from __future__ import annotations

import copy
import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock
from threading import Thread
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.json_processor import (
    ProcessContext,
    ProcessOptions,
    ProcessingCancelledError,
    estimate_progress_units,
    process_json_document,
)
from app.utils.diff_tracker import DiffTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Multilingual Text Review API", version="1.0.0")

raw_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,https://ai-translator-a89r.vercel.app",
)
allow_origins = [o.strip() for o in raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_DIR = Path(tempfile.gettempdir()) / "ai_translator_review_jobs"
JOB_DIR.mkdir(parents=True, exist_ok=True)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = Lock()


class ReviewBody(BaseModel):
    data: Any
    target_language: str = Field(default="pt-BR")
    treat_biblical_texto_as_google: bool = Field(default=False)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/review")
def review_json(body: ReviewBody) -> dict[str, Any]:
    tracker = DiffTracker()
    opts = ProcessOptions(
        target_language=body.target_language,
        treat_biblical_texto_as_google=body.treat_biblical_texto_as_google,
    )
    try:
        result = process_json_document(body.data, opts, tracker, progress=None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("review failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "result": result,
        "changes": tracker.to_json(),
        "change_count": len(tracker.changes),
    }


def _form_bool(raw: str) -> bool:
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_result_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.json"


def _job_checkpoint_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.checkpoint.json"


def _preview_text(text: str, limit: int = 220) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _serialize_recent_changes(tracker: DiffTracker, limit: int = 8) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for ch in tracker.changes[-limit:]:
        out.append(
            {
                "path": ch.path,
                "before_preview": _preview_text(ch.before),
                "after_preview": _preview_text(ch.after),
            }
        )
    return out


def _set_job(job_id: str, updates: dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)
        _jobs[job_id]["updated_at"] = _now_iso()


def _is_daily_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    has_rate_limit_signal = any(
        s in msg
        for s in (
            "rate limit",
            "too many requests",
            "error code: 429",
            "status code 429",
        )
    )
    has_daily_signal = any(
        s in msg
        for s in (
            "requests per day",
            "rpd",
            "max_requests_per_1_day",
            "per_1_day",
        )
    )
    return has_rate_limit_signal and has_daily_signal


def _next_utc_midnight_iso() -> str:
    now = datetime.now(timezone.utc)
    next_day = (now + timedelta(days=1)).date()
    midnight = datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)
    return midnight.isoformat()


def _seconds_until_next_utc_midnight(min_seconds: int = 300) -> int:
    now = datetime.now(timezone.utc)
    next_day = (now + timedelta(days=1)).date()
    midnight = datetime.combine(next_day, datetime.min.time(), tzinfo=timezone.utc)
    seconds = int((midnight - now).total_seconds())
    return max(seconds, min_seconds)


def _wait_until_next_utc_midnight_or_cancel(should_cancel: Any) -> bool:
    # Sleep in short intervals so cancel requests can be honored.
    while True:
        if should_cancel():
            return True
        remaining = _seconds_until_next_utc_midnight(min_seconds=0)
        if remaining <= 0:
            return False
        time.sleep(min(remaining, 60))


def _save_checkpoint(job_id: str, payload: dict[str, Any]) -> None:
    _job_checkpoint_path(job_id).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_checkpoint(job_id: str) -> dict[str, Any] | None:
    p = _job_checkpoint_path(job_id)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _clear_checkpoint(job_id: str) -> None:
    p = _job_checkpoint_path(job_id)
    if p.exists():
        p.unlink()


def _spawn_resume_thread(
    job_id: str,
    data: Any,
    filename: str,
    target_language: str,
    treat_biblical_texto_as_google: bool,
) -> None:
    t = Thread(
        target=_run_review_job,
        args=(
            job_id,
            data,
            filename,
            target_language,
            treat_biblical_texto_as_google,
        ),
        daemon=True,
    )
    t.start()


def _recover_jobs_on_startup() -> None:
    for cp_path in JOB_DIR.glob("*.checkpoint.json"):
        try:
            cp = json.loads(cp_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to parse checkpoint: %s", cp_path)
            continue
        partial = cp.get("partial_result")
        if not isinstance(partial, dict):
            logger.warning("checkpoint missing partial_result: %s", cp_path)
            continue
        # "<job_id>.checkpoint.json" -> "<job_id>"
        suffix = ".checkpoint.json"
        name = cp_path.name
        if not name.endswith(suffix):
            continue
        job_id = name[: -len(suffix)]
        now = _now_iso()
        filename = str(cp.get("filename", "document.json"))
        target_language = str(cp.get("target_language", "pt-BR"))
        treat_flag = bool(cp.get("treat_biblical_texto_as_google", False))
        total_units = cp.get("total_units")
        completed_units = int(cp.get("completed_units", 0))
        progress_pct = float(cp.get("progress_pct", 0.0))
        with _jobs_lock:
            if job_id not in _jobs:
                _jobs[job_id] = {
                    "job_id": job_id,
                    "status": "queued",
                    "cancel_requested": False,
                    "filename": filename,
                    "target_language": target_language,
                    "treat_biblical_texto_as_google": treat_flag,
                    "change_count": None,
                    "error": None,
                    "result_file": None,
                    "total_units": total_units,
                    "completed_units": completed_units,
                    "progress_pct": progress_pct,
                    "current_path": None,
                    "changes_live_count": int(cp.get("changes_live_count", 0)),
                    "recent_changes": [],
                    "resume_after": None,
                    "created_at": str(cp.get("created_at", now)),
                    "updated_at": now,
                }
        logger.info("recovering paused review job %s from checkpoint", job_id)
        _spawn_resume_thread(job_id, partial, filename, target_language, treat_flag)


def _run_review_job(
    job_id: str,
    data: Any,
    filename: str,
    target_language: str,
    treat_biblical_texto_as_google: bool,
) -> None:
    _set_job(job_id, {"status": "processing", "resume_after": None})
    tracker = DiffTracker()
    opts = ProcessOptions(
        target_language=target_language,
        treat_biblical_texto_as_google=treat_biblical_texto_as_google,
    )
    try:
        total_units = estimate_progress_units(data, opts)
        def should_cancel() -> bool:
            with _jobs_lock:
                job = _jobs.get(job_id)
                return bool(job and job.get("cancel_requested"))

        _set_job(
            job_id,
            {
                "total_units": total_units,
                "completed_units": 0,
                "progress_pct": 0.0,
                "current_path": None,
            },
        )

        completed_units = 0
        last_pushed = 0

        def on_progress(event: str, path: str | None) -> None:
            nonlocal completed_units, last_pushed
            if event in ("chunk", "title_gen", "memo_hit"):
                completed_units += 1
            if path is None:
                return
            # Throttle frequent updates for massive documents.
            if (
                completed_units == total_units
                or completed_units - last_pushed >= 8
                or event in ("title_gen",)
            ):
                last_pushed = completed_units
                pct = round((100.0 * min(completed_units, total_units)) / max(total_units, 1), 3)
                _set_job(
                    job_id,
                    {
                        "completed_units": min(completed_units, total_units),
                        "progress_pct": pct,
                        "current_path": path,
                        "changes_live_count": len(tracker.changes),
                        "recent_changes": _serialize_recent_changes(tracker),
                    },
                )

        can_resume_by_livro = (
            isinstance(data, dict) and isinstance(data.get("livros"), list)
        )
        if can_resume_by_livro:
            root = copy.deepcopy(data)
            livros = root["livros"]
            ctx = ProcessContext(
                reviewed_text_cache={},
                generated_title_cache={},
            )
            cp = _load_checkpoint(job_id)
            next_idx = 0
            if cp:
                next_idx = int(cp.get("next_livro_index", 0))
                saved = cp.get("partial_result")
                if isinstance(saved, dict) and isinstance(saved.get("livros"), list):
                    root = saved
                    livros = root["livros"]
            end_idx = len(livros)
            while next_idx < end_idx:
                if should_cancel():
                    raise ProcessingCancelledError("Processing cancelled by user")
                wrapped = {"livros": [copy.deepcopy(livros[next_idx])]}
                try:
                    processed = process_json_document(
                        wrapped,
                        opts,
                        tracker,
                        progress=on_progress,
                        ctx=ctx,
                        should_cancel=should_cancel,
                    )
                except Exception as e:
                    _save_checkpoint(
                        job_id,
                        {
                            "next_livro_index": next_idx,
                            "partial_result": root,
                            "filename": filename,
                            "target_language": target_language,
                            "treat_biblical_texto_as_google": treat_biblical_texto_as_google,
                            "total_units": total_units,
                            "completed_units": min(completed_units, total_units),
                            "progress_pct": round(
                                (100.0 * min(completed_units, total_units))
                                / max(total_units, 1),
                                3,
                            ),
                            "changes_live_count": len(tracker.changes),
                            "created_at": _jobs.get(job_id, {}).get("created_at", _now_iso()),
                            "error": str(e),
                            "updated_at": _now_iso(),
                        },
                    )
                    if _is_daily_rate_limit_error(e):
                        _set_job(
                            job_id,
                            {
                                "status": "paused_daily_limit",
                                "error": str(e),
                                "resume_after": _next_utc_midnight_iso(),
                            },
                        )
                        cancelled = _wait_until_next_utc_midnight_or_cancel(should_cancel)
                        if cancelled:
                            raise ProcessingCancelledError("Processing cancelled by user")
                        _set_job(
                            job_id,
                            {
                                "status": "processing",
                                "error": None,
                                "resume_after": None,
                            },
                        )
                        continue
                    raise
                livros[next_idx] = processed["livros"][0]
                next_idx += 1
                _save_checkpoint(
                    job_id,
                    {
                        "next_livro_index": next_idx,
                        "partial_result": root,
                        "filename": filename,
                        "target_language": target_language,
                        "treat_biblical_texto_as_google": treat_biblical_texto_as_google,
                        "total_units": total_units,
                        "completed_units": min(completed_units, total_units),
                        "progress_pct": round(
                            (100.0 * min(completed_units, total_units)) / max(total_units, 1),
                            3,
                        ),
                        "changes_live_count": len(tracker.changes),
                        "created_at": _jobs.get(job_id, {}).get("created_at", _now_iso()),
                        "updated_at": _now_iso(),
                    },
                )
            result = root
        else:
            result = process_json_document(
                data,
                opts,
                tracker,
                progress=on_progress,
                should_cancel=should_cancel,
            )
        if should_cancel():
            raise ProcessingCancelledError("Processing cancelled by user")
        payload = {
            "result": result,
            "changes": tracker.to_json(),
            "change_count": len(tracker.changes),
            "filename": filename,
        }
        out_path = _job_result_path(job_id)
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        _set_job(
            job_id,
            {
                "status": "completed",
                "cancel_requested": False,
                "change_count": len(tracker.changes),
                "result_file": str(out_path),
                "completed_units": total_units,
                "progress_pct": 100.0,
                "current_path": None,
                "changes_live_count": len(tracker.changes),
                "recent_changes": _serialize_recent_changes(tracker),
                "resume_after": None,
            },
        )
        _clear_checkpoint(job_id)
    except ProcessingCancelledError as e:
        _set_job(
            job_id,
            {
                "status": "cancelled",
                "error": str(e),
                "current_path": None,
            },
        )
    except Exception as e:
        logger.exception("async review job failed")
        _set_job(
            job_id,
            {
                "status": "failed",
                "error": str(e),
            },
        )


def _get_job_or_404(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/review/upload")
async def review_upload(
    file: UploadFile = File(...),
    target_language: str = Form(default="pt-BR"),
    treat_biblical_texto_as_google: str = Form(default="false"),
) -> dict[str, Any]:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="File must be UTF-8 JSON") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    tracker = DiffTracker()
    opts = ProcessOptions(
        target_language=target_language,
        treat_biblical_texto_as_google=_form_bool(treat_biblical_texto_as_google),
    )
    try:
        result = process_json_document(data, opts, tracker, progress=None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("review failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return {
        "result": result,
        "changes": tracker.to_json(),
        "change_count": len(tracker.changes),
        "filename": file.filename or "document.json",
    }


@app.post("/api/review/upload/async")
async def review_upload_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_language: str = Form(default="pt-BR"),
    treat_biblical_texto_as_google: str = Form(default="false"),
) -> dict[str, Any]:
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
        data = json.loads(text)
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="File must be UTF-8 JSON") from e
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    job_id = str(uuid.uuid4())
    now = _now_iso()
    filename = file.filename or "document.json"
    treat_flag = _form_bool(treat_biblical_texto_as_google)
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "cancel_requested": False,
            "filename": filename,
            "target_language": target_language,
            "treat_biblical_texto_as_google": treat_flag,
            "change_count": None,
            "error": None,
            "result_file": None,
            "total_units": None,
            "completed_units": 0,
            "progress_pct": 0.0,
            "current_path": None,
            "changes_live_count": 0,
            "recent_changes": [],
            "resume_after": None,
            "created_at": now,
            "updated_at": now,
        }

    background_tasks.add_task(
        _run_review_job,
        job_id,
        data,
        filename,
        target_language,
        treat_flag,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "status_url": f"/api/review/jobs/{job_id}",
        "result_url": f"/api/review/jobs/{job_id}/result",
        "download_url": f"/api/review/jobs/{job_id}/download",
    }


@app.get("/api/review/jobs/{job_id}")
def get_review_job(job_id: str) -> dict[str, Any]:
    job = _get_job_or_404(job_id)
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "cancel_requested": job["cancel_requested"],
        "filename": job["filename"],
        "target_language": job["target_language"],
        "treat_biblical_texto_as_google": job["treat_biblical_texto_as_google"],
        "change_count": job["change_count"],
        "error": job["error"],
        "total_units": job["total_units"],
        "completed_units": job["completed_units"],
        "progress_pct": job["progress_pct"],
        "current_path": job["current_path"],
        "changes_live_count": job["changes_live_count"],
        "recent_changes": job["recent_changes"],
        "resume_after": job.get("resume_after"),
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


@app.post("/api/review/jobs/{job_id}/cancel")
def cancel_review_job(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        if job["status"] in ("completed", "failed", "cancelled"):
            return {"job_id": job_id, "status": job["status"], "cancel_requested": False}
        job["cancel_requested"] = True
        job["status"] = "cancelling"
        job["updated_at"] = _now_iso()
        return {"job_id": job_id, "status": "cancelling", "cancel_requested": True}


@app.get("/api/review/jobs/{job_id}/result")
def get_review_job_result(job_id: str) -> dict[str, Any]:
    job = _get_job_or_404(job_id)
    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job["error"] or "Job failed")
    if job["status"] != "completed" or not job["result_file"]:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    result_file = Path(job["result_file"])
    if not result_file.exists():
        raise HTTPException(status_code=500, detail="Result file is missing")
    return json.loads(result_file.read_text(encoding="utf-8"))


@app.get("/api/review/jobs/{job_id}/download")
def download_review_job_result(job_id: str) -> FileResponse:
    job = _get_job_or_404(job_id)
    if job["status"] == "failed":
        raise HTTPException(status_code=409, detail=job["error"] or "Job failed")
    if job["status"] != "completed" or not job["result_file"]:
        raise HTTPException(status_code=409, detail="Job is not completed yet")
    result_file = Path(job["result_file"])
    if not result_file.exists():
        raise HTTPException(status_code=500, detail="Result file is missing")
    output_name = f"reviewed-{job['filename']}"
    return FileResponse(
        path=result_file,
        media_type="application/json",
        filename=output_name,
    )


@app.on_event("startup")
def startup_recover_jobs() -> None:
    _recover_jobs_on_startup()
