from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from threading import Thread
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import AuthenticationError, PermissionDeniedError
from pydantic import BaseModel, Field

from app.services.json_processor import (
    ProcessContext,
    ProcessOptions,
    ProcessingCancelledError,
    estimate_progress_units,
    process_json_document,
    process_json_document_async,
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


def _job_source_path(job_id: str) -> Path:
    """Pristine uploaded document, kept on disk for crash/restart recovery."""
    return JOB_DIR / f"{job_id}.source.json"


def _job_cache_path(job_id: str) -> Path:
    """Append-only, content-addressed store of every corrected unit."""
    return JOB_DIR / f"{job_id}.cache.jsonl"


def _job_meta_path(job_id: str) -> Path:
    return JOB_DIR / f"{job_id}.meta.json"


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


def _save_source(job_id: str, data: Any) -> None:
    _job_source_path(job_id).write_text(
        json.dumps(data, ensure_ascii=False),
        encoding="utf-8",
    )


def _save_meta(job_id: str, meta: dict[str, Any]) -> None:
    _job_meta_path(job_id).write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )


def _load_meta(job_id: str) -> dict[str, Any] | None:
    p = _job_meta_path(job_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("failed to parse job meta: %s", p)
        return None


def _cleanup_job_files(job_id: str) -> None:
    """Remove the recovery artifacts once a job reaches a clean terminal state.

    Their absence is the signal that the job is no longer mid-flight: any
    leftover `*.source.json` means the process died before finishing and the
    job should be resumed on startup.
    """
    for p in (_job_source_path(job_id), _job_cache_path(job_id), _job_meta_path(job_id)):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            logger.warning("could not remove job artifact %s", p)


class _DurableCache:
    """Content-addressed, append-only result store for one job.

    Each corrected chunk / generated title is written as a JSON line the moment
    it is produced, so a restart or retry serves it from disk instead of calling
    the API again. Keyed by a content hash, it makes resume work for any JSON
    shape and replaces the old whole-document checkpoint (which rewrote the
    entire file after every book).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._store: dict[str, str] = {}
        self._lock = Lock()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                k, v = obj.get("k"), obj.get("v")
                if isinstance(k, str) and isinstance(v, str):
                    self._store[k] = v
        self._fh = path.open("a", encoding="utf-8")

    @property
    def hits_on_load(self) -> int:
        return len(self._store)

    def get(self, key: str) -> str | None:
        return self._store.get(key)

    def put(self, key: str, value: str) -> None:
        with self._lock:
            if key in self._store:
                return
            self._store[key] = value
            self._fh.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")
            self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except OSError:
            pass


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
    """Resume any job whose source file is still present (i.e. it was mid-flight
    when the process stopped). The durable cache makes this near-instant for
    work already completed before the interruption."""
    for src_path in JOB_DIR.glob("*.source.json"):
        suffix = ".source.json"
        name = src_path.name
        if not name.endswith(suffix):
            continue
        job_id = name[: -len(suffix)]
        if _job_result_path(job_id).exists():
            continue
        try:
            data = json.loads(src_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("failed to parse job source: %s", src_path)
            continue
        meta = _load_meta(job_id) or {}
        now = _now_iso()
        filename = str(meta.get("filename", "document.json"))
        target_language = str(meta.get("target_language", "pt-BR"))
        treat_flag = bool(meta.get("treat_biblical_texto_as_google", False))
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
                    "total_units": meta.get("total_units"),
                    "completed_units": 0,
                    "progress_pct": 0.0,
                    "current_path": None,
                    "changes_live_count": 0,
                    "recent_changes": [],
                    "resume_after": None,
                    "created_at": str(meta.get("created_at", now)),
                    "updated_at": now,
                }
        logger.info("recovering interrupted review job %s from source + cache", job_id)
        _spawn_resume_thread(job_id, data, filename, target_language, treat_flag)


# Bounded full-document restarts for *unexpected* errors. Rate limits and
# transient API failures are handled inside the pipeline and never reach here;
# these restarts cover crashes/bugs, and the durable cache makes each restart
# skip all previously completed work.
_JOB_MAX_RESTARTS = 5


def _run_review_job(
    job_id: str,
    data: Any,
    filename: str,
    target_language: str,
    treat_biblical_texto_as_google: bool,
) -> None:
    _set_job(job_id, {"status": "processing", "resume_after": None})
    opts = ProcessOptions(
        target_language=target_language,
        treat_biblical_texto_as_google=treat_biblical_texto_as_google,
    )

    def should_cancel() -> bool:
        with _jobs_lock:
            job = _jobs.get(job_id)
            return bool(job and job.get("cancel_requested"))

    try:
        # Persist the pristine source so the job can be resumed after a crash or
        # restart. Skip if it already exists (we are resuming such a job now).
        if not _job_source_path(job_id).exists():
            _save_source(job_id, data)

        total_units = estimate_progress_units(data, opts)
        _save_meta(
            job_id,
            {
                "filename": filename,
                "target_language": target_language,
                "treat_biblical_texto_as_google": treat_biblical_texto_as_google,
                "total_units": total_units,
                "created_at": _jobs.get(job_id, {}).get("created_at", _now_iso()),
            },
        )
        _set_job(
            job_id,
            {
                "total_units": total_units,
                "completed_units": 0,
                "progress_pct": 0.0,
                "current_path": None,
            },
        )

        durable = _DurableCache(_job_cache_path(job_id))
        if durable.hits_on_load:
            logger.info(
                "job %s resuming with %d unit(s) already cached on disk",
                job_id,
                durable.hits_on_load,
            )
        ctx = ProcessContext(durable_get=durable.get, durable_put=durable.put)

        attempt = 0
        tracker = DiffTracker()
        try:
            while True:
                # Each attempt walks the whole (pristine) document. Anything done
                # in a prior attempt is served from the durable cache, so this is
                # cheap; only un-finished units actually hit the API.
                tracker = DiffTracker()
                ctx.reviewed_text_cache = {}
                ctx.generated_title_cache = {}
                state = {"completed": 0, "last_pushed": 0}

                def on_progress(event: str, path: str | None) -> None:
                    if event == "paused":
                        _set_job(
                            job_id,
                            {
                                "status": "paused_daily_limit",
                                "resume_after": path,
                                "error": None,
                            },
                        )
                        return
                    if event == "resumed":
                        _set_job(job_id, {"status": "processing", "resume_after": None})
                    if event in ("chunk", "title_gen", "memo_hit"):
                        state["completed"] += 1
                    completed = state["completed"]
                    if path is None and event != "resumed":
                        return
                    if (
                        completed == total_units
                        or completed - state["last_pushed"] >= 8
                        or event in ("title_gen", "resumed")
                    ):
                        state["last_pushed"] = completed
                        pct = round(
                            (100.0 * min(completed, total_units)) / max(total_units, 1), 3
                        )
                        _set_job(
                            job_id,
                            {
                                "completed_units": min(completed, total_units),
                                "progress_pct": pct,
                                "current_path": path,
                                "changes_live_count": len(tracker.changes),
                                "recent_changes": _serialize_recent_changes(tracker),
                            },
                        )

                try:
                    result = process_json_document(
                        data,
                        opts,
                        tracker,
                        progress=on_progress,
                        ctx=ctx,
                        should_cancel=should_cancel,
                        in_place=True,
                    )
                    break
                except ProcessingCancelledError:
                    raise
                except (ValueError, AuthenticationError, PermissionDeniedError):
                    # Configuration / credential problems: retrying cannot help.
                    raise
                except Exception:
                    attempt += 1
                    if attempt >= _JOB_MAX_RESTARTS:
                        raise
                    logger.exception(
                        "review job %s attempt %d failed; reloading source and retrying",
                        job_id,
                        attempt,
                    )
                    # Reload a pristine copy: the previous attempt mutated `data`
                    # in place, and re-walking that would re-send corrected text.
                    data = json.loads(_job_source_path(job_id).read_text(encoding="utf-8"))
                    _set_job(job_id, {"status": "processing", "error": None})
                    time.sleep(min(2 ** attempt, 30))
                    continue
        finally:
            durable.close()

        if should_cancel():
            raise ProcessingCancelledError("Processing cancelled by user")

        payload = {
            "result": result,
            "changes": tracker.to_json(),
            "change_count": len(tracker.changes),
            "filename": filename,
        }
        out_path = _job_result_path(job_id)
        out_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
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
        _cleanup_job_files(job_id)
    except ProcessingCancelledError as e:
        _set_job(
            job_id,
            {"status": "cancelled", "error": str(e), "current_path": None},
        )
        _cleanup_job_files(job_id)
    except Exception as e:
        logger.exception("async review job failed")
        _set_job(job_id, {"status": "failed", "error": str(e)})
        _cleanup_job_files(job_id)


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
        # This handler is `async def`, so we already have a running event
        # loop and must use the async-native entry point.
        result = await process_json_document_async(
            data, opts, tracker, progress=None, in_place=True
        )
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
