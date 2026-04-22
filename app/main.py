from __future__ import annotations

import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services.json_processor import (
    ProcessOptions,
    estimate_progress_units,
    process_json_document,
)
from app.utils.diff_tracker import DiffTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Multilingual Text Review API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


def _set_job(job_id: str, updates: dict[str, Any]) -> None:
    with _jobs_lock:
        if job_id not in _jobs:
            return
        _jobs[job_id].update(updates)
        _jobs[job_id]["updated_at"] = _now_iso()


def _run_review_job(
    job_id: str,
    data: Any,
    filename: str,
    target_language: str,
    treat_biblical_texto_as_google: bool,
) -> None:
    _set_job(job_id, {"status": "processing"})
    tracker = DiffTracker()
    opts = ProcessOptions(
        target_language=target_language,
        treat_biblical_texto_as_google=treat_biblical_texto_as_google,
    )
    try:
        total_units = estimate_progress_units(data, opts)
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
            if event in ("chunk", "title_gen"):
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
                    },
                )

        result = process_json_document(data, opts, tracker, progress=on_progress)
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
                "change_count": len(tracker.changes),
                "result_file": str(out_path),
                "completed_units": total_units,
                "progress_pct": 100.0,
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
        "filename": job["filename"],
        "target_language": job["target_language"],
        "treat_biblical_texto_as_google": job["treat_biblical_texto_as_google"],
        "change_count": job["change_count"],
        "error": job["error"],
        "total_units": job["total_units"],
        "completed_units": job["completed_units"],
        "progress_pct": job["progress_pct"],
        "current_path": job["current_path"],
        "created_at": job["created_at"],
        "updated_at": job["updated_at"],
    }


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
