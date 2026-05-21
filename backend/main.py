from __future__ import annotations

import csv
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .dependency_check import check_dependencies, optional_dependency_report, pytorch_install_advice
from .diagnostics import collect_system_report
from .runtime_dirs import APP_DIR, IMG_DIR, OUTPUT_DIR, ROOT, ensure_runtime_dirs
from .image_scan import IMAGE_EXTENSIONS, list_image_folders, scan_images
from .jobs import cancel_job, create_clip_projection_job, get_job, job_counts, list_jobs, list_projection_files
from .schemas import ClipProjectionRequest
from .encoders.registry import get_model_spec, public_model_registry
from .search import list_search_indexes, run_text_search
from .search.schemas import RebuildSearchIndexRequest, TextSearchRequest
from .settings import apply_runtime_settings, delete_huggingface_token, model_guide, runtime_settings_public, save_huggingface_token
from .version import APP_VERSION
from .analysis import (
    cluster_report,
    compare_projections,
    delete_session,
    export_html_package,
    list_sessions,
    load_session,
    projection_catalog,
    save_session,
)

ensure_runtime_dirs()
apply_runtime_settings()

app = FastAPI(title="ImageCluster")
app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/clip", response_class=HTMLResponse)
def clip() -> str:
    return (APP_DIR / "clip_projection.html").read_text(encoding="utf-8")


@app.get("/settings", response_class=HTMLResponse)
def settings_page() -> str:
    return (APP_DIR / "settings.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    ok, report = check_dependencies()
    return {
        "ok": ok,
        "dependencies": [{"name": d.name, "installed": d.installed, "required": d.required, "note": d.note} for d in report],
        "default_image_dir": "img",
        "version": APP_VERSION,
        "features": {"optional_clustering": True, "model_registry": True, "debug_reports": True, "embedding_recovery": True, "semantic_search": True},
        "settings": runtime_settings_public(),
        "optional_dependencies": optional_dependency_report(),
        "install_advice": pytorch_install_advice(),
        "jobs": job_counts(),
    }


class HuggingFaceTokenPayload(BaseModel):
    token: str


@app.get("/api/settings")
def settings_read() -> dict:
    return runtime_settings_public()


@app.put("/api/settings/huggingface-token")
def settings_save_hf_token(payload: HuggingFaceTokenPayload) -> dict:
    try:
        return {"huggingface": save_huggingface_token(payload.token)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/settings/huggingface-token")
def settings_delete_hf_token() -> dict:
    return {"huggingface": delete_huggingface_token()}


@app.get("/api/model-guide")
def settings_model_guide() -> dict:
    return model_guide()


@app.get("/api/debug/system")
def debug_system() -> dict:
    return collect_system_report({"source": "api/debug/system"})


@app.get("/api/diagnostics")
def diagnostics() -> dict:
    report = collect_system_report({"source": "api/diagnostics"})
    report["version"] = APP_VERSION
    report["jobs"] = job_counts()
    return report


@app.get("/api/models")
def models() -> dict:
    return {"models": public_model_registry()}


@app.get("/api/search/indexes")
def search_indexes() -> dict:
    return {"indexes": list_search_indexes()}


@app.get("/api/analysis/projections")
def analysis_projections() -> dict:
    return {"projections": projection_catalog()}


@app.get("/api/analysis/cluster-report")
def analysis_cluster_report(projection: str, representatives: int = 5) -> dict:
    try:
        return cluster_report(projection, representatives)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/model-comparison")
def analysis_model_comparison(payload: dict) -> dict:
    try:
        return compare_projections(payload.get("projections") or [], int(payload.get("top_k") or 5))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/projection-comparison")
def analysis_projection_comparison(payload: dict) -> dict:
    try:
        return compare_projections(payload.get("projections") or [], int(payload.get("top_k") or 5))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/export/html")
def html_export(payload: dict) -> dict:
    try:
        return export_html_package(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/sessions")
def sessions() -> dict:
    return {"sessions": list_sessions()}


@app.post("/api/sessions")
def session_create(payload: dict) -> dict:
    return save_session(payload)


@app.get("/api/sessions/{session_id}")
def session_read(session_id: str) -> dict:
    try:
        data = load_session(session_id)
        if data is None:
            raise FileNotFoundError("Session not found")
        return data
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}")
def session_update(session_id: str, payload: dict) -> dict:
    return save_session(payload, session_id=session_id)


@app.delete("/api/sessions/{session_id}")
def session_delete(session_id: str) -> dict:
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.post("/api/search/text")
def search_text(req: TextSearchRequest) -> dict:
    try:
        return run_text_search(
            query=req.query.strip(),
            model_key=req.model_key,
            image_dir=req.image_dir,
            embedding_id=req.embedding_id,
            top_k=req.top_k,
            threshold=req.threshold,
            normalize=req.normalize,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/search/rebuild-index")
def search_rebuild_index(req: RebuildSearchIndexRequest) -> dict:
    spec = get_model_spec(req.model_key)
    if not spec.supports_text_search:
        raise HTTPException(status_code=400, detail="This model can create image projections but does not support text search.")
    projection_req = ClipProjectionRequest(image_dir=req.image_dir, model_key=req.model_key, use_cache=True)
    job = create_clip_projection_job(projection_req, job_type="search_rebuild_index")
    return {"job_id": job.job_id, "status": job.status, "message": "Queued embedding/projection job for searchable index."}


@app.get("/api/image-folders")
def image_folders() -> dict:
    folders = ["img" if folder == "." else f"img/{folder}" for folder in list_image_folders(IMG_DIR)]
    return {"root": "img", "folders": folders, "default_image_dir": "img", "current_image_dir": "img"}


@app.get("/api/images")
def images(dir: str = "img") -> dict:
    image_dir = _safe_dir(dir)
    paths = scan_images(image_dir)
    return {
        "dir": str(image_dir.relative_to(ROOT)),
        "count": len(paths),
        "images": [{"filename": p.name, "relative_path": p.relative_to(ROOT).as_posix()} for p in paths],
    }


@app.get("/api/images/scan")
def image_scan(image_dir: str = "img") -> dict:
    try:
        resolved = _safe_dir(image_dir)
    except HTTPException as exc:
        return {"ok": False, "image_dir": image_dir, "error": str(exc.detail)}
    if not resolved.exists():
        return {"ok": False, "image_dir": image_dir, "absolute_path": str(resolved), "error": "The selected image folder does not exist."}
    if not resolved.is_dir():
        return {"ok": False, "image_dir": image_dir, "absolute_path": str(resolved), "error": "The selected image path is not a folder."}
    try:
        paths = scan_images(resolved)
    except OSError:
        return {"ok": False, "image_dir": image_dir, "absolute_path": str(resolved), "error": "The selected image folder cannot be read."}
    extensions = Counter(p.suffix.lower() for p in paths)
    all_files = [p for p in resolved.rglob("*") if p.is_file()]
    unsupported_count = len([p for p in all_files if p.suffix.lower() not in IMAGE_EXTENSIONS])
    warnings = []
    if not paths:
        warnings.append("No supported image files were found.")
    if unsupported_count:
        warnings.append(f"{unsupported_count} unsupported file(s) were ignored.")
    return {
        "ok": True,
        "image_dir": str(resolved.relative_to(ROOT)),
        "absolute_path": str(resolved),
        "count": len(paths),
        "extensions": dict(sorted(extensions.items())),
        "unsupported_count": unsupported_count,
        "warnings": warnings,
    }


@app.post("/api/jobs/clip-projection")
def create_clip_projection(req: ClipProjectionRequest) -> dict:
    job = create_clip_projection_job(req)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs")
def jobs(status: str | None = None, job_type: str | None = None, limit: int | None = None, newest_first: bool = True) -> dict:
    return {"jobs": list_jobs(status=status, job_type=job_type, limit=limit, newest_first=newest_first)}


@app.get("/api/projections")
def projections() -> dict:
    return {"projections": list_projection_files()}


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    job = cancel_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.get("/api/jobs/{job_id}/result")
def job_result(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != "completed" or not job.result_path:
        raise HTTPException(status_code=409, detail="Job is not completed")
    return _read_tsv_result(ROOT / job.result_path, {"job_id": job_id, "result_path": job.result_path})


@app.get("/api/projections/read")
def projection_read(path: str) -> JSONResponse:
    requested = (ROOT / path).resolve()
    output_root = (OUTPUT_DIR / "projections").resolve()
    if requested != output_root and output_root not in requested.parents:
        raise HTTPException(status_code=400, detail="Projection path must be inside output/projections/.")
    if not requested.exists():
        raise HTTPException(status_code=404, detail="Projection file not found")
    return _read_tsv_result(requested, {"result_path": requested.relative_to(ROOT).as_posix()})


@app.get("/api/jobs/{job_id}/debug")
def job_debug(job_id: str) -> JSONResponse:
    job = get_job(job_id)
    if job is None or not job.debug_path:
        raise HTTPException(status_code=404, detail="Debug report not found")
    path = ROOT / job.debug_path
    if not path.exists():
        raise HTTPException(status_code=404, detail="Debug report file not found")
    import json
    return JSONResponse(json.loads(path.read_text(encoding="utf-8", errors="replace")))


@app.get("/api/jobs/{job_id}/download")
def job_download(job_id: str) -> FileResponse:
    job = get_job(job_id)
    if job is None or not job.result_path:
        raise HTTPException(status_code=404, detail="Result not found")
    path = ROOT / job.result_path
    return FileResponse(path, filename=path.name, media_type="text/tab-separated-values")


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str) -> PlainTextResponse:
    job = get_job(job_id)
    if job is None or not job.log_path:
        raise HTTPException(status_code=404, detail="Log not found")
    path = ROOT / job.log_path
    return PlainTextResponse(path.read_text(encoding="utf-8", errors="replace"))


def _read_tsv_result(path: Path, extra: dict) -> JSONResponse:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
    columns = list(rows[0].keys()) if rows else []
    payload = {"columns": columns, "rows": rows}
    payload.update(extra)
    return JSONResponse(payload)


def _safe_dir(dir_value: str) -> Path:
    if dir_value == "img":
        return IMG_DIR.resolve()
    path = (ROOT / dir_value).resolve()
    root = IMG_DIR.resolve()
    if path == root or root in path.parents:
        return path
    raise HTTPException(status_code=400, detail="Directory must be img/ or a subfolder of img/.")
