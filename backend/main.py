from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path
from collections import Counter

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .config_store import apply_hf_token_to_environment, get_hf_token, remove_hf_token, set_hf_token
from .dependency_check import check_dependencies, optional_dependency_report, pytorch_install_advice
from .diagnostics import collect_system_report
from .runtime_dirs import APP_DIR, IMG_DIR, OUTPUT_DIR, ROOT, ensure_runtime_dirs
from .image_scan import list_image_folders, scan_images
from .jobs import cancel_job, create_clip_projection_job, get_job, list_jobs, list_projection_files
from .schemas import ClipProjectionRequest, LocalTokenRequest
from .encoders.registry import get_model_spec, public_model_registry
from .search import list_search_indexes, run_text_search
from .search.schemas import RebuildSearchIndexRequest, TextSearchRequest

ensure_runtime_dirs()

app = FastAPI(title="ImagePlot-CLIP")
app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return (APP_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/clip", response_class=HTMLResponse)
def clip() -> str:
    return (APP_DIR / "clip_projection.html").read_text(encoding="utf-8")


@app.get("/integrated", include_in_schema=False)
def integrated() -> RedirectResponse:
    return RedirectResponse(url="/clip", status_code=302)


@app.get("/models", response_class=HTMLResponse)
def models_page() -> str:
    return (APP_DIR / "models_tokens.html").read_text(encoding="utf-8")


THUMB_CACHE = OUTPUT_DIR / "thumb_cache"


@app.get("/api/thumb")
def thumbnail(path: str, w: int = 160) -> Response:
    """Serve a downscaled JPEG thumbnail for canvas/gallery rendering.

    The full-resolution images are still served by the /img mount for the preview
    modal; this keeps decoded image memory and per-frame draw cost low when many
    thumbnails are shown at once. Results are cached on disk (keyed by path, width
    and mtime) so a large projection only pays the resize cost once.
    """
    width = max(16, min(512, int(w)))
    candidate = (ROOT / path).resolve()
    img_root = IMG_DIR.resolve()
    if not (candidate == img_root or candidate.is_relative_to(img_root)) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    try:
        from PIL import Image

        stat = candidate.stat()
        key = hashlib.sha1(f"{candidate}|{width}|{int(stat.st_mtime)}|{stat.st_size}".encode("utf-8")).hexdigest()
        THUMB_CACHE.mkdir(parents=True, exist_ok=True)
        cached = THUMB_CACHE / f"{key}.jpg"
        if cached.is_file():
            data = cached.read_bytes()
        else:
            with Image.open(candidate) as im:
                im = im.convert("RGB")
                im.thumbnail((width, width))
                buffer = io.BytesIO()
                im.save(buffer, format="JPEG", quality=82)
                data = buffer.getvalue()
            cached.write_bytes(data)
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - decode/format errors are environment-specific
        raise HTTPException(status_code=415, detail=f"Cannot render thumbnail: {exc}")
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/status")
def status() -> dict:
    ok, report = check_dependencies()
    return {
        "ok": ok,
        "dependencies": [{"name": d.name, "installed": d.installed, "required": d.required, "note": d.note} for d in report],
        "default_image_dir": "img",
        "version": "5.5-dev",
        "features": {"optional_clustering": True, "model_registry": True, "debug_reports": True, "embedding_recovery": True, "semantic_search": True},
        "optional_dependencies": optional_dependency_report(),
        "install_advice": pytorch_install_advice(),
    }


@app.get("/api/debug/system")
def debug_system() -> dict:
    return collect_system_report({"source": "api/debug/system"})


@app.get("/api/models")
def models() -> dict:
    return {"models": public_model_registry()}


@app.get("/api/config/local")
def local_config() -> dict:
    token = apply_hf_token_to_environment()
    tail = token[-4:] if token else ""
    return {
        "huggingface_token_configured": bool(token),
        "huggingface_token_tail": tail,
        "huggingface_token_runtime_enabled": bool(token),
        "token_used_by_model_loaders": bool(token),
        "storage_path": "output/local_settings.json",
        "storage_note": "Do not commit local settings to source control.",
    }


@app.put("/api/config/local/hf-token")
def save_local_token(req: LocalTokenRequest) -> dict:
    token = req.token.strip()
    set_hf_token(token)
    return local_config()


@app.delete("/api/config/local/hf-token")
def delete_local_token() -> dict:
    remove_hf_token()
    return local_config()


@app.get("/api/search/indexes")
def search_indexes() -> dict:
    return {"indexes": list_search_indexes()}


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
    job = create_clip_projection_job(projection_req)
    return {"job_id": job.job_id, "status": job.status, "message": "Started embedding/projection job for searchable index."}


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
    warnings = []
    if not paths:
        warnings.append("No supported image files were found.")
    return {
        "ok": True,
        "image_dir": str(resolved.relative_to(ROOT)),
        "absolute_path": str(resolved),
        "count": len(paths),
        "extensions": dict(sorted(extensions.items())),
        "warnings": warnings,
    }


@app.post("/api/jobs/clip-projection")
def create_clip_projection(req: ClipProjectionRequest) -> dict:
    job = create_clip_projection_job(req)
    return {"job_id": job.job_id, "status": job.status}


@app.get("/api/jobs")
def jobs() -> dict:
    return {"jobs": list_jobs()}


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
