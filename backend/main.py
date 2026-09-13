from __future__ import annotations

import csv
import hashlib
import io
import re
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from .analysis import (
    cluster_summaries,
    compare_images,
    duplicate_groups,
    outlier_ranking,
    similar_images,
)
from .app_config import clustering_defaults, search_defaults, view_defaults
from .app_config import get as config_get
from .config_store import apply_hf_token_to_environment, remove_hf_token, set_hf_token
from .dependency_check import check_dependencies, optional_dependency_report, pytorch_install_advice
from .diagnostics import collect_system_report
from .disagreement import model_disagreement
from .encoders.registry import get_model_spec, public_model_registry
from .exports import _safe_name, export_clusters
from .housekeeping import THUMB_CACHE_DIR, prune_thumb_cache, startup_sweep
from .image_scan import list_image_folders, scan_images
from .jobs import cancel_job, create_clip_projection_job, get_job, list_jobs, list_projection_files
from .pdf_export import (
    PDF_AVAILABLE,
    PDF_MISSING_MESSAGE,
    SheetItem,
    build_image_sheet,
)
from .pdf_export import stamp as pdf_stamp
from .runtime_dirs import APP_DIR, IMG_DIR, OUTPUT_DIR, ROOT, ensure_runtime_dirs
from .schemas import (
    ClipProjectionRequest,
    ClusterSummaryRequest,
    CompareImagesRequest,
    ConceptArithmeticRequest,
    ConceptScoresRequest,
    DisagreementRequest,
    DistinguishRequest,
    DuplicateGroupsRequest,
    ExportClustersRequest,
    ImageSheetRequest,
    LocalTokenRequest,
    MosaicRequest,
    NeighborGraphRequest,
    OutlierRequest,
    SemanticAxisPreviewRequest,
    SimilarImagesRequest,
)
from .search import list_search_indexes, run_text_search
from .search.schemas import RebuildSearchIndexRequest, TextSearchRequest
from .semantic import (
    DEFAULT_VOCABULARY,
    concept_arithmetic,
    concept_scores,
    distinguish,
    semantic_axis_preview,
)
from .version import __app_name__, __version__
from .views import mosaic_view, neighbor_graph_view, projection_supports_graph, validated_projection

ensure_runtime_dirs()
startup_sweep()

app = FastAPI(title=__app_name__, version=__version__)
app.mount("/static", StaticFiles(directory=APP_DIR), name="static")
app.mount("/img", StaticFiles(directory=IMG_DIR), name="img")
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


_ASSET_URL = re.compile(r'(?P<attr>href|src)="(?P<url>/static/[^"?]+)"')


def _page(name: str) -> str:
    """Serve a page with its own CSS/JS stamped with the app version.

    Without this, a browser that cached the stylesheet from a previous version
    keeps using it after an update: the new markup arrives, the styles that go
    with it do not, and the page looks broken for no visible reason. Stamping the
    version into the URL makes each release a distinct resource.
    """
    html = (APP_DIR / name).read_text(encoding="utf-8")
    return _ASSET_URL.sub(lambda m: f'{m.group("attr")}="{m.group("url")}?v={__version__}"', html)


@app.get("/", response_class=HTMLResponse)
def dashboard() -> str:
    return _page("index.html")


@app.get("/clip", response_class=HTMLResponse)
def clip() -> str:
    return _page("clip_projection.html")


@app.get("/integrated", include_in_schema=False)
def integrated() -> RedirectResponse:
    return RedirectResponse(url="/clip", status_code=302)


@app.get("/models", response_class=HTMLResponse)
def models_page() -> str:
    return _page("models_tokens.html")


THUMB_CACHE = THUMB_CACHE_DIR


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
        key = hashlib.sha1(f"{candidate}|{width}|{int(stat.st_mtime)}|{stat.st_size}".encode()).hexdigest()
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
            prune_thumb_cache()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - decode/format errors are environment-specific
        raise HTTPException(status_code=415, detail=f"Cannot render thumbnail: {exc}") from exc
    return Response(content=data, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/api/status")
def status() -> dict:
    ok, report = check_dependencies()
    return {
        "ok": ok,
        "dependencies": [{"name": d.name, "installed": d.installed, "required": d.required, "note": d.note} for d in report],
        "default_image_dir": config_get("default_image_dir", "img"),
        "version": __version__,
        "app_name": __app_name__,
        "features": {
            "optional_clustering": True,
            "model_registry": True,
            "debug_reports": bool(config_get("debug.json_debug_reports", True)),
            "embedding_recovery": bool(config_get("debug.batch_retry", True)),
            "semantic_search": bool(config_get("semantic_search.enabled", True)),
            "grid_mosaic": bool(config_get("views.grid_mosaic_enabled", True)),
            "neighbor_graph": bool(config_get("views.neighbor_graph_enabled", True)),
        },
        "optional_dependencies": optional_dependency_report(),
        "install_advice": pytorch_install_advice(),
    }


@app.get("/api/config")
def ui_config() -> dict:
    """Defaults the UI should start from, so config.json actually configures the app."""
    return {
        "version": __version__,
        "app_name": __app_name__,
        "default_image_dir": config_get("default_image_dir", "img"),
        "default_model_key": config_get("default_model_key", "openclip_vit_b_32"),
        "default_reducer": config_get("default_reducer", "umap"),
        "batch_size": config_get("batch_size", 32),
        "clustering": clustering_defaults(),
        "semantic_search": search_defaults(),
        "views": view_defaults(),
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
    if not config_get("semantic_search.enabled", True):
        raise HTTPException(status_code=403, detail="Semantic search is disabled in config.json.")
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
    if not config_get("semantic_search.enabled", True):
        raise HTTPException(status_code=403, detail="Semantic search is disabled in config.json.")
    try:
        spec = get_model_spec(req.model_key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not spec.supports_text_search:
        raise HTTPException(status_code=400, detail="This model can create image projections but does not support text search.")
    projection_req = ClipProjectionRequest(image_dir=req.image_dir, model_key=req.model_key, use_cache=True)
    job = create_clip_projection_job(projection_req)
    return {"job_id": job.job_id, "status": job.status, "message": "Started embedding/projection job for searchable index."}


@app.get("/graph", response_class=HTMLResponse)
def neighbor_graph_page() -> str:
    return _page("neighbor_graph.html")


@app.post("/api/views/mosaic")
def views_mosaic(req: MosaicRequest) -> dict:
    """Grid-mosaic assignment for a saved projection."""
    if not config_get("views.grid_mosaic_enabled", True):
        raise HTTPException(status_code=403, detail="The grid mosaic view is disabled in config.json.")
    try:
        return mosaic_view(
            req.projection_path,
            columns=req.columns,
            aspect=req.aspect,
            algorithm=req.algorithm,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/views/neighbor-graph")
def views_neighbor_graph(req: NeighborGraphRequest) -> dict:
    """k-NN graph over the embeddings a saved projection was built from."""
    if not config_get("views.neighbor_graph_enabled", True):
        raise HTTPException(status_code=403, detail="The neighbor graph view is disabled in config.json.")
    try:
        return neighbor_graph_view(
            req.projection_path,
            k=req.k,
            min_similarity=req.min_similarity,
            mutual_only=req.mutual_only,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/views/neighbor-graph/available")
def views_neighbor_graph_available(path: str) -> dict:
    """Whether a given projection can produce a graph, and why not when it cannot."""
    return projection_supports_graph(path)


# --- Analysis over a saved projection's embeddings -------------------------
# These need no encoder: the vectors are already on disk, which is why they also
# work for image-only models that cannot do text search.


@app.post("/api/analysis/similar")
def analysis_similar(req: SimilarImagesRequest) -> dict:
    """Images most similar to one image already in the projection."""
    try:
        return similar_images(
            req.projection_path,
            relative_path=req.relative_path,
            index=req.index,
            top_k=req.top_k,
            min_similarity=req.min_similarity,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/duplicates")
def analysis_duplicates(req: DuplicateGroupsRequest) -> dict:
    """Groups of images that are effectively the same picture."""
    try:
        return duplicate_groups(req.projection_path, threshold=req.threshold, max_pairs=req.max_pairs)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/outliers")
def analysis_outliers(req: OutlierRequest) -> dict:
    """Images ranked by how isolated they are from the rest of the collection."""
    try:
        return outlier_ranking(req.projection_path, k=req.k, top_k=req.top_k)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# --- The shared text-image space ------------------------------------------
# Unlike /api/analysis/*, these load the model: a phrase has to be encoded before
# anything can be compared with it. The first call of a session pays that cost.


@app.get("/api/semantic/vocabulary")
def semantic_vocabulary() -> dict:
    """The default term list used to put a difference between groups into words."""
    return {"terms": list(DEFAULT_VOCABULARY)}


@app.post("/api/semantic/concept-scores")
def semantic_concept_scores(req: ConceptScoresRequest) -> dict:
    """Score every image against one concept, for shading the plot by it."""
    try:
        return concept_scores(req.projection_path, phrases=req.phrases, normalize=req.normalize)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/semantic/axis-preview")
def semantic_axes_preview(req: SemanticAxisPreviewRequest) -> dict:
    """Coordinates on two named axes, without re-running the projection."""
    try:
        return semantic_axis_preview(
            req.projection_path,
            x_positive=req.x_positive,
            x_negative=req.x_negative,
            y_positive=req.y_positive,
            y_negative=req.y_negative,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/semantic/distinguish")
def semantic_distinguish(req: DistinguishRequest) -> dict:
    """What separates one group of images from another, in words."""
    try:
        return distinguish(
            req.projection_path,
            paths_a=req.paths_a,
            paths_b=req.paths_b,
            vocabulary=req.vocabulary,
            top_k=req.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/semantic/arithmetic")
def semantic_arithmetic(req: ConceptArithmeticRequest) -> dict:
    """An image or phrase, minus an idea, plus another."""
    try:
        return concept_arithmetic(
            req.projection_path,
            relative_path=req.relative_path,
            base_phrases=req.base_phrases,
            add=req.add,
            subtract=req.subtract,
            strength=req.strength,
            top_k=req.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/disagreement")
def analysis_disagreement(req: DisagreementRequest) -> dict:
    """Images that two models of the same folder place differently."""
    try:
        return model_disagreement(
            req.projection_a,
            req.projection_b,
            neighbours=req.neighbours,
            top_k=req.top_k,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/lens", response_class=HTMLResponse)
def lens_page() -> str:
    return _page("lens.html")


BRAND_ICON = ROOT / "icone" / "ImageClusetIcon.ico"


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    """The project's own icon, for the tab and for the request browsers make anyway.

    Browsers ask for /favicon.ico on every page whether or not a page declares
    one, so without this every single page view logged a 404 and the tab showed
    a blank sheet. The file is the same icon the desktop launcher uses, so the
    app looks like itself in the tab strip.
    """
    if not BRAND_ICON.is_file():
        # A missing brand asset is not worth a 500; the tab simply stays blank.
        return Response(status_code=204)
    return FileResponse(
        BRAND_ICON,
        media_type="image/x-icon",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/guide", response_class=HTMLResponse)
def guide_page() -> str:
    """Plain-language account of what every function tells you, and what it cannot."""
    return _page("guide.html")


@app.get("/guide/it", response_class=HTMLResponse)
def guide_page_it() -> str:
    """The same account in Italian, for readers who are scholars rather than developers.

    A parallel file rather than a branch or a string catalogue: this page is
    prose with no logic and no interpolation, so a second copy costs nothing to
    serve and cannot drift from any code. English stays the source - where the
    two disagree the English one is right - and the interface itself is
    unchanged, which the page says in its own opening lines so that nobody
    expects otherwise.
    """
    return _page("guide.it.html")


@app.post("/api/analysis/compare")
def analysis_compare(req: CompareImagesRequest) -> dict:
    """How related two images of the same projection actually are."""
    try:
        return compare_images(
            req.projection_path,
            path_a=req.path_a,
            path_b=req.path_b,
            neighbours=req.neighbours,
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/analysis/clusters")
def analysis_clusters(req: ClusterSummaryRequest) -> dict:
    """Size, cohesion and representative image (medoid) for each cluster."""
    try:
        return cluster_summaries(req.projection_path)
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/exports/sheet")
def export_image_sheet(req: ImageSheetRequest) -> Response:
    """A result, as a PDF contact sheet the images can actually be read in.

    One route serves every surface that produces a set of images - a community,
    a neighbourhood, a similarity result - so the documents they generate are
    the same artefact and can be filed together.

    What could not be included is reported in headers rather than dropped in
    silence: a sheet holding fewer images than the result it came from would be
    quietly wrong, and the caller shows the difference.
    """
    if not PDF_AVAILABLE:
        raise HTTPException(status_code=503, detail=PDF_MISSING_MESSAGE)
    try:
        # Provenance must be a real projection: this path is quoted in the
        # sheet's heading, and it is caller input like everything else here.
        validated_projection(req.projection_path)
        result = build_image_sheet(
            title=req.title,
            subtitle=req.subtitle,
            meta_lines=list(req.meta_lines),
            items=[
                SheetItem(relative_path=i.relative_path, caption=i.caption, detail=i.detail)
                for i in req.items
            ],
        )
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    name = _safe_name(req.filename or f"imagecluster_sheet_{pdf_stamp()}")
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    return Response(
        content=result.pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{name}"',
            "X-Sheet-Included": str(result.included),
            "X-Sheet-Skipped": str(len(result.skipped)),
            "X-Sheet-Undersized": str(len(result.undersized)),
        },
    )


@app.post("/api/clusters/export")
def clusters_export(req: ExportClustersRequest) -> dict:
    try:
        return export_clusters(req.projection_path, req.cluster, req.export_name)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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
    path = ROOT / job.result_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The result file is no longer on disk.")
    return _read_tsv_result(path, {"job_id": job_id, "result_path": job.result_path})


@app.get("/api/projections/read")
def projection_read(path: str) -> JSONResponse:
    requested = (ROOT / path).resolve()
    output_root = (OUTPUT_DIR / "projections").resolve()
    if requested == output_root or output_root not in requested.parents:
        raise HTTPException(status_code=400, detail="Projection path must be a file inside output/projections/.")
    # exists() is true for a directory, and opening one raises rather than 404s.
    if not requested.is_file():
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
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The result file is no longer on disk.")
    return FileResponse(path, filename=path.name, media_type="text/tab-separated-values")


@app.get("/api/jobs/{job_id}/log")
def job_log(job_id: str) -> PlainTextResponse:
    job = get_job(job_id)
    if job is None or not job.log_path:
        raise HTTPException(status_code=404, detail="Log not found")
    path = ROOT / job.log_path
    if not path.is_file():
        # Logs are pruned by housekeeping while the job is still in memory.
        raise HTTPException(status_code=404, detail="The log for this job has been pruned.")
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
