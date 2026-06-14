from __future__ import annotations

import csv
import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from .cache import dataset_hash, write_manifest
from .diagnostics import user_facing_error, write_debug_report
from .encoders.base import encode_images
from .encoders.registry import get_model_spec
from .image_scan import scan_images
from .reducers import cluster_projection, reduce_embeddings
from .schemas import ClipProjectionRequest

from project_paths import get_project_root

ROOT = get_project_root()
OUTPUT = ROOT / "output"


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | running | completed | failed | cancelled | cancelling
    stage: str = "queued"
    done: int = 0
    total: int = 0
    message: str = "Queued"
    error: str | None = None
    result_path: str | None = None
    log_path: str | None = None
    debug_path: str | None = None
    recovery_hint: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    params: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


JOBS: dict[str, JobState] = {}
CANCEL_REQUESTS: set[str] = set()
LOCK = threading.Lock()


def _set(job_id: str, **updates) -> None:
    with LOCK:
        job = JOBS[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        job.updated_at = time.time()


def create_clip_projection_job(req: ClipProjectionRequest) -> JobState:
    job_id = f"clip-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    params = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    job = JobState(job_id=job_id, params=params)
    with LOCK:
        JOBS[job_id] = job
    thread = threading.Thread(target=_run_clip_projection, args=(job_id, req), daemon=True)
    thread.start()
    return job


def cancel_job(job_id: str) -> JobState | None:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.status in {"completed", "failed", "cancelled"}:
            return job
        CANCEL_REQUESTS.add(job_id)
        job.status = "cancelling"
        job.message = "Cancellation requested"
        job.updated_at = time.time()
        return job


def _check_cancel(job_id: str) -> None:
    with LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled("Job cancelled by user")


def get_job(job_id: str) -> JobState | None:
    with LOCK:
        return JOBS.get(job_id)


def list_jobs() -> list[dict]:
    with LOCK:
        return [job.to_dict() for job in sorted(JOBS.values(), key=lambda j: j.created_at, reverse=True)]


def list_projection_files() -> list[dict]:
    proj_dir = OUTPUT / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(proj_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime, reverse=True):
        items.append({
            "name": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "modified_at": path.stat().st_mtime,
            "size": path.stat().st_size,
        })
    return items


class JobCancelled(RuntimeError):
    pass


def _safe_image_dir(image_dir: str) -> Path:
    root = ROOT / "img"
    requested = (ROOT / image_dir).resolve() if image_dir != "img" else root.resolve()
    root_resolved = root.resolve()
    if requested == root_resolved or root_resolved in requested.parents:
        return requested
    raise ValueError("Image directory must be img/ or one of its subfolders.")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value)


def _run_clip_projection(job_id: str, req: ClipProjectionRequest) -> None:
    try:
        params = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        debug_path = write_debug_report(job_id, {"job_id": job_id, "request": params, "phase": "job_start"})
        _set(job_id, debug_path=str(debug_path.relative_to(ROOT)))
        _check_cancel(job_id)
        image_dir = _safe_image_dir(req.image_dir)
        image_dir_label = str(image_dir.relative_to(ROOT))
        _set(job_id, status="running", stage="scan", message=f"Scanning {image_dir_label}")
        paths = scan_images(image_dir)
        if not paths:
            raise RuntimeError(f"No images found in {image_dir_label}. Add images to img/ or choose a subfolder.")
        _set(job_id, total=len(paths), done=0, message=f"Found {len(paths)} images")
        _check_cancel(job_id)

        spec = get_model_spec(req.model_key, req.model, req.pretrained)
        if spec.status == "planned":
            raise RuntimeError("The selected model is planned and is not available for local embedding generation yet.")
        if spec.status == "unavailable":
            raise RuntimeError("The selected model is unavailable for local embedding generation.")
        if not spec.supports_image_embedding:
            raise RuntimeError("The selected model does not support image embeddings.")
        if not spec.supports_projection:
            raise RuntimeError("The selected model does not support projection generation.")
        _set(job_id, message=f"Selected model: {spec.label} ({spec.provider})")
        ds_hash = dataset_hash(paths, spec.key, spec.model_id, spec.pretrained)
        emb_dir = OUTPUT / "embeddings"
        proj_dir = OUTPUT / "projections"
        emb_dir.mkdir(parents=True, exist_ok=True)
        proj_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_tag = _safe_name(f"{spec.key}_{spec.model_id}_{spec.pretrained}")
        emb_path = emb_dir / f"{ds_hash}_{model_tag}.npy"
        cluster_tag = ""
        if req.cluster_enabled:
            if req.cluster_auto:
                cluster_tag = f"_clusters-auto-{req.cluster_min_k}-{req.cluster_max_k}"
            else:
                cluster_tag = f"_clusters-k{req.cluster_k}"
        projection_path = proj_dir / f"{timestamp}_{ds_hash}_{model_tag}_{req.reducer}{cluster_tag}_projection.tsv"
        projection_meta_path = proj_dir / f"{timestamp}_{ds_hash}_{model_tag}_{req.reducer}{cluster_tag}_projection.json"
        manifest_path = emb_dir / f"{ds_hash}_{model_tag}_manifest.json"

        if req.use_cache and emb_path.exists():
            _set(job_id, stage="cache", done=len(paths), total=len(paths), message="Loading cached embeddings")
            embeddings = np.load(emb_path)
            encoded_paths = paths
        else:
            def progress(done: int, total: int, msg: str) -> None:
                _check_cancel(job_id)
                _set(job_id, status="running", stage="embedding", done=done, total=total, message=msg)

            embeddings, encoded_paths = encode_images(paths, spec, req.batch_size, progress)
            if len(encoded_paths) != len(paths):
                paths = encoded_paths
                if not paths:
                    raise RuntimeError("No valid images could be encoded.")
            _check_cancel(job_id)
            np.save(emb_path, embeddings)
            write_manifest(manifest_path, {
                "dataset_hash": ds_hash,
                "model_key": spec.key,
                "model_family": spec.family,
                "model_label": spec.label,
                "model": spec.model_id,
                "pretrained": spec.pretrained,
                "provider": spec.provider,
                "image_dir": image_dir_label,
                "count": len(paths),
                "embedding_path": str(emb_path.relative_to(ROOT)),
                "created_at": time.time(),
            })

        _check_cancel(job_id)
        _set(job_id, stage="reduction", done=0, total=len(paths), message=f"Running {req.reducer.upper()} projection")
        coords = reduce_embeddings(
            embeddings,
            req.reducer,
            umap_n_neighbors=req.umap_n_neighbors,
            umap_min_dist=req.umap_min_dist,
            tsne_perplexity=req.tsne_perplexity,
            tsne_max_iter=req.tsne_max_iter,
        )
        _check_cancel(job_id)

        cluster_info = None
        cluster_warning = None
        if req.cluster_enabled:
            _set(job_id, stage="clustering", done=0, total=len(paths), message="Running optional K-Means clustering")
            try:
                cluster_info = cluster_projection(
                    coords,
                    auto=req.cluster_auto,
                    fixed_k=req.cluster_k,
                    min_k=req.cluster_min_k,
                    max_k=req.cluster_max_k,
                )
            except Exception as cluster_exc:
                cluster_warning = f"Clustering failed and projection was saved without clusters: {cluster_exc}"
                cluster_log_path = OUTPUT / "logs" / f"{job_id}-clustering.log"
                cluster_log_path.parent.mkdir(parents=True, exist_ok=True)
                cluster_log_path.write_text(traceback.format_exc(), encoding="utf-8")
                _set(job_id, message=cluster_warning, log_path=str(cluster_log_path.relative_to(ROOT)))
            _check_cancel(job_id)

        rows = []
        for idx, (path, xy) in enumerate(zip(paths, coords), start=1):
            row = {
                "filename": path.name,
                "relative_path": path.relative_to(ROOT).as_posix(),
                "x": float(xy[0]),
                "y": float(xy[1]),
                "model_key": spec.key,
                "model_family": spec.family,
                "embedding_model": spec.label,
                "model_id": spec.model_id,
                "pretrained": spec.pretrained,
                "provider": spec.provider,
                "reducer": req.reducer,
                "row_index": idx,
            }
            if cluster_info is not None:
                row["cluster"] = cluster_info["labels"][idx - 1]
                row["cluster_k"] = cluster_info["k"]
                row["cluster_score"] = round(float(cluster_info["score"]), 6)
                row["cluster_method"] = cluster_info["method"]
            rows.append(row)

        with projection_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

        projection_meta_path.write_text(json.dumps({
            "model_key": spec.key,
            "model_family": spec.family,
            "model_label": spec.label,
            "provider": spec.provider,
            "model_id": spec.model_id,
            "pretrained": spec.pretrained,
            "reducer": req.reducer,
            "image_dir": image_dir_label,
            "image_count": len(paths),
            "dataset_hash": ds_hash,
            "embedding_id": emb_path.stem,
            "embedding_path": str(emb_path.relative_to(ROOT)),
            "projection_path": str(projection_path.relative_to(ROOT)),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "clustering": {
                "enabled": bool(req.cluster_enabled),
                "method": "kmeans" if req.cluster_enabled else None,
                "auto": bool(req.cluster_auto) if req.cluster_enabled else None,
                "k": cluster_info.get("k") if cluster_info else (req.cluster_k if req.cluster_enabled and not req.cluster_auto else None),
                "score": round(float(cluster_info["score"]), 6) if cluster_info else None,
                "applied": cluster_info is not None,
                "warning": cluster_warning,
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        _set(
            job_id,
            status="completed",
            stage="completed",
            done=len(paths),
            total=len(paths),
            message="Projection completed" + (f" ({cluster_warning})" if cluster_warning else ""),
            result_path=str(projection_path.relative_to(ROOT)),
        )
    except JobCancelled as exc:
        with LOCK:
            CANCEL_REQUESTS.discard(job_id)
        _set(job_id, status="cancelled", stage="cancelled", error=None, message=str(exc))
    except Exception as exc:
        log_path = OUTPUT / "logs" / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        params = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        debug_path = write_debug_report(job_id, {
            "job_id": job_id,
            "request": params,
            "phase": "job_failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback_log": str(log_path.relative_to(ROOT)),
        })
        friendly = user_facing_error(exc)
        _set(
            job_id,
            status="failed",
            stage="failed",
            error=str(exc),
            recovery_hint=friendly,
            message=f"Failed: {friendly}",
            log_path=str(log_path.relative_to(ROOT)),
            debug_path=str(debug_path.relative_to(ROOT)),
        )
