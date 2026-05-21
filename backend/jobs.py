from __future__ import annotations

import csv
import json
import os
import queue
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np

from .cache import dataset_fingerprint, dataset_hash, write_manifest
from .diagnostics import user_facing_error, write_debug_report
from .encoders.base import encode_images
from .encoders.registry import get_model_spec
from .image_scan import scan_images
from .reducers import cluster_projection, reduce_embeddings
from .runtime_dirs import JOB_LOGS_DIR, JOBS_DIR, OUTPUT_DIR, ROOT
from .schemas import ClipProjectionRequest
from .version import APP_VERSION

OUTPUT = OUTPUT_DIR
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
ACTIVE_STATUSES = {"queued", "running", "cancelling"}
MAX_WORKERS = max(1, int(os.environ.get("CLUSTERIMG_MAX_WORKERS", "1")))
QUEUE_ENABLED = os.environ.get("CLUSTERIMG_JOB_QUEUE_ENABLED", "1") != "0"


@dataclass
class JobState:
    job_id: str
    job_type: str = "clip_projection"
    status: str = "queued"
    stage: str = "queued"
    done: int = 0
    total: int = 0
    progress: float | None = None
    message: str = "Queued"
    current_step: str | None = None
    error: str | None = None
    warning: str | None = None
    result_path: str | None = None
    embedding_path: str | None = None
    search_index_path: str | None = None
    log_path: str | None = None
    debug_path: str | None = None
    recovery_hint: str | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    created_at_iso: str = field(default_factory=lambda: _iso_now())
    started_at_iso: str | None = None
    finished_at_iso: str | None = None
    updated_at_iso: str = field(default_factory=lambda: _iso_now())
    params: dict = field(default_factory=dict)
    runtime: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class JobStore:
    def __init__(self, directory: Path = JOBS_DIR):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def path_for(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def save(self, job: JobState) -> None:
        payload = job.to_dict()
        path = self.path_for(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def load_all(self) -> dict[str, JobState]:
        jobs: dict[str, JobState] = {}
        for path in sorted(self.directory.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                known = {field_name for field_name in JobState.__dataclass_fields__}
                job = JobState(**{key: value for key, value in data.items() if key in known})
                jobs[job.job_id] = job
            except Exception:
                continue
        return jobs


STORE = JobStore()
JOBS: dict[str, JobState] = {}
CANCEL_REQUESTS: set[str] = set()
LOCK = threading.RLock()
WORK_QUEUE: queue.Queue[str] = queue.Queue()
WORKERS_STARTED = False
RECOVERY_DONE = False


class JobCancelled(RuntimeError):
    pass


def initialize_jobs() -> None:
    global RECOVERY_DONE, WORKERS_STARTED
    with LOCK:
        if not RECOVERY_DONE:
            JOBS.update(STORE.load_all())
            changed: list[JobState] = []
            for job in JOBS.values():
                if job.status in {"running", "cancelling", "queued"}:
                    job.status = "interrupted"
                    job.stage = "interrupted"
                    job.message = "Backend restarted before this job finished."
                    job.error = None
                    job.finished_at = time.time()
                    job.finished_at_iso = _iso_now()
                    job.updated_at = time.time()
                    job.updated_at_iso = _iso_now()
                    changed.append(job)
            for job in changed:
                STORE.save(job)
            RECOVERY_DONE = True
        if not WORKERS_STARTED and QUEUE_ENABLED:
            for index in range(MAX_WORKERS):
                thread = threading.Thread(target=_worker_loop, name=f"clusterimg-job-worker-{index + 1}", daemon=True)
                thread.start()
            WORKERS_STARTED = True


def create_clip_projection_job(req: ClipProjectionRequest, job_type: str = "clip_projection") -> JobState:
    initialize_jobs()
    job_id = f"clip-{time.strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    params = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    log_path = JOB_LOGS_DIR / f"{job_id}.log"
    job = JobState(
        job_id=job_id,
        job_type=job_type,
        params=params,
        log_path=str(log_path.relative_to(ROOT)),
        runtime=_runtime_metadata(),
        message="Queued. Waiting for the local job worker.",
    )
    with LOCK:
        JOBS[job_id] = job
        STORE.save(job)
    _job_log(job_id, f"created job_type={job_type}")
    if QUEUE_ENABLED:
        WORK_QUEUE.put(job_id)
        _job_log(job_id, "queued")
    else:
        thread = threading.Thread(target=_execute_job, args=(job_id,), daemon=True)
        thread.start()
    return job


def cancel_job(job_id: str) -> JobState | None:
    initialize_jobs()
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return None
        if job.status in TERMINAL_STATUSES:
            return job
        CANCEL_REQUESTS.add(job_id)
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "Cancelled before execution."
            job.finished_at = time.time()
            job.finished_at_iso = _iso_now()
        else:
            job.status = "cancelling"
            job.message = "Cancellation requested. The job will stop at the next safe checkpoint."
        job.updated_at = time.time()
        job.updated_at_iso = _iso_now()
        STORE.save(job)
    _job_log(job_id, "cancellation requested")
    return job


def get_job(job_id: str) -> JobState | None:
    initialize_jobs()
    with LOCK:
        return JOBS.get(job_id)


def list_jobs(status: str | None = None, job_type: str | None = None, limit: int | None = None, newest_first: bool = True) -> list[dict]:
    initialize_jobs()
    with LOCK:
        jobs = list(JOBS.values())
    if status:
        jobs = [job for job in jobs if job.status == status]
    if job_type:
        jobs = [job for job in jobs if job.job_type == job_type]
    jobs.sort(key=lambda j: j.created_at, reverse=newest_first)
    if limit:
        jobs = jobs[:limit]
    return [job.to_dict() for job in jobs]


def job_counts() -> dict:
    initialize_jobs()
    with LOCK:
        active = len([job for job in JOBS.values() if job.status in {"running", "cancelling"}])
        queued = len([job for job in JOBS.values() if job.status == "queued"])
    return {
        "active": active,
        "queued": queued,
        "max_workers": MAX_WORKERS,
        "queue_enabled": QUEUE_ENABLED,
        "job_store": str(JOBS_DIR),
    }


def list_projection_files() -> list[dict]:
    proj_dir = OUTPUT / "projections"
    proj_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for path in sorted(proj_dir.glob("*.tsv"), key=lambda p: p.stat().st_mtime, reverse=True):
        validation = validate_projection_tsv(path)
        items.append({
            "name": path.name,
            "relative_path": path.relative_to(ROOT).as_posix(),
            "modified_at": path.stat().st_mtime,
            "size": path.stat().st_size,
            "valid": validation["valid"],
            "validation_error": validation.get("error"),
        })
    return items


def validate_projection_tsv(path: Path) -> dict:
    required = {"filename", "relative_path", "x", "y", "reducer"}
    try:
        if not path.exists():
            return {"valid": False, "error": "missing file"}
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh, delimiter="\t")
            header = next(reader, [])
        columns = set(header)
        if not ({"model_key"} & columns or {"embedding_model", "model_family"} & columns):
            return {"valid": False, "error": "missing model column"}
        missing = sorted(required - columns)
        if missing:
            return {"valid": False, "error": f"missing columns: {', '.join(missing)}"}
        return {"valid": True, "columns": header}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}


def _worker_loop() -> None:
    while True:
        job_id = WORK_QUEUE.get()
        try:
            _execute_job(job_id)
        finally:
            WORK_QUEUE.task_done()


def _execute_job(job_id: str) -> None:
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        if job.status == "cancelled":
            _job_log(job_id, "skipping cancelled queued job")
            return
        if job.status != "queued":
            return
        params = dict(job.params)
        job.status = "running"
        job.stage = "starting"
        job.current_step = "starting"
        job.message = "Starting job."
        job.started_at = time.time()
        job.started_at_iso = _iso_now()
        job.updated_at = time.time()
        job.updated_at_iso = _iso_now()
        STORE.save(job)
    _job_log(job_id, "started")
    req = ClipProjectionRequest(**params)
    _run_clip_projection(job_id, req)


def _set(job_id: str, **updates) -> None:
    with LOCK:
        job = JOBS[job_id]
        for key, value in updates.items():
            setattr(job, key, value)
        if "stage" in updates and "current_step" not in updates:
            job.current_step = updates["stage"]
        total = getattr(job, "total", 0) or 0
        done = getattr(job, "done", 0) or 0
        job.progress = round(done / total, 4) if total else None
        job.updated_at = time.time()
        job.updated_at_iso = _iso_now()
        STORE.save(job)
    if "message" in updates:
        _job_log(job_id, str(updates["message"]))


def _check_cancel(job_id: str) -> None:
    with LOCK:
        if job_id in CANCEL_REQUESTS:
            raise JobCancelled("Job cancelled by user")


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

        manifest = _load_manifest(manifest_path)
        can_use_cache = bool(
            req.use_cache
            and emb_path.exists()
            and manifest.get("complete", True)
            and manifest.get("count") == len(paths)
        )
        if can_use_cache:
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
            _atomic_save_npy(emb_path, embeddings)
            write_manifest(manifest_path, {
                "dataset_hash": ds_hash,
                "dataset_fingerprint": dataset_fingerprint(paths),
                "model_key": spec.key,
                "model_family": spec.family,
                "model_label": spec.label,
                "model": spec.model_id,
                "pretrained": spec.pretrained,
                "provider": spec.provider,
                "image_dir": image_dir_label,
                "count": len(paths),
                "embedding_path": str(emb_path.relative_to(ROOT)),
                "app_version": APP_VERSION,
                "created_at": _iso_now(),
                "complete": True,
            })

        current_artifacts = dict(JOBS[job_id].artifacts) if job_id in JOBS else {}
        current_artifacts["embedding_path"] = str(emb_path.relative_to(ROOT))
        _set(job_id, embedding_path=str(emb_path.relative_to(ROOT)), artifacts=current_artifacts)
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
                cluster_log_path = JOB_LOGS_DIR / f"{job_id}-clustering.log"
                cluster_log_path.parent.mkdir(parents=True, exist_ok=True)
                cluster_log_path.write_text(traceback.format_exc(), encoding="utf-8")
                _set(job_id, warning=cluster_warning, message=cluster_warning, log_path=str(cluster_log_path.relative_to(ROOT)))
            _check_cancel(job_id)

        rows = []
        for idx, (path, xy) in enumerate(zip(paths, coords), start=1):
            _check_cancel(job_id)
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

        _atomic_write_tsv(projection_path, rows)
        projection_meta = {
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
            "artifact_status": "complete",
            "app_version": APP_VERSION,
            "created_at": _iso_now(),
            "clustering": {
                "enabled": bool(req.cluster_enabled),
                "method": "kmeans" if req.cluster_enabled else None,
                "auto": bool(req.cluster_auto) if req.cluster_enabled else None,
                "k": cluster_info.get("k") if cluster_info else (req.cluster_k if req.cluster_enabled and not req.cluster_auto else None),
                "score": round(float(cluster_info["score"]), 6) if cluster_info else None,
                "applied": cluster_info is not None,
                "warning": cluster_warning,
            },
        }
        _atomic_write_json(projection_meta_path, projection_meta)

        _set(
            job_id,
            status="completed",
            stage="completed",
            done=len(paths),
            total=len(paths),
            message="Projection completed" + (f" ({cluster_warning})" if cluster_warning else ""),
            warning=cluster_warning,
            result_path=str(projection_path.relative_to(ROOT)),
            artifacts={
                "projection_tsv": str(projection_path.relative_to(ROOT)),
                "projection_metadata": str(projection_meta_path.relative_to(ROOT)),
                "embedding_path": str(emb_path.relative_to(ROOT)),
                "embedding_manifest": str(manifest_path.relative_to(ROOT)),
            },
            finished_at=time.time(),
            finished_at_iso=_iso_now(),
        )
        _job_log(job_id, "completed")
    except JobCancelled as exc:
        with LOCK:
            CANCEL_REQUESTS.discard(job_id)
        _set(job_id, status="cancelled", stage="cancelled", error=None, message=str(exc), finished_at=time.time(), finished_at_iso=_iso_now())
        _job_log(job_id, "cancelled")
    except Exception as exc:
        log_path = JOB_LOGS_DIR / f"{job_id}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("\n" + traceback.format_exc())
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
            finished_at=time.time(),
            finished_at_iso=_iso_now(),
        )
        _job_log(job_id, f"failed: {friendly}")


def _atomic_write_tsv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        np.save(fh, array)
    os.replace(tmp, path)


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _runtime_metadata() -> dict:
    return {
        "app_version": APP_VERSION,
        "python_version": sys.version,
        "platform": sys.platform,
        "max_workers": MAX_WORKERS,
    }


def _job_log(job_id: str, message: str) -> None:
    JOB_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_iso_now()} {message}\n"
    with (JOB_LOGS_DIR / f"{job_id}.log").open("a", encoding="utf-8") as fh:
        fh.write(line)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


initialize_jobs()
