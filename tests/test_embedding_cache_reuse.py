"""The cached-embedding path must reuse the manifest's recorded images.

This is the end-to-end version of the manifest fix. Encoders skip files they
cannot read, so a job's output can cover fewer images than the folder holds. The
second run must reuse exactly the images the vectors were built from — before
v6.0 it re-scanned the folder and zipped a longer path list against a shorter
vector array, quietly attributing rows to the wrong images.

The encoder, model spec and reducer are stubbed so the test needs neither torch
nor a real checkpoint.
"""

from __future__ import annotations

import csv
import json
import shutil
import time

import numpy as np
import pytest

import backend.jobs as jobs
from backend.encoders.registry import ModelSpec
from backend.runtime_dirs import IMG_DIR, OUTPUT_DIR, ROOT
from backend.schemas import ClipProjectionRequest

STUB_SPEC = ModelSpec(
    key="pytest_stub",
    family="Stub",
    label="Stub encoder",
    provider="pytest_stub",
    model_id="stub",
    pretrained="v1",
    requires=(),
)

SKIPPED_INDEX = 2
IMAGE_COUNT = 6


@pytest.fixture
def stub_pipeline(request, monkeypatch):
    """A folder of images plus an encoder that always drops one of them."""
    from PIL import Image

    folder = IMG_DIR / "_pytest_cache_reuse"
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(IMAGE_COUNT):
        Image.new("RGB", (16, 16), (index * 40 % 255, 60, 90)).save(folder / f"p{index}.jpg")

    def fake_encode(image_paths, spec, batch_size, progress=None):
        kept = [p for i, p in enumerate(image_paths) if i != SKIPPED_INDEX]
        vectors = np.random.default_rng(0).normal(0, 1, (len(kept), 8)).astype("float32")
        return vectors, kept

    def fake_reduce(embeddings, reducer, **kwargs):
        return np.column_stack([embeddings[:, 0], embeddings[:, 1]]).astype("float32")

    monkeypatch.setattr(jobs, "encode_images", fake_encode)
    monkeypatch.setattr(jobs, "get_model_spec", lambda *a, **k: STUB_SPEC)
    monkeypatch.setattr(jobs, "reduce_embeddings", fake_reduce)

    started_at = time.time()

    def cleanup():
        shutil.rmtree(folder, ignore_errors=True)
        for directory in ("embeddings", "projections"):
            for path in (OUTPUT_DIR / directory).glob("*pytest_stub*"):
                path.unlink(missing_ok=True)
        # Each run also writes a per-job debug report; without this the suite
        # leaves a growing pile of them behind in output/logs.
        for path in (OUTPUT_DIR / "logs").glob("clip-*"):
            try:
                if path.stat().st_mtime >= started_at - 1:
                    path.unlink(missing_ok=True)
            except OSError:
                pass

    request.addfinalizer(cleanup)
    return {"dir": "img/_pytest_cache_reuse"}


def _run(image_dir: str, use_cache: bool):
    request = ClipProjectionRequest(image_dir=image_dir, model_key="pytest_stub", use_cache=use_cache)
    job = jobs.create_clip_projection_job(request)
    deadline = time.time() + 30
    while time.time() < deadline:
        current = jobs.get_job(job.job_id)
        if current.status in {"completed", "failed", "cancelled"}:
            return current
        time.sleep(0.02)
    raise AssertionError("the stubbed job did not finish in time")


def _filenames(job) -> list[str]:
    with (ROOT / job.result_path).open(encoding="utf-8", newline="") as fh:
        return [row["filename"] for row in csv.DictReader(fh, delimiter="\t")]


def _manifest():
    matches = list((OUTPUT_DIR / "embeddings").glob("*pytest_stub*_manifest.json"))
    assert len(matches) == 1, matches
    return matches[0]


def test_first_run_excludes_the_skipped_image(stub_pipeline):
    job = _run(stub_pipeline["dir"], use_cache=False)
    assert job.status == "completed", job.error
    names = _filenames(job)
    assert len(names) == IMAGE_COUNT - 1
    assert f"p{SKIPPED_INDEX}.jpg" not in names


def test_manifest_records_the_encoded_paths_not_the_scan(stub_pipeline):
    _run(stub_pipeline["dir"], use_cache=False)
    manifest = json.loads(_manifest().read_text(encoding="utf-8"))
    assert manifest["count"] == IMAGE_COUNT - 1
    assert len(manifest["paths"]) == IMAGE_COUNT - 1
    assert not any(f"p{SKIPPED_INDEX}.jpg" in path for path in manifest["paths"])


def test_cached_run_reproduces_the_same_images_in_the_same_order(stub_pipeline):
    first = _run(stub_pipeline["dir"], use_cache=False)
    second = _run(stub_pipeline["dir"], use_cache=True)
    assert second.status == "completed", second.error
    assert _filenames(second) == _filenames(first)


def test_cached_run_reports_the_cache_stage(stub_pipeline):
    _run(stub_pipeline["dir"], use_cache=False)
    second = _run(stub_pipeline["dir"], use_cache=True)
    assert second.total == IMAGE_COUNT - 1


def test_a_legacy_manifest_without_paths_is_re_encoded(stub_pipeline):
    first = _run(stub_pipeline["dir"], use_cache=False)
    manifest_path = _manifest()
    legacy = json.loads(manifest_path.read_text(encoding="utf-8"))
    legacy.pop("paths")
    manifest_path.write_text(json.dumps(legacy), encoding="utf-8")

    second = _run(stub_pipeline["dir"], use_cache=True)
    assert second.status == "completed", second.error
    assert _filenames(second) == _filenames(first)
    # Re-encoding must leave a manifest that the next run *can* trust.
    assert "paths" in json.loads(manifest_path.read_text(encoding="utf-8"))


def test_a_manifest_listing_a_deleted_image_is_re_encoded(stub_pipeline):
    _run(stub_pipeline["dir"], use_cache=False)
    manifest_path = _manifest()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["paths"] = manifest["paths"] + ["img/_pytest_cache_reuse/ghost.jpg"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    second = _run(stub_pipeline["dir"], use_cache=True)
    assert second.status == "completed", second.error
    assert len(_filenames(second)) == IMAGE_COUNT - 1


def test_the_projection_sidecar_points_at_the_embedding_file(stub_pipeline):
    job = _run(stub_pipeline["dir"], use_cache=False)
    sidecar = (ROOT / job.result_path).with_suffix(".json")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert (ROOT / meta["embedding_path"]).is_file()
    assert meta["image_count"] == IMAGE_COUNT - 1
