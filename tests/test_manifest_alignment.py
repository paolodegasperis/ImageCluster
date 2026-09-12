"""The manifest must pin embeddings to the exact images they were built from.

Encoders skip files they cannot read, so the encoded set can be a strict subset
of a directory scan. Before v6.0 only a count was stored, and both the cached
embedding path and text search re-derived the order from a fresh scan: an added
file plus a deleted file left the counts equal and every row attributed to the
wrong image, silently.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.cache import manifest_paths, read_manifest, write_manifest
from backend.jobs import _usable_cached_paths
from backend.runtime_dirs import ROOT
from backend.search.semantic_search import _index_paths_for


def test_manifest_paths_returns_none_for_legacy_manifests():
    assert manifest_paths({"count": 3}) is None
    assert manifest_paths({}) is None
    assert manifest_paths(None) is None


def test_manifest_paths_rejects_malformed_entries():
    assert manifest_paths({"paths": "img/a.jpg"}) is None
    assert manifest_paths({"paths": []}) is None
    assert manifest_paths({"paths": ["img/a.jpg", 7]}) is None


def test_manifest_paths_resolves_against_the_project_root():
    resolved = manifest_paths({"paths": ["img/a.jpg", "img/b.jpg"]})
    assert resolved == [ROOT / "img/a.jpg", ROOT / "img/b.jpg"]


def test_read_manifest_tolerates_corrupt_json(tmp_path):
    broken = tmp_path / "broken_manifest.json"
    broken.write_text("{not json", encoding="utf-8")
    assert read_manifest(broken) is None
    assert read_manifest(tmp_path / "absent.json") is None


def test_write_then_read_manifest_roundtrip(tmp_path):
    target = tmp_path / "sub" / "m.json"
    write_manifest(target, {"count": 2, "paths": ["img/a.jpg", "img/b.jpg"]})
    assert json.loads(target.read_text(encoding="utf-8"))["count"] == 2
    assert len(manifest_paths(read_manifest(target))) == 2


def _write_pair(tmp_path, paths, rows):
    manifest = tmp_path / "m_manifest.json"
    emb = tmp_path / "m.npy"
    write_manifest(manifest, {"count": len(paths), "paths": paths})
    np.save(emb, np.zeros((rows, 4), dtype="float32"))
    return manifest, emb


def test_cached_paths_rejected_when_manifest_has_no_path_list(tmp_path):
    manifest = tmp_path / "legacy_manifest.json"
    emb = tmp_path / "legacy.npy"
    write_manifest(manifest, {"count": 2})
    np.save(emb, np.zeros((2, 4), dtype="float32"))
    assert _usable_cached_paths(manifest, emb) is None


def test_cached_paths_rejected_when_counts_disagree(tmp_path, image_files):
    rel = [p.as_posix() for p in image_files]
    manifest, emb = _write_pair(tmp_path, rel, rows=2)
    assert _usable_cached_paths(manifest, emb) is None


def test_cached_paths_rejected_when_a_listed_image_is_gone(tmp_path, image_files):
    rel = [p.as_posix() for p in image_files]
    manifest, emb = _write_pair(tmp_path, rel, rows=3)
    image_files[1].unlink()
    assert _usable_cached_paths(manifest, emb) is None


def test_index_paths_raises_instead_of_misaligning(tmp_path, image_files):
    index = {"manifest_path": None, "image_dir": "img"}
    # A recorded list whose length disagrees with the vectors must fail loudly.
    manifest = tmp_path / "idx_manifest.json"
    write_manifest(manifest, {"paths": [p.as_posix() for p in image_files]})
    index = {
        "manifest_path": manifest.relative_to(manifest.anchor).as_posix(),
        "image_dir": "img",
    }
    # Point the index at the real file through ROOT so index_image_paths finds it.
    import backend.search.index_store as store

    original = store.ROOT
    try:
        store.ROOT = manifest.parent
        index["manifest_path"] = manifest.name
        with pytest.raises(RuntimeError, match="inconsistent"):
            _index_paths_for(index, vector_count=99)
        resolved = _index_paths_for(index, vector_count=3)
        assert len(resolved) == 3
    finally:
        store.ROOT = original
