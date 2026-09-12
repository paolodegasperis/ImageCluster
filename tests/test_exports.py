"""Cluster export behaviour, including the deterministic collision suffix.

The suffix used to come from ``hash()``, which Python randomises per process, so
re-running the same export produced different filenames every time.
"""

from __future__ import annotations

import pytest

from backend.exports import _collision_suffix, _safe_name, export_clusters
from backend.runtime_dirs import IMG_DIR, OUTPUT_DIR


def test_safe_name_strips_separators_and_never_returns_empty():
    assert _safe_name("cluster 1") == "cluster-1"
    assert _safe_name("../../etc") == "etc"
    assert _safe_name("///") == "export"
    assert _safe_name("") == "export"


def test_collision_suffix_is_stable_across_calls():
    first = _collision_suffix("img/a/painting.jpg")
    second = _collision_suffix("img/a/painting.jpg")
    assert first == second
    assert first != _collision_suffix("img/b/painting.jpg")
    assert len(first) == 8


def _projection(tmp_name: str, rows: list[str]) -> str:
    projections = OUTPUT_DIR / "projections"
    projections.mkdir(parents=True, exist_ok=True)
    path = projections / tmp_name
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return f"output/projections/{tmp_name}"


def test_export_requires_a_cluster_column():
    rel = _projection("_pytest_nocluster.tsv", ["filename\trelative_path", "a.jpg\timg/a.jpg"])
    try:
        with pytest.raises(ValueError, match="no clusters"):
            export_clusters(rel)
    finally:
        (OUTPUT_DIR / "projections" / "_pytest_nocluster.tsv").unlink(missing_ok=True)


def test_export_rejects_an_empty_projection():
    rel = _projection("_pytest_empty.tsv", ["filename\trelative_path\tcluster"])
    try:
        with pytest.raises(ValueError, match="empty"):
            export_clusters(rel)
    finally:
        (OUTPUT_DIR / "projections" / "_pytest_empty.tsv").unlink(missing_ok=True)


def test_export_reports_an_unknown_cluster():
    rel = _projection(
        "_pytest_cluster.tsv",
        ["filename\trelative_path\tcluster", "a.jpg\timg/a.jpg\t1"],
    )
    try:
        with pytest.raises(ValueError, match="No images found for cluster"):
            export_clusters(rel, cluster="9")
    finally:
        (OUTPUT_DIR / "projections" / "_pytest_cluster.tsv").unlink(missing_ok=True)


def test_export_copies_real_files_and_skips_missing_ones(tmp_path):
    import shutil

    from PIL import Image

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    real = IMG_DIR / "_pytest_export.jpg"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(real)
    rel = _projection(
        "_pytest_export.tsv",
        [
            "filename\trelative_path\tcluster",
            "_pytest_export.jpg\timg/_pytest_export.jpg\t1",
            "ghost.jpg\timg/ghost.jpg\t1",
            "outside.jpg\tconfig.json\t1",
        ],
    )
    dest = None
    try:
        result = export_clusters(rel, cluster="1", export_name="_pytest_run")
        assert result["exported_count"] == 1
        assert result["skipped_count"] == 2
        assert result["folders"] == {"cluster_1": 1}
        # _safe_name strips leading separators, so "_pytest_run" lands as "pytest_run".
        assert result["dest"].endswith("pytest_run")
        dest = OUTPUT_DIR / "exports" / "pytest_run"
        assert (dest / "cluster_1" / "_pytest_export.jpg").is_file()
    finally:
        real.unlink(missing_ok=True)
        (OUTPUT_DIR / "projections" / "_pytest_export.tsv").unlink(missing_ok=True)
        if dest and dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
