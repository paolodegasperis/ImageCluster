"""Path-containment tests.

These guard the only functions in the project that decide whether a
caller-supplied string may touch the filesystem. They had no coverage before
v6.0, which is exactly backwards: everything else fails loudly, these fail
quietly and hand out files.
"""

from __future__ import annotations

import pytest

from backend.exports import _safe_source, _validated_projection
from backend.jobs import _safe_image_dir
from backend.runtime_dirs import IMG_DIR, OUTPUT_DIR, ROOT
from backend.search.index_store import safe_image_dir

TRAVERSAL_ATTEMPTS = [
    "../",
    "../../",
    "img/../..",
    "img/../../etc",
    "img/../output",
    "output",
    "output/embeddings",
    "..",
    "./..",
    "img/subfolder/../../..",
]


@pytest.mark.parametrize("attempt", TRAVERSAL_ATTEMPTS)
def test_jobs_image_dir_rejects_traversal(attempt):
    with pytest.raises(ValueError):
        _safe_image_dir(attempt)


@pytest.mark.parametrize("attempt", TRAVERSAL_ATTEMPTS)
def test_index_store_image_dir_rejects_traversal(attempt):
    with pytest.raises(ValueError):
        safe_image_dir(attempt)


@pytest.mark.parametrize("attempt", ["/etc/passwd", "C:/Windows/System32"])
def test_image_dir_rejects_absolute_paths(attempt):
    with pytest.raises(ValueError):
        _safe_image_dir(attempt)
    with pytest.raises(ValueError):
        safe_image_dir(attempt)


def test_image_dir_accepts_img_root_and_subfolders():
    assert _safe_image_dir("img") == IMG_DIR.resolve()
    assert safe_image_dir("img") == IMG_DIR.resolve()
    nested = _safe_image_dir("img/paintings")
    assert nested == (IMG_DIR / "paintings").resolve()
    # The folder need not exist yet; containment is what is being asserted.
    assert IMG_DIR.resolve() in nested.parents


def test_validated_projection_rejects_paths_outside_projections(tmp_path):
    for attempt in ["output/embeddings/x.npy", "config.json", "../config.json", "output"]:
        with pytest.raises(ValueError):
            _validated_projection(attempt)


def test_validated_projection_rejects_missing_file():
    with pytest.raises(ValueError, match="not found"):
        _validated_projection("output/projections/does-not-exist.tsv")


def test_validated_projection_accepts_real_file():
    projections = OUTPUT_DIR / "projections"
    projections.mkdir(parents=True, exist_ok=True)
    sample = projections / "_pytest_projection.tsv"
    sample.write_text("filename\trelative_path\n", encoding="utf-8")
    try:
        resolved = _validated_projection("output/projections/_pytest_projection.tsv")
        assert resolved == sample.resolve()
    finally:
        sample.unlink(missing_ok=True)


def test_safe_source_rejects_paths_outside_img():
    assert _safe_source("config.json") is None
    assert _safe_source("../config.json") is None
    assert _safe_source("output/local_settings.json") is None
    assert _safe_source("") is None


def test_safe_source_accepts_a_real_image_under_img():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    sample = IMG_DIR / "_pytest_sample.txt"
    sample.write_text("x", encoding="utf-8")
    try:
        resolved = _safe_source("img/_pytest_sample.txt")
        assert resolved == sample.resolve()
    finally:
        sample.unlink(missing_ok=True)


def test_safe_source_rejects_a_path_that_is_not_a_file():
    assert _safe_source("img") is None
    assert _safe_source("img/definitely-missing.jpg") is None


def test_root_is_the_project_folder():
    assert (ROOT / "backend").is_dir()
    assert IMG_DIR.parent == ROOT
