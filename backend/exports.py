"""Export the images of a projection's clusters into folders.

Copies the original image files (under ``img/``) into
``output/exports/<name>/cluster_<id>/`` so an archive can be reorganized
according to the clustering produced by any embedding or visual-feature model.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import time
from pathlib import Path
from typing import Any

from .runtime_dirs import IMG_DIR, OUTPUT_DIR, ROOT


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-._")
    return cleaned or "export"


def _collision_suffix(relative_path: str) -> str:
    """Stable disambiguator for same-named images from different subfolders.

    Python randomises ``hash()`` per process, so the previous implementation gave
    the same image a different filename on every export run.
    """
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]


def _validated_projection(projection_path: str) -> Path:
    requested = (ROOT / projection_path).resolve()
    proj_root = (OUTPUT_DIR / "projections").resolve()
    if requested != proj_root and proj_root not in requested.parents:
        raise ValueError("Projection path must be inside output/projections/.")
    if not requested.is_file():
        raise ValueError("Projection file not found.")
    return requested


def _read_cluster_rows(projection_path: Path) -> list[dict[str, str]]:
    with projection_path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def _safe_source(relative_path: str) -> Path | None:
    """Resolve a TSV ``relative_path`` to a real file under img/, or None."""
    if not relative_path:
        return None
    candidate = (ROOT / relative_path).resolve()
    img_root = IMG_DIR.resolve()
    if candidate != img_root and img_root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def export_clusters(projection_path: str, cluster: str = "all", export_name: str | None = None) -> dict[str, Any]:
    proj = _validated_projection(projection_path)
    rows = _read_cluster_rows(proj)
    if not rows:
        raise ValueError("The projection file is empty.")
    if "cluster" not in rows[0]:
        raise ValueError("This projection has no clusters. Re-run it with clustering enabled before exporting.")

    selected = str(cluster).strip()
    wants_all = selected.lower() == "all" or selected == ""

    base_name = _safe_name(export_name) if export_name else f"{proj.stem}_{time.strftime('%Y%m%d-%H%M%S')}"
    dest_root = OUTPUT_DIR / "exports" / base_name

    folders: dict[str, int] = {}
    skipped: list[str] = []
    exported = 0
    matched_target = False

    for row in rows:
        label = (row.get("cluster") or "").strip()
        if not label:
            continue
        if not wants_all and label != selected:
            continue
        matched_target = True
        source = _safe_source(row.get("relative_path", ""))
        if source is None:
            skipped.append(row.get("relative_path", "<missing>"))
            continue
        cluster_dir = dest_root / f"cluster_{_safe_name(label)}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        target = cluster_dir / source.name
        # Avoid clobbering same-named files coming from different subfolders.
        if target.exists() and target.stat().st_size != source.stat().st_size:
            target = cluster_dir / f"{source.stem}_{_collision_suffix(row.get('relative_path', ''))}{source.suffix}"
        shutil.copy2(source, target)
        exported += 1
        folders[cluster_dir.name] = folders.get(cluster_dir.name, 0) + 1

    if not matched_target:
        raise ValueError(f"No images found for cluster '{selected}'.")

    return {
        "dest": dest_root.relative_to(ROOT).as_posix(),
        "folders": dict(sorted(folders.items())),
        "exported_count": exported,
        "skipped_count": len(skipped),
        "skipped": skipped[:20],
    }
