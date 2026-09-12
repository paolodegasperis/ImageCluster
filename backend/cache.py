from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .runtime_dirs import ROOT


def dataset_hash(paths: list[Path], *model_parts: str) -> str:
    h = hashlib.sha256()
    for part in model_parts:
        h.update(str(part).encode("utf-8"))
    for path in paths:
        stat = path.stat()
        h.update(str(path.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode("utf-8"))
        h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def manifest_paths(manifest: dict | None) -> list[Path] | None:
    """Return the exact image list an embedding array was built from.

    Encoders skip unreadable files, so the encoded set can be a strict subset of
    what ``scan_images`` returns. Before v6.0 the manifest only stored a count,
    and both the cached-embedding path and text search re-derived the order from
    a fresh directory scan. When the counts happened to match after an add and a
    delete, rows were silently attributed to the wrong images. Storing the paths
    removes the guesswork; a manifest without them is treated as unusable.
    """
    if not manifest:
        return None
    raw = manifest.get("paths")
    if not isinstance(raw, list) or not raw:
        return None
    resolved: list[Path] = []
    for item in raw:
        if not isinstance(item, str):
            return None
        resolved.append(ROOT / item)
    return resolved
