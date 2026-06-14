from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
