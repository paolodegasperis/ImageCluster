from __future__ import annotations

import hashlib
import json
import os
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
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def dataset_fingerprint(paths: list[Path]) -> dict:
    files = []
    h = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        item = {
            "relative_path": path.as_posix(),
            "size": stat.st_size,
            "mtime": int(stat.st_mtime),
        }
        files.append(item)
        h.update(item["relative_path"].encode("utf-8"))
        h.update(str(item["size"]).encode("utf-8"))
        h.update(str(item["mtime"]).encode("utf-8"))
    return {"count": len(files), "hash": h.hexdigest()[:16], "files": files}
