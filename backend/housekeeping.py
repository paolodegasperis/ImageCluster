"""Bounded growth for the on-disk artefacts the app produces.

Thumbnails, job logs and debug reports all used to accumulate forever: a long
session on a few large folders could leave hundreds of megabytes of thumbnails
and thousands of log files behind, on a machine whose owner never opens
output/. Each helper enforces a ceiling from config.json's ``limits`` section.

Eviction is by modification time, so it is oldest-written first rather than
least-recently-used: reading a cached thumbnail deliberately does not touch its
mtime, because doing so would add a write to every thumbnail request to manage a
cache whose contents are regenerated on demand anyway.

Every operation is best-effort: housekeeping must never turn a working request
into an error, so filesystem races (a file removed by the user mid-sweep) are
swallowed.
"""

from __future__ import annotations

import threading
from pathlib import Path

from .app_config import get as config_get
from .runtime_dirs import LOGS_DIR, THUMB_CACHE_DIR

# Sweeping on every request would stat the whole cache directory for each
# thumbnail. Sweeping is therefore throttled to one pass per N writes.
_THUMB_SWEEP_EVERY = 200
_thumb_writes_since_sweep = 0
_thumb_lock = threading.Lock()


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _entries(directory: Path, pattern: str = "*") -> list[tuple[float, int, Path]]:
    if not directory.is_dir():
        return []
    items: list[tuple[float, int, Path]] = []
    for path in directory.glob(pattern):
        try:
            if not path.is_file():
                continue
            stat = path.stat()
        except OSError:
            continue
        items.append((stat.st_mtime, stat.st_size, path))
    return items


def prune_log_dir() -> int:
    """Keep only the newest ``limits.max_log_files`` files in output/logs."""
    max_files = int(config_get("limits.max_log_files", 200) or 0)
    if max_files <= 0:
        return 0
    items = _entries(LOGS_DIR)
    if len(items) <= max_files:
        return 0
    items.sort(key=lambda item: item[0], reverse=True)
    removed = 0
    for _, _, path in items[max_files:]:
        _safe_unlink(path)
        removed += 1
    return removed


def prune_thumb_cache(force: bool = False) -> int:
    """Trim the thumbnail cache to ``limits.thumb_cache_max_mb``, oldest first.

    Called after each cache write, but only sweeps every ``_THUMB_SWEEP_EVERY``
    calls: statting the whole directory on every thumbnail request would cost
    more than the cache saves. ``force`` bypasses the throttle for the startup
    sweep and for tests.
    """
    global _thumb_writes_since_sweep

    if not force:
        with _thumb_lock:
            _thumb_writes_since_sweep += 1
            if _thumb_writes_since_sweep < _THUMB_SWEEP_EVERY:
                return 0
            _thumb_writes_since_sweep = 0

    max_mb = float(config_get("limits.thumb_cache_max_mb", 512) or 0)
    if max_mb <= 0:
        return 0
    budget = int(max_mb * 1024 * 1024)

    items = _entries(THUMB_CACHE_DIR, "*.jpg")
    total = sum(size for _, size, _ in items)
    if total <= budget:
        return 0

    items.sort(key=lambda item: item[0])  # oldest first
    removed = 0
    for _, size, path in items:
        if total <= budget:
            break
        _safe_unlink(path)
        total -= size
        removed += 1
    return removed


def startup_sweep() -> dict[str, int]:
    """Run every sweep once at startup so a stale cache never grows unchecked."""
    return {
        "logs_removed": prune_log_dir(),
        "thumbnails_removed": prune_thumb_cache(force=True),
    }
