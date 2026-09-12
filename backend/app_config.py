"""Typed access to config.json.

Until v6.0 config.json was mostly decorative: only host, port and
block_start_if_dependencies_missing were ever read, while the clustering,
semantic-search and debug sections were silently ignored. This module loads the
file once, merges it over a complete set of defaults and exposes the values that
the rest of the application actually consumes.

Unknown keys are preserved so a user-edited file is never rewritten lossily, and
a malformed file degrades to the defaults instead of preventing startup.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from project_paths import get_project_root

CONFIG_PATH = get_project_root() / "config.json"

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8765,
    "default_image_dir": "img",
    "output_dir": "output",
    "default_model_key": "openclip_vit_b_32",
    "default_reducer": "umap",
    "batch_size": 32,
    "block_start_if_dependencies_missing": True,
    "optional_clustering": {
        "enabled_by_default": False,
        "method": "kmeans",
        "auto_min_k": 2,
        "auto_max_k": 8,
        "fixed_k": 5,
    },
    "semantic_search": {
        "enabled": True,
        "default_top_k": 30,
        "default_threshold": None,
        "save_search_results": False,
        "highlight_results_on_plot": True,
    },
    "debug": {
        "job_logs": True,
        "json_debug_reports": True,
        "batch_retry": True,
    },
    # Bounded on-disk / in-memory growth. Before v6.0 none of these had a ceiling.
    "limits": {
        "max_jobs_in_memory": 50,
        "thumb_cache_max_mb": 512,
        "max_log_files": 200,
        "max_concurrent_jobs": 1,
    },
    "views": {
        "grid_mosaic_enabled": True,
        "neighbor_graph_enabled": True,
        "neighbor_graph_default_k": 4,
        "thumbnail_opacity": 1.0,
    },
}


def _compatible(default: Any, value: Any) -> bool:
    """Whether ``value`` may replace ``default``, judged by shape.

    config.json is hand-edited, and a wrong type used to propagate straight into
    the running configuration: a section replaced by a string made
    ``dict(CONFIG["limits"])`` raise, so /api/config answered 500 and the
    dashboard broke. A value that does not match the default's shape is
    discarded, which is the whole point of having defaults.
    """
    if default is None:
        # Genuinely nullable settings (a threshold that may be unset).
        return True
    if isinstance(default, dict):
        return isinstance(value, dict)
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, (int, float)):
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if isinstance(default, str):
        return isinstance(value, str)
    if isinstance(default, list):
        return isinstance(value, list)
    return True


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Overlay ``override`` on ``base``, keeping any value of the wrong shape out."""
    merged = deepcopy(base)
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if key not in merged:
            # Unknown keys are preserved so a user-edited file is not silently
            # trimmed, but nothing reads them.
            merged[key] = value
            continue
        default = merged[key]
        if isinstance(default, dict):
            merged[key] = _deep_merge(default, value) if isinstance(value, dict) else default
        elif _compatible(default, value):
            merged[key] = value
    return merged


def _load() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return deepcopy(DEFAULTS)
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return deepcopy(DEFAULTS)
    if not isinstance(raw, dict):
        return deepcopy(DEFAULTS)
    return _deep_merge(DEFAULTS, raw)


CONFIG: dict[str, Any] = _load()


def reload_config() -> dict[str, Any]:
    """Re-read config.json. Used by tests and by the launcher before startup."""
    global CONFIG
    CONFIG = _load()
    return CONFIG


def get(path: str, fallback: Any = None) -> Any:
    """Read a dotted path, e.g. get("semantic_search.default_top_k")."""
    node: Any = CONFIG
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return fallback
        node = node[part]
    return node


def _section(name: str) -> dict[str, Any]:
    """A config section as a plain dict, falling back to the shipped defaults."""
    value = CONFIG.get(name)
    if isinstance(value, dict):
        return dict(value)
    return deepcopy(DEFAULTS[name])


def clustering_defaults() -> dict[str, Any]:
    return _section("optional_clustering")


def search_defaults() -> dict[str, Any]:
    return _section("semantic_search")


def debug_flags() -> dict[str, Any]:
    return _section("debug")


def limits() -> dict[str, Any]:
    return _section("limits")


def view_defaults() -> dict[str, Any]:
    return _section("views")
