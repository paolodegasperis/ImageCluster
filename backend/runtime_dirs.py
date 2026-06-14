from __future__ import annotations

from pathlib import Path

from project_paths import get_project_root

PROJECT_ROOT = get_project_root()
ROOT = PROJECT_ROOT
APP_DIR = PROJECT_ROOT / "app"
IMG_DIR = PROJECT_ROOT / "img"
OUTPUT_DIR = PROJECT_ROOT / "output"
EMBEDDINGS_DIR = OUTPUT_DIR / "embeddings"
PROJECTIONS_DIR = OUTPUT_DIR / "projections"
LOGS_DIR = OUTPUT_DIR / "logs"
SEARCH_DIR = OUTPUT_DIR / "search"


def ensure_runtime_dirs() -> None:
    """Create folders required at runtime.

    Zip archives often omit empty folders. The application must therefore create
    these directories before Starlette mounts static directories or jobs write
    embeddings, projections and logs.
    """
    for path in (IMG_DIR, OUTPUT_DIR, EMBEDDINGS_DIR, PROJECTIONS_DIR, LOGS_DIR, SEARCH_DIR):
        path.mkdir(parents=True, exist_ok=True)
