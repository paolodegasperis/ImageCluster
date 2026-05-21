from __future__ import annotations

from pathlib import Path

VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"
APP_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip() if VERSION_FILE.exists() else "0.5.3"
