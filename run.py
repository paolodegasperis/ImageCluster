from __future__ import annotations

import json
import sys
import threading
import time
import webbrowser
from pathlib import Path

from backend.dependency_check import check_dependencies, format_report
from backend.runtime_dirs import ensure_runtime_dirs

from project_paths import get_project_root

ROOT = get_project_root()
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main() -> int:
    ensure_runtime_dirs()
    config = load_config()
    ok, report = check_dependencies()
    if not ok and config.get("block_start_if_dependencies_missing", True):
        print("ImagePlot-CLIP cannot start.\n")
        print(format_report(report))
        return 1

    host = config.get("host", "127.0.0.1")
    port = int(config.get("port", 8765))
    url = f"http://{host}:{port}"
    print(format_report(report))
    print(f"Starting ImageCluster at {url}")
    open_browser_later(url)
    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, factory=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
