from __future__ import annotations

import sys
import threading
import time
import webbrowser

from backend.app_config import get as config_get
from backend.dependency_check import check_dependencies, format_report
from backend.runtime_dirs import ensure_runtime_dirs
from backend.version import __app_name__, __version__


def open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def main() -> int:
    ensure_runtime_dirs()
    ok, report = check_dependencies()
    if not ok and config_get("block_start_if_dependencies_missing", True):
        print(f"{__app_name__} cannot start.")
        print("")
        print(format_report(report))
        return 1

    host = config_get("host", "127.0.0.1")
    port = int(config_get("port", 8765))
    url = f"http://{host}:{port}"
    print(format_report(report))
    print(f"Starting {__app_name__} v{__version__} at {url}")
    open_browser_later(url)
    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, factory=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
