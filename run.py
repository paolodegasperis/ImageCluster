from __future__ import annotations

import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


from backend.dependency_check import check_dependencies, format_report
from backend.runtime_dirs import ensure_runtime_dirs

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def open_browser_later(url: str) -> None:
    def _open() -> None:
        time.sleep(1.0)
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()


def app_is_already_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=1.5) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
            return bool(payload.get("features", {}).get("model_registry") or payload.get("version"))
    except (OSError, ValueError, urllib.error.URLError):
        return False


def port_is_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) != 0


def find_available_port(host: str, preferred_port: int, attempts: int = 20) -> int:
    if port_is_available(host, preferred_port):
        return preferred_port
    for port in range(preferred_port + 1, preferred_port + attempts + 1):
        if port_is_available(host, port):
            return port
    raise RuntimeError(f"No free local port found near {preferred_port}. Close another running instance and try again.")


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
    if app_is_already_running(url):
        print(f"ClusterIMG-V-52 is already running at {url}")
        webbrowser.open(url)
        return 0
    if not port_is_available(host, port):
        original_port = port
        port = find_available_port(host, port)
        url = f"http://{host}:{port}"
        print(f"Port {original_port} is already in use by another process. Starting on {port} instead.")
    print(f"Starting ImagePlot-CLIP at {url}")
    open_browser_later(url)
    import uvicorn
    uvicorn.run("backend.main:app", host=host, port=port, reload=False, factory=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
