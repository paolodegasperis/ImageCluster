from __future__ import annotations

import argparse
import hashlib
import os
import platform
import subprocess
import sys
import webbrowser
import shutil
from pathlib import Path

from project_paths import get_project_root

ROOT = get_project_root()
VENV = ROOT / ".venv"
REQ_CORE = ROOT / "requirements-core.txt"
REQ_FULL = ROOT / "requirements.txt"
STAMP = VENV / ".imageplot_launcher.sha256"
RUNTIME_DIRS = [
    ROOT / "img",
    ROOT / "output",
    ROOT / "output" / "embeddings",
    ROOT / "output" / "projections",
    ROOT / "output" / "logs",
]

PYTORCH_COMMANDS = {
    "windows_cpu": [
        "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ],
    "windows_cuda": [
        "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cu128",
    ],
    "macos": [
        "pip", "install", "torch", "torchvision",
    ],
    "linux_cpu": [
        "pip", "install", "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
    ],
}

MIN_PYTHON = (3, 10)
PYTHON_DOWNLOAD_URL = "https://www.python.org/downloads/"
PYTORCH_SELECTOR_URL = "https://pytorch.org/get-started/locally/"



def ensure_runtime_dirs() -> None:
    for path in RUNTIME_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def print_step(message: str) -> None:
    print("")
    print("=" * 72)
    print(message)
    print("=" * 72, flush=True)


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


def venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        raise RuntimeError(
            "Python 3.10 or newer is required. "
            f"Detected Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}."
        )


def open_url(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def show_fatal_dialog(title: str, message: str, *, help_url: str | None = None) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            if help_url:
                wants_help = messagebox.askyesno(title, message + "\n\nAprire il link di aiuto?", parent=root)
                if wants_help:
                    open_url(help_url)
            else:
                messagebox.showerror(title, message, parent=root)
        finally:
            root.destroy()
    except Exception:
        print(f"{title}: {message}", file=sys.stderr)
        if help_url:
            print(help_url, file=sys.stderr)


def file_hash(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def launcher_hash(torch_variant: str) -> str:
    parts = [
        f"torch_variant={torch_variant}",
        f"requirements_core={file_hash(REQ_CORE)}",
        f"requirements_full={file_hash(REQ_FULL)}",
        "launcher_version=6.0",
    ]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def ensure_venv() -> Path:
    if not VENV.exists():
        print_step("Creating the local Python environment")
        bootstrap = bootstrap_python_command()
        run(bootstrap + ["-m", "venv", str(VENV)])
    py = venv_python()
    if not py.exists():
        raise RuntimeError(f"Virtual environment Python not found: {py}")
    return py


def bootstrap_python_command() -> list[str]:
    if not getattr(sys, "frozen", False):
        return [sys.executable]

    candidates: list[list[str]] = []
    if platform.system() == "Windows":
        candidates = [["py", "-3"], ["python"], ["python3"]]
    else:
        candidates = [["python3"], ["python"]]

    for candidate in candidates:
        executable = candidate[0]
        if shutil.which(executable):
            return candidate

    raise RuntimeError(
        "A system Python interpreter could not be found. "
        "Install Python 3.10 or newer and run the launcher again."
    )


def python_exec(py: Path, pip_args: list[str]) -> list[str]:
    return [str(py), "-m"] + pip_args


def detect_variant(requested: str) -> str:
    if requested != "auto":
        return requested
    system = platform.system()
    if system == "Windows":
        return "cpu"
    if system == "Darwin":
        return "macos"
    return "cpu"


def torch_command_for_platform(variant: str) -> list[str]:
    system = platform.system()
    if system == "Windows":
        if variant == "cuda":
            return PYTORCH_COMMANDS["windows_cuda"]
        return PYTORCH_COMMANDS["windows_cpu"]
    if system == "Darwin":
        return PYTORCH_COMMANDS["macos"]
    if variant == "cuda":
        # Linux CUDA users should normally use the official selector because CUDA
        # compatibility varies. cu128 is used as the current stable default.
        return [
            "pip", "install", "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        ]
    return PYTORCH_COMMANDS["linux_cpu"]


def module_available(py: Path, module_name: str) -> bool:
    code = f"import importlib.util; raise SystemExit(0 if importlib.util.find_spec('{module_name}') else 1)"
    result = subprocess.run([str(py), "-c", code], cwd=ROOT)
    return result.returncode == 0


def torch_available(py: Path) -> bool:
    code = "import torch, torchvision; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"
    result = subprocess.run([str(py), "-c", code], cwd=ROOT)
    return result.returncode == 0


def install_pytorch(py: Path, variant: str, force: bool = False) -> None:
    if torch_available(py) and not force:
        print("PyTorch is already installed in .venv.", flush=True)
        return
    print_step(f"Installing PyTorch for variant: {variant}")
    pip_cmd = torch_command_for_platform(variant)
    run(python_exec(py, pip_cmd))
    if not torch_available(py):
        raise RuntimeError("PyTorch installation did not pass the verification step.")


def install_core_requirements(py: Path) -> None:
    if not REQ_CORE.exists():
        raise RuntimeError("requirements-core.txt not found. The project folder may be incomplete.")
    print_step("Installing ImagePlot-CLIP libraries")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", "-r", str(REQ_CORE)])


def requirements_current(torch_variant: str) -> bool:
    if not STAMP.exists():
        return False
    return STAMP.read_text(encoding="utf-8").strip() == launcher_hash(torch_variant)


def write_stamp(torch_variant: str) -> None:
    STAMP.write_text(launcher_hash(torch_variant), encoding="utf-8")


def print_platform_advice(torch_variant: str) -> None:
    system = platform.system()
    machine = platform.machine()
    print("")
    print("Detected platform:", system, machine)
    print("Selected PyTorch variant:", torch_variant)
    print("")
    if system == "Windows":
        print("Windows 11:")
        print("- Use the CPU launcher if you are unsure.")
        print("- Use the CUDA launcher only with a compatible NVIDIA GPU and driver.")
    elif system == "Darwin":
        print("macOS:")
        print("- On Apple Silicon, the standard PyTorch wheel can use MPS acceleration when available.")
        print("- CUDA is not used on macOS.")
    else:
        print("For Linux or other systems, check the official PyTorch selector if installation fails:")
        print(PYTORCH_SELECTOR_URL)
    print("")


def verify_required_modules(py: Path) -> None:
    required = [
        "fastapi", "uvicorn", "PIL", "numpy", "pandas", "sklearn", "umap",
        "torch", "torchvision", "open_clip", "transformers", "huggingface_hub",
    ]
    missing = [name for name in required if not module_available(py, name)]
    if missing:
        raise RuntimeError("Required modules are still missing after installation: " + ", ".join(missing))


def launch_app(py: Path) -> None:
    print_step("Starting ImageCluster")
    run([str(py), "run.py"])


def dependency_install_failed(message: str) -> None:
    show_fatal_dialog("ImagePlot-CLIP - dependency problem", message)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="ImagePlot-CLIP one-click launcher")
    parser.add_argument(
        "--torch",
        choices=["auto", "cpu", "cuda", "macos"],
        default=os.environ.get("IMAGEPLOT_TORCH_VARIANT", "auto"),
        help="PyTorch installation variant.",
    )
    parser.add_argument(
        "--reinstall",
        action="store_true",
        help="Reinstall PyTorch and Python libraries even if the launcher stamp is current.",
    )
    parser.add_argument(
        "--no-start",
        action="store_true",
        help="Install dependencies but do not start the web app.",
    )
    args = parser.parse_args(argv)

    check_python_version()
    ensure_runtime_dirs()
    torch_variant = detect_variant(args.torch)
    print_step("ImagePlot-CLIP v6.0 launcher")
    print(f"Project folder: {ROOT}")
    print_platform_advice(torch_variant)

    py = ensure_venv()

    try:
        if args.reinstall or not requirements_current(torch_variant):
            install_pytorch(py, torch_variant, force=args.reinstall)
            install_core_requirements(py)
            verify_required_modules(py)
            write_stamp(torch_variant)
        else:
            print("Local environment already prepared. Starting without reinstalling libraries.", flush=True)

        if not args.no_start:
            launch_app(py)
    except RuntimeError as exc:
        dependency_install_failed(
            "ImagePlot-CLIP could not complete the first start.\n\n"
            f"{exc}\n\n"
            "The app can usually recover by reconnecting to the internet and running the launcher again."
        )
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except subprocess.CalledProcessError as exc:
        message = (
            "A command failed while preparing or starting ImagePlot-CLIP.\n\n"
            f"Exit code: {exc.returncode}\n\n"
            "Suggested checks:\n"
            "- Confirm that Python 3.10 or newer is installed.\n"
            "- Confirm that the computer is connected to the internet on first launch.\n"
            "- On Windows, use the CPU launcher first unless NVIDIA CUDA is already configured.\n"
            f"- If PyTorch fails, use the official selector: {PYTORCH_SELECTOR_URL}"
        )
        print("", file=sys.stderr)
        print(message, file=sys.stderr)
        show_fatal_dialog(
            "ImagePlot-CLIP launch failed",
            message,
            help_url=PYTHON_DOWNLOAD_URL,
        )
        sys.exit(1)
    except Exception as exc:
        message = (
            f"Launcher error: {exc}\n\n"
            "The project files are intact, but the local environment could not be prepared.\n"
            "Use the CPU launcher as the safest first attempt."
        )
        print("", file=sys.stderr)
        print(message, file=sys.stderr)
        show_fatal_dialog(
            "ImagePlot-CLIP launch failed",
            message,
            help_url=PYTHON_DOWNLOAD_URL,
        )
        sys.exit(1)
