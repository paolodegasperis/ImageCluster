from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass


@dataclass
class DependencyReport:
    name: str
    import_name: str
    installed: bool
    required: bool = True
    note: str = ""


DEPENDENCIES = [
    DependencyReport("fastapi", "fastapi", False),
    DependencyReport("uvicorn", "uvicorn", False),
    DependencyReport("pillow", "PIL", False),
    DependencyReport("numpy", "numpy", False),
    DependencyReport("pandas", "pandas", False),
    DependencyReport("scikit-learn", "sklearn", False),
    DependencyReport("umap-learn", "umap", False),
    DependencyReport("torch", "torch", False),
    DependencyReport("torchvision", "torchvision", False),
    DependencyReport("open_clip_torch", "open_clip", False),
    DependencyReport("transformers", "transformers", False),
    DependencyReport("huggingface_hub", "huggingface_hub", False),
    DependencyReport("einops", "einops", False, required=False, note="Needed by some optional Hugging Face models, including Nomic Embed Vision."),
]

OPTIONAL_DEPENDENCIES = [
    DependencyReport("imagebind", "imagebind", False, required=False, note="Optional. Required only for Meta ImageBind. Use the optional ImageBind installer script."),
]


def check_dependencies() -> tuple[bool, list[DependencyReport]]:
    result: list[DependencyReport] = []
    ok = True
    for dep in DEPENDENCIES:
        installed = importlib.util.find_spec(dep.import_name) is not None
        item = DependencyReport(dep.name, dep.import_name, installed, dep.required, dep.note)
        result.append(item)
        if dep.required and not installed:
            ok = False
    return ok, result


def optional_dependency_report() -> list[dict]:
    items = []
    for dep in OPTIONAL_DEPENDENCIES:
        items.append({
            "name": dep.name,
            "import_name": dep.import_name,
            "installed": importlib.util.find_spec(dep.import_name) is not None,
            "required": dep.required,
            "note": dep.note,
        })
    return items


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows":
        return "windows"
    if system == "darwin" and ("arm" in machine or "aarch64" in machine):
        return "macos_apple_silicon"
    if system == "darwin":
        return "macos_intel"
    return "other"


def pytorch_install_advice() -> dict:
    key = platform_key()
    common = {
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "official_selector": "https://pytorch.org/get-started/locally/",
        "note": "The one-click launchers install libraries automatically inside .venv. Use the manual commands only for troubleshooting.",
    }
    if key == "windows":
        return {
            **common,
            "platform": "Windows 11 / Windows",
            "recommended": "Run Start Windows 11 CPU.bat first. Use Start Windows 11 CUDA.bat only with a compatible NVIDIA GPU and driver.",
            "cpu_launcher": "Start Windows 11 CPU.bat",
            "cuda_launcher": "Start Windows 11 CUDA.bat",
            "cpu_command": r".venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
            "cuda_command": r".venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
            "after_pytorch": r".venv\Scripts\python.exe -m pip install -r requirements-core.txt",
            "imagebind_note": "ImageBind is optional. Use Install Optional ImageBind Windows.bat after the app has created .venv.",
        }
    if key == "macos_apple_silicon":
        return {
            **common,
            "platform": "macOS Apple Silicon",
            "recommended": "Run Start macOS Apple Silicon.command. The standard macOS PyTorch wheel can use MPS acceleration when available.",
            "cpu_launcher": "Start macOS Apple Silicon.command",
            "cuda_launcher": "CUDA is not used on Apple Silicon.",
            "cpu_command": ".venv/bin/python -m pip install torch torchvision",
            "cuda_command": "CUDA is not used on macOS.",
            "after_pytorch": ".venv/bin/python -m pip install -r requirements-core.txt",
            "imagebind_note": "ImageBind is optional. Use Install Optional ImageBind macOS.command after the app has created .venv.",
        }
    if key == "macos_intel":
        return {
            **common,
            "platform": "macOS Intel",
            "recommended": "Run Start macOS.command. CUDA is not used on macOS.",
            "cpu_launcher": "Start macOS.command",
            "cuda_launcher": "CUDA is not used on macOS.",
            "cpu_command": ".venv/bin/python -m pip install torch torchvision",
            "cuda_command": "CUDA is not used on macOS.",
            "after_pytorch": ".venv/bin/python -m pip install -r requirements-core.txt",
            "imagebind_note": "ImageBind is optional. Use Install Optional ImageBind macOS.command after the app has created .venv.",
        }
    return {
        **common,
        "platform": platform.system() or "Unknown",
        "recommended": "Use launcher.py --torch cpu, or consult the official PyTorch selector.",
        "cpu_launcher": "python launcher.py --torch cpu",
        "cuda_launcher": "python launcher.py --torch cuda",
        "cpu_command": "python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu",
        "cuda_command": "python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        "after_pytorch": "python -m pip install -r requirements-core.txt",
        "imagebind_note": "ImageBind is optional and may require additional setup.",
    }


def format_install_advice() -> str:
    advice = pytorch_install_advice()
    lines = [
        "",
        "Installation guidance:",
        f"Detected platform: {advice['platform']}",
        advice["recommended"],
        "",
        "Recommended launcher:",
        advice["cpu_launcher"],
        "",
        "CUDA launcher, when applicable:",
        advice["cuda_launcher"],
        "",
        "Manual PyTorch CPU command:",
        advice["cpu_command"],
        "",
        "Manual PyTorch CUDA command:",
        advice["cuda_command"],
        "",
        "Then install the remaining ImagePlot-CLIP dependencies:",
        advice["after_pytorch"],
        "",
        advice.get("imagebind_note", ""),
        f"Official PyTorch selector: {advice['official_selector']}",
    ]
    return "\n".join(lines)


def format_report(report: list[DependencyReport]) -> str:
    missing = [d for d in report if d.required and not d.installed]
    lines = ["Dependency check:"]
    for dep in report:
        mark = "OK" if dep.installed else "MISSING"
        required = "required" if dep.required else "optional"
        note = f" ({dep.note})" if dep.note else ""
        lines.append(f"- {dep.name}: {mark} [{required}]{note}")
    for dep in optional_dependency_report():
        mark = "OK" if dep["installed"] else "MISSING"
        lines.append(f"- {dep['name']}: {mark} [optional] ({dep['note']})")
    if missing:
        lines.extend([
            "",
            "Missing dependencies detected.",
            "For non-technical use, start the app with the appropriate Start script rather than installing packages manually.",
            format_install_advice(),
        ])
    else:
        lines.append("All required dependencies are available.")
    return "\n".join(lines)


if __name__ == "__main__":
    ok, report = check_dependencies()
    print(format_report(report))
    raise SystemExit(0 if ok else 1)
