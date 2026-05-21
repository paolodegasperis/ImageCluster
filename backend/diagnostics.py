from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

IMPORTANT_IMPORTS = [
    "fastapi",
    "uvicorn",
    "PIL",
    "numpy",
    "pandas",
    "sklearn",
    "umap",
    "torch",
    "torchvision",
    "open_clip",
    "transformers",
    "huggingface_hub",
    "einops",
    "imagebind",
    "timm",
    "safetensors",
]


def torch_report() -> dict[str, Any]:
    data: dict[str, Any] = {"available": False}
    try:
        import torch

        data["available"] = True
        data["version"] = getattr(torch, "__version__", "unknown")
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            try:
                data["cuda_device_count"] = torch.cuda.device_count()
                data["cuda_device_name"] = torch.cuda.get_device_name(0)
            except Exception as exc:
                data["cuda_device_error"] = str(exc)
        data["mps_available"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    except Exception as exc:
        data["error"] = str(exc)
    return data


def import_report() -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in IMPORTANT_IMPORTS}


def package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except Exception:
        return None


def collect_system_report(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "project_root": str(ROOT),
        "cwd": str(Path.cwd()),
        "output_dir": str(OUTPUT),
        "output_dir_exists": OUTPUT.exists(),
        "output_dir_writable": _is_writable(OUTPUT),
        "img_dir": str(ROOT / "img"),
        "img_dir_exists": (ROOT / "img").exists(),
        "img_dir_readable": (ROOT / "img").is_dir(),
        "imports": import_report(),
        "package_versions": {
            "fastapi": package_version("fastapi"),
            "uvicorn": package_version("uvicorn"),
            "pillow": package_version("pillow"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "umap-learn": package_version("umap-learn"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "open_clip_torch": package_version("open_clip_torch"),
            "transformers": package_version("transformers"),
            "huggingface_hub": package_version("huggingface_hub"),
            "timm": package_version("timm"),
            "safetensors": package_version("safetensors"),
        },
        "torch": torch_report(),
    }
    if extra:
        report["context"] = extra
    return report


def _is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def write_debug_report(job_id: str, extra: dict[str, Any] | None = None) -> Path:
    log_dir = OUTPUT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{job_id}.debug.json"
    path.write_text(json.dumps(collect_system_report(extra), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def user_facing_error(exc: Exception) -> str:
    text = str(exc)
    lower = text.lower()
    if "basemodeloutputwithpooling" in lower and "norm" in lower:
        return (
            "The selected Transformers model returned a structured output instead of a tensor. "
            "This is fixed in v5.2.2. As a temporary workaround, select OpenCLIP or CLIP, "
            "or disable cache and rerun after updating."
        )
    if "out of memory" in lower or "cuda" in lower and "memory" in lower:
        return (
            "The embedding step ran out of memory. Reduce Batch size to 1, use the CPU launcher, "
            "or choose a lighter model such as OpenCLIP ViT-B-32 or MobileCLIP."
        )
    if "no images found" in lower:
        return "No images were found. Add .jpg, .png, .webp, .tif or .tiff files inside img/ or one of its subfolders."
    if "trust_remote_code" in lower:
        return "This model requires Hugging Face remote code. Check the internet connection and rerun the job."
    return text
