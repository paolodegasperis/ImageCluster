from __future__ import annotations

import json
import os

from .runtime_dirs import OUTPUT_DIR

LOCAL_SETTINGS_PATH = OUTPUT_DIR / "local_settings.json"


def load_local_settings() -> dict:
    if not LOCAL_SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(LOCAL_SETTINGS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_local_settings(settings: dict) -> None:
    LOCAL_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOCAL_SETTINGS_PATH.write_text(json.dumps(settings, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def get_hf_token() -> str:
    settings = load_local_settings()
    token = settings.get("huggingface_token", "")
    return token if isinstance(token, str) else ""


def apply_hf_token_to_environment() -> str:
    """Expose the stored Hugging Face token to libraries that read env vars."""
    token = get_hf_token().strip()
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if token:
            os.environ[name] = token
        else:
            os.environ.pop(name, None)
    return token


def huggingface_loader_kwargs() -> dict:
    token = apply_hf_token_to_environment()
    return {"token": token} if token else {}


def set_hf_token(token: str) -> dict:
    settings = load_local_settings()
    settings["huggingface_token"] = token.strip()
    save_local_settings(settings)
    return settings


def remove_hf_token() -> dict:
    settings = load_local_settings()
    settings.pop("huggingface_token", None)
    save_local_settings(settings)
    return settings
