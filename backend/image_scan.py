from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def list_image_folders(root: Path) -> list[str]:
    root = root.resolve()
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
    folders = ["."]
    for p in sorted(root.rglob("*")):
        if p.is_dir():
            rel = p.relative_to(root).as_posix()
            if rel and not rel.startswith("."):
                folders.append(rel)
    return folders


def scan_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        return []
    return sorted([p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])
