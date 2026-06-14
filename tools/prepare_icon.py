from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CANDIDATES = [
    ROOT / "icone" / "ImageClusetIcon.png",
]
OUT_DIR = ROOT / "icone"


def main() -> None:
    source = next((path for path in SOURCE_CANDIDATES if path.exists()), None)
    if source is None:
        raise FileNotFoundError("Source icon not found in icone/.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "ImageClusetIcon.png"
    ico_path = OUT_DIR / "ImageClusetIcon.ico"
    icns_path = OUT_DIR / "ImageClusetIcon.icns"

    shutil.copy2(source, png_path)
    img = Image.open(source).convert("RGBA")
    img.save(ico_path, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    img.save(icns_path)

    print(png_path)
    print(ico_path)
    print(icns_path)


if __name__ == "__main__":
    main()
