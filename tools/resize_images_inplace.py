#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import tempfile
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg",
    ".png",
    ".webp",
    ".tif", ".tiff",
}

JPEG_EXTENSIONS = {".jpg", ".jpeg"}
WEBP_EXTENSIONS = {".webp"}
PNG_EXTENSIONS = {".png"}
TIFF_EXTENSIONS = {".tif", ".tiff"}


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )


def collect_images(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def compute_new_size(
    width: int,
    height: int,
    max_side: int,
    allow_upscale: bool,
) -> Tuple[int, int]:
    longest_side = max(width, height)

    if longest_side <= 0:
        raise ValueError("Dimensioni immagine non valide.")

    if longest_side <= max_side and not allow_upscale:
        return width, height

    scale = max_side / float(longest_side)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    return new_width, new_height


def format_from_extension(path: Path) -> str:
    ext = path.suffix.lower()

    if ext in JPEG_EXTENSIONS:
        return "JPEG"
    if ext in WEBP_EXTENSIONS:
        return "WEBP"
    if ext in PNG_EXTENSIONS:
        return "PNG"
    if ext in TIFF_EXTENSIONS:
        return "TIFF"

    raise ValueError(f"Formato non supportato: {path.suffix}")


def flatten_for_jpeg(img: Image.Image) -> Image.Image:
    """
    Converte immagini con trasparenza in RGB su fondo bianco,
    necessario per il salvataggio JPEG.
    """
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.getchannel("A")
        background.paste(img.convert("RGB"), mask=alpha)
        return background

    if img.mode == "P":
        return img.convert("RGBA").convert("RGB")

    if img.mode not in ("RGB", "L", "CMYK"):
        return img.convert("RGB")

    return img


def build_save_kwargs(
    img: Image.Image,
    path: Path,
    quality: int,
    icc_profile: bytes | None,
) -> dict:
    ext = path.suffix.lower()
    kwargs: dict = {}

    if icc_profile:
        kwargs["icc_profile"] = icc_profile

    if ext in JPEG_EXTENSIONS:
        kwargs.update(
            {
                "quality": quality,
                "optimize": True,
                "progressive": True,
            }
        )

    elif ext in WEBP_EXTENSIONS:
        kwargs.update(
            {
                "quality": quality,
                "method": 6,
            }
        )

    elif ext in PNG_EXTENSIONS:
        kwargs.update(
            {
                "optimize": True,
                "compress_level": 9,
            }
        )

    elif ext in TIFF_EXTENSIONS:
        kwargs.update(
            {
                "compression": "tiff_lzw",
            }
        )

    return kwargs


def process_image(
    path: Path,
    max_side: int,
    quality: int,
    allow_upscale: bool,
    allow_larger_output: bool,
    dry_run: bool,
) -> tuple[str, int, int]:
    """
    Restituisce:
    - status: processed / skipped / error
    - original_size_bytes
    - final_size_bytes
    """
    original_size_bytes = path.stat().st_size
    tmp_name = None

    try:
        with Image.open(path) as original_img:
            # Evita di alterare file animati o multipagina.
            if getattr(original_img, "is_animated", False):
                logging.warning("SKIP animated | %s", path)
                return "skipped", original_size_bytes, original_size_bytes

            original_format = original_img.format
            icc_profile = original_img.info.get("icc_profile")

            # Applica l'orientamento EXIF ai pixel.
            img = ImageOps.exif_transpose(original_img)
            img.load()

            old_width, old_height = img.size
            new_width, new_height = compute_new_size(
                old_width,
                old_height,
                max_side=max_side,
                allow_upscale=allow_upscale,
            )

            resized = (new_width, new_height) != (old_width, old_height)

            if resized:
                img = img.resize(
                    (new_width, new_height),
                    resample=Image.Resampling.LANCZOS,
                    reducing_gap=3.0,
                )

            output_format = format_from_extension(path)

            if output_format == "JPEG":
                img = flatten_for_jpeg(img)

            save_kwargs = build_save_kwargs(
                img=img,
                path=path,
                quality=quality,
                icc_profile=icc_profile,
            )

            if dry_run:
                logging.info(
                    "DRY-RUN | %s | %sx%s -> %sx%s | original=%s bytes",
                    path,
                    old_width,
                    old_height,
                    new_width,
                    new_height,
                    original_size_bytes,
                )
                return "processed", original_size_bytes, original_size_bytes

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=path.parent,
                prefix=f".{path.stem}_",
                suffix=path.suffix,
            ) as tmp_file:
                tmp_name = tmp_file.name

            img.save(tmp_name, format=output_format, **save_kwargs)
            new_size_bytes = Path(tmp_name).stat().st_size

            # Se l'immagine non è stata ridimensionata e la ricompressione produce
            # un file più grande, mantiene l'originale salvo diversa opzione.
            if (
                new_size_bytes > original_size_bytes
                and not resized
                and not allow_larger_output
            ):
                os.remove(tmp_name)
                logging.info(
                    "SKIP larger output | %s | %sx%s unchanged | original=%s bytes | candidate=%s bytes",
                    path,
                    old_width,
                    old_height,
                    original_size_bytes,
                    new_size_bytes,
                )
                return "skipped", original_size_bytes, original_size_bytes

            os.replace(tmp_name, path)

            logging.info(
                "OK | %s | format=%s/%s | %sx%s -> %sx%s | %s -> %s bytes",
                path,
                original_format,
                output_format,
                old_width,
                old_height,
                new_width,
                new_height,
                original_size_bytes,
                new_size_bytes,
            )

            return "processed", original_size_bytes, new_size_bytes

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        if tmp_name and Path(tmp_name).exists():
            Path(tmp_name).unlink(missing_ok=True)

        logging.exception("ERROR | %s | %s", path, exc)
        return "error", original_size_bytes, original_size_bytes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Ridimensiona e comprime ricorsivamente immagini in-place "
            "per embeddings, UMAP e workflow di analisi visuale."
        )
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Cartella principale contenente immagini e sottocartelle.",
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=800,
        help="Dimensione massima del lato più lungo. Default: 800.",
    )

    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Qualità per JPEG/WebP. Default: 90.",
    )

    parser.add_argument(
        "--log",
        type=Path,
        default=Path("resize_images.log"),
        help="Percorso del file di log. Default: resize_images.log.",
    )

    parser.add_argument(
        "--allow-upscale",
        action="store_true",
        help="Ridimensiona anche immagini più piccole fino al lato lungo indicato.",
    )

    parser.add_argument(
        "--allow-larger-output",
        action="store_true",
        help="Sostituisce il file anche se la ricompressione genera un file più grande.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simula l'operazione senza modificare i file.",
    )

    args = parser.parse_args()

    if not args.root.exists() or not args.root.is_dir():
        raise SystemExit(f"Cartella non valida: {args.root}")

    if not 1 <= args.quality <= 95:
        raise SystemExit("Per JPEG è consigliabile usare una qualità compresa tra 1 e 95.")

    setup_logging(args.log)

    images = collect_images(args.root)

    logging.info("START | root=%s | files=%s", args.root, len(images))

    processed = 0
    skipped = 0
    errors = 0
    total_before = 0
    total_after = 0

    for path in tqdm(images, desc="Elaborazione immagini", unit="img"):
        status, before, after = process_image(
            path=path,
            max_side=args.max_side,
            quality=args.quality,
            allow_upscale=args.allow_upscale,
            allow_larger_output=args.allow_larger_output,
            dry_run=args.dry_run,
        )

        total_before += before
        total_after += after

        if status == "processed":
            processed += 1
        elif status == "skipped":
            skipped += 1
        else:
            errors += 1

    saved_bytes = total_before - total_after
    saved_mb = saved_bytes / (1024 * 1024)

    logging.info(
        "END | processed=%s | skipped=%s | errors=%s | before=%s bytes | after=%s bytes | saved=%.2f MB",
        processed,
        skipped,
        errors,
        total_before,
        total_after,
        saved_mb,
    )

    print("\nCompletato.")
    print(f"Immagini trovate: {len(images)}")
    print(f"Processate: {processed}")
    print(f"Saltate: {skipped}")
    print(f"Errori: {errors}")
    print(f"Risparmio stimato: {saved_mb:.2f} MB")
    print(f"Log: {args.log.resolve()}")


if __name__ == "__main__":
    main()