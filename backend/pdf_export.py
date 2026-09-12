"""Contact sheets of a result, as PDF.

Every view in this app is a working instrument, and a result that cannot leave
the screen cannot be worked on. A community, a neighbourhood, a set of similar
images: each is an answer worth taking away, annotating and coming back to. This
module turns any of them into the same document, so the three surfaces produce
one recognisable artefact rather than three.

The sheet is built server-side deliberately. The originals sit on disk here at
full resolution, while the browser can only reach them through ``/api/thumb``,
which stops at 512px - no headroom at all for a page meant to be looked at and
printed. The images therefore never make the round trip.

``fpdf2`` is an optional dependency. Without it everything else works unchanged
and the callers report that PDF export is unavailable, rather than failing at
the point of use.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from PIL import Image

from .runtime_dirs import IMG_DIR

try:  # fpdf2 is optional; see check_dependencies()
    from fpdf import FPDF

    PDF_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only where fpdf2 is absent
    FPDF = None  # type: ignore[assignment]
    PDF_AVAILABLE = False

PDF_MISSING_MESSAGE = (
    "PDF export needs the 'fpdf2' package, which is not installed in this "
    "environment. Everything else works without it. Install it with: "
    "pip install fpdf2"
)

# The requirement the sheet is built around: no image is embedded smaller than
# this on its long side, so the page can actually be read.
MIN_LONG_SIDE = 500
# And an upper bound, or a forty-image community becomes a file nobody can open.
MAX_LONG_SIDE = 1400

# A4 portrait, in millimetres.
PAGE_W, PAGE_H = 210.0, 297.0
MARGIN_X, MARGIN_TOP, MARGIN_BOTTOM = 15.0, 16.0, 14.0
COLUMNS = 2
GUTTER = 8.0
CAPTION_H = 11.0


@dataclass
class SheetItem:
    """One image on the sheet, with whatever the surface knows about it."""

    relative_path: str
    caption: str = ""
    detail: str = ""


@dataclass
class SheetResult:
    """The document, plus an honest account of what did not make it in."""

    pdf: bytes
    included: int = 0
    skipped: list[dict[str, str]] = field(default_factory=list)
    undersized: list[str] = field(default_factory=list)


def _safe_source(relative_path: str) -> Any:
    """Resolve an image path, refusing anything outside ``img/``.

    The three callers all send a list of paths chosen in the browser. That list
    is user input travelling over HTTP, so it is re-validated here rather than
    trusted: without this, a crafted request would read any file on the machine
    into a PDF.
    """
    try:
        candidate = (IMG_DIR / relative_path.replace("img/", "", 1)).resolve()
    except (OSError, ValueError):
        return None
    root = IMG_DIR.resolve()
    if candidate != root and root not in candidate.parents:
        return None
    return candidate if candidate.is_file() else None


def _latin1(text: str) -> str:
    """The core PDF fonts cover latin-1 only; keep the rest legible, not fatal."""
    return str(text or "").encode("latin-1", "replace").decode("latin-1")


def _prepare(path: Any) -> tuple[Image.Image, bool]:
    """An RGB copy sized for the page, and whether it fell short of the minimum.

    Images larger than needed are reduced, which keeps the file openable. Images
    already below the minimum are left at their own size: enlarging them would
    invent detail and hide the fact that the source is small, and this document
    exists to be looked at closely.
    """
    image = Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")

    long_side = max(image.size)
    undersized = long_side < MIN_LONG_SIDE
    if long_side > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    return image, undersized


class _Sheet(FPDF if PDF_AVAILABLE else object):  # type: ignore[misc]
    """Page furniture: the running title and the page numbers."""

    def __init__(self, running_title: str) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.running_title = _latin1(running_title)
        self.set_auto_page_break(False)
        self.set_margins(MARGIN_X, MARGIN_TOP, MARGIN_X)

    def footer(self) -> None:
        self.set_y(-MARGIN_BOTTOM)
        self.set_font("Helvetica", "", 7.5)
        self.set_text_color(130)
        self.cell(0, 6, self.running_title, align="L")
        self.set_x(-MARGIN_X - 40)
        self.cell(40, 6, f"{self.page_no()} / {{nb}}", align="R")
        self.set_text_color(0)


def build_image_sheet(
    *,
    title: str,
    subtitle: str = "",
    meta_lines: list[str] | None = None,
    items: list[SheetItem],
    note: str = "",
) -> SheetResult:
    """One contact sheet, whatever the surface that asked for it.

    Keeping the layout in a single function is the point: a community, a
    neighbourhood and a similarity result are different questions, but the
    document that answers them should be the same object, so that a folder of
    these can be read without relearning the format each time.
    """
    if not PDF_AVAILABLE:
        raise RuntimeError(PDF_MISSING_MESSAGE)
    if not items:
        raise ValueError("There is nothing to export.")

    pdf = _Sheet(title)
    pdf.alias_nb_pages()
    pdf.add_page()

    cell_w = (PAGE_W - 2 * MARGIN_X - (COLUMNS - 1) * GUTTER) / COLUMNS
    image_box = cell_w  # a square box: rows stay level whatever the aspect ratios
    cell_h = image_box + CAPTION_H

    # --- heading ---------------------------------------------------------
    text_w = PAGE_W - 2 * MARGIN_X

    def paragraph(text: str, size: float, style: str = "", grey: int = 0, height: float = 5.0) -> None:
        """One block of heading text.

        Width and x are set explicitly on every call: left to itself the cursor
        carries over from the previous block and a later line can be handed a
        width of nearly zero, which fpdf2 reports only as a failure to fit a
        single character.
        """
        pdf.set_x(MARGIN_X)
        pdf.set_font("Helvetica", style, size)
        if grey:
            pdf.set_text_color(grey)
        pdf.multi_cell(text_w, height, _latin1(text))
        if grey:
            pdf.set_text_color(0)

    pdf.set_xy(MARGIN_X, MARGIN_TOP)
    paragraph(title, 15, "B", height=7)
    if subtitle:
        paragraph(subtitle, 10.5, grey=90, height=5.4)
    for line in meta_lines or []:
        paragraph(line, 8.5, grey=120, height=4.4)
    if note:
        pdf.ln(1.5)
        paragraph(note, 8.5, "I", grey=120, height=4.4)
    pdf.ln(3)

    top = pdf.get_y()
    column = 0
    row_top = top
    included = 0
    skipped: list[dict[str, str]] = []
    undersized: list[str] = []

    for item in items:
        source = _safe_source(item.relative_path)
        if source is None:
            # Reported rather than silently dropped: a sheet with fewer images
            # than the result it came from would be quietly wrong.
            skipped.append({"relative_path": item.relative_path, "reason": "not found under img/"})
            continue
        try:
            image, small = _prepare(source)
        except (OSError, ValueError) as exc:
            skipped.append({"relative_path": item.relative_path, "reason": f"unreadable ({exc})"})
            continue
        if small:
            undersized.append(item.relative_path)

        if row_top + cell_h > PAGE_H - MARGIN_BOTTOM:
            pdf.add_page()
            row_top = MARGIN_TOP
            column = 0

        x = MARGIN_X + column * (cell_w + GUTTER)

        # Fit inside the square box, preserving the aspect, centred.
        ratio = min(image_box / image.width, image_box / image.height)
        draw_w, draw_h = image.width * ratio, image.height * ratio
        offset_x = x + (image_box - draw_w) / 2
        offset_y = row_top + (image_box - draw_h) / 2

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
        buffer.seek(0)
        pdf.image(buffer, x=offset_x, y=offset_y, w=draw_w, h=draw_h)

        pdf.set_draw_color(210)
        pdf.rect(offset_x, offset_y, draw_w, draw_h)
        pdf.set_draw_color(0)

        caption = item.caption or item.relative_path.rsplit("/", 1)[-1]
        pdf.set_xy(x, row_top + image_box + 1.2)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(cell_w, 3.8, _latin1(caption[:48]), align="C")
        detail = item.detail + ("  ·  under 500px" if small else "")
        if detail:
            pdf.set_xy(x, row_top + image_box + 5.2)
            pdf.set_font("Helvetica", "", 7.5)
            pdf.set_text_color(120)
            pdf.cell(cell_w, 3.6, _latin1(detail[:60]), align="C")
            pdf.set_text_color(0)

        included += 1
        column += 1
        if column >= COLUMNS:
            column = 0
            row_top += cell_h

    if included == 0:
        raise ValueError("None of these images could be read from img/.")

    return SheetResult(
        pdf=bytes(pdf.output()),
        included=included,
        skipped=skipped,
        undersized=undersized,
    )


def stamp() -> str:
    """A timestamp for filenames, sortable and free of characters paths dislike."""
    return datetime.now().strftime("%Y%m%d-%H%M")
