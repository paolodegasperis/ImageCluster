"""Contact sheets: what goes in, what is refused, what is reported.

The sheets are built from a list of paths chosen in the browser, which makes
that list untrusted input arriving over HTTP. Most of what follows is about the
boundary rather than the layout.
"""

from __future__ import annotations

import re

import pytest
from PIL import Image

from backend import pdf_export as px
from backend.runtime_dirs import IMG_DIR

pytestmark = pytest.mark.skipif(not px.PDF_AVAILABLE, reason="fpdf2 is not installed")

PREFIX = "_pytest_sheet"


@pytest.fixture
def images(request):
    """Three images on disk: one large, one small, one well under the minimum."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    made = []
    for name, size in [("big", (1800, 1200)), ("mid", (640, 480)), ("tiny", (220, 300))]:
        path = IMG_DIR / f"{PREFIX}_{name}.jpg"
        Image.new("RGB", size, (90, 120, 160)).save(path, "JPEG")
        made.append(path)

    def cleanup():
        for path in made:
            path.unlink(missing_ok=True)

    request.addfinalizer(cleanup)
    return {name: f"img/{PREFIX}_{name}.jpg" for name in ("big", "mid", "tiny")}


def _items(*paths):
    return [px.SheetItem(relative_path=p, caption=p.rsplit("/", 1)[-1]) for p in paths]


def _embedded_long_sides(pdf: bytes) -> list[int]:
    widths = [int(v) for v in re.findall(rb"/Width\s+(\d+)", pdf)]
    heights = [int(v) for v in re.findall(rb"/Height\s+(\d+)", pdf)]
    return [max(w, h) for w, h in zip(widths, heights, strict=True)]


# --------------------------------------------------------------------------
# The boundary


@pytest.mark.parametrize("hostile", [
    "../backend/main.py",
    "img/../../backend/main.py",
    "img/../../../Windows/win.ini",
    "/etc/passwd",
    "img/../output/local_settings.json",
])
def test_paths_outside_the_image_folder_are_refused(hostile):
    """The path list comes from the browser; a crafted one must read nothing."""
    assert px._safe_source(hostile) is None


def test_a_real_image_inside_the_folder_resolves(images):
    assert px._safe_source(images["big"]) is not None


def test_a_sheet_of_only_unreachable_paths_fails_rather_than_producing_a_blank(images):
    with pytest.raises(ValueError, match="None of these images could be read"):
        px.build_image_sheet(title="T", items=_items("img/nope.jpg", "../secret.txt"))


def test_an_empty_sheet_is_refused():
    with pytest.raises(ValueError, match="nothing to export"):
        px.build_image_sheet(title="T", items=[])


# --------------------------------------------------------------------------
# What comes out


def test_the_result_is_a_pdf(images):
    result = px.build_image_sheet(title="Community 1", items=_items(images["big"], images["mid"]))
    assert result.pdf.startswith(b"%PDF-")
    assert result.included == 2


def test_every_readable_image_is_embedded(images):
    result = px.build_image_sheet(title="T", items=_items(*images.values()))
    assert result.included == 3
    assert len(_embedded_long_sides(result.pdf)) == 3


def test_large_images_are_reduced_but_stay_well_above_the_minimum(images):
    """1800px would bloat the file; the floor that matters is 500."""
    result = px.build_image_sheet(title="T", items=_items(images["big"]))
    long_side = _embedded_long_sides(result.pdf)[0]
    assert long_side <= px.MAX_LONG_SIDE
    assert long_side >= px.MIN_LONG_SIDE


def test_an_image_already_within_range_is_embedded_untouched(images):
    result = px.build_image_sheet(title="T", items=_items(images["mid"]))
    assert _embedded_long_sides(result.pdf)[0] == 640


def test_an_undersized_image_is_reported_and_never_enlarged(images):
    """Upscaling would invent detail and hide that the source is small."""
    result = px.build_image_sheet(title="T", items=_items(images["tiny"]))
    assert result.undersized == [images["tiny"]]
    assert _embedded_long_sides(result.pdf)[0] == 300


def test_a_missing_image_is_reported_not_silently_dropped(images):
    """A sheet holding fewer images than the result it came from is wrong."""
    result = px.build_image_sheet(title="T", items=_items(images["big"], "img/absent.jpg"))
    assert result.included == 1
    assert len(result.skipped) == 1
    assert result.skipped[0]["relative_path"] == "img/absent.jpg"


def test_the_sheet_carries_its_heading(images):
    result = px.build_image_sheet(
        title="Community 3",
        subtitle="27 images of 165",
        meta_lines=["Neighbor graph - k=4"],
        items=_items(images["big"]),
    )
    assert result.pdf.startswith(b"%PDF-")
    assert result.included == 1


def test_text_outside_latin1_does_not_break_the_document(images):
    """Core PDF fonts are latin-1; a Cyrillic filename must not raise."""
    result = px.build_image_sheet(
        title="Сообщество",
        items=[px.SheetItem(relative_path=images["big"], caption="Тест.jpg")],
    )
    assert result.pdf.startswith(b"%PDF-")


def test_many_images_flow_onto_several_pages(images):
    one = list(images.values())[0]
    result = px.build_image_sheet(title="T", items=_items(*([one] * 12)))
    pages = len(re.findall(rb"/Type\s*/Page[^s]", result.pdf))
    assert pages >= 2
    assert result.included == 12


def test_the_filename_stamp_is_path_safe():
    assert re.fullmatch(r"\d{8}-\d{4}", px.stamp())
