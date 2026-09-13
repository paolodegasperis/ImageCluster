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


# --------------------------------------------------------------------------
# Over HTTP
#
# The module tests above call the builder directly. These cover the route: the
# contract the browser actually depends on, and the validation that stands
# between a posted path list and the filesystem.


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from backend import main

    return TestClient(main.app)


@pytest.fixture
def projection():
    """Any real saved projection: the route checks provenance before building."""
    from backend.runtime_dirs import PROJECTIONS_DIR

    found = sorted(PROJECTIONS_DIR.glob("*.tsv"))
    if not found:
        pytest.skip("no saved projection to attribute the sheet to")
    return f"output/projections/{found[0].name}"


def _payload(projection, items, **extra):
    return {"projection_path": projection, "title": "Community 1", "items": items, **extra}


def test_the_route_returns_a_pdf_as_an_attachment(client, projection, images):
    response = client.post("/api/exports/sheet", json=_payload(
        projection, [{"relative_path": images["big"], "caption": "big.jpg"}]))
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert "attachment" in response.headers["content-disposition"]
    assert response.content.startswith(b"%PDF-")


def test_the_route_reports_what_it_could_not_include(client, projection, images):
    """The browser shows these counts; without them a short sheet looks complete."""
    response = client.post("/api/exports/sheet", json=_payload(projection, [
        {"relative_path": images["big"]},
        {"relative_path": images["tiny"]},
        {"relative_path": "img/absent.jpg"},
    ]))
    assert response.status_code == 200
    assert response.headers["X-Sheet-Included"] == "2"
    assert response.headers["X-Sheet-Skipped"] == "1"
    assert response.headers["X-Sheet-Undersized"] == "1"


def test_the_route_refuses_a_path_list_that_escapes_the_image_folder(client, projection):
    """Every posted path is re-validated; none of these may be read."""
    response = client.post("/api/exports/sheet", json=_payload(projection, [
        {"relative_path": "../backend/main.py"},
        {"relative_path": "img/../../output/local_settings.json"},
    ]))
    assert response.status_code == 400
    assert b"main.py" not in response.content


def test_the_route_refuses_a_projection_outside_the_projections_folder(client, images):
    response = client.post("/api/exports/sheet", json=_payload(
        "../backend/main.py", [{"relative_path": images["big"]}]))
    assert response.status_code in (400, 404)


def test_the_route_rejects_an_empty_item_list(client, projection):
    response = client.post("/api/exports/sheet", json=_payload(projection, []))
    assert response.status_code == 422  # the schema requires at least one


def test_the_filename_is_sanitised(client, projection, images):
    """A filename is user input and ends up in a header."""
    response = client.post("/api/exports/sheet", json=_payload(
        projection, [{"relative_path": images["big"]}], filename="../../evil name.pdf"))
    assert response.status_code == 200
    disposition = response.headers["content-disposition"]
    assert ".." not in disposition
    assert "/" not in disposition.split("filename=")[1]


# --------------------------------------------------------------------------
# Images that fail only when their pixels are read
#
# PIL decodes lazily. An image small enough to skip the resize is first touched
# at save() time, which used to sit outside the per-item guard: identical
# corruption produced a clean "1 missing" note when the file was large and a
# 500 when it was small.


def test_a_truncated_small_image_is_skipped_not_fatal(request):
    path = IMG_DIR / f"{PREFIX}_truncated.jpg"
    Image.new("RGB", (800, 600), (200, 100, 50)).save(path, "JPEG")
    whole = path.read_bytes()
    path.write_bytes(whole[: len(whole) // 2])   # header intact, pixels missing
    request.addfinalizer(lambda: path.unlink(missing_ok=True))

    # Paired with a readable image so the sheet itself can still be produced.
    good = IMG_DIR / f"{PREFIX}_ok.jpg"
    Image.new("RGB", (900, 700), (10, 80, 30)).save(good, "JPEG")
    request.addfinalizer(lambda: good.unlink(missing_ok=True))

    result = px.build_image_sheet(
        title="T", items=_items(f"img/{PREFIX}_truncated.jpg", f"img/{PREFIX}_ok.jpg"))
    assert result.included == 1
    assert len(result.skipped) == 1
    assert result.skipped[0]["relative_path"].endswith("_truncated.jpg")


def test_a_decompression_bomb_is_skipped_not_fatal(request, monkeypatch):
    """DecompressionBombError descends from Exception, not OSError."""
    good = IMG_DIR / f"{PREFIX}_bomb.jpg"
    Image.new("RGB", (900, 700), (10, 80, 30)).save(good, "JPEG")
    request.addfinalizer(lambda: good.unlink(missing_ok=True))

    real_open = Image.open

    def exploding_open(path, *args, **kwargs):
        if str(path).endswith("_bomb.jpg"):
            raise Image.DecompressionBombError("too many pixels")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Image, "open", exploding_open)

    other = IMG_DIR / f"{PREFIX}_plain.jpg"
    Image.new("RGB", (600, 400), (30, 30, 90)).save(other, "JPEG")
    request.addfinalizer(lambda: other.unlink(missing_ok=True))

    result = px.build_image_sheet(
        title="T", items=_items(f"img/{PREFIX}_bomb.jpg", f"img/{PREFIX}_plain.jpg"))
    assert result.included == 1
    assert len(result.skipped) == 1
    assert "too many pixels" in result.skipped[0]["reason"]
