"""Shared fixtures.

The project is run from its own folder rather than installed, so tests need the
repository root on sys.path exactly the way run.py provides it at runtime.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def image_files(tmp_path: Path):
    """Three tiny real JPEGs, so encoders and scanners see valid files."""
    from PIL import Image

    folder = tmp_path / "pics"
    folder.mkdir()
    paths = []
    for index, colour in enumerate([(200, 30, 30), (30, 200, 30), (30, 30, 200)]):
        path = folder / f"img_{index}.jpg"
        Image.new("RGB", (32, 32), colour).save(path)
        paths.append(path)
    return paths
