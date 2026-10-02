"""Single source of truth for the application name and version.

Before v6.0 the version was duplicated across main.py, config.json, launcher.py,
requirements.txt and the HTML titles, and the product name alternated between
"ImagePlot-CLIP" and "ImageCluster". Everything now reads from here.
"""

from __future__ import annotations

__version__ = "6.8.0"
__app_name__ = "ImageCluster"
# Kept for log/debug output so older bug reports remain searchable.
__legacy_name__ = "ImagePlot-CLIP"


def version_string() -> str:
    return f"{__app_name__} v{__version__}"
