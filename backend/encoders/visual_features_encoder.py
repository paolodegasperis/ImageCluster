"""Visual Features encoder.

Computes interpretable, formal image descriptors with OpenCV / scikit-image
instead of a multimodal model. The result is a feature matrix that can be
clustered (K-Means on the full standardized vector) and projected onto fixed,
interpretable axes (each axis is one named scalar feature), so the same image
always lands in the same spot regardless of the dataset.

The first ``len(FEATURE_NAMES)`` columns of the matrix are the named scalar
features (kept in their natural range so they can be normalized with
``FEATURE_RANGES`` for plotting). The remaining columns are HSV/Lab histogram
bins used only to enrich clustering.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image

# Scalar features that double as selectable, interpretable projection axes.
FEATURE_NAMES: list[str] = [
    "brightness",
    "saturation",
    "hue",
    "contrast",
    "edge_density",
    "texture",
    "symmetry",
]

# Fixed (dataset-independent) value ranges used to normalize an axis to [0, 1].
# Keeping these fixed is what makes the projection reproducible and comparable
# across different image sets.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "brightness": (0.0, 255.0),
    "saturation": (0.0, 255.0),
    "hue": (0.0, 180.0),  # OpenCV H channel is 0..180
    "contrast": (0.0, 128.0),
    "edge_density": (0.0, 1.0),
    "texture": (0.0, 1.0),
    "symmetry": (-1.0, 1.0),  # correlation between mirrored halves
}

# Histogram layout appended after the scalar features (channel, bin count).
_HISTOGRAMS: list[tuple[str, int]] = [
    ("hsv_h", 16),
    ("hsv_s", 8),
    ("hsv_v", 8),
    ("lab_l", 8),
    ("lab_a", 8),
    ("lab_b", 8),
]

# Images are analysed at a fixed size so histograms and the symmetry measure are
# comparable and the cost is bounded.
_ANALYSIS_SIZE = 256


def feature_column_names() -> list[str]:
    """Full ordered list of column names for the feature matrix."""
    names = list(FEATURE_NAMES)
    for channel, bins in _HISTOGRAMS:
        names.extend(f"{channel}_bin{i}" for i in range(bins))
    return names


def _load_rgb(path: Path) -> np.ndarray:
    """Load an image as an RGB uint8 array via PIL (robust to Unicode paths)."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _scalar_features(bgr: np.ndarray, gray: np.ndarray) -> dict[str, float]:
    import cv2
    from skimage.feature import local_binary_pattern

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = float(hsv[:, :, 0].mean())
    saturation = float(hsv[:, :, 1].mean())
    brightness = float(hsv[:, :, 2].mean())

    contrast = float(gray.std())

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float((edges > 0).mean())

    # LBP histogram entropy as a stable [0, 1] texture measure.
    p = 8
    lbp = local_binary_pattern(gray, P=p, R=1, method="uniform")
    n_bins = p + 2
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    hist = hist[hist > 0]
    if hist.size:
        entropy = float(-(hist * np.log2(hist)).sum())
        texture = entropy / np.log2(n_bins)
    else:
        texture = 0.0

    # Horizontal symmetry: correlation between the left half and the mirrored
    # right half.
    h, w = gray.shape
    half = w // 2
    if half > 0:
        left = gray[:, :half].astype("float64")
        right = cv2.flip(gray[:, w - half:], 1).astype("float64")
        lf, rf = left.ravel(), right.ravel()
        if lf.std() > 1e-6 and rf.std() > 1e-6:
            symmetry = float(np.corrcoef(lf, rf)[0, 1])
        else:
            symmetry = 1.0  # uniform halves are perfectly symmetric
    else:
        symmetry = 0.0

    return {
        "brightness": brightness,
        "saturation": saturation,
        "hue": hue,
        "contrast": contrast,
        "edge_density": edge_density,
        "texture": texture,
        "symmetry": symmetry,
    }


def _histogram_features(bgr: np.ndarray) -> np.ndarray:
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    ranges = {
        "hsv_h": (hsv, 0, 180),
        "hsv_s": (hsv, 1, 256),
        "hsv_v": (hsv, 2, 256),
        "lab_l": (lab, 0, 256),
        "lab_a": (lab, 1, 256),
        "lab_b": (lab, 2, 256),
    }
    parts: list[np.ndarray] = []
    for channel, bins in _HISTOGRAMS:
        source, idx, top = ranges[channel]
        hist = cv2.calcHist([source], [idx], None, [bins], [0, top]).flatten()
        total = hist.sum()
        parts.append(hist / total if total > 0 else hist)
    return np.concatenate(parts).astype("float32")


def _encode_one(path: Path) -> np.ndarray:
    import cv2

    rgb = _load_rgb(path)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (_ANALYSIS_SIZE, _ANALYSIS_SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    scalars = _scalar_features(bgr, gray)
    scalar_vec = np.array([scalars[name] for name in FEATURE_NAMES], dtype="float32")
    hist_vec = _histogram_features(bgr)
    return np.concatenate([scalar_vec, hist_vec]).astype("float32")


def encode_visual_features(
    image_paths: list[Path],
    spec=None,
    batch_size: int = 16,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, list[Path]]:
    """Compute the visual-feature matrix for the given images.

    Mirrors the ``encode_with_*`` contract: returns ``(matrix, encoded_paths)``
    and skips images that cannot be read instead of failing the whole job.
    The ``spec`` and ``batch_size`` arguments are accepted for signature
    compatibility with the other encoders.
    """
    vectors: list[np.ndarray] = []
    encoded_paths: list[Path] = []
    skipped: list[str] = []
    total = len(image_paths)

    for done, path in enumerate(image_paths, start=1):
        try:
            vectors.append(_encode_one(path))
            encoded_paths.append(path)
        except Exception as exc:  # pragma: no cover - per-image decode errors
            skipped.append(f"{path.name}: visual-feature extraction failed: {exc}")
        if progress:
            progress(done, total, f"Computed visual features for {done} of {total} images")

    if not vectors:
        details = "\n".join(skipped[:20])
        raise RuntimeError("No valid images could be analysed." + (f" Details:\n{details}" if details else ""))
    return np.vstack(vectors).astype("float32"), encoded_paths
