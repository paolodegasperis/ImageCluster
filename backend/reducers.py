"""Turning an embedding matrix into two coordinates.

Four methods, and which is "best" depends on what the map is being read for.
Measured on the 165-image reference collection, against both OpenCLIP B/32 and
EmbeddingGemma 2 embeddings, with the same stress the Map honesty panel reports
plus the share of each image's ten true nearest neighbours still among its ten
nearest on the map:

| method            | stress | rank corr. | kNN@10 |
|-------------------|--------|-----------|--------|
| UMAP (default)    | 0.376  | 0.609     | 50.6%  |
| t-SNE             | 0.358  | 0.687     | 52.2%  |
| MDS               | 0.311  | 0.764     | 36.4%  |
| densMAP           | 0.442  | 0.777     | 46.8%  |

Read that as a trade, not a ranking. MDS minimises the distortion of distances
directly - it is the literal answer to "lay this out so the picture agrees with
the vectors" - and pays for it in neighbourhoods. densMAP gets the *order* of
distances right more often than anything else while bending their scale most.
UMAP and t-SNE sit in between, which is why they stay the everyday choices.

The numbers above are for 165 images. They are not a law: re-measure on a larger
collection with the Map honesty panel before concluding anything about one.
"""

from __future__ import annotations

import numpy as np

# MDS is O(n**2) per iteration and runs several restarts: ~2 s at 1500 images,
# ~55 s at 3000 on the reference machine. Past that it stops being a projection
# a user waits for, so it is refused with an explanation rather than left to
# look like a hang.
MAX_MDS_IMAGES = 3_000


def reduce_embeddings(
    embeddings: np.ndarray,
    reducer: str,
    *,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    tsne_perplexity: int = 30,
    tsne_max_iter: int = 1000,
) -> np.ndarray:
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("At least two embeddings are required for projection.")

    if reducer in {"umap", "densmap"}:
        import umap
        n_neighbors = min(max(2, umap_n_neighbors), max(2, embeddings.shape[0] - 1))
        model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=42,
            # densMAP is the same algorithm with a density term added to the
            # objective: it keeps dense regions dense instead of inflating them
            # to fill the page, which is what makes it read the order of
            # distances better than plain UMAP.
            densmap=reducer == "densmap",
        )
        return model.fit_transform(embeddings).astype("float32")

    if reducer == "tsne":
        from sklearn.manifold import TSNE
        perplexity = min(tsne_perplexity, max(1, (embeddings.shape[0] - 1) // 3))
        perplexity = max(1, min(perplexity, embeddings.shape[0] - 1))
        kwargs = dict(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            metric="cosine",
            random_state=42,
        )
        try:
            model = TSNE(max_iter=tsne_max_iter, **kwargs)
        except TypeError:
            model = TSNE(n_iter=tsne_max_iter, **kwargs)
        return model.fit_transform(embeddings).astype("float32")

    if reducer == "mds":
        from sklearn.manifold import MDS

        count = embeddings.shape[0]
        if count > MAX_MDS_IMAGES:
            raise ValueError(
                f"This folder has {count} images; the MDS projection is limited to {MAX_MDS_IMAGES} "
                "because it compares every pair at every step. Use UMAP or t-SNE for a set this size, "
                "or project a subfolder."
            )

        # MDS is fed the cosine distances rather than the vectors: the whole
        # point of this method is that it reproduces *these* distances, so they
        # have to be the ones the rest of the application uses.
        unit = np.asarray(embeddings, dtype="float64")
        norms = np.linalg.norm(unit, axis=1, keepdims=True)
        unit = unit / np.where(norms > 0, norms, 1.0)
        distances = 1.0 - unit @ unit.T
        np.fill_diagonal(distances, 0.0)
        np.clip(distances, 0.0, None, out=distances)

        model = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            normalized_stress=False,
            n_init=4,
            max_iter=400,
        )
        return model.fit_transform(distances).astype("float32")

    raise ValueError(f"Unsupported reducer: {reducer}")


def project_visual_features(
    matrix: np.ndarray,
    feature_names: list[str],
    x_feature: str,
    y_feature: str,
) -> np.ndarray:
    """Project a visual-feature matrix onto two fixed, interpretable axes.

    Each axis is one named scalar feature, normalized with its fixed
    (dataset-independent) range so the same image always maps to the same
    coordinates. Returns coordinates scaled to a 0..100 plot area.
    """
    from .encoders.visual_features_encoder import FEATURE_RANGES

    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError("Visual-feature projection needs at least one image.")

    def axis(name: str) -> np.ndarray:
        if name not in feature_names:
            raise ValueError(f"Unknown visual-feature axis '{name}'. Available: {', '.join(feature_names)}")
        col = matrix[:, feature_names.index(name)].astype("float64")
        lo, hi = FEATURE_RANGES.get(name, (float(col.min()), float(col.max())))
        span = hi - lo
        if span <= 0:
            span = 1.0
        scaled = (col - lo) / span
        return np.clip(scaled, 0.0, 1.0) * 100.0

    return np.column_stack([axis(x_feature), axis(y_feature)]).astype("float32")


def cluster_feature_matrix(
    matrix: np.ndarray,
    *,
    auto: bool = True,
    fixed_k: int = 5,
    min_k: int = 2,
    max_k: int = 8,
) -> dict:
    """Cluster images by their full standardized visual-feature vector.

    Same auto-k / silhouette strategy as :func:`cluster_projection`, but runs on
    the N-dimensional feature matrix so groups reflect overall formal
    characteristics rather than only the two displayed axes.
    """
    if matrix.ndim != 2:
        raise ValueError("Cluster input must be a two-dimensional feature matrix.")

    n = matrix.shape[0]
    if n < 3:
        raise ValueError("At least three images are required for clustering.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(matrix.astype("float64"))

    def fit_k(k: int):
        k = int(max(2, min(k, n - 1)))
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X)
        except TypeError:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        unique = np.unique(labels)
        score = float(silhouette_score(X, labels)) if 2 <= unique.size < n else -1.0
        return k, labels, score

    if auto:
        min_k = int(max(2, min_k))
        max_k = int(min(max(min_k, max_k), n - 1))
        candidates = [k for k in range(min_k, max_k + 1) if 2 <= k < n] or [min(2, n - 1)]
        best = None
        for k in candidates:
            result = fit_k(k)
            if best is None or result[2] > best[2]:
                best = result
        selected_k, labels, score = best
    else:
        selected_k, labels, score = fit_k(fixed_k)

    return {
        "method": "kmeans_features",
        "auto": bool(auto),
        "k": int(selected_k),
        "score": float(score),
        "labels": (labels.astype(int) + 1).tolist(),
    }


def cluster_projection(
    coords: np.ndarray,
    *,
    auto: bool = True,
    fixed_k: int = 5,
    min_k: int = 2,
    max_k: int = 8,
) -> dict:
    """Cluster 2D projection coordinates with K-Means.

    Returns a serializable dictionary containing 1-based labels. If auto=True,
    the best k is chosen by silhouette score over the requested range.
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Cluster input must be a two-dimensional projection.")

    n = coords.shape[0]
    if n < 3:
        raise ValueError("At least three projected points are required for clustering.")

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    X = StandardScaler().fit_transform(coords.astype("float64"))

    def fit_k(k: int):
        k = int(max(2, min(k, n - 1)))
        try:
            model = KMeans(n_clusters=k, random_state=42, n_init="auto")
            labels = model.fit_predict(X)
        except TypeError:
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        unique = np.unique(labels)
        if unique.size < 2 or unique.size >= n:
            score = -1.0
        else:
            score = float(silhouette_score(X, labels))
        return k, labels, score

    if auto:
        min_k = int(max(2, min_k))
        max_k = int(max(min_k, max_k))
        max_k = int(min(max_k, n - 1))
        candidates = [k for k in range(min_k, max_k + 1) if 2 <= k < n]
        if not candidates:
            candidates = [min(2, n - 1)]
        best = None
        for k in candidates:
            result = fit_k(k)
            if best is None or result[2] > best[2]:
                best = result
        selected_k, labels, score = best
    else:
        selected_k, labels, score = fit_k(fixed_k)

    return {
        "method": "kmeans",
        "auto": bool(auto),
        "k": int(selected_k),
        "score": float(score),
        "labels": (labels.astype(int) + 1).tolist(),
    }


def project_semantic_axes(
    embeddings: np.ndarray,
    axis_x: np.ndarray,
    axis_y: np.ndarray,
) -> np.ndarray:
    """Project embeddings onto two named directions of the shared text-image space.

    The same shape as :func:`project_visual_features`: fixed, interpretable axes
    rather than a learned flattening. UMAP and t-SNE decide for themselves what
    the axes mean, and the answer is "nothing expressible"; here the direction is
    given, so left and right have names.

    Coordinates are the raw cosine of each image against each axis, in [-1, 1].
    They are emitted unscaled so the values in the TSV stay interpretable — the
    plot frames them from their own bounds, as it does for any projection.
    """
    values = np.asarray(embeddings, dtype="float32")
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("Semantic projection needs at least one embedding.")
    if axis_x.shape[-1] != values.shape[1] or axis_y.shape[-1] != values.shape[1]:
        raise ValueError(
            "The semantic axes do not match the embedding width, which means they came "
            "from a different model than the images."
        )

    norms = np.linalg.norm(values, axis=1, keepdims=True)
    unit = values / np.clip(norms, 1e-12, None)
    return np.column_stack([unit @ axis_x, unit @ axis_y]).astype("float32")
