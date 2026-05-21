from __future__ import annotations

import numpy as np


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

    if reducer == "umap":
        import umap
        n_neighbors = min(max(2, umap_n_neighbors), max(2, embeddings.shape[0] - 1))
        model = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=42,
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

    raise ValueError(f"Unsupported reducer: {reducer}")


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
