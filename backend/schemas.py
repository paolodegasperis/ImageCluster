from __future__ import annotations

from pydantic import BaseModel, Field

from .app_config import get as config_get

# Every default is resolved per request through a default_factory. A snapshot
# taken at import time would ignore app_config.reload_config(), which is what the
# tests and a config edit between runs rely on.


class ClipProjectionRequest(BaseModel):
    image_dir: str = Field(default_factory=lambda: config_get("default_image_dir", "img"))
    # v5 field: registry key for the selected embedding model.
    # model/pretrained are retained for backward compatibility with v4 clients.
    model_key: str | None = Field(default=None)
    model: str = Field(default="ViT-B-32")
    pretrained: str = Field(default="laion2b_s34b_b79k")
    reducer: str = Field(default_factory=lambda: config_get("default_reducer", "umap"), pattern="^(umap|tsne|semantic)$")
    batch_size: int = Field(default_factory=lambda: int(config_get("batch_size", 32)), ge=1, le=256)
    use_cache: bool = True
    umap_n_neighbors: int = Field(default=15, ge=2, le=200)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    tsne_perplexity: int = Field(default=30, ge=2, le=100)
    tsne_max_iter: int = Field(default=1000, ge=250, le=5000)

    # Optional projection clustering. Labels are written into the TSV as
    # cluster, cluster_k and cluster_score columns when enabled.
    cluster_enabled: bool = Field(default_factory=lambda: bool(config_get("optional_clustering.enabled_by_default", False)))
    cluster_auto: bool = True
    cluster_k: int = Field(default_factory=lambda: int(config_get("optional_clustering.fixed_k", 5)), ge=2, le=50)
    cluster_min_k: int = Field(default_factory=lambda: int(config_get("optional_clustering.auto_min_k", 2)), ge=2, le=50)
    cluster_max_k: int = Field(default_factory=lambda: int(config_get("optional_clustering.auto_max_k", 8)), ge=2, le=50)

    # Visual Features provider only: which scalar feature maps to each axis.
    # Ignored by the embedding-based providers.
    vf_x_feature: str = Field(default="hue")
    vf_y_feature: str = Field(default="brightness")

    # Semantic axes (reducer="semantic"). Each axis is the direction between two
    # text poles, so the plot's axes carry names the user chose. Requires a
    # text-capable model; ignored by every other reducer.
    semantic_x_positive: list[str] = Field(default_factory=list)
    semantic_x_negative: list[str] = Field(default_factory=list)
    semantic_y_positive: list[str] = Field(default_factory=list)
    semantic_y_negative: list[str] = Field(default_factory=list)
    semantic_x_label: str = Field(default="", max_length=120)
    semantic_y_label: str = Field(default="", max_length=120)


class LocalTokenRequest(BaseModel):
    token: str = Field(default="")


class ExportClustersRequest(BaseModel):
    projection_path: str
    # "all" exports every cluster into its own subfolder; otherwise the id of a
    # single cluster to export.
    cluster: str = Field(default="all")
    export_name: str | None = Field(default=None)


class SheetItemModel(BaseModel):
    """One image on an exported sheet, as the browser describes it."""

    relative_path: str
    caption: str = Field(default="", max_length=200)
    detail: str = Field(default="", max_length=200)


class ImageSheetRequest(BaseModel):
    """A contact sheet of whatever result the user is looking at.

    The items come from the browser because the sheet should show exactly what
    was on screen. Every path in them is re-validated against ``img/`` before
    anything is read: this list is user input, not a trusted query.
    """

    projection_path: str
    title: str = Field(default="Export", max_length=200)
    subtitle: str = Field(default="", max_length=300)
    meta_lines: list[str] = Field(default_factory=list, max_length=8)
    items: list[SheetItemModel] = Field(min_length=1, max_length=600)
    filename: str | None = Field(default=None, max_length=120)


class MosaicRequest(BaseModel):
    """Grid-mosaic layout of an existing projection."""

    projection_path: str
    # Left unset the grid is derived from the image count and the canvas aspect.
    columns: int | None = Field(default=None, ge=1, le=400)
    aspect: float = Field(default=1.0, gt=0.05, le=20.0)
    algorithm: str = Field(default="auto", pattern="^(auto|optimal|split)$")


class NeighborGraphRequest(BaseModel):
    """k-nearest-neighbour graph over the embeddings behind a projection."""

    projection_path: str
    k: int = Field(default_factory=lambda: int(config_get("views.neighbor_graph_default_k", 4)), ge=1, le=20)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)
    mutual_only: bool = False


class SimilarImagesRequest(BaseModel):
    """Rank a projection's images by similarity to one of its own images."""

    projection_path: str
    # Identify the query by path (what the UI has) or by row index (what a script has).
    relative_path: str | None = None
    index: int | None = Field(default=None, ge=0)
    top_k: int = Field(default=24, ge=1, le=500)
    min_similarity: float | None = Field(default=None, ge=-1.0, le=1.0)


class DuplicateGroupsRequest(BaseModel):
    """Find images that are effectively the same picture."""

    projection_path: str
    threshold: float = Field(default=0.98, gt=0.0, le=1.0)
    max_pairs: int = Field(default=20000, ge=100, le=200000)


class OutlierRequest(BaseModel):
    """Rank images by how unlike the rest of the collection they are."""

    projection_path: str
    k: int = Field(default=5, ge=1, le=20)
    top_k: int = Field(default=40, ge=1, le=500)


class ClusterSummaryRequest(BaseModel):
    """Size, cohesion and representative image for each cluster."""

    projection_path: str


class CompareImagesRequest(BaseModel):
    """Side-by-side comparison of two images in the same projection."""

    projection_path: str
    path_a: str
    path_b: str
    neighbours: int = Field(default=12, ge=1, le=100)


class ConceptScoresRequest(BaseModel):
    """Score every image of a projection against one concept."""

    projection_path: str
    phrases: list[str] = Field(min_length=1)
    normalize: bool = True


class SemanticAxisPreviewRequest(BaseModel):
    """Try two named axes against a loaded projection without re-running it."""

    projection_path: str
    x_positive: list[str] = Field(min_length=1)
    x_negative: list[str] = Field(min_length=1)
    y_positive: list[str] = Field(min_length=1)
    y_negative: list[str] = Field(min_length=1)


class DistinguishRequest(BaseModel):
    """Read the difference between two groups of images back out as words."""

    projection_path: str
    paths_a: list[str] = Field(min_length=1)
    paths_b: list[str] = Field(min_length=1)
    vocabulary: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class ConceptArithmeticRequest(BaseModel):
    """An image (or a phrase), minus an idea, plus another."""

    projection_path: str
    relative_path: str | None = None
    base_phrases: list[str] | None = None
    add: list[str] | None = None
    subtract: list[str] | None = None
    # How hard the phrases pull against the starting image. 0 returns the
    # image's plain neighbours; the upper end lets the concepts dominate.
    #
    # 1.0 measured over 25 starting images x 4 terms on a real collection:
    # 64% of results differ from the plain neighbours, and the concept lift is
    # 1.50 of the 2.02 sigma that searching the term alone can reach. At 3.0 it
    # is 86% and 1.95 sigma - so close to the ceiling that the starting image
    # has stopped mattering, which is not what "images like this one, but more
    # X" means. The slider still goes there for anyone who wants it.
    strength: float = Field(default=1.0, ge=0.0, le=5.0)
    top_k: int = Field(default=24, ge=1, le=200)


class DisagreementRequest(BaseModel):
    """Where two projections of the same images place them differently."""

    projection_a: str
    projection_b: str
    neighbours: int = Field(default=15, ge=2, le=50)
    top_k: int = Field(default=40, ge=1, le=500)
