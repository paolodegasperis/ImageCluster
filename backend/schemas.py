from __future__ import annotations

from pydantic import BaseModel, Field


class ClipProjectionRequest(BaseModel):
    image_dir: str = Field(default="img")
    # v5 field: registry key for the selected embedding model.
    # model/pretrained are retained for backward compatibility with v4 clients.
    model_key: str | None = Field(default=None)
    model: str = Field(default="ViT-B-32")
    pretrained: str = Field(default="laion2b_s34b_b79k")
    reducer: str = Field(default="umap", pattern="^(umap|tsne)$")
    batch_size: int = Field(default=32, ge=1, le=256)
    use_cache: bool = True
    umap_n_neighbors: int = Field(default=15, ge=2, le=200)
    umap_min_dist: float = Field(default=0.1, ge=0.0, le=1.0)
    tsne_perplexity: int = Field(default=30, ge=2, le=100)
    tsne_max_iter: int = Field(default=1000, ge=250, le=5000)

    # Optional projection clustering. Labels are written into the TSV as
    # cluster, cluster_k and cluster_score columns when enabled.
    cluster_enabled: bool = False
    cluster_auto: bool = True
    cluster_k: int = Field(default=5, ge=2, le=50)
    cluster_min_k: int = Field(default=2, ge=2, le=50)
    cluster_max_k: int = Field(default=8, ge=2, le=50)


class JobCreated(BaseModel):
    job_id: str
    status: str


class LocalTokenRequest(BaseModel):
    token: str = Field(default="")
