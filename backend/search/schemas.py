from __future__ import annotations

from pydantic import BaseModel, Field

from ..app_config import get as config_get


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model_key: str
    image_dir: str = Field(default_factory=lambda: config_get("default_image_dir", "img"))
    embedding_id: str | None = None
    top_k: int = Field(default_factory=lambda: int(config_get("semantic_search.default_top_k", 30)), ge=1, le=500)
    threshold: float | None = Field(default_factory=lambda: config_get("semantic_search.default_threshold"), ge=-1.0, le=1.0)
    normalize: bool = True


class RebuildSearchIndexRequest(BaseModel):
    image_dir: str = Field(default_factory=lambda: config_get("default_image_dir", "img"))
    model_key: str
