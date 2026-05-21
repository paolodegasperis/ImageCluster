from __future__ import annotations

from pydantic import BaseModel, Field


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1)
    model_key: str
    image_dir: str = "img"
    embedding_id: str | None = None
    top_k: int = Field(default=30, ge=1, le=500)
    threshold: float | None = Field(default=None, ge=-1.0, le=1.0)
    normalize: bool = True


class RebuildSearchIndexRequest(BaseModel):
    image_dir: str = "img"
    model_key: str
