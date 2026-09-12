"""Process-wide cache for loaded encoder models.

Before v6.0 every encoder except the Sentence-Transformers one rebuilt its model
from scratch on each call. The visible symptom was text search: encoding a single
short query reloaded a multi-GB checkpoint every time. Projection jobs paid the
same cost on every run even when the embeddings themselves were cached.

There are two caches, because the objects have wildly different costs:

``weights``
    Model checkpoints — hundreds of megabytes to several gigabytes each. Kept in
    a very small LRU, since holding every model a session has touched would
    exhaust RAM/VRAM faster than reloading costs time.

``aux``
    Processors, image transforms and tokenizers — a few megabytes at most, but
    each ``from_pretrained`` still hits disk. These must not share the weights
    budget: a single Transformers model needs a processor *and* a text processor
    *and* its weights, so counting them together made the small objects evict
    each other (and pushed the weights to the eviction edge) on every call.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

# Two checkpoints let the common "encode images, then search with the same model"
# flow stay warm while still allowing a switch to another model.
MAX_CACHED_WEIGHTS = 2
# Processors are cheap to hold; this only exists so the cache cannot grow without
# bound across a long session of model switching.
MAX_CACHED_AUX = 12

WEIGHTS = "weights"
AUX = "aux"

_CACHES: dict[str, OrderedDict[str, Any]] = {WEIGHTS: OrderedDict(), AUX: OrderedDict()}
_LIMITS = {WEIGHTS: MAX_CACHED_WEIGHTS, AUX: MAX_CACHED_AUX}
_LOCK = threading.Lock()


def _release_accelerator_memory() -> None:
    """Best-effort release of accelerator memory held by an evicted checkpoint."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _evict_locked(kind: str) -> None:
    cache = _CACHES[kind]
    limit = _LIMITS[kind]
    while len(cache) > limit:
        cache.popitem(last=False)
        if kind == WEIGHTS:
            _release_accelerator_memory()


def get_or_load(key: str, loader: Callable[[], Any], kind: str = WEIGHTS) -> Any:
    """Return the cached object for ``key``, loading it with ``loader`` if absent.

    The loader runs outside the lock because loading can take minutes; a race
    where two threads load the same key wastes work once but never corrupts the
    cache, and the second result is simply discarded.
    """
    if kind not in _CACHES:
        raise ValueError(f"Unknown cache kind '{kind}'. Use 'weights' or 'aux'.")

    with _LOCK:
        cache = _CACHES[kind]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]

    value = loader()

    with _LOCK:
        cache = _CACHES[kind]
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        cache[key] = value
        _evict_locked(kind)
        return value


def clear() -> None:
    with _LOCK:
        for kind, cache in _CACHES.items():
            had_weights = kind == WEIGHTS and bool(cache)
            cache.clear()
            if had_weights:
                _release_accelerator_memory()


def cached_keys(kind: str | None = None) -> list[str]:
    with _LOCK:
        if kind is None:
            return [key for cache in _CACHES.values() for key in cache]
        return list(_CACHES[kind])
