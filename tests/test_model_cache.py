"""The encoder cache must actually stop checkpoints from reloading.

The first version of this cache held processors, tokenizers and weights in one
two-entry LRU. A single Transformers model needs three of those, so the small
objects evicted each other on every call and the expensive checkpoint sat one
step from eviction — the cache existed but delivered almost nothing.
"""

from __future__ import annotations

import threading

import pytest

from backend.encoders import model_cache
from backend.encoders.model_cache import AUX, WEIGHTS, cached_keys, clear, get_or_load


@pytest.fixture(autouse=True)
def empty_cache():
    clear()
    yield
    clear()


class Recorder:
    """Counts how many times each key was actually loaded."""

    def __init__(self):
        self.loads: list[str] = []

    def loader(self, name: str):
        def inner():
            self.loads.append(name)
            return f"<{name}>"

        return inner

    def count(self, name: str) -> int:
        return self.loads.count(name)


def test_a_second_request_for_the_same_key_does_not_reload():
    rec = Recorder()
    first = get_or_load("k", rec.loader("k"))
    second = get_or_load("k", rec.loader("k"))
    assert first is second
    assert rec.count("k") == 1


def test_weights_are_evicted_least_recently_used_first():
    rec = Recorder()
    for name in ["a", "b", "c"]:
        get_or_load(name, rec.loader(name))
    assert cached_keys(WEIGHTS) == ["b", "c"]
    # "a" is gone and must reload; "c" is still resident.
    get_or_load("a", rec.loader("a"))
    get_or_load("c", rec.loader("c"))
    assert rec.count("a") == 2
    assert rec.count("c") == 1


def test_using_a_key_refreshes_its_position():
    rec = Recorder()
    get_or_load("a", rec.loader("a"))
    get_or_load("b", rec.loader("b"))
    get_or_load("a", rec.loader("a"))  # a is now the most recent
    get_or_load("c", rec.loader("c"))  # evicts b, not a
    assert cached_keys(WEIGHTS) == ["a", "c"]
    assert rec.count("a") == 1


def test_the_aux_cache_does_not_compete_with_weights():
    """The defect this replaced: processors and weights shared one small budget."""
    rec = Recorder()

    def encode(model: str):
        get_or_load(f"proc|{model}", rec.loader(f"proc:{model}"), kind=AUX)
        get_or_load(f"weights|{model}", rec.loader(f"weights:{model}"))

    def search(model: str):
        get_or_load(f"textproc|{model}", rec.loader(f"textproc:{model}"), kind=AUX)
        get_or_load(f"weights|{model}", rec.loader(f"weights:{model}"))

    for _ in range(3):
        encode("m")
        search("m")

    assert rec.count("weights:m") == 1
    assert rec.count("proc:m") == 1
    assert rec.count("textproc:m") == 1


def test_two_models_both_stay_resident_at_the_default_cap():
    rec = Recorder()
    for _ in range(3):
        get_or_load("weights|a", rec.loader("a"))
        get_or_load("weights|b", rec.loader("b"))
    assert rec.count("a") == 1
    assert rec.count("b") == 1


def test_processors_survive_a_checkpoint_eviction():
    rec = Recorder()
    for model in ["a", "b", "c"]:
        get_or_load(f"proc|{model}", rec.loader(f"proc:{model}"), kind=AUX)
        get_or_load(f"weights|{model}", rec.loader(f"weights:{model}"))
    # Model "a"'s checkpoint was evicted by "c"; its processor must not have been.
    assert "weights|a" not in cached_keys(WEIGHTS)
    assert "proc|a" in cached_keys(AUX)


def test_the_aux_cache_is_still_bounded():
    rec = Recorder()
    for index in range(model_cache.MAX_CACHED_AUX + 5):
        get_or_load(f"aux|{index}", rec.loader(str(index)), kind=AUX)
    assert len(cached_keys(AUX)) == model_cache.MAX_CACHED_AUX


def test_clear_empties_both_caches():
    rec = Recorder()
    get_or_load("w", rec.loader("w"))
    get_or_load("a", rec.loader("a"), kind=AUX)
    clear()
    assert cached_keys() == []


def test_an_unknown_cache_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown cache kind"):
        get_or_load("k", lambda: 1, kind="somewhere-else")


def test_a_loader_failure_is_not_cached():
    calls = []

    def failing():
        calls.append(1)
        raise RuntimeError("checkpoint download failed")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            get_or_load("broken", failing)
    assert len(calls) == 2
    assert cached_keys() == []


def test_concurrent_requests_for_one_key_converge_on_one_object():
    """A load race may duplicate work once, but everyone must get the same object."""
    barrier = threading.Barrier(4)
    results = []

    def loader():
        barrier.wait()
        return object()

    def worker():
        results.append(get_or_load("shared", loader))

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 4
    assert len({id(item) for item in results}) == 1
