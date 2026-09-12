"""In-memory job bookkeeping: pruning, cancellation and the concurrency slot."""

from __future__ import annotations

import time

import pytest

from backend import jobs
from backend.jobs import TERMINAL_STATES, JobState, cancel_job, get_job, list_jobs


@pytest.fixture(autouse=True)
def clean_registry():
    jobs.JOBS.clear()
    jobs.CANCEL_REQUESTS.clear()
    yield
    jobs.JOBS.clear()
    jobs.CANCEL_REQUESTS.clear()


def _add(job_id: str, status: str = "completed", age: float = 0.0) -> JobState:
    job = JobState(job_id=job_id, status=status)
    job.created_at = time.time() - age
    job.updated_at = time.time() - age
    jobs.JOBS[job_id] = job
    return job


def test_pruning_drops_the_oldest_finished_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "config_get", lambda path, fallback=None: 3 if path == "limits.max_jobs_in_memory" else fallback)
    for index in range(6):
        _add(f"job-{index}", "completed", age=100 - index)
    with jobs.LOCK:
        jobs._prune_jobs_locked()
    assert sorted(jobs.JOBS) == ["job-3", "job-4", "job-5"]


def test_pruning_never_drops_a_running_job(monkeypatch):
    monkeypatch.setattr(jobs, "config_get", lambda path, fallback=None: 2 if path == "limits.max_jobs_in_memory" else fallback)
    _add("old-running", "running", age=1000)
    for index in range(5):
        _add(f"done-{index}", "completed", age=100 - index)
    with jobs.LOCK:
        jobs._prune_jobs_locked()
    assert "old-running" in jobs.JOBS


def test_pruning_forgets_cancel_requests_for_removed_jobs(monkeypatch):
    monkeypatch.setattr(jobs, "config_get", lambda path, fallback=None: 1 if path == "limits.max_jobs_in_memory" else fallback)
    _add("stale", "cancelled", age=500)
    jobs.CANCEL_REQUESTS.add("stale")
    _add("fresh", "completed", age=1)
    with jobs.LOCK:
        jobs._prune_jobs_locked()
    assert "stale" not in jobs.JOBS
    assert "stale" not in jobs.CANCEL_REQUESTS


def test_pruning_is_disabled_when_the_limit_is_zero(monkeypatch):
    monkeypatch.setattr(jobs, "config_get", lambda path, fallback=None: 0 if path == "limits.max_jobs_in_memory" else fallback)
    for index in range(20):
        _add(f"job-{index}")
    with jobs.LOCK:
        jobs._prune_jobs_locked()
    assert len(jobs.JOBS) == 20


def test_cancel_marks_a_running_job_and_records_the_request():
    _add("live", "running")
    result = cancel_job("live")
    assert result.status == "cancelling"
    assert "live" in jobs.CANCEL_REQUESTS


def test_cancel_is_a_no_op_on_a_finished_job():
    _add("done", "completed")
    result = cancel_job("done")
    assert result.status == "completed"
    assert "done" not in jobs.CANCEL_REQUESTS


def test_cancel_returns_none_for_an_unknown_job():
    assert cancel_job("nope") is None
    assert get_job("nope") is None


def test_list_jobs_is_newest_first():
    _add("older", age=100)
    _add("newer", age=1)
    assert [job["job_id"] for job in list_jobs()] == ["newer", "older"]


def test_terminal_states_cover_every_finished_status():
    assert TERMINAL_STATES == {"completed", "failed", "cancelled"}


def test_safe_name_is_filesystem_safe():
    assert jobs._safe_name("ViT-B/32:laion2b") == "ViT-B-32-laion2b"
    assert jobs._safe_name("a b c") == "a-b-c"
