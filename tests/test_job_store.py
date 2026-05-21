import csv
import time

from backend import jobs
from backend.jobs import JobState, JobStore, validate_projection_tsv
from backend.schemas import ClipProjectionRequest


def test_job_store_writes_and_reads_metadata(tmp_path):
    store = JobStore(tmp_path)
    job = JobState(job_id="job-1", status="completed", result_path="output/projections/a.tsv")
    store.save(job)

    loaded = store.load_all()

    assert loaded["job-1"].status == "completed"
    assert loaded["job-1"].result_path == "output/projections/a.tsv"
    assert store.path_for("job-1").read_text(encoding="utf-8").startswith("{")


def test_projection_tsv_validation_detects_missing_columns(tmp_path):
    invalid = tmp_path / "bad.tsv"
    invalid.write_text("filename\tx\ty\nimage.jpg\t1\t2\n", encoding="utf-8")

    result = validate_projection_tsv(invalid)

    assert result["valid"] is False
    assert "missing" in result["error"]


def test_projection_tsv_validation_accepts_current_schema(tmp_path):
    valid = tmp_path / "ok.tsv"
    with valid.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "relative_path", "x", "y", "model_key", "reducer"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"filename": "a.jpg", "relative_path": "img/a.jpg", "x": 1, "y": 2, "model_key": "openclip", "reducer": "umap"})

    result = validate_projection_tsv(valid)

    assert result["valid"] is True


def test_queue_executes_jobs_sequentially_by_default(monkeypatch, tmp_path):
    store = JobStore(tmp_path)
    monkeypatch.setattr(jobs, "STORE", store)
    with jobs.LOCK:
        jobs.JOBS.clear()
        jobs.CANCEL_REQUESTS.clear()

    events = []

    def fake_run(job_id, req):
        events.append(("start", job_id))
        time.sleep(0.05)
        jobs._set(job_id, status="completed", stage="completed", message="done", finished_at=time.time(), finished_at_iso=jobs._iso_now())
        events.append(("done", job_id))

    monkeypatch.setattr(jobs, "_run_clip_projection", fake_run)
    first = jobs.create_clip_projection_job(ClipProjectionRequest(model_key="openclip_vit_b_32"))
    second = jobs.create_clip_projection_job(ClipProjectionRequest(model_key="openclip_vit_b_32"))

    deadline = time.time() + 3
    while time.time() < deadline:
        if jobs.get_job(first.job_id).status == "completed" and jobs.get_job(second.job_id).status == "completed":
            break
        time.sleep(0.02)

    assert events == [("start", first.job_id), ("done", first.job_id), ("start", second.job_id), ("done", second.job_id)]
