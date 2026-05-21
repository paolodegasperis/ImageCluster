# Backend Job Robustness Checklist

Use this checklist after backend job, cache, diagnostics or launcher changes.

## Job Persistence

- [ ] Start one projection job and verify `output/jobs/<job_id>.json` is created.
- [ ] Verify the job metadata moves from `queued` to `running` to `completed`.
- [ ] Verify completed jobs remain visible from `GET /api/jobs` after backend restart.
- [ ] Manually mark a job metadata file as `running`, restart backend, and verify it becomes `interrupted`.
- [ ] Verify failed jobs include an error summary and a log/debug path.

## Queue and Cancellation

- [ ] Start one projection job.
- [ ] Start a second expensive job and verify it stays queued until the first finishes.
- [ ] Cancel a queued job and verify it never starts.
- [ ] Cancel a running job and verify it becomes `cancelled` at a safe checkpoint.
- [ ] Verify cancellation does not mark the job as failed unless an unexpected exception occurs.

## Outputs and Cache

- [ ] Verify projection TSV files are readable and include the existing schema.
- [ ] Verify projection metadata JSON has `artifact_status: complete`.
- [ ] Verify embedding manifests include `complete: true`, app version and dataset fingerprint.
- [ ] Interrupt a job and verify incomplete cache artifacts are not reused.
- [ ] Verify search outputs still save JSON and TSV under `output/search/`.

## Diagnostics

- [ ] Open `GET /api/status` and verify existing frontend status still works.
- [ ] Open `GET /api/diagnostics` and verify Python, platform, output/img paths, package versions, Torch/CUDA/MPS and job counts are present.
- [ ] Verify diagnostics works in CPU mode without optional GPU libraries.

## Launchers and Workflows

- [ ] Verify the app still launches from Windows CPU launcher.
- [ ] Verify the app still launches from Windows CUDA launcher where CUDA is available.
- [ ] Verify macOS launchers still call the same startup path.
- [ ] Run the README workflow: scan `img/`, choose model, generate graph, explore, search, export PNG/TSV.
