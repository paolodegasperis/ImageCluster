# Phase 5 Advanced Analysis Audit

## Cluster Reports

Status: partially implemented.

Existing pieces to reuse:
- Projection TSV files already contain `cluster`, `cluster_k`, `cluster_score` and `cluster_method` when K-Means succeeds.
- The canvas already colors cluster outlines and has a small cluster legend.
- Search result cards already show cluster values when available.

Missing pieces:
- No backend cluster report endpoint.
- No per-cluster counts, percentages, representatives, centroid-nearest images or exportable report.
- No frontend panel to isolate clusters from a report.

Risk:
- Old TSV files may not contain cluster columns. The report must return a clear no-clustering message.

## Model Comparison

Status: partially implemented.

Existing pieces to reuse:
- Saved projection TSV and sidecar JSON files include model metadata.
- `/api/projections` lists saved projection files.

Missing pieces:
- No projection catalog with extracted metadata.
- No endpoint to compare multiple projection artifacts.
- No frontend UI for model metadata side-by-side or neighborhood overlap.

Risk:
- UMAP/t-SNE coordinates from independent runs are not axis-aligned. Comparison must use shared images and neighborhood structure.

## Projection Comparison

Status: partially implemented.

Existing pieces to reuse:
- Multiple UMAP/t-SNE outputs can be saved and loaded one at a time.
- TSV rows share `relative_path`, which can align images across runs.

Missing pieces:
- No multi-projection comparison endpoint.
- No stability/nearest-neighbor overlap metrics.
- No UI for comparing UMAP vs t-SNE runs.

Risk:
- Raw x/y equality is misleading across independent reductions.

## Complete Session Saving

Status: missing.

Existing pieces to reuse:
- Frontend already tracks active projection, model, reducer, display settings, search results and filters such as search-result-only.
- Backend job persistence already writes JSON metadata under `output/jobs`.

Missing pieces:
- No `output/sessions` manifest store.
- No session API.
- No save/load UI.

Risk:
- Referenced projections or search files may be missing later. Loading must be partial and warn rather than fail hard.

## Standalone HTML Export

Status: missing.

Existing pieces to reuse:
- Projection TSV, search results, current session state and image-serving relative paths.
- Existing static canvas behavior can be approximated in an export-specific HTML file.

Missing pieces:
- No HTML package export endpoint.
- No frontend export action.

Risk:
- Fully self-contained base64 HTML can become very large. Package mode is safer for Phase 5.

## Combined Filters

Status: partially implemented.

Existing pieces to reuse:
- `showOnlySearch` already filters to semantic result paths.
- Search result scores are available in frontend state.
- Cluster, filename, relative path and subfolder metadata are already loaded in projection rows.

Missing pieces:
- No combined AND filter state.
- No cluster dropdown, score range, filename or subfolder controls.
- No visible count summary.
- Filter state is not saved in a session.

Risk:
- Filtering must not alter original projection/search TSV exports.

## Files Affected

- Backend: `backend/analysis.py`, `backend/main.py`, `backend/runtime_dirs.py`.
- Frontend: `app/clip_projection.html`, `app/js/clip_projection.js`, `app/css/style.css`.
- Documentation/tests: `docs/phase5-advanced-analysis-checklist.md`, `tests/test_analysis.py`.
