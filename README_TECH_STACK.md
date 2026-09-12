# ImageCluster V53 - Web App Technical Stack

This document explains how the local web app is built, so the UI redesign can be reviewed with precise technical instructions.

## Purpose

ImageCluster is a local web application. The user starts the app from the simplified launcher, then opens a browser-based interface served from the local machine.

The web app has three main areas:

- Dashboard: first entry point with links to the workflow and settings.
- Embedding Projection: guided workflow for image projection, semantic search and cluster gallery.
- Info and settings: guide, model descriptions, Hugging Face token settings and glossary.

## Frontend Stack

The production frontend is intentionally simple:

- Static HTML pages served by FastAPI.
- Plain CSS in one main stylesheet.
- Vanilla JavaScript modules loaded directly by each page.
- HTML canvas for the projection graph.
- No frontend build step is currently required.
- No React runtime is used by the shipped app.

### Frontend Source Files

- `app/index.html`: dashboard / start page.
- `app/clip_projection.html`: embedding workflow, semantic search, graph canvas and cluster gallery.
- `app/models_tokens.html`: Info and settings page with Guide, Models and Glossary.
- `app/css/style.css`: global legacy styles plus V53 visual refinements.
- `app/js/dashboard.js`: dashboard scan/status behavior.
- `app/js/clip_projection.js`: projection workflow, canvas rendering, semantic search and cluster gallery behavior.
- `app/js/models_tokens.js`: model registry UI, token form, model drawer and filters.

### Design Reference Files

The shared UI design lives in:

- `V-52 _ Image Cluster-NEW-UI/Image Projector Redesign (standalone).html`
- `V-52 _ Image Cluster-NEW-UI/Image Projector Redesign.html`
- `V-52 _ Image Cluster-NEW-UI/main-view.jsx`
- `V-52 _ Image Cluster-NEW-UI/models-tokens.jsx`
- `V-52 _ Image Cluster-NEW-UI/start-menu.jsx`
- `V-52 _ Image Cluster-NEW-UI/tweaks-panel.jsx`

Important note: the JSX files are design/source references, not the current production frontend runtime. The production app must be updated in the static files under `app/`.

## Current UI Architecture

The current app still contains two visual layers:

- Legacy dark/global base styles at the top of `app/css/style.css`.
- V53 scoped styles applied through classes such as `v53-home`, `v53-app`, `v53-workflow`, `v53-panel` and `v53-hero`.

This is why some spacing, card and form details can still feel inconsistent. A complete UI pass should either:

- Continue hardening the V53 scoped layer until all active pages are visually consistent.
- Or split legacy and V53 styles into separate CSS files, then load only the V53 stylesheet on the new pages.

Recommended design pass order:

1. Define final design tokens: colors, typography, radii, shadows, spacing scale.
2. Normalize page shells: dashboard, workflow page, Info and settings page.
3. Normalize cards and panels: same radius, border, shadow, padding and heading rhythm.
4. Normalize controls: buttons, inputs, selects, checkboxes, chips and badges.
5. Normalize data areas: canvas panel, search results, cluster gallery and model cards.
6. Remove or isolate unused legacy CSS only after the active pages are verified.

## Frontend Behavior

### Projection Workflow

File:

- `app/js/clip_projection.js`

Main responsibilities:

- Load dependency and model status.
- Scan the `img` folder.
- Start projection jobs through the backend API.
- Poll job progress.
- Render the graph on `<canvas id="plot">`.
- Load saved projections.
- Export TSV/PNG.
- Run semantic search.
- Render search result thumbnails.
- Render Cluster Gallery when cluster data is available.

Important HTML anchors:

- `#modelKey`
- `#reducer`
- `#startBtn`
- `#plot`
- `#searchQuery`
- `#searchResults`
- `#clusterPanel`
- `#clusterSelect`
- `#clusterGallery`

When moving UI sections, keep these ids unchanged unless the JavaScript is updated at the same time.

### Info and Settings

File:

- `app/js/models_tokens.js`

Main responsibilities:

- Load `/api/models`.
- Render model cards.
- Render model details drawer.
- Filter models by search, status and capability.
- Load and save the local Hugging Face token through `/api/config/local`.

The Hugging Face token is not only visual configuration. The backend now reads the stored token and applies it to model loaders.

## Backend Stack

The backend is Python-based:

- FastAPI for the local HTTP API.
- Uvicorn as ASGI server.
- StaticFiles for serving `app/`, `img/` and `output/`.
- PyTorch for model execution.
- OpenCLIP for OpenCLIP-compatible models.
- Transformers for Hugging Face model loading.
- UMAP / t-SNE reducers for projection.
- scikit-learn K-Means and silhouette score for clustering.
- Pillow and NumPy for image processing and vectors.

## Backend Source Files

- `backend/main.py`: FastAPI app, routes and API endpoints.
- `backend/jobs.py`: projection job creation, progress and result management.
- `backend/cache.py`: embedding cache helpers.
- `backend/reducers.py`: UMAP/t-SNE projection and clustering.
- `backend/image_scan.py`: image folder scanning.
- `backend/config_store.py`: local settings and Hugging Face token storage.
- `backend/dependency_check.py`: runtime dependency checks.
- `backend/diagnostics.py`: debug report helpers.
- `backend/encoders/registry.py`: model registry shown in the UI.
- `backend/encoders/openclip_encoder.py`: OpenCLIP image/text embeddings.
- `backend/encoders/transformers_encoder.py`: Transformers image/text embeddings.
- `backend/encoders/imagebind_encoder.py`: optional ImageBind path.
- `backend/search/semantic_search.py`: text search over generated embeddings.
- `backend/search/index_store.py`: saved search index handling.

## Main Routes

Pages:

- `/`: dashboard.
- `/clip`: embedding projection workflow.
- `/models`: Info and settings.
- `/integrated`: redirects to `/clip`.

APIs:

- `/api/status`: dependency and runtime status.
- `/api/models`: model registry.
- `/api/config/local`: local configuration and token status.
- `/api/config/local/hf-token`: save or remove Hugging Face token.
- `/api/images/scan`: scan image folder.
- `/api/jobs/clip-projection`: start projection job.
- `/api/jobs/{job_id}`: job status.
- `/api/jobs/{job_id}/result`: projection TSV as JSON.
- `/api/search/text`: semantic search.
- `/api/search/rebuild-index`: rebuild searchable embeddings.

## Hugging Face Token Flow

User flow:

1. User opens `/models`.
2. User enters a Hugging Face token.
3. Frontend saves it through `/api/config/local/hf-token`.
4. Backend stores it locally in `output/local_settings.json`.
5. Model loaders read the stored token before downloading/loading models.

Backend behavior:

- `backend/config_store.py` exposes the token through standard environment variables.
- `backend/encoders/transformers_encoder.py` passes the token to `from_pretrained`.
- `backend/encoders/openclip_encoder.py` applies the token before OpenCLIP/HF Hub loading.

## Runtime And Launch Flow

The simplified launch flow remains outside the web UI:

- `Start Windows.bat`
- `Start macOS.command`
- `launcher.py`
- `installer_ui.py`
- `bootstrap/`
- `tools/`

The launcher checks Python, virtual environment, PyTorch and requirements before starting the local web app.

## Deploy Folder Policy

The final deploy folder should be a clean copy of the app, not a full working directory.

It should include:

- `app/`
- `backend/`
- `bootstrap/`
- `tools/`
- `icone/`
- `img/` if sample/empty image folder is required.
- `config.json`
- `installer_ui.py`
- `launcher.py`
- `project_paths.py`
- `run.py`
- `requirements-core.txt`
- `requirements.txt`
- `README.md`
- platform release folders under `Deploy/release/`.

It should not include:

- `.venv/`
- `__pycache__/`
- `output/` runtime results, unless intentionally shipping examples.
- `build*/`
- `dist*/`
- old build snapshots.
- temporary ZIP files generated during local packaging.
- local settings containing tokens.

## Suggested Next UI Review Inputs

To guide the UI pass precisely, provide notes in this format:

- Page: dashboard, workflow, Info and settings.
- Section: hero, controls, canvas, search, cluster gallery, model card, drawer.
- Desired change: spacing, columns, typography, color, card shape, button style, mobile behavior.
- Reference: line/section in the standalone HTML or a screenshot crop.

This will let the implementation stay aligned with the shared design without changing the working backend features.
