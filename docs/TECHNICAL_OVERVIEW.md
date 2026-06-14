# ImageCluster — Technical Overview & Handover

**Purpose.** Snapshot of the existing application (state: V-5.3 frozen, V-5.4 in
development) for transfer to other teams, forks, and further development. It
describes what the app does, how it is built, which embedding models work and
which do not, and where the known limits are.

**Version at time of writing:** `/api/status` reports `5.4-dev`. The frozen
baseline is V-5.3 (see `CHANGELOG.md`).

---

## 1. What ImageCluster does

ImageCluster is a **local desktop web app** for exploring image collections by
visual/semantic similarity. The user flow:

1. Put images in the `img/` folder.
2. Pick an **embedding model** (CLIP-family or vision-only).
3. **Generate** a 2D projection (UMAP or t-SNE) of the image embeddings, with
   optional **K-Means clustering**.
4. **Explore** the projection on an interactive canvas: pan/zoom, thumbnails,
   colored clusters, a **cluster gallery**, and **semantic text search** (for
   text-capable models).

It runs entirely on the local machine; no telemetry, no external services
except downloading model weights from Hugging Face on first use.

### Three screens
| Route | File(s) | Purpose |
|-------|---------|---------|
| `/` | `app/index.html` + `app/js/dashboard.js` | Dashboard: scan status, dependency status, two workflow cards |
| `/clip` | `app/clip_projection.html` + `app/js/clip_projection.js` (+ `redesign_ui.js`) | Embedding projection: the core workspace |
| `/models` | `app/models_tokens.html` + `app/js/models_tokens.js` (+ `redesign_settings.js`) | Info & settings: HF token, model registry, guide, glossary |
| `/integrated` | — | 302 redirect to `/clip` (legacy compatibility) |

---

## 2. Technology stack

### Frontend (intentionally framework-free)
- **Static HTML** served by FastAPI; **one CSS file** (`app/css/style.css`,
  token-first under `:root`); **vanilla JS modules** loaded per page.
- **HTML `<canvas>` 2D** for the projection plot (id `#plot`) — custom renderer
  with DPR handling, zoom/pan, hit-testing, level-of-detail and viewport culling.
- Fonts: **Geist** (UI) + **Geist Mono** (paths/ids/numbers).
- No build step, no React runtime in production. (The `UI-check/` prototype is
  React, used only as a design reference.)

### Backend (Python)
- **FastAPI** + **Uvicorn** (ASGI). `StaticFiles` serves `app/`, `img/`, `output/`.
- **PyTorch** for model execution.
- **OpenCLIP** for OpenCLIP-compatible models; **Transformers** for HF models;
  optional **ImageBind**.
- **UMAP** / **t-SNE** (scikit-learn) for projection; **K-Means** + silhouette
  score for clustering.
- **Pillow** + **NumPy** for image I/O and vectors.

### Backend source map
| File | Responsibility |
|------|----------------|
| `backend/main.py` | FastAPI app, routes, API endpoints, thumbnail endpoint |
| `backend/jobs.py` | Projection job creation, progress, results |
| `backend/cache.py` | Embedding cache helpers |
| `backend/reducers.py` | UMAP/t-SNE projection + K-Means clustering |
| `backend/image_scan.py` | Image folder scanning |
| `backend/config_store.py` | Local settings + HF token storage |
| `backend/dependency_check.py` | Runtime dependency checks |
| `backend/encoders/registry.py` | **Model registry** (single source of truth) |
| `backend/encoders/base.py` | Provider → encoder routing |
| `backend/encoders/openclip_encoder.py` | OpenCLIP image/text embeddings |
| `backend/encoders/transformers_encoder.py` | Transformers image/text embeddings |
| `backend/encoders/imagebind_encoder.py` | Optional ImageBind path |
| `backend/search/semantic_search.py` | Text search over embeddings |
| `backend/search/index_store.py` | Saved search index handling |

### Launcher / packaging
`Start Windows.bat` / `Start macOS.command` → `launcher.py` → creates `.venv`,
installs PyTorch + `requirements-core.txt`, then runs `run.py` (Uvicorn on
`127.0.0.1:8765`). Build scripts under `tools/build/`.

---

## 3. API surface (the contract)

Pages: `/`, `/clip`, `/models`, `/integrated`→`/clip`.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Dependency & runtime status, version, install advice |
| `GET /api/models` | Model registry (with `available` + `missing_requirements`) |
| `GET /api/config/local` · `POST /api/config/local/hf-token` | Local config + HF token save/remove |
| `GET /api/images/scan` | Scan the `img` folder |
| `POST /api/jobs/clip-projection` | Start a projection job |
| `GET /api/jobs/{id}` · `/result` · `/cancel` · `/download` | Job status / result TSV / cancel / TSV download |
| `GET /api/projections` · `/api/projections/read` | List / load saved projections |
| `POST /api/search/text` · `POST /api/search/rebuild-index` | Semantic search / rebuild index |
| `GET /api/thumb?path=&w=` | **Downscaled JPEG thumbnail** (disk-cached, path-safe) — added V-5.3 |

### HF token flow
`/models` saves a token via `/api/config/local/hf-token` → stored in
`output/local_settings.json` (git-ignored) → `config_store.py` exposes it to the
environment → the OpenCLIP/Transformers loaders read it before downloading gated
models.

---

## 4. The `/clip` rendering engine (performance-critical)

The plot is a **single `<canvas>`**, not thousands of DOM nodes. Key design,
after the V-5.3 performance pass (`app/js/clip_projection.js`):

- **Memoized data bounds** + **cached canvas rect** — the projection math was
  O(n²) with thousands of forced reflows per frame; now O(n) per frame.
- **rAF-coalesced redraws** (`scheduleDraw`) — one paint per frame regardless of
  hover/pan/wheel event bursts.
- **Viewport culling** — off-screen points are skipped.
- **Level of detail (LOD)** — above 500 images, the canvas draws colored points
  when zoomed out and switches to thumbnails at zoom ≥ 2×.
- **Low-resolution thumbnails** — canvas/gallery use `/api/thumb` (≈20× smaller
  than originals; disk-cached); the preview modal uses the full-resolution image.
- **Progressive image loading** — points render immediately; thumbnails stream in
  and repaint as they decode (no blocking on a full preload).

Measured: ~2,686 real images stay interactive (≈0.25 ms per hover event;
thumbnails ≈6.8 KB vs ≈133 KB originals).

**Stable DOM anchor ids** (the JS binds to these — do not rename without updating
the JS): `#modelKey`, `#reducer`, `#startBtn`, `#plot`, `#searchQuery`,
`#searchResults`, `#clusterPanel`, `#clusterSelect`, `#clusterGallery`.

**Cluster colors** are defined once in JS (`CLUSTER_PALETTE`) and mirrored by CSS
`--c1…--c6` so canvas points, legend, gallery swatches and badges never drift.

**Run state machine** (`idle → scanned → ready → running → done`) drives the
left-rail step badges, the empty state, and `#startBtn`.

---

## 5. Embedding models — what works and what does not

The registry (`backend/encoders/registry.py`) is the single source of truth.
**24 models** across three statuses (4 stable / 15 experimental / 5 planned). A model is **selectable** in the UI only
when `available` is true — i.e. its declared `requires` are importable and its
status is not `planned`/`unavailable`.

### Encoder routing (how a model becomes runnable)
| provider | encoder | loads via |
|----------|---------|-----------|
| `openclip`, `openclip_hf_hub` | `openclip_encoder.py` | `open_clip.create_model_and_transforms(model_id, pretrained=…)` or `hf-hub:` |
| `transformers_clip`, `transformers_image_features`, `transformers_vision_pool`, `transformers_metaclip2`, `nomic_transformers` | `transformers_encoder.py` | `AutoModel`/`AutoProcessor`, `get_image_features`/`get_text_features` |
| `sentence_transformers` | `sentence_transformers_encoder.py` | `SentenceTransformer(model_id, trust_remote_code=…).encode(...)` (optional dependency) |
| `imagebind` | `imagebind_encoder.py` | bespoke (manual install) |
| `planned` | — | not wired (placeholder) |

Both main encoders are **generic**: adding a model is usually a registry-data
change (checkpoint + provider + capability flags), not new code.

### ✅ Stable (4) — verified working
| Model | Provider | Text search | Notes |
|-------|----------|:-----------:|-------|
| OpenCLIP ViT-B-32 (LAION-2B) | openclip | yes | **Default**; CPU-friendly; end-to-end verified |
| CLIP OpenAI ViT-B/32 | transformers_clip | yes | CPU-friendly |
| SigLIP Base patch16 224 | transformers_image_features | yes | CPU-friendly |
| DINOv2 Base | transformers_vision_pool | **no** | Vision-only (projection/clustering; search bar disabled by design) |

### ◐ Experimental (13) — wired to real encoders; work when deps + checkpoint present
Downloads the checkpoint on first use; not all hardware-verified.

| Model | Provider | Text search | Caveat |
|-------|----------|:-----------:|--------|
| SigLIP 2 Base patch16 224 | transformers_image_features | yes | GPU recommended |
| MobileCLIP B (OpenCLIP) | openclip_hf_hub | yes | Lightweight |
| MobileCLIP2 S2 / B / S4 | openclip_hf_hub | yes | S4 GPU-recommended |
| MetaCLIP ViT-B/32 | openclip | yes | Pretrained tag must exist in the installed OpenCLIP build |
| MetaCLIP ViT-L/14 | openclip | yes | GPU recommended |
| MetaCLIP 2 Worldwide B/32 | transformers_metaclip2 | yes | Needs recent Transformers |
| Nomic Embed Vision v1.5 | nomic_transformers | **no** | `trust_remote_code=True`; vision-only |
| **MetaCLIP 2 Worldwide H/14** (V-5.4) | transformers_metaclip2 | yes | `facebook/metaclip-2-worldwide-huge-quickgelu`; large GPU; **not yet hardware-verified** |
| **EVA-CLIP L/14** (V-5.4) | openclip | yes | `EVA02-L-14`/`merged2b_s4b_b131k`; GPU; **not yet hardware-verified** |
| **EVA-CLIP bigE/14** (V-5.4) | openclip | yes | `EVA02-E-14`/`laion2b_s4b_b115k`; large GPU, multi-GB; **not yet hardware-verified** |
| **ImageBind Huge** | imagebind | no | ⚠️ **Inactive unless installed manually** — `imagebind` is not on PyPI; install from Meta's repo, then it activates automatically |
| **Qwen3-VL Embedding 2B** (V-5.5) | sentence_transformers | yes | ⚠️ Optional deps: `sentence-transformers` + `qwen-vl-utils` + `transformers>=4.57`; large GPU; inactive until installed; **not yet hardware-verified** |
| **Jina v5 Omni Small** (V-5.5) | sentence_transformers | yes | ⚠️ Optional dep `sentence-transformers`; GPU; **license CC BY-NC 4.0 (non-commercial)**; inactive until installed; **not yet hardware-verified** |

### ✖️ Planned (5) — NOT functional (no checkpoint/loader wired)
Shown in `/models` for roadmap visibility; disabled in the `/clip` selector.

| Model | Why still planned |
|-------|-------------------|
| MetaCLIP 2 · 2B Worldwide | Code path ready (`transformers_metaclip2`) but ~2B params → very large GPU/host RAM; niche |
| HQ-CLIP B/16 | Needs checkpoint + slot confirmation |
| Long-CLIP B/32 | Extended 248-token context → needs a custom loader (modified positional embeddings) |
| LaCLIP | Standard CLIP arch, but official weights are `.pt` files in `LijieFan/LaCLIP` — not an OpenCLIP `pretrained` tag nor an HF Transformers repo; needs a mirror or checkpoint-file loader |
| CLOC | Region-aware design + uncertain public checkpoint |

See `docs/PLANNED_MODELS_FEASIBILITY.md` for the integration playbook and the
promotion criteria (`planned → experimental → stable`).

---

## 6. Capability model

Each `ModelSpec` declares booleans that drive the UI and the engine:
`supports_image_embedding`, `supports_text_embedding`, `supports_text_search`,
`supports_projection`, plus `hardware_tier` (`cpu_ok` / `gpu_recommended` /
`large_gpu`) and `status` (`stable` / `experimental` / `planned`).

- **Projection/clustering** works for any `supports_projection` model.
- **Semantic text search** is enabled only when `supports_text_search` is true
  **and** the model is available; the search bar is disabled with an inline reason
  otherwise (driven by the flag, not a hard-coded list). Example: DINOv2 and Nomic
  are vision-only → no text search.

---

## 7. Storage & runtime folders
- `img/` — input collections.
- `output/embeddings/` — cached embeddings (`.npy` + manifest).
- `output/projections/` — projection results (`.json` + `.tsv`).
- `output/search/` — search exports.
- `output/thumb_cache/` — disk-cached canvas thumbnails (V-5.3).
- `output/local_settings.json` — local config incl. HF token (**git-ignored**).
- `output/logs/` — debug reports.

---

## 8. Known limitations & next steps
- **Memory at very large scale:** low-res thumbnails are still all preloaded; for
  collections well beyond a few thousand images an **LRU + lazy-load** (decode
  only visible, release off-screen) and **DOM gallery virtualization** are the
  recommended next optimizations (P3 in the performance plan).
- **Model verification debt:** the V-5.4 experimental models (EVA-CLIP L/14,
  EVA-CLIP bigE/14, MetaCLIP 2 H/14) are wired and structurally verified but need
  real load+encode verification on a GPU machine before promotion to `stable`.
- **Manual-dependency models:** ImageBind requires a manual install.
- **Desktop only:** no responsive/mobile layout (min usable width ~1180px).

---

## 9. How to run / verify (developer quickstart)
```
python -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements-core.txt
python run.py            # serves http://127.0.0.1:8765
python -m pytest tests   # registry + capability + search tests
```
Frontend has no build step; edit the static files under `app/` and reload.

---

## 10. Document map
- `README.md` — user-facing run instructions.
- `README_TECH_STACK.md` — stack details, file map, routes, anchor ids.
- `CHANGELOG.md` — V-5.3 (frozen) / V-5.4 (in development).
- `docs/PLANNED_MODELS_FEASIBILITY.md` — planned-model integration playbook.
- `docs/TECHNICAL_OVERVIEW.md` — this document.
- `UI-check/` — design reference prototype (React; not shipped).
