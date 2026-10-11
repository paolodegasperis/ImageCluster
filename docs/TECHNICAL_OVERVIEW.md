# ImageCluster — Technical Overview & Handover

**Purpose.** Snapshot of the existing application for transfer to other teams,
forks, and further development. It describes what the app does, how it is built,
which embedding models work and which do not, and where the known limits are.

**Version at time of writing:** `/api/status` reports `6.6.0`
(`backend/version.py` is the single source of truth; see `CHANGELOG.md`).

---

## 1. What ImageCluster does

ImageCluster is a **local desktop web app** for exploring image collections by
visual/semantic similarity. The user flow:

1. Put images in the `img/` folder.
2. Pick an **embedding model** (CLIP-family or vision-only).
3. **Generate** a 2D projection (UMAP, t-SNE, MDS or densMAP) of the image embeddings, with
   optional **K-Means clustering**.
4. **Explore** the projection on an interactive canvas: pan/zoom, thumbnails,
   colored clusters, a **cluster gallery**, and **semantic text search** (for
   text-capable models).

It runs entirely on the local machine; no telemetry, no external services
except downloading model weights from Hugging Face on first use.

### Seven screens
| Route | File(s) | Purpose |
|-------|---------|---------|
| `/` | `app/index.html` + `app/js/dashboard.js` | Dashboard: scan status, dependency status, three workflow cards |
| `/clip` | `app/clip_projection.html` + `app/js/clip_projection.js` (+ `analysis_panels.js`, `redesign_ui.js`) | Embedding projection: the core workspace |
| `/graph` | `app/neighbor_graph.html` + `app/js/neighbor_graph.js` | Nearest-neighbor graph: standalone view over a saved projection's embeddings |
| `/lens` | `app/lens.html` + `app/js/lens.js` | Lenses: what separates two groups in words, concept arithmetic, model disagreement |
| `/guide` | `app/guide.html` | Plain-language explanation of every feature, its limits included |
| `/guide/it` | `app/guide.it.html` | The same guide in Italian, for a readership of scholars |
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
- **UMAP** / **densMAP** (umap-learn) and **t-SNE** / **MDS** (scikit-learn) for projection, each
  faithful to something different — see `backend/reducers.py` for the measured trade-off;
  **K-Means** + silhouette
  score for clustering.
- **Pillow** + **NumPy** for image I/O and vectors.

### Backend source map
| File | Responsibility |
|------|----------------|
| `backend/version.py` | **Single source of truth** for app name and version |
| `backend/app_config.py` | Typed access to `config.json`, merged over complete defaults |
| `backend/housekeeping.py` | Ceilings on log files and the thumbnail cache |
| `backend/main.py` | FastAPI app, routes, API endpoints, thumbnail endpoint |
| `backend/jobs.py` | Projection job creation, progress, results |
| `backend/cache.py` | Embedding cache helpers |
| `backend/reducers.py` | UMAP/t-SNE/MDS/densMAP/semantic-axis projection + K-Means clustering |
| `backend/layouts.py` | Grid-mosaic assignment (exact LAP, or recursive median split) |
| `backend/neighbors.py` | k-NN graph over embeddings: cosine, mutual flags, components, Louvain communities, bridges, articulation points |
| `backend/views.py` | Mosaic and graph views over a saved projection on disk |
| `backend/analysis.py` | Similar images, duplicates, outliers, cluster summaries, pair comparison |
| `backend/pdf_export.py` | PDF contact sheets shared by every export surface; re-validates caller-supplied paths against `img/` |
| `backend/semantic.py` | The shared text-image space beyond search: concept scores, semantic axes, distinguish, concept arithmetic |
| `backend/disagreement.py` | Neighbourhood overlap between two projections of the same folder |
| `backend/image_scan.py` | Image folder scanning |
| `backend/config_store.py` | Local settings + HF token storage |
| `backend/dependency_check.py` | Runtime dependency checks |
| `backend/encoders/registry.py` | **Model registry** (single source of truth) |
| `backend/encoders/base.py` | Provider → encoder routing |
| `backend/encoders/model_cache.py` | Shared LRU of loaded checkpoints: 2 weight entries, 12 auxiliary (processors, tokenizers) — a single budget let the two evict each other |
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

Pages: `/`, `/clip`, `/graph`, `/lens`, `/guide`, `/guide/it`, `/models`,
`/integrated`→`/clip`.

**Italian documentation.** Only the documentation is translated, never the
interface: the guide page, the two in-place help panels and the glossary
definitions. Parallel content rather than a branch or a string catalogue -
this is prose with no logic and no interpolation, so a second copy costs
nothing and cannot drift from any code. English is the source and the fallback;
where the two disagree the English one is right. The panels carry both
languages in the markup with one hidden, and `app/js/help_language.js` holds
the single shared preference (English by default). UI labels stay English even
inside Italian prose, so a reader can still find them on screen -
`docs/GLOSSARIO_IT.md` is the binding rule sheet.

**Concept arithmetic combines in score space, not vector space** (v6.4.1). CLIP
keeps images and text in two separate cones: on a real collection an image sits
0.13-0.23 from every phrase whatever either depicts, while image-to-image
similarity reaches 0.81. Adding a text vector to an image vector is therefore
swamped, and the endpoint used to return the base image's neighbours for every
query. Each term is now scored separately and standardised (`semantic._standardise`)
before being summed, with a `strength` weight on the phrases. Scores returned are
standard deviations, not cosines.

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
| `GET /api/config` | UI defaults sourced from `config.json` — added v6.0 |
| `POST /api/views/mosaic` | Grid-mosaic assignment for a saved projection — added v6.0 |
| `POST /api/views/neighbor-graph` | k-NN graph over a projection's embeddings — added v6.0 |
| `GET /api/views/neighbor-graph/available?path=` | Pre-flight: can this projection produce a graph — added v6.0 |
| `POST /api/analysis/similar` · `/duplicates` · `/outliers` · `/clusters` | Four analyses over a projection's stored embeddings — added v6.2 |
| `POST /api/analysis/compare` | Two images side by side with reciprocal ranks and collection context — added v6.3 |
| `GET /api/semantic/vocabulary` | The default term list used by `distinguish` — added v6.4 |
| `POST /api/semantic/concept-scores` | Every image scored against a phrase, with percentile ramp bounds — added v6.4 |
| `POST /api/semantic/axis-preview` | Coordinates on two named axes, without re-running the projection — added v6.4 |
| `POST /api/semantic/distinguish` | The direction between two groups, read back as words — added v6.4 |
| `POST /api/semantic/arithmetic` | Image ± phrases, ranked against the collection — added v6.4 |
| `POST /api/analysis/similar-set` | Rank the collection by likeness to several images at once, with unwanted ones subtracted - added v6.13 |
| `POST /api/analysis/shepard` | Embedding distance against map distance over a sample of pairs, with rank correlation, Kruskal stress-1 and the worst-described pairs in each direction - added v6.10 |
| `GET /compare` | The Compare page: pick two images from a mosaic of the collection and see everything it knows about the pair - added v6.12 |
| `POST /api/analysis/chain` | Shortest route between two images through the k-NN graph, weighted by `1 - similarity` - added v6.10 |
| `POST /api/analysis/disagreement` | Neighbourhood overlap between two projections — added v6.4 |
| `POST /api/exports/sheet` | Any result as a PDF contact sheet; one route serves all three surfaces — added v6.5 |
| `GET /favicon.ico` | The project icon, also answering the request browsers make unprompted — added v6.5 |

**Exports.** PDFs are generated server-side (`pdf_export.py`) because `/api/thumb` clamps at 512px, which
leaves no headroom under the 500px floor the sheets are built around; on the server the originals are on
disk. The item list is chosen in the browser and re-validated path by path in `_safe_source()` — it is user
input over HTTP, not a trusted query. `fpdf2` is optional: absent, the route answers 503 and the rest of the
app is unchanged, the same treatment networkx gets. CSV is written in the browser (`download_helpers.js`)
because the data is already there; its escaping is covered by `node --test`.

**Semantic endpoints and the model.** The five `/api/semantic/*` routes and
`axis-preview` load the projection's *own* text encoder — using a different model
would compare vectors from two unrelated spaces and yield numbers that look fine
and mean nothing. `semantic.text_model_for()` enforces this and refuses
image-only models (DINOv2, Visual Features) by name. `/api/analysis/disagreement`
loads no model at all: it reads two sets of stored embeddings and compares
neighbourhoods, because two spaces share no coordinate frame.

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

### ◐ Experimental (15) — wired to real encoders; work when deps + checkpoint present
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
| **HQ-CLIP B/16** (v6.7.0) | openclip_hf_hub | yes | `hf-hub:zhixiangwei/vlm150m-hqclip-large-vitb16`, Apache-2.0; **hardware-verified on CPU** (~47 ms/image, about 2× the B/32 default) |
| **PE-Core B/16** (v6.7.0) | openclip_hf_hub | yes | `hf-hub:timm/PE-Core-B-16`, Apache-2.0; pulls its vision tower through `timm`; 1024-d embedding; **text context 32 tokens** — long queries are truncated; **hardware-verified on CPU** (~112 ms/image) |
| **CultureCLIP B/32** (v6.8.0) | transformers_clip | yes | `lukahh/cultureclip_sd_0303_32`, a fine-tune of `openai/clip-vit-base-patch32`; **licence undeclared on the Hub** — the CulTwin training set is CC BY-NC 4.0, so treat it as non-commercial; **hardware-verified** (~30 ms/image on CPU, ~11 ms on an RTX 5070 - within a millisecond of the B/32 default either way) |
| **EmbeddingGemma 2** (v6.9.0) | sentence_transformers | yes | `google/embeddinggemma-2`, Apache-2.0; one shared 768-d space for text and images, 100+ languages, 8192-token context; ⚠️ optional dep `sentence-transformers>=6.1`, inactive until installed; **hardware-verified** (~107 ms/image on an RTX 5070, **~2.2 s/image on CPU**) |

### ✖️ Planned (4) — NOT functional (no checkpoint/loader wired)
Shown in `/models` for roadmap visibility; disabled in the `/clip` selector.

| Model | Why still planned |
|-------|-------------------|
| MetaCLIP 2 · 2B Worldwide | Code path ready (`transformers_metaclip2`) but ~2B params → very large GPU/host RAM; niche |
| Long-CLIP B/32 | Extended 248-token context → needs a custom loader (modified positional embeddings) |
| LaCLIP | **Not integrated by decision, not by obstacle** (v6.7.0). Official weights remain `.pt` files in `LijieFan/LaCLIP`. A community mirror (`helena-balabin/laclip-vit-b16`, Oct 2025) is a structurally valid CLIP B/16 and would load through `transformers_clip` with no new code, but it declares no licence, ships an empty auto-generated model card and offers no way to prove the weights are the ones the paper describes. Provenance that thin does not belong in a citable tool |
| CLOC | Region-aware design + uncertain public checkpoint |

### Licences

Nine active checkpoints restrict use. The registry records this per model in
`ModelSpec.license`, and both `/clip` and `/models` display it beside the
publication date.

| Models | Licence |
|---|---|
| MetaCLIP B/32 and L/14, MetaCLIP 2 Worldwide B/32 and H/14, Jina v5 Omni Small | **CC BY-NC 4.0 — non-commercial** |
| MobileCLIP B, MobileCLIP2 S2 / B / S4 | **`apple-amlr` — research use only, and revocable** |
| OpenCLIP, EVA-CLIP | MIT |
| SigLIP, SigLIP 2, DINOv2, Nomic, HQ-CLIP, PE-Core, Qwen3-VL, EmbeddingGemma 2 | Apache-2.0 |
| CultureCLIP B/32 | **undeclared** — code Apache-2.0, CulTwin dataset CC BY-NC 4.0; assume non-commercial |
| Visual Features | none — computed locally, no weights |

### Text query padding

Two settings that decide whether search works, both verified and both covered by
`tests/test_text_query_padding.py`:

- The Transformers path pads queries with `padding="max_length"`. SigLIP and
  SigLIP 2 are trained on captions padded to the full 64-token context; padded
  to the longest line of a batch instead, their text towers return near-identical
  vectors for unrelated queries and search ranks at chance while the images
  still encode correctly. CLIP-family towers are indifferent to the choice.
- The OpenCLIP path blanks everything after the end-of-text token. MobileCLIP2
  S2 and S4 ship a tokenizer that pads with that token repeated, and their towers
  collapse on it — pairwise similarity 0.95 between unrelated sentences, against
  0.50 once blanked.

A degenerate text tower is invisible on real images, because a plausible picture
always comes back. Four solid-colour images and the queries "a solid red image",
"a solid green image" and so on expose it in seconds: a healthy model scores 4/4,
a collapsed one answers every query with the same image.

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
- `output/search/` — search exports (opt-in via `semantic_search.save_search_results`, off by default since v6.0).
- `output/thumb_cache/` — disk-cached canvas thumbnails (V-5.3), trimmed to `limits.thumb_cache_max_mb`.
- `output/exports/` — cluster exports.
- `output/local_settings.json` — local config incl. HF token (**git-ignored**).
- `output/logs/` — debug reports, trimmed to `limits.max_log_files`.

Embedding manifests record the **exact encoded paths** (v6.0+), not just a count:
encoders skip unreadable files, so a fresh directory scan cannot be assumed to
match the stored vectors row for row.

---

## 8. Known limitations & next steps
- **Gallery virtualization:** canvas thumbnails are lazy-loaded with an LRU since
  v6.0, but the cluster and search galleries still render every card into the DOM;
  virtualizing them is the remaining P3 optimization.
- **Neighbor graph scale:** capped at 5000 nodes. Beyond that the cosine k-NN and
  the edge payload both need an approximate index rather than a dense pass.
- **Mosaic exactness:** the optimal assignment is O(n^3) and limited to 2000
  images; larger sets use the recursive split, which preserves locality well but
  is not globally optimal.
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
python -m pytest         # run from .venv, which has networkx, httpx and fpdf2
python -m ruff check .   # lint (config in pyproject.toml)
node --test tests/js/*.test.js   # the CSV writer and export filenames
```
Frontend has no build step; edit the static files under `app/` and reload.

`node --test` uses Node's built-in runner, so there is no package.json and
nothing to install. It covers `app/js/download_helpers.js`, whose CSV escaping
is the one piece of frontend formatting whose mistakes are invisible until a
file is opened in a spreadsheet. Note the runner does not accept a bare
directory here: pass the glob.

---

## 10. Document map
- `README.md` — user-facing run instructions.
- `README_TECH_STACK.md` — stack details, file map, routes, anchor ids.
- `CHANGELOG.md` — release history from v5.x onwards.
- `docs/PLANNED_MODELS_FEASIBILITY.md` — planned-model integration playbook.
- `docs/TECHNICAL_OVERVIEW.md` — this document.
- `UI-check/` — design reference prototype (React; not shipped).
