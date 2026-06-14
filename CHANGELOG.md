# Changelog

## V-5.5 (in development) — multimodal embedding models (Sentence-Transformers)

Added two recent multimodal embedding models that use the **Sentence-Transformers
API** (`model.encode`) rather than CLIP-style `get_image_features`. This required a
**new encoder + provider**, not pure wiring:

- New `backend/encoders/sentence_transformers_encoder.py` (image + text encoding,
  per-process model cache, batching, progress, per-image resilience), registered as
  provider `sentence_transformers` in `backend/encoders/base.py`.

New models (status `experimental`):

- **Qwen3-VL Embedding · 2B** — `Qwen/Qwen3-VL-Embedding-2B`, shared text/image space
  (2048-dim). Needs `sentence-transformers`, `transformers>=4.57`, `qwen-vl-utils`;
  large GPU.
- **Jina v5 Omni · Small** — `jinaai/jina-embeddings-v5-omni-small` (~1.74B, 1024-dim,
  Matryoshka). Needs `sentence-transformers`, `transformers>=4.57`; GPU recommended.
  **License CC BY-NC 4.0 (non-commercial).**

Both are **optional-dependency models** (like ImageBind): until `sentence-transformers`
(+ `qwen-vl-utils` for Qwen) is installed they report `available: false` and are
disabled in the selector. They activate automatically once installed.

> Verification status: registry/encoder wiring and UI behaviour verified structurally
> (12/12 tests; `/api/models` shows both `experimental`, disabled until deps present).
> Real load+encode must be verified on a GPU machine after installing the optional deps:
> `pip install sentence-transformers "transformers>=4.57" qwen-vl-utils peft`
> (Jina v5 Omni loads PEFT adapters, so `peft` is required).

## V-5.4 (in development) — more embedding models

**Wave A — planned roadmap models promoted to `experimental`** (wiring onto the
existing encoders + structural verification; real model load/encode must be
verified on a machine with the runtime + GPU, see below):

- **EVA-CLIP · L/14** — OpenCLIP path, `EVA02-L-14` / `merged2b_s4b_b131k` (GPU recommended).
- **EVA-CLIP · bigE/14** — OpenCLIP path, `EVA02-E-14` / `laion2b_s4b_b115k` (large GPU; multi-GB download, not a default).
- **MetaCLIP 2 · Worldwide H/14** — Transformers MetaCLIP 2 path, `facebook/metaclip-2-worldwide-huge-quickgelu` (large GPU; reuses the proven B/32 sibling loader).

All three are CLIP-family: image + text embeddings, projection and text search.
No encoder code changed — only registry entries (`backend/encoders/registry.py`).

**Still planned (not pure-wiring):**

- **LaCLIP** — standard CLIP architecture, but official weights ship as `.pt`
  checkpoints in the `LijieFan/LaCLIP` repo, not an OpenCLIP `pretrained` tag nor
  a Hugging Face Transformers repo. Needs a verified `hf-hub:` mirror or a
  checkpoint-file loader before promotion.
- HQ-CLIP B/16, Long-CLIP B/32, MetaCLIP 2 2B, CLOC — see
  `docs/PLANNED_MODELS_FEASIBILITY.md`.

### How to verify the Wave A models on a real machine
With the full runtime installed (torch + open_clip + transformers, GPU):
1. Open `/models`, filter by **experimental**, confirm the three cards render.
2. In `/clip`, scan `img`, select each model in turn, and **Generate** a small
   projection — first run downloads the checkpoint.
3. For each, run a text query in the search bar (all three are text-capable).
4. Promote to `stable` once load + projection + search are verified and the
   resource needs are acceptable for the declared hardware tier.

---

## V-5.3 (frozen) — performance + UI consolidation

Frozen baseline. Highlights:

- **Rendering performance** (`app/js/clip_projection.js`): memoized data bounds,
  cached canvas rect, `requestAnimationFrame`-coalesced redraws, viewport culling,
  level-of-detail (colored points when zoomed out, thumbnails when zoomed in), and
  a low-resolution thumbnail endpoint (`/api/thumb`, disk-cached) with progressive
  loading. Restores interactivity at ~2.5k+ images.
- **UI redesign** ported to the production static stack (dashboard, `/clip`,
  `/models`): three-column `/clip` shell, run-state machine driving the workflow
  steps, capability-aware semantic search bar, collapsible setup-help box.
- **ImagePlot workflow removed** (page, `/original` route, layout localStorage).
- Token-first CSS pass; legacy dark control leakage neutralized.
