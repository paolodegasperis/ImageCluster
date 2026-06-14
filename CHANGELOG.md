# Changelog

## V-5.5 — multimodal embedding models via Sentence-Transformers

### New models

Two new multimodal embedding models added as optional dependencies, using the
**Sentence-Transformers API** (`model.encode`) rather than the CLIP-style
`get_image_features`. This required a dedicated encoder provider, not just
registry wiring.

- **Qwen3-VL Embedding · 2B** — `Qwen/Qwen3-VL-Embedding-2B`. Joint text/image
  embedding space (2048-dim). Requires `sentence-transformers`, `transformers>=4.57`,
  `qwen-vl-utils`. Large GPU.
- **Jina v5 Omni · Small** — `jinaai/jina-embeddings-v5-omni-small` (~1.74B,
  1024-dim, Matryoshka). Requires `sentence-transformers`, `transformers>=4.57`,
  `peft`. GPU recommended. **License: CC BY-NC 4.0 (non-commercial use only).**

Both models follow the optional-dependency pattern: they are listed in the model
selector but disabled until the matching packages are installed. A double-click
installer in the project root activates them without requiring terminal knowledge.

### New encoder

- `backend/encoders/sentence_transformers_encoder.py` — new provider handling image
  and text encoding, per-process model cache, batching, progress reporting, and
  per-image error resilience.
- `backend/encoders/base.py` — routing for `provider = "sentence_transformers"` in
  both `encode_images()` and `encode_texts()`.

### New installer scripts

- `Install additional models (Windows).bat` — root shortcut (double-click)
- `Install additional models (macOS).command` — root shortcut (double-click)
- `bootstrap/windows/install_optional_multimodal.bat`
- `bootstrap/macos/install_optional_multimodal.command`

All install: `sentence-transformers transformers>=4.57 qwen-vl-utils peft`

### Other changes

- Version string updated to `5.5-dev` in `backend/main.py`.
- Registry dependency key `peft` added to `IMPORT_NAMES`.
- Tests extended: `test_v55_multimodal_models_are_registered()` covers both new models.

> Verification status: registry, encoder wiring, and UI behaviour verified
> structurally (12/12 tests). Real load + encode must be confirmed on a GPU machine
> after installing the optional deps.

---

## V-5.4 — model registry expansion (Wave A)

Three roadmap models promoted from `planned` to `experimental` — wired to existing
encoders, no new encoder code required.

### Promoted models

- **EVA-CLIP · L/14** — OpenCLIP path, `EVA02-L-14` / `merged2b_s4b_b131k`.
  GPU recommended.
- **EVA-CLIP · bigE/14** — OpenCLIP path, `EVA02-E-14` / `laion2b_s4b_b115k`.
  Large GPU; multi-GB checkpoint, not a default choice.
- **MetaCLIP 2 · Worldwide H/14** — Transformers MetaCLIP 2 path,
  `facebook/metaclip-2-worldwide-huge-quickgelu`. Large GPU. Reuses the proven
  MetaCLIP B/32 sibling loader.

All three support text search (joint image/text embedding space).

### Still planned

- **LaCLIP** — architecture is standard CLIP, but official weights ship as `.pt`
  checkpoints in the `LijieFan/LaCLIP` GitHub repo, not as an OpenCLIP pretrained
  tag or a Hugging Face Transformers repo. Needs a verified `hf-hub:` mirror or a
  checkpoint-file loader before promotion.
- HQ-CLIP B/16, Long-CLIP B/32, MetaCLIP 2 2B, CLOC — see
  `docs/PLANNED_MODELS_FEASIBILITY.md`.

### How to verify Wave A models

With torch + open_clip + transformers installed (GPU recommended):
1. Open `/models`, filter by **Experimental** — confirm the three cards appear.
2. In `/clip`, scan `img`, select each model, click **Generate** — first run
   downloads the checkpoint.
3. Run a text search query for each — all three are text-capable.
4. Promote to `stable` once load, projection, and search are verified and hardware
   requirements are acceptable.

---

## V-5.3 — performance and UI consolidation (frozen baseline)

### Rendering performance (`app/js/clip_projection.js`)

- **Memoized bounds** — data extent computed once per projection load (O(n)
  single-pass), invalidated on new data.
- **Cached canvas rect** — `getBoundingClientRect()` called once and reused across
  mouse events instead of once per point per event.
- **rAF coalescence** — `scheduleDraw()` prevents more than one redraw per animation
  frame regardless of event rate.
- **Viewport culling** — off-screen points are skipped in `draw()`.
- **Level-of-detail** — colored points at zoom < 2× when the collection has more than
  500 images; thumbnails at zoom ≥ 2×. Restores smooth interaction at 2,500+ images.
- **`/api/thumb` endpoint** — PIL resize, SHA1 disk cache in `output/thumb_cache/`,
  path-traversal protected, ~20× smaller than originals. `Cache-Control: max-age=86400`.
- **Progressive thumbnail loading** — non-blocking; each thumbnail triggers a
  `scheduleDraw()` on load so the canvas fills in without freezing.

### UI redesign (production stack)

- Three-column `/clip` shell: left rail (workflow steps), centre (canvas), right panel
  (search + cluster gallery).
- Run-state machine: `idle → scanned → ready → running → done` drives step badges
  and the Generate button.
- Capability-aware semantic search bar: 3-state indicator (ok / warn / danger) with
  tooltip distinguishing capability from runtime availability.
- Step panels are collapsible (click the step header).
- Collapsible setup-help box in the dashboard (replaces inline install-advice block
  that was leaking into the `/clip` left rail).
- Token-first CSS pass; legacy dark control leakage fixed (selects, checkboxes,
  range inputs, modal, model detail buttons).
- `localStorage` namespace: `imageplot.*` → `imagecluster.*`.
- Download filenames: `imageplot-*` → `imagecluster-*`.
- Keyboard shortcuts: `G` (generate), `F` (fit), `R` (reset zoom), `C` (cluster
  gallery toggle), `Esc` (close modal) — guarded against INPUT/SELECT focus.

### Removed

- `/original` route and the `imageplot_original.html` page.
- `renderInstallAdvice()` from the clip projection module.
