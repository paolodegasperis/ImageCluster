# Planned models — integration feasibility report

**Scope:** the 8 models currently flagged `status="planned"` in
`backend/encoders/registry.py`. This report explains how integration works in
this codebase, assesses each planned model, lists the concrete work needed, and
proposes a promotion path to `experimental` and then `stable`.

> Data source: `backend/encoders/registry.py` (registry), `backend/encoders/base.py`
> (provider routing), `backend/encoders/openclip_encoder.py`,
> `backend/encoders/transformers_encoder.py`, `backend/encoders/imagebind_encoder.py`.
> Checkpoint identifiers marked “verify” must be confirmed on Hugging Face / the
> OpenCLIP build before wiring — they are best-known candidates, not guarantees.

---

## 1. How model integration actually works here

A model is a `ModelSpec` entry routed to an encoder by its `provider`
(`backend/encoders/base.py` → `encode_images` / `encode_texts`):

| provider | encoder | loads via | reusable for new models? |
|----------|---------|-----------|--------------------------|
| `openclip`, `openclip_hf_hub` | `openclip_encoder.py` | `open_clip.create_model_and_transforms(model_id, pretrained=…)` or `hf-hub:` | **Yes — generic.** Any OpenCLIP-known arch/pretrained tag, or any `hf-hub:` OpenCLIP checkpoint. |
| `transformers_clip`, `transformers_image_features`, `transformers_vision_pool`, `transformers_metaclip2`, `nomic_transformers` | `transformers_encoder.py` | `AutoModel` + `AutoProcessor`, `get_image_features` / `get_text_features`, optional `trust_remote_code` | **Yes — generic** for any HF model exposing `get_image_features` (CLIP-like) or poolable hidden states. |
| `imagebind` | `imagebind_encoder.py` | bespoke (manual install) | No — bespoke path. |
| `planned` | — | not wired | n/a (placeholder) |

Because both main encoders are generic, **most integrations are “data” changes,
not new code**: pick a real checkpoint, set `provider` + `model_id`
(+ `pretrained` for OpenCLIP, + `trust_remote_code` if needed), set the
capability flags, then verify. A *new* encoder is only required when a model
needs a non-standard loading path (custom positional embeddings, region pooling,
non-PyPI package).

**Proof the paths already work:** the `experimental` tier already uses exactly
these slots — `metaclip2_worldwide_b32` (`transformers_metaclip2`),
`mobileclip2_{s2,b,s4}` and `mobileclip_b_openclip` (`openclip_hf_hub`),
`nomic_embed_vision_v1_5` (`nomic_transformers`, `trust_remote_code=True`).

### Integration checklist (per model)
1. **Locate + verify a checkpoint** (HF Hub id or OpenCLIP arch+pretrained tag); confirm licence/gating (HF token already supported via `config_store`).
2. **Choose the slot:** existing generic encoder, or write a new encoder module + register its provider in `base.py`.
3. **Fill the `ModelSpec`:** real `model_id`/`pretrained`, `requires`, `recommended_batch_size`, `hardware_tier`, `trust_remote_code`, and the capability flags (`supports_image_embedding`, `supports_text_embedding`, `supports_text_search`, `supports_projection`).
4. **Flip `status`** `planned` → `experimental`.
5. **Verify locally:** load + encode a small image batch (projection), and—if text-capable—a text query; confirm the embedding dim/shape and that `/api/jobs/clip-projection` + `/api/search/text` succeed.
6. **Extend tests:** `tests/test_model_registry.py`, `tests/test_encoder_capabilities.py`.

### Criteria to promote `experimental` → `stable`
- Checkpoint is public and reliably downloadable (no fragile remote code / gated surprises).
- Verified end-to-end on a reference machine: scan → projection → (search if text-capable) → cluster gallery.
- Resource needs documented and reasonable for the declared `hardware_tier` (a `large_gpu`-only model can be stable but should say so).
- Capability flags match observed behaviour; test suite green.
- Stable in this project means “verified + dependable”, not necessarily CPU-only.

---

## 2. Per-model feasibility

The 8 planned entries split into two structural groups:

- **Pure placeholders** (`provider="planned"`, empty `model_id`, all capability flags `False`): `metaclip2_worldwide_h14`, `metaclip2_2b_worldwide`, `hq_clip_b16`, `long_clip_b32`, `eva_clip_l14`, `eva_clip_bigE14`, `cloc_roadmap`, `laclip_roadmap`.

> **Resolved in v6.7.0.** Rows 4 and 5 below are settled and kept only as the
> record of how. `hq_clip_b16` is **wired and experimental**: the author's own
> OpenCLIP-compatible repo `zhixiangwei/vlm150m-hqclip-large-vitb16` (Apache-2.0,
> `open_clip_config.json` + `open_clip_model.safetensors`) needed no new loader,
> only the existing `hf-hub:` branch. `laclip_roadmap` is **deliberately not
> integrated**: a community mirror appeared in October 2025 and would have
> loaded, but with no licence, an empty model card and no way to prove the
> weights' identity, wiring it would have put an unverifiable dependency inside
> a tool that carries a DOI.

| # | Model (key) | Family / arch | Reusable slot? | Effort | Feasibility | Blocking work |
|---|-------------|---------------|----------------|--------|-------------|---------------|
| 1 | `eva_clip_l14` — EVA-CLIP · L/14 | EVA-CLIP (ViT-L/14) | **Yes — `openclip`** (open_clip ships EVA archs, e.g. `EVA02-L-14` + a `pretrained` tag) | Low | **High** | Confirm the exact OpenCLIP arch name + `pretrained` tag in the installed build; set `model_id`/`pretrained`; verify. |
| 2 | `eva_clip_bigE14` — EVA-CLIP · bigE/14 | EVA-CLIP (enormous, e.g. EVA02-E-14) | **Yes — `openclip`** | Low (wiring) / High (runtime) | **High (tech), gated by hardware** | Same as L/14; `large_gpu`, multi-GB weights. Keep experimental, document VRAM. |
| 3 | `metaclip2_worldwide_h14` — MetaCLIP 2 · Worldwide H/14 | MetaCLIP 2 (ViT-H/14) | **Yes — `transformers_metaclip2`** (B/32 sibling already wired) | Low–Med | **High** | Find the worldwide H/14 HF id (verify, e.g. `facebook/metaclip-2-worldwide-h14`); needs a recent Transformers with MetaCLIP 2 support; `large_gpu`. |
| 4 | `laclip_roadmap` — LACLIP | LaCLIP — standard CLIP arch, caption-rewritten training | **Likely — `transformers_clip`** or `openclip` | Low–Med | **Medium-High** | Locate a real LaCLIP checkpoint (verify availability/format on HF); if it’s a vanilla CLIP state-dict it drops into `transformers_clip`/`openclip`. |
| 5 | `hq_clip_b16` — HQ-CLIP · B/16 | HQ-CLIP (high-quality-data CLIP, ViT-B/16) | **Probably — `transformers_clip`** if standard CLIP arch; else `openclip` | Med | **Medium** | Verify checkpoint exists and its arch/format; choose slot accordingly; verify image+text encode. |
| 6 | `long_clip_b32` — Long-CLIP · B/32 | Long-CLIP (extended 248-token context, modified positional embeddings) | **No — needs custom loader** (or `trust_remote_code` path) | Med–High | **Medium-Low** | Long-CLIP changes the text positional embeddings; standard CLIP loaders truncate/break. Write/adapt a loader (or use the authors’ code via `trust_remote_code`) and validate long-query behaviour. |
| 7 | `metaclip2_2b_worldwide` — MetaCLIP 2 · 2B Worldwide | MetaCLIP 2 (2B params) | **Yes — `transformers_metaclip2`** | Low (wiring) / Very high (runtime) | **Low (practical)** | Same code path as #3, but 2B params → very large GPU/host RAM and slow; checkpoint availability to verify. Niche; likely stays experimental. |
| 8 | `cloc_roadmap` — CLOC | CLOC (contrastive *localized* / region-aware pretraining) | **No — design mismatch + uncertain checkpoint** | High | **Low** | CLOC targets region-level embeddings; for our global image embedding we’d need a pooled global feature, plus a verified public checkpoint + loader. Keep on roadmap until both exist. |

---

## 3. Recommended action plan

**Wave A — promote to `experimental` now (reuse existing encoders, wiring + verify only):**
- `eva_clip_l14` (provider `openclip`) — strongest quick win.
- `metaclip2_worldwide_h14` (provider `transformers_metaclip2`) — sibling path proven.
- ~~`laclip_roadmap`~~ — **closed in v6.7.0 as will-not-integrate**; the only available mirror has no verifiable provenance.
- `eva_clip_bigE14` (provider `openclip`) — wire it, but label clearly `large_gpu`.

**Wave B — `experimental` after small targeted work:**
- ~~`hq_clip_b16`~~ — **done in v6.7.0**, `openclip_hf_hub`; the checkpoint turned out to be a clean OpenCLIP repo, so no loader work was needed at all.
- `long_clip_b32` — requires a custom/`trust_remote_code` loader for the extended context; this is the one genuine code task.

**Wave C — keep `planned` (or experimental-with-warnings) until prerequisites exist:**
- `metaclip2_2b_worldwide` — code path is ready, but resource cost makes it niche; only wire when there’s a tested large-GPU target.
- `cloc_roadmap` — needs a public global-embedding checkpoint and a loader; architecturally not a drop-in. Keep as roadmap.

**Then `experimental` → `stable`** for any model that clears the Section-1 promotion
criteria on a reference machine. Realistic near-term stable candidates are the
ones that reuse a proven path and run within a normal GPU budget (EVA-CLIP L/14,
MetaCLIP 2 Worldwide H/14, LaCLIP, HQ-CLIP). The very large variants
(EVA bigE/14, MetaCLIP 2 2B) can be stable only with the GPU requirement
explicitly documented.

## 4. Notes / risks
- “Verify” checkpoint ids must be confirmed against Hugging Face and the *installed* OpenCLIP/Transformers versions before wiring; OpenCLIP arch names and HF repo ids drift between releases.
- Gated repos: the existing HF-token mechanism (`/api/config/local/hf-token` → `config_store`) covers authentication; no new auth code is needed.
- `trust_remote_code=True` (Long-CLIP, possibly others) executes downloaded code — acceptable for a local tool but worth flagging to users.
- No backend contract changes are required for Waves A/B beyond registry entries and (for Long-CLIP) one new loader path registered in `base.py`.

---

## 5. Survey of September 2026

The roadmap above was drawn up against the models of its own moment, and a year
of releases has passed since. This section is a fresh reading of what the
Hugging Face Hub actually offered on **13 September 2026**, taken from the Hub
API rather than from memory: repository existence, gating, licence and file
layout were each checked, and the two entries marked *verified* below were run
against the 165-image reference collection with the application's own encoders.

### Wired in v6.7.0

| Model | Repo | Licence | Evidence |
|---|---|---|---|
| HQ-CLIP · B/16 | `zhixiangwei/vlm150m-hqclip-large-vitb16` | Apache-2.0 | **verified** — grounded retrieval ranked the target 1st, 1st, 2nd of 165; ~47 ms/image on CPU |
| PE-Core · B/16 | `timm/PE-Core-B-16` | Apache-2.0 | **verified** — same three cases, same ranks; 1024-d embedding; ~112 ms/image on CPU |

### Drop-in candidates — one registry entry each, no new code

All carry `open_clip_config.json` + `open_clip_model.safetensors`, so the
existing `hf-hub:` branch loads them unchanged.

| Repo | Licence | What it would add |
|---|---|---|
| `timm/PE-Core-L-14-336` | Apache-2.0 | The L/14 sibling at 336 px — more detail, more cost |
| `imageomics/bioclip-2` | MIT | Biological taxonomy. Narrow, but decisive for a natural-history collection |
| `laion/CLIP-convnext_base_w-laion2B` | MIT | A convolutional CLIP rather than a ViT: a different inductive bias produces a genuinely different map of the same collection, which is the point of offering lenses at all |
| `laion/CLIP-ViT-L-14-laion2B`, `-H-14`, `-bigG-14` | MIT | Larger siblings of the current default |
| `timm/ViT-B-16-SigLIP2`, `timm/ViT-SO400M-14-SigLIP-384` | Apache-2.0 | SigLIP through OpenCLIP, alongside the Transformers path already wired |
| `apple/DFN2B-CLIP-ViT-B-16` | ⚠️ `apple-amlr` | Strong data-filtered CLIP, but the Apple ML Research licence must be read before use — it is not an ordinary open licence |

### Real work required

- **DINOv3** (`facebook/dinov3-vitb16-pretrain-lvd1689m`) — the natural successor
  to DINOv2 on the `transformers_vision_pool` path, and the installed
  Transformers 4.57.6 already supports the architecture. But the repository is
  **`gated: manual`**: it needs an access request approved by hand and a token.
  That contradicts the application's own promise of working without an account
  or an API key, and the licence is a custom Meta one rather than Apache. Keep
  `planned` with the reason recorded, rather than shipping a model most users
  cannot load.
- **TIPSv2** (`google/tipsv2-b14`, Apache-2.0, updated 2026-08-17) — conceptually
  the closest thing to what this application does: dense spatial features in the
  DINO manner *and* text alignment, from one model, so the map and the search
  would share a space. Not a drop-in: it needs `trust_remote_code=True`, a
  non-standard preprocessing (`ToTensor()` at 448×448 with **no** ImageNet
  normalisation, where every current encoder normalises), and its config
  declares `transformers_version 5.10.0.dev0` against the installed 4.57.6.
  Worth a scoped feasibility spike of its own.

### Ruled out

- `microsoft/colipri` — trending, but a **3D encoder for chest CT scans**. Nothing to do with 2D images.
- `jinaai/jina-clip-v2` — multilingual and good, but **CC BY-NC 4.0**, the same non-commercial constraint already documented for Jina v5 Omni.
- Multilingual search generally — the only open option is `laion/CLIP-ViT-B-32-xlm-roberta-base-laion5B` (MIT), unchanged since 2022 and without an `open_clip_config.json`. Now that the documentation exists in Italian the question will be asked; today the honest answer is SigLIP 2, already wired and trained multilingually.
- Art- or heritage-specific CLIP checkpoints — searched for, and nothing serious exists. Only hobby fine-tunes with single-digit download counts.
