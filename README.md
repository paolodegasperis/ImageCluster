https://doi.org/10.5281/zenodo.20691432
# ImageCluster

ImageCluster is a local web application for exploring image collections through AI embedding models. It projects thousands of images onto an interactive 2D map, lets you search them with natural language, and groups them into visual clusters — all running on your own machine, without sending data anywhere.

---

## What it does

- **2D projection** — generates a visual map of your images based on their visual and semantic similarity. Images that look or feel alike end up close together.
- **Semantic search** — type a description in plain language (`gold background Madonna and Child`, `woman in red dress`) and the app finds the matching images using the same AI model that produced the projection.
- **Cluster gallery** — automatically groups images into clusters and shows them as a browsable gallery with color-coded labels.
- **Multiple embedding models** — choose from a curated set of vision-language models (OpenCLIP, SigLIP, CLIP, DINOv2, MetaCLIP, EVA-CLIP and more), each with different strengths.
- **Fully local** — no cloud account, no API key required for basic use. Your images never leave your computer.

---

## First launch

### Windows

1. Put your images in the `img` folder inside the project.
2. Double-click **`Start Windows.bat`**.
3. On the first run, a small window will ask whether to use **CPU or CUDA**. Choose **CPU** unless you have an NVIDIA GPU and know it is set up correctly.
4. The launcher installs everything automatically. This may take a few minutes on the first run (it downloads Python packages).
5. A browser tab opens at `http://127.0.0.1:8765`. If it does not open automatically, copy that address into your browser.

> If Windows shows a security warning ("Windows protected your PC"), click **More info → Run anyway**. The file is safe — Windows warns about any unsigned script.

### macOS

1. Put your images in the `img` folder inside the project.
2. Double-click **`Start macOS.command`**.
3. If macOS blocks the file because it was downloaded from the internet, right-click it → **Open** → **Open** in the dialog. You only need to do this once.
4. The launcher installs everything automatically in the background. First run may take a few minutes.
5. A browser tab opens at `http://127.0.0.1:8765`.

---

## Guided workflow

Once the app is open in the browser:

1. Go to **`/clip`** (the app opens there directly).
2. In the left panel, click **Scan img folder** to count your images.
3. Choose an **embedding model**. The default (`OpenCLIP ViT-B/32 · LAION-2B`) works well for most collections.
4. Choose a **projection method**. UMAP is the recommended default; t-SNE is an alternative for smaller collections.
5. Optionally enable **K-Means clustering** to group images automatically.
6. Click **Generate** and wait. The first run downloads the model weights (several hundred MB); subsequent runs reuse the cache.
7. The 2D map appears. Use scroll to zoom, drag to pan, and hover to see image previews.
8. Type a query in the **Search** bar to highlight matching images on the map and see ranked results.
9. Switch to the **Cluster gallery** tab to browse grouped images.

---

## Optional models

A few models need extra packages that are not installed by default because they are large and require a GPU. The main app works without them. After the first normal start, double-click the matching file in the project root:

| File | Models it enables |
|---|---|
| `Install additional models (Windows).bat` | Qwen3-VL Embedding (2B), Jina v5 Omni (Small) |
| `Install ImageBind (Windows).bat` | ImageBind (Huge) |
| `Install additional models (macOS).command` | Qwen3-VL Embedding (2B), Jina v5 Omni (Small) |
| `Install ImageBind (macOS).command` | ImageBind (Huge) |

After running an installer, restart the app. The newly installed models appear enabled in the model selector. Until installed they are listed but greyed out, with a note explaining the missing dependency.

> **Jina v5 Omni is licensed CC BY-NC 4.0 — non-commercial use only.**

---

## Available embedding models

| Model | Text search | Hardware |
|---|:---:|---|
| OpenCLIP · ViT-B/32 · LAION-2B | ✓ | CPU / GPU |
| CLIP · OpenAI ViT-B/32 | ✓ | CPU / GPU |
| SigLIP · Base patch16 224 | ✓ | CPU / GPU |
| SigLIP 2 · Base patch16 224 | ✓ | CPU / GPU |
| MobileCLIP · B | ✓ | CPU / GPU |
| DINOv2 · Base | — | CPU / GPU |
| Nomic Embed Vision · v1.5 | ✓ | CPU / GPU |
| EVA-CLIP · L/14 | ✓ | GPU recommended |
| EVA-CLIP · bigE/14 | ✓ | Large GPU |
| MetaCLIP · B/32 | ✓ | CPU / GPU |
| MetaCLIP · H/14 | ✓ | GPU recommended |
| MetaCLIP 2 · Worldwide H/14 | ✓ | Large GPU |
| ImageBind · Huge *(optional)* | ✓ | GPU recommended |
| Qwen3-VL Embedding · 2B *(optional)* | ✓ | Large GPU |
| Jina v5 Omni · Small *(optional, NC)* | ✓ | GPU recommended |

Models marked *optional* require a separate installer (see above). Models shown as greyed-out in the UI are registered but not yet available on your system.

---

## Pages and endpoints

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8765/` | Dashboard — system status, quick actions |
| `http://127.0.0.1:8765/clip` | Main guided workflow |
| `http://127.0.0.1:8765/models` | Model guide, capability cards, HF token settings, glossary |
| `http://127.0.0.1:8765/api/status` | JSON dependency status |
| `http://127.0.0.1:8765/api/models` | JSON model registry |

---

## Prerequisites

- Python 3.10 or newer (the Windows launcher can install it automatically via `winget` if missing).
- Internet connection on first launch to download Python packages and model weights.
- At least 4 GB of free disk space for the default models; more for optional large models.
- A GPU is not required for the default models but speeds up embedding generation significantly.

---

## Manual setup 

```bash
python -m venv .venv
```

Windows:
```powershell
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

macOS / Linux:
```bash
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch torchvision
.venv/bin/python -m pip install -r requirements-core.txt
```

Then start the app:
```bash
python run.py
```

---

## Runtime folders

| Folder | Contents |
|---|---|
| `img/` | Place your image collection here |
| `output/embeddings/` | Cached embedding vectors (reused between runs) |
| `output/projections/` | Generated projection TSV files |
| `output/search/` | Exported search results |
| `output/logs/` | Debug logs |

These folders are tracked as empty (`.gitkeep`) and excluded from git content.

---

## Hugging Face token

Some models require accepting a license on Hugging Face before download. If a model fails to download, generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it in the **Models** page (`/models → Token`). The token is saved locally in `output/local_settings.json` and never committed.

---

## Version

V-5.5 · June 2026
This software was primarily developed using generative AI. It is intended for educational and research purposes only. It is not recommended for commercial or production use.
