https://doi.org/10.5281/zenodo.20691432


# ImageCluster

ImageCluster is a local web application for exploring image collections through AI embedding models. It projects thousands of images onto an interactive 2D map, lets you search them with natural language, and groups them into visual clusters — all running on your own machine, without sending data anywhere.

---

## What it does

- **2D projection** — generates a visual map of your images based on their visual and semantic similarity. Images that look or feel alike end up close together.
- **Semantic search** — type a description in plain language (`gold background Madonna and Child`, `woman in red dress`) and the app finds the matching images using the same AI model that produced the projection.
- **Cluster gallery** — automatically groups images into clusters and shows them as a browsable gallery with color-coded labels.
- **Grid mosaic** — the same projection laid out as a dense grid where every image gets one equal cell. A scatter of thumbnails hides its own dense regions; the mosaic keeps them readable while preserving which images sit near which.
- **Nearest-neighbor graph** — a separate view that links every image to its closest neighbours in the *full embedding space*, where the 2D projection's distortions do not apply. Useful for finding hubs, outliers and groups the map flattens together.
- **Find similar images** - pick any image and rank the whole collection by likeness to it. No model is loaded, so this works even with the image-only models that cannot search by text.
- **Duplicate detection** - find images that are effectively the same picture: re-saves, crops, colour variants.
- **Outlier ranking** - find the images least like anything else, which is usually where cataloguing errors hide.
- **Compare two images** - pick any two and see exactly how they relate: the similarity, where each ranks in the other's neighbour list, and the neighbours they share.
- **Multiple embedding models** — choose from a curated set of vision-language models (OpenCLIP, SigLIP, CLIP, DINOv2, MetaCLIP, EVA-CLIP and more), each with different strengths.
- **Fully local** — no cloud account, no API key required for basic use. Your images never leave your computer.

---

## First launch (no technical knowledge required)

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

## What each function is for

If any of this is unfamiliar, open **What you can find out** — the first card on
the dashboard, or `/guide` directly — before anything else. It explains every
feature in plain language: the question it answers, what comes back, and how it
can mislead you. Nothing here is magic, and several of the readings are easy to
misinterpret; the page says exactly where.

## Guided workflow (step by step)

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

### Analysing a projection

The bottom dock has four tabs. **Cluster gallery** browses each cluster, most representative image first,
with a strip at the top showing every cluster's *medoid* - the image closest to the cluster's centre - and
a bar for its cohesion, so you can see at a glance which clusters are one thing and which are several
merged. **Search results** holds the last text query. **Similar images** ranks the collection by likeness
to one image you picked: double-click any image on the plot, or press **Similar** on any card.
**Analysis** holds two whole-collection checks:

- **Duplicates** - groups of images that are effectively the same picture. The threshold defaults to 0.98,
  which catches re-saves and light crops; lower it to around 0.92 for colour variants. A group marked
  *chained* was joined through intermediaries rather than every member matching every other one, which is
  the signal to look before deleting anything.
- **Outliers** - the images least like anything else, judged by their k-th nearest neighbour so that a pair
  of duplicates does not read as typical.

All four read the embeddings the projection already saved. Nothing is re-encoded and no model is loaded,
which is why they answer instantly and why they work with every model - including DINOv2 and Visual
Features, which have no text search at all.

### What the neighbor graph tells you

Beyond drawing the links, the graph reads three things off them:

- **Communities** - groups whose members link to each other far more than outward. Unlike connected
  components, which only separate images with no path between them, communities divide a graph that is
  fully connected, which is the usual case. This is the graph's own division of the collection, reached
  without the 2D projection and without K-Means; where it disagrees with the cluster colours, one of the
  two has distorted something. The *modularity* figure says how clear-cut the split is: below about 0.3
  the collection is fairly homogeneous, above 0.5 the groups are strongly distinct.
- **Bridges** - links whose removal would split the graph. Each is the single connection holding two
  groups together, often the work that belongs to both worlds. Drawn heavier and in amber.
- **Load-bearing images** - the images whose removal would split the graph. Ringed in amber.

All three are listed with thumbnails in the **Structure** tab of the right-hand panel.

Colouring edges by **projection stress** compares each link's length on screen with its real similarity.
The warm links are where the 2D map separated images the model considers close - that is, exactly where
the scatter plot is misleading you.

### Other ways to read the same projection

Once a projection exists you can look at it in two more ways, without recomputing anything:

- **Grid mosaic** — in the plot toolbar, click **grid** (or press `M` to cycle scatter → points → mosaic).
  Every image gets its own equal cell, so nothing overlaps. Two assignment methods are available:
  *Optimal* is exact and used automatically up to 2000 images; *Recursive split* is much faster and
  handles large collections. You can pin the number of columns, or leave it to follow the window shape.
- **Nearest-neighbor graph** — click **graph** in the toolbar, or open **Neighbor graph** from the
  dashboard. Choose how many neighbours per image (k), optionally drop weak links with a similarity
  threshold or keep only reciprocal ones, then build. Click any node to see its nearest neighbours
  ranked by similarity. The layout can follow the projection's coordinates or be relaxed with a
  force-directed pass.

### Taking results away

Every view here is a working instrument, and a result that cannot leave the screen cannot be worked on
afterwards. Three surfaces now export what they are showing, in two formats:

| Where | What it exports |
|---|---|
| Neighbor graph → **Structure** → a community | Every image in that community |
| Neighbor graph → **Selected** → an image | That image and its nearest neighbours, in order |
| Embedding projection → **Similar images** | The query and its matches, ranked |

**PDF** produces a contact sheet in which the images are large enough to actually look at — no image is
embedded smaller than 500px on its long side, two per row on A4, each with its filename and what the view
knows about it (similarity, rank, cluster, number of links). Images whose original is smaller than that are
included at their own size and marked, rather than enlarged: blowing them up would invent detail and hide
that the source is small.

**CSV** gives the same result as numbers, for whatever comes after looking at it. The columns differ between
surfaces, but the shape is constant: an ordering column, then filename and path, then what the view knows,
then the plot coordinates. In the two neighbour exports the image itself is row 0 with no similarity — a
neighbour list without the image it belongs to cannot be read back months later. Files use `;` and a UTF-8
BOM, so they open correctly in an Italian Excel.

The PDFs are built on the server, where the originals are on disk at full resolution. The browser can only
reach images through the thumbnail endpoint, which stops at 512px.

### Making the results panel bigger

The panel below the plot is where results are read. Drag its top edge to trade plot height for panel height;
double-click to reset, or use the arrow keys once it has focus. The chosen height is remembered. The plot
always keeps at least a third of the space.

### Asking the model questions (optional)

A CLIP-family model does not only turn images into numbers: it puts images and words in the *same*
space. Search uses one sentence of that shared vocabulary — "which images are nearest this phrase".
Three optional features use the rest of it. All of them are **off until you turn them on**, and none
of them changes how a projection is generated by default.

- **Named axes** — in the reducer list, next to UMAP and t-SNE, choose **Named axes**. Give a phrase
  for each end of the horizontal axis ("a portrait of a person" ↔ "a landscape") and each end of the
  vertical one, then press **Preview**: the collection is rearranged on two axes you chose, with your
  names drawn on the plot. Unlike UMAP and t-SNE, the position of an image now means something you
  can state. Three presets (subject, mood, setting) are there to start from.
- **Colour by a concept** — in the **Explore** panel, type a phrase such as "a ruin" and every image
  is shaded by how much of that concept it carries. This shows the *distribution* rather than a top
  thirty: the concept may turn out to be spread across two clusters, or concentrated in the corner of
  one. One click restores the cluster colours.
- **Lenses** (`/lens`, or **Lenses** on the dashboard) — three questions the map cannot answer:
  *what separates two groups, in words*; *concept arithmetic* (an image, minus an idea, plus
  another); and *where two models disagree* — generate the same folder with a CLIP-family model and
  with DINOv2, and the images the two place differently are the ones whose subject and form part
  company. Neither model can point at those alone.

The first two need a model that can read text, so the controls appear only for CLIP-family
projections and stay hidden otherwise.

### Readability controls

Thumbnails on a light background can wash out, and in dense areas they hide each other. Open the
**Explore** panel on the right rail and use **Thumbnail opacity** (5–100%) together with
**Thumbnail size**. At the low end the images become faint ghosts, which is the quickest way to read
the cluster colours underneath a crowded region. The graph view has its own opacity controls for
nodes and edges.

Every control on the projection and graph pages carries a tooltip explaining what it does and when
to reach for it — hover to read it. The graph view also has a **How to read this** button (or press
`?`) that opens a full explanation of what the graph shows, how it is built, and how each setting
changes the picture.

---

## Optional models (advanced)

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
| `http://127.0.0.1:8765/graph` | Nearest-neighbor graph — communities, bridges, projection stress |
| `http://127.0.0.1:8765/guide` | What each function does — every feature in plain language |
| `POST /api/exports/sheet` | A result as a PDF contact sheet (needs the optional `fpdf2` package) |
| `http://127.0.0.1:8765/lens` | Lenses — what separates two groups, concept arithmetic, model disagreement |
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

## Manual setup (technical users)

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

v6.6.0 · September 2026
