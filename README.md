# ImageCluster

[![DOI](https://zenodo.org/badge/1245869180.svg)](https://doi.org/10.5281/zenodo.20344849)

ImageCluster is a local web application for analyzing digital image collections with visual embeddings, CLIP-like multimodal models, UMAP/t-SNE projection maps, optional clustering, and semantic text search.

The goal of ImageCluster is to give non-technical users a practical interface for exploring image collections and comparing different embedding models on those collections.

The project is inspired by Lev Manovich's ImagePlot software - https://github.com/culturevis/imageplot - and adapts that tradition of visual cultural analytics to current embedding-based image analysis. 

ImageCluster runs on your computer. Images are read from the local `img/` folder and output files are written under `output/`.

## What You Can Do

- Scan a local image folder.
- Generate image embeddings with OpenCLIP, CLIP, SigLIP, SigLIP 2, MetaCLIP, MobileCLIP and other supported models.
- Create interactive UMAP or t-SNE projection maps.
- Explore the projection as points or image thumbnails.
- Compare how different embedding models organize the same image collection.
- Run natural-language semantic search over image embeddings.
- View ranked search results with thumbnails and image detail previews.
- Optionally run K-Means clustering.
- Export PNG charts, projection TSV files, search TSV files, sessions and debug reports.

## Requirements

- Windows 11 or macOS.
- Python 3.10 or newer.
- Internet connection for the first installation and for first-time model downloads.
- Disk space for Python packages and model weights.

A GPU is optional. The CPU launchers are safer and easier to start with. CUDA can be faster on Windows machines with a compatible NVIDIA GPU.

## Project Folders

Put your images here:

```text
img/
```

Generated files are saved here:

```text
output/
```

Important output folders:

```text
output/embeddings/
output/projections/
output/search/
output/logs/
output/jobs/
output/sessions/
```

Do not delete `output/embeddings/` or `output/projections/` if you want to reuse cached embeddings and saved maps.

## Start the Web App on Windows 11

### Recommended CPU Start

1. Copy your images into `img/`.
2. Double-click:

```text
Start Windows 11 CPU.bat
```

3. Wait for the local environment to be prepared.
4. The browser should open automatically.
5. If it does not open, go to:

```text
http://127.0.0.1:8765/
```

Use this option first if you are not sure which launcher to choose.

### NVIDIA CUDA Start

Use this only if your Windows machine has a compatible NVIDIA GPU and CUDA-capable PyTorch can be installed:

```text
Start Windows 11 CUDA.bat
```

If CUDA startup fails, close the terminal and use the CPU launcher instead:

```text
Start Windows 11 CPU.bat
```

### Menu Launcher

You can also use:

```text
Start Windows 11.bat
```

This launcher provides a simpler startup menu.

## Start the Web App on macOS

1. Copy your images into `img/`.
2. Double-click:

```text
Start macOS.command
```

3. Wait for the local environment to be prepared.
4. The browser should open automatically.
5. If it does not open, go to:

```text
http://127.0.0.1:8765/
```

Optional macOS launchers:

```text
Start macOS Apple Silicon.command
Start macOS Intel.command
```

If macOS blocks the command file, open Terminal in the project folder and run:

```bash
chmod +x "Start macOS.command"
chmod +x "Start macOS Apple Silicon.command"
chmod +x "Start macOS Intel.command"
```

Then try the launcher again.

## If Python is missing

### Windows

Use:

```text
Install Python Windows 11.bat
```

Or install Python 3.10+ from [python.org](https://www.python.org/) and enable `Add python.exe to PATH` during installation.

### macOS

Use:

```text
Install Python macOS.command
```

Or install Python 3.10+ from [python.org](https://www.python.org/).

## First launch

The first launch can take several minutes.

The launcher creates a local Python virtual environment:

```text
.venv/
```

Then it installs the required packages, including:

- FastAPI and Uvicorn for the local web server;
- PyTorch;
- NumPy, pandas, Pillow and scikit-learn;
- UMAP and t-SNE support;
- OpenCLIP;
- Transformers and Hugging Face Hub;
- model support packages such as `timm` and `safetensors`.

Some embedding models download weights the first time you use them. This can take additional time.

## Main URLs

Dashboard:

```text
http://127.0.0.1:8765/
```

Embedding workflow:

```text
http://127.0.0.1:8765/clip
```

Models and Hugging Face token:

```text
http://127.0.0.1:8765/settings
```

System status:

```text
http://127.0.0.1:8765/api/status
```

The old ImagePlot + embeddings panel workspace has been removed in this release.

## Basic workflow

1. Put images in `img/`.
2. Start the app.
3. Open the dashboard at `http://127.0.0.1:8765/`.
4. Choose `Embedding guided workflow`.
5. Click `Scan img folder`.
6. Choose an embedding model.
7. Choose `UMAP` or `t-SNE`.
8. Optional: enable K-Means clustering in advanced settings.
9. Click `Generate projection graph`.
10. Explore the plot.
11. Use semantic search if the selected model supports text search.
12. Export PNG, TSV, search results or a session if needed.

## Interface layout

The embedding workflow uses a three-column desktop layout:

- Left rail: scan folder, choose model, choose projection method, generate graph.
- Center: dominant interactive projection plot and floating semantic search bar.
- Right rail: drawers for Explore, Search Results, Filters, Session, Cluster, Compare, Export and Info.

The plot remains visible while drawers are open.

## Supported image formats

The app scans common image formats:

```text
.jpg
.jpeg
.png
.webp
.bmp
.tif
.tiff
```

Unsupported files inside `img/` are ignored.

## Projection map controls

After a projection is loaded or generated, you can:

- zoom with the mouse wheel;
- pan by dragging the canvas;
- hover images or points for details;
- fit the map to the visible data;
- reset the view;
- switch between thumbnail mode and point mode;
- change thumbnail size;
- show or hide thumbnail outlines;
- load saved projections;
- export the projection as PNG;
- download the projection TSV.

For large collections, the app may switch to point mode for performance. You can force thumbnails from the Explore drawer.

## Semantic Search

Semantic search compares a natural-language text query against cached image embeddings in the same multimodal embedding space.

It is not filename search.

Example queries:

```text
Madonna con bambino
portrait with dark background
gold background Byzantine icon
landscape with ruins
Caravaggio-like chiaroscuro
blue abstract composition
female saint holding a book
decorative floral pattern
```

How to search:

1. Select a model that supports text search.
2. Generate or load image embeddings/projection data.
3. Type a query in the floating search bar.
4. Choose top-k results.
5. Optional: set a threshold.
6. Click `Search`.

After a successful search, the `Search Results` drawer opens automatically.

The results drawer includes:

- ranked result cards;
- image thumbnails;
- similarity scores;
- cluster labels when available;
- `Details` buttons for image preview;
- direct file links;
- thumbnail-size control;
- TSV export.

Search results can also be highlighted on the projection. Non-matching images can be dimmed or hidden from the Explore drawer.

If the selected model does not support text embeddings, the search button is disabled.

## Models

The model registry is capability-aware. A model can support some or all of:

- image embeddings;
- text embeddings;
- semantic search;
- image-text similarity;
- projection;
- optional local features.

Some models are image-only. They can create projections but cannot run semantic search.

Examples of image-only models:

```text
DINOv2
LLaVA-OneVision image-only
Qwen2.5-VL image-only
```

Recommended starting model:

```text
OpenCLIP ViT-B/32
```

Models and token settings are available at:

```text
http://127.0.0.1:8765/settings
```

## Hugging Face Token

Some models are downloaded from Hugging Face. Public downloads may work without a token, but a token can improve reliability and avoid rate limits.

To configure a local token:

1. Open:

```text
http://127.0.0.1:8765/settings
```

2. Paste the token into the Hugging Face token field.
3. Click `Save`.

The token is stored locally in:

```text
output/local_settings.json
```

Do not commit or share this file.

## Optional Clustering

K-Means clustering can be enabled in the advanced projection settings.

Options include:

- automatic K selection;
- fixed K;
- minimum and maximum K for automatic selection.

When clustering is enabled and successful, exported TSV files include cluster metadata:

```text
cluster
cluster_k
cluster_score
cluster_method
```

If clustering fails, the projection can still be saved without cluster metadata.

## Exported Files

Projection TSV files include image paths, coordinates, model metadata and optional cluster data.

Search exports include:

```text
rank
filename
relative_path
score
query
model_key
embedding_id
x
y
cluster
```

Typical output folders:

```text
output/projections/
output/search/
output/logs/
output/jobs/
output/sessions/
```

## Manual Start for Technical Users

You can start the app from a terminal.

CPU:

```bash
python launcher.py --torch cpu
```

Windows CUDA:

```bash
python launcher.py --torch cuda
```

macOS:

```bash
python launcher.py --torch macos
```

Install dependencies without starting the server:

```bash
python launcher.py --torch cpu --no-start
```

Start the already-prepared app directly:

```bash
python run.py
```

Default server:

```text
http://127.0.0.1:8765/
```

The default host and port are configured in:

```text
config.json
```

## API Endpoints

Common endpoints:

```text
GET  /api/status
GET  /api/diagnostics
GET  /api/models
GET  /api/model-guide
GET  /api/images/scan?image_dir=img
GET  /api/projections
GET  /api/jobs
GET  /api/jobs/{job_id}
POST /api/jobs/{job_id}/cancel
POST /api/jobs/clip-projection
GET  /api/search/indexes
POST /api/search/text
POST /api/search/rebuild-index
GET  /api/sessions
POST /api/sessions
POST /api/export/html
```

Diagnostics:

```text
http://127.0.0.1:8765/api/diagnostics
```

## Troubleshooting

### The App Does Not Start

Try these steps:

1. Use the CPU launcher.
2. Confirm Python 3.10+ is installed.
3. Run the repair script:

```text
Repair Runtime Folders Windows.bat
Repair Runtime Folders macOS.command
```

4. Start the app again.

### The Browser Does Not Open

Open this URL manually:

```text
http://127.0.0.1:8765/
```

### First Launch Is Slow

This is normal. The app may be creating `.venv/`, installing packages and downloading model weights.

### A Model Fails to Load

Possible causes:

- first-time model download failed;
- internet connection is unavailable;
- Hugging Face rate limit;
- missing optional dependency;
- not enough RAM or GPU memory;
- experimental model support.

Use `OpenCLIP ViT-B/32` as the safest default model.

### CUDA Fails on Windows

Use:

```text
Start Windows 11 CPU.bat
```

The CPU workflow is slower but more reliable.

### Semantic Search Is Disabled

The selected model probably does not support text embeddings.

Choose a text-search-capable model such as:

```text
OpenCLIP
CLIP
SigLIP
SigLIP 2
MetaCLIP
MetaCLIP 2 B/32
MobileCLIP
MobileCLIP2
```

### No Images Are Found

Check that images are inside:

```text
img/
```

Then click:

```text
Scan img folder
```

### Search Returns No Results

Check:

- a text-search model is selected;
- embeddings/projection have been generated or loaded;
- `top-k` is greater than zero;
- threshold is empty or not too strict;
- the image folder matches the embedding cache.

## Privacy Notes

The app is local-first. It reads your images from `img/` and does not intentionally upload the image collection.

However, model weights may be downloaded from external model hosts such as Hugging Face when a model is used for the first time.

## Version

Current version:

```text
0.5.3
```

The canonical version is stored in:

```text
VERSION
```

The running backend also reports the version at:

```text
http://127.0.0.1:8765/api/status
```
