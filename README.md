# ImageCluster V53 Final

Frozen clean release of the current working ImageCluster / ImagePlot-CLIP app.

This folder is intended to be self-contained for development, maintenance and local execution. It excludes local virtual environments, previous builds, generated outputs, logs, caches and UI work files.

## Prerequisites

- Python 3.10 or newer.
- Internet connection on first launch, because the launcher installs Python packages.
- Windows users can choose CPU or CUDA on first launch. Use CPU if unsure.
- macOS users can use the macOS launcher; CUDA is not used on macOS.

## Recommended Start

### Windows

Double-click:

```text
Start Windows.bat
```

The launcher checks Python, asks for CPU/CUDA on first start, prepares `.venv`, installs PyTorch and required packages, then starts the web app.

### macOS

Double-click:

```text
Start macOS.command
```

If macOS blocks the file because it was downloaded from the internet, allow it from System Settings or run it once from Terminal.

## Manual Setup

Create and activate a virtual environment:

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

## Run

```bash
python run.py
```

The app starts on:

```text
http://127.0.0.1:8765
```

Embedding projection UI:

```text
http://127.0.0.1:8765/clip
```

Models and settings:

```text
http://127.0.0.1:8765/models
```

## Optional Models

A few embedding models need extra packages that are **not installed by default**
(they are large and/or GPU-only). The main app works without them. To enable a
group, **double-click** the matching installer in the project root after the first
normal start — no terminal or pip knowledge required — then restart the app.

In the project root (next to `Start Windows.bat`):

```text
Install additional models (Windows).bat    → Qwen3-VL Embedding (2B), Jina v5 Omni (Small)
Install ImageBind (Windows).bat             → ImageBind (Huge)
Install additional models (macOS).command
Install ImageBind (macOS).command
```

(These are thin shortcuts to the scripts under `bootstrap/windows/` and
`bootstrap/macos/`, which can also be run directly.)

After running an installer, restart with `Start Windows.bat` / `Start macOS.command`.
The newly installed models then appear **enabled** in the model selector on `/clip`
(until installed they are shown but greyed out, with the reason in the model details
on `/models`). On macOS, if the `.command` is blocked, allow it in System Settings or
run it once from Finder with right-click → Open.

Notes:
- Multimodal models (Qwen3-VL, Jina v5 Omni) require a GPU and download large
  checkpoints on first use. **Jina v5 Omni is licensed CC BY-NC 4.0 (non-commercial).**
- ImageBind is large and may not work on every setup.

## Test And Validation

If `pytest` is installed:

```bash
python -m pytest tests
```

Syntax-only backend check:

```bash
python -m compileall backend app launcher.py run.py project_paths.py
```

JavaScript syntax check, if Node.js is available:

```bash
node --check app/js/clip_projection.js
node --check app/js/redesign_ui.js
node --check app/js/models_tokens.js
node --check app/js/dashboard.js
```

## Build / Packaging

Windows installer/app build script:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\build\build_installer_windows.ps1
```

macOS packaging script:

```bash
./tools/build/build_installer_macos.command
```

Build artifacts are intentionally not included in this clean folder.

## Runtime Folders

- `img/`: place image collections here.
- `output/embeddings/`: generated embedding caches.
- `output/projections/`: generated projection files.
- `output/search/`: generated semantic search exports.
- `output/logs/`: generated debug reports and logs.

These runtime folders are included empty and are ignored by Git except for `.gitkeep` placeholders.

## Environment Variables

No `.env` file is required for normal local use.

Optional Hugging Face token configuration is managed from the app UI and stored locally in `output/local_settings.json`, which should not be committed.

## Frozen Version

- Version label: V53-final
- Source state: current verified UI and launcher flow
- Date: 2026-06-13
