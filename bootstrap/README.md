# Internal startup helpers

This folder keeps the non-public startup scripts used by the two root launchers:

- `Start Windows.bat`
- `Start macOS.command`

The root files are the only entry points meant for end users.

## Layout

- `windows/` contains the Windows startup flow, Python bootstrap, dependency helpers and maintenance shortcuts.
- `macos/` contains the macOS startup flow, Python bootstrap, dependency helpers and maintenance shortcuts.

These helpers are intentionally grouped away from the project root so the first
launch is easier to understand for non-technical users.

## Optional model installers (double-click, no terminal)

Run after the first normal start, then restart the app:

- `install_optional_multimodal.{bat,command}` — Qwen3-VL Embedding (2B) and
  Jina v5 Omni (Small). Large, GPU-only; Jina is CC BY-NC 4.0 (non-commercial).
- `install_optional_imagebind.{bat,command}` — ImageBind (Huge).

Each script installs into the existing `.venv` and is safe to re-run. The main
app works without any of them; the matching models stay greyed out in the model
selector until their packages are present.
