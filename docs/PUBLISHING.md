# Publishing / updating the GitHub repository

Target repo: <https://github.com/paolodegasperis/ImageCluster>

The local working folder is **not** a git repository, and the GitHub repo already
has history. The safe way to publish a new version is to work from a fresh clone,
replace its contents with the new version, and commit. This preserves history and
records file deletions (e.g. the removed `imageplot_original.html`).

> Prerequisites: `git` (and optionally the `gh` CLI) installed and authenticated
> to GitHub (`gh auth login`, or an SSH key / credential helper).

## One-time vs each release
`.gitignore` already excludes work artifacts: `.venv/`, `__pycache__/`,
`.pytest_cache/`, `.claude/`, `*.zip`, and the runtime contents of `output/` and
`img/` (only the empty folder structure + `.gitkeep` is tracked).

The share zip (`ImageCluster-V5.5-share.zip`) is intentionally **git-ignored** —
publish it as a **Release asset**, not as a tracked file.

## Procedure (Windows PowerShell)

```powershell
# 0. Pick a clean working location OUTSIDE the project folder
cd $env:TEMP

# 1. Clone the existing repo
git clone https://github.com/paolodegasperis/ImageCluster.git
cd ImageCluster

# 2. Clear tracked files (keeps .git) so deletions are captured in the diff.
#    If the repo contains files you want to keep that are NOT in the new version
#    (e.g. LICENSE, .github/ workflows), note them and re-add them in step 4.
git rm -r --quiet .

# 3. Copy the new version in, excluding work/runtime folders.
#    Replace <WORKDIR> with the real path:
#    "C:\Users\paolo\Documents\Web\Progetti vari\ImageCluster\Claude-version"
robocopy "<WORKDIR>" . /E /XD .git .venv .claude .pytest_cache __pycache__ UI-check img output node_modules /XF *.zip *.pyc local_settings.json

# 4. Recreate the empty runtime structure so it is tracked
New-Item -ItemType Directory -Force img,output\embeddings,output\projections,output\search,output\logs | Out-Null
"" | Set-Content img\.gitkeep, output\embeddings\.gitkeep, output\projections\.gitkeep, output\search\.gitkeep, output\logs\.gitkeep

# 5. Review, commit, push
git add -A
git status            # sanity-check the diff (adds, modifications, deletions)
git commit -m "V-5.5: rendering performance, expanded model registry (EVA-CLIP, MetaCLIP 2 H/14, Qwen3-VL, Jina v5 Omni), optional installers, docs"
git push origin main  # or 'master' — check the repo's default branch
```

> `git rm -r .` is the deliberate step that lets the commit reflect removed files.
> If you would rather not clear everything, skip step 2 and instead `git rm` the
> specific obsolete paths (e.g. `git rm app/imageplot_original.html`) after copying.

## Tag + Release (recommended)
```powershell
git tag v5.5
git push origin v5.5

# Attach the downloadable zip to a GitHub Release (requires the gh CLI):
gh release create v5.5 "<WORKDIR>\ImageCluster-V5.5-share.zip" `
  --title "ImageCluster V5.5" `
  --notes-file "<WORKDIR>\CHANGELOG.md"
```

## Notes
- **`UI-check/`** (the React design prototype) is excluded above as a work file.
  If you want it in the repo for design reference, drop it from the `/XD` list.
- The default branch may be `main` or `master`; confirm with
  `git remote show origin` and push to the correct one.
- No secrets are tracked: the HF token lives in `output/local_settings.json`,
  which is git-ignored.
