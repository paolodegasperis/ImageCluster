# Phase 1 UX Test Checklist

Use this checklist after frontend changes to the Embedding Projection workflow.

## Page Load

- Open `http://127.0.0.1:8765/clip`.
- Confirm the page shows five workflow steps and a Session Summary panel.
- Confirm `/api/status` errors, if any, are shown as user-facing messages.

## Image Folder

- Click `Scan img folder`.
- Confirm the image count appears in Step 1 and Session Summary.
- Confirm supported formats are shown.
- Confirm unsupported files, if present, are reported as ignored.
- Confirm `Generate projection graph` remains disabled when zero images are found.

## Model Selection

- Change the embedding model.
- Confirm model badges update.
- Confirm model explanation updates.
- Confirm semantic-search capability text updates.
- Enable `Show planned roadmap models`.
- Select a planned model and confirm generation is disabled with a readable reason.

## Projection Settings

- Confirm UMAP is available and marked as recommended.
- Confirm t-SNE remains available and is labeled as slower/advanced.
- Open advanced projection settings and confirm batch size/cache/reducer parameters remain available.
- Open advanced clustering options and confirm K-Means controls remain available.

## Generation

- Confirm generation summary includes image count, model, reducer, clustering and semantic-search state.
- Start a projection with an available model.
- Confirm job progress/status updates.
- Confirm cancel remains available while the job runs.
- Confirm completion shows output path and enables explore/export controls.

## Explore

- Confirm `Reset view` and `Fit to data` are separate controls.
- Zoom and pan, then confirm both controls restore a predictable centered view.
- Confirm thumbnail/points mode still works.
- Confirm thumbnail size still works.
- Confirm colored outline toggle still works.
- Confirm PNG and TSV export are disabled before a projection and enabled after one.

## Semantic Search

- Select an image-only model such as DINOv2 and confirm search is visibly disabled with a reason.
- Select a text-search model and confirm examples are visible.
- Click an example query and confirm it fills the search box.
- Run a search after embeddings exist.
- Confirm ranked result cards show image, rank, filename and score.
- Confirm search result thumbnail size works.
- Confirm double-click opens preview and Escape closes it.
- Confirm result highlighting and show-only filters still work.
- Confirm `Clear search filter` clears search/filter state without resetting the projection view.

## Error Handling

- Attempt search before embeddings exist and confirm the UI shows a readable message.
- Confirm traceback/debug links remain separate from the main user-facing message.
- Confirm clustering failure, if forced, does not prevent projection output from being saved.
