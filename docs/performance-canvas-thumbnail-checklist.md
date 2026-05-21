# Performance Canvas and Thumbnail Checklist

Use this checklist after changes to `clip_projection.js` or the projection canvas UI.

## Dataset sizes

- [ ] Small dataset under 500 images opens in thumbnail mode and loads only visible thumbnails.
- [ ] Medium dataset around 1,000 images defaults or recommends points mode.
- [ ] Large dataset above 5,000 images uses points mode and shows a performance notice.
- [ ] A repeated/synthetic large dataset can be panned and zoomed without requesting every image immediately.

## Rendering

- [ ] Initial projection load computes bounds once and then uses cached normalized coordinates.
- [ ] Panning draws only visible or near-viewport items.
- [ ] Zooming in reveals thumbnails when the adaptive threshold is reached.
- [ ] Zooming out returns to points/placeholder rendering for large datasets.
- [ ] Search-highlighted off-screen items do not trigger thumbnail drawing work.
- [ ] Reset View and Fit to Data remain separate and predictable.
- [ ] PNG export captures the current visible canvas state.

## Thumbnail cache

- [ ] Initial page load does not load every thumbnail.
- [ ] Newly visible thumbnails load progressively while panning.
- [ ] Failed thumbnail loads do not retry in a tight loop.
- [ ] Cache status shows loaded, loading, queued and failed counts.
- [ ] Browser memory remains bounded during repeated pan/zoom sessions.
- [ ] Previewed images remain visible while the preview/modal is open.

## Interaction

- [ ] Hover preview remains responsive on medium and large datasets.
- [ ] Mouse move does not trigger redraws when the hover target is unchanged.
- [ ] Semantic search result thumbnails load correctly and stay resizable.
- [ ] Show-only-search-results mode updates culling and hover behavior.
- [ ] Colored thumbnail outline toggle still affects only the thumbnail outline.
- [ ] Force thumbnails can be enabled by advanced users and persists during the session.

## Debug signals

- [ ] Render debug shows total items, visible items, drawn thumbnails, drawn points, render time and last render reason.
- [ ] Thumbnail cache status updates when visible thumbnails are queued, loaded, failed or evicted.
- [ ] Adaptive rendering notice explains points mode, lazy thumbnail rendering, or zoom threshold behavior.
