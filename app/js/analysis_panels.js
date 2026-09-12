// Analysis panels: similar images, duplicates, outliers, cluster summaries.
//
// All four read the projection's stored embeddings, so none of them loads a model
// and none of them needs a text-capable one — they work for DINOv2 and Visual
// Features exactly as they do for CLIP.
//
// They deliberately share one card component and one interaction contract with the
// existing search results: click focuses the image on the plot, double-click opens
// the preview, and a small button asks "what else looks like this?". Learning one
// panel teaches all of them.
//
// Loaded after clip_projection.js, whose globals (state, apiJson, draw, ...) it uses.

const ANALYSIS_DEFAULTS = {
  similarTopK: 24,
  duplicateThreshold: 0.98,
  outlierK: 5,
  outlierTopK: 40,
};

// --------------------------------------------------------------------------
// Shared card rendering

function cardMarkup(item, options = {}) {
  const badges = (options.badges || []).filter(Boolean);
  const size = options.size || 160;
  return `
    <article class="search-card analysis-card${options.extraClass ? ` ${options.extraClass}` : ''}"
             data-path="${escapeHtml(item.relative_path)}" title="${escapeHtml(item.filename)}">
      <img src="${escapeHtml(thumbUrl(item, Math.max(160, size * 1.5)))}" alt="" loading="lazy" />
      ${badges.length ? `<div class="card-badges">${badges.map(b => `<span class="card-badge ${escapeHtml(b.tone || '')}">${escapeHtml(b.label)}</span>`).join('')}</div>` : ''}
      <div class="search-meta">${(options.meta || []).map(m => `<span>${escapeHtml(m)}</span>`).join('')}</div>
      <div class="search-title">${escapeHtml(item.filename)}</div>
      <div class="card-actions">
        <button class="btn btn-xs" type="button" data-similar-to="${escapeHtml(item.relative_path)}"
                title="Find the images most similar to this one">Similar</button>
        <button class="btn btn-xs" type="button" data-compare-with="${escapeHtml(item.relative_path)}"
                title="Add to the comparison tray">Compare</button>
        <a class="search-open-link" href="${escapeHtml(imageUrl(item))}" target="_blank" rel="noopener"
           title="Open the full-resolution image in a new tab">Open</a>
      </div>
    </article>`;
}

// One place wires every card, so a new panel gets the same behaviour for free.
function wireCards(container) {
  if (!container) return;
  container.querySelectorAll('.analysis-card').forEach(card => {
    card.addEventListener('click', event => {
      if (event.target.closest('a') || event.target.closest('button')) return;
      focusSearchResult(card.dataset.path);
    });
    card.addEventListener('dblclick', event => {
      if (event.target.closest('a') || event.target.closest('button')) return;
      const row = state.rows.find(item => item.relative_path === card.dataset.path);
      if (row) openImagePreview(row);
    });
  });
  container.querySelectorAll('[data-similar-to]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      runSimilarSearch(button.dataset.similarTo);
    });
  });
  container.querySelectorAll('[data-compare-with]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      addToComparison(button.dataset.compareWith);
    });
  });
}

function analysisReady(statusId) {
  if (state.resultPath) return true;
  setText(statusId, 'Load or generate a projection first.');
  return false;
}

async function analysisRequest(url, body) {
  return apiJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ projection_path: state.resultPath, ...body }),
  });
}

// --------------------------------------------------------------------------
// 1.1 Similar images

async function runSimilarSearch(relativePath) {
  if (!relativePath) return;
  if (!analysisReady('similarStatus')) return;
  selectDockPane('similar', true);
  setText('similarStatus', 'Searching...');
  setHTML('similarResults', '');
  try {
    const data = await analysisRequest('/api/analysis/similar', {
      relative_path: relativePath,
      top_k: Number($('similarTopK')?.value || ANALYSIS_DEFAULTS.similarTopK),
    });
    state.similarQuery = data.query;
    state.similarResults = data.results;
    // The query is part of the highlighted set so it is visible on the plot too.
    state.similarPaths = new Set([data.query.relative_path, ...data.results.map(r => r.relative_path)]);
    renderSimilar(data);
    draw();
  } catch (err) {
    setText('similarStatus', `Similar search failed: ${err.message}`);
    setHTML('similarResults', '');
  }
}

function renderSimilar(data) {
  setText('similarStatus', `${data.count} of ${data.searched} images${data.model_label ? ` · ${data.model_label}` : ''}`);
  const query = `
    <div class="similar-query" title="The image everything below is compared with">
      <img src="${escapeHtml(thumbUrl(data.query, 200))}" alt="" />
      <div class="sq-meta">
        <span class="sq-label">Query image</span>
        <strong>${escapeHtml(data.query.filename)}</strong>
        ${hasNumber(data.query.cluster) ? `<span class="sq-sub">cluster ${Number(data.query.cluster)}</span>` : ''}
      </div>
      <button class="btn btn-sm btn-ghost" type="button" id="similarClearBtn"
              title="Clear these results and remove the highlight from the plot">Clear</button>
    </div>`;

  const cards = data.results.map(item => cardMarkup(item, {
    size: state.searchThumbSize,
    meta: [`Rank ${item.rank}`, `Similarity ${item.score.toFixed(4)}`],
  })).join('');

  setHTML('similarResults', query + `<div class="gallery-grid analysis-grid">${cards || '<p class="help">No image cleared the threshold.</p>'}</div>`);
  $('similarResults').style.setProperty('--search-thumb-size', `${state.searchThumbSize}px`);
  wireCards($('similarResults'));
  $('similarClearBtn')?.addEventListener('click', () => clearSimilar());
}

function clearSimilar(redraw = true) {
  state.similarQuery = null;
  state.similarResults = [];
  state.similarPaths = new Set();
  setHTML('similarResults', '');
  if (has('similarStatus')) setText('similarStatus', 'Double-click an image on the plot, or use the Similar button on any card.');
  if (redraw) draw();
}

// --------------------------------------------------------------------------
// 1.2 and 1.3 share the Analysis pane

function showAnalysisTool(name) {
  document.querySelectorAll('[data-analysis-tool]').forEach(button => {
    button.classList.toggle('active', button.dataset.analysisTool === name);
  });
  document.querySelectorAll('[data-analysis-panel]').forEach(panel => {
    panel.hidden = panel.dataset.analysisPanel !== name;
  });
}

async function runDuplicateScan() {
  if (!analysisReady('duplicateStatus')) return;
  const threshold = Number($('duplicateThreshold')?.value || ANALYSIS_DEFAULTS.duplicateThreshold);
  setText('duplicateStatus', 'Scanning...');
  setHTML('duplicateResults', '');
  try {
    const data = await analysisRequest('/api/analysis/duplicates', { threshold });
    renderDuplicates(data);
  } catch (err) {
    setText('duplicateStatus', `Duplicate scan failed: ${err.message}`);
  }
}

function renderDuplicates(data) {
  if (!data.group_count) {
    setText('duplicateStatus', `No image pairs reached ${data.threshold} similarity across ${data.searched} images.`);
    setHTML('duplicateResults', '<div class="empty-state"><h3>Nothing looks duplicated</h3><p>Lower the threshold to catch looser matches — crops, re-saves and colour variants usually sit between 0.92 and 0.98.</p></div>');
    return;
  }

  const chainedNote = data.chained_group_count
    ? ` · ${data.chained_group_count} group(s) linked through intermediaries — review before deleting`
    : '';
  setText('duplicateStatus',
    `${data.group_count} group(s) · ${data.redundant_image_count} redundant image(s) of ${data.searched}${chainedNote}`);

  const html = data.groups.map((group, index) => {
    const cards = group.images.map(item => cardMarkup(item, {
      size: 140,
      extraClass: item.most_central ? 'is-keeper' : '',
      badges: [item.most_central ? { label: 'most central', tone: 'badge-ok' } : null],
      meta: [`Similarity to group ${item.score.toFixed(4)}`],
    })).join('');
    return `
      <section class="dup-group">
        <header class="dup-head">
          <span class="dup-index">Group ${index + 1}</span>
          <span class="dup-size">${group.size} images</span>
          <span class="dup-metric" title="Average similarity between all pairs in this group">avg ${group.mean_similarity.toFixed(4)}</span>
          <span class="dup-metric" title="The weakest pair in this group. Well below the threshold means the members were linked through intermediaries.">min ${group.min_similarity.toFixed(4)}</span>
          ${group.chained ? '<span class="dup-warn" title="Not every member matches every other one directly; they were joined through intermediaries. This can happen when the threshold is below a tight series’ internal similarity.">chained</span>' : ''}
        </header>
        <div class="gallery-grid analysis-grid dup-grid">${cards}</div>
      </section>`;
  }).join('');

  setHTML('duplicateResults', html);
  $('duplicateResults').style.setProperty('--search-thumb-size', '140px');
  wireCards($('duplicateResults'));
}

async function runOutlierScan() {
  if (!analysisReady('outlierStatus')) return;
  setText('outlierStatus', 'Scanning...');
  setHTML('outlierResults', '');
  try {
    const data = await analysisRequest('/api/analysis/outliers', {
      k: Number($('outlierK')?.value || ANALYSIS_DEFAULTS.outlierK),
      top_k: Number($('outlierTopK')?.value || ANALYSIS_DEFAULTS.outlierTopK),
    });
    renderOutliers(data);
  } catch (err) {
    setText('outlierStatus', `Outlier scan failed: ${err.message}`);
  }
}

function renderOutliers(data) {
  setText('outlierStatus',
    `${data.count} most isolated of ${data.searched} · typical image scores ${data.median_similarity.toFixed(3)} against its ${data.k}th neighbour`);

  const cards = data.results.map(item => {
    // Well under the median means genuinely unlike anything else, not just
    // slightly peripheral.
    const stark = item.score < data.median_similarity * 0.6;
    return cardMarkup(item, {
      size: 140,
      extraClass: stark ? 'is-outlier' : '',
      badges: [stark ? { label: 'unlike everything', tone: 'badge-warn' } : null],
      meta: [`#${item.rank}`, `${data.k}th neighbour ${item.score.toFixed(4)}`],
    });
  }).join('');

  setHTML('outlierResults', `<div class="gallery-grid analysis-grid">${cards}</div>`);
  $('outlierResults').style.setProperty('--search-thumb-size', '140px');
  wireCards($('outlierResults'));
}

// --------------------------------------------------------------------------
// 1.4 Cluster summaries, folded into the existing cluster gallery

async function loadClusterStats() {
  if (!state.resultPath || !clusterGroups().length) {
    state.clusterStats = null;
    renderClusterStrip();
    return;
  }
  try {
    state.clusterStats = await analysisRequest('/api/analysis/clusters', {});
  } catch (err) {
    state.clusterStats = null;
    setText('clusterSummary', `Cluster summaries unavailable: ${err.message}`);
  }
  renderClusterStrip();
  // The gallery's default order depends on the typicality scores we just loaded.
  if (typeof renderClusterCards === 'function') renderClusterCards();
}

function clusterStatFor(cluster) {
  if (!state.clusterStats) return null;
  return state.clusterStats.clusters.find(item => String(item.cluster) === String(cluster)) || null;
}

// Typicality per image, so the gallery can sort by it without a second request.
function typicalityMap() {
  const map = new Map();
  if (!state.clusterStats) return map;
  for (const cluster of state.clusterStats.clusters) {
    for (const member of cluster.members) map.set(member.relative_path, member);
  }
  return map;
}

function renderClusterStrip() {
  const strip = $('clusterStrip');
  if (!strip) return;
  const stats = state.clusterStats;
  if (!stats || !stats.clusters.length) {
    strip.hidden = true;
    setHTML('clusterStrip', '');
    return;
  }
  strip.hidden = false;
  setHTML('clusterStrip', stats.clusters.map(cluster => {
    const color = CLUSTER_PALETTE[(Number(cluster.cluster) - 1) % CLUSTER_PALETTE.length];
    const selected = String(cluster.cluster) === String(state.selectedCluster);
    // Cohesion runs roughly 0..1; the bar makes "one thing" vs "several things
    // merged" readable at a glance without reading the number.
    const pct = Math.max(0, Math.min(100, Math.round(cluster.cohesion * 100)));
    return `
      <button class="cluster-chip${selected ? ' selected' : ''}" type="button" data-cluster-chip="${escapeHtml(cluster.cluster)}"
              title="Cluster ${escapeHtml(cluster.cluster)}: ${cluster.size} images. Cohesion ${cluster.cohesion.toFixed(3)} is the average similarity to the cluster centre — high means the cluster is one thing, low means K-Means merged several. The thumbnail is the medoid: the member closest to that centre, i.e. the most representative image.">
        <img src="${escapeHtml(thumbUrl(cluster.medoid, 120))}" alt="" loading="lazy" />
        <span class="cc-body">
          <span class="cc-title"><i style="background:${color}"></i>Cluster ${escapeHtml(cluster.cluster)}</span>
          <span class="cc-sub mono">${cluster.size} imgs · coh ${cluster.cohesion.toFixed(2)}</span>
          <span class="cc-bar"><i style="width:${pct}%;background:${color}"></i></span>
        </span>
      </button>`;
  }).join(''));

  strip.querySelectorAll('[data-cluster-chip]').forEach(chip => {
    chip.addEventListener('click', () => {
      const value = chip.dataset.clusterChip;
      if (has('clusterSelect')) {
        $('clusterSelect').value = value;
        $('clusterSelect').dispatchEvent(new Event('change'));
      }
    });
  });
}

// --------------------------------------------------------------------------
// Wiring

document.querySelectorAll('[data-analysis-tool]').forEach(button => {
  button.addEventListener('click', () => showAnalysisTool(button.dataset.analysisTool));
});
$('duplicateScanBtn')?.addEventListener('click', runDuplicateScan);
$('outlierScanBtn')?.addEventListener('click', runOutlierScan);
$('similarTopK')?.addEventListener('change', () => {
  if (state.similarQuery) runSimilarSearch(state.similarQuery.relative_path);
});
$('highlightSimilarToggle')?.addEventListener('change', event => {
  state.highlightSimilar = event.target.checked;
  draw();
});
$('clusterSortSelect')?.addEventListener('change', event => {
  state.clusterSort = event.target.value;
  localStorage.setItem('imagecluster.clusterSort', state.clusterSort);
  if (typeof renderClusterCards === 'function') renderClusterCards();
});

// "Find similar" from the full-size preview, where the user is looking closest.
$('previewSimilarBtn')?.addEventListener('click', () => {
  const path = $('previewSimilarBtn').dataset.path;
  if (!path) return;
  closeImagePreview();
  runSimilarSearch(path);
});

// The preview modal is opened by clip_projection.js; record which image it holds
// so the Similar button knows its subject.
const originalOpenPreview = window.openImagePreview;
if (typeof originalOpenPreview === 'function') {
  window.openImagePreview = function openPreviewWithSimilar(result) {
    originalOpenPreview(result);
    const button = $('previewSimilarBtn');
    if (button) button.dataset.path = result?.relative_path || '';
  };
}

// Cluster summaries are cheap and always relevant once clusters exist, so they
// load with the projection rather than behind another click.
const originalRenderClusterGallery = window.renderClusterGallery;
if (typeof originalRenderClusterGallery === 'function') {
  window.renderClusterGallery = function renderClusterGalleryWithStats() {
    originalRenderClusterGallery();
    loadClusterStats();
  };
}

// Order the gallery by typicality (medoid first) and badge the medoid. The
// projection's own cluster_score is one silhouette value repeated on every row,
// so sorting by it is effectively sorting by filename.
const originalRenderClusterCards = window.renderClusterCards;
if (typeof originalRenderClusterCards === 'function') {
  window.renderClusterCards = function renderClusterCardsRanked() {
    const container = $('clusterGallery');
    if (!container || !state.selectedCluster) {
      originalRenderClusterCards();
      return;
    }
    const typicality = typicalityMap();
    if (!typicality.size || state.clusterSort === 'filename') {
      originalRenderClusterCards();
      return;
    }
    const rows = state.rows
      .filter(row => String(row.cluster) === String(state.selectedCluster))
      .map(row => ({ row, stat: typicality.get(row.relative_path) }))
      .sort((a, b) => (b.stat?.score ?? -Infinity) - (a.stat?.score ?? -Infinity));

    container.style.setProperty('--search-thumb-size', `${state.clusterThumbSize}px`);
    setHTML('clusterGallery', rows.map(({ row, stat }) => cardMarkup(
      { ...row, index: stat?.index ?? 0 },
      {
        size: state.clusterThumbSize,
        extraClass: stat?.is_medoid ? 'is-medoid' : '',
        badges: [stat?.is_medoid ? { label: 'most representative', tone: 'badge-ok' } : null],
        meta: [stat ? `Typicality ${stat.score.toFixed(4)}` : 'Typicality n/a'],
      },
    )).join(''));
    wireCards(container);
  };
}

if (has('clusterSortSelect')) $('clusterSortSelect').value = state.clusterSort;
if (has('highlightSimilarToggle')) $('highlightSimilarToggle').checked = state.highlightSimilar;
clearSimilar(false);
showAnalysisTool('duplicates');


// --------------------------------------------------------------------------
// Two-image comparison
//
// A similarity figure means little on its own - whether 0.72 is "close" depends
// on the model and the collection. The comparison puts three kinds of context
// around it: where each image ranks in the other's neighbour list, where the
// pair sits among all pairs, and which neighbours the two share.
//
// The picker is a tray rather than a mode: it stays hidden until an image is
// picked and disappears when cleared, so nothing is added to the workspace for
// a feature that is not in use.

const comparePicks = { a: null, b: null };

function rowFor(relativePath) {
  return state.rows.find(row => row.relative_path === relativePath) || null;
}

function addToComparison(relativePath) {
  const row = rowFor(relativePath);
  if (!row) return;
  if (comparePicks.a && comparePicks.a.relative_path === relativePath) return;
  if (comparePicks.b && comparePicks.b.relative_path === relativePath) return;

  // Fill A first, then B; a third pick replaces B so the tray never needs a reset.
  if (!comparePicks.a) comparePicks.a = row;
  else comparePicks.b = row;
  renderCompareTray();
}

function clearComparisonSlot(slot) {
  comparePicks[slot] = null;
  // Keep A filled when only B remains, so the tray reads left to right.
  if (!comparePicks.a && comparePicks.b) {
    comparePicks.a = comparePicks.b;
    comparePicks.b = null;
  }
  renderCompareTray();
}

function clearComparison() {
  comparePicks.a = null;
  comparePicks.b = null;
  renderCompareTray();
}

function renderCompareTray() {
  const tray = $('compareTray');
  if (!tray) return;
  const any = Boolean(comparePicks.a || comparePicks.b);
  tray.hidden = !any;
  if (!any) return;

  for (const slot of ['a', 'b']) {
    const button = tray.querySelector(`[data-compare-slot="${slot}"]`);
    const pick = comparePicks[slot];
    button.classList.toggle('filled', Boolean(pick));
    button.innerHTML = pick
      ? `<img src="${escapeHtml(thumbUrl(pick, 96))}" alt="" /><span class="ct-name">${escapeHtml(pick.filename)}</span>`
      : `<span class="ct-empty">${slot.toUpperCase()}</span>`;
    button.title = pick ? `${pick.filename} — click to remove` : `Pick image ${slot.toUpperCase()}`;
  }

  const ready = Boolean(comparePicks.a && comparePicks.b);
  $('compareRunBtn').disabled = !ready;
  setText('compareHint', ready ? 'Ready' : 'Pick a second image');
}

async function runComparison() {
  if (!comparePicks.a || !comparePicks.b) return;
  if (!state.resultPath) return;
  const body = $('compareBody');
  $('compareModal').hidden = false;
  body.innerHTML = '<p class="help">Comparing...</p>';
  try {
    const data = await analysisRequest('/api/analysis/compare', {
      path_a: comparePicks.a.relative_path,
      path_b: comparePicks.b.relative_path,
    });
    renderComparison(data);
  } catch (err) {
    body.innerHTML = `<p class="help">Comparison failed: ${escapeHtml(err.message)}</p>`;
  }
}

// Cosine runs -1..1; the bar maps that range so a negative similarity reads as
// "actively unalike" rather than simply as zero.
function similarityPercent(value) {
  return Math.max(0, Math.min(100, ((value + 1) / 2) * 100));
}

// What makes a relationship one-sided is the *gap* between the two ranks, not a
// low minimum: #4 and #7 is a balanced pair, #1 and #200 is not.
function rankVerdict(data) {
  const best = Math.min(data.rank_of_b_in_a, data.rank_of_a_in_b);
  const worst = Math.max(data.rank_of_b_in_a, data.rank_of_a_in_b);
  if (worst === 1) return "Each is the other's closest match.";
  if (worst <= 10) return "Each sits near the top of the other's neighbours \u2014 a mutual relationship.";
  if (best <= 10 && worst >= best * 6) {
    return "One ranks the other highly, the other does not \u2014 a one-sided relationship, usually because the second image is a popular match for many others.";
  }
  if (worst <= 30) return "Related, but neither is central to the other.";
  return "Distant: neither appears among the other's close neighbours.";
}


function renderComparison(data) {
  const shared = data.shared_neighbours.map(item => cardMarkup(item, {
    size: 120,
    meta: [`A ${item.similarity_to_a.toFixed(3)}`, `B ${item.similarity_to_b.toFixed(3)}`],
  })).join('');

  const pct = similarityPercent(data.similarity);
  const medianPct = similarityPercent(data.median_similarity);

  $('compareBody').innerHTML = `
    <div class="cmp-pair">
      <figure class="cmp-side">
        <img src="${escapeHtml(thumbUrl(data.a, 420))}" alt="" />
        <figcaption><span class="cmp-tag">A</span>${escapeHtml(data.a.filename)}</figcaption>
      </figure>

      <div class="cmp-middle">
        <div class="cmp-score" title="Cosine similarity in the model's embedding space: 1 is identical, 0 unrelated, below 0 actively unalike.">
          <span class="cmp-score-value">${data.similarity.toFixed(4)}</span>
          <span class="cmp-score-label">cosine similarity</span>
        </div>
        <div class="cmp-bar" title="Where this pair sits on the -1..1 scale. The marker is the collection's median pair, for comparison.">
          <i class="cmp-bar-fill" style="width:${pct}%"></i>
          <i class="cmp-bar-median" style="left:${medianPct}%"></i>
        </div>
        <p class="cmp-note">Median pair in this collection: <strong>${data.median_similarity.toFixed(4)}</strong></p>

        <dl class="cmp-facts">
          <dt title="Where B sits in A's ranked neighbour list, out of every other image.">B in A's neighbours</dt>
          <dd>#${data.rank_of_b_in_a} <span class="muted">of ${data.searched - 1}</span></dd>
          <dt title="Where A sits in B's ranked neighbour list. A large gap between the two ranks means the relationship is one-sided.">A in B's neighbours</dt>
          <dd>#${data.rank_of_a_in_b} <span class="muted">of ${data.searched - 1}</span></dd>
          <dt title="Share of A's pairs that are weaker than this one.">Closer than</dt>
          <dd>${data.percentile_in_a}% <span class="muted">of A's pairs</span></dd>
          <dt title="Images among the closest neighbours of both. Two works can be only moderately similar to each other and still belong together because they share a neighbourhood.">Shared neighbours</dt>
          <dd>${data.shared_count} <span class="muted">of top ${data.neighbours}</span></dd>
        </dl>

        <p class="cmp-verdict">${escapeHtml(rankVerdict(data))}</p>
        ${data.same_cluster ? '<span class="cmp-badge">same cluster</span>' : ''}
      </div>

      <figure class="cmp-side">
        <img src="${escapeHtml(thumbUrl(data.b, 420))}" alt="" />
        <figcaption><span class="cmp-tag">B</span>${escapeHtml(data.b.filename)}</figcaption>
      </figure>
    </div>

    <section class="cmp-shared">
      <h3>Neighbours they have in common <span class="count mono">${data.shared_count}</span></h3>
      <p class="help">
        Images among the closest neighbours of both. A high count with a middling similarity means the two belong to
        the same neighbourhood without being alike in themselves — often a more interesting relationship than a direct
        resemblance.
      </p>
      ${shared
        ? `<div class="gallery-grid analysis-grid cmp-shared-grid">${shared}</div>`
        : '<p class="help cmp-empty">These two share no close neighbours at all — whatever relationship they have is direct, not through a common neighbourhood.</p>'}
    </section>`;

  wireCards($('compareBody'));
}

function closeComparison() {
  const modal = $('compareModal');
  if (modal) modal.hidden = true;
}

document.querySelectorAll('[data-compare-slot]').forEach(button => {
  button.addEventListener('click', () => clearComparisonSlot(button.dataset.compareSlot));
});
$('compareRunBtn')?.addEventListener('click', runComparison);
$('compareCloseBtn')?.addEventListener('click', clearComparison);
document.querySelectorAll('[data-close-compare]').forEach(el => el.addEventListener('click', closeComparison));
$('previewCompareBtn')?.addEventListener('click', () => {
  const path = $('previewSimilarBtn')?.dataset.path;
  if (!path) return;
  closeImagePreview();
  addToComparison(path);
});

// Escape closes the comparison before anything else on the page reacts to it.
window.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  const modal = $('compareModal');
  if (modal && !modal.hidden) {
    closeComparison();
    event.stopImmediatePropagation();
  }
}, true);

// A new projection invalidates the picks: the paths may not exist in it.
const originalIngestForCompare = window.ingestRows;
if (typeof originalIngestForCompare === 'function') {
  window.ingestRows = async function ingestRowsClearingComparison(...args) {
    clearComparison();
    closeComparison();
    return originalIngestForCompare(...args);
  };
}

renderCompareTray();
