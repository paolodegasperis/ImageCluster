// Compare: a place rather than a dialog.
//
// The projection page can compare two images in a modal, which is right for a
// glance while reading the map. This page is for the other case - when the pair
// *is* the work. Everything the collection knows about two images is on one
// screen: the similarity and its context, the chain of neighbours that carries
// one into the other, and the neighbours they share.
//
// Picking is the part that had to change. In the modal the two images arrive
// from wherever the user happened to be; here they are chosen from a mosaic of
// the whole collection, which is the only way to compare two works that are not
// already near each other on the map.
//
// Everything is read from embeddings already on disk: no model is loaded, so
// this works with the image-only encoders too.

const $ = id => document.getElementById(id);

// Images are added to the mosaic a page at a time. A collection of ten thousand
// images is ten thousand <img> elements and ten thousand thumbnail requests if
// it is built in one go, which locks the browser for seconds.
const PAGE = 150;

const state = {
  projections: [],
  path: '',
  rows: [],
  filtered: [],
  shown: 0,
  picks: { a: null, b: null },
  comparison: null,
  chain: null,
  k: 6,
  thumb: 92,
};

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

async function apiJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `${response.status} ${response.statusText}`);
  return payload;
}

function thumbUrl(row, width) {
  return `/api/thumb?path=${encodeURIComponent(row.relative_path)}&w=${width}`;
}

function imageUrl(row) {
  // The image folder is mounted directly; the projection page resolves it the
  // same way, so a full-resolution link works without a dedicated endpoint.
  return `/${row.relative_path}`;
}

// --------------------------------------------------------------------------
// Source

async function loadProjections() {
  try {
    const data = await apiJson('/api/projections');
    state.projections = data.projections || [];
  } catch {
    state.projections = [];
  }
  const select = $('cmpProjection');
  if (!state.projections.length) {
    select.innerHTML = '<option value="">No saved projections</option>';
    setText('cmpSourceInfo', 'Generate a projection first, on the Embedding Projection page.');
    return;
  }
  select.innerHTML = '<option value="">Choose a saved projection...</option>'
    + state.projections.map(p => `<option value="${escapeHtml(p.relative_path)}">${escapeHtml(p.name)}</option>`).join('');

  // A deep link from the projection page carries the pair it was looking at.
  const params = new URLSearchParams(location.search);
  const wanted = params.get('path');
  const known = state.projections.some(p => p.relative_path === wanted);
  const chosen = known ? wanted : state.projections[0].relative_path;
  select.value = chosen;
  await loadProjection(chosen, params.get('a'), params.get('b'));
}

async function loadProjection(path, wantA, wantB) {
  if (!path) return;
  state.path = path;
  setText('cmpSourceInfo', 'Loading...');
  try {
    const data = await apiJson(`/api/projections/read?path=${encodeURIComponent(path)}`);
    state.rows = (data.rows || []).map((row, index) => ({ ...row, index }));
  } catch (err) {
    state.rows = [];
    setText('cmpSourceInfo', `Could not read this projection: ${err.message}`);
    return;
  }
  state.picks = { a: null, b: null };
  state.comparison = null;
  state.chain = null;

  const clusters = [...new Set(state.rows.map(r => r.cluster).filter(c => c !== undefined && c !== null && c !== ''))]
    .sort((x, y) => Number(x) - Number(y));
  $('cmpCluster').innerHTML = '<option value="">All clusters</option>'
    + clusters.map(c => `<option value="${escapeHtml(c)}">Cluster ${escapeHtml(c)}</option>`).join('');

  setText('cmpSourceInfo', `${state.rows.length} images ready to compare.`);
  setText('cmpStatusChip', `${state.rows.length} images`);
  $('cmpStatusChip').className = 'chip dot';

  applyFilter();
  if (wantA) pick(state.rows.find(r => r.relative_path === wantA));
  if (wantB) pick(state.rows.find(r => r.relative_path === wantB));
}

// --------------------------------------------------------------------------
// The mosaic

function applyFilter() {
  const needle = ($('cmpFilter').value || '').trim().toLowerCase();
  const cluster = $('cmpCluster').value;
  state.filtered = state.rows.filter(row => {
    if (cluster !== '' && String(row.cluster) !== cluster) return false;
    if (needle && !String(row.filename || '').toLowerCase().includes(needle)) return false;
    return true;
  });
  state.shown = 0;
  $('cmpMosaic').innerHTML = '';
  setText('cmpCount', String(state.filtered.length));
  showMore();
}

function showMore() {
  const mosaic = $('cmpMosaic');
  const next = state.filtered.slice(state.shown, state.shown + PAGE);
  if (next.length) {
    mosaic.insertAdjacentHTML('beforeend', next.map(tileMarkup).join(''));
    state.shown += next.length;
  }
  $('cmpMore').hidden = state.shown >= state.filtered.length;
  markPicked();
}

function tileMarkup(row) {
  return `
    <button class="cmp-tile" type="button" role="option" aria-selected="false"
            data-path="${escapeHtml(row.relative_path)}" title="${escapeHtml(row.filename)}">
      <img src="${escapeHtml(thumbUrl(row, 160))}" alt="" loading="lazy" />
      <span class="cmp-tile-name">${escapeHtml(row.filename)}</span>
    </button>`;
}

function markPicked() {
  const a = state.picks.a?.relative_path;
  const b = state.picks.b?.relative_path;
  $('cmpMosaic').querySelectorAll('.cmp-tile').forEach(tile => {
    const path = tile.dataset.path;
    const slot = path === a ? 'a' : (path === b ? 'b' : '');
    tile.classList.toggle('picked', Boolean(slot));
    tile.dataset.pickedSlot = slot;
    tile.setAttribute('aria-selected', slot ? 'true' : 'false');
  });
}

// --------------------------------------------------------------------------
// Picking

function pick(row) {
  if (!row) return;
  const { a, b } = state.picks;
  // Clicking something already chosen removes it: one gesture, both directions.
  if (a && a.relative_path === row.relative_path) return clearSlot('a');
  if (b && b.relative_path === row.relative_path) return clearSlot('b');

  if (!a) state.picks.a = row;
  else if (!b) state.picks.b = row;
  else state.picks.b = row; // a third pick replaces B, so the page never needs a reset
  afterPickChange();
}

function clearSlot(slot) {
  state.picks[slot] = null;
  if (!state.picks.a && state.picks.b) {
    state.picks.a = state.picks.b;
    state.picks.b = null;
  }
  afterPickChange();
}

function afterPickChange() {
  renderSlots();
  markPicked();
  state.comparison = null;
  state.chain = null;
  if (state.picks.a && state.picks.b) run();
  else renderResult();
}

function renderSlots() {
  for (const slot of ['a', 'b']) {
    const button = $(slot === 'a' ? 'cmpSlotA' : 'cmpSlotB');
    const pick_ = state.picks[slot];
    button.classList.toggle('filled', Boolean(pick_));
    button.innerHTML = pick_
      ? `<span class="cmp-slot-tag">${slot.toUpperCase()}</span>
         <img src="${escapeHtml(thumbUrl(pick_, 120))}" alt="" />
         <span class="cmp-slot-name">${escapeHtml(pick_.filename)}</span>`
      : `<span class="cmp-slot-tag">${slot.toUpperCase()}</span>
         <span class="cmp-slot-empty">${slot === 'a' ? 'Click a thumbnail' : 'Click a second one'}</span>`;
  }
  const both = Boolean(state.picks.a && state.picks.b);
  $('cmpSwapBtn').disabled = !both;
  $('cmpClearBtn').disabled = !(state.picks.a || state.picks.b);
}

// --------------------------------------------------------------------------
// The comparison

async function run() {
  const { a, b } = state.picks;
  if (!a || !b) return;
  $('cmpPlaceholder').hidden = true;
  $('cmpResult').hidden = false;
  $('cmpResult').innerHTML = '<p class="help">Comparing...</p>';
  try {
    const [comparison, chain] = await Promise.all([
      apiJson('/api/analysis/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ projection_path: state.path, path_a: a.relative_path, path_b: b.relative_path }),
      }),
      apiJson('/api/analysis/chain', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projection_path: state.path, from_path: a.relative_path, to_path: b.relative_path, k: state.k,
        }),
      }).catch(err => ({ error: err.message })),
    ]);
    state.comparison = comparison;
    state.chain = chain;
  } catch (err) {
    $('cmpResult').innerHTML = `<p class="help">Comparison failed: ${escapeHtml(err.message)}</p>`;
    return;
  }
  renderResult();
}

async function runChainOnly() {
  const { a, b } = state.picks;
  if (!a || !b) return;
  try {
    state.chain = await apiJson('/api/analysis/chain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        projection_path: state.path, from_path: a.relative_path, to_path: b.relative_path, k: state.k,
      }),
    });
  } catch (err) {
    state.chain = { error: err.message };
  }
  renderResult();
}

function similarityPercent(value) {
  return Math.max(0, Math.min(100, ((value + 1) / 2) * 100));
}

function rankVerdict(data) {
  const best = Math.min(data.rank_of_b_in_a, data.rank_of_a_in_b);
  const worst = Math.max(data.rank_of_b_in_a, data.rank_of_a_in_b);
  if (worst === 1) return "Each is the other's closest match.";
  if (worst <= 10) return "Each sits near the top of the other's neighbours — a mutual relationship.";
  if (best <= 10 && worst >= best * 6) {
    return "One ranks the other highly, the other does not — usually because the second image is a popular match for many others.";
  }
  if (worst <= 30) return 'Related, but neither is central to the other.';
  return "Distant: neither appears among the other's close neighbours.";
}

function chainMarkup(chain) {
  if (!chain) return '';
  if (chain.error) return `<p class="help">Chain unavailable: ${escapeHtml(chain.error)}</p>`;
  if (!chain.connected) {
    return `<p class="help">${escapeHtml(chain.reason)}</p>`;
  }
  const parts = [];
  chain.steps.forEach((step, index) => {
    const end = index === 0 || index === chain.steps.length - 1;
    parts.push(`
      <figure class="chain-step${end ? ' chain-end' : ''}" data-path="${escapeHtml(step.relative_path)}">
        <img src="${escapeHtml(thumbUrl(step, 140))}" alt="" />
        <figcaption>${escapeHtml(step.filename)}</figcaption>
      </figure>`);
    if (index < chain.links.length) {
      parts.push(`<div class="chain-link" title="Similarity between these two neighbours">
        <span class="chain-arrow">&rarr;</span><span class="mono">${chain.links[index].toFixed(3)}</span></div>`);
    }
  });
  return `
    <p class="cmp-line">${chain.hops} step${chain.hops === 1 ? '' : 's'} at k=${chain.k}
      <span class="muted">&middot; weakest link ${chain.weakest_link.toFixed(3)}</span></p>
    <div class="chain-strip">${parts.join('')}</div>`;
}

function sharedMarkup(item) {
  return `
    <article class="cmp-neighbour" data-path="${escapeHtml(item.relative_path)}" title="${escapeHtml(item.filename)}">
      <img src="${escapeHtml(thumbUrl(item, 200))}" alt="" loading="lazy" />
      <div class="cmp-neighbour-name">${escapeHtml(item.filename)}</div>
      <div class="cmp-neighbour-scores mono">
        <span title="Similarity to A">A ${item.similarity_to_a.toFixed(3)}</span>
        <span title="Similarity to B">B ${item.similarity_to_b.toFixed(3)}</span>
      </div>
    </article>`;
}

function renderResult() {
  const data = state.comparison;
  if (!data) {
    $('cmpResult').hidden = true;
    $('cmpPlaceholder').hidden = false;
    return;
  }
  $('cmpPlaceholder').hidden = true;
  $('cmpResult').hidden = false;

  const pct = similarityPercent(data.similarity);
  const medianPct = similarityPercent(data.median_similarity);
  const shared = data.shared_neighbours.map(sharedMarkup).join('');

  $('cmpResult').innerHTML = `
    <div class="cmp-pair">
      <figure class="cmp-side">
        <a href="${escapeHtml(imageUrl(data.a))}" target="_blank" rel="noopener"
           title="Open the full-resolution image in a new tab">
          <img src="${escapeHtml(thumbUrl(data.a, 520))}" alt="" />
        </a>
        <figcaption><span class="cmp-tag">A</span>${escapeHtml(data.a.filename)}</figcaption>
      </figure>

      <div class="cmp-middle">
        <div class="cmp-score" title="Cosine similarity in the model's embedding space: 1 is identical, 0 unrelated, below 0 actively unalike.">
          <span class="cmp-score-value">${data.similarity.toFixed(4)}</span>
          <span class="cmp-score-label">cosine similarity</span>
        </div>
        <div class="cmp-bar" title="Where this pair sits on the -1..1 scale. The marker is this collection's median pair.">
          <i class="cmp-bar-fill" style="width:${pct}%"></i>
          <i class="cmp-bar-median" style="left:${medianPct}%"></i>
        </div>
        <p class="cmp-note">Median pair here: <strong>${data.median_similarity.toFixed(4)}</strong></p>

        <dl class="cmp-facts">
          <dt title="Where B sits in A's ranked neighbour list, out of every other image.">B in A's neighbours</dt>
          <dd>#${data.rank_of_b_in_a} <span class="muted">of ${data.searched - 1}</span></dd>
          <dt title="Where A sits in B's ranked neighbour list. A large gap between the two ranks means the relationship is one-sided.">A in B's neighbours</dt>
          <dd>#${data.rank_of_a_in_b} <span class="muted">of ${data.searched - 1}</span></dd>
          <dt title="Share of A's pairs that are weaker than this one.">Closer than</dt>
          <dd>${data.percentile_in_a}% <span class="muted">of A's pairs</span></dd>
          <dt title="Images among the closest neighbours of both.">Shared neighbours</dt>
          <dd>${data.shared_count} <span class="muted">of top ${data.neighbours}</span></dd>
        </dl>

        <p class="cmp-verdict">${escapeHtml(rankVerdict(data))}</p>
        ${data.same_cluster ? '<span class="cmp-badge">same cluster</span>' : ''}
      </div>

      <figure class="cmp-side">
        <a href="${escapeHtml(imageUrl(data.b))}" target="_blank" rel="noopener"
           title="Open the full-resolution image in a new tab">
          <img src="${escapeHtml(thumbUrl(data.b, 520))}" alt="" />
        </a>
        <figcaption><span class="cmp-tag">B</span>${escapeHtml(data.b.filename)}</figcaption>
      </figure>
    </div>

    <section class="cmp-block">
      <h3>The chain between them
        <span class="hint" title="The shortest route from A to B through the nearest-neighbour graph. Each step shows the similarity of that single hop; adjust k on the left.">?</span>
      </h3>
      ${chainMarkup(state.chain)}
    </section>

    <section class="cmp-block">
      <h3>Neighbours they have in common
        <span class="count mono">${data.shared_count}</span>
        <span class="hint" title="Images among the closest neighbours of both. A high count with a middling similarity means the two belong to the same neighbourhood without being alike in themselves.">?</span>
      </h3>
      ${shared
        ? `<div class="cmp-neighbours">${shared}</div>`
        : '<p class="help">No close neighbours in common: whatever relationship these two have is direct, not through a shared neighbourhood.</p>'}
    </section>`;

  // Clicking anything in the result puts it back in the picker, which is the
  // only navigation this page needs.
  $('cmpResult').querySelectorAll('[data-path]').forEach(node => {
    node.addEventListener('click', event => {
      if (event.target.closest('a')) return;
      const row = state.rows.find(r => r.relative_path === node.dataset.path);
      if (row) pick(row);
    });
  });
}

// --------------------------------------------------------------------------
// Wiring

$('cmpProjection').addEventListener('change', event => loadProjection(event.target.value));
$('cmpFilter').addEventListener('input', applyFilter);
$('cmpCluster').addEventListener('change', applyFilter);
$('cmpMoreBtn').addEventListener('click', showMore);
$('cmpMosaic').addEventListener('click', event => {
  const tile = event.target.closest('.cmp-tile');
  if (!tile) return;
  pick(state.rows.find(r => r.relative_path === tile.dataset.path));
});
$('cmpSlotA').addEventListener('click', () => clearSlot('a'));
$('cmpSlotB').addEventListener('click', () => clearSlot('b'));
$('cmpSwapBtn').addEventListener('click', () => {
  const { a, b } = state.picks;
  state.picks = { a: b, b: a };
  afterPickChange();
});
$('cmpClearBtn').addEventListener('click', () => {
  state.picks = { a: null, b: null };
  afterPickChange();
});
$('cmpBigThumbs').addEventListener('change', event => {
  $('cmpMosaic').classList.toggle('large', event.target.checked);
});
$('cmpK').addEventListener('input', event => {
  state.k = Number(event.target.value);
  setText('cmpKValue', String(state.k));
});
$('cmpK').addEventListener('change', runChainOnly);

// Scrolling to the end of the mosaic loads the next page: the button stays for
// anyone who prefers it, and for keyboard users who never scroll into view.
$('cmpMosaic').addEventListener('scroll', () => {
  const el = $('cmpMosaic');
  if (el.scrollTop + el.clientHeight >= el.scrollHeight - 200) showMore();
});

renderSlots();
loadProjections();
