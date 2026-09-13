// Lenses: three questions the map cannot answer.
//
// The projection, the mosaic and the graph all reply with a picture of the
// collection. These reply with something else — a list of words, a derived query,
// a ranking of disagreements — which is why they live in their own view instead
// of crowding a workspace built around a plot.
//
// Standalone page: it shares the stylesheet with the rest of the app but none of
// its globals.

const state = {
  projections: [],
  projectionPath: '',
  rows: [],
  clusters: new Map(),
  vocabulary: [],
  defaultVocabulary: [],
  lens: 'distinguish',
};

const $ = id => document.getElementById(id);

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function setHTML(id, value) {
  const el = $(id);
  if (el) el.innerHTML = value;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

function hasNumber(value) {
  if (value === null || value === undefined || value === '') return false;
  return Number.isFinite(Number(value));
}

async function apiJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let text = await res.text();
    try { text = JSON.parse(text).detail || text; } catch (_) { /* plain-text error */ }
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function post(url, body) {
  return apiJson(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

function thumbUrl(item, width = 160) {
  return `/api/thumb?path=${encodeURIComponent(item.relative_path)}&w=${Math.round(width)}`;
}

function imageUrl(item) {
  return `/${item.relative_path}`;
}

function phraseList(id) {
  return String($(id)?.value || '').split(';').map(part => part.trim()).filter(Boolean);
}

function setBanner(id, message, tone = 'idle') {
  setText(id, message);
  const el = $(id);
  if (el) el.className = `status-banner ${tone}`;
}

// One card, used by every lens that answers with images.
function imageCard(item, meta = []) {
  return `
    <article class="search-card lens-card" data-path="${escapeHtml(item.relative_path)}"
             title="${escapeHtml(item.filename)}">
      <img src="${escapeHtml(thumbUrl(item, 220))}" alt="" loading="lazy" />
      <div class="search-meta">${meta.map(m => `<span>${escapeHtml(m)}</span>`).join('')}</div>
      <div class="search-title">${escapeHtml(item.filename)}</div>
      <a class="search-open-link" href="${escapeHtml(imageUrl(item))}" target="_blank" rel="noopener">Open</a>
    </article>`;
}

// --------------------------------------------------------------------------
// Source

async function loadProjections() {
  try {
    const data = await apiJson('/api/projections');
    state.projections = data.projections || [];
    const options = state.projections.length
      ? '<option value="">Choose a projection...</option>' +
        state.projections.map(p => `<option value="${escapeHtml(p.relative_path)}">${escapeHtml(p.name)}</option>`).join('')
      : '<option value="">No saved projections</option>';
    setHTML('lensProjection', options);
    setHTML('disagreementA', options);
    setHTML('disagreementB', options);
    if (!state.projections.length) {
      setBanner('lensSourceInfo', 'Generate a projection first — every lens reads one.', 'warn');
    }
  } catch (err) {
    setBanner('lensSourceInfo', `Cannot list projections: ${err.message}`, 'warn');
  }
}

async function chooseProjection(path) {
  state.projectionPath = path;
  state.rows = [];
  state.clusters = new Map();
  if (!path) {
    setBanner('lensSourceInfo', 'Choose a saved projection.');
    setText('lensStatusChip', 'Choose a projection');
    return;
  }
  setBanner('lensSourceInfo', 'Reading...');
  try {
    const data = await apiJson(`/api/projections/read?path=${encodeURIComponent(path)}`);
    state.rows = data.rows || [];
    groupClusters();
    populateGroupPickers();
    setBanner('lensSourceInfo', `${state.rows.length} images · ${state.clusters.size} cluster(s).`, 'ok');
    setText('lensStatusChip', `${state.rows.length} images`);
    $('lensStatusChip').className = 'chip chip-success dot';
  } catch (err) {
    setBanner('lensSourceInfo', `Cannot read this projection: ${err.message}`, 'warn');
  }
}

function groupClusters() {
  state.clusters = new Map();
  for (const row of state.rows) {
    if (!hasNumber(row.cluster)) continue;
    const key = String(Number(row.cluster));
    if (!state.clusters.has(key)) state.clusters.set(key, []);
    state.clusters.get(key).push(row);
  }
}

// Group choices: each cluster, plus "everything else" so a cluster can always be
// contrasted with its complement even when there is only one of them.
function groupOptions() {
  const options = [];
  const keys = [...state.clusters.keys()].sort((a, b) => Number(a) - Number(b));
  for (const key of keys) {
    options.push({ value: `cluster:${key}`, label: `Cluster ${key} (${state.clusters.get(key).length})` });
  }
  for (const key of keys) {
    options.push({ value: `not:${key}`, label: `Everything except cluster ${key}` });
  }
  return options;
}

// Clusters are only available when the projection has them; the concept mode
// works on any projection, so it is the default and the fallback.
function updateGroupMode() {
  const mode = $('groupMode')?.value || 'concept';
  document.querySelectorAll('[data-group-mode]').forEach(block => {
    block.hidden = block.dataset.groupMode !== mode;
  });
  const clusterOption = $('groupMode')?.querySelector('option[value="cluster"]');
  if (clusterOption) {
    const available = state.clusters.size >= 1;
    clusterOption.disabled = !available;
    clusterOption.textContent = available ? 'Clusters' : 'Clusters (this projection has none)';
    if (!available && mode === 'cluster') {
      $('groupMode').value = 'concept';
      updateGroupMode();
    }
  }
}

function populateGroupPickers() {
  const options = groupOptions();
  const markup = options.length
    ? options.map(o => `<option value="${escapeHtml(o.value)}">${escapeHtml(o.label)}</option>`).join('')
    : '<option value="">This projection has no clusters</option>';
  setHTML('distinguishA', markup);
  setHTML('distinguishB', markup);
  // Default to contrasting the first cluster with its complement, which is the
  // most immediately informative pairing.
  if (options.length >= 2) {
    $('distinguishA').value = options[0].value;
    $('distinguishB').value = options.find(o => o.value.startsWith('not:'))?.value || options[1].value;
  }

  const rows = state.rows.slice(0, 2000);
  setHTML('arithmeticBase', '<option value="">(start from a phrase instead)</option>' +
    rows.map(r => `<option value="${escapeHtml(r.relative_path)}">${escapeHtml(r.filename)}</option>`).join(''));

  updateGroupMode();
}

// Score the collection against a phrase and take the two ends. The middle is
// deliberately excluded: contrasting the extremes is what makes the difference
// legible, and the images in between belong to neither side.
async function conceptGroups() {
  const phrases = phraseList('groupConcept');
  if (!phrases.length) throw new Error('Enter a concept to split the collection by.');
  const size = Math.max(2, Math.min(500, Number($('groupSize').value || 40)));

  const data = await post('/api/semantic/concept-scores', {
    projection_path: state.projectionPath,
    phrases,
  });
  const scored = data.scores
    .filter(item => hasNumber(item.score))
    .sort((a, b) => b.score - a.score);
  if (scored.length < 4) throw new Error('Too few scored images to form two groups.');

  const take = Math.min(size, Math.floor(scored.length / 2));
  return {
    a: scored.slice(0, take).map(item => item.relative_path),
    b: scored.slice(-take).map(item => item.relative_path),
    label: `most vs least "${phrases.join(' · ')}"`,
    take,
  };
}

function resolveGroup(value) {
  if (!value) return [];
  const [kind, key] = value.split(':');
  if (kind === 'cluster') return (state.clusters.get(key) || []).map(r => r.relative_path);
  if (kind === 'not') {
    return state.rows
      .filter(r => String(Number(r.cluster)) !== key)
      .map(r => r.relative_path);
  }
  return [];
}

// --------------------------------------------------------------------------
// Lens switching

function showLens(name) {
  state.lens = name;
  document.querySelectorAll('[data-lens]').forEach(button => {
    button.classList.toggle('active', button.dataset.lens === name);
  });
  document.querySelectorAll('[data-lens-pane]').forEach(pane => {
    pane.classList.toggle('active', pane.dataset.lensPane === name);
  });
}

// --------------------------------------------------------------------------
// 1.4 What separates two groups

async function loadVocabulary() {
  try {
    const data = await apiJson('/api/semantic/vocabulary');
    state.defaultVocabulary = data.terms || [];
    state.vocabulary = [...state.defaultVocabulary];
    $('vocabularyText').value = state.vocabulary.join('\n');
    updateVocabularyCount();
  } catch (_) {
    state.defaultVocabulary = [];
  }
}

function currentVocabulary() {
  const text = String($('vocabularyText')?.value || '');
  const terms = text.split('\n').map(t => t.trim()).filter(Boolean);
  return terms.length ? terms : null;
}

function updateVocabularyCount() {
  const terms = currentVocabulary() || [];
  setText('vocabularyCount', `${terms.length} term${terms.length === 1 ? '' : 's'}`);
}

async function runDistinguish() {
  if (!state.projectionPath) { setBanner('distinguishStatus', 'Choose a projection first.', 'warn'); return; }
  setBanner('distinguishStatus', 'Reading the difference... the first run loads the model.');
  setHTML('distinguishResult', '');
  try {
    let pathsA;
    let pathsB;
    let label = '';
    if (($('groupMode')?.value || 'concept') === 'concept') {
      const groups = await conceptGroups();
      pathsA = groups.a;
      pathsB = groups.b;
      label = `${groups.label} — ${groups.take} images each. `;
    } else {
      pathsA = resolveGroup($('distinguishA').value);
      pathsB = resolveGroup($('distinguishB').value);
      if (!pathsA.length || !pathsB.length) {
        setBanner('distinguishStatus', 'Both groups need at least one image.', 'warn');
        return;
      }
    }
    const data = await post('/api/semantic/distinguish', {
      projection_path: state.projectionPath,
      paths_a: pathsA,
      paths_b: pathsB,
      vocabulary: currentVocabulary(),
      top_k: Number($('distinguishTopK').value || 10),
    });
    renderDistinguish(data, label);
  } catch (err) {
    setBanner('distinguishStatus', `Failed: ${err.message}`, 'warn');
  }
}

function renderDistinguish(data, label = '') {
  // Centroid similarity is the caveat that makes the word list readable: a long
  // list means little when the two groups nearly coincide.
  const closeness = data.centroid_similarity;
  const caveat = closeness > 0.95
    ? ' These two groups sit almost on top of each other, so the terms below separate very little.'
    : closeness > 0.85
      ? ' The two groups overlap considerably; the terms mark a tendency, not a division.'
      : '';
  setBanner('distinguishStatus',
    `${label}A: ${data.size_a} images · B: ${data.size_b} · centroid similarity ${closeness.toFixed(3)} · ` +
    `${data.vocabulary_size} terms.${caveat}`, closeness > 0.95 ? 'warn' : 'ok');

  const scale = Math.max(
    ...data.towards_a.map(t => Math.abs(t.score)),
    ...data.towards_b.map(t => Math.abs(t.score)),
    1e-6,
  );

  const column = (title, subtitle, terms, tone) => `
    <section class="dist-column ${tone}">
      <h3>${escapeHtml(title)}</h3>
      <p class="help">${escapeHtml(subtitle)}</p>
      <ol class="dist-terms">
        ${terms.map(term => `
          <li title="How far this term lies towards this group along the direction between the two centroids.">
            <span class="dt-term">${escapeHtml(term.term)}</span>
            <span class="dt-bar"><i style="width:${Math.max(2, (Math.abs(term.score) / scale) * 100).toFixed(1)}%"></i></span>
            <span class="dt-score mono">${term.score.toFixed(3)}</span>
          </li>`).join('')}
      </ol>
    </section>`;

  setHTML('distinguishResult', `
    <div class="dist-grid">
      ${column('Towards A', 'What group A has that group B does not.', data.towards_a, 'tone-a')}
      ${column('Towards B', 'What group B has that group A does not.', data.towards_b, 'tone-b')}
    </div>
    <p class="help dist-footnote">
      Read these as the model's vocabulary for the gap, not as labels. Terms the vocabulary does not contain cannot
      appear, however well they would fit &mdash; edit the list if the answer feels like it is missing the point.
    </p>`);
}

// --------------------------------------------------------------------------
// 1.6 Concept arithmetic

async function runArithmetic() {
  if (!state.projectionPath) { setBanner('arithmeticStatus', 'Choose a projection first.', 'warn'); return; }
  const base = $('arithmeticBase').value;
  const add = phraseList('arithmeticAdd');
  const subtract = phraseList('arithmeticSub');
  const strength = Number($('arithmeticStrength')?.value ?? 1);
  if (!base && !add.length) {
    setBanner('arithmeticStatus', 'Start from an image, or give at least one term to add.', 'warn');
    return;
  }
  setBanner('arithmeticStatus', 'Computing... the first run loads the model.');
  setHTML('arithmeticResult', '');
  try {
    const data = await post('/api/semantic/arithmetic', {
      projection_path: state.projectionPath,
      relative_path: base || null,
      add,
      subtract,
      strength,
      top_k: 24,
    });
    setBanner('arithmeticStatus',
      `${data.expression || 'query'} → ${data.count} of ${data.searched} images.`, 'ok');
    const baseRow = state.rows.find(r => r.relative_path === base);
    const header = baseRow ? `
      <div class="arith-expression">
        <figure><img src="${escapeHtml(thumbUrl(baseRow, 160))}" alt="" /><figcaption>${escapeHtml(baseRow.filename)}</figcaption></figure>
        ${subtract.map(t => `<span class="ae-op">&minus;</span><span class="ae-term">${escapeHtml(t)}</span>`).join('')}
        ${add.map(t => `<span class="ae-op">+</span><span class="ae-term">${escapeHtml(t)}</span>`).join('')}
        <span class="ae-op">=</span>
      </div>` : '';
    setHTML('arithmeticResult', header +
      `<div class="lens-grid">${data.results.map(item =>
        imageCard(item, [`#${item.rank}`, `${item.score >= 0 ? '+' : ''}${item.score.toFixed(2)}σ`])).join('')}</div>`);
  } catch (err) {
    setBanner('arithmeticStatus', `Failed: ${err.message}`, 'warn');
  }
}

// --------------------------------------------------------------------------
// 3. Where two models disagree

async function runDisagreement() {
  const a = $('disagreementA').value;
  const b = $('disagreementB').value;
  if (!a || !b) { setBanner('disagreementStatus', 'Pick two projections.', 'warn'); return; }
  if (a === b) { setBanner('disagreementStatus', 'Pick two different projections.', 'warn'); return; }
  setBanner('disagreementStatus', 'Comparing the two neighbourhoods of every image...');
  setHTML('disagreementResult', '');
  try {
    const data = await post('/api/analysis/disagreement', {
      projection_a: a,
      projection_b: b,
      neighbours: Number($('disagreementK').value || 15),
      top_k: 40,
    });
    renderDisagreement(data);
  } catch (err) {
    setBanner('disagreementStatus', `Failed: ${err.message}`, 'warn');
  }
}

function renderDisagreement(data) {
  const verdict = data.mean_agreement > 0.6
    ? 'The two models largely agree about this collection.'
    : data.mean_agreement > 0.25
      ? 'The two models agree in part: enough overlap to be comparable, enough difference to be worth reading.'
      : 'The two models see this collection very differently — they share few neighbours anywhere.';
  setBanner('disagreementStatus',
    `${data.model_a || 'A'} vs ${data.model_b || 'B'} · ${data.count} images · ` +
    `mean overlap ${(data.mean_agreement * 100).toFixed(0)}% of ${data.neighbours} neighbours · ` +
    `${data.total_disagreement} share none. ${verdict}`, 'ok');

  // Overlap is a count out of k, so ties are the rule rather than the exception.
  // Saying so is the honest framing: without it the arbitrary order inside a tie
  // reads as a ranking, and the first few look singled out when nothing
  // distinguishes them from the other sixty.
  const tiedNote = data.tied_at_top > 1
    ? `<strong>${data.tied_at_top} images share the lowest overlap</strong>, so the order among them is arbitrary &mdash;
       read them as one group, not as a ranking. `
    : '';

  setHTML('disagreementResult', `
    <p class="help">
      ${tiedNote}Lowest overlap first: these are the images whose closest relatives change completely depending on
      which model you ask. A low score is the interesting end.
    </p>
    <div class="lens-grid">
      ${data.results.map(item => imageCard(item, [
        // A shared rank is shown as "=3" rather than "#3": the position is not
        // this image's own.
        item.tied_with > 1 ? `=${item.rank} of ${item.tied_with} tied` : `#${item.rank}`,
        `overlap ${(item.score * 100).toFixed(0)}%`,
        `${item.shared_neighbours}/${data.neighbours} shared`,
      ])).join('')}
    </div>`);
}

// --------------------------------------------------------------------------
// Wiring

document.querySelectorAll('[data-lens]').forEach(button => {
  button.addEventListener('click', () => showLens(button.dataset.lens));
});
$('lensProjection').addEventListener('change', e => chooseProjection(e.target.value));
$('distinguishRunBtn').addEventListener('click', runDistinguish);
$('groupMode').addEventListener('change', updateGroupMode);
$('arithmeticRunBtn').addEventListener('click', runArithmetic);
if ($('arithmeticStrength')) {
  const syncStrength = () => setText('arithmeticStrengthOut', `${Number($('arithmeticStrength').value).toFixed(2)}×`);
  $('arithmeticStrength').addEventListener('input', syncStrength);
  syncStrength();
}
$('disagreementRunBtn').addEventListener('click', runDisagreement);
$('vocabularyText').addEventListener('input', updateVocabularyCount);
$('vocabularyResetBtn').addEventListener('click', () => {
  $('vocabularyText').value = state.defaultVocabulary.join('\n');
  updateVocabularyCount();
});
document.querySelectorAll('[data-toggle-vocabulary]').forEach(button => {
  button.addEventListener('click', () => {
    const box = $('vocabularyBox');
    if (box) box.hidden = !box.hidden;
  });
});

function setHelpOpen(open) {
  const modal = $('lensHelpModal');
  if (!modal) return;
  modal.hidden = !open;
}
document.querySelectorAll('[data-open-help]').forEach(el => el.addEventListener('click', () => setHelpOpen(true)));
document.querySelectorAll('[data-close-help]').forEach(el => el.addEventListener('click', () => setHelpOpen(false)));
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') setHelpOpen(false);
});

function init() {
  loadProjections().then(() => {
    // Arriving from the plot's "Lenses" link carries the projection along.
    const requested = new URLSearchParams(window.location.search).get('projection');
    if (requested && state.projections.some(p => p.relative_path === requested)) {
      $('lensProjection').value = requested;
      chooseProjection(requested);
      $('disagreementA').value = requested;
    }
  });
  loadVocabulary();
  showLens('distinguish');
}

init();
