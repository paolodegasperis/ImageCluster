// Thumbnails can be dimmed so overlapping images stay readable. The floor is a
// faint ghost rather than nothing at all: at 0 an image would vanish and the plot
// would look broken, so the slider stops at 5%.
const MIN_THUMB_OPACITY = 0.05;

function clampOpacity(value, fallback = 1) {
  // localStorage.getItem returns null when nothing is stored, and Number(null) is 0 —
  // a finite number, so a plain isFinite guard would silently pick fully transparent.
  // Absent values must fall through to the caller's default instead.
  if (value === null || value === undefined || value === '') return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(MIN_THUMB_OPACITY, Math.min(1, number));
}

const state = {
  rows: [],
  images: new Map(),
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStart: null,
  hover: null,
  // 'images' = thumbnails at their projected position, 'points' = colored marks,
  // 'mosaic' = every image in its own grid cell (see loadMosaic).
  mode: 'images',
  mosaic: null,
  mosaicCells: new Map(),
  mosaicAlgorithm: localStorage.getItem('imagecluster.mosaicAlgorithm') || 'auto',
  mosaicColumns: null,
  mosaicBusy: false,
  thumbSize: 48,
  showThumbnailOutline: localStorage.getItem('imagecluster.showThumbnailOutline') !== 'false',
  // Thumbnails on a light plot background can wash out or, when they overlap,
  // hide each other. A user-controlled alpha makes dense regions readable.
  thumbOpacity: clampOpacity(localStorage.getItem('imagecluster.thumbOpacity')),
  pointSize: 6,
  jobId: null,
  resultPath: null,
  polling: false,
  scanCount: null,
  modelReady: false,
  models: [],
  showPlannedModels: localStorage.getItem('imagecluster.showPlannedModels') === 'true',
  searchResults: [],
  searchResultPaths: new Set(),
  lastSearchPayload: null,
  highlightSearch: true,
  showOnlySearch: false,
  searchThumbSize: Number(localStorage.getItem('imagecluster.searchThumbSize') || 160),
  // Result set of "find images similar to this one". Kept separate from the
  // text-search set so the two can coexist: a curator often wants to see where
  // a text query landed *and* what a particular image is close to.
  similarQuery: null,
  similarResults: [],
  similarPaths: new Set(),
  highlightSimilar: true,
  // Continuous score colouring. One mechanism, many questions: concept
  // intensity today, and anything else that produces a number per image.
  // Null means "colour by cluster", which is the default and the fallback.
  scoreColor: null,
  // Cluster summaries (size, cohesion, medoid) for the loaded projection.
  clusterStats: null,
  clusterSort: localStorage.getItem('imagecluster.clusterSort') || 'typicality',
  selectedCluster: '',
  highlightCluster: true,
  showOnlyCluster: false,
  clusterThumbSize: Number(localStorage.getItem('imagecluster.clusterThumbSize') || 160),
};

const CLUSTER_PALETTE = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899'];


// Level-of-detail thresholds: above LOD_THUMB_MIN_ROWS images, the canvas shows colored
// points until the zoom reaches LOD_THUMB_ZOOM, then switches to thumbnails.
const LOD_THUMB_MIN_ROWS = 500;
const LOD_THUMB_ZOOM = 2;
// Source width (px) of canvas thumbnails fetched from /api/thumb — small enough to keep
// decoded image memory low, large enough for crisp rendering on HiDPI displays.
const CANVAS_THUMB_WIDTH = 192;
// Thumbnails are fetched only for rows the current frame actually draws, and only a
// few at a time: eagerly instantiating one Image per row meant a 10k-image projection
// queued 10k requests and held 10k decoded bitmaps before a single one was visible.
const MAX_CONCURRENT_THUMB_LOADS = 12;
// Once this many bitmaps are resident, the least-recently-drawn ones are dropped.
const MAX_RESIDENT_THUMBS = 1500;

const $ = id => document.getElementById(id);
const canvas = $('plot');
const ctx = canvas.getContext('2d');
const has = id => Boolean($(id));
let resizeFrame = null;
// Perf caches: data bounds are constant per projection, and the canvas rect only
// changes on resize/scroll — recomputing either per point made draw()/findHover() O(n²)
// with thousands of forced reflows per frame. draw() is also rAF-coalesced via scheduleDraw().
let boundsCache = null;
let plotRect = null;
let drawScheduled = false;

function refreshPlotRect() {
  plotRect = canvas.getBoundingClientRect();
  return plotRect;
}

function currentRect() {
  return plotRect || refreshPlotRect();
}

function scheduleDraw() {
  if (drawScheduled) return;
  drawScheduled = true;
  requestAnimationFrame(() => { drawScheduled = false; draw(); });
}

function setText(id, value) {
  const el = $(id);
  if (!el) return;
  el.textContent = value;
}

function setHTML(id, value) {
  const el = $(id);
  if (!el) return;
  el.innerHTML = value;
}

function showElement(id, display = 'block') {
  const el = $(id);
  if (!el) return;
  el.style.display = display;
}

function hideElement(id) {
  const el = $(id);
  if (!el) return;
  el.style.display = 'none';
}

// Tab name -> pane element id. Data-driven rather than a chain of ifs: this used
// to name the two panes explicitly in two different files, so adding a pane meant
// finding both.
const DOCK_PANES = {
  cluster: 'clusterPanel',
  search: 'searchDockPane',
  similar: 'similarDockPane',
  analysis: 'analysisDockPane',
};

function selectDockPane(name, expand = false) {
  if (!DOCK_PANES[name]) return;
  document.querySelectorAll('[data-dock-tab]').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.dockTab === name);
  });
  for (const [tabName, paneId] of Object.entries(DOCK_PANES)) {
    const pane = $(paneId);
    if (pane) pane.classList.toggle('active', tabName === name);
  }
  const dock = document.getElementById('projectionDock');
  if (dock && expand) dock.classList.add('is-expanded');
}

async function apiJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let text = await res.text();
    try { text = JSON.parse(text).detail || text; } catch (_) {}
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

async function loadStatus() {
  try {
    const data = await apiJson('/api/status');
    const missing = (data.dependencies || []).filter(d => d.required && !d.installed);
    if (has('topbarStatusChip')) {
      setText('topbarStatusChip', missing.length ? 'Setup needed' : 'System ready');
      $('topbarStatusChip').className = missing.length ? 'chip chip-danger dot' : 'chip chip-success dot';
    }
    if (has('systemStatus')) {
      setText('systemStatus', missing.length ? 'setup needed' : '0 imgs');
      $('systemStatus').className = missing.length ? 'small danger' : 'small ok';
    }
    if (missing.length && has('startBtn')) $('startBtn').disabled = true;
  } catch (err) {
    if (has('topbarStatusChip')) {
      setText('topbarStatusChip', 'Status unknown');
      $('topbarStatusChip').className = 'chip chip-warn dot';
    }
    setText('systemStatus', `Cannot read system status: ${err.message}`);
  }
}

async function loadModels() {
  try {
    const data = await apiJson('/api/models');
    state.models = data.models || [];
    // Stable 1-based number per model (registry order), surfaced in the model list and
    // referenced compactly in the search bar ("mod N") so a long label never bloats the bar.
    state.models.forEach((m, i) => { m._num = i + 1; });
    renderModelOptions();
    updateModelInfo();
  } catch (err) {
    setText('modelInfo', `Cannot load model registry: ${err.message}`);
  }
}

function renderModelOptions() {
  const select = $('modelKey');
  if (!select) return;
  const previousValue = select.value;
  const models = state.models.filter(m => {
    if (m.supports_image_embedding && m.supports_projection && m.status !== 'planned') return true;
    return state.showPlannedModels && m.status === 'planned';
  });
  select.innerHTML = models.map(m => {
    const disabled = m.status === 'unavailable' || (m.status !== 'planned' && (!m.available || !m.supports_image_embedding || !m.supports_projection));
    const suffix = m.status === 'planned'
      ? ' (planned)'
      : (disabled ? ` (${m.status || 'unavailable'})` : (m.status === 'experimental' ? ' (experimental)' : ''));
    return `<option value="${escapeHtml(m.key)}" ${m.default ? 'selected' : ''} ${disabled ? 'disabled' : ''}>${m._num}. ${escapeHtml(m.label)}${escapeHtml(suffix)}</option>`;
  }).join('');
  if (previousValue && [...select.options].some(option => option.value === previousValue && !option.disabled)) {
    select.value = previousValue;
  }
  renderModelOptionCards(models);
  updateModelOptionSelection();
}

function renderModelOptionCards(models) {
  if (!has('modelOptionList')) return;
  const visibleModels = models.slice(0, 3);
  setHTML('modelOptionList', visibleModels.map(model => {
    const disabled = model.status === 'unavailable' || (model.status !== 'planned' && (!model.available || !model.supports_image_embedding || !model.supports_projection));
    const checked = has('modelKey') && $('modelKey').value === model.key;
    const provider = model.provider || model.family || 'local';
    const dim = model.embedding_dim ? `${model.embedding_dim} dim` : '';
    return `
      <button class="model-opt-card ${checked ? 'selected' : ''}" type="button" data-model-option="${escapeHtml(model.key)}" ${disabled ? 'disabled' : ''}>
        <span class="model-radio" aria-hidden="true"></span>
        <span class="model-opt-copy">
          <strong><span class="model-num">mod ${model._num}</span>${escapeHtml(model.label)}</strong>
          <span>${escapeHtml(provider)}${dim ? ` - ${escapeHtml(dim)}` : ''}</span>
        </span>
      </button>
    `;
  }).join(''));
  document.querySelectorAll('[data-model-option]').forEach(button => {
    button.addEventListener('click', () => {
      if (!has('modelKey')) return;
      $('modelKey').value = button.dataset.modelOption;
      updateModelInfo();
    });
  });
}

function updateModelOptionSelection() {
  if (!has('modelKey')) return;
  document.querySelectorAll('[data-model-option]').forEach(button => {
    button.classList.toggle('selected', button.dataset.modelOption === $('modelKey').value);
  });
}

function updateModelInfo() {
  if (!has('modelKey') || !has('modelInfo')) return;
  const selected = state.models.find(m => m.key === $('modelKey').value);
  if (!selected) { setText('modelInfo', ''); return; }
  const missing = selected.missing_requirements && selected.missing_requirements.length ? `missing: ${selected.missing_requirements.join(', ')}` : '';
  const labels = [
    selected.default ? 'Recommended' : '',
    selected.recommended_for && selected.recommended_for.includes('fast') ? 'Fast' : '',
    selected.status === 'experimental' ? 'Experimental' : '',
    selected.hardware_tier === 'gpu_recommended' || selected.hardware_tier === 'large_gpu' ? 'Requires GPU' : '',
    selected.supports_text_search ? 'Text search supported' : 'Projection only',
  ].filter(Boolean).join(' · ');
  const parts = [selected.family, labels, selected.published ? `published ${selected.published}` : '', selected.provider ? `provider ${selected.provider}` : '', missing, selected.notes || selected.description || ''];
  setText('modelInfo', parts.filter(Boolean).join(' · '));
  setText('workflowModelShort', selected.family || selected.label || 'Model');
  if (has('batchSize') && selected.recommended_batch_size) $('batchSize').value = selected.recommended_batch_size;
  updateModelOptionSelection();
  updateProjectionCapability(selected);
  updateVisualFeaturesAxes(selected);
  updateSemanticAxesVisibility();
  updateSearchCapability();
}

// Visual Features models project onto two fixed, interpretable axes. Show the
// axis pickers (populated from the model's feature_axes) only for that provider
// and hide the embedding-only reducer controls that do not apply.
function updateVisualFeaturesAxes(selected) {
  const wrap = $('visualFeaturesAxes');
  if (!wrap) return;
  const isVf = Boolean(selected && selected.provider === 'visual_features');
  wrap.style.display = isVf ? '' : 'none';
  const reducerBlock = document.querySelector('.projection-seg');
  if (reducerBlock) reducerBlock.style.display = isVf ? 'none' : '';
  if (!isVf) return;
  const axes = (selected.feature_axes && selected.feature_axes.length)
    ? selected.feature_axes
    : ['brightness', 'saturation', 'hue', 'contrast', 'edge_density', 'texture', 'symmetry'];
  for (const [id, fallback] of [['vfXFeature', 'hue'], ['vfYFeature', 'brightness']]) {
    const sel = $(id);
    if (!sel) continue;
    const previous = sel.value;
    sel.innerHTML = axes.map(a => `<option value="${escapeHtml(a)}">${escapeHtml(a)}</option>`).join('');
    sel.value = axes.includes(previous) ? previous : (axes.includes(fallback) ? fallback : axes[0]);
  }
}

function updateProjectionCapability(selected) {
  if (!has('startBtn')) return;
  const canProject = Boolean(
    selected &&
    selected.status !== 'planned' &&
    selected.status !== 'unavailable' &&
    selected.available &&
    selected.supports_image_embedding &&
    selected.supports_projection
  );
  state.modelReady = canProject;
  $('startBtn').disabled = !canProject;
  updateWorkflowState();
  if (canProject || !selected) return;
  if (selected.status === 'planned') {
    setJobText(`${selected.label} is planned and not wired to a verified local loader yet.`);
  } else if (selected.status === 'unavailable') {
    setJobText(`${selected.label} is unavailable: ${selected.notes || 'missing local support.'}`);
  } else if (!selected.available) {
    const missingReqs = selected.missing_requirements && selected.missing_requirements.length ? selected.missing_requirements.join(', ') : 'runtime requirements';
    setJobText(`${selected.label} cannot run until these requirements are available: ${missingReqs}.`);
  } else {
    setJobText(`${selected.label} does not support image projection.`);
  }
}

// Run state machine (CODEX_GUIDE §7): idle → scanned → ready → running → done.
// Drives the left-rail step badges (number → active ink → done green-check). #startBtn
// enablement stays governed by model capability so the backend's own img scan still works
// without forcing an explicit scan first.
function computeRunState() {
  if (state.polling) return 'running';
  if (state.rows.length) return 'done';
  const scanned = Number.isFinite(state.scanCount) && state.scanCount > 0;
  if (scanned && state.modelReady) return 'ready';
  if (scanned) return 'scanned';
  return 'idle';
}

function setStepState(section, num, done, active) {
  if (!section) return;
  section.classList.toggle('done', done);
  section.classList.toggle('active', active);
  const badge = section.querySelector('.num');
  if (badge) {
    badge.className = 'num';
    badge.textContent = done ? '✓' : String(num);
  }
}

function updateWorkflowState() {
  const runState = computeRunState();
  const steps = document.querySelectorAll('.left-rail .step');
  if (steps.length >= 3) {
    const folderDone = ['scanned', 'ready', 'running', 'done'].includes(runState);
    const modelDone = ['ready', 'running', 'done'].includes(runState);
    const projectionDone = runState === 'done';
    setStepState(steps[0], 1, folderDone, !folderDone);
    setStepState(steps[1], 2, modelDone, folderDone && !modelDone);
    setStepState(steps[2], 3, projectionDone, modelDone && !projectionDone);
  }
  document.body.dataset.runState = runState;
}

async function loadProjectionList() {
  if (!has('projectionList')) return;
  try {
    const data = await apiJson('/api/projections');
    if (!data.projections.length) {
      setHTML('projectionList', '<option value="">No saved projections</option>');
      return;
    }
    setHTML('projectionList', '<option value="">Choose saved projection...</option>' + data.projections.map(p => `<option value="${escapeHtml(p.relative_path)}">${escapeHtml(p.name)}</option>`).join(''));
  } catch (err) {
    setHTML('projectionList', '<option value="">Cannot list projections</option>');
  }
}

async function scanImages() {
  setText('imageDirStatus', 'Scanning img folder...');
  try {
    const data = await apiJson('/api/images/scan?image_dir=img');
    if (!data.ok) {
      setText('imageDirStatus', data.error || 'The img folder could not be scanned.');
      return;
    }
    const warning = data.warnings && data.warnings.length ? ` ${data.warnings.join(' ')}` : '';
    state.scanCount = data.count || 0;
    setText('imageDirStatus', `${data.count || 0} image(s) found in img.${warning}`);
    if (has('imageDirStatus')) $('imageDirStatus').className = `status-banner ${state.scanCount > 0 ? 'ok' : 'warn'}`;
    if (has('workflowImageCount')) setText('workflowImageCount', `${state.scanCount} imgs`);
    updateWorkflowState();
  } catch (err) {
    setText('imageDirStatus', `Scan error: ${err.message}`);
  }
}

// A field may hold several phrasings separated by ";": averaging them describes
// a direction more reliably than any single wording, which carries the quirks of
// how that phrase happened to appear in the training captions.
function phraseList(id) {
  const raw = has(id) ? $(id).value : '';
  return String(raw).split(';').map(part => part.trim()).filter(Boolean);
}

function semanticAxisPayload() {
  return {
    semantic_x_positive: phraseList('semanticXPos'),
    semantic_x_negative: phraseList('semanticXNeg'),
    semantic_y_positive: phraseList('semanticYPos'),
    semantic_y_negative: phraseList('semanticYNeg'),
    semantic_x_label: axisLabel('semanticXNeg', 'semanticXPos'),
    semantic_y_label: axisLabel('semanticYNeg', 'semanticYPos'),
  };
}

function axisLabel(negId, posId) {
  const negative = phraseList(negId);
  const positive = phraseList(posId);
  if (!negative.length || !positive.length) return '';
  return `${negative[0]} → ${positive[0]}`;
}

const AXIS_PRESETS = {
  subject: { xNeg: 'a landscape', xPos: 'a portrait of a person', yNeg: 'a single figure', yPos: 'a crowd of people' },
  mood: { xNeg: 'a calm scene', xPos: 'a dramatic scene', yNeg: 'a melancholy mood', yPos: 'a festive scene' },
  setting: { xNeg: 'an indoor scene', xPos: 'an outdoor scene', yNeg: 'night', yPos: 'daylight' },
};

function applyAxisPreset(name) {
  const preset = AXIS_PRESETS[name];
  if (!preset) return;
  if (has('semanticXNeg')) $('semanticXNeg').value = preset.xNeg;
  if (has('semanticXPos')) $('semanticXPos').value = preset.xPos;
  if (has('semanticYNeg')) $('semanticYNeg').value = preset.yNeg;
  if (has('semanticYPos')) $('semanticYPos').value = preset.yPos;
}

// Semantic axes are a projection method, so they appear where UMAP and t-SNE are
// chosen and hide again when another method is picked.
function updateSemanticAxesVisibility() {
  const panel = $('semanticAxes');
  if (!panel) return;
  const active = has('reducer') && $('reducer').value === 'semantic';
  const selected = has('modelKey') ? state.models.find(m => m.key === $('modelKey').value) : null;
  const textCapable = Boolean(selected && selected.supports_text_search);
  panel.hidden = !active;
  if (!active) return;
  if (has('semanticStatus')) {
    setText('semanticStatus', textCapable
      ? 'Ready. The first run loads the model to encode the axis phrases.'
      : `${selected ? selected.label : 'This model'} encodes images only, so it has no text side to build axes from. Pick a CLIP-family model.`);
    $('semanticStatus').className = `status-banner ${textCapable ? 'ok' : 'warn'}`;
  }
  if (has('startBtn') && active && !textCapable) $('startBtn').disabled = true;
}

async function startJob() {
  setBusy(true);
  setHTML('jobLinks', '');
  if (has('downloadBtn')) $('downloadBtn').disabled = true;
  if (has('cancelBtn')) $('cancelBtn').disabled = false;
  state.rows = [];
  clearSearch(false);
  state.resultPath = null;
  // The old grid belongs to the old projection; without this the stale mosaic is
  // still painted behind the empty-state message until the new run finishes.
  invalidateMosaic();
  resetThumbLoader();
  updateGraphLink();
  draw();
  const payload = {
    image_dir: 'img',
    model_key: has('modelKey') ? $('modelKey').value : 'openclip_vit_b_32',
    // Legacy fields retained for v4 compatibility. The backend uses model_key when present.
    model: 'ViT-B-32',
    pretrained: 'laion2b_s34b_b79k',
    reducer: $('reducer').value,
    batch_size: Number($('batchSize').value || 32),
    use_cache: has('useCache') ? $('useCache').checked : true,
    umap_n_neighbors: has('umapNeighbors') ? Number($('umapNeighbors').value || 15) : 15,
    umap_min_dist: has('umapMinDist') ? Number($('umapMinDist').value || 0.1) : 0.1,
    tsne_perplexity: has('tsnePerplexity') ? Number($('tsnePerplexity').value || 30) : 30,
    tsne_max_iter: has('tsneIterations') ? Number($('tsneIterations').value || 1000) : 1000,
    cluster_enabled: has('clusterEnabled') ? $('clusterEnabled').checked : false,
    cluster_auto: has('clusterMode') ? $('clusterMode').value === 'auto' : true,
    cluster_k: has('clusterK') ? Number($('clusterK').value || 5) : 5,
    cluster_min_k: has('clusterMinK') ? Number($('clusterMinK').value || 2) : 2,
    cluster_max_k: has('clusterMaxK') ? Number($('clusterMaxK').value || 8) : 8,
    vf_x_feature: has('vfXFeature') ? $('vfXFeature').value : 'hue',
    vf_y_feature: has('vfYFeature') ? $('vfYFeature').value : 'brightness',
    ...semanticAxisPayload(),
  };
  try {
    const data = await apiJson('/api/jobs/clip-projection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    state.jobId = data.job_id;
    state.polling = true;
    updateWorkflowState();
    pollJob();
  } catch (err) {
    setJobText(`Failed to start job: ${err.message}`);
    setBusy(false);
  }
}

async function cancelJob() {
  if (!state.jobId) return;
  try {
    await apiJson(`/api/jobs/${state.jobId}/cancel`, { method: 'POST' });
    setJobText('Cancellation requested.');
  } catch (err) {
    setJobText(`Cannot cancel job: ${err.message}`);
  }
}

async function pollJob() {
  if (!state.jobId || !state.polling) return;
  try {
    const job = await apiJson(`/api/jobs/${state.jobId}`);
    const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
    if (has('progressBar')) $('progressBar').style.width = `${pct}%`;
    setJobText(`${job.status} · ${job.stage} · ${job.done || 0}/${job.total || 0} · ${job.message || ''}`);
    if (job.status === 'completed') {
      state.polling = false;
      state.resultPath = job.result_path;
      setBusy(false);
      if (has('downloadBtn')) $('downloadBtn').disabled = false;
      if (has('cancelBtn')) $('cancelBtn').disabled = true;
      renderJobDebugLinks(null);
      await loadResult();
      await loadProjectionList();
      return;
    }
    if (job.status === 'failed' || job.status === 'cancelled') {
      state.polling = false;
      setBusy(false);
      if (has('cancelBtn')) $('cancelBtn').disabled = true;
      setJobText(`${job.status} · ${job.recovery_hint || job.error || job.message}`);
      renderJobDebugLinks(job);
      updateWorkflowState();
      return;
    }
    setTimeout(pollJob, 1000);
  } catch (err) {
    state.polling = false;
    setBusy(false);
    setJobText(`Job status error: ${err.message}`);
  }
}

async function loadResult(jobId = state.jobId) {
  if (!jobId) return;
  const data = await apiJson(`/api/jobs/${jobId}/result`);
  ingestRows(data.rows, data.result_path, jobId);
}

async function loadProjectionFile(path) {
  if (!path) return;
  const data = await apiJson(`/api/projections/read?path=${encodeURIComponent(path)}`);
  ingestRows(data.rows, data.result_path, null);
}

async function ingestRows(rows, resultPath, jobId) {
  // Assigned, not merged: loading a saved projection passes null, and keeping the
  // previous run's id made "Download TSV" hand over the wrong file.
  state.jobId = jobId;
  state.resultPath = resultPath;
  state.rows = rows.map(r => ({
    ...r,
    x: Number(r.x),
    y: Number(r.y),
    cluster: parseOptionalNumber(r.cluster),
    cluster_k: parseOptionalNumber(r.cluster_k),
    cluster_score: parseOptionalNumber(r.cluster_score),
  })).filter(r => Number.isFinite(r.x) && Number.isFinite(r.y));
  boundsCache = null;
  invalidateMosaic();
  clearSimilar(false);
  state.axisLabels = null;
  state.scoreColor = null;
  if (has('scoreLegend')) $('scoreLegend').hidden = true;
  state.clusterStats = null;
  resetThumbLoader();
  syncSearchWithRows();
  resetView(false);
  setJobText(`Loaded ${state.rows.length} projected images${resultPath ? ` · ${resultPath}` : ''}.`);
  if (has('downloadBtn')) $('downloadBtn').disabled = !downloadUrl();
  updateRunSummary();
  renderClusterGallery();
  renderPlotLegend();
  updateWorkflowState();
  updateGraphLink();
  draw();
  if (state.mode === 'mosaic') loadMosaic(true).then(draw);
}

// The graph view needs a saved projection to read embeddings from, so its entry
// point stays disabled (with the reason on the tooltip) until one is loaded.
function updateGraphLink() {
  const link = $('openGraphBtn');
  if (!link) return;
  if (!state.resultPath) {
    link.classList.add('is-disabled');
    link.removeAttribute('href');
    link.title = 'Load or generate a projection first.';
    return;
  }
  link.classList.remove('is-disabled');
  link.href = `/graph?projection=${encodeURIComponent(state.resultPath)}`;
  link.title = 'Open the nearest-neighbor graph for this projection';
}

// --- On-demand thumbnail loading -----------------------------------------
// Only rows the current frame draws are fetched, at most MAX_CONCURRENT_THUMB_LOADS
// at a time, and the resident set is capped. Loading every row up front (the pre-v6.0
// behaviour) queued one request and one decoded bitmap per image in the projection,
// most of which the LOD renderer never draws.
const thumbQueue = [];
const thumbQueued = new Set();
const thumbFailed = new Set();
let thumbInFlight = 0;
// Recency is tracked in a side map of frame numbers rather than by re-keying
// state.images: a delete+insert per drawn row per frame churns the Map (hundreds
// of tombstones a second on a dense plot), while overwriting an existing key is
// a plain slot update.
let frameId = 0;
const thumbSeen = new Map();

function touchThumb(key) {
  const img = state.images.get(key);
  if (img === undefined) return undefined;
  thumbSeen.set(key, frameId);
  return img;
}

function evictThumbs() {
  if (state.images.size <= MAX_RESIDENT_THUMBS) return;
  // Drop the least recently drawn, oldest first.
  const byAge = [...state.images.keys()].sort((a, b) => (thumbSeen.get(a) || 0) - (thumbSeen.get(b) || 0));
  const excess = state.images.size - MAX_RESIDENT_THUMBS;
  for (let i = 0; i < excess; i++) {
    state.images.delete(byAge[i]);
    thumbSeen.delete(byAge[i]);
  }
}

function resetThumbLoader() {
  state.images.clear();
  thumbSeen.clear();
  thumbQueue.length = 0;
  thumbQueued.clear();
  thumbFailed.clear();
}

function requestThumb(row) {
  const key = row.relative_path;
  if (state.images.has(key) || thumbQueued.has(key) || thumbFailed.has(key)) return;
  thumbQueued.add(key);
  thumbQueue.push(row);
  pumpThumbQueue();
}

function pumpThumbQueue() {
  while (thumbInFlight < MAX_CONCURRENT_THUMB_LOADS && thumbQueue.length) {
    // Newest request first: the queue is refilled every frame, so the tail holds the
    // rows the user is looking at right now while the head may already be off-screen.
    const row = thumbQueue.pop();
    const key = row.relative_path;
    thumbInFlight += 1;
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => {
      thumbInFlight -= 1;
      thumbQueued.delete(key);
      state.images.set(key, img);
      evictThumbs();
      scheduleDraw();
      pumpThumbQueue();
    };
    img.onerror = () => {
      thumbInFlight -= 1;
      thumbQueued.delete(key);
      thumbFailed.add(key);
      pumpThumbQueue();
    };
    img.src = thumbUrl(row, CANVAS_THUMB_WIDTH);
  }
}

function resizeCanvas() {
  const rect = refreshPlotRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

// The grid's aspect follows the canvas, so a resize means a different mosaic.
let mosaicResizeTimer = null;
function scheduleMosaicRefresh() {
  if (state.mode !== 'mosaic' || !state.resultPath || state.mosaicColumns) return;
  if (mosaicResizeTimer) clearTimeout(mosaicResizeTimer);
  mosaicResizeTimer = setTimeout(() => {
    mosaicResizeTimer = null;
    loadMosaic(true).then(draw);
  }, 400);
}

function scheduleResizeCanvas() {
  if (resizeFrame) cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = null;
    resizeCanvas();
    scheduleMosaicRefresh();
  });
}

function bounds() {
  if (boundsCache) return boundsCache;
  if (!state.rows.length) {
    boundsCache = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    return boundsCache;
  }
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const r of state.rows) {
    if (Number.isFinite(r.x)) { if (r.x < minX) minX = r.x; if (r.x > maxX) maxX = r.x; }
    if (Number.isFinite(r.y)) { if (r.y < minY) minY = r.y; if (r.y > maxY) maxY = r.y; }
  }
  boundsCache = { minX, maxX, minY, maxY };
  return boundsCache;
}

// Single definition of the on-plot row set. draw() and findHover() used to filter
// separately and had drifted: hover still matched rows that the cluster filter had
// removed from the canvas, so the tooltip could describe an invisible image.
function visibleRows() {
  let rows = state.rows;
  if (state.showOnlySearch && state.searchResultPaths.size) {
    rows = rows.filter(row => state.searchResultPaths.has(row.relative_path));
  }
  if (state.showOnlyCluster && state.selectedCluster) {
    rows = rows.filter(row => String(row.cluster) === String(state.selectedCluster));
  }
  return rows;
}

// --- Grid mosaic ---------------------------------------------------------
// A scatter of thumbnails hides its own dense regions: images overlap, and the
// space an image occupies carries no meaning. The mosaic asks the backend to
// assign each image to a distinct cell of a rectangular grid (optimal linear
// assignment for small sets, recursive median split for large ones) so nothing
// overlaps, every work gets equal area, and neighbours stay neighbours.
const MOSAIC_CELL_GAP = 2;
// Below this drawn size a thumbnail conveys nothing, so the mosaic falls back to
// colored cells. The second, harder limit is the resident-thumbnail cap: a large
// mosaic can put every cell on screen at once, and asking for more bitmaps than
// the cache holds makes every frame evict what the next frame requests again —
// an endless load loop. Requiring the cell to be big enough that the visible
// cells fit in the cache turns that into ordinary level-of-detail behaviour.
const MOSAIC_MIN_THUMB_PX = 12;

function mosaicGeometry() {
  const rect = currentRect();
  const cols = Math.max(1, state.mosaic ? state.mosaic.columns : 1);
  const rows = Math.max(1, state.mosaic ? state.mosaic.rows : 1);
  const availWidth = Math.max(1, rect.width - 120);
  const availHeight = Math.max(1, rect.height - 104);
  const cell = Math.min(availWidth / cols, availHeight / rows);
  return {
    cols,
    rows,
    cell,
    originX: 60 + (availWidth - cell * cols) / 2,
    originY: 42 + (availHeight - cell * rows) / 2,
  };
}

function mosaicActive() {
  return state.mode === 'mosaic' && state.mosaic && state.mosaicCells.size > 0;
}

// True when the mosaic's cells are large enough to be worth fetching images for.
function mosaicThumbsWorthwhile() {
  const rect = currentRect();
  const drawn = currentThumbSize();
  if (drawn < MOSAIC_MIN_THUMB_PX) return false;
  const area = Math.max(1, rect.width) * Math.max(1, rect.height);
  const cellArea = Math.max(1, drawn * drawn);
  return (area / cellArea) <= MAX_RESIDENT_THUMBS;
}

function baseProjectMosaic(row) {
  const geometry = mosaicGeometry();
  const cell = state.mosaicCells.get(row.relative_path);
  if (!cell) return { x: -9999, y: -9999 };
  return {
    x: geometry.originX + (cell.col + 0.5) * geometry.cell,
    y: geometry.originY + (cell.row + 0.5) * geometry.cell,
  };
}

// Thumbnails fill their cell in mosaic mode; elsewhere the user's size slider wins.
function currentThumbSize() {
  if (!mosaicActive()) return state.thumbSize;
  return Math.max(8, (mosaicGeometry().cell * state.zoom) - MOSAIC_CELL_GAP);
}

// Cluster frames and focus rings are drawn *outside* the thumbnail. In the scatter
// there is empty space for them; in the mosaic the cells are touching, so an
// unclamped expansion draws over the neighbouring images. Shrink the decorations
// to whatever gap the current cell size actually leaves.
function decorationExpand(preferred) {
  if (!mosaicActive()) return preferred;
  return Math.max(0, Math.min(preferred, MOSAIC_CELL_GAP));
}

async function loadMosaic(force = false) {
  if (!state.resultPath) {
    setJobText('Load or generate a projection before switching to the mosaic view.');
    return false;
  }
  if (state.mosaic && !force) return true;
  const rect = currentRect();
  state.mosaicBusy = true;
  setText('mosaicStatus', 'Computing mosaic layout...');
  try {
    const data = await apiJson('/api/views/mosaic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        projection_path: state.resultPath,
        columns: state.mosaicColumns || null,
        // The grid follows the plot's shape so a wide canvas gets a wide mosaic.
        aspect: Math.max(0.1, Math.min(20, (rect.width || 1) / (rect.height || 1))),
        algorithm: state.mosaicAlgorithm,
      }),
    });
    state.mosaic = data;
    state.mosaicCells = new Map(data.cells.map(cell => [cell.relative_path, cell]));
    const detail = data.algorithm === 'optimal' ? 'optimal assignment' : 'recursive split';
    setText('mosaicStatus', `${data.columns} x ${data.rows} grid - ${data.count} images - ${detail}${data.note ? ` - ${data.note}` : ''}`);
    return true;
  } catch (err) {
    state.mosaic = null;
    state.mosaicCells = new Map();
    setText('mosaicStatus', `Mosaic failed: ${err.message}`);
    setJobText(`Mosaic failed: ${err.message}`);
    return false;
  } finally {
    state.mosaicBusy = false;
    updateMosaicControls();
  }
}

function invalidateMosaic() {
  state.mosaic = null;
  state.mosaicCells = new Map();
  if (has('mosaicStatus')) setText('mosaicStatus', 'Not computed yet.');
}

function updateMosaicControls() {
  const wrap = $('mosaicControls');
  if (wrap) wrap.hidden = state.mode !== 'mosaic';
  if (has('mosaicRecomputeBtn')) $('mosaicRecomputeBtn').disabled = state.mosaicBusy || !state.resultPath;
  if (has('mosaicAlgorithm')) $('mosaicAlgorithm').value = state.mosaicAlgorithm;
}

async function setViewMode(mode) {
  if (mode === state.mode) return;
  const previous = state.mode;
  state.mode = mode;
  if (mode === 'mosaic') {
    const ok = await loadMosaic();
    if (!ok) {
      state.mode = previous;
      updateModeButton();
      updateMosaicControls();
      draw();
      return;
    }
    // The mosaic is its own coordinate system; a pan carried over from the
    // scatter would drop the user somewhere outside the grid.
    resetView(false);
  }
  updateModeButton();
  updateMosaicControls();
  draw();
}

function updateModeButton() {
  if (!has('toggleModeBtn')) return;
  const labels = { images: 'img', points: 'pts', mosaic: 'grid' };
  $('toggleModeBtn').textContent = labels[state.mode] || 'img';
  $('toggleModeBtn').title = `View mode: ${state.mode} (click to cycle)`;
  document.querySelectorAll('[data-view-mode]').forEach(button => {
    button.classList.toggle('active', button.dataset.viewMode === state.mode);
  });
}

function baseProject(row) {
  if (mosaicActive()) return baseProjectMosaic(row);
  const rect = currentRect();
  const b = bounds();
  const pad = 0.08;
  const rangeX = b.maxX - b.minX || 1;
  const rangeY = b.maxY - b.minY || 1;
  const x = 72 + ((row.x - b.minX + rangeX * pad) / (rangeX * (1 + pad * 2))) * (rect.width - 144);
  const y = rect.height - 62 - ((row.y - b.minY + rangeY * pad) / (rangeY * (1 + pad * 2))) * (rect.height - 122);
  return { x, y };
}

function project(row) {
  const rect = currentRect();
  const p = baseProject(row);
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  return { x: cx + (p.x - cx) * state.zoom + state.panX, y: cy + (p.y - cy) * state.zoom + state.panY };
}

function draw() {
  frameId += 1;
  updatePlotStatus();
  const rect = refreshPlotRect();
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalAlpha = 1;
  ctx.fillStyle = '#f6f4ef';
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (mosaicActive()) drawMosaicGrid(rect); else drawGrid(rect);

  if (!state.rows.length) {
    ctx.fillStyle = '#15171a';
    ctx.font = '15px Geist, Arial';
    ctx.fillText('Add images to img, then generate an embedding projection.', 82, 82);
    ctx.fillStyle = '#7A7C85';
    ctx.font = '12px Geist, Arial';
    ctx.fillText('Images are read from the project img folder. The final plot remains interactive.', 82, 106);
    return;
  }

  const visible = visibleRows();

  // LOD: on large datasets, draw cheap colored points when zoomed out and switch to
  // thumbnails only when zoomed in (where few points are visible). Small datasets keep
  // thumbnails at every zoom. Point mode (toggled by the user) always draws points.
  const inMosaic = mosaicActive();
  const thumbSize = currentThumbSize();
  // In the mosaic every image already has its own cell, so there is nothing to
  // thin out: LOD only applies to the overlapping scatter.
  const lodActive = !inMosaic && state.rows.length > LOD_THUMB_MIN_ROWS;
  const showThumbs = inMosaic
    ? mosaicThumbsWorthwhile()
    : (state.mode === 'images' && (!lodActive || state.zoom >= LOD_THUMB_ZOOM));
  const cullMargin = (showThumbs ? thumbSize : state.pointSize * 2) + 24;
  for (const row of visible) {
    const p = project(row);
    // Viewport culling: skip points outside the canvas (+margin). Invisible to the user,
    // but at high zoom/pan it avoids drawing the off-screen majority of the dataset.
    if (p.x < -cullMargin || p.x > rect.width + cullMargin || p.y < -cullMargin || p.y > rect.height + cullMargin) continue;
    const clusterColor = getMarkColor(row);
    const isSearchResult = state.searchResultPaths.has(row.relative_path);
    const isSelectedCluster = state.selectedCluster && String(row.cluster) === String(state.selectedCluster);
    if (state.highlightSearch && state.searchResultPaths.size && !isSearchResult) ctx.globalAlpha = 0.22;
    if (state.selectedCluster && !isSelectedCluster) ctx.globalAlpha = Math.min(ctx.globalAlpha, state.showOnlyCluster ? 0.12 : 0.30);
    if (showThumbs) {
      const img = touchThumb(row.relative_path);
      if (img === undefined) requestThumb(row);
      if (clusterColor && state.showThumbnailOutline) drawClusterFrame(p.x, p.y, clusterColor);
      if (img) drawImageThumb(img, p.x, p.y, clusterColor);
      else drawPointBox(p.x, p.y, clusterColor);
    } else {
      drawPointBox(p.x, p.y, clusterColor);
    }
    ctx.globalAlpha = 1;
    if (state.highlightSearch && isSearchResult) drawSearchRing(p.x, p.y);
    if (state.highlightSimilar && state.similarPaths.has(row.relative_path)) {
      drawSimilarRing(p.x, p.y, row.relative_path === state.similarQuery?.relative_path);
    }
    if (state.selectedCluster && isSelectedCluster && state.highlightCluster) drawClusterFocusRing(p.x, p.y, clusterColor || '#ffffff');
  }

  if (!inMosaic) drawLabels(rect);
  if (state.hover) drawTooltip(state.hover);
}

function drawMosaicGrid(rect) {
  ctx.save();
  ctx.strokeStyle = 'rgba(21,23,26,0.10)';
  ctx.lineWidth = 1;
  const geometry = mosaicGeometry();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const scaled = geometry.cell * state.zoom;
  const left = cx + (geometry.originX - cx) * state.zoom + state.panX;
  const top = cy + (geometry.originY - cy) * state.zoom + state.panY;
  ctx.strokeRect(left, top, scaled * geometry.cols, scaled * geometry.rows);
  // Cell guides are only legible, and only useful, when the cells are big enough.
  if (scaled >= 14) {
    ctx.strokeStyle = 'rgba(21,23,26,0.05)';
    for (let col = 1; col < geometry.cols; col++) {
      const x = left + scaled * col;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + scaled * geometry.rows); ctx.stroke();
    }
    for (let row = 1; row < geometry.rows; row++) {
      const y = top + scaled * row;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + scaled * geometry.cols, y); ctx.stroke();
    }
  }
  ctx.restore();
}

function drawGrid(rect) {
  ctx.strokeStyle = 'rgba(21,23,26,0.10)';
  ctx.lineWidth = 1;
  ctx.strokeRect(60, 42, rect.width - 120, rect.height - 104);
  ctx.strokeStyle = 'rgba(21,23,26,0.045)';
  for (let i = 1; i < 6; i++) {
    const x = 60 + (rect.width - 120) * i / 6;
    const y = 42 + (rect.height - 104) * i / 6;
    ctx.beginPath(); ctx.moveTo(x, 42); ctx.lineTo(x, rect.height - 62); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(rect.width - 60, y); ctx.stroke();
  }
}

// A semantic projection carries its axis names in every row; showing them is the
// entire point of the feature, so they replace the generic labels when present.
function axisTitles() {
  const preview = state.axisLabels;
  if (preview && (preview.x || preview.y)) {
    return { x: preview.x || 'Projection X', y: preview.y || 'Projection Y' };
  }
  const row = state.rows[0];
  if (row && (row.semantic_x_label || row.semantic_y_label)) {
    return { x: row.semantic_x_label || 'Projection X', y: row.semantic_y_label || 'Projection Y' };
  }
  return { x: 'Projection X', y: 'Projection Y' };
}

function drawLabels(rect) {
  const titles = axisTitles();
  ctx.fillStyle = 'rgba(21,23,26,0.55)';
  ctx.font = '12px Geist, Arial';
  ctx.textAlign = 'center';
  ctx.fillText(titles.x, rect.width / 2, rect.height - 24);
  ctx.save();
  ctx.translate(24, rect.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText(titles.y, 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';
}

function drawImageThumb(img, x, y, color = null) {
  const box = thumbnailBox(x, y, currentThumbSize());
  ctx.save();
  // The backing card fades with the thumbnail. Held at a fixed 0.78 it turned a
  // dimmed thumbnail into a white card instead of a transparent one, so lowering
  // the slider washed the plot out rather than letting it show through.
  ctx.fillStyle = `rgba(255,255,255,${(0.78 * state.thumbOpacity).toFixed(3)})`;
  roundedRect(box.x, box.y, box.size, box.size, box.radius);
  ctx.fill();
  ctx.clip();

  const naturalW = img.naturalWidth || 1;
  const naturalH = img.naturalHeight || 1;
  const aspect = naturalW / naturalH;
  const inner = box.size - 4;
  const w = aspect >= 1 ? inner : inner * aspect;
  const h = aspect >= 1 ? inner / aspect : inner;
  // Multiplied into the alpha already set by the search/cluster dimming, so the two
  // effects compose instead of one overwriting the other.
  const previousAlpha = ctx.globalAlpha;
  ctx.globalAlpha = previousAlpha * state.thumbOpacity;
  ctx.drawImage(img, x - w / 2, y - h / 2, w, h);
  ctx.globalAlpha = previousAlpha;
  ctx.restore();

  ctx.save();
  roundedRect(box.x, box.y, box.size, box.size, box.radius);
  ctx.strokeStyle = color || 'rgba(21,23,26,0.26)';
  ctx.lineWidth = color ? 2 : 1;
  ctx.stroke();
  ctx.restore();
}

function drawTooltip(row) {
  const p = project(row);
  const img = state.images.get(row.relative_path);
  const preview = 160;
  const pad = 12;
  let w = 280;
  let h = img ? 252 : 105;
  // The cached rect: measuring here ran a forced layout on every hover frame,
  // which is exactly what plotRect exists to avoid.
  const rect = currentRect();
  let x = p.x + 18;
  let y = p.y - h - 10;
  if (x + w > rect.width - 10) x = p.x - w - 18;
  if (y < 10) y = p.y + 18;
  if (x < 10) x = 10;
  ctx.fillStyle = '#202020';
  ctx.shadowColor = 'rgba(0,0,0,.45)';
  ctx.shadowBlur = 20;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, 10); ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(255,255,255,.18)'; ctx.stroke();
  let textY = y + pad + 14;
  if (img) {
    const aspect = img.naturalWidth / img.naturalHeight;
    const iw = aspect >= 1 ? preview : preview * aspect;
    const ih = aspect >= 1 ? preview / aspect : preview;
    ctx.drawImage(img, x + (w - iw) / 2, y + pad, iw, ih);
    textY = y + pad + ih + 24;
  }
  ctx.fillStyle = '#fff'; ctx.font = '12px Arial';
  ctx.fillText(row.filename, x + pad, textY);
  ctx.fillStyle = '#bbb'; ctx.font = '11px Arial';
  ctx.fillText(`x ${row.x.toFixed(3)} · y ${row.y.toFixed(3)}`, x + pad, textY + 17);
  ctx.fillText(`${row.reducer || ''} · ${row.embedding_model || row.model_family || ''}`, x + pad, textY + 34);
  ctx.fillStyle = '#8a8a8a';
  ctx.fillText('double-click for similar images', x + pad, textY + 51);
  ctx.fillStyle = '#bbb';
  const searchResult = state.searchResults.find(result => result.relative_path === row.relative_path);
  if (searchResult) ctx.fillText(`score ${Number(searchResult.score).toFixed(4)}`, x + pad, textY + 68);
  if (hasCluster(row)) ctx.fillText(`cluster ${row.cluster}${hasClusterK(row) ? ` / ${row.cluster_k}` : ''}`, x + pad, textY + (searchResult ? 85 : 68));
}

// --- Continuous score colouring -------------------------------------------
// A blue-to-warm ramp over the score's own range. The range comes from the
// data's 2nd and 98th percentiles rather than its extremes: cosine scores sit in
// a narrow band, so a ramp stretched over the theoretical -1..1 would render the
// whole collection one flat colour.
function scoreRampColor(value) {
  const ramp = state.scoreColor;
  if (!ramp) return null;
  const span = ramp.high - ramp.low;
  const t = span > 1e-9 ? Math.max(0, Math.min(1, (value - ramp.low) / span)) : 0.5;
  const hue = 215 - 195 * t;            // blue -> amber -> red
  const light = 62 - 14 * t;
  return `hsl(${hue.toFixed(0)}, 78%, ${light.toFixed(0)}%)`;
}

function getScoreColor(row) {
  if (!state.scoreColor) return undefined;
  const value = state.scoreColor.byPath.get(row.relative_path);
  // An image with no score is grey rather than invisible: the absence is data.
  if (!hasNumber(value)) return '#B8B7B0';
  return scoreRampColor(Number(value));
}

// The plot colours by score when one is loaded, by cluster otherwise.
function getMarkColor(row) {
  const score = getScoreColor(row);
  return score !== undefined ? score : getClusterColor(row);
}

function getClusterColor(row) {
  if (!hasCluster(row)) return null;
  const idx = Math.max(0, Number(row.cluster) - 1) % CLUSTER_PALETTE.length;
  return CLUSTER_PALETTE[idx];
}

// ingestRows stores a missing cluster as null, and Number(null) is 0 - a finite
// number. Testing finiteness alone therefore reported every unclustered image as
// belonging to cluster 0: the gallery invented a group, auto-selected it, painted
// a blue frame on every thumbnail, and then dimmed them all to 30% because
// String(null) never equals "0". The null has to be rejected explicitly.
function hasNumber(value) {
  if (value === null || value === undefined || value === '') return false;
  return Number.isFinite(Number(value));
}

function hasCluster(row) {
  return Boolean(row && hasNumber(row.cluster));
}

function hasClusterK(row) {
  return Boolean(row && hasNumber(row.cluster_k));
}

function thumbnailBox(x, y, sizeOverride = null, expand = 0) {
  const base = Number(sizeOverride) || currentThumbSize();
  const size = Math.max(16, base) + expand;
  return {
    x: x - size / 2,
    y: y - size / 2,
    size,
    radius: Math.max(5, Math.min(10, size * 0.16)),
  };
}

function roundedRect(x, y, width, height, radius) {
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(x, y, width, height, radius);
    return;
  }
  const r = Math.min(radius, width / 2, height / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

function drawClusterFrame(x, y, color) {
  const box = thumbnailBox(x, y, currentThumbSize(), decorationExpand(8));
  if (mosaicActive() && box.size < 18) return;  // no room for a legible frame
  ctx.save();
  roundedRect(box.x, box.y, box.size, box.size, box.radius);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.14;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawSearchRing(x, y) {
  ctx.save();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  const box = thumbnailBox(x, y, currentThumbSize(), decorationExpand(12));
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
  ctx.fill();
  ctx.restore();
}

// A solid accent ring, visually distinct from the white search ring, so a text
// search and a similarity set can be on screen at once without ambiguity. The
// query image itself gets a heavier, filled marker.
const SIMILAR_RING_COLOR = '#2A6FDB';

function drawSimilarRing(x, y, isQuery) {
  ctx.save();
  const box = thumbnailBox(x, y, currentThumbSize(), decorationExpand(isQuery ? 14 : 10));
  ctx.strokeStyle = SIMILAR_RING_COLOR;
  ctx.lineWidth = isQuery ? 4 : 2.5;
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
  ctx.stroke();
  if (isQuery) {
    ctx.fillStyle = 'rgba(42,111,219,0.18)';
    roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawClusterFocusRing(x, y, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 5]);
  const box = thumbnailBox(x, y, currentThumbSize(), decorationExpand(18));
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 3);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

function setScoreColor(ramp) {
  state.scoreColor = ramp;
  renderScoreLegend();
  renderPlotLegend();
  draw();
}

function clearScoreColor() {
  setScoreColor(null);
  if (has('conceptStatus')) setText('conceptStatus', 'Colouring by cluster.');
}

function renderScoreLegend() {
  const legend = $('scoreLegend');
  if (!legend) return;
  const ramp = state.scoreColor;
  legend.hidden = !ramp;
  if (!ramp) return;
  setText('scoreLegendLabel', ramp.label);
  const low = $('scoreLegendLow');
  const high = $('scoreLegendHigh');
  if (low) { low.style.background = scoreRampColor(ramp.low); low.textContent = ramp.low.toFixed(3); }
  if (high) { high.style.background = scoreRampColor(ramp.high); high.textContent = ramp.high.toFixed(3); }
}

function renderPlotLegend() {
  if (!has('plotLegend')) return;
  const groups = clusterGroups();
  if (state.scoreColor || !groups.length) {
    $('plotLegend').hidden = true;
    setHTML('plotLegend', '');
    return;
  }
  $('plotLegend').hidden = false;
  setHTML('plotLegend', [
    '<span class="plot-legend-title">Clusters</span>',
    ...groups.slice(0, 8).map(([cluster, rows]) => {
      const color = CLUSTER_PALETTE[(Number(cluster) - 1) % CLUSTER_PALETTE.length];
      return `<span class="plot-legend-item"><i style="background:${color}"></i>Cluster ${escapeHtml(cluster)}<b>${rows.length}</b></span>`;
    })
  ].join(''));
}

function updatePlotStatus() {
  const stats = document.querySelectorAll('.plot-status .ps-stat .v');
  if (stats[0]) stats[0].textContent = state.rows.length ? String(state.rows.length) : 'canvas';
  if (stats[1]) stats[1].textContent = state.mode;
  if (stats[2]) stats[2].textContent = `${state.zoom.toFixed(2)}x`;
}

function updateRunSummary() {
  const values = document.querySelectorAll('.gen-summary .gv');
  if (values[0]) values[0].textContent = state.rows.length ? String(state.rows.length) : 'img';
  if (values[1] && has('modelKey')) {
    const selected = state.models.find(m => m.key === $('modelKey').value);
    values[1].textContent = selected?.family || selected?.label || 'selected';
  }
  if (values[2] && has('reducer')) values[2].textContent = $('reducer').value.toUpperCase();
  if (values[3]) values[3].textContent = clusterGroups().length ? `k=${clusterGroups().length}` : 'optional';
  setText('workflowImageCount', state.rows.length ? `${state.rows.length} imgs` : 'img');
  if (has('systemStatus')) setText('systemStatus', state.rows.length ? `${state.rows.length} imgs` : '0 imgs');
}

function drawPointBox(x, y, color = null) {
  const size = Math.max(10, state.pointSize * 2.8);
  const box = thumbnailBox(x, y, size, 0);
  ctx.save();
  roundedRect(box.x, box.y, box.size, box.size, 4);
  ctx.fillStyle = color ? `${color}33` : '#eee';
  ctx.fill();
  ctx.strokeStyle = color || 'rgba(0,0,0,0.28)';
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.restore();
}

function resetView(redraw = true) {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  state.dragging = false;
  state.dragStart = null;
  state.hover = null;
  if (redraw) draw();
}

// "Fit" used to be an alias for "Reset", which is only correct when nothing is
// filtered. It now frames the rows currently on the plot, so fitting after a
// cluster or search filter zooms to that subset instead of the whole dataset.
function fitView(redraw = true) {
  const rows = visibleRows();
  if (!rows.length) { resetView(redraw); return; }

  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  const rect = refreshPlotRect();

  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const row of rows) {
    const p = baseProject(row);
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }

  // Leave room for the thumbnail drawn around each extreme point, plus a margin.
  // currentThumbSize() follows the mode, so a mosaic is framed by its cell size
  // rather than by the (irrelevant) thumbnail slider.
  const halfMark = (state.mode === 'points' ? state.pointSize * 2 : currentThumbSize() / 2) + 16;
  const spanX = (maxX - minX) + halfMark * 2;
  const spanY = (maxY - minY) + halfMark * 2;
  const zoom = Math.max(0.2, Math.min(12, Math.min(rect.width / Math.max(1, spanX), rect.height / Math.max(1, spanY))));

  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  state.zoom = zoom;
  state.panX = cx - (cx + (midX - cx) * zoom);
  state.panY = cy - (cy + (midY - cy) * zoom);
  state.hover = null;
  if (redraw) draw();
}

function syncReducerVisuals() {
  if (!has('reducer')) return;
  const value = $('reducer').value;
  setText('workflowReducerShort', value === 'semantic' ? 'NAMED' : value.toUpperCase());
  updateSemanticAxesVisibility();
  document.querySelectorAll('[data-reducer-choice]').forEach(button => {
    button.classList.toggle('active', button.dataset.reducerChoice === value);
  });
  updateRunSummary();
}

function findHover(evt) {
  if (!state.rows.length) return null;
  const rect = refreshPlotRect();
  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;
  let best = null;
  let bestD = Math.max(22, currentThumbSize() * 0.55);
  for (const row of visibleRows()) {
    const p = project(row);
    const d = Math.hypot(mx - p.x, my - p.y);
    if (d < bestD) { bestD = d; best = row; }
  }
  return best;
}

function exportPng() {
  const link = document.createElement('a');
  link.download = 'imagecluster-clip-projection.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function updateSearchCapability() {
  if (!has('modelKey') || !has('searchCapability')) return;
  const selected = state.models.find(m => m.key === $('modelKey').value);
  if (!selected) return;
  // Capability (can this model search by text at all?) is separate from availability
  // (are its runtime deps installed / embeddings generated?). CODEX_GUIDE §4.3: the bar is
  // disabled only when the model lacks the text-search capability flag; otherwise it stays
  // usable and we surface the availability reason inline.
  const supportsText = Boolean(selected.supports_text_search);
  const searchable = Boolean(supportsText && selected.available);
  // The bar shows a compact indicator ("mod N · <status>") so a long model label can't make
  // the search field unusable; the full sentence lives in the title tooltip.
  let label;
  let tone;
  let full;
  if (searchable) {
    label = 'text ✓';
    tone = 'cap-ok';
    full = `${selected.label} can search cached image embeddings with text queries.`;
  } else if (!supportsText) {
    label = 'no text';
    tone = 'cap-danger';
    full = `${selected.label} does not support text search. Pick a CLIP-family model to search by text.`;
  } else {
    const missingReqs = selected.missing_requirements && selected.missing_requirements.length
      ? selected.missing_requirements.join(', ')
      : 'runtime requirements';
    label = 'setup';
    tone = 'cap-warn';
    full = `${selected.label} supports text search once ${missingReqs} are installed and embeddings are generated.`;
  }
  const ref = selected._num ? `mod ${selected._num}` : 'model';
  setText('searchCapability', `${ref} · ${label}`);
  $('searchCapability').className = `sb-meta ${tone}`;
  $('searchCapability').title = full;
  if (has('searchQuery')) {
    $('searchQuery').disabled = !supportsText;
    $('searchQuery').placeholder = supportsText
      ? "Describe what you're looking for - e.g. gold background Madonna and Child"
      : 'Text search is not available for this model.';
  }
  if (has('semanticSearchBar')) $('semanticSearchBar').classList.toggle('is-disabled', !supportsText);
  if (has('searchBtn')) $('searchBtn').disabled = !searchable;
  if (has('buildIndexBtn')) $('buildIndexBtn').disabled = !searchable;
}

async function runSemanticSearch() {
  const query = has('searchQuery') ? $('searchQuery').value.trim() : '';
  if (!query) {
    setSearchError('Enter a search query.');
    return;
  }
  setSearchBusy(true);
  setSearchError('');
  const thresholdRaw = has('searchThreshold') ? $('searchThreshold').value.trim() : '';
  const payload = {
    query,
    model_key: has('modelKey') ? $('modelKey').value : 'openclip_vit_b_32',
    image_dir: 'img',
    embedding_id: null,
    top_k: has('searchTopK') ? Number($('searchTopK').value || 30) : 30,
    threshold: thresholdRaw === '' ? null : Number(thresholdRaw),
    normalize: true,
  };
  try {
    const data = await apiJson('/api/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    state.lastSearchPayload = data;
    state.searchResults = data.results || [];
    state.searchResultPaths = new Set(state.searchResults.map(r => r.relative_path));
    renderSearchResults();
    selectDockPane('search', true);
    syncSearchWithRows();
    if (has('searchStatus')) $('searchStatus').textContent = `${state.searchResults.length} result(s) · ${data.embedding_id}`;
    if (has('exportSearchBtn')) $('exportSearchBtn').disabled = !state.searchResults.length;
    draw();
  } catch (err) {
    setSearchError(err.message);
    if (has('searchStatus')) $('searchStatus').textContent = 'Search failed.';
  } finally {
    setSearchBusy(false);
  }
}

function renderSearchResults() {
  if (!has('searchResults')) return;
  if (!state.searchResults.length) {
    setHTML('searchResults', '');
    return;
  }
  $('searchResults').style.setProperty('--search-thumb-size', `${state.searchThumbSize}px`);
  setHTML('searchResults', state.searchResults.map(result => `
    <article class="search-card" data-path="${escapeHtml(result.relative_path)}">
      <img src="${escapeHtml(thumbUrl(result, Math.max(160, state.searchThumbSize * 1.5)))}" alt="" loading="lazy" />
      <div class="search-meta">
        <span class="search-rank">Rank: ${result.rank}</span>
        <span class="search-score">Score: ${Number(result.score).toFixed(4)}</span>
        ${hasNumber(result.cluster) ? `<span class="search-cluster">Cluster: ${Number(result.cluster)}</span>` : ''}
      </div>
      <div class="search-title">${escapeHtml(result.filename)}</div>
      <a class="search-open-link" href="${escapeHtml(imageUrl(result))}" target="_blank" rel="noopener">Open image</a>
    </article>
  `).join(''));
  document.querySelectorAll('.search-card').forEach(card => {
    card.addEventListener('click', event => {
      if (event.target.closest('a')) return;
      focusSearchResult(card.dataset.path);
    });
    card.addEventListener('dblclick', () => {
      const result = state.searchResults.find(item => item.relative_path === card.dataset.path);
      if (result) openImagePreview(result);
    });
  });
}

function clusterGroups() {
  const groups = new Map();
  for (const row of state.rows) {
    if (!hasCluster(row)) continue;
    const key = String(Number(row.cluster));
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(row);
  }
  return [...groups.entries()].sort((a, b) => Number(a[0]) - Number(b[0]));
}

function renderClusterGallery() {
  if (!has('clusterGallery') || !has('clusterSelect') || !has('clusterSummary')) return;
  const groups = clusterGroups();
  const clusterSelect = $('clusterSelect');
  const currentValue = clusterSelect.value || state.selectedCluster || '';
  if (!groups.length) {
    state.selectedCluster = '';
    clusterSelect.innerHTML = '<option value="">No clusters yet</option>';
    setText('clusterSummary', 'Enable clustering to browse the images inside each cluster.');
    setHTML('clusterGallery', '');
    if (has('showOnlyClusterToggle')) $('showOnlyClusterToggle').checked = false;
    if (has('highlightClusterToggle')) $('highlightClusterToggle').checked = true;
    hideElement('clusterPanel');
    renderPlotLegend();
    draw();
    return;
  }

  showElement('clusterPanel', 'block');
  selectDockPane('cluster');
  const options = groups.map(([cluster, rows]) => `<option value="${escapeHtml(cluster)}">Cluster ${escapeHtml(cluster)} (${rows.length})</option>`);
  clusterSelect.innerHTML = options.join('');
  const preferred = groups.some(([cluster]) => cluster === currentValue) ? currentValue : groups[0][0];
  clusterSelect.value = preferred;
  state.selectedCluster = preferred;
  setText('clusterSummary', `${groups.length} cluster(s) available · ${state.rows.filter(hasCluster).length} labelled image(s).`);
  renderClusterCards();
  draw();
}

async function exportClusters(cluster) {
  if (!has('clusterSummary')) return;
  if (!state.resultPath) {
    setText('clusterSummary', 'Save or load a projection before exporting clusters.');
    return;
  }
  if (!clusterGroups().length) {
    setText('clusterSummary', 'This projection has no clusters. Re-run it with clustering enabled.');
    return;
  }
  const scope = cluster === 'all' ? 'all clusters' : `cluster ${cluster}`;
  setText('clusterSummary', `Exporting ${scope}…`);
  try {
    const data = await apiJson('/api/clusters/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projection_path: state.resultPath, cluster: String(cluster) })
    });
    const folders = Object.keys(data.folders || {}).length;
    const skipped = data.skipped_count ? ` · ${data.skipped_count} skipped` : '';
    setText('clusterSummary', `Exported ${data.exported_count} image(s) into ${folders} folder(s) → ${data.dest}${skipped}`);
  } catch (err) {
    setText('clusterSummary', `Export failed: ${err.message}`);
  }
}

function renderClusterCards() {
  if (!has('clusterGallery') || !state.selectedCluster) {
    setHTML('clusterGallery', '');
    return;
  }
  const rows = state.rows
    .filter(row => String(row.cluster) === String(state.selectedCluster))
    .sort((a, b) => (Number(b.cluster_score || 0) - Number(a.cluster_score || 0)) || String(a.filename).localeCompare(String(b.filename)));
  if (!rows.length) {
    setHTML('clusterGallery', '<div class="empty-state"><h3>No images in this cluster</h3><p>Try selecting another cluster or rerun the projection with clustering enabled.</p></div>');
    return;
  }
  $('clusterGallery').style.setProperty('--search-thumb-size', `${state.clusterThumbSize}px`);
  setHTML('clusterGallery', rows.map(row => `
    <article class="search-card cluster-card" data-path="${escapeHtml(row.relative_path)}">
      <img src="${escapeHtml(thumbUrl(row, Math.max(160, state.clusterThumbSize * 1.5)))}" alt="" loading="lazy" />
      <div class="search-meta">
        <span class="search-rank">Cluster: ${escapeHtml(row.cluster)}</span>
        ${hasNumber(row.cluster_score) ? `<span class="search-score">Score: ${Number(row.cluster_score).toFixed(4)}</span>` : ''}
      </div>
      <div class="search-title">${escapeHtml(row.filename)}</div>
      <a class="search-open-link" href="${escapeHtml(imageUrl(row))}" target="_blank" rel="noopener">Open image</a>
    </article>
  `).join(''));
  document.querySelectorAll('#clusterGallery .cluster-card').forEach(card => {
    card.addEventListener('click', event => {
      if (event.target.closest('a')) return;
      focusSearchResult(card.dataset.path);
    });
    card.addEventListener('dblclick', () => {
      const row = state.rows.find(item => item.relative_path === card.dataset.path);
      if (row) openImagePreview(row);
    });
  });
}

function setThumbOpacity(value) {
  state.thumbOpacity = clampOpacity(value);
  localStorage.setItem('imagecluster.thumbOpacity', String(state.thumbOpacity));
  if (has('thumbOpacity')) $('thumbOpacity').value = String(Math.round(state.thumbOpacity * 100));
  setText('thumbOpacityValue', `${Math.round(state.thumbOpacity * 100)}%`);
  scheduleDraw();
}

function updateSearchThumbSize(value) {
  state.searchThumbSize = Math.max(80, Math.min(320, Number(value || 160)));
  localStorage.setItem('imagecluster.searchThumbSize', String(state.searchThumbSize));
  if (has('searchResults')) $('searchResults').style.setProperty('--search-thumb-size', `${state.searchThumbSize}px`);
}

function openImagePreview(result) {
  if (!has('imagePreviewModal')) return;
  const url = imageUrl(result);
  $('previewImage').src = url;
  $('previewImage').alt = result.filename || '';
  $('previewTitle').textContent = result.filename || 'Image preview';
  const scoreValue = hasNumber(result.score)
    ? `Score: ${Number(result.score).toFixed(4)}`
    : (hasNumber(result.cluster_score) ? `Cluster score: ${Number(result.cluster_score).toFixed(4)}` : '');
  const details = [
    hasNumber(result.rank) ? `Rank: ${Number(result.rank)}` : '',
    scoreValue,
    hasNumber(result.cluster) ? `Cluster: ${Number(result.cluster)}` : '',
  ].filter(Boolean).join(' · ');
  $('previewDetails').textContent = details;
  $('previewOpenLink').href = url;
  $('imagePreviewModal').hidden = false;
}

function closeImagePreview() {
  if (!has('imagePreviewModal')) return;
  $('imagePreviewModal').hidden = true;
  if (has('previewImage')) $('previewImage').removeAttribute('src');
}

function focusSearchResult(relativePath) {
  const row = state.rows.find(item => item.relative_path === relativePath);
  if (!row) return;
  const rect = currentRect();
  const p = project(row);
  state.panX += rect.width / 2 - p.x;
  state.panY += rect.height / 2 - p.y;
  state.hover = row;
  draw();
}

function syncSearchWithRows() {
  if (!state.searchResults.length || !state.rows.length) return;
  const rowPaths = new Set(state.rows.map(row => row.relative_path));
  state.searchResultPaths = new Set(state.searchResults.map(r => r.relative_path).filter(path => rowPaths.has(path)));
}

function clearSearch(redraw = true) {
  state.searchResults = [];
  state.searchResultPaths = new Set();
  state.lastSearchPayload = null;
  setHTML('searchResults', '');
  if (has('searchStatus')) $('searchStatus').textContent = 'No search yet.';
  if (has('searchError')) { $('searchError').hidden = true; $('searchError').textContent = ''; }
  if (has('exportSearchBtn')) $('exportSearchBtn').disabled = true;
  if (redraw) draw();
}

async function buildSearchIndex() {
  if (!has('modelKey')) return;
  setSearchBusy(true);
  setSearchError('');
  try {
    const data = await apiJson('/api/search/rebuild-index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_key: $('modelKey').value, image_dir: 'img' })
    });
    state.jobId = data.job_id;
    state.polling = true;
    if (has('cancelBtn')) $('cancelBtn').disabled = false;
    setJobText(data.message || 'Started searchable index job.');
    pollJob();
  } catch (err) {
    setSearchError(err.message);
  } finally {
    setSearchBusy(false);
  }
}

function exportSearchTsv() {
  if (!state.lastSearchPayload || !state.searchResults.length) return;
  const rows = ['rank\tfilename\trelative_path\tscore\tquery\tmodel_key\tembedding_id\tx\ty\tcluster'];
  for (const result of state.searchResults) {
    rows.push([
      result.rank,
      result.filename,
      result.relative_path,
      result.score,
      state.lastSearchPayload.query,
      state.lastSearchPayload.model_key,
      state.lastSearchPayload.embedding_id,
      result.x ?? '',
      result.y ?? '',
      result.cluster ?? '',
    ].map(tsvCell).join('\t'));
  }
  const blob = new Blob([rows.join('\n')], { type: 'text/tab-separated-values' });
  const link = document.createElement('a');
  link.download = 'imagecluster-search-results.tsv';
  link.href = URL.createObjectURL(blob);
  link.click();
  URL.revokeObjectURL(link.href);
}

function tsvCell(value) {
  return String(value ?? '').replace(/\t/g, ' ').replace(/\r?\n/g, ' ');
}

function setSearchBusy(busy) {
  if (has('searchBtn')) $('searchBtn').disabled = busy;
  if (has('buildIndexBtn')) $('buildIndexBtn').disabled = busy;
  if (has('searchStatus') && busy) $('searchStatus').textContent = 'Searching...';
  if (!busy) updateSearchCapability();
}

function setSearchError(message) {
  if (!has('searchError')) return;
  $('searchError').hidden = !message;
  $('searchError').textContent = message || '';
}

function setBusy(busy) {
  if (has('startBtn')) $('startBtn').disabled = busy;
}

function setJobText(text) {
  if (has('jobBox')) $('jobBox').textContent = text;
}

function renderJobDebugLinks(job) {
  if (!has('jobLinks')) return;
  if (!job || (!job.log_path && !job.debug_path)) {
    setHTML('jobLinks', '');
    return;
  }
  const parts = [];
  if (job.log_path) parts.push(`<a href="/api/jobs/${encodeURIComponent(job.job_id)}/log" target="_blank" rel="noopener">Open traceback log</a>`);
  if (job.debug_path) parts.push(`<a href="/api/jobs/${encodeURIComponent(job.job_id)}/debug" target="_blank" rel="noopener">Open debug report</a>`);
  setHTML('jobLinks', parts.join(' · '));
}

function imageUrl(row) {
  return `/${row.relative_path}`;
}

function thumbUrl(row, w = 160) {
  return `/api/thumb?path=${encodeURIComponent(row.relative_path)}&w=${Math.round(w)}`;
}

function parseOptionalNumber(value) {
  if (value === undefined || value === null || value === '' || value === 'null') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

// Zoom keeps the point under the cursor fixed. Zooming about the canvas centre
// (the pre-v6.0 behaviour) pushed whatever the user was inspecting off-screen on
// dense projections, forcing a pan after every wheel tick.
function zoomAt(factor, anchorX, anchorY) {
  const rect = currentRect();
  const previous = state.zoom;
  const next = Math.max(0.2, Math.min(12, previous * factor));
  if (next === previous) return;
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const ratio = next / previous;
  // Solve for the pan that leaves (anchorX, anchorY) mapping to itself.
  state.panX = anchorX - cx - ratio * (anchorX - cx - state.panX);
  state.panY = anchorY - cy - ratio * (anchorY - cy - state.panY);
  state.zoom = next;
}

function zoomAtCentre(factor) {
  const rect = currentRect();
  zoomAt(factor, rect.width / 2, rect.height / 2);
}

canvas.addEventListener('wheel', evt => {
  evt.preventDefault();
  const rect = refreshPlotRect();
  zoomAt(evt.deltaY < 0 ? 1.1 : 0.9, evt.clientX - rect.left, evt.clientY - rect.top);
  scheduleDraw();
}, { passive: false });

canvas.addEventListener('mousedown', evt => {
  state.dragging = true;
  state.dragStart = { x: evt.clientX - state.panX, y: evt.clientY - state.panY };
});
window.addEventListener('mouseup', () => { state.dragging = false; });
canvas.addEventListener('mousemove', evt => {
  if (state.dragging) {
    state.panX = evt.clientX - state.dragStart.x;
    state.panY = evt.clientY - state.dragStart.y;
  } else {
    state.hover = findHover(evt);
  }
  scheduleDraw();
});
canvas.addEventListener('mouseleave', () => { state.hover = null; scheduleDraw(); });

// Double-clicking an image asks "what else looks like this?" - the gesture the
// galleries already use to open a preview, applied to the plot where the natural
// follow-up question is about neighbours rather than detail.
canvas.addEventListener('dblclick', evt => {
  const row = findHover(evt);
  if (row && typeof runSimilarSearch === 'function') runSimilarSearch(row.relative_path);
});

if (has('scanImageDirBtn')) $('scanImageDirBtn').addEventListener('click', scanImages);
if (has('modelKey')) $('modelKey').addEventListener('change', updateModelInfo);
if (has('reducer')) $('reducer').addEventListener('change', syncReducerVisuals);
document.querySelectorAll('[data-reducer-choice]').forEach(button => {
  button.addEventListener('click', () => {
    if (!has('reducer')) return;
    $('reducer').value = button.dataset.reducerChoice;
    syncReducerVisuals();
  });
});
if (has('startBtn')) $('startBtn').addEventListener('click', startJob);
if (has('cancelBtn')) $('cancelBtn').addEventListener('click', cancelJob);
if (has('resetViewBtn')) $('resetViewBtn').addEventListener('click', () => resetView());
if (has('fitViewBtn')) $('fitViewBtn').addEventListener('click', () => fitView());
if (has('zoomInBtn')) $('zoomInBtn').addEventListener('click', () => { zoomAtCentre(1.2); draw(); });
if (has('zoomOutBtn')) $('zoomOutBtn').addEventListener('click', () => { zoomAtCentre(1 / 1.2); draw(); });
if (has('exportPngBtn')) $('exportPngBtn').addEventListener('click', exportPng);
const VIEW_MODES = ['images', 'points', 'mosaic'];
if (has('toggleModeBtn')) $('toggleModeBtn').addEventListener('click', () => {
  const next = VIEW_MODES[(VIEW_MODES.indexOf(state.mode) + 1) % VIEW_MODES.length];
  setViewMode(next);
});
document.querySelectorAll('[data-view-mode]').forEach(button => {
  button.addEventListener('click', () => setViewMode(button.dataset.viewMode));
});
if (has('mosaicAlgorithm')) $('mosaicAlgorithm').addEventListener('change', e => {
  state.mosaicAlgorithm = e.target.value;
  localStorage.setItem('imagecluster.mosaicAlgorithm', state.mosaicAlgorithm);
  if (state.mode === 'mosaic') loadMosaic(true).then(draw);
});
if (has('mosaicColumns')) $('mosaicColumns').addEventListener('change', e => {
  const value = Number(e.target.value);
  state.mosaicColumns = Number.isFinite(value) && value >= 1 ? Math.round(value) : null;
  if (state.mode === 'mosaic') loadMosaic(true).then(draw);
});
if (has('mosaicRecomputeBtn')) $('mosaicRecomputeBtn').addEventListener('click', () => {
  loadMosaic(true).then(draw);
});
if (has('thumbSize')) $('thumbSize').addEventListener('input', e => { state.thumbSize = Number(e.target.value || 48); draw(); });
if (has('thumbOpacity')) $('thumbOpacity').addEventListener('input', e => setThumbOpacity(Number(e.target.value) / 100));

document.querySelectorAll('[data-axis-preset]').forEach(button => {
  button.addEventListener('click', () => applyAxisPreset(button.dataset.axisPreset));
});

// Preview runs the same maths as the job, against the projection already loaded,
// so the axes can be tried and adjusted before committing to a full run.
if (has('semanticPreviewBtn')) $('semanticPreviewBtn').addEventListener('click', async () => {
  if (!state.resultPath) {
    setText('semanticStatus', 'Load a projection first — the preview reuses its embeddings.');
    return;
  }
  const payload = semanticAxisPayload();
  if (!payload.semantic_x_positive.length || !payload.semantic_x_negative.length
      || !payload.semantic_y_positive.length || !payload.semantic_y_negative.length) {
    setText('semanticStatus', 'Each axis needs a phrase at both ends.');
    return;
  }
  setText('semanticStatus', 'Encoding the axes... the first run loads the model.');
  try {
    const data = await apiJson('/api/semantic/axis-preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        projection_path: state.resultPath,
        x_positive: payload.semantic_x_positive,
        x_negative: payload.semantic_x_negative,
        y_positive: payload.semantic_y_positive,
        y_negative: payload.semantic_y_negative,
      }),
    });
    applyAxisPreview(data, payload);
  } catch (err) {
    setText('semanticStatus', `Preview failed: ${err.message}`);
    $('semanticStatus').className = 'status-banner warn';
  }
});

// The preview replaces the coordinates in place: same images, new axes, no job.
function applyAxisPreview(data, payload) {
  const byPath = new Map(data.points.map(point => [point.relative_path, point]));
  let moved = 0;
  for (const row of state.rows) {
    const point = byPath.get(row.relative_path);
    if (!point) continue;
    row.x = point.x;
    row.y = point.y;
    moved += 1;
  }
  boundsCache = null;
  invalidateMosaic();
  state.axisLabels = { x: payload.semantic_x_label, y: payload.semantic_y_label };
  resetView(false);
  draw();
  const warning = data.independent
    ? ''
    : ` The two axes overlap (${data.axis_overlap.toFixed(2)}), so they measure much the same thing and the plot collapses towards a diagonal — try poles that differ more.`;
  setText('semanticStatus', `Previewing ${moved} images on your axes.${warning}`);
  $('semanticStatus').className = `status-banner ${data.independent ? 'ok' : 'warn'}`;
}

// --- Concept colouring -----------------------------------------------------
if (has('conceptApplyBtn')) $('conceptApplyBtn').addEventListener('click', async () => {
  if (!state.resultPath) {
    setText('conceptStatus', 'Load or generate a projection first.');
    return;
  }
  const phrases = phraseList('conceptPhrase');
  if (!phrases.length) {
    setText('conceptStatus', 'Enter a concept to score the collection against.');
    return;
  }
  setText('conceptStatus', 'Scoring every image... the first run loads the model.');
  try {
    const data = await apiJson('/api/semantic/concept-scores', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projection_path: state.resultPath, phrases }),
    });
    setScoreColor({
      label: phrases.join(' · '),
      byPath: new Map(data.scores.map(item => [item.relative_path, item.score])),
      low: data.ramp_low,
      high: data.ramp_high,
    });
    setText('conceptStatus',
      `Scored ${data.count} images · range ${data.min.toFixed(3)} to ${data.max.toFixed(3)}, median ${data.median.toFixed(3)}.`);
    $('conceptStatus').className = 'status-banner ok';
  } catch (err) {
    setText('conceptStatus', `Scoring failed: ${err.message}`);
    $('conceptStatus').className = 'status-banner warn';
  }
});

if (has('conceptClearBtn')) $('conceptClearBtn').addEventListener('click', () => clearScoreColor());
if (has('scoreClearBtn')) $('scoreClearBtn').addEventListener('click', () => clearScoreColor());
document.querySelectorAll('[data-semantic-help]').forEach(button => {
  button.addEventListener('click', () => {
    const drawer = document.getElementById('workflowDrawer');
    if (drawer) openWorkflowDrawer('help');
  });
});
if (has('showThumbnailOutline')) $('showThumbnailOutline').addEventListener('change', e => {
  state.showThumbnailOutline = e.target.checked;
  localStorage.setItem('imagecluster.showThumbnailOutline', state.showThumbnailOutline ? 'true' : 'false');
  draw();
});
if (has('showPlannedModels')) $('showPlannedModels').addEventListener('change', e => {
  state.showPlannedModels = e.target.checked;
  localStorage.setItem('imagecluster.showPlannedModels', state.showPlannedModels ? 'true' : 'false');
  renderModelOptions();
  updateModelInfo();
});
// A job result is served through the job endpoint; a projection loaded from disk is
// served by the /output static mount. Either way the button downloads what is on screen.
function downloadUrl() {
  if (state.jobId) return `/api/jobs/${encodeURIComponent(state.jobId)}/download`;
  if (state.resultPath) return `/${state.resultPath.split('/').map(encodeURIComponent).join('/')}`;
  return null;
}

if (has('downloadBtn')) $('downloadBtn').addEventListener('click', () => {
  const url = downloadUrl();
  if (url) window.location.href = url;
});
if (has('projectionList')) $('projectionList').addEventListener('change', e => loadProjectionFile(e.target.value));
if (has('searchBtn')) $('searchBtn').addEventListener('click', runSemanticSearch);
if (has('searchQuery')) $('searchQuery').addEventListener('keydown', e => { if (e.key === 'Enter') runSemanticSearch(); });
if (has('clearSearchBtn')) $('clearSearchBtn').addEventListener('click', () => clearSearch());
if (has('buildIndexBtn')) $('buildIndexBtn').addEventListener('click', buildSearchIndex);
if (has('exportSearchBtn')) $('exportSearchBtn').addEventListener('click', exportSearchTsv);
if (has('highlightSearchToggle')) $('highlightSearchToggle').addEventListener('change', e => { state.highlightSearch = e.target.checked; draw(); });
if (has('showOnlySearchToggle')) $('showOnlySearchToggle').addEventListener('change', e => { state.showOnlySearch = e.target.checked; draw(); });
if (has('searchThumbSize')) $('searchThumbSize').addEventListener('input', e => updateSearchThumbSize(e.target.value));
if (has('clusterSelect')) $('clusterSelect').addEventListener('change', e => {
  state.selectedCluster = e.target.value;
  renderClusterCards();
  draw();
});
if (has('clusterThumbSize')) $('clusterThumbSize').addEventListener('input', e => {
  state.clusterThumbSize = Math.max(80, Math.min(320, Number(e.target.value || 160)));
  localStorage.setItem('imagecluster.clusterThumbSize', String(state.clusterThumbSize));
  if (has('clusterGallery')) $('clusterGallery').style.setProperty('--search-thumb-size', `${state.clusterThumbSize}px`);
  renderClusterCards();
});
if (has('highlightClusterToggle')) $('highlightClusterToggle').addEventListener('change', e => {
  state.highlightCluster = e.target.checked;
  draw();
});
if (has('showOnlyClusterToggle')) $('showOnlyClusterToggle').addEventListener('change', e => {
  state.showOnlyCluster = e.target.checked;
  draw();
});
if (has('exportClusterBtn')) $('exportClusterBtn').addEventListener('click', () => exportClusters(state.selectedCluster || 'all'));
if (has('exportAllClustersBtn')) $('exportAllClustersBtn').addEventListener('click', () => exportClusters('all'));
document.querySelectorAll('[data-close-preview]').forEach(el => el.addEventListener('click', closeImagePreview));

// Keyboard shortcuts (CODEX_GUIDE): / focus search · Esc close panel / clear search ·
// Enter run search · G Generate · F Fit · R Reset · C toggle cluster dock. Single-letter
// shortcuts are ignored while focus is in an INPUT/TEXTAREA/SELECT. (/ and Enter are wired
// elsewhere: redesign_ui.js and the #searchQuery keydown handler.)
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    // Escape dismisses one layer at a time, innermost first. It used to also call
    // clearSearch() unconditionally, so closing an image preview threw away the
    // search results underneath it. Clearing results is now the last resort,
    // reached only when there is nothing left to dismiss.
    const preview = $('imagePreviewModal');
    if (preview && !preview.hidden) { closeImagePreview(); return; }

    const drawer = document.getElementById('workflowDrawer');
    if (drawer && !drawer.hidden) {
      drawer.hidden = true;
      document.querySelectorAll('[data-open-drawer]').forEach(b => b.classList.remove('active'));
      return;
    }

    const opts = document.getElementById('searchOptionsPanel');
    if (opts && !opts.hidden) { opts.hidden = true; return; }

    if (has('searchQuery') && document.activeElement === $('searchQuery')) {
      $('searchQuery').blur();
      return;
    }

    if (state.searchResults.length) clearSearch();
    return;
  }
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  switch (event.key.toLowerCase()) {
    case 'g':
      if (has('startBtn') && !$('startBtn').disabled) { event.preventDefault(); $('startBtn').click(); }
      break;
    case 'f':
      if (has('fitViewBtn')) { event.preventDefault(); $('fitViewBtn').click(); }
      break;
    case 'r':
      if (has('resetViewBtn')) { event.preventDefault(); $('resetViewBtn').click(); }
      break;
    case 'm':
      event.preventDefault();
      setViewMode(VIEW_MODES[(VIEW_MODES.indexOf(state.mode) + 1) % VIEW_MODES.length]);
      break;
    case 'c': {
      const toggle = document.querySelector('[data-dock-toggle]');
      if (toggle) { event.preventDefault(); toggle.click(); }
      break;
    }
    default:
      break;
  }
});
window.addEventListener('resize', scheduleResizeCanvas);
if ('ResizeObserver' in window) {
  const resizeObserver = new ResizeObserver(scheduleResizeCanvas);
  resizeObserver.observe(canvas.parentElement || canvas);
}

async function initFromQuery() {
  if (has('showThumbnailOutline')) $('showThumbnailOutline').checked = state.showThumbnailOutline;
  setThumbOpacity(state.thumbOpacity);
  if (has('showPlannedModels')) $('showPlannedModels').checked = state.showPlannedModels;
  if (has('searchThumbSize')) $('searchThumbSize').value = state.searchThumbSize;
  if (has('clusterThumbSize')) $('clusterThumbSize').value = state.clusterThumbSize;
  if (has('highlightClusterToggle')) $('highlightClusterToggle').checked = state.highlightCluster;
  if (has('showOnlyClusterToggle')) $('showOnlyClusterToggle').checked = state.showOnlyCluster;
  updateSearchThumbSize(state.searchThumbSize);
  updateModeButton();
  updateMosaicControls();
  updateGraphLink();
  updateSemanticAxesVisibility();
  syncReducerVisuals();
  await loadStatus();
  await loadModels();
  await loadProjectionList();
  resizeCanvas();
  const params = new URLSearchParams(window.location.search);
  const modelKey = params.get('model_key');
  if (modelKey && has('modelKey') && [...$('modelKey').options].some(option => option.value === modelKey && !option.disabled)) {
    $('modelKey').value = modelKey;
    updateModelInfo();
  }
  const jobId = params.get('job_id');
  if (!jobId) return;
  state.jobId = jobId;
  try {
    const job = await apiJson(`/api/jobs/${jobId}`);
    if (job.status === 'completed') {
      state.resultPath = job.result_path;
      if (has('downloadBtn')) $('downloadBtn').disabled = false;
      await loadResult(jobId);
    } else {
      setJobText(`Job ${jobId}: ${job.status} · ${job.stage}`);
      state.polling = true;
      pollJob();
    }
  } catch (err) {
    setJobText(`Cannot load job from URL: ${err.message}`);
  }
}

initFromQuery().catch(err => setJobText(`Startup error: ${err.message}`));
