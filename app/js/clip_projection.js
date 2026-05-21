class ThumbnailCache {
  constructor({ maxItems = 500, concurrency = 12 } = {}) {
    this.maxItems = maxItems;
    this.concurrency = concurrency;
    this.items = new Map();
    this.queue = [];
    this.activeLoads = 0;
    this.pinned = new Set();
    this.onChange = null;
    this.generation = 0;
  }

  get(path) {
    const item = this.items.get(path);
    if (item && item.status === 'loaded') item.lastUsed = performance.now();
    return item || { status: 'unloaded', image: null };
  }

  request(path, priority = 0) {
    if (!path) return;
    const existing = this.items.get(path);
    if (existing) {
      existing.lastUsed = performance.now();
      if (existing.status === 'queued' && priority > existing.priority) {
        existing.priority = priority;
        const queued = this.queue.find(item => item.path === path);
        if (queued) queued.priority = priority;
        this.queue.sort((a, b) => b.priority - a.priority || a.queuedAt - b.queuedAt);
      }
      if (existing.status === 'loaded' || existing.status === 'loading' || existing.status === 'failed') return;
      if (existing.status === 'queued') {
        this.pump();
        this.notify();
        return;
      }
    }
    this.items.set(path, { status: 'queued', image: null, lastUsed: performance.now(), priority });
    this.queue.push({ path, priority, queuedAt: performance.now() });
    this.queue.sort((a, b) => b.priority - a.priority || a.queuedAt - b.queuedAt);
    this.pump();
    this.notify();
  }

  pin(path) {
    if (path) this.pinned.add(path);
  }

  unpin(path) {
    if (path) this.pinned.delete(path);
  }

  pump() {
    while (this.activeLoads < this.concurrency && this.queue.length) {
      const next = this.queue.shift();
      const item = this.items.get(next.path);
      if (!item || item.status !== 'queued') continue;
      item.status = 'loading';
      this.activeLoads += 1;
      const generation = this.generation;
      const img = new Image();
      img.onload = () => {
        if (generation !== this.generation) return;
        item.status = 'loaded';
        item.image = img;
        item.lastUsed = performance.now();
        this.activeLoads -= 1;
        this.evict();
        this.pump();
        this.notify();
        scheduleRender('thumbnail-loaded');
      };
      img.onerror = () => {
        if (generation !== this.generation) return;
        item.status = 'failed';
        item.image = null;
        item.lastUsed = performance.now();
        this.activeLoads -= 1;
        this.pump();
        this.notify();
        scheduleRender('thumbnail-failed');
      };
      img.src = imagePathToUrl(next.path);
    }
  }

  evict(visiblePaths = new Set()) {
    if (!visiblePaths.size && typeof state !== 'undefined' && state.visibleRows) {
      visiblePaths = new Set(state.visibleRows.map(row => row.relative_path));
    }
    const loaded = [...this.items.entries()].filter(([, item]) => item.status === 'loaded');
    if (loaded.length <= this.maxItems) return;
    loaded
      .filter(([path]) => !visiblePaths.has(path) && !this.pinned.has(path))
      .sort((a, b) => a[1].lastUsed - b[1].lastUsed)
      .slice(0, Math.max(0, loaded.length - this.maxItems))
      .forEach(([path]) => this.items.delete(path));
  }

  clear() {
    this.generation += 1;
    this.items.clear();
    this.queue = [];
    this.activeLoads = 0;
    this.pinned.clear();
    this.notify();
  }

  stats() {
    let loaded = 0;
    let loading = this.activeLoads;
    let failed = 0;
    for (const item of this.items.values()) {
      if (item.status === 'loaded') loaded += 1;
      if (item.status === 'failed') failed += 1;
    }
    return { loaded, loading, queued: this.queue.length, failed, max: this.maxItems };
  }

  notify() {
    if (this.onChange) this.onChange(this.stats());
  }
}

const state = {
  rows: [],
  scannedImageCount: null,
  scannedFormats: {},
  scanWarnings: [],
  selectedModel: null,
  currentJobStatus: 'ready',
  projectionBounds: null,
  transformDirty: true,
  visibleRows: [],
  spatialIndex: new Map(),
  spatialCellSize: 72,
  renderScheduled: false,
  lastRenderReason: 'initial',
  lastRenderStats: null,
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStart: null,
  hover: null,
  mode: 'images',
  thumbSize: 48,
  showThumbnailOutline: localStorage.getItem('projector.showThumbnailOutline') !== 'false',
  forceThumbnails: localStorage.getItem('projector.forceThumbnails') === 'true',
  pointSize: 6,
  jobId: null,
  resultPath: null,
  polling: false,
  models: [],
  showPlannedModels: localStorage.getItem('projector.showPlannedModels') === 'true',
  searchResults: [],
  searchResultPaths: new Set(),
  lastSearchPayload: null,
  highlightSearch: true,
  showOnlySearch: false,
  searchThumbSize: Number(localStorage.getItem('projector.searchThumbSize') || 160),
  previewPath: null,
  filters: { cluster: '', filename: '', path: '', subfolder: '', scoreMin: '', scoreMax: '' },
  lastClusterReport: null,
  lastComparison: null,
  analysisProjections: [],
  sessions: [],
};

const thumbnailCache = new ThumbnailCache({ maxItems: 500, concurrency: 12 });
thumbnailCache.onChange = () => updateThumbnailStatus();

const CLUSTER_PALETTE = ['#ff6b6b', '#4ecdc4', '#ffe66d', '#a29bfe', '#fd79a8', '#74b9ff', '#55efc4', '#fab1a0', '#c7ecee', '#badc58'];

const $ = id => document.getElementById(id);
const canvas = $('plot');
const ctx = canvas.getContext('2d');
const has = id => Boolean($(id));

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

function imagePathToUrl(path) {
  return `/${path}`;
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
    if (has('systemStatus')) {
      setText('systemStatus', missing.length ? `Missing dependencies: ${missing.map(d => d.name).join(', ')}` : 'System ready.');
      $('systemStatus').className = missing.length ? 'status-chip small danger' : 'status-chip small ok';
      renderInstallAdvice(missing.length ? data.install_advice : null);
    }
    if (missing.length && has('startBtn')) $('startBtn').disabled = true;
  } catch (err) {
    setText('systemStatus', `Cannot read system status: ${err.message}`);
  }
}

async function loadModels() {
  try {
    const data = await apiJson('/api/models');
    state.models = data.models || [];
    renderModelOptions();
    updateModelInfo();
  } catch (err) {
    setText('modelInfo', `Cannot load model registry: ${err.message}`);
  }
}

function isImageOnlyProjectionModel(model) {
  return Boolean(model && model.supports_image_embedding && model.supports_projection && !model.supports_text_search);
}

function modelOptionLabel(model) {
  const label = model.label || model.key;
  if (!isImageOnlyProjectionModel(model)) return label;
  return label.toLowerCase().includes('image only') ? label : `${label} (image only)`;
}

function renderModelOptions() {
  const select = $('modelKey');
  if (!select) return;
  const previousValue = select.value;
  const requestedValue = new URLSearchParams(window.location.search).get('model');
  const models = state.models.filter(m => {
    if (m.supports_image_embedding && m.supports_projection && m.status !== 'planned') return true;
    return state.showPlannedModels && m.status === 'planned';
  });
  select.innerHTML = models.map(m => {
    const disabled = m.status === 'unavailable' || (m.status !== 'planned' && (!m.available || !m.supports_image_embedding || !m.supports_projection));
    const suffix = m.status === 'planned'
      ? ' (planned)'
      : (disabled ? ` (${m.status || 'unavailable'})` : (m.status === 'experimental' ? ' (experimental)' : ''));
    return `<option value="${escapeHtml(m.key)}" ${m.default ? 'selected' : ''} ${disabled ? 'disabled' : ''}>${escapeHtml(modelOptionLabel(m))}${escapeHtml(suffix)}</option>`;
  }).join('');
  if (requestedValue && [...select.options].some(option => option.value === requestedValue && !option.disabled)) {
    select.value = requestedValue;
  } else if (previousValue && [...select.options].some(option => option.value === previousValue && !option.disabled)) {
    select.value = previousValue;
  }
}

function updateModelInfo() {
  if (!has('modelKey') || !has('modelInfo')) return;
  const selected = state.models.find(m => m.key === $('modelKey').value);
  state.selectedModel = selected || null;
  if (!selected) { setText('modelInfo', ''); return; }
  const missing = selected.missing_requirements && selected.missing_requirements.length ? `missing: ${selected.missing_requirements.join(', ')}` : '';
  renderModelBadges(selected);
  const searchText = selected.supports_text_search
    ? 'It can also run semantic text search after embeddings are available.'
    : 'It can create image projections but cannot process text queries.';
  const availabilityText = selected.status === 'planned'
    ? 'Planned roadmap model: not yet available for local projection.'
    : (selected.status === 'experimental' ? 'Experimental model: first use may download weights and may be slower.' : 'Available for the standard projection workflow.');
  const parts = [selected.description || selected.family, searchText, availabilityText, missing, selected.notes || ''];
  setText('modelInfo', parts.filter(Boolean).join(' '));
  if (has('batchSize') && selected.recommended_batch_size) $('batchSize').value = selected.recommended_batch_size;
  updateProjectionCapability(selected);
  updateSearchCapability();
  updateWorkflowState();
}

function renderModelBadges(model) {
  if (!has('modelBadges')) return;
  const badges = [];
  if (model.default) badges.push(['Recommended', 'good']);
  if (model.recommended_for && model.recommended_for.includes('fast')) badges.push(['Fast', 'good']);
  if (model.hardware_tier === 'cpu_ok') badges.push(['CPU-friendly', 'good']);
  if (model.hardware_tier === 'gpu_recommended' || model.hardware_tier === 'large_gpu') badges.push(['GPU recommended', 'warn']);
  if (model.supports_text_search) badges.push(['Supports semantic search', 'good']);
  if (!model.supports_text_search) badges.push(['Image-only', 'neutral']);
  if (model.status === 'experimental') badges.push(['Experimental', 'warn']);
  if (model.status === 'planned') badges.push(['Planned', 'danger']);
  if (model.status === 'unavailable' || !model.available) badges.push(['Unavailable', 'danger']);
  setHTML('modelBadges', badges.map(([text, kind]) => `<span class="capability-badge ${kind}">${escapeHtml(text)}</span>`).join(''));
}

function updateProjectionCapability(selected) {
  if (!has('startBtn')) return false;
  const canProject = Boolean(
    selected &&
    selected.status !== 'planned' &&
    selected.status !== 'unavailable' &&
    selected.available &&
    selected.supports_image_embedding &&
    selected.supports_projection &&
    Number(state.scannedImageCount) > 0 &&
    !state.polling
  );
  $('startBtn').disabled = !canProject;
  if (canProject || !selected) return canProject;
  if (!Number(state.scannedImageCount)) {
    setText('generateHint', 'Scan the img folder and make sure at least one supported image is available.');
    return canProject;
  }
  if (selected.status === 'planned') {
    setText('generateHint', `${selected.label} is planned and not wired to a verified local loader yet.`);
  } else if (selected.status === 'unavailable') {
    setText('generateHint', `${selected.label} is unavailable: ${selected.notes || 'missing local support.'}`);
  } else if (!selected.available) {
    const missingReqs = selected.missing_requirements && selected.missing_requirements.length ? selected.missing_requirements.join(', ') : 'runtime requirements';
    setText('generateHint', `${selected.label} cannot run until these requirements are available: ${missingReqs}.`);
  } else {
    setText('generateHint', `${selected.label} does not support image projection.`);
  }
  return canProject;
}

function canUseSelectedModel() {
  const model = state.selectedModel;
  return Boolean(
    model &&
    model.status !== 'planned' &&
    model.status !== 'unavailable' &&
    model.available &&
    model.supports_image_embedding &&
    model.supports_projection
  );
}

function updateWorkflowState() {
  const model = state.selectedModel || (has('modelKey') ? state.models.find(m => m.key === $('modelKey').value) : null);
  const imageCount = Number.isFinite(Number(state.scannedImageCount)) ? Number(state.scannedImageCount) : null;
  const hasImages = imageCount !== null && imageCount > 0;
  const hasProjection = state.rows.length > 0;
  const searchReady = Boolean(model && model.supports_text_search && model.available && hasProjection);
  const canGenerate = Boolean(hasImages && canUseSelectedModel() && !state.polling);

  if (has('startBtn')) $('startBtn').disabled = !canGenerate;
  if (has('cancelBtn')) $('cancelBtn').disabled = !state.polling;
  if (has('exportPngBtn')) $('exportPngBtn').disabled = !hasProjection;
  if (has('downloadBtn')) $('downloadBtn').disabled = !hasProjection || !state.resultPath;
  if (has('resetViewBtn')) $('resetViewBtn').disabled = !hasProjection;
  if (has('fitDataBtn')) $('fitDataBtn').disabled = !hasProjection;
  if (has('toggleModeBtn')) $('toggleModeBtn').disabled = !hasProjection;
  if (has('clearSearchFilterBtn')) $('clearSearchFilterBtn').disabled = !state.searchResultPaths.size && !state.showOnlySearch;
  if (has('clusterReportBtn')) $('clusterReportBtn').disabled = !hasProjection;
  if (has('exportHtmlBtn')) $('exportHtmlBtn').disabled = !hasProjection;

  if (has('searchBtn')) $('searchBtn').disabled = !searchReady;
  if (has('buildIndexBtn')) $('buildIndexBtn').disabled = !(model && model.supports_text_search && model.available && !state.polling);
  if (has('semanticSearchPanel')) $('semanticSearchPanel').classList.toggle('disabled-panel', !(model && model.supports_text_search && model.available));

  renderGenerationSummary(model, imageCount);
  renderSessionSummary(model, imageCount);
  renderGenerateHint(model, imageCount, canGenerate);
  updateFilterSummary();
}

function renderGenerationSummary(model, imageCount) {
  if (!has('generationSummary')) return;
  const rows = [
    ['Images', imageCount === null ? 'Scan needed' : `${imageCount} found`],
    ['Model', model ? model.label : 'Not selected'],
    ['Projection', has('reducer') ? $('reducer').value.toUpperCase() : 'UMAP'],
    ['Clustering', has('clusterEnabled') && $('clusterEnabled').checked ? 'Enabled' : 'Disabled'],
    ['Semantic search', model && model.supports_text_search ? 'Available after embeddings' : 'Not available for this model'],
  ];
  setHTML('generationSummary', rows.map(([label, value]) => `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`).join(''));
}

function renderGenerateHint(model, imageCount, canGenerate) {
  if (!has('generateHint')) return;
  if (state.polling) {
    setText('generateHint', 'A projection job is already running. You can cancel it before starting another.');
  } else if (canGenerate) {
    setText('generateHint', 'Ready to generate the projection graph.');
  } else if (imageCount === null) {
    setText('generateHint', 'Scan the img folder before generating a projection.');
  } else if (imageCount < 1) {
    setText('generateHint', 'No supported images were found. Add images to img and scan again.');
  } else if (!model) {
    setText('generateHint', 'Choose an embedding model.');
  } else if (model.status === 'planned') {
    setText('generateHint', 'This planned roadmap model is not yet available for local projection.');
  } else if (!model.available) {
    const missing = model.missing_requirements && model.missing_requirements.length ? model.missing_requirements.join(', ') : 'required packages';
    setText('generateHint', `This model is unavailable until ${missing} are installed.`);
  } else {
    setText('generateHint', 'The selected model cannot generate image projections.');
  }
}

function renderSessionSummary(model, imageCount) {
  setText('summaryImages', imageCount === null ? 'Not scanned' : `${imageCount} image(s)`);
  setText('summaryModel', model ? `${model.label} (${model.key})` : 'Not selected');
  setText('summaryCapability', model ? (model.supports_text_search ? 'Projection + text search' : 'Projection only') : 'Unknown');
  setText('summaryReducer', has('reducer') ? $('reducer').value.toUpperCase() : 'UMAP');
  setText('summaryClustering', has('clusterEnabled') && $('clusterEnabled').checked ? 'Enabled' : 'Disabled');
  setText('summaryJob', state.currentJobStatus || 'Ready');
  setText('summaryOutput', state.resultPath || 'None');
  const searchText = state.lastSearchPayload
    ? `${state.searchResults.length} result(s)`
    : (model && model.supports_text_search ? 'Available after projection' : 'Not available');
  setText('summarySearch', searchText);
  setText('summaryReducerDrawer', has('reducer') ? $('reducer').value.toUpperCase() : 'UMAP');
  setText('summaryClusteringDrawer', has('clusterEnabled') && $('clusterEnabled').checked ? 'Enabled' : 'Disabled');
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
  state.scannedImageCount = null;
  state.scannedFormats = {};
  state.scanWarnings = [];
  updateWorkflowState();
  try {
    const data = await apiJson('/api/images/scan?image_dir=img');
    if (!data.ok) {
      state.scannedImageCount = 0;
      state.scanWarnings = [data.error || 'The img folder could not be scanned.'];
      setText('imageDirStatus', data.error || 'The img folder could not be scanned.');
      updateWorkflowState();
      return;
    }
    state.scannedImageCount = data.count || 0;
    state.scannedFormats = data.extensions || {};
    state.scanWarnings = [...(data.warnings || [])];
    if (state.scannedImageCount > 1000) {
      state.scanWarnings.push('Large collection: projection and thumbnails may take longer.');
    }
    const formats = Object.keys(state.scannedFormats).length ? ` Formats: ${Object.keys(state.scannedFormats).join(', ')}.` : '';
    const warning = state.scanWarnings.length ? ` ${state.scanWarnings.join(' ')}` : '';
    setText('imageDirStatus', `${state.scannedImageCount} image(s) found in img.${formats}${warning}`);
  } catch (err) {
    state.scannedImageCount = 0;
    state.scanWarnings = [err.message];
    setText('imageDirStatus', `Scan error: ${err.message}`);
  } finally {
    updateWorkflowState();
  }
}

async function startJob() {
  if (!updateProjectionCapability(state.selectedModel)) {
    updateWorkflowState();
    return;
  }
  setBusy(true);
  state.currentJobStatus = 'starting';
  setHTML('jobLinks', '');
  if (has('downloadBtn')) $('downloadBtn').disabled = true;
  if (has('cancelBtn')) $('cancelBtn').disabled = false;
  state.rows = [];
  clearSearch(false);
  state.resultPath = null;
  updateWorkflowState();
  scheduleRender('job-start');
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
  };
  try {
    const data = await apiJson('/api/jobs/clip-projection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    state.jobId = data.job_id;
    state.polling = true;
    state.currentJobStatus = 'queued';
    updateWorkflowState();
    pollJob();
  } catch (err) {
    setJobText(`Failed to start job: ${err.message}`);
    state.currentJobStatus = 'failed';
    setBusy(false);
    updateWorkflowState();
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
    state.currentJobStatus = job.status || 'running';
    const pct = job.total > 0 ? Math.round((job.done / job.total) * 100) : 0;
    if (has('progressBar')) $('progressBar').style.width = `${pct}%`;
    setJobText(`${job.status} · ${job.stage} · ${job.done || 0}/${job.total || 0} · ${job.message || ''}`);
    if (job.status === 'completed') {
      state.polling = false;
      state.currentJobStatus = 'completed';
      state.resultPath = job.result_path;
      setBusy(false);
      if (has('downloadBtn')) $('downloadBtn').disabled = false;
      if (has('cancelBtn')) $('cancelBtn').disabled = true;
      renderJobDebugLinks(null);
      await loadResult();
      await loadProjectionList();
      updateWorkflowState();
      return;
    }
    if (job.status === 'failed' || job.status === 'cancelled' || job.status === 'interrupted') {
      state.polling = false;
      state.currentJobStatus = job.status;
      setBusy(false);
      if (has('cancelBtn')) $('cancelBtn').disabled = true;
      setJobText(`${job.status} · ${job.recovery_hint || job.error || job.message}`);
      renderJobDebugLinks(job);
      updateWorkflowState();
      return;
    }
    updateWorkflowState();
    setTimeout(pollJob, 1000);
  } catch (err) {
    state.polling = false;
    state.currentJobStatus = 'failed';
    setBusy(false);
    setJobText(`Job status error: ${err.message}`);
    updateWorkflowState();
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
  state.jobId = jobId || state.jobId;
  state.resultPath = resultPath;
  state.rows = rows.map(r => ({
    ...r,
    x: Number(r.x),
    y: Number(r.y),
    cluster: parseOptionalNumber(r.cluster),
    cluster_k: parseOptionalNumber(r.cluster_k),
    cluster_score: parseOptionalNumber(r.cluster_score),
  })).filter(r => Number.isFinite(r.x) && Number.isFinite(r.y));
  prepareProjectionCache();
  thumbnailCache.clear();
  syncSearchWithRows();
  applyAdaptiveDisplayMode();
  resetView(false);
  setJobText(`Loaded ${state.rows.length} projected images${resultPath ? ` · ${resultPath}` : ''}.`);
  if (has('downloadBtn')) $('downloadBtn').disabled = !state.resultPath;
  updateFilterOptions();
  updateWorkflowState();
  scheduleRender('projection-loaded');
}

async function preloadImages() {
  // Kept as a compatibility hook. Canvas thumbnails are now loaded lazily.
  requestVisibleThumbnails(2);
}

function prepareProjectionCache() {
  state.projectionBounds = computeProjectionBounds(state.rows);
  for (const row of state.rows) {
    if (!state.projectionBounds) continue;
    const rangeX = state.projectionBounds.maxX - state.projectionBounds.minX || 1;
    const rangeY = state.projectionBounds.maxY - state.projectionBounds.minY || 1;
    row._normX = (row.x - state.projectionBounds.minX) / rangeX;
    row._normY = (row.y - state.projectionBounds.minY) / rangeY;
    row._screenX = 0;
    row._screenY = 0;
  }
  markTransformDirty();
}

function computeProjectionBounds(points) {
  if (!points.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (const point of points) {
    if (!Number.isFinite(point.x) || !Number.isFinite(point.y)) continue;
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }
  if (!Number.isFinite(minX)) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  return { minX, maxX, minY, maxY };
}

function markTransformDirty() {
  state.transformDirty = true;
}

function applyAdaptiveDisplayMode() {
  if (state.rows.length > 500 && !state.forceThumbnails && state.mode === 'images') {
    state.mode = 'points';
  }
  updateModeButton();
}

function updateModeButton() {
  if (has('toggleModeBtn')) {
    $('toggleModeBtn').textContent = `Mode: ${state.mode === 'images' ? 'thumbnails' : 'points'}`;
  }
}

function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  markTransformDirty();
  scheduleRender('resize');
}

function bounds() {
  return state.projectionBounds || computeProjectionBounds(state.rows);
}

function baseProject(row) {
  const rect = canvas.getBoundingClientRect();
  const pad = 0.08;
  const nx = Number.isFinite(row._normX) ? row._normX : 0.5;
  const ny = Number.isFinite(row._normY) ? row._normY : 0.5;
  const x = 72 + ((nx + pad) / (1 + pad * 2)) * (rect.width - 144);
  const y = rect.height - 62 - ((ny + pad) / (1 + pad * 2)) * (rect.height - 122);
  return { x, y };
}

function project(row) {
  const rect = canvas.getBoundingClientRect();
  const p = baseProject(row);
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  return { x: cx + (p.x - cx) * state.zoom + state.panX, y: cy + (p.y - cy) * state.zoom + state.panY };
}

function scheduleRender(reason = 'update') {
  state.lastRenderReason = reason;
  if (state.renderScheduled) return;
  state.renderScheduled = true;
  requestAnimationFrame(() => {
    state.renderScheduled = false;
    draw();
  });
}

function updateProjectionTransform() {
  if (!state.transformDirty) return;
  const rect = canvas.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const pad = 0.08;
  for (const row of state.rows) {
    const nx = Number.isFinite(row._normX) ? row._normX : 0.5;
    const ny = Number.isFinite(row._normY) ? row._normY : 0.5;
    const baseX = 72 + ((nx + pad) / (1 + pad * 2)) * (rect.width - 144);
    const baseY = rect.height - 62 - ((ny + pad) / (1 + pad * 2)) * (rect.height - 122);
    row._screenX = cx + (baseX - cx) * state.zoom + state.panX;
    row._screenY = cy + (baseY - cy) * state.zoom + state.panY;
  }
  rebuildSpatialIndex();
  state.transformDirty = false;
}

function draw() {
  const started = performance.now();
  const rect = canvas.getBoundingClientRect();
  updateProjectionTransform();
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = '#f1efe8';
  ctx.fillRect(0, 0, rect.width, rect.height);
  drawGrid(rect);

  if (!state.rows.length) {
    ctx.fillStyle = 'rgba(21,23,26,0.82)';
    ctx.font = '15px Inter, Arial';
    ctx.fillText('Add images to img, then generate an embedding projection.', 82, 82);
    ctx.fillStyle = 'rgba(21,23,26,0.56)';
    ctx.font = '12px Inter, Arial';
    ctx.fillText('Images are read from the project img folder. The final plot remains interactive.', 82, 106);
    return;
  }

  const visibleRows = getVisibleRows(rect);
  state.visibleRows = visibleRows;
  let drawnThumbs = 0;
  let drawnPoints = 0;
  const shouldUseThumbs = shouldRenderThumbnails();

  for (const row of visibleRows) {
    const p = { x: row._screenX, y: row._screenY };
    const clusterColor = getClusterColor(row);
    const isSearchResult = state.searchResultPaths.has(row.relative_path);
    if (state.highlightSearch && state.searchResultPaths.size && !isSearchResult) ctx.globalAlpha = 0.22;
    if (state.mode === 'images' && (shouldUseThumbs || isSearchResult || state.hover === row)) {
      const cached = thumbnailCache.get(row.relative_path);
      const img = cached.image;
      if (clusterColor && state.showThumbnailOutline) drawClusterHalo(p.x, p.y, clusterColor);
      if (img) {
        drawImageThumb(img, p.x, p.y);
        drawnThumbs += 1;
      } else {
        drawThumbnailPlaceholder(p.x, p.y, clusterColor);
        drawnPoints += 1;
      }
    } else {
      drawPoint(p.x, p.y, clusterColor);
      drawnPoints += 1;
    }
    ctx.globalAlpha = 1;
    if (state.highlightSearch && isSearchResult) drawSearchRing(p.x, p.y);
  }
  requestVisibleThumbnails(shouldUseThumbs ? 1 : 0);

  drawClusterLegend(rect);
  drawLabels(rect);
  if (state.hover) drawTooltip(state.hover);
  state.lastRenderStats = {
    total: state.rows.length,
    visible: visibleRows.length,
    drawnThumbs,
    drawnPoints,
    renderMs: performance.now() - started,
    reason: state.lastRenderReason,
  };
  updateRenderStatus();
}

function drawGrid(rect) {
  ctx.strokeStyle = 'rgba(255,255,255,0.22)';
  ctx.lineWidth = 1;
  ctx.strokeRect(60, 42, rect.width - 120, rect.height - 104);
  ctx.strokeStyle = 'rgba(255,255,255,0.09)';
  for (let i = 1; i < 6; i++) {
    const x = 60 + (rect.width - 120) * i / 6;
    const y = 42 + (rect.height - 104) * i / 6;
    ctx.beginPath(); ctx.moveTo(x, 42); ctx.lineTo(x, rect.height - 62); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(60, y); ctx.lineTo(rect.width - 60, y); ctx.stroke();
  }
}

function drawLabels(rect) {
  ctx.fillStyle = 'rgba(255,255,255,0.68)';
  ctx.font = '12px Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Projection X', rect.width / 2, rect.height - 24);
  ctx.save();
  ctx.translate(24, rect.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Projection Y', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';
}

function drawImageThumb(img, x, y) {
  const size = state.thumbSize;
  const aspect = img.naturalWidth / img.naturalHeight;
  const w = aspect >= 1 ? size : size * aspect;
  const h = aspect >= 1 ? size / aspect : size;
  ctx.drawImage(img, x - w / 2, y - h / 2, w, h);
}

function drawThumbnailPlaceholder(x, y, color = null) {
  const radius = Math.max(5, Math.min(12, state.thumbSize * 0.16));
  ctx.beginPath();
  ctx.arc(x, y, radius, 0, Math.PI * 2);
  ctx.fillStyle = color || 'rgba(255,255,255,0.76)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.28)';
  ctx.stroke();
}

function getFilteredRows() {
  return state.rows.filter(row => rowPassesFilters(row));
}

function rowPassesFilters(row) {
  if (state.showOnlySearch && state.searchResultPaths.size && !state.searchResultPaths.has(row.relative_path)) return false;
  const filters = state.filters || {};
  if (filters.cluster !== '' && String(row.cluster) !== String(filters.cluster)) return false;
  if (filters.filename && !String(row.filename || '').toLowerCase().includes(filters.filename.toLowerCase())) return false;
  if (filters.path && !String(row.relative_path || '').toLowerCase().includes(filters.path.toLowerCase())) return false;
  if (filters.subfolder && subfolderOf(row.relative_path) !== filters.subfolder) return false;
  const searchResult = state.searchResults.find(result => result.relative_path === row.relative_path);
  const score = searchResult ? Number(searchResult.score) : null;
  if (filters.scoreMin !== '' && (score === null || score < Number(filters.scoreMin))) return false;
  if (filters.scoreMax !== '' && (score === null || score > Number(filters.scoreMax))) return false;
  return true;
}

function subfolderOf(relativePath) {
  const parts = String(relativePath || '').replace(/\\/g, '/').split('/');
  return parts.length > 1 ? parts.slice(0, -1).join('/') : '';
}

function getVisibleRows(rect) {
  const margin = Math.max(state.thumbSize, state.pointSize, 64);
  const rows = getFilteredRows();
  return rows.filter(row => {
    const x = row._screenX;
    const y = row._screenY;
    return x >= -margin && x <= rect.width + margin && y >= -margin && y <= rect.height + margin;
  });
}

function shouldRenderThumbnails() {
  if (state.mode !== 'images') return false;
  const count = getFilteredRows().length;
  if (state.forceThumbnails) return true;
  if (count <= 500) return true;
  if (count <= 5000) return state.zoom >= 1.8;
  return state.zoom >= 3;
}

function adaptiveRenderMessage() {
  const count = getFilteredRows().length;
  if (state.mode !== 'images') {
    return count > 500 ? 'Point mode is active for performance. Switch to thumbnails or force thumbnails when you need image previews.' : 'Point mode is active.';
  }
  if (state.forceThumbnails) return 'Forced thumbnail rendering is enabled.';
  if (count <= 500) return 'Lazy thumbnail rendering is active.';
  const threshold = count <= 5000 ? 1.8 : 3;
  if (state.zoom < threshold) return `Using points for performance. Zoom in to ${threshold}x to show visible thumbnails.`;
  return 'Rendering visible thumbnails only.';
}

function requestVisibleThumbnails(priority = 1) {
  if (!state.visibleRows.length) return;
  if (!shouldRenderThumbnails() && priority < 2) return;
  const visiblePaths = new Set(state.visibleRows.map(row => row.relative_path));
  for (const row of state.visibleRows) {
    thumbnailCache.request(row.relative_path, priority);
  }
  thumbnailCache.evict(visiblePaths);
  updateThumbnailStatus();
}

function updateThumbnailStatus() {
  if (!has('thumbnailStatus')) return;
  const stats = thumbnailCache.stats();
  $('thumbnailStatus').textContent = `Thumbnail cache: ${stats.loaded}/${stats.max} loaded, ${stats.loading} loading, ${stats.queued} queued${stats.failed ? `, ${stats.failed} failed` : ''}.`;
}

function updateRenderStatus() {
  if (has('renderNotice')) {
    $('renderNotice').textContent = adaptiveRenderMessage();
    $('renderNotice').classList.toggle('warn', state.mode === 'images' && !shouldRenderThumbnails());
  }
  if (has('renderDebug') && state.lastRenderStats) {
    const stats = state.lastRenderStats;
    $('renderDebug').textContent = [
      `items=${stats.total}`,
      `visible=${stats.visible}`,
      `thumbs=${stats.drawnThumbs}`,
      `points=${stats.drawnPoints}`,
      `render=${stats.renderMs.toFixed(1)}ms`,
      `reason=${stats.reason}`,
    ].join(' | ');
  }
  updateThumbnailStatus();
}

function drawTooltip(row) {
  const p = { x: row._screenX, y: row._screenY };
  thumbnailCache.request(row.relative_path, 3);
  const img = thumbnailCache.get(row.relative_path).image;
  const preview = 160;
  const pad = 12;
  let w = 280;
  let h = img ? 235 : 88;
  const rect = canvas.getBoundingClientRect();
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
  const searchResult = state.searchResults.find(result => result.relative_path === row.relative_path);
  if (searchResult) ctx.fillText(`score ${Number(searchResult.score).toFixed(4)}`, x + pad, textY + 51);
  if (hasCluster(row)) ctx.fillText(`cluster ${row.cluster}${hasClusterK(row) ? ` / ${row.cluster_k}` : ''}${Number.isFinite(row.cluster_score) ? ` · score ${Number(row.cluster_score).toFixed(3)}` : ''}`, x + pad, textY + (searchResult ? 68 : 51));
}

function getClusterColor(row) {
  if (!hasCluster(row)) return null;
  const idx = Math.max(0, Number(row.cluster) - 1) % CLUSTER_PALETTE.length;
  return CLUSTER_PALETTE[idx];
}

function hasCluster(row) {
  return Boolean(row && Number.isFinite(Number(row.cluster)));
}

function hasClusterK(row) {
  return Boolean(row && Number.isFinite(Number(row.cluster_k)));
}

function drawClusterHalo(x, y, color) {
  ctx.beginPath();
  ctx.arc(x, y, Math.max(13, state.thumbSize * 0.55), 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.18;
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawSearchRing(x, y) {
  ctx.save();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(x, y, Math.max(18, state.thumbSize * 0.62), 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  ctx.beginPath();
  ctx.arc(x, y, Math.max(18, state.thumbSize * 0.62), 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

function drawClusterLegend(rect) {
  const clusters = [...new Set(state.rows.map(r => r.cluster).filter(v => Number.isFinite(Number(v))))].sort((a, b) => a - b);
  if (!clusters.length) return;
  const x = rect.width - 170;
  let y = 22;
  ctx.save();
  ctx.fillStyle = 'rgba(20,20,20,0.72)';
  ctx.strokeStyle = 'rgba(255,255,255,0.18)';
  ctx.beginPath(); ctx.roundRect(x - 12, y - 10, 150, 24 + clusters.length * 18, 8); ctx.fill(); ctx.stroke();
  ctx.fillStyle = '#fff'; ctx.font = '12px Arial'; ctx.fillText('Clusters', x, y + 2);
  y += 18;
  for (const c of clusters.slice(0, 12)) {
    const color = CLUSTER_PALETTE[(Number(c) - 1) % CLUSTER_PALETTE.length];
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(x + 6, y, 5, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#ddd'; ctx.font = '11px Arial'; ctx.fillText(`Cluster ${c}`, x + 18, y + 4);
    y += 18;
  }
  if (clusters.length > 12) { ctx.fillStyle = '#aaa'; ctx.fillText(`+ ${clusters.length - 12} more`, x, y + 4); }
  ctx.restore();
}

function drawPoint(x, y, color = null) {
  ctx.beginPath();
  ctx.arc(x, y, state.pointSize, 0, Math.PI * 2);
  ctx.fillStyle = color || '#eee'; ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.28)'; ctx.stroke();
}

function resetView(redraw = true) {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  state.dragging = false;
  state.dragStart = null;
  state.hover = null;
  markTransformDirty();
  if (redraw) scheduleRender('reset-view');
}

function fitToData() {
  // The base projection maps current data bounds into the canvas at zoom=1.
  resetView(false);
  scheduleRender('fit-to-data');
}

function findHover(evt) {
  if (!state.rows.length) return null;
  const rect = canvas.getBoundingClientRect();
  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;
  const candidates = nearbyRows(mx, my);
  let best = null;
  let bestD = Math.max(22, state.thumbSize * 0.55);
  for (const row of candidates) {
    const p = { x: row._screenX, y: row._screenY };
    const d = Math.hypot(mx - p.x, my - p.y);
    if (d < bestD) { bestD = d; best = row; }
  }
  return best;
}

function rebuildSpatialIndex() {
  state.spatialIndex.clear();
  const rows = getFilteredRows();
  const cellSize = state.spatialCellSize;
  for (const row of rows) {
    const cx = Math.floor(row._screenX / cellSize);
    const cy = Math.floor(row._screenY / cellSize);
    const key = `${cx},${cy}`;
    if (!state.spatialIndex.has(key)) state.spatialIndex.set(key, []);
    state.spatialIndex.get(key).push(row);
  }
}

function nearbyRows(x, y) {
  updateProjectionTransform();
  const cellSize = state.spatialCellSize;
  const cx = Math.floor(x / cellSize);
  const cy = Math.floor(y / cellSize);
  const rows = [];
  for (let gx = cx - 1; gx <= cx + 1; gx++) {
    for (let gy = cy - 1; gy <= cy + 1; gy++) {
      const bucket = state.spatialIndex.get(`${gx},${gy}`);
      if (bucket) rows.push(...bucket);
    }
  }
  return rows;
}

function exportPng() {
  if (!state.rows.length) {
    setJobText('Generate or load a projection before exporting PNG.');
    return;
  }
  const link = document.createElement('a');
  link.download = 'projector-projection.png';
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function downloadProjectionTsv() {
  if (!state.resultPath) {
    setJobText('Generate or load a projection before downloading TSV.');
    return;
  }
  if (state.jobId) {
    window.location.href = `/api/jobs/${state.jobId}/download`;
    return;
  }
  window.open(`/${state.resultPath}`, '_blank', 'noopener');
}

function updateSearchCapability() {
  if (!has('modelKey') || !has('searchCapability')) return;
  const selected = state.models.find(m => m.key === $('modelKey').value);
  if (!selected) return;
  const searchable = Boolean(selected.supports_text_search && selected.available);
  $('searchCapability').textContent = searchable
    ? `${selected.label} can search image embeddings with text queries after a projection/index exists.`
    : 'The selected model supports image projection only and cannot process text queries.';
  $('searchCapability').className = searchable ? 'search-capability small ok' : 'search-capability small danger';
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
    syncSearchWithRows();
    updateFilterSummary();
    if (has('searchStatus')) $('searchStatus').textContent = `${state.searchResults.length} result(s) · ${data.embedding_id}`;
    if (has('exportSearchBtn')) $('exportSearchBtn').disabled = !state.searchResults.length;
    updateWorkflowState();
    openDrawer('results');
    scheduleRender('search-results');
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
  for (const result of state.searchResults) {
    thumbnailCache.request(result.relative_path, 4);
  }
  $('searchResults').style.setProperty('--search-thumb-size', `${state.searchThumbSize}px`);
  setHTML('searchResults', state.searchResults.map(result => `
    <article class="search-card" data-path="${escapeHtml(result.relative_path)}">
      <img src="${escapeHtml(imageUrl(result))}" alt="" loading="lazy" />
      <div class="search-meta">
        <span class="search-rank">Rank: ${result.rank}</span>
        <span class="search-score">Score: ${Number(result.score).toFixed(4)}</span>
        ${Number.isFinite(Number(result.cluster)) ? `<span class="search-cluster">Cluster: ${Number(result.cluster)}</span>` : ''}
      </div>
      <div class="search-title">${escapeHtml(result.filename)}</div>
      <div class="search-card-actions">
        <button type="button" class="search-detail-btn" data-preview-path="${escapeHtml(result.relative_path)}">Details</button>
        <a class="search-open-link" href="${escapeHtml(imageUrl(result))}" target="_blank" rel="noopener">Open file</a>
      </div>
    </article>
  `).join(''));
  document.querySelectorAll('.search-card').forEach(card => {
    card.addEventListener('click', event => {
      if (event.target.closest('a') || event.target.closest('button')) return;
      focusSearchResult(card.dataset.path);
    });
    card.addEventListener('dblclick', () => {
      const result = state.searchResults.find(item => item.relative_path === card.dataset.path);
      if (result) openImagePreview(result);
    });
  });
  document.querySelectorAll('.search-detail-btn').forEach(button => {
    button.addEventListener('click', () => {
      const result = state.searchResults.find(item => item.relative_path === button.dataset.previewPath);
      if (result) openImagePreview(result);
    });
  });
}

function updateSearchThumbSize(value) {
  state.searchThumbSize = Math.max(80, Math.min(320, Number(value || 160)));
  localStorage.setItem('projector.searchThumbSize', String(state.searchThumbSize));
  if (has('searchResults')) $('searchResults').style.setProperty('--search-thumb-size', `${state.searchThumbSize}px`);
}

function openImagePreview(result) {
  if (!has('imagePreviewModal')) return;
  if (state.previewPath) thumbnailCache.unpin(state.previewPath);
  state.previewPath = result.relative_path;
  thumbnailCache.pin(state.previewPath);
  thumbnailCache.request(state.previewPath, 5);
  const url = imageUrl(result);
  $('previewImage').src = url;
  $('previewImage').alt = result.filename || '';
  $('previewTitle').textContent = result.filename || 'Image preview';
  const details = [
    `Rank: ${result.rank}`,
    `Score: ${Number(result.score).toFixed(4)}`,
    Number.isFinite(Number(result.cluster)) ? `Cluster: ${Number(result.cluster)}` : '',
  ].filter(Boolean).join(' · ');
  $('previewDetails').textContent = details;
  $('previewOpenLink').href = url;
  $('imagePreviewModal').hidden = false;
}

function closeImagePreview() {
  if (!has('imagePreviewModal')) return;
  $('imagePreviewModal').hidden = true;
  if (has('previewImage')) $('previewImage').removeAttribute('src');
  if (state.previewPath) thumbnailCache.unpin(state.previewPath);
  state.previewPath = null;
}

function focusSearchResult(relativePath) {
  const row = state.rows.find(item => item.relative_path === relativePath);
  if (!row) return;
  const rect = canvas.getBoundingClientRect();
  const p = project(row);
  state.panX += rect.width / 2 - p.x;
  state.panY += rect.height / 2 - p.y;
  state.hover = row;
  markTransformDirty();
  thumbnailCache.request(row.relative_path, 5);
  scheduleRender('focus-search-result');
}

function syncSearchWithRows() {
  if (!state.searchResults.length || !state.rows.length) return;
  const rowPaths = new Set(state.rows.map(row => row.relative_path));
  state.searchResultPaths = new Set(state.searchResults.map(r => r.relative_path).filter(path => rowPaths.has(path)));
  markTransformDirty();
}

function clearSearch(redraw = true) {
  state.searchResults = [];
  state.searchResultPaths = new Set();
  state.lastSearchPayload = null;
  setHTML('searchResults', '');
  if (has('searchStatus')) $('searchStatus').textContent = 'No search yet.';
  if (has('searchError')) { $('searchError').hidden = true; $('searchError').textContent = ''; }
  if (has('exportSearchBtn')) $('exportSearchBtn').disabled = true;
  updateFilterSummary();
  updateWorkflowState();
  markTransformDirty();
  if (redraw) scheduleRender('clear-search');
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
  link.download = 'projector-search-results.tsv';
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
  if (!busy) updateWorkflowState();
}

function setSearchError(message) {
  if (!has('searchError')) return;
  $('searchError').hidden = !message;
  $('searchError').textContent = message || '';
}

function setBusy(busy) {
  if (has('startBtn')) $('startBtn').disabled = busy || !canUseSelectedModel() || !Number(state.scannedImageCount);
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
  return imagePathToUrl(row.relative_path);
}

function updateFilterOptions() {
  if (has('filterCluster')) {
    const current = state.filters.cluster;
    const clusters = [...new Set(state.rows.map(row => row.cluster).filter(v => Number.isFinite(Number(v))))].sort((a, b) => a - b);
    setHTML('filterCluster', '<option value="">All clusters</option>' + clusters.map(cluster => `<option value="${cluster}">Cluster ${cluster}</option>`).join(''));
    $('filterCluster').value = clusters.map(String).includes(String(current)) ? current : '';
    state.filters.cluster = $('filterCluster').value;
  }
  if (has('filterSubfolder')) {
    const current = state.filters.subfolder;
    const folders = [...new Set(state.rows.map(row => subfolderOf(row.relative_path)).filter(Boolean))].sort();
    setHTML('filterSubfolder', '<option value="">All subfolders</option>' + folders.map(folder => `<option value="${escapeHtml(folder)}">${escapeHtml(folder)}</option>`).join(''));
    $('filterSubfolder').value = folders.includes(current) ? current : '';
    state.filters.subfolder = $('filterSubfolder').value;
  }
  updateFilterSummary();
}

function updateFilterFromInputs() {
  state.filters = {
    cluster: has('filterCluster') ? $('filterCluster').value : '',
    filename: has('filterFilename') ? $('filterFilename').value.trim() : '',
    path: has('filterPath') ? $('filterPath').value.trim() : '',
    subfolder: has('filterSubfolder') ? $('filterSubfolder').value : '',
    scoreMin: has('filterScoreMin') ? $('filterScoreMin').value.trim() : '',
    scoreMax: has('filterScoreMax') ? $('filterScoreMax').value.trim() : '',
  };
  markTransformDirty();
  updateWorkflowState();
  scheduleRender('filters');
}

function applyFilterState(filters = {}) {
  state.filters = { cluster: '', filename: '', path: '', subfolder: '', scoreMin: '', scoreMax: '', ...filters };
  if (has('filterCluster')) $('filterCluster').value = state.filters.cluster || '';
  if (has('filterFilename')) $('filterFilename').value = state.filters.filename || '';
  if (has('filterPath')) $('filterPath').value = state.filters.path || '';
  if (has('filterSubfolder')) $('filterSubfolder').value = state.filters.subfolder || '';
  if (has('filterScoreMin')) $('filterScoreMin').value = state.filters.scoreMin || '';
  if (has('filterScoreMax')) $('filterScoreMax').value = state.filters.scoreMax || '';
  markTransformDirty();
  updateWorkflowState();
  scheduleRender('filters-restored');
}

function clearAllFilters() {
  state.showOnlySearch = false;
  if (has('showOnlySearchToggle')) $('showOnlySearchToggle').checked = false;
  applyFilterState({ cluster: '', filename: '', path: '', subfolder: '', scoreMin: '', scoreMax: '' });
}

function updateFilterSummary() {
  if (!has('filterSummary')) return;
  const visible = getFilteredRows().length;
  const total = state.rows.length;
  const active = Object.entries(state.filters || {}).filter(([, value]) => value !== '').map(([key]) => key);
  $('filterSummary').textContent = total
    ? `Showing ${visible} of ${total} images${active.length ? ` · active filters: ${active.join(', ')}` : ''}.`
    : 'No projection loaded.';
}

async function loadAnalysisProjectionOptions() {
  try {
    const data = await apiJson('/api/analysis/projections');
    state.analysisProjections = data.projections || [];
    const options = '<option value="">Choose projection...</option>' + state.analysisProjections.map(p => `<option value="${escapeHtml(p.relative_path)}">${escapeHtml(p.name || p.relative_path)}</option>`).join('');
    if (has('compareProjectionA')) setHTML('compareProjectionA', options);
    if (has('compareProjectionB')) setHTML('compareProjectionB', options);
  } catch (err) {
    if (has('comparisonResults')) setText('comparisonResults', `Cannot load projection catalog: ${err.message}`);
  }
}

async function loadSessionsList() {
  if (!has('sessionList')) return;
  try {
    const data = await apiJson('/api/sessions');
    state.sessions = data.sessions || [];
    if (!state.sessions.length) {
      setHTML('sessionList', '<option value="">No saved sessions</option>');
      return;
    }
    setHTML('sessionList', '<option value="">Choose session...</option>' + state.sessions.map(s => `<option value="${escapeHtml(s.session_id)}">${escapeHtml(s.name || s.session_id)}</option>`).join(''));
  } catch (err) {
    setHTML('sessionList', '<option value="">Cannot load sessions</option>');
  }
}

async function generateClusterReport() {
  if (!state.resultPath) {
    setText('clusterReportStatus', 'Load or generate a projection first.');
    return;
  }
  try {
    const report = await apiJson(`/api/analysis/cluster-report?projection=${encodeURIComponent(state.resultPath)}`);
    state.lastClusterReport = report;
    if (has('clusterReportJsonBtn')) $('clusterReportJsonBtn').disabled = false;
    renderClusterReport(report);
  } catch (err) {
    setText('clusterReportStatus', `Cluster report error: ${err.message}`);
  }
}

function renderClusterReport(report) {
  if (has('clusterReportStatus')) {
    setText('clusterReportStatus', report.clustering_available ? `${report.clusters.length} cluster(s), ${report.total_images} images.` : (report.message || 'No cluster data available.'));
  }
  if (!has('clusterReportResults')) return;
  if (!report.clustering_available) {
    setHTML('clusterReportResults', `<div class="status-box">${escapeHtml(report.message || 'No cluster data available.')}</div>`);
    return;
  }
  setHTML('clusterReportResults', report.clusters.map(cluster => `
    <article class="analysis-card">
      <div class="analysis-card-head">
        <strong>Cluster ${cluster.cluster}</strong>
        <span>${cluster.count} images · ${cluster.percentage}%</span>
        <button type="button" data-isolate-cluster="${cluster.cluster}">Isolate</button>
      </div>
      <div class="small">Dominant subfolder: ${escapeHtml(cluster.dominant_subfolder || 'none')}</div>
      <div class="representatives">
        ${cluster.representative_images.map(img => `<img src="${escapeHtml(imagePathToUrl(img.relative_path))}" alt="${escapeHtml(img.filename)}" title="${escapeHtml(img.filename)}" loading="lazy" />`).join('')}
      </div>
    </article>
  `).join(''));
  document.querySelectorAll('[data-isolate-cluster]').forEach(button => {
    button.addEventListener('click', () => {
      if (has('filterCluster')) $('filterCluster').value = button.dataset.isolateCluster;
      updateFilterFromInputs();
    });
  });
}

async function compareSelectedProjections() {
  const projections = [has('compareProjectionA') ? $('compareProjectionA').value : '', has('compareProjectionB') ? $('compareProjectionB').value : ''].filter(Boolean);
  if (projections.length < 2) {
    setText('comparisonResults', 'Choose two projections to compare.');
    return;
  }
  try {
    const data = await apiJson('/api/analysis/model-comparison', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projections, top_k: 5 }),
    });
    state.lastComparison = data;
    if (has('comparisonJsonBtn')) $('comparisonJsonBtn').disabled = false;
    renderComparison(data);
  } catch (err) {
    setText('comparisonResults', `Comparison error: ${err.message}`);
  }
}

function renderComparison(data) {
  if (!has('comparisonResults')) return;
  const pair = data.pairwise && data.pairwise[0];
  setHTML('comparisonResults', `
    <div class="status-box">${escapeHtml(data.warning || '')}</div>
    <div class="summary-grid">
      <div><span>Common images</span><strong>${data.common_images_count}</strong></div>
      <div><span>Mean neighbor overlap</span><strong>${pair && pair.nearest_neighbor_overlap.mean !== null ? pair.nearest_neighbor_overlap.mean : 'n/a'}</strong></div>
      <div><span>Cluster changes</span><strong>${pair ? `${pair.cluster_membership_difference.changed}/${pair.cluster_membership_difference.comparable}` : 'n/a'}</strong></div>
    </div>
    <div class="analysis-columns">
      ${(data.projections || []).map(p => `<div class="status-box"><strong>${escapeHtml(p.embedding_model || p.model_key || 'Projection')}</strong><br>${escapeHtml(p.reducer || '')} · ${p.image_count} images<br>${escapeHtml(p.relative_path)}</div>`).join('')}
    </div>
  `);
}

function collectSessionPayload() {
  return {
    name: has('sessionName') && $('sessionName').value.trim() ? $('sessionName').value.trim() : `Session ${new Date().toLocaleString()}`,
    active_projection: state.resultPath,
    active_model: state.selectedModel,
    active_reducer: has('reducer') ? $('reducer').value : null,
    display_settings: {
      zoom: state.zoom,
      panX: state.panX,
      panY: state.panY,
      mode: state.mode,
      thumbSize: state.thumbSize,
      showThumbnailOutline: state.showThumbnailOutline,
      showOnlySearch: state.showOnlySearch,
    },
    filters: state.filters,
    semantic_search: {
      last_payload: state.lastSearchPayload,
      result_paths: [...state.searchResultPaths],
      query: has('searchQuery') ? $('searchQuery').value : '',
      top_k: has('searchTopK') ? $('searchTopK').value : '',
    },
    selected_image: state.hover ? state.hover.relative_path : null,
  };
}

async function saveCurrentSession() {
  try {
    const data = await apiJson('/api/sessions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(collectSessionPayload()),
    });
    setText('sessionStatus', `Saved session ${data.session_id}.`);
    await loadSessionsList();
  } catch (err) {
    setText('sessionStatus', `Session save error: ${err.message}`);
  }
}

async function loadSelectedSession() {
  if (!has('sessionList') || !$('sessionList').value) return;
  try {
    const data = await apiJson(`/api/sessions/${encodeURIComponent($('sessionList').value)}`);
    if (has('sessionName')) $('sessionName').value = data.name || data.session_id || '';
    if (data.active_projection) await loadProjectionFile(data.active_projection);
    const display = data.display_settings || {};
    state.zoom = Number(display.zoom || 1);
    state.panX = Number(display.panX || 0);
    state.panY = Number(display.panY || 0);
    state.mode = display.mode || state.mode;
    state.thumbSize = Number(display.thumbSize || state.thumbSize);
    state.showThumbnailOutline = display.showThumbnailOutline !== false;
    state.showOnlySearch = Boolean(display.showOnlySearch);
    if (has('thumbSize')) $('thumbSize').value = state.thumbSize;
    if (has('showThumbnailOutline')) $('showThumbnailOutline').checked = state.showThumbnailOutline;
    if (has('showOnlySearchToggle')) $('showOnlySearchToggle').checked = state.showOnlySearch;
    if (data.semantic_search && data.semantic_search.last_payload) {
      state.lastSearchPayload = data.semantic_search.last_payload;
      state.searchResults = state.lastSearchPayload.results || [];
      state.searchResultPaths = new Set(state.searchResults.map(result => result.relative_path));
      if (has('searchQuery')) $('searchQuery').value = data.semantic_search.query || state.lastSearchPayload.query || '';
      renderSearchResults();
    }
    updateModeButton();
    applyFilterState(data.filters || {});
    setText('sessionStatus', `Loaded session ${data.session_id}${data.warnings && data.warnings.length ? ` · ${data.warnings.join(' ')}` : ''}`);
  } catch (err) {
    setText('sessionStatus', `Session load error: ${err.message}`);
  }
}

async function exportStandaloneHtml() {
  if (!state.resultPath) {
    setText('sessionStatus', 'Load or generate a projection before exporting HTML.');
    return;
  }
  try {
    const data = await apiJson('/api/export/html', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ projection: state.resultPath, session: collectSessionPayload(), mode: 'package' }),
    });
    setHTML('sessionStatus', `HTML export created: <a href="/${escapeHtml(data.index_path)}" target="_blank" rel="noopener">${escapeHtml(data.index_path)}</a>`);
  } catch (err) {
    setText('sessionStatus', `HTML export error: ${err.message}`);
  }
}

function exportJson(data, filename) {
  if (!data) return;
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const link = document.createElement('a');
  link.download = filename;
  link.href = URL.createObjectURL(blob);
  link.click();
  URL.revokeObjectURL(link.href);
}

function parseOptionalNumber(value) {
  if (value === undefined || value === null || value === '' || value === 'null') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function renderInstallAdvice(advice) {
  let box = document.getElementById('installAdvice');
  if (!box && has('systemStatus')) {
    box = document.createElement('div');
    box.id = 'installAdvice';
    box.className = 'install-advice';
    $('systemStatus').insertAdjacentElement('afterend', box);
  }
  if (!box) return;
  if (!advice) { box.style.display = 'none'; box.textContent = ''; return; }
  box.style.display = 'block';
  box.textContent = [
    `Detected platform: ${advice.platform}`,
    advice.recommended || '',
    '',
    'Recommended launcher:',
    advice.cpu_launcher || '',
    '',
    'CUDA launcher, when applicable:',
    advice.cuda_launcher || '',
    '',
    'Manual CPU command:',
    advice.cpu_command || '',
    '',
    'Manual CUDA command:',
    advice.cuda_command || '',
    '',
    'Then install remaining packages:',
    advice.after_pytorch || '',
    '',
    advice.imagebind_note || '',
  ].join('\n');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

const drawerTitles = {
  explore: 'Explore',
  results: 'Search Results',
  filters: 'Filters',
  session: 'Session',
  cluster: 'Cluster',
  compare: 'Compare',
  export: 'Export',
  info: 'Info',
};

function openDrawer(name) {
  if (!has('toolDrawer')) return;
  document.querySelectorAll('.drawer-pane').forEach(pane => {
    pane.hidden = pane.dataset.pane !== name;
  });
  document.querySelectorAll('.right-rail [data-drawer]').forEach(button => {
    button.classList.toggle('active', button.dataset.drawer === name);
  });
  setText('drawerTitle', drawerTitles[name] || 'Tools');
  $('toolDrawer').hidden = false;
  if (has('drawerBackdrop')) $('drawerBackdrop').hidden = false;
}

function closeDrawer() {
  if (!has('toolDrawer')) return;
  $('toolDrawer').hidden = true;
  if (has('drawerBackdrop')) $('drawerBackdrop').hidden = true;
  document.querySelectorAll('.right-rail [data-drawer]').forEach(button => button.classList.remove('active'));
}

function syncReducerRadios() {
  if (!has('reducer')) return;
  document.querySelectorAll('input[name="reducerChoice"]').forEach(input => {
    input.checked = input.value === $('reducer').value;
  });
}

function setReducerFromRadio(value) {
  if (!has('reducer')) return;
  $('reducer').value = value;
  $('reducer').dispatchEvent(new Event('change'));
  syncReducerRadios();
}

canvas.addEventListener('wheel', evt => {
  evt.preventDefault();
  state.zoom = Math.max(0.2, Math.min(12, state.zoom * (evt.deltaY < 0 ? 1.1 : 0.9)));
  markTransformDirty();
  scheduleRender('zoom');
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
    markTransformDirty();
    scheduleRender('pan');
  } else {
    const nextHover = findHover(evt);
    if (nextHover !== state.hover) {
      state.hover = nextHover;
      if (state.hover) thumbnailCache.request(state.hover.relative_path, 4);
      scheduleRender('hover');
    }
  }
});
canvas.addEventListener('mouseleave', () => {
  if (!state.hover) return;
  state.hover = null;
  scheduleRender('hover-clear');
});

if (has('scanImageDirBtn')) $('scanImageDirBtn').addEventListener('click', scanImages);
if (has('modelKey')) $('modelKey').addEventListener('change', updateModelInfo);
if (has('reducer')) $('reducer').addEventListener('change', () => {
  syncReducerRadios();
  updateWorkflowState();
});
document.querySelectorAll('input[name="reducerChoice"]').forEach(input => {
  input.addEventListener('change', event => setReducerFromRadio(event.target.value));
});
['clusterEnabled', 'clusterMode', 'clusterMinK', 'clusterK', 'clusterMaxK'].forEach(id => {
  if (has(id)) $(id).addEventListener('change', updateWorkflowState);
});
if (has('startBtn')) $('startBtn').addEventListener('click', startJob);
if (has('cancelBtn')) $('cancelBtn').addEventListener('click', cancelJob);
if (has('resetViewBtn')) $('resetViewBtn').addEventListener('click', () => resetView());
if (has('fitDataBtn')) $('fitDataBtn').addEventListener('click', fitToData);
if (has('exportPngBtn')) $('exportPngBtn').addEventListener('click', exportPng);
if (has('toggleModeBtn')) $('toggleModeBtn').addEventListener('click', () => {
  state.mode = state.mode === 'images' ? 'points' : 'images';
  updateModeButton();
  scheduleRender('display-mode');
});
if (has('thumbSize')) $('thumbSize').addEventListener('input', e => {
  state.thumbSize = Number(e.target.value || 48);
  scheduleRender('thumbnail-size');
});
if (has('showThumbnailOutline')) $('showThumbnailOutline').addEventListener('change', e => {
  state.showThumbnailOutline = e.target.checked;
  localStorage.setItem('projector.showThumbnailOutline', state.showThumbnailOutline ? 'true' : 'false');
  scheduleRender('thumbnail-outline');
});
if (has('forceThumbnails')) $('forceThumbnails').addEventListener('change', e => {
  state.forceThumbnails = e.target.checked;
  localStorage.setItem('projector.forceThumbnails', state.forceThumbnails ? 'true' : 'false');
  if (!state.forceThumbnails) applyAdaptiveDisplayMode();
  scheduleRender('force-thumbnails');
});
if (has('showPlannedModels')) $('showPlannedModels').addEventListener('change', e => {
  state.showPlannedModels = e.target.checked;
  localStorage.setItem('projector.showPlannedModels', state.showPlannedModels ? 'true' : 'false');
  renderModelOptions();
  updateModelInfo();
});
if (has('downloadBtn')) $('downloadBtn').addEventListener('click', downloadProjectionTsv);
if (has('clearSearchFilterBtn')) $('clearSearchFilterBtn').addEventListener('click', () => {
  state.showOnlySearch = false;
  if (has('showOnlySearchToggle')) $('showOnlySearchToggle').checked = false;
  clearSearch();
});
if (has('projectionList')) $('projectionList').addEventListener('change', e => loadProjectionFile(e.target.value));
if (has('searchBtn')) $('searchBtn').addEventListener('click', runSemanticSearch);
if (has('searchQuery')) $('searchQuery').addEventListener('keydown', e => { if (e.key === 'Enter') runSemanticSearch(); });
if (has('clearSearchBtn')) $('clearSearchBtn').addEventListener('click', () => clearSearch());
if (has('buildIndexBtn')) $('buildIndexBtn').addEventListener('click', buildSearchIndex);
if (has('exportSearchBtn')) $('exportSearchBtn').addEventListener('click', exportSearchTsv);
if (has('highlightSearchToggle')) $('highlightSearchToggle').addEventListener('change', e => {
  state.highlightSearch = e.target.checked;
  scheduleRender('search-highlight');
});
if (has('showOnlySearchToggle')) $('showOnlySearchToggle').addEventListener('change', e => {
  state.showOnlySearch = e.target.checked;
  markTransformDirty();
  scheduleRender('search-filter');
});
if (has('searchThumbSize')) $('searchThumbSize').addEventListener('input', e => updateSearchThumbSize(e.target.value));
['filterCluster', 'filterFilename', 'filterPath', 'filterSubfolder', 'filterScoreMin', 'filterScoreMax'].forEach(id => {
  if (has(id)) $(id).addEventListener('input', updateFilterFromInputs);
  if (has(id)) $(id).addEventListener('change', updateFilterFromInputs);
});
if (has('clearFiltersBtn')) $('clearFiltersBtn').addEventListener('click', clearAllFilters);
if (has('clusterReportBtn')) $('clusterReportBtn').addEventListener('click', generateClusterReport);
if (has('clusterReportJsonBtn')) $('clusterReportJsonBtn').addEventListener('click', () => exportJson(state.lastClusterReport, 'cluster-report.json'));
if (has('compareBtn')) $('compareBtn').addEventListener('click', compareSelectedProjections);
if (has('comparisonJsonBtn')) $('comparisonJsonBtn').addEventListener('click', () => exportJson(state.lastComparison, 'projection-comparison.json'));
if (has('saveSessionBtn')) $('saveSessionBtn').addEventListener('click', saveCurrentSession);
if (has('loadSessionBtn')) $('loadSessionBtn').addEventListener('click', loadSelectedSession);
if (has('exportHtmlBtn')) $('exportHtmlBtn').addEventListener('click', exportStandaloneHtml);
document.querySelectorAll('[data-query]').forEach(button => {
  button.addEventListener('click', () => {
    if (has('searchQuery')) $('searchQuery').value = button.dataset.query || '';
  });
});
document.querySelectorAll('[data-close-preview]').forEach(el => el.addEventListener('click', closeImagePreview));
document.querySelectorAll('.right-rail [data-drawer]').forEach(button => {
  button.addEventListener('click', () => {
    const isOpen = has('toolDrawer') && !$('toolDrawer').hidden && button.classList.contains('active');
    if (isOpen) closeDrawer();
    else openDrawer(button.dataset.drawer);
  });
});
if (has('drawerClose')) $('drawerClose').addEventListener('click', closeDrawer);
if (has('drawerBackdrop')) $('drawerBackdrop').addEventListener('click', closeDrawer);
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    closeImagePreview();
    closeDrawer();
    if (has('searchQuery') && document.activeElement === $('searchQuery')) $('searchQuery').blur();
  }
  if (event.key === '/' && has('searchQuery') && document.activeElement !== $('searchQuery')) {
    event.preventDefault();
    $('searchQuery').focus();
  }
  if ((event.key === 'g' || event.key === 'G') && has('startBtn') && !$('startBtn').disabled) startJob();
  if ((event.key === 'f' || event.key === 'F') && state.rows.length) fitToData();
  if ((event.key === 'r' || event.key === 'R') && state.rows.length) resetView();
  const drawerKeys = { '1': 'explore', '2': 'results', '3': 'filters', '4': 'session', '5': 'cluster', '6': 'compare', '7': 'export' };
  if (drawerKeys[event.key] && document.activeElement?.tagName !== 'INPUT') openDrawer(drawerKeys[event.key]);
});
window.addEventListener('resize', resizeCanvas);
if ('ResizeObserver' in window) {
  const canvasResizeObserver = new ResizeObserver(() => resizeCanvas());
  canvasResizeObserver.observe(canvas.parentElement || canvas);
}

async function initFromQuery() {
  if (has('showThumbnailOutline')) $('showThumbnailOutline').checked = state.showThumbnailOutline;
  if (has('forceThumbnails')) $('forceThumbnails').checked = state.forceThumbnails;
  if (has('showPlannedModels')) $('showPlannedModels').checked = state.showPlannedModels;
  if (has('searchThumbSize')) $('searchThumbSize').value = state.searchThumbSize;
  updateSearchThumbSize(state.searchThumbSize);
  syncReducerRadios();
  updateModeButton();
  updateThumbnailStatus();
  await loadStatus();
  await loadModels();
  await loadProjectionList();
  await loadAnalysisProjectionOptions();
  await loadSessionsList();
  resizeCanvas();
  updateWorkflowState();
  const params = new URLSearchParams(window.location.search);
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
