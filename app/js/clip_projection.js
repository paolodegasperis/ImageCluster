const state = {
  rows: [],
  images: new Map(),
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStart: null,
  hover: null,
  mode: 'images',
  thumbSize: 48,
  showThumbnailOutline: localStorage.getItem('imagecluster.showThumbnailOutline') !== 'false',
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

function selectDockPane(name, expand = false) {
  document.querySelectorAll('[data-dock-tab]').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.dockTab === name);
  });
  const cluster = $('clusterPanel');
  const search = $('searchDockPane');
  if (cluster) cluster.classList.toggle('active', name === 'cluster');
  if (search) search.classList.toggle('active', name === 'search');
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
  updateSearchCapability();
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

async function startJob() {
  setBusy(true);
  setHTML('jobLinks', '');
  if (has('downloadBtn')) $('downloadBtn').disabled = true;
  if (has('cancelBtn')) $('cancelBtn').disabled = false;
  state.rows = [];
  clearSearch(false);
  state.resultPath = null;
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
  boundsCache = null;
  preloadImages();
  syncSearchWithRows();
  resetView(false);
  setJobText(`Loaded ${state.rows.length} projected images${resultPath ? ` · ${resultPath}` : ''}.`);
  if (has('downloadBtn')) $('downloadBtn').disabled = !state.jobId;
  updateRunSummary();
  renderClusterGallery();
  renderPlotLegend();
  updateWorkflowState();
  draw();
}

function preloadImages() {
  // Non-blocking + progressive: with LOD the canvas can render colored points immediately,
  // and each thumbnail repaints (coalesced) as it arrives. Awaiting all thumbnails up front
  // would freeze the first paint for seconds (cold thumbnail cache = one resize per image).
  state.images.clear();
  for (const row of state.rows) {
    const img = new Image();
    img.onload = () => { state.images.set(row.relative_path, img); scheduleDraw(); };
    img.onerror = () => {};
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

function scheduleResizeCanvas() {
  if (resizeFrame) cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => {
    resizeFrame = null;
    resizeCanvas();
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

function baseProject(row) {
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
  drawGrid(rect);

  if (!state.rows.length) {
    ctx.fillStyle = '#15171a';
    ctx.font = '15px Geist, Arial';
    ctx.fillText('Add images to img, then generate an embedding projection.', 82, 82);
    ctx.fillStyle = '#7A7C85';
    ctx.font = '12px Geist, Arial';
    ctx.fillText('Images are read from the project img folder. The final plot remains interactive.', 82, 106);
    return;
  }

  const searchFilteredRows = state.showOnlySearch && state.searchResultPaths.size
    ? state.rows.filter(row => state.searchResultPaths.has(row.relative_path))
    : state.rows;
  const visibleRows = state.showOnlyCluster && state.selectedCluster
    ? searchFilteredRows.filter(row => String(row.cluster) === String(state.selectedCluster))
    : searchFilteredRows;

  // LOD: on large datasets, draw cheap colored points when zoomed out and switch to
  // thumbnails only when zoomed in (where few points are visible). Small datasets keep
  // thumbnails at every zoom. Point mode (toggled by the user) always draws points.
  const lodActive = state.rows.length > LOD_THUMB_MIN_ROWS;
  const showThumbs = state.mode === 'images' && (!lodActive || state.zoom >= LOD_THUMB_ZOOM);
  const cullMargin = (showThumbs ? state.thumbSize : state.pointSize * 2) + 24;
  for (const row of visibleRows) {
    const p = project(row);
    // Viewport culling: skip points outside the canvas (+margin). Invisible to the user,
    // but at high zoom/pan it avoids drawing the off-screen majority of the dataset.
    if (p.x < -cullMargin || p.x > rect.width + cullMargin || p.y < -cullMargin || p.y > rect.height + cullMargin) continue;
    const clusterColor = getClusterColor(row);
    const isSearchResult = state.searchResultPaths.has(row.relative_path);
    const isSelectedCluster = state.selectedCluster && String(row.cluster) === String(state.selectedCluster);
    if (state.highlightSearch && state.searchResultPaths.size && !isSearchResult) ctx.globalAlpha = 0.22;
    if (state.selectedCluster && !isSelectedCluster) ctx.globalAlpha = Math.min(ctx.globalAlpha, state.showOnlyCluster ? 0.12 : 0.30);
    if (showThumbs) {
      const img = state.images.get(row.relative_path);
      if (clusterColor && state.showThumbnailOutline) drawClusterFrame(p.x, p.y, clusterColor);
      if (img) drawImageThumb(img, p.x, p.y, clusterColor);
      else drawPointBox(p.x, p.y, clusterColor);
    } else {
      drawPointBox(p.x, p.y, clusterColor);
    }
    ctx.globalAlpha = 1;
    if (state.highlightSearch && isSearchResult) drawSearchRing(p.x, p.y);
    if (state.selectedCluster && isSelectedCluster && state.highlightCluster) drawClusterFocusRing(p.x, p.y, clusterColor || '#ffffff');
  }

  drawLabels(rect);
  if (state.hover) drawTooltip(state.hover);
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

function drawLabels(rect) {
  ctx.fillStyle = 'rgba(21,23,26,0.55)';
  ctx.font = '12px Geist, Arial';
  ctx.textAlign = 'center';
  ctx.fillText('Projection X', rect.width / 2, rect.height - 24);
  ctx.save();
  ctx.translate(24, rect.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText('Projection Y', 0, 0);
  ctx.restore();
  ctx.textAlign = 'left';
}

function drawImageThumb(img, x, y, color = null) {
  const box = thumbnailBox(x, y, state.thumbSize);
  ctx.save();
  ctx.fillStyle = 'rgba(255,255,255,0.78)';
  roundedRect(box.x, box.y, box.size, box.size, box.radius);
  ctx.fill();
  ctx.clip();

  const naturalW = img.naturalWidth || 1;
  const naturalH = img.naturalHeight || 1;
  const aspect = naturalW / naturalH;
  const inner = box.size - 4;
  const w = aspect >= 1 ? inner : inner * aspect;
  const h = aspect >= 1 ? inner / aspect : inner;
  ctx.drawImage(img, x - w / 2, y - h / 2, w, h);
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

function thumbnailBox(x, y, sizeOverride = state.thumbSize, expand = 0) {
  const size = Math.max(16, Number(sizeOverride || state.thumbSize)) + expand;
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
  const box = thumbnailBox(x, y, state.thumbSize, 8);
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
  const box = thumbnailBox(x, y, state.thumbSize, 12);
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,0.16)';
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 2);
  ctx.fill();
  ctx.restore();
}

function drawClusterFocusRing(x, y, color) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 3;
  ctx.setLineDash([8, 5]);
  const box = thumbnailBox(x, y, state.thumbSize, 18);
  roundedRect(box.x, box.y, box.size, box.size, box.radius + 3);
  ctx.stroke();
  ctx.setLineDash([]);
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

function renderPlotLegend() {
  if (!has('plotLegend')) return;
  const groups = clusterGroups();
  if (!groups.length) {
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

function syncReducerVisuals() {
  if (!has('reducer')) return;
  const value = $('reducer').value;
  setText('workflowReducerShort', value.toUpperCase());
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
  let bestD = Math.max(22, state.thumbSize * 0.55);
  const rows = state.showOnlySearch && state.searchResultPaths.size
    ? state.rows.filter(row => state.searchResultPaths.has(row.relative_path))
    : state.rows;
  for (const row of rows) {
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
        ${Number.isFinite(Number(result.cluster)) ? `<span class="search-cluster">Cluster: ${Number(result.cluster)}</span>` : ''}
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
        ${Number.isFinite(Number(row.cluster_score)) ? `<span class="search-score">Score: ${Number(row.cluster_score).toFixed(4)}</span>` : ''}
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
  const scoreValue = Number.isFinite(Number(result.score))
    ? `Score: ${Number(result.score).toFixed(4)}`
    : (Number.isFinite(Number(result.cluster_score)) ? `Cluster score: ${Number(result.cluster_score).toFixed(4)}` : '');
  const details = [
    Number.isFinite(Number(result.rank)) ? `Rank: ${Number(result.rank)}` : '',
    scoreValue,
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
}

function focusSearchResult(relativePath) {
  const row = state.rows.find(item => item.relative_path === relativePath);
  if (!row) return;
  const rect = canvas.getBoundingClientRect();
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

canvas.addEventListener('wheel', evt => {
  evt.preventDefault();
  state.zoom = Math.max(0.2, Math.min(12, state.zoom * (evt.deltaY < 0 ? 1.1 : 0.9)));
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
if (has('fitViewBtn')) $('fitViewBtn').addEventListener('click', () => resetView());
if (has('zoomInBtn')) $('zoomInBtn').addEventListener('click', () => { state.zoom = Math.min(12, state.zoom * 1.2); draw(); });
if (has('zoomOutBtn')) $('zoomOutBtn').addEventListener('click', () => { state.zoom = Math.max(0.2, state.zoom / 1.2); draw(); });
if (has('exportPngBtn')) $('exportPngBtn').addEventListener('click', exportPng);
if (has('toggleModeBtn')) $('toggleModeBtn').addEventListener('click', () => {
  state.mode = state.mode === 'images' ? 'points' : 'images';
  $('toggleModeBtn').textContent = state.mode === 'images' ? 'img' : 'pts';
  $('toggleModeBtn').title = `Mode: ${state.mode}`;
  draw();
});
if (has('thumbSize')) $('thumbSize').addEventListener('input', e => { state.thumbSize = Number(e.target.value || 48); draw(); });
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
if (has('downloadBtn')) $('downloadBtn').addEventListener('click', () => { if (state.jobId) window.location.href = `/api/jobs/${state.jobId}/download`; });
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
document.querySelectorAll('[data-close-preview]').forEach(el => el.addEventListener('click', closeImagePreview));

// Keyboard shortcuts (CODEX_GUIDE): / focus search · Esc close panel / clear search ·
// Enter run search · G Generate · F Fit · R Reset · C toggle cluster dock. Single-letter
// shortcuts are ignored while focus is in an INPUT/TEXTAREA/SELECT. (/ and Enter are wired
// elsewhere: redesign_ui.js and the #searchQuery keydown handler.)
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') {
    closeImagePreview();
    const opts = document.getElementById('searchOptionsPanel');
    if (opts && !opts.hidden) opts.hidden = true;
    const drawer = document.getElementById('workflowDrawer');
    if (drawer && !drawer.hidden) {
      drawer.hidden = true;
      document.querySelectorAll('[data-open-drawer]').forEach(b => b.classList.remove('active'));
    }
    if (has('searchQuery') && document.activeElement === $('searchQuery')) $('searchQuery').blur();
    clearSearch();
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
  if (has('showPlannedModels')) $('showPlannedModels').checked = state.showPlannedModels;
  if (has('searchThumbSize')) $('searchThumbSize').value = state.searchThumbSize;
  if (has('clusterThumbSize')) $('clusterThumbSize').value = state.clusterThumbSize;
  if (has('highlightClusterToggle')) $('highlightClusterToggle').checked = state.highlightCluster;
  if (has('showOnlyClusterToggle')) $('showOnlyClusterToggle').checked = state.showOnlyCluster;
  updateSearchThumbSize(state.searchThumbSize);
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
