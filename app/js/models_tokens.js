const state = {
  models: [],
  tokenConfigured: false,
  tokenTail: '',
  selected: null,
};

const MODEL_IMAGE_GUIDANCE = {
  openclip_vit_b_32: 'Balanced baseline for subjects, scenes, broad composition, recurring iconography and text-to-image search across mixed collections.',
  openai_clip_vit_b_32: 'Good for common objects, recognizable subjects, visual concepts and natural-language search terms that match everyday descriptions.',
  siglip_base_patch16_224: 'Strong visual-semantic alignment for subjects, scene context, object relationships and cleaner text-to-image matching.',
  siglip2_base_patch16_224: 'Useful for multilingual or culturally varied collections, with better attention to broad scene meaning and contextual visual cues.',
  mobileclip_b_openclip: 'Fast option for first passes over large collections; focuses on main subjects, shapes and broad composition more than fine detail.',
  dinov2_base: 'Best for visual similarity without text search: composition, texture, pose, color rhythm, style, layout and formal image structure.',
  nomic_embed_vision_v1_5: 'Designed for reusable visual embeddings; useful for broad similarity, aesthetic families and visual relationships across a collection.',
  imagebind_huge: 'Advanced multimodal option for high-level contextual relations between images; heavier and better suited to specialist experiments.',
  metaclip_b32: 'Web-scale CLIP-like model for broad subject recognition, visual concepts and general search across varied image archives.',
  metaclip_l14: 'Larger CLIP-like option that can capture finer semantic distinctions, complex scenes and more subtle subject relationships.',
  metaclip2_worldwide_h14: 'Roadmap model intended for broad worldwide visual-language coverage and culturally diverse image descriptions.',
  metaclip2_worldwide_b32: 'Experimental worldwide model for multilingual concepts, global visual categories and text search over diverse collections.',
  metaclip2_2b_worldwide: 'Roadmap entry for very large multilingual visual-language understanding, intended for richer semantic distinctions.',
  mobileclip2_s2: 'Very lightweight option for fast previews, main subjects, broad shapes and large local archives.',
  mobileclip2_b: 'Balanced MobileCLIP2 option for quick local runs with better subject and composition awareness than smaller variants.',
  mobileclip2_s4: 'Larger MobileCLIP2 option for stronger local visual-language matching while remaining lighter than large CLIP families.',
  hq_clip_b16: 'Roadmap model for higher-quality image-text alignment, useful for curated collections where precise captions matter.',
  long_clip_b32: 'Roadmap model for longer catalogue-style search terms and descriptive prompts.',
  eva_clip_l14: 'Roadmap model for strong visual similarity and art-oriented semantic grouping.',
  eva_clip_bigE14: 'Roadmap large EVA-CLIP option for richer visual detail, style and semantic grouping on powerful hardware.',
  cloc_roadmap: 'Roadmap entry for region-aware and local-alignment retrieval, useful when parts of an image matter more than the whole.',
  laclip_roadmap: 'Roadmap entry for patch-aware cultural heritage retrieval and fine-grained local visual correspondences.',
};

const $ = id => document.getElementById(id);

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

async function apiJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let text = await res.text();
    try {
      const parsed = JSON.parse(text);
      text = parsed.detail || text;
    } catch (_) {}
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

function normalizeText(value) {
  return String(value || '').toLowerCase();
}

function tokenStatusLabel() {
  if (state.tokenConfigured) {
    return state.tokenTail ? `Configured - ****${state.tokenTail}` : 'Configured';
  }
  return 'Not set';
}

function renderTokenStatus() {
  setText('hfTokenStatus', tokenStatusLabel());
  const meta = state.tokenConfigured
    ? 'The token is stored locally and can be removed at any time.'
    : 'No local token stored yet.';
  setText('hfTokenMeta', `${meta} Do not commit local_settings.json to source control.`);
  const input = $('hfTokenInput');
  if (input && !input.value && state.tokenConfigured) {
    input.placeholder = 'Token already configured';
  }
}

async function loadLocalConfig() {
  const data = await apiJson('/api/config/local');
  state.tokenConfigured = Boolean(data.huggingface_token_configured);
  state.tokenTail = data.huggingface_token_tail || '';
  renderTokenStatus();
}

async function loadModels() {
  const data = await apiJson('/api/models');
  state.models = data.models || [];
  populateCapabilityFilter();
  renderModelCounts();
  renderModelGrid();
}

function populateCapabilityFilter() {
  const select = $('modelCapabilityFilter');
  if (!select) return;
  const capabilities = new Set();
  state.models.forEach(model => {
    buildModelChips(model).forEach(chip => capabilities.add(chip));
  });
  const current = select.value || 'all';
  const options = ['<option value="all">All capabilities</option>'];
  [...capabilities].sort((a, b) => a.localeCompare(b)).forEach(cap => {
    options.push(`<option value="${escapeHtml(cap)}">${escapeHtml(cap)}</option>`);
  });
  select.innerHTML = options.join('');
  select.value = [...select.options].some(option => option.value === current) ? current : 'all';
}

function renderModelCounts() {
  const counts = { stable: 0, experimental: 0, planned: 0 };
  state.models.forEach(model => {
    if (counts[model.status] !== undefined) counts[model.status] += 1;
  });
  setText('modelCounts', `Stable ${counts.stable} - Experimental ${counts.experimental} - Planned ${counts.planned}`);
}

function buildModelChips(model) {
  const chips = new Set();
  if (model.supports_projection) chips.add('projection');
  if (model.supports_text_search) chips.add('semantic search');
  if (model.supports_text_embedding) chips.add('text embedding');
  if (model.supports_image_embedding === false) chips.add('image only');
  if (model.hardware_tier === 'cpu_ok') chips.add('CPU-friendly');
  if (model.hardware_tier === 'gpu_recommended') chips.add('GPU recommended');
  if (model.hardware_tier === 'large_gpu') chips.add('Large GPU');
  (model.recommended_for || []).forEach(tag => chips.add(tag.replace(/_/g, ' ')));
  if (model.trust_remote_code) chips.add('remote code');
  return [...chips];
}

function imageGuidance(model) {
  if (MODEL_IMAGE_GUIDANCE[model.key]) return MODEL_IMAGE_GUIDANCE[model.key];
  const family = normalizeText(model.family);
  if (family.includes('dino')) {
    return 'Best for formal visual similarity: composition, texture, shape, color and style rather than text search.';
  }
  if (family.includes('clip') || family.includes('siglip')) {
    return 'Best for subject recognition, semantic concepts, visual-language search and broad scene understanding.';
  }
  if (family.includes('imagebind')) {
    return 'Best for advanced multimodal relationships and high-level contextual similarity.';
  }
  return 'Use this model to compare images through the visual features exposed by its embedding space.';
}

function filterModels() {
  const query = normalizeText($('modelSearch')?.value || '');
  const status = $('modelStatusFilter')?.value || 'all';
  const capability = $('modelCapabilityFilter')?.value || 'all';
  return state.models.filter(model => {
    if (status !== 'all' && model.status !== status) return false;
    const chips = buildModelChips(model);
    if (capability !== 'all' && !chips.some(chip => normalizeText(chip) === normalizeText(capability))) return false;
    if (!query) return true;
    const haystack = [
      model.family,
      model.label,
      model.model_id,
      model.description,
      model.notes,
      model.published,
      model.provider,
      model.status,
      imageGuidance(model),
      ...chips,
    ].map(normalizeText).join(' ');
    return haystack.includes(query);
  });
}

function renderModelGrid() {
  const grid = $('modelGrid');
  if (!grid) return;
  const filtered = filterModels();
  setText('modelShowing', `Showing ${filtered.length} / ${state.models.length}`);
  if (!filtered.length) {
    grid.innerHTML = `
      <div class="empty-state">
        <h3>No models match these filters.</h3>
        <p>Try clearing the filters or broadening the search terms.</p>
      </div>
    `;
    return;
  }
  grid.innerHTML = filtered.map(model => renderModelCard(model)).join('');
  grid.querySelectorAll('[data-model-key]').forEach(card => {
    card.addEventListener('click', () => openDrawer(card.dataset.modelKey));
  });
}

function renderModelCard(model) {
  const chips = buildModelChips(model);
  const chipHtml = chips.map(chip => `<span class="chip">${escapeHtml(chip)}</span>`).join('');
  const statusClass = `status-${model.status}`;
  return `
    <article class="model-card" data-model-key="${escapeHtml(model.key)}">
      <div class="model-card-head">
        <div>
          <h3>${escapeHtml(model.label)}</h3>
          <div class="small mono">${escapeHtml(model.model_id || model.pretrained || model.key)}</div>
        </div>
        <span class="status-chip ${statusClass}">${escapeHtml(model.status)}</span>
      </div>
      <p>${escapeHtml(model.description || model.notes || 'No description available.')}</p>
      <p class="model-focus"><strong>Image focus:</strong> ${escapeHtml(imageGuidance(model))}</p>
      <div class="chip-row">${chipHtml}</div>
      <div class="model-card-foot">
        <span class="small">${escapeHtml(model.family)}</span>
        <button type="button" class="btn btn-sm btn-ghost">Details</button>
      </div>
    </article>
  `;
}

function openDrawer(modelKey) {
  const model = state.models.find(item => item.key === modelKey);
  if (!model) return;
  state.selected = model;
  setText('drawerTitle', model.label);
  setText('drawerSubtitle', [
    model.family,
    model.model_id || model.pretrained || 'local registry entry',
    model.published ? `published ${model.published}` : '',
  ].filter(Boolean).join(' - '));
  setHTML('drawerBody', renderDrawerBody(model));
  const blocked = model.status === 'planned' || model.status === 'unavailable';
  $('drawerUse').href = blocked ? '#' : `/clip?model_key=${encodeURIComponent(model.key)}`;
  $('drawerUse').classList.toggle('is-disabled', blocked);
  $('drawerUse').setAttribute('aria-disabled', String(blocked));
  $('drawerUse').textContent = blocked ? 'Not available yet' : 'Use this model';
  $('modelDrawer').hidden = false;
  showElement('modelDrawer', 'grid');
  document.body.classList.add('drawer-open');
}

function closeDrawer() {
  $('modelDrawer').hidden = true;
  hideElement('modelDrawer');
  document.body.classList.remove('drawer-open');
  state.selected = null;
}

function renderDrawerBody(model) {
  const chips = buildModelChips(model);
  const specs = [
    ['Status', model.status],
    ['Provider', model.provider || ''],
    ['Model ID', model.model_id || ''],
    ['Pretrained', model.pretrained || ''],
    ['Recommended batch', model.recommended_batch_size || ''],
    ['Hardware tier', model.hardware_tier || ''],
    ['Remote code', model.trust_remote_code ? 'Yes' : 'No'],
  ].filter(([, value]) => value !== '');

  return `
    <p>${escapeHtml(model.description || model.notes || 'No long description available.')}</p>
    <section class="drawer-insight">
      <h3>Image aspects considered</h3>
      <p>${escapeHtml(imageGuidance(model))}</p>
    </section>
    ${model.notes ? `<p class="drawer-note">${escapeHtml(model.notes)}</p>` : ''}
    <div class="chip-row drawer-chip-row">${chips.map(chip => `<span class="chip">${escapeHtml(chip)}</span>`).join('')}</div>
    <div class="spec-grid">
      ${specs.map(([key, value]) => `
        <div class="spec-item">
          <div class="spec-key">${escapeHtml(key)}</div>
          <div class="spec-value">${escapeHtml(String(value))}</div>
        </div>
      `).join('')}
    </div>
  `;
}

function applyFilters() {
  renderModelGrid();
}

async function saveToken() {
  const token = $('hfTokenInput')?.value.trim() || '';
  if (!token) {
    setText('hfTokenMeta', 'Enter a token before saving it locally.');
    return;
  }
  await apiJson('/api/config/local/hf-token', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token }),
  });
  $('hfTokenInput').value = '';
  await loadLocalConfig();
}

async function clearToken() {
  await apiJson('/api/config/local/hf-token', { method: 'DELETE' });
  $('hfTokenInput').value = '';
  await loadLocalConfig();
}

function updateRevealState() {
  const input = $('hfTokenInput');
  if (!input) return;
  input.type = $('hfTokenReveal')?.checked ? 'text' : 'password';
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
}

document.querySelectorAll('[data-close-drawer]').forEach(el => el.addEventListener('click', closeDrawer));
window.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeDrawer();
});

$('hfTokenSave')?.addEventListener('click', saveToken);
$('hfTokenClear')?.addEventListener('click', clearToken);
$('hfTokenReveal')?.addEventListener('change', updateRevealState);
$('modelSearch')?.addEventListener('input', applyFilters);
$('modelStatusFilter')?.addEventListener('change', applyFilters);
$('modelCapabilityFilter')?.addEventListener('change', applyFilters);
$('modelClearFilters')?.addEventListener('click', () => {
  if ($('modelSearch')) $('modelSearch').value = '';
  if ($('modelStatusFilter')) $('modelStatusFilter').value = 'all';
  if ($('modelCapabilityFilter')) $('modelCapabilityFilter').value = 'all';
  applyFilters();
});

async function init() {
  updateRevealState();
  try {
    await loadLocalConfig();
  } catch (err) {
    setText('hfTokenStatus', 'Unavailable');
    setText('hfTokenMeta', `Cannot read local configuration: ${err.message}`);
  }
  try {
    await loadModels();
  } catch (err) {
    setText('modelCounts', `Cannot load models: ${err.message}`);
    setHTML('modelGrid', `<div class="empty-state"><h3>Model registry unavailable</h3><p>${escapeHtml(err.message)}</p></div>`);
  }
}

init();
