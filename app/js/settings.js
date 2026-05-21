const $ = id => document.getElementById(id);

const state = {
  models: [],
  selectedModel: null,
};

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#039;',
  }[ch]));
}

async function apiJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) throw new Error(data.detail || data.error || response.statusText);
  return data;
}

function renderTokenStatus(huggingface) {
  const configured = Boolean(huggingface && huggingface.configured);
  $('hfTokenBadge').textContent = configured ? 'Configured' : 'Not configured';
  $('hfTokenBadge').className = configured ? 'badge badge-ok' : 'badge badge-warn';
  const source = huggingface?.source || 'none';
  const masked = huggingface?.masked ? ` (${huggingface.masked})` : '';
  $('hfTokenStatus').textContent = configured
    ? `Hugging Face token is active from ${source}${masked}. New model downloads will use it while the app is running.`
    : 'No Hugging Face token is configured. Public downloads still work, but rate limits and speed may be worse.';
}

async function loadSettings() {
  const data = await apiJson('/api/settings');
  renderTokenStatus(data.huggingface);
}

async function saveToken() {
  const token = $('hfTokenInput').value.trim();
  if (!token) {
    $('hfTokenStatus').textContent = 'Enter a token before saving.';
    return;
  }
  const data = await apiJson('/api/settings/huggingface-token', {
    method: 'PUT',
    body: JSON.stringify({ token }),
  });
  $('hfTokenInput').value = '';
  renderTokenStatus(data.huggingface);
}

async function removeToken() {
  const data = await apiJson('/api/settings/huggingface-token', { method: 'DELETE' });
  renderTokenStatus(data.huggingface);
}

function matchesModel(model) {
  const filter = $('modelGuideFilter').value.trim().toLowerCase();
  const status = $('modelGuideStatus').value;
  const capability = $('modelGuideCapability').value;
  if (status && model.status !== status) return false;
  if (capability && !(model.capabilities || []).includes(capability)) return false;
  if (!filter) return true;
  const haystack = [
    model.label,
    model.family,
    model.status,
    model.hardware_label,
    model.description,
    model.notes,
    model.limitation,
    ...(model.capabilities || []),
    ...(model.recommended_for || []),
  ].join(' ').toLowerCase();
  return haystack.includes(filter);
}

function renderModelGuide() {
  const models = state.models.filter(matchesModel);
  $('modelGuideCount').textContent = `${models.length} model(s)`;
  $('modelGuideGrid').innerHTML = models.map(model => `
    <article class="model-guide-card" data-model-key="${escapeHtml(model.key)}" tabindex="0">
      <div class="model-guide-head">
        <div>
          <h3>${escapeHtml(model.label)}</h3>
          <div class="mono small">${escapeHtml(model.key)}</div>
        </div>
        <span class="capability-badge ${model.status === 'experimental' ? 'warn' : model.status === 'planned' ? 'danger' : 'good'}">${escapeHtml(model.status)}</span>
      </div>
      <div class="badge-row">
        ${(model.capabilities || []).map(item => `<span class="capability-badge ${item.includes('semantic') ? 'good' : 'neutral'}">${escapeHtml(item)}</span>`).join('')}
        <span class="capability-badge ${model.hardware_tier === 'large_gpu' ? 'warn' : 'neutral'}">${escapeHtml(model.hardware_label)}</span>
      </div>
      <p>${escapeHtml(model.description)}</p>
      <p class="small">${escapeHtml(model.limitation)}</p>
      ${(model.recommended_for || []).length ? `<div class="model-tags">${model.recommended_for.map(item => `<span>${escapeHtml(item.replaceAll('_', ' '))}</span>`).join('')}</div>` : ''}
      <button class="details-link" type="button" data-model-key="${escapeHtml(model.key)}">Details</button>
    </article>
  `).join('');
  document.querySelectorAll('[data-model-key]').forEach(el => {
    el.addEventListener('click', event => {
      const key = event.currentTarget.dataset.modelKey;
      if (key) openModelDetail(key);
    });
    el.addEventListener('keydown', event => {
      if ((event.key === 'Enter' || event.key === ' ') && event.currentTarget.classList.contains('model-guide-card')) {
        event.preventDefault();
        openModelDetail(event.currentTarget.dataset.modelKey);
      }
    });
  });
}

function clearModelFilters() {
  $('modelGuideFilter').value = '';
  $('modelGuideStatus').value = '';
  $('modelGuideCapability').value = '';
  renderModelGuide();
}

function openModelDetail(key) {
  const model = state.models.find(item => item.key === key);
  if (!model || !$('modelDetail')) return;
  state.selectedModel = model;
  $('modelDetailFamily').textContent = model.family || 'Model';
  $('modelDetailTitle').textContent = model.label;
  $('modelDetailBody').innerHTML = `
    <div class="badge-row">
      <span class="capability-badge ${model.status === 'experimental' ? 'warn' : model.status === 'planned' ? 'danger' : 'good'}">${escapeHtml(model.status)}</span>
      <span class="capability-badge neutral">${escapeHtml(model.hardware_label)}</span>
      ${(model.capabilities || []).map(item => `<span class="capability-badge ${item.includes('semantic') ? 'good' : 'neutral'}">${escapeHtml(item)}</span>`).join('')}
    </div>
    <p>${escapeHtml(model.profile || model.description || '')}</p>
    <dl class="spec-grid">
      <div><dt>Key</dt><dd>${escapeHtml(model.key)}</dd></div>
      <div><dt>Status</dt><dd>${escapeHtml(model.status)}</dd></div>
      <div><dt>Hardware</dt><dd>${escapeHtml(model.hardware_label)}</dd></div>
      <div><dt>Available</dt><dd>${model.available ? 'Yes' : 'No'}</dd></div>
    </dl>
    ${model.notes ? `<p class="small">${escapeHtml(model.notes)}</p>` : ''}
    <p class="small">${escapeHtml(model.limitation || '')}</p>
  `;
  $('modelDetailUse').href = `/clip?model=${encodeURIComponent(model.key)}`;
  $('modelDetailUse').classList.toggle('disabled-link', model.status === 'planned' || !model.available);
  $('modelDetail').hidden = false;
}

function closeModelDetail() {
  if ($('modelDetail')) $('modelDetail').hidden = true;
}

async function loadModelGuide() {
  const data = await apiJson('/api/model-guide');
  state.models = data.models || [];
  renderModelGuide();
}

document.addEventListener('DOMContentLoaded', () => {
  $('saveHfTokenBtn')?.addEventListener('click', () => saveToken().catch(err => { $('hfTokenStatus').textContent = err.message; }));
  $('removeHfTokenBtn')?.addEventListener('click', () => removeToken().catch(err => { $('hfTokenStatus').textContent = err.message; }));
  $('modelGuideFilter')?.addEventListener('input', renderModelGuide);
  $('modelGuideStatus')?.addEventListener('change', renderModelGuide);
  $('modelGuideCapability')?.addEventListener('change', renderModelGuide);
  $('modelGuideClear')?.addEventListener('click', clearModelFilters);
  $('modelDetailClose')?.addEventListener('click', closeModelDetail);
  loadSettings().catch(err => { $('hfTokenStatus').textContent = `Cannot load settings: ${err.message}`; });
  loadModelGuide().catch(err => { $('modelGuideGrid').innerHTML = `<div class="search-error">${escapeHtml(err.message)}</div>`; });
});
