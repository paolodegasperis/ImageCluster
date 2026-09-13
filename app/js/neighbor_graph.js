// Nearest-neighbor graph view.
//
// This is a standalone view rather than a mode of the projection page: it answers a
// different question. The scatter plot shows where an image sits in the overall space;
// the graph shows which images are actually closest to it in the full embedding space,
// where the 2D projection's distortions do not apply.
//
// The backend computes the edges (see backend/neighbors.py). Everything here is layout,
// rendering and interaction.

// Number(null) is 0, a finite number, so testing finiteness alone reads "this
// image has no cluster" as "this image is in cluster 0" - it coloured unclustered
// nodes as if they were grouped and printed "cluster null" in the inspector.
function hasNumber(value) {
  if (value === null || value === undefined || value === '') return false;
  return Number.isFinite(Number(value));
}

function clamp01(value, fallback) {
  // localStorage.getItem returns null when nothing is stored, and Number(null) is 0 —
  // a finite number, so a plain isFinite guard would read "never stored" as "fully
  // transparent" and paint nothing.
  if (value === null || value === undefined || value === '') return fallback;
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(0, Math.min(1, number));
}

const state = {
  graph: null,
  nodes: [],
  edges: [],
  positions: [],
  projections: [],
  projectionPath: '',
  selected: null,
  hover: null,
  images: new Map(),
  zoom: 1,
  panX: 0,
  panY: 0,
  dragging: false,
  dragStart: null,
  layout: localStorage.getItem('imagecluster.graphLayout') || 'projection',
  // Community is the default because it is the reading that says something on a
  // connected graph; component collapses to one colour as soon as k is high enough.
  colorBy: localStorage.getItem('imagecluster.graphColorBy') || 'community',
  edgeColor: localStorage.getItem('imagecluster.graphEdgeColor') || 'similarity',
  inspectorTab: 'selected',
  highlightCommunity: null,
  showThumbs: localStorage.getItem('imagecluster.graphShowThumbs') !== 'false',
  nodeSize: Number(localStorage.getItem('imagecluster.graphNodeSize') || 26),
  thumbOpacity: clamp01(localStorage.getItem('imagecluster.graphOpacity'), 1),
  edgeOpacity: clamp01(localStorage.getItem('imagecluster.graphEdgeOpacity'), 0.35),
  busy: false,
};

const PALETTE = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#14B8A6', '#F97316', '#6366F1', '#84CC16'];
const THUMB_WIDTH = 128;
const MAX_CONCURRENT_THUMB_LOADS = 12;
// Above this many nodes the per-node thumbnail is both illegible and expensive,
// so the renderer falls back to colored dots regardless of the toggle.
const THUMB_NODE_LIMIT = 1200;
// Bridges and articulation points are structural findings rather than data, so
// they get one signal colour that no palette entry uses.
const STRUCTURE_COLOR = '#C77A18';

const $ = id => document.getElementById(id);
const canvas = $('graphCanvas');
const ctx = canvas.getContext('2d');
let plotRect = null;
let drawScheduled = false;

function refreshRect() {
  plotRect = canvas.getBoundingClientRect();
  return plotRect;
}

function currentRect() {
  return plotRect || refreshRect();
}

function scheduleDraw() {
  if (drawScheduled) return;
  drawScheduled = true;
  requestAnimationFrame(() => { drawScheduled = false; draw(); });
}

function setText(id, value) {
  const el = $(id);
  if (el) el.textContent = value;
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]));
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

function thumbUrl(node, width = THUMB_WIDTH) {
  return `/api/thumb?path=${encodeURIComponent(node.relative_path)}&w=${Math.round(width)}`;
}

function imageUrl(node) {
  return `/${node.relative_path}`;
}

// --- Thumbnail loading ----------------------------------------------------
// Same discipline as the projection canvas: fetch only what is drawn, a few at a time.
const thumbQueue = [];
const thumbQueued = new Set();
const thumbFailed = new Set();
let thumbInFlight = 0;

function requestThumb(node) {
  const key = node.relative_path;
  if (state.images.has(key) || thumbQueued.has(key) || thumbFailed.has(key)) return;
  thumbQueued.add(key);
  thumbQueue.push(node);
  pumpThumbQueue();
}

function pumpThumbQueue() {
  while (thumbInFlight < MAX_CONCURRENT_THUMB_LOADS && thumbQueue.length) {
    const node = thumbQueue.pop();
    const key = node.relative_path;
    thumbInFlight += 1;
    const img = new Image();
    img.decoding = 'async';
    img.onload = () => { thumbInFlight -= 1; thumbQueued.delete(key); state.images.set(key, img); scheduleDraw(); pumpThumbQueue(); };
    img.onerror = () => { thumbInFlight -= 1; thumbQueued.delete(key); thumbFailed.add(key); pumpThumbQueue(); };
    img.src = thumbUrl(node);
  }
}

// --- Layout ---------------------------------------------------------------

function layoutFromProjection() {
  // Normalise the projection's own coordinates into the unit square. This is
  // instant and keeps the graph spatially comparable to the scatter view.
  // Scanned in a loop rather than with Math.min(...array): spreading thousands of
  // elements into a call is bounded by the engine's argument limit.
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const node of state.nodes) {
    if (Number.isFinite(node.x)) {
      if (node.x < minX) minX = node.x;
      if (node.x > maxX) maxX = node.x;
    }
    if (Number.isFinite(node.y)) {
      if (node.y < minY) minY = node.y;
      if (node.y > maxY) maxY = node.y;
    }
  }
  if (!Number.isFinite(minX)) { minX = 0; maxX = 1; }
  if (!Number.isFinite(minY)) { minY = 0; maxY = 1; }
  const spanX = (maxX - minX) || 1;
  const spanY = (maxY - minY) || 1;
  state.positions = state.nodes.map(node => ({
    x: Number.isFinite(node.x) ? (node.x - minX) / spanX : 0.5,
    // Screen y grows downwards; flip so the graph matches the scatter's orientation.
    y: Number.isFinite(node.y) ? 1 - (node.y - minY) / spanY : 0.5,
  }));
}

// Force-directed relaxation, seeded from the projection so it converges in a
// handful of passes instead of from random noise. Repulsion is approximated on a
// uniform grid: only nodes in neighbouring cells push each other, which keeps a
// pass O(n) instead of O(n^2).
function layoutForce(iterations = 120) {
  layoutFromProjection();
  const count = state.positions.length;
  if (count < 2) return;

  const ideal = 1 / Math.sqrt(count);
  const repulsion = ideal * ideal;
  const cellSize = ideal * 2;

  for (let step = 0; step < iterations; step++) {
    const temperature = 0.08 * (1 - step / iterations);
    const dispX = new Float64Array(count);
    const dispY = new Float64Array(count);

    const buckets = new Map();
    for (let i = 0; i < count; i++) {
      const key = `${Math.floor(state.positions[i].x / cellSize)},${Math.floor(state.positions[i].y / cellSize)}`;
      let bucket = buckets.get(key);
      if (!bucket) { bucket = []; buckets.set(key, bucket); }
      bucket.push(i);
    }

    for (let i = 0; i < count; i++) {
      const pi = state.positions[i];
      const cx = Math.floor(pi.x / cellSize);
      const cy = Math.floor(pi.y / cellSize);
      for (let ox = -1; ox <= 1; ox++) {
        for (let oy = -1; oy <= 1; oy++) {
          const bucket = buckets.get(`${cx + ox},${cy + oy}`);
          if (!bucket) continue;
          for (const j of bucket) {
            if (j === i) continue;
            let dx = pi.x - state.positions[j].x;
            let dy = pi.y - state.positions[j].y;
            let distSq = dx * dx + dy * dy;
            if (distSq < 1e-9) {
              // Perfectly coincident nodes need a nudge or they never separate.
              dx = (Math.random() - 0.5) * 1e-4;
              dy = (Math.random() - 0.5) * 1e-4;
              distSq = dx * dx + dy * dy;
            }
            const force = repulsion / distSq;
            const dist = Math.sqrt(distSq);
            dispX[i] += (dx / dist) * force;
            dispY[i] += (dy / dist) * force;
          }
        }
      }
    }

    for (const edge of state.edges) {
      const a = state.positions[edge.source];
      const b = state.positions[edge.target];
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1e-6;
      // Stronger edges pull harder, so highly similar pairs end up adjacent.
      const strength = (dist * dist / ideal) * (0.4 + 0.6 * Math.max(0, edge.similarity));
      const fx = (dx / dist) * strength;
      const fy = (dy / dist) * strength;
      dispX[edge.source] -= fx;
      dispY[edge.source] -= fy;
      dispX[edge.target] += fx;
      dispY[edge.target] += fy;
    }

    for (let i = 0; i < count; i++) {
      const length = Math.sqrt(dispX[i] * dispX[i] + dispY[i] * dispY[i]) || 1e-9;
      const limit = Math.min(length, temperature);
      state.positions[i].x = Math.max(0, Math.min(1, state.positions[i].x + (dispX[i] / length) * limit));
      state.positions[i].y = Math.max(0, Math.min(1, state.positions[i].y + (dispY[i] / length) * limit));
    }
  }
}

function applyLayout() {
  if (!state.nodes.length) return;
  if (state.layout === 'force') layoutForce();
  else layoutFromProjection();
}

// --- Projection stress ----------------------------------------------------
// Every edge carries two measurements that ought to agree: how far apart the two
// images sit on screen, and how similar the model says they are. Their
// disagreement is where the 2D projection had to break a real relationship.
// Nothing new is computed - both numbers are already here.
let stressCache = null;

function invalidateStress() {
  stressCache = null;
}

function edgeStressValues() {
  if (stressCache) return stressCache;
  const lengths = state.edges.map(edge => {
    const a = state.positions[edge.source];
    const b = state.positions[edge.target];
    if (!a || !b) return 0;
    return Math.hypot(a.x - b.x, a.y - b.y);
  });
  const longest = Math.max(1e-6, ...lengths);
  // proximity 1 = adjacent on screen, 0 = as far apart as any linked pair.
  // stress > 0 means "more similar than the projection makes them look".
  stressCache = state.edges.map((edge, index) => {
    const proximity = 1 - (lengths[index] / longest);
    return Math.max(0, Math.min(1, edge.similarity - proximity));
  });
  return stressCache;
}

function stressAvailable() {
  return state.layout === 'projection';
}

// --- Projection to screen -------------------------------------------------

function screenPosition(index) {
  const rect = currentRect();
  const position = state.positions[index];
  if (!position) return { x: -9999, y: -9999 };
  const margin = 50;
  const baseX = margin + position.x * Math.max(1, rect.width - margin * 2);
  const baseY = margin + position.y * Math.max(1, rect.height - margin * 2);
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  return {
    x: cx + (baseX - cx) * state.zoom + state.panX,
    y: cy + (baseY - cy) * state.zoom + state.panY,
  };
}

function nodeColor(node) {
  if (state.colorBy === 'community') {
    if (!hasNumber(node.community)) return '#94A3B8';
    return PALETTE[node.community % PALETTE.length];
  }
  if (state.colorBy === 'cluster') {
    if (!hasNumber(node.cluster)) return '#94A3B8';
    return PALETTE[(Number(node.cluster) - 1 + PALETTE.length) % PALETTE.length];
  }
  if (state.colorBy === 'degree') {
    const maxDegree = state.graph ? Math.max(1, ...state.nodes.map(n => n.degree)) : 1;
    const ratio = Math.min(1, node.degree / maxDegree);
    // Cool (few links) to warm (hub), on a fixed hue ramp.
    const hue = 210 - 190 * ratio;
    return `hsl(${hue}, 70%, 52%)`;
  }
  return PALETTE[node.component % PALETTE.length];
}

function nodeRadius() {
  return Math.max(2, state.nodeSize * state.zoom * 0.5);
}

// --- Rendering ------------------------------------------------------------

function resizeCanvas() {
  const rect = refreshRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function draw() {
  const rect = refreshRect();
  const dpr = window.devicePixelRatio || 1;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#f6f4ef';
  ctx.fillRect(0, 0, rect.width, rect.height);

  setText('statZoom', `${state.zoom.toFixed(2)}x`);
  if (!state.nodes.length) return;

  const points = state.nodes.map((_, index) => screenPosition(index));
  const neighbourSet = state.selected === null ? null : neighboursOf(state.selected);

  drawEdges(points, neighbourSet);
  drawNodes(points, rect, neighbourSet);
  if (state.hover !== null) drawTooltip(points[state.hover], state.nodes[state.hover]);
}

// Warm = the projection separated images the model considers close.
function stressColor(value) {
  const hue = 210 - 200 * Math.max(0, Math.min(1, value / 0.6));
  return `hsl(${hue}, 75%, 45%)`;
}

function drawEdges(points, neighbourSet) {
  if (state.edgeOpacity <= 0) return;
  const useStress = state.edgeColor === 'stress' && stressAvailable();
  const stress = useStress ? edgeStressValues() : null;
  ctx.save();
  ctx.lineWidth = 1;
  state.edges.forEach((edge, index) => {
    const a = points[edge.source];
    const b = points[edge.target];
    const touchesSelection = neighbourSet !== null && (edge.source === state.selected || edge.target === state.selected);
    // With a community filter on, only the links inside it carry the structure
    // being asked about; the rest recede the same way non-neighbours do.
    const outsideCommunity = state.highlightCommunity !== null && !(
      state.nodes[edge.source]?.community === state.highlightCommunity
      && state.nodes[edge.target]?.community === state.highlightCommunity
    );

    if ((neighbourSet !== null && !touchesSelection) || outsideCommunity) {
      ctx.globalAlpha = state.edgeOpacity * 0.15;
      ctx.strokeStyle = '#15171a';
      ctx.lineWidth = 1;
    } else if (edge.bridge) {
      // A bridge holds two groups together; it should never be lost in the mesh.
      ctx.globalAlpha = Math.min(1, state.edgeOpacity * 2.2);
      ctx.strokeStyle = STRUCTURE_COLOR;
      ctx.lineWidth = touchesSelection ? 3.4 : 2.6;
    } else if (useStress) {
      const value = stress[index];
      // Low-stress links are the ones the map already shows correctly, so they
      // recede; the disagreements come forward.
      ctx.globalAlpha = state.edgeOpacity * (0.25 + 0.75 * Math.min(1, value / 0.6));
      ctx.strokeStyle = stressColor(value);
      ctx.lineWidth = touchesSelection ? 2.4 : (0.8 + 2.2 * Math.min(1, value / 0.6));
    } else if (state.edgeColor === 'uniform') {
      ctx.globalAlpha = state.edgeOpacity;
      ctx.strokeStyle = '#15171a';
      ctx.lineWidth = touchesSelection ? 2 : 1;
    } else {
      // Similarity drives both opacity and weight, so the strong structure reads first.
      ctx.globalAlpha = state.edgeOpacity * (0.35 + 0.65 * Math.max(0, edge.similarity));
      ctx.strokeStyle = touchesSelection ? '#111827' : '#15171a';
      ctx.lineWidth = touchesSelection ? 2 : (edge.mutual ? 1.4 : 0.8);
    }

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  });
  ctx.restore();
}

function drawNodes(points, rect, neighbourSet) {
  const useThumbs = state.showThumbs && state.nodes.length <= THUMB_NODE_LIMIT;
  for (let index = 0; index < state.nodes.length; index++) {
    const node = state.nodes[index];
    const point = points[index];
    const radius = nodeRadius();
    const margin = radius + 24;
    if (point.x < -margin || point.x > rect.width + margin || point.y < -margin || point.y > rect.height + margin) continue;

    const outsideCommunity = state.highlightCommunity !== null && node.community !== state.highlightCommunity;
    const dimmed = (neighbourSet !== null && index !== state.selected && !neighbourSet.has(index)) || outsideCommunity;
    ctx.globalAlpha = dimmed ? 0.18 : 1;
    const color = nodeColor(node);

    if (useThumbs) {
      const img = state.images.get(node.relative_path);
      if (!img) requestThumb(node);
      if (img) drawThumbNode(img, point, radius, color);
      else drawDotNode(point, radius, color);
    } else {
      drawDotNode(point, radius, color);
    }

    // An articulation point is an image the graph depends on: remove it and the
    // structure falls apart. Marked with the same signal colour as the bridges.
    if (node.articulation && !dimmed) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = STRUCTURE_COLOR;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius + 4, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (index === state.selected) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#111827';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius + 6, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }
}

function drawDotNode(point, radius, color) {
  ctx.beginPath();
  ctx.arc(point.x, point.y, Math.max(2, radius * 0.5), 0, Math.PI * 2);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.strokeStyle = 'rgba(0,0,0,0.30)';
  ctx.lineWidth = 1;
  ctx.stroke();
}

function drawThumbNode(img, point, radius, color) {
  const size = radius * 2;
  ctx.save();
  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,255,255,0.85)';
  ctx.fill();
  ctx.clip();
  const aspect = (img.naturalWidth || 1) / (img.naturalHeight || 1);
  const width = aspect >= 1 ? size * aspect : size;
  const height = aspect >= 1 ? size : size / aspect;
  const previousAlpha = ctx.globalAlpha;
  ctx.globalAlpha = previousAlpha * state.thumbOpacity;
  ctx.drawImage(img, point.x - width / 2, point.y - height / 2, width, height);
  ctx.globalAlpha = previousAlpha;
  ctx.restore();

  ctx.beginPath();
  ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
}

function drawTooltip(point, node) {
  const rect = currentRect();
  const width = 260;
  const height = 74;
  let x = point.x + 16;
  let y = point.y - height - 10;
  if (x + width > rect.width - 10) x = point.x - width - 16;
  if (y < 10) y = point.y + 16;
  if (x < 10) x = 10;

  ctx.save();
  ctx.fillStyle = '#202020';
  ctx.shadowColor = 'rgba(0,0,0,.45)';
  ctx.shadowBlur = 18;
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, width, height, 10); else ctx.rect(x, y, width, height);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = 'rgba(255,255,255,.18)';
  ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.font = '12px Geist, Arial';
  ctx.fillText(truncate(node.filename, 34), x + 12, y + 24);
  ctx.fillStyle = '#bbb';
  ctx.font = '11px Geist, Arial';
  ctx.fillText(`${node.degree} link(s) - component ${node.component + 1}`, x + 12, y + 43);
  ctx.fillText(hasNumber(node.cluster) ? `cluster ${node.cluster}` : 'no cluster', x + 12, y + 60);
  ctx.restore();
}

function truncate(value, max) {
  const text = String(value ?? '');
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

// --- Interaction ----------------------------------------------------------

function neighboursOf(index) {
  const set = new Set();
  for (const edge of state.edges) {
    if (edge.source === index) set.add(edge.target);
    else if (edge.target === index) set.add(edge.source);
  }
  return set;
}

function findNode(evt) {
  if (!state.nodes.length) return null;
  const rect = refreshRect();
  const mx = evt.clientX - rect.left;
  const my = evt.clientY - rect.top;
  let best = null;
  let bestDistance = Math.max(14, state.nodeSize * state.zoom * 0.6);
  for (let index = 0; index < state.nodes.length; index++) {
    const point = screenPosition(index);
    const distance = Math.hypot(mx - point.x, my - point.y);
    if (distance < bestDistance) { bestDistance = distance; best = index; }
  }
  return best;
}

function selectNode(index) {
  state.selected = index;
  // Picking a node is a question about one image; a community filter is a
  // question about the whole collection. Left combined, the graph showed only
  // their intersection - a handful of nodes - and choosing a community appeared
  // to do nothing at all.
  if (index !== null && state.highlightCommunity !== null) {
    state.highlightCommunity = null;
    renderStructure();
  }
  renderInspector();
  draw();
}

function renderInspector() {
  const body = $('graphInspectorBody');
  if (!body) return;
  if (state.selected === null || !state.nodes[state.selected]) {
    body.innerHTML = '<p class="help">Click a node to see its nearest neighbours and how similar they are.</p>';
    return;
  }
  const node = state.nodes[state.selected];
  const links = state.edges
    .filter(edge => edge.source === state.selected || edge.target === state.selected)
    .map(edge => ({
      other: edge.source === state.selected ? edge.target : edge.source,
      similarity: edge.similarity,
      mutual: edge.mutual,
    }))
    .sort((a, b) => b.similarity - a.similarity);

  const neighbourCards = links.map(link => {
    const other = state.nodes[link.other];
    return `
      <button class="gi-neighbour" type="button" data-node-index="${link.other}">
        <img src="${escapeHtml(thumbUrl(other, 96))}" alt="" loading="lazy" />
        <span class="gi-n-meta">
          <span class="gi-n-name">${escapeHtml(truncate(other.filename, 26))}</span>
          <span class="gi-n-score mono">${link.similarity.toFixed(4)}${link.mutual ? ' · mutual' : ''}</span>
        </span>
      </button>`;
  }).join('');

  body.innerHTML = `
    <div class="gi-hero">
      <img src="${escapeHtml(thumbUrl(node, 320))}" alt="" />
      <div class="gi-title">${escapeHtml(node.filename)}</div>
      <div class="gi-meta mono">${node.degree} link(s) · component ${node.component + 1}${hasNumber(node.cluster) ? ` · cluster ${node.cluster}` : ''}</div>
      <div class="gi-actions">
        <a class="search-open-link" href="${escapeHtml(imageUrl(node))}" target="_blank" rel="noopener">Open full image</a>
        <button class="sr-export" type="button" data-export-node
                title="Save this image and its nearest neighbours as a PDF, large enough to compare them">PDF</button>
        <button class="sr-export" type="button" data-export-node-csv
                title="Save the neighbour list as a CSV: rank, filename, similarity and whether the link is mutual">CSV</button>
      </div>
    </div>
    <div class="gi-section-title">Nearest neighbours</div>
    <div class="gi-neighbours">${neighbourCards || '<p class="help">This image has no edges at the current k and threshold.</p>'}</div>`;

  const exportButton = body.querySelector('[data-export-node]');
  if (exportButton) {
    exportButton.addEventListener('click', () => exportNodePdf(node, links, exportButton));
  }
  const csvButton = body.querySelector('[data-export-node-csv]');
  if (csvButton) {
    csvButton.addEventListener('click', () => exportNodeCsv(node, links));
  }

  body.querySelectorAll('[data-node-index]').forEach(button => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.nodeIndex);
      centerOnNode(index);
      selectNode(index);
    });
  });
}

/* The same two results as numbers.
 *
 * The columns differ between surfaces because the questions do, but the shape
 * is deliberately constant: an ordering column first, then filename and path,
 * then what the view knows, then the plot coordinates last. That is what makes
 * two exports from different panels line up in a spreadsheet.
 */
function exportCommunityCsv(community) {
  const nodes = nodesOfCommunity(community);
  if (!nodes.length) return;
  const header = ['position', 'filename', 'relative_path', 'community', 'cluster',
                  'links', 'load_bearing', 'x', 'y'];
  const rows = nodes.map((node, index) => [
    index + 1,
    node.filename,
    node.relative_path,
    community + 1,
    hasNumber(node.cluster) ? Number(node.cluster) : '',
    node.degree,
    node.articulation ? 'yes' : '',
    node.x ?? '',
    node.y ?? '',
  ]);
  const label = `Community ${community + 1}`;
  const report = saveCsv(header, rows, exportFilename('community', label, 'csv'));
  setText('graphMessage', report.ok
    ? `${label}: ${rows.length} rows exported.`
    : `CSV export failed: ${report.error}`);
  $('graphMessage').className = `status-banner ${report.ok ? 'ok' : 'warn'}`;
}

function exportNodeCsv(node, links) {
  const header = ['rank', 'filename', 'relative_path', 'similarity', 'mutual',
                  'cluster', 'community', 'x', 'y'];
  // The selected image is row 0 with no similarity: a neighbour list without
  // the image it belongs to cannot be read back months later.
  const row = (n, rank, similarity, mutual) => [
    rank, n.filename, n.relative_path, similarity, mutual,
    hasNumber(n.cluster) ? Number(n.cluster) : '',
    hasNumber(n.community) ? n.community + 1 : '',
    n.x ?? '', n.y ?? '',
  ];
  const rows = [row(node, 0, '', '')];
  links.forEach((link, position) => {
    const other = state.nodes[link.other];
    if (other) rows.push(row(other, position + 1, link.similarity.toFixed(6), link.mutual ? 'yes' : ''));
  });
  const report = saveCsv(header, rows, exportFilename('neighbours', node.filename, 'csv'));
  setText('graphMessage', report.ok
    ? `${node.filename}: ${rows.length} rows exported, the image itself included.`
    : `CSV export failed: ${report.error}`);
  $('graphMessage').className = `status-banner ${report.ok ? 'ok' : 'warn'}`;
}

// One image and its neighbourhood, as the same sheet a community produces.
//
// The selected image leads, then its neighbours in the order the graph ranks
// them, so the page reads as the question it answers: what does this model
// consider closest to this, and how close is close.
async function exportNodePdf(node, links, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = '...';
  try {
    const graph = state.graph || {};
    const items = [{
      relative_path: node.relative_path,
      caption: node.filename,
      detail: 'the selected image',
    }];
    links.forEach((link, position) => {
      const other = state.nodes[link.other];
      if (!other) return;
      items.push({
        relative_path: other.relative_path,
        caption: other.filename,
        detail: `#${position + 1}  ·  similarity ${link.similarity.toFixed(4)}`
          + (link.mutual ? '  ·  mutual' : ''),
      });
    });

    const report = await downloadFromServer('/api/exports/sheet', {
      projection_path: state.projectionPath,
      title: node.filename,
      subtitle: `${links.length} nearest neighbour${links.length === 1 ? '' : 's'} in the graph`,
      meta_lines: [
        `Neighbor graph · k=${graph.k ?? $('graphK')?.value ?? '?'}`
          + (hasNumber(node.community) ? ` · community ${node.community + 1}` : '')
          + (hasNumber(node.cluster) ? ` · cluster ${node.cluster}` : ''),
        `${graph.model_label || ''} · ${state.projectionPath.split('/').pop()}`.replace(/^ · /, ''),
      ].filter(line => line.trim()),
      items,
      filename: exportFilename('neighbours', node.filename, 'pdf'),
    });

    const notes = [];
    if (report.skipped) notes.push(`${report.skipped} missing from img/`);
    if (report.undersized) notes.push(`${report.undersized} smaller than 500px, shown at their own size`);
    setText('graphMessage', `${node.filename}: ${report.included} image${report.included === 1 ? '' : 's'} exported`
      + (notes.length ? ` · ${notes.join(' · ')}` : '.'));
    $('graphMessage').className = `status-banner ${notes.length ? 'warn' : 'ok'}`;
  } catch (err) {
    setText('graphMessage', `PDF export failed: ${err.message}`);
    $('graphMessage').className = 'status-banner warn';
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

// --- Structure panel ------------------------------------------------------
// The right-hand panel answers two different questions, so it has two tabs:
// "Selected" is about one image, "Structure" is about the collection as a whole.
// Putting the second one here keeps the left rail from growing, and uses space
// that sits empty whenever nothing is selected.

function showInspectorTab(name) {
  state.inspectorTab = name;
  document.querySelectorAll('[data-inspector-tab]').forEach(tab => {
    tab.classList.toggle('active', tab.dataset.inspectorTab === name);
  });
  document.querySelectorAll('[data-inspector-pane]').forEach(pane => {
    pane.classList.toggle('active', pane.dataset.inspectorPane === name);
  });
}

// Save a community as a contact sheet.
//
// The images are sent by path and rebuilt server-side at full resolution: the
// browser can only reach them through the thumbnail endpoint, which stops at
// 512px, and this document exists to be looked at closely.
async function exportCommunityPdf(community, button) {
  const nodes = nodesOfCommunity(community);
  if (!nodes.length) return;

  const label = `Community ${community + 1}`;
  const original = button.textContent;
  button.disabled = true;
  button.textContent = '...';
  try {
    const graph = state.graph || {};
    const report = await downloadFromServer('/api/exports/sheet', {
      projection_path: state.projectionPath,
      title: label,
      subtitle: `${nodes.length} image${nodes.length === 1 ? '' : 's'} of ${state.nodes.length}`
        + (hasNumber(graph.modularity) ? ` · graph modularity ${graph.modularity.toFixed(3)}` : ''),
      meta_lines: [
        `Neighbor graph · k=${graph.k ?? $('graphK')?.value ?? '?'} · ${state.edges.length} edges`,
        `${graph.model_label || ''} · ${state.projectionPath.split('/').pop()}`.replace(/^ · /, ''),
      ].filter(line => line.trim()),
      items: nodes.map(node => ({
        relative_path: node.relative_path,
        caption: node.filename,
        // What the graph knows about this image, so the sheet can be read back
        // against the view it came from.
        detail: [
          `${node.degree} link${node.degree === 1 ? '' : 's'}`,
          hasNumber(node.cluster) ? `cluster ${node.cluster}` : '',
          node.articulation ? 'load-bearing' : '',
        ].filter(Boolean).join('  ·  '),
      })),
      filename: exportFilename('community', label, 'pdf'),
    });
    const notes = [];
    if (report.skipped) notes.push(`${report.skipped} missing from img/`);
    if (report.undersized) notes.push(`${report.undersized} smaller than 500px, shown at their own size`);
    setText('graphMessage', `${label}: ${report.included} image${report.included === 1 ? '' : 's'} exported`
      + (notes.length ? ` · ${notes.join(' · ')}` : '.'));
    $('graphMessage').className = `status-banner ${notes.length ? 'warn' : 'ok'}`;
  } catch (err) {
    // A failure has to say so: a download that silently does nothing reads as a
    // broken button.
    setText('graphMessage', `PDF export failed: ${err.message}`);
    $('graphMessage').className = 'status-banner warn';
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function nodesOfCommunity(community) {
  return state.nodes.filter(node => node.community === community);
}

function thumbStrip(nodes, limit = 5) {
  return nodes.slice(0, limit).map(node =>
    `<img src="${escapeHtml(thumbUrl(node, 72))}" alt="" loading="lazy" title="${escapeHtml(node.filename)}" />`
  ).join('');
}

function renderStructure() {
  const body = $('graphStructureBody');
  if (!body) return;
  const graph = state.graph;
  if (!graph) {
    body.innerHTML = '<p class="help">Build the graph to see its structure.</p>';
    return;
  }
  if (graph.structure_available === false) {
    // Why it is unavailable decides what the reader should do about it.
    body.innerHTML = graph.structure_reason === 'no_edges'
      ? '<p class="help">No links survived the current settings, so there is no structure to read. Lower the minimum similarity, raise k, or turn off <em>only reciprocal edges</em>.</p>'
      : '<p class="help">The structural analysis needs the <code>networkx</code> package, which is not installed in this environment. The graph itself is unaffected.</p>';
    return;
  }

  // --- communities
  const sizes = new Map();
  for (const node of state.nodes) {
    if (!hasNumber(node.community)) continue;
    sizes.set(node.community, (sizes.get(node.community) || 0) + 1);
  }
  const ordered = [...sizes.entries()].sort((a, b) => b[1] - a[1]);
  const modularity = graph.modularity;
  const verdict = modularity === null ? ''
    : modularity >= 0.5 ? 'strongly distinct groups'
    : modularity >= 0.3 ? 'clear but overlapping groups'
    : 'a fairly homogeneous collection';

  // The row is a button, so the export control cannot live inside it; the two
  // sit side by side in a wrapper instead.
  const communityHtml = ordered.map(([community, size]) => {
    const color = PALETTE[community % PALETTE.length];
    const selected = state.highlightCommunity === community;
    return `
      <div class="structure-row-wrap">
        <button class="structure-row${selected ? ' selected' : ''}" type="button" data-community="${community}"
                title="Show only this community on the graph. Click again to show everything.">
          <span class="sr-dot" style="background:${color}"></span>
          <span class="sr-body">
            <span class="sr-title">Community ${community + 1}</span>
            <span class="sr-sub mono">${size} image${size === 1 ? '' : 's'}</span>
          </span>
          <span class="sr-thumbs">${thumbStrip(nodesOfCommunity(community), 3)}</span>  <!-- three, not five: with both export buttons the strip is clipped, and a
                 half-cut thumbnail reads as a rendering fault rather than a preview -->
        </button>
        <button class="sr-export" type="button" data-export-community="${community}"
                title="Save every image in this community as a PDF, large enough to see what is in them">PDF</button>
        <button class="sr-export" type="button" data-export-community-csv="${community}"
                title="Save this community as a CSV: filename, links, cluster and plot coordinates">CSV</button>
      </div>`;
  }).join('');

  // --- bridges
  const bridgeHtml = (graph.bridges || []).slice(0, 40).map(([a, b], index) => {
    const nodeA = state.nodes[a];
    const nodeB = state.nodes[b];
    if (!nodeA || !nodeB) return '';
    const edge = state.edges.find(e =>
      (e.source === a && e.target === b) || (e.source === b && e.target === a));
    return `
      <button class="structure-row bridge-row" type="button" data-bridge="${a},${b}"
              title="Frame this link on the graph">
        <span class="sr-body">
          <span class="sr-title">Bridge ${index + 1}</span>
          <span class="sr-sub mono">similarity ${edge ? edge.similarity.toFixed(4) : '--'}</span>
        </span>
        <span class="sr-pair">
          <img src="${escapeHtml(thumbUrl(nodeA, 72))}" alt="" loading="lazy" title="${escapeHtml(nodeA.filename)}" />
          <span class="sr-link"></span>
          <img src="${escapeHtml(thumbUrl(nodeB, 72))}" alt="" loading="lazy" title="${escapeHtml(nodeB.filename)}" />
        </span>
      </button>`;
  }).join('');

  // --- articulation points
  const articulation = (graph.articulation_points || [])
    .map(index => state.nodes[index])
    .filter(Boolean);
  const articulationHtml = articulation.slice(0, 24).map(node => `
    <button class="structure-thumb" type="button" data-node-index="${node.index}"
            title="${escapeHtml(node.filename)} — removing this image would split the graph">
      <img src="${escapeHtml(thumbUrl(node, 96))}" alt="" loading="lazy" />
    </button>`).join('');

  body.innerHTML = `
    <section class="structure-section">
      <h4>Communities</h4>
      <p class="help">
        Groups the graph finds on its own, by looking for images that link to each other far more than outward.
        ${modularity === null ? '' : `Modularity <strong>${modularity.toFixed(3)}</strong> — ${verdict}.`}
        ${graph.component_count > 1 ? `The graph also falls into ${graph.component_count} disconnected islands.` : 'The graph is fully connected, so components would show a single colour — this is the division that matters.'}
      </p>
      <div class="structure-list">${communityHtml || '<p class="help">No communities found.</p>'}</div>
    </section>

    <section class="structure-section">
      <h4>Bridges <span class="count mono">${(graph.bridges || []).length}</span></h4>
      <p class="help">
        Links whose removal would split the graph. Each one is the single connection holding two groups together —
        often the work that belongs to both. Drawn heavier and in amber on the graph.
      </p>
      <div class="structure-list">${bridgeHtml || '<p class="help">No bridges: every group is held together by more than one link.</p>'}</div>
    </section>

    <section class="structure-section">
      <h4>Load-bearing images <span class="count mono">${articulation.length}</span></h4>
      <p class="help">
        Remove one of these and the graph falls apart. They are ringed in amber on the graph.
      </p>
      <div class="structure-thumbs">${articulationHtml || '<p class="help">None: the graph has no single point of failure.</p>'}</div>
    </section>`;

  body.querySelectorAll('[data-export-community-csv]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();
      exportCommunityCsv(Number(button.dataset.exportCommunityCsv));
    });
  });
  body.querySelectorAll('[data-export-community]').forEach(button => {
    button.addEventListener('click', event => {
      event.stopPropagation();  // the row underneath filters the graph
      exportCommunityPdf(Number(button.dataset.exportCommunity), button);
    });
  });
  body.querySelectorAll('[data-community]').forEach(button => {
    button.addEventListener('click', () => {
      const community = Number(button.dataset.community);
      state.highlightCommunity = state.highlightCommunity === community ? null : community;
      // Drop any node selection: its neighbour highlight would mask the
      // community, leaving only the few nodes in both.
      if (state.highlightCommunity !== null && state.selected !== null) {
        state.selected = null;
        renderInspector();
      }
      renderStructure();
      draw();
    });
  });
  body.querySelectorAll('[data-bridge]').forEach(button => {
    button.addEventListener('click', () => {
      const [a, b] = button.dataset.bridge.split(',').map(Number);
      frameNodes([a, b]);
    });
  });
  body.querySelectorAll('[data-node-index]').forEach(button => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.nodeIndex);
      centerOnNode(index);
      selectNode(index);
      showInspectorTab('selected');
    });
  });
}

// Centre the view on a pair of nodes, so a bridge can be inspected as a unit.
function frameNodes(indices) {
  const rect = currentRect();
  const points = indices.map(index => screenPosition(index));
  const midX = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const midY = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  state.panX += rect.width / 2 - midX;
  state.panY += rect.height / 2 - midY;
  state.selected = indices[0];
  renderInspector();
  draw();
}

function centerOnNode(index) {
  const rect = currentRect();
  const point = screenPosition(index);
  state.panX += rect.width / 2 - point.x;
  state.panY += rect.height / 2 - point.y;
}

function zoomAt(factor, anchorX, anchorY) {
  const rect = currentRect();
  const previous = state.zoom;
  const next = Math.max(0.2, Math.min(12, previous * factor));
  if (next === previous) return;
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const ratio = next / previous;
  state.panX = anchorX - cx - ratio * (anchorX - cx - state.panX);
  state.panY = anchorY - cy - ratio * (anchorY - cy - state.panY);
  state.zoom = next;
}

function resetView() {
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  draw();
}

function fitView() {
  if (!state.nodes.length) { resetView(); return; }
  state.zoom = 1;
  state.panX = 0;
  state.panY = 0;
  const rect = refreshRect();
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let index = 0; index < state.nodes.length; index++) {
    const point = screenPosition(index);
    if (point.x < minX) minX = point.x;
    if (point.x > maxX) maxX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.y > maxY) maxY = point.y;
  }
  const pad = state.nodeSize + 24;
  const zoom = Math.max(0.2, Math.min(12, Math.min(
    rect.width / Math.max(1, (maxX - minX) + pad * 2),
    rect.height / Math.max(1, (maxY - minY) + pad * 2),
  )));
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  const midX = (minX + maxX) / 2;
  const midY = (minY + maxY) / 2;
  state.zoom = zoom;
  state.panX = cx - (cx + (midX - cx) * zoom);
  state.panY = cy - (cy + (midY - cy) * zoom);
  draw();
}

// --- Data loading ---------------------------------------------------------

async function loadProjections() {
  try {
    const data = await apiJson('/api/projections');
    state.projections = data.projections || [];
    const select = $('graphProjection');
    if (!state.projections.length) {
      select.innerHTML = '<option value="">No saved projections</option>';
      setText('graphSourceInfo', 'Generate a projection first — the graph reads its embeddings.');
      return;
    }
    select.innerHTML = '<option value="">Choose a projection...</option>' +
      state.projections.map(p => `<option value="${escapeHtml(p.relative_path)}">${escapeHtml(p.name)}</option>`).join('');

    // A projection can be passed in from the plot view's "graph" link.
    const requested = new URLSearchParams(window.location.search).get('projection');
    if (requested && state.projections.some(p => p.relative_path === requested)) {
      select.value = requested;
      await onProjectionChosen(requested);
    }
  } catch (err) {
    setText('graphSourceInfo', `Cannot list projections: ${err.message}`);
  }
}

async function onProjectionChosen(path) {
  state.projectionPath = path;
  $('graphBuildBtn').disabled = true;
  if (!path) {
    setText('graphSourceInfo', 'Choose a saved projection.');
    return;
  }
  setText('graphSourceInfo', 'Checking this projection...');
  try {
    const info = await apiJson(`/api/views/neighbor-graph/available?path=${encodeURIComponent(path)}`);
    if (!info.ok) {
      setText('graphSourceInfo', info.reason);
      $('graphSourceInfo').className = 'status-banner warn';
      return;
    }
    const label = info.model_label ? `${info.model_label} · ` : '';
    setText('graphSourceInfo', `${label}${info.image_count ?? '?'} images. Ready to build.`);
    $('graphSourceInfo').className = 'status-banner ok';
    $('graphBuildBtn').disabled = false;
  } catch (err) {
    setText('graphSourceInfo', `Cannot read this projection: ${err.message}`);
    $('graphSourceInfo').className = 'status-banner warn';
  }
}

async function buildGraph() {
  if (!state.projectionPath || state.busy) return;
  state.busy = true;
  $('graphBuildBtn').disabled = true;
  setText('graphMessage', 'Building the neighbor graph...');
  $('graphMessage').className = 'status-banner idle';
  setText('graphStatusChip', 'Building...');
  $('graphStatusChip').className = 'chip chip-warn dot';

  const thresholdRaw = Number($('graphThreshold').value);
  try {
    const data = await apiJson('/api/views/neighbor-graph', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        projection_path: state.projectionPath,
        k: Number($('graphK').value || 4),
        // 0 on the slider means "no threshold" rather than "similarity >= 0".
        min_similarity: thresholdRaw > 0 ? thresholdRaw / 100 : null,
        mutual_only: $('graphMutualOnly').checked,
      }),
    });
    state.graph = data;
    state.nodes = data.nodes;
    state.edges = data.edges;
    state.selected = null;
    state.hover = null;
    state.images.clear();
    thumbQueue.length = 0;
    thumbQueued.clear();
    thumbFailed.clear();

    applyLayout();
    invalidateStress();
    state.highlightCommunity = null;
    $('graphEmpty').hidden = true;
    setText('statNodes', String(data.node_count));
    setText('statEdges', String(data.edge_count));
    setText('statCommunities', data.structure_available ? String(data.community_count) : 'n/a');
    setText('statModularity', data.modularity === null || data.modularity === undefined ? 'n/a' : data.modularity.toFixed(3));
    setText('statBridges', data.structure_available ? String((data.bridges || []).length) : 'n/a');
    setText('statIsolated', String(data.nodes.filter(n => n.degree === 0).length));
    const unusable = (data.unusable_vectors || []).length;
    setText('graphMessage', `Graph built with k=${data.k}.` +
      (unusable ? ` ${unusable} image(s) had an unusable embedding vector and were treated as unrelated.` : ''));
    $('graphMessage').className = 'status-banner ok';
    setText('graphStatusChip', `${data.node_count} nodes · ${data.edge_count} edges`);
    $('graphStatusChip').className = 'chip chip-success dot';
    renderInspector();
    renderStructure();
    updateEdgeColorHint();
    fitView();
  } catch (err) {
    setText('graphMessage', `Graph failed: ${err.message}`);
    $('graphMessage').className = 'status-banner warn';
    setText('graphStatusChip', 'Graph failed');
    $('graphStatusChip').className = 'chip chip-danger dot';
  } finally {
    state.busy = false;
    $('graphBuildBtn').disabled = !state.projectionPath;
  }
}

function exportPng() {
  saveCanvasPng(canvas, exportFilename('graph', '', 'png'));
}

// --- Events ---------------------------------------------------------------

canvas.addEventListener('wheel', evt => {
  evt.preventDefault();
  const rect = refreshRect();
  zoomAt(evt.deltaY < 0 ? 1.1 : 0.9, evt.clientX - rect.left, evt.clientY - rect.top);
  scheduleDraw();
}, { passive: false });

canvas.addEventListener('mousedown', evt => {
  state.dragging = true;
  state.dragStart = { x: evt.clientX - state.panX, y: evt.clientY - state.panY, originX: evt.clientX, originY: evt.clientY };
});

window.addEventListener('mouseup', evt => {
  if (state.dragging && state.dragStart) {
    const moved = Math.hypot(evt.clientX - state.dragStart.originX, evt.clientY - state.dragStart.originY);
    // A click is a drag that went nowhere; anything further was a pan.
    if (moved < 4) {
      const index = findNode(evt);
      if (index !== null) selectNode(index);
      else selectNode(null);
    }
  }
  state.dragging = false;
});

canvas.addEventListener('mousemove', evt => {
  if (state.dragging && state.dragStart) {
    state.panX = evt.clientX - state.dragStart.x;
    state.panY = evt.clientY - state.dragStart.y;
  } else {
    state.hover = findNode(evt);
    canvas.style.cursor = state.hover === null ? 'grab' : 'pointer';
  }
  scheduleDraw();
});

canvas.addEventListener('mouseleave', () => { state.hover = null; scheduleDraw(); });

$('graphProjection').addEventListener('change', e => onProjectionChosen(e.target.value));
$('graphBuildBtn').addEventListener('click', buildGraph);

$('graphK').addEventListener('input', e => setText('graphKValue', e.target.value));
$('graphThreshold').addEventListener('input', e => {
  const value = Number(e.target.value);
  setText('graphThresholdValue', value > 0 ? (value / 100).toFixed(2) : 'off');
});

$('graphLayout').addEventListener('change', e => {
  state.layout = e.target.value;
  localStorage.setItem('imagecluster.graphLayout', state.layout);
  if (!state.nodes.length) return;
  setText('graphMessage', state.layout === 'force' ? 'Relaxing layout...' : 'Using projection coordinates.');
  // Let the message paint before the synchronous relaxation blocks the thread.
  // setTimeout rather than requestAnimationFrame: rAF is throttled to zero in a
  // background tab, which would leave the view stuck on "Relaxing layout...".
  setTimeout(() => {
    applyLayout();
    invalidateStress();
    updateEdgeColorHint();
    setText('graphMessage', state.layout === 'force' ? 'Force layout applied.' : 'Using projection coordinates.');
    fitView();
  }, 30);
});

// Projection stress is only meaningful where the positions come from the
// projection rather than from the links themselves.
function updateEdgeColorHint() {
  const hint = $('edgeColorHint');
  if (hint) hint.hidden = !(state.edgeColor === 'stress' && !stressAvailable());
}

document.querySelectorAll('[data-inspector-tab]').forEach(tab => {
  tab.addEventListener('click', () => showInspectorTab(tab.dataset.inspectorTab));
});

$('graphEdgeColor').addEventListener('change', e => {
  state.edgeColor = e.target.value;
  localStorage.setItem('imagecluster.graphEdgeColor', state.edgeColor);
  updateEdgeColorHint();
  draw();
});

$('graphColorBy').addEventListener('change', e => {
  state.colorBy = e.target.value;
  localStorage.setItem('imagecluster.graphColorBy', state.colorBy);
  draw();
});

$('graphShowThumbs').addEventListener('change', e => {
  state.showThumbs = e.target.checked;
  localStorage.setItem('imagecluster.graphShowThumbs', state.showThumbs ? 'true' : 'false');
  draw();
});

$('graphNodeSize').addEventListener('input', e => {
  state.nodeSize = Number(e.target.value || 26);
  localStorage.setItem('imagecluster.graphNodeSize', String(state.nodeSize));
  scheduleDraw();
});

$('graphOpacity').addEventListener('input', e => {
  state.thumbOpacity = clamp01(Number(e.target.value) / 100, 1);
  localStorage.setItem('imagecluster.graphOpacity', String(state.thumbOpacity));
  setText('graphOpacityValue', `${Math.round(state.thumbOpacity * 100)}%`);
  scheduleDraw();
});

$('graphEdgeOpacity').addEventListener('input', e => {
  state.edgeOpacity = clamp01(Number(e.target.value) / 100, 0.35);
  localStorage.setItem('imagecluster.graphEdgeOpacity', String(state.edgeOpacity));
  setText('graphEdgeOpacityValue', `${Math.round(state.edgeOpacity * 100)}%`);
  scheduleDraw();
});

$('graphZoomIn').addEventListener('click', () => { const r = currentRect(); zoomAt(1.2, r.width / 2, r.height / 2); draw(); });
$('graphZoomOut').addEventListener('click', () => { const r = currentRect(); zoomAt(1 / 1.2, r.width / 2, r.height / 2); draw(); });
$('graphFit').addEventListener('click', fitView);
$('graphReset').addEventListener('click', resetView);
$('graphExportPng').addEventListener('click', exportPng);

// --- Help panel -----------------------------------------------------------
// The view answers a different question from the scatter plot, and the difference
// is not self-evident, so the explanation is one click away from every entry point
// rather than buried in the README.
function setHelpOpen(open) {
  const modal = $('graphHelpModal');
  if (!modal) return;
  modal.hidden = !open;
  if (open) modal.querySelector('[data-close-help]')?.focus();
}

document.querySelectorAll('[data-open-help]').forEach(el => el.addEventListener('click', () => setHelpOpen(true)));
document.querySelectorAll('[data-close-help]').forEach(el => el.addEventListener('click', () => setHelpOpen(false)));

function helpIsOpen() {
  const modal = $('graphHelpModal');
  return Boolean(modal && !modal.hidden);
}

window.addEventListener('keydown', event => {
  // Escape closes the help panel first, wherever focus happens to be.
  if (event.key === 'Escape' && helpIsOpen()) { setHelpOpen(false); return; }
  const tag = document.activeElement && document.activeElement.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (event.key === 'Escape') {
    state.highlightCommunity = null;
    renderStructure();
    selectNode(null);
    return;
  }
  switch (event.key.toLowerCase()) {
    case 'f': event.preventDefault(); fitView(); break;
    case 'r': event.preventDefault(); resetView(); break;
    case 'b': if (!$('graphBuildBtn').disabled) { event.preventDefault(); buildGraph(); } break;
    case '?': event.preventDefault(); setHelpOpen(true); break;
    default: break;
  }
});

let resizeFrame = null;
window.addEventListener('resize', () => {
  if (resizeFrame) cancelAnimationFrame(resizeFrame);
  resizeFrame = requestAnimationFrame(() => { resizeFrame = null; resizeCanvas(); });
});
if ('ResizeObserver' in window) {
  new ResizeObserver(() => {
    if (resizeFrame) cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => { resizeFrame = null; resizeCanvas(); });
  }).observe(canvas.parentElement || canvas);
}

function init() {
  $('graphLayout').value = state.layout;
  $('graphColorBy').value = state.colorBy;
  $('graphEdgeColor').value = state.edgeColor;
  updateEdgeColorHint();
  $('graphShowThumbs').checked = state.showThumbs;
  $('graphNodeSize').value = String(state.nodeSize);
  $('graphOpacity').value = String(Math.round(state.thumbOpacity * 100));
  setText('graphOpacityValue', `${Math.round(state.thumbOpacity * 100)}%`);
  $('graphEdgeOpacity').value = String(Math.round(state.edgeOpacity * 100));
  setText('graphEdgeOpacityValue', `${Math.round(state.edgeOpacity * 100)}%`);
  resizeCanvas();
  loadProjections();
}

init();
