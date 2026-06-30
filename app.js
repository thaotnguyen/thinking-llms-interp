"use strict";

// Data sources. Relative `data/` paths resolve correctly both on GitHub Pages
// (served from a subpath) and via the local Flask dev server (which serves the
// same paths). Precomputed by build_static.py / served by app.py.
const SRC = {
  index: "data/index.json",
  grid: (model) => `data/grid/${model}.json`,
  data: (model, layer) => `data/annotated_${model}_layer${layer}.json`,
};

const state = {
  index: {},        // {model: [layers]}
  model: null,
  grid: null,       // {layers, sizes, scores}
  layer: null,      // selected layer (number)
  size: null,       // selected n_clusters (string)
  data: null,       // current annotation file (for state.layer)
  traceIdx: null,   // selected trace index
};

const $ = (id) => document.getElementById(id);

// ---- cluster color palette (Tableau 20: distinct but muted, up to 20 clusters) ----
const CLUSTER_PALETTE = [
  "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1",
  "#76b7b2", "#edc948", "#ff9da7", "#9c755f", "#bab0ac",
  "#a0cbe8", "#ffbe7d", "#8cd17d", "#ff9d9a", "#d4a6c8",
  "#86bcb6", "#b6992d", "#d37295", "#79706e", "#fabfd2",
];
function clusterColor(id) {                    // saturated (legend swatch / detailed accent)
  return CLUSTER_PALETTE[((id % CLUSTER_PALETTE.length) + CLUSTER_PALETTE.length) % CLUSTER_PALETTE.length];
}

// ---- grid heatmap color: matplotlib/seaborn RdBu (matches visualize_sae.py) ----
// ColorBrewer RdBu-11, ordered blue -> white -> red so low scores are blue, high scores red.
const RDBU = [
  [5, 48, 97], [33, 102, 172], [67, 147, 195], [146, 197, 222], [209, 229, 240],
  [247, 247, 247],
  [253, 219, 199], [244, 165, 130], [214, 96, 77], [178, 24, 43], [103, 0, 31],
];
function gridColor(t) {                          // t in [0,1]
  const x = Math.min(1, Math.max(0, t)) * (RDBU.length - 1);
  const i = Math.min(RDBU.length - 2, Math.floor(x));
  const u = x - i;
  const c = RDBU[i].map((a, k) => Math.round(a + (RDBU[i + 1][k] - a) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

// ---- init ------------------------------------------------------------------
async function init() {
  state.index = await (await fetch(SRC.index)).json();
  const modelSel = $("model-select");
  modelSel.innerHTML = "";
  const models = Object.keys(state.index);
  if (models.length === 0) {
    $("trace-view").innerHTML = '<p class="hint">No annotation files found. Run annotate_thinking_grid.py first.</p>';
    return;
  }
  models.forEach((m) => {
    const o = document.createElement("option");
    o.value = m; o.textContent = m;
    modelSel.appendChild(o);
  });
  modelSel.onchange = onModelChange;
  await onModelChange();
}

async function onModelChange() {
  state.model = $("model-select").value;
  state.grid = await (await fetch(SRC.grid(state.model))).json();

  // default selection: best (layer, size) by final score
  let best = null;
  for (const layer of state.grid.layers) {
    const row = state.grid.scores[String(layer)] || {};
    for (const size of state.grid.sizes) {
      const v = row[String(size)];
      if (v === null || v === undefined) continue;
      if (!best || v > best.v) best = { layer, size: String(size), v };
    }
  }
  state.layer = best ? best.layer : state.grid.layers[0];
  state.size = best ? best.size : String(state.grid.sizes[0]);
  state.traceIdx = null;

  renderGrid();
  await loadLayer(state.layer);
  renderMetrics();
  renderLegend();
  $("trace-view").innerHTML = '<p class="hint">Select a trace from the left.</p>';
}

async function loadLayer(layer) {
  state.data = await (await fetch(SRC.data(state.model, layer))).json();
  // keep trace selection across layers (same trace order); drop if out of range
  if (state.traceIdx !== null && state.traceIdx >= state.data.traces.length) {
    state.traceIdx = null;
  }
  buildTraceList();
}

// ---- grid selector ---------------------------------------------------------
function renderGrid() {
  const g = state.grid;
  // normalization range across all present scores
  let min = Infinity, max = -Infinity, bestKey = null;
  for (const layer of g.layers) {
    const row = g.scores[String(layer)] || {};
    for (const size of g.sizes) {
      const v = row[String(size)];
      if (v === null || v === undefined) continue;
      if (v < min) min = v;
      if (v > max) { max = v; bestKey = `${layer}:${size}`; }
    }
  }
  const span = max - min || 1;

  const sizesDesc = [...g.sizes].sort((a, b) => b - a);
  let html = '<table class="grid"><thead><tr><th class="corner">k \\ L</th>';
  g.layers.forEach((l) => { html += `<th>${l}</th>`; });
  html += "</tr></thead><tbody>";

  sizesDesc.forEach((size) => {
    html += `<tr><th>${size}</th>`;
    g.layers.forEach((layer) => {
      const v = (g.scores[String(layer)] || {})[String(size)];
      if (v === null || v === undefined) {
        html += '<td class="empty">·</td>';
        return;
      }
      const t = (v - min) / span;
      const isSel = state.layer === layer && state.size === String(size);
      const isBest = bestKey === `${layer}:${size}`;
      const cls = ["cell", isSel ? "sel" : "", isBest ? "best" : ""].filter(Boolean).join(" ");
      const fg = (t > 0.82 || t < 0.18) ? "#fff" : "#1f2430";  // legible on dark blue/red ends
      html += `<td class="${cls}" style="background:${gridColor(t)};color:${fg}"
        data-layer="${layer}" data-size="${size}" title="layer ${layer}, k=${size} — final ${v.toFixed(3)}">${v.toFixed(2)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody></table>";
  $("grid-select").innerHTML = html;

  $("grid-select").querySelectorAll("td.cell").forEach((td) => {
    td.onclick = () => selectCell(Number(td.dataset.layer), td.dataset.size);
  });
}

async function selectCell(layer, size) {
  const layerChanged = layer !== state.layer;
  state.layer = layer;
  state.size = size;
  renderGrid();
  if (layerChanged) await loadLayer(layer);
  renderMetrics();
  renderLegend();
  if (state.traceIdx !== null) renderTrace();
  else $("trace-view").innerHTML = '<p class="hint">Select a trace from the left.</p>';
}

// ---- sidebar panels --------------------------------------------------------
function buildTraceList() {
  const ul = $("trace-list");
  ul.innerHTML = "";
  state.data.traces.forEach((t, i) => {
    const li = document.createElement("li");
    li.className = i === state.traceIdx ? "active" : "";
    const ds = document.createElement("span");
    ds.className = "ds"; ds.textContent = t.dataset || "?";
    const ok = document.createElement("span");
    ok.textContent = t.is_correct ? "✓" : "✗";
    ok.className = t.is_correct ? "badge-ok" : "badge-no";
    const q = document.createElement("span");
    q.className = "q";
    q.textContent = (t.diagnosis || t.question || t.pmcid || `trace ${i}`);
    li.append(ds, ok, q);
    li.onclick = () => { state.traceIdx = i; buildTraceList(); renderTrace(); };
    ul.appendChild(li);
  });
}

function renderMetrics() {
  const m = (state.data.metrics_by_size || {})[state.size] || {};
  const fmt = (v) => (v === null || v === undefined) ? "–" : Number(v).toFixed(3);
  const items = [
    ["Layer", state.layer],
    ["n_clusters", state.size],
    ["Final score", fmt(m.final_score)],
    ["F1", fmt(m.avg_f1)],
    ["Completeness", fmt(m.avg_confidence)],
    ["Sem. orth.", fmt(m.semantic_orthogonality_score)],
  ];
  $("metrics").innerHTML = items.map(([label, v]) =>
    `<div class="metric"><div class="label">${label}</div><div class="value">${v}</div></div>`
  ).join("");
}

function renderLegend() {
  const cats = (state.data.categories_by_size || {})[state.size] || [];
  $("legend").innerHTML = cats.map((c) => {
    const [id, title, desc] = c;
    const color = clusterColor(Number(id));
    const safeDesc = (desc || "").replace(/"/g, "&quot;");
    return `<span class="chip" title="${safeDesc}">
      <span class="swatch" style="background:${color}"></span>
      <b>${id}</b> ${title}
    </span>`;
  }).join("");
}

function catTitle(id) {
  const cats = (state.data.categories_by_size || {})[state.size] || [];
  const found = cats.find((c) => String(c[0]) === String(id));
  return found ? found[1] : `cluster ${id}`;
}

function renderTrace() {
  const t = state.data.traces[state.traceIdx];
  const labels = (t.labels_by_size || {})[state.size] || [];
  const maxAct = Math.max(0.0001, ...labels.filter(Boolean).map((l) => Math.abs(l.activation)));

  let html = `<div class="trace-header">
      <div><b>${t.dataset || "?"}</b> · ${t.pmcid || ""} ·
        <span class="${t.is_correct ? "badge-ok" : "badge-no"}">${t.is_correct ? "correct" : "incorrect"}</span></div>
      ${t.diagnosis ? `<div class="dx"><span class="k">diagnosis:</span> ${escapeHtml(t.diagnosis)}</div>` : ""}
      ${t.gold_answer ? `<div class="dx gold"><span class="k">gold:</span> ${escapeHtml(t.gold_answer)}</div>` : ""}
    </div>`;

  html += '<div class="trace-view">';
  t.sentences.forEach((s, i) => {
    const lab = labels[i];
    const color = lab ? clusterColor(lab.cluster_id) : "#ccc";
    const barW = lab ? Math.round((Math.abs(lab.activation) / maxAct) * 80) : 0;
    html += `<div class="sentence-block" style="border-left-color:${color}">
      <div class="head">
        <span class="lbl">${lab ? `${lab.cluster_id} · ${escapeHtml(catTitle(lab.cluster_id))}` : "unlabeled"}</span>
        <span class="act">${lab ? `act ${lab.activation}` : ""}<span class="actbar" style="width:${barW}px;background:${color}"></span></span>
      </div>
      <div class="txt">${escapeHtml(s)}</div>
    </div>`;
  });
  html += "</div>";
  $("trace-view").innerHTML = html;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
}

init();
