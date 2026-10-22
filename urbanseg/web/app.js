// UrbanSeg web app: choose an area on the map, review the cost, run, explore.
// Plain ES module, no build step. Talks to the JSON API in urbanseg/api.py.
import { initGold, openFromJob } from './gold.js';

const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
const LEGEND_ORDER = ['residential', 'commercial', 'free_land', 'other'];
const UNIT_FMT = { m2: [0, 'm²'], ha: [2, 'ha'], acres: [2, 'acres'], sq_yd: [0, 'sq yd'], guntas: [1, 'guntas'] };
const STYLE = {
  zone: { strokeColor: '#ffffff', strokeWeight: 2, fillColor: '#ffffff', fillOpacity: 0.08,
          editable: true, draggable: true, zIndex: 5 },
  draft: { strokeColor: '#ffffff', strokeWeight: 2, fillColor: '#ffffff', fillOpacity: 0.05,
           clickable: false, zIndex: 6 },
  extent: { strokeColor: '#4f8ef7', strokeWeight: 2, fillOpacity: 0, clickable: false, zIndex: 3 },
  grid: { strokeColor: '#4f8ef7', strokeOpacity: 0.55, strokeWeight: 1, clickable: false, zIndex: 3 },
  coverage: { strokeColor: '#ffffff', strokeOpacity: 0.35, strokeWeight: 1, fillOpacity: 0,
              clickable: false, zIndex: 1 },
};

const S = {
  cfg: null, map: null, mapsOk: false, mode: 'zone',
  zone: null, point: null, pointMark: null, draw: null,
  plan: null, planSeq: 0, planTimer: null, gridLines: [],
  job: null, pollTimer: null, jobId: null, result: null,
  overlays: [], canvases: {}, urls: {},
  hidden: new Set(), version: 'fused', scope: 'zone',
  unit: store('unit') || 'ha', viewer: null,
};
window.__urbanseg = S;  // read-only handle for debugging and browser tests

// ------------------------------------------------------------------ helpers

function store(k, v) {
  try {
    if (v === undefined) return localStorage.getItem('urbanseg.' + k);
    localStorage.setItem('urbanseg.' + k, v);
  } catch { /* storage may be unavailable */ }
  return null;
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  let body = null;
  try { body = await r.json(); } catch { /* not JSON */ }
  if (!r.ok) {
    const e = body?.error || { code: 'http_' + r.status,
      message: typeof body?.detail === 'string' ? body.detail : `HTTP ${r.status}` };
    throw Object.assign(new Error(e.message), { code: e.code, hint: e.hint, status: r.status });
  }
  return body;
}
const postJSON = (path, data) => api(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data) });

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const rgb = c => `rgb(${c.join(',')})`;
const colorKey = (r, g, b) => (r << 16) | (g << 8) | b;
const classById = id => S.cfg.classes[id];
const className = n => S.cfg.classes.find(c => c.name === n);

function fmtArea(units, unit = S.unit) {
  const [dp, label] = UNIT_FMT[unit];
  const v = units?.[unit] ?? 0;
  return `${v.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp })} ${label}`;
}
function haToUnits(ha) {
  const m2 = ha * 1e4;
  return { m2, ha, acres: m2 / 4046.8564224, sq_yd: m2 / 0.83612736, guntas: m2 / 101.17141056 };
}

// Web-Mercator world pixels, exactly as urbanseg/imagery.py
const WORLD = z => 256 * 2 ** z;
function latLngToWorld(lat, lng, z) {
  const w = WORLD(z), s = Math.sin(lat * Math.PI / 180);
  return [(lng + 180) / 360 * w, (0.5 - Math.log((1 + s) / (1 - s)) / (4 * Math.PI)) * w];
}
function worldToLatLng(x, y, z) {
  const w = WORLD(z);
  return { lat: Math.atan(Math.sinh(Math.PI * (1 - 2 * y / w))) * 180 / Math.PI, lng: x / w * 360 - 180 };
}

// ------------------------------------------------------------------ startup

init();

async function init() {
  applyTheme(store('theme'));
  $('#theme').onclick = () => {
    const next = { null: 'dark', dark: 'light', light: null }[store('theme') ?? 'null'] ?? null;
    applyTheme(next);
    store('theme', next ?? '');
  };
  try {
    S.cfg = await api('/api/config');
  } catch (e) {
    showError({ message: 'The service is not reachable: ' + e.message });
    return;
  }
  renderChips();
  renderLegend();
  renderModels();
  initGold({ api, esc, showError, cfg: S.cfg });
  $('#gold-correct').onclick = () => openFromJob(S.jobId, $('#result-title').textContent,
    $('#gold-seed').value).catch(showError);
  for (const [g, s] of Object.entries(S.cfg.grid_sizes)) {
    $('#grid').insertAdjacentHTML('beforeend', `<option value="${g}" ${g === '2' ? 'selected' : ''}>${esc(s.name)}: ${Math.round(s.extent_m[0])} × ${Math.round(s.extent_m[1])} m (${s.calls} call${s.calls > 1 ? 's' : ''})</option>`);
  }
  $('#unit').value = S.unit;
  bindControls();
  if (S.cfg.maps.key) {
    try {
      await loadMaps(S.cfg.maps.key);
      await initMap();
    } catch (e) {
      mapOff(e.message);
    }
  } else {
    mapOff(S.cfg.maps.reason || 'no Google Maps key is configured');
  }
  renderHistory();
  schedulePlan();
}

function applyTheme(t) {
  if (t === 'dark' || t === 'light') document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
}

function renderChips() {
  const c = S.cfg;
  const chip = (on, yes, no) => `<span class="chip ${on ? 'on' : 'off'}">${esc(on ? yes : no)}</span>`;
  $('#chips').innerHTML = chip(c.coordinates_enabled, 'Imagery: Google key set', 'Imagery: no Google key')
    + chip(c.footprint_index, 'Footprint index built', 'No footprint index')
    + `<span class="chip">Fusion: ${esc(c.fusion_mode)}</span>`;
}

function renderModels() {
  // a choice only when more than one model is installed; else the server default
  const ms = (S.cfg.models || []).filter(m => m.available);
  const sel = $('#opt-model');
  sel.innerHTML = ms.map(m => `<option value="${esc(m.id)}" ${m.id === S.cfg.default_model ? 'selected' : ''}>${esc(m.label)}${m.id === S.cfg.default_model ? ' (default)' : ''}</option>`).join('');
  const about = () => { $('#opt-model-about').textContent = ms.find(m => m.id === sel.value)?.about || ''; };
  sel.onchange = about;
  about();
  document.querySelectorAll('.model-opt').forEach(el => { el.hidden = ms.length < 2; });
}

const chosenModel = () => ($('#opt-model').hidden ? undefined : $('#opt-model').value || undefined);

function renderLegend() {
  $('#legend').innerHTML = LEGEND_ORDER.map(n => className(n))
    .map(c => `<div><i class="sw" style="background:${rgb(c.color)}"></i>${esc(c.label)}</div>`).join('');
}

function loadMaps(key) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error('Google Maps did not load in time (network?)')), 20000);
    window.__urbansegMapsReady = () => { clearTimeout(timer); resolve(); };
    // Google calls this when the key is not valid for the Maps JavaScript API
    window.gm_authFailure = () => mapOff('Google rejected the key for the Maps JavaScript API. '
      + 'Enable the "Maps JavaScript API" for this key in the Google Cloud console, '
      + 'or set GOOGLE_MAPS_BROWSER_KEY to a key that has it.');
    const s = document.createElement('script');
    s.src = 'https://maps.googleapis.com/maps/api/js?' + new URLSearchParams({
      key, v: 'weekly', callback: '__urbansegMapsReady', loading: 'async' });
    s.async = true;
    s.onerror = () => { clearTimeout(timer); reject(new Error('Google Maps could not be loaded')); };
    document.head.append(s);
  });
}

async function initMap() {
  await google.maps.importLibrary('maps');
  await google.maps.importLibrary('core');
  const m = S.map = new google.maps.Map($('#map'), {
    center: { lat: 17.42, lng: 78.44 }, zoom: 12, mapTypeId: 'satellite', tilt: 0,
    streetViewControl: false, rotateControl: false, fullscreenControl: false,
    clickableIcons: false, gestureHandling: 'greedy',
    mapTypeControlOptions: { mapTypeIds: ['satellite', 'hybrid', 'roadmap'] },
  });
  S.mapsOk = true;
  const [w, s, e, n] = S.cfg.coverage_bbox;
  new google.maps.Rectangle({ map: m, bounds: { west: w, south: s, east: e, north: n }, ...STYLE.coverage });
  m.data.setStyle({ strokeColor: '#ffffff', strokeWeight: 2, fillOpacity: 0, clickable: false, zIndex: 4 });
  m.addListener('click', onMapClick);
  m.addListener('dblclick', () => { if (S.draw?.kind === 'poly') finishPoly(); });
  m.addListener('mousemove', onMapMove);
  m.addListener('mouseout', () => { $('#readout').hidden = true; });
}

function mapOff(why) {
  S.mapsOk = false;
  $('#map-off').hidden = false;
  $('#map-off-why').textContent = why;
  if (S.mode === 'zone') setMode('point');
}

// ------------------------------------------------------------------ controls

function bindControls() {
  $$('.tabs [data-mode]').forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $('#locate-form').onsubmit = ev => { ev.preventDefault(); locate($('#locate').value); };
  $('#draw-rect').onclick = () => startDraw('rect');
  $('#draw-poly').onclick = () => startDraw('poly');
  $('#clear-zone').onclick = () => { cancelDraw(); clearZone(); schedulePlan(); };
  $('#grid').onchange = schedulePlan;
  $('#opt-fuse').onchange = schedulePlan;
  $('#opt-quality').onchange = schedulePlan;
  $('#geojson-use').onclick = () => useGeojson($('#geojson-text').value);
  $('#geojson-file').onchange = async ev => {
    const f = ev.target.files[0];
    if (f) { $('#geojson-text').value = await f.text(); useGeojson($('#geojson-text').value); }
  };
  $('#upload-file').onchange = schedulePlan;
  $$('input[name=scale]').forEach(r => r.onchange = () => {
    $$('.sub-opts').forEach(d => { d.hidden = d.dataset.for !== r.value; });
    schedulePlan();
  });
  $('#run').onclick = run;
  $('#cancel').onclick = cancelJob;
  $('#unit').onchange = () => { S.unit = $('#unit').value; store('unit', S.unit); renderStats(); renderPlan(); };
  $$('#scope [data-scope]').forEach(b => b.onclick = () => { S.scope = b.dataset.scope; renderStats(); });
  $$('#version [data-ver]').forEach(b => b.onclick = () => { S.version = b.dataset.ver; refreshLayers(); });
  $('#opacity').oninput = () => {
    const v = +$('#opacity').value;
    S.overlays.forEach(o => o.kind === 'classes' && o.ov.setOpacity(v));
    if (S.viewer) drawViewer();
  };
  ['lay-lowconf', 'lay-imagery', 'lay-outside'].forEach(id => { $('#' + id).onchange = refreshLayers; });
  $('#viewer-close').onclick = closeViewer;
  document.addEventListener('keydown', ev => { if (ev.key === 'Escape') cancelDraw(); });
}

function setMode(mode) {
  if (mode === 'zone' && !S.mapsOk) {
    showError({ message: 'Drawing a zone needs the map. Use Point (with “Go to”) or Import instead.' });
    mode = 'point';
  }
  S.mode = mode;
  cancelDraw();
  $$('.tabs [data-mode]').forEach(b => b.setAttribute('aria-selected', String(b.dataset.mode === mode)));
  $$('.panel .mode').forEach(d => { d.hidden = d.dataset.mode !== mode; });
  $('#locate-form').hidden = mode === 'upload';
  if (S.zone?.shape) S.zone.shape.setMap(mode === 'zone' ? S.map : null);
  if (S.pointMark) S.pointMark.setMap(mode === 'point' ? S.map : null);
  if (S.map && !S.result) S.map.data.forEach(f => mode === 'geojson' || S.map.data.remove(f));
  // the previous mode's plan must not linger while the new one is computed
  S.plan = null;
  clearGrid();
  $('#plan').innerHTML = '<span class="muted">…</span>';
  $('#run').disabled = true;
  schedulePlan();
}

async function locate(text) {
  $('#locate-msg').textContent = '';
  if (!text.trim()) return;
  try {
    const p = await postJSON('/api/locate', { text });
    $('#locate-msg').textContent = `${p.lat.toFixed(6)}, ${p.lng.toFixed(6)}`
      + (p.in_coverage ? '' : ' (outside greater Hyderabad)');
    if (S.map) { S.map.panTo(p); if (S.map.getZoom() < 16) S.map.setZoom(17); }
    if (S.mode === 'point' || S.mode === 'zone' && !S.zone && !S.mapsOk) {
      if (S.mode !== 'point') setMode('point');
      setPoint(p.lat, p.lng);
    }
  } catch (e) {
    $('#locate-msg').innerHTML = esc(e.message) + (e.code === 'swapped' && e.hint
      ? ` <button type="button" id="swap-fix" class="ghost">Use ${e.hint.lat}, ${e.hint.lng}</button>` : '');
    const b = $('#swap-fix');
    if (b) b.onclick = () => { $('#locate').value = `${e.hint.lat}, ${e.hint.lng}`; locate($('#locate').value); };
  }
}

function setPoint(lat, lng) {
  S.point = { lat, lng };
  if (S.map) {
    S.pointMark?.setMap(null);
    S.pointMark = new google.maps.Circle({ map: S.map, center: S.point, radius: 6, strokeColor: '#ffffff',
      strokeWeight: 2, fillColor: '#4f8ef7', fillOpacity: 1, clickable: false, zIndex: 7 });
  }
  schedulePlan();
}

// ------------------------------------------------------------------ drawing

function startDraw(kind) {
  if (!S.mapsOk) return;
  if (S.draw?.kind === kind) { cancelDraw(); return; }
  cancelDraw();
  clearZone();
  S.draw = { kind, path: [], anchor: null, preview: null, line: null };
  S.map.setOptions({ draggableCursor: 'crosshair', disableDoubleClickZoom: true });
  $('#draw-rect').setAttribute('aria-pressed', String(kind === 'rect'));
  $('#draw-poly').setAttribute('aria-pressed', String(kind === 'poly'));
}

function cancelDraw() {
  if (!S.draw) return;
  S.draw.preview?.setMap(null);
  S.draw.line?.setMap(null);
  S.draw = null;
  S.map?.setOptions({ draggableCursor: null, disableDoubleClickZoom: false });
  $('#draw-rect').setAttribute('aria-pressed', 'false');
  $('#draw-poly').setAttribute('aria-pressed', 'false');
}

function clearZone() {
  S.zone?.shape?.setMap(null);
  S.zone = null;
  if (S.map && !S.result) S.map.data.forEach(f => S.map.data.remove(f));
}

function boundsOf(a, b) {
  return { west: Math.min(a.lng(), b.lng()), east: Math.max(a.lng(), b.lng()),
           south: Math.min(a.lat(), b.lat()), north: Math.max(a.lat(), b.lat()) };
}

function screenPx(ll) {
  const p = S.map.getProjection().fromLatLngToPoint(ll), z = 2 ** S.map.getZoom();
  return [p.x * z, p.y * z];
}
const pxDist = (a, b) => { const [ax, ay] = screenPx(a), [bx, by] = screenPx(b); return Math.hypot(ax - bx, ay - by); };

function onMapClick(e) {
  const d = S.draw;
  if (d?.kind === 'rect') {
    if (!d.anchor) {
      d.anchor = e.latLng;
      d.preview = new google.maps.Rectangle({ map: S.map, bounds: boundsOf(e.latLng, e.latLng), ...STYLE.draft });
    } else {
      const b = boundsOf(d.anchor, e.latLng);
      cancelDraw();
      setZoneRect(b);
    }
  } else if (d?.kind === 'poly') {
    if (d.path.length >= 3 && pxDist(e.latLng, d.path[0]) < 10) { finishPoly(); return; }
    d.path.push(e.latLng);
    drawPolyLine();
  } else if (S.mode === 'point') {
    setPoint(e.latLng.lat(), e.latLng.lng());
  }
}

function drawPolyLine(cursor) {
  const d = S.draw;
  const path = cursor ? [...d.path, cursor] : d.path;
  if (!d.line) d.line = new google.maps.Polyline({ map: S.map, path, strokeColor: '#ffffff', strokeWeight: 2, clickable: false, zIndex: 6 });
  else d.line.setPath(path);
}

function finishPoly() {
  const path = [];
  for (const ll of S.draw.path) {  // a double-click also delivered two clicks
    if (!path.length || pxDist(ll, path[path.length - 1]) > 3) path.push(ll);
  }
  cancelDraw();
  if (path.length < 3) { showError({ message: 'A polygon needs at least 3 corners.' }); return; }
  const p = new google.maps.Polygon({ map: S.map, paths: path, ...STYLE.zone });
  ['set_at', 'insert_at', 'remove_at'].forEach(ev => p.getPath().addListener(ev, schedulePlan));
  p.addListener('dragend', schedulePlan);
  hoverThrough(p);
  S.zone = { kind: 'poly', shape: p };
  schedulePlan();
}

function setZoneRect(b) {
  const r = new google.maps.Rectangle({ map: S.map, bounds: b, ...STYLE.zone });
  r.addListener('bounds_changed', schedulePlan);
  hoverThrough(r);
  S.zone = { kind: 'rect', shape: r };
  schedulePlan();
}

// an editable zone takes the mouse events over its area, where the results
// are: pass its moves on so the hover readout works there too
function hoverThrough(shape) {
  shape.addListener('mousemove', onMapMove);
  shape.addListener('mouseout', () => { $('#readout').hidden = true; });
}

function zoneAoi() {
  const z = S.zone;
  if (!z) return null;
  if (z.kind === 'rect') {
    const b = z.shape.getBounds(), sw = b.getSouthWest(), ne = b.getNorthEast();
    return [sw.lng(), sw.lat(), ne.lng(), ne.lat()];
  }
  if (z.kind === 'poly') {
    const ring = z.shape.getPath().getArray().map(ll => [ll.lng(), ll.lat()]);
    return { type: 'Polygon', coordinates: [[...ring, ring[0]]] };
  }
  return z.aoi;
}

function useGeojson(text) {
  hideError();
  let obj;
  try { obj = JSON.parse(text); } catch { showError({ message: 'That is not valid JSON.' }); return; }
  S.zone = { kind: 'geojson', aoi: obj };
  schedulePlan();
}

// ------------------------------------------------------------------ plan

function areaRequest() {
  const base = { fuse: $('#opt-fuse').checked, quality: $('#opt-quality').value, model: chosenModel() };
  if (S.mode === 'zone' || S.mode === 'geojson') {
    const a = S.zone && (S.mode === 'zone' ? S.zone.kind !== 'geojson' : S.zone.kind === 'geojson') ? zoneAoi() : null;
    return a ? { ...base, aoi: a } : null;
  }
  if (S.mode === 'point') {
    return S.point ? { ...base, lat: S.point.lat, lng: S.point.lng, grid: +$('#grid').value } : null;
  }
  return null;
}

function schedulePlan() {
  clearTimeout(S.planTimer);
  S.planTimer = setTimeout(doPlan, 300);
}

async function doPlan() {
  clearGrid();
  if (S.mode === 'upload') {
    const f = $('#upload-file').files[0];
    S.plan = null;
    $('#plan').innerHTML = f ? `<div class="big">${esc(f.name)}</div><div class="muted">${(f.size / 2 ** 20).toFixed(1)} MB · analysed on this server, not stored</div>`
      : '<span class="muted">Choose an image file.</span>';
    $('#run').disabled = !f;
    $('#run').textContent = 'Analyse image';
    return;
  }
  const req = areaRequest();
  const seq = ++S.planSeq;
  if (!req) {
    S.plan = null;
    $('#plan').innerHTML = `<span class="muted">${S.mode === 'point' ? 'Click the map or use “Go to” to set the centre.' : S.mode === 'geojson' ? 'Load or paste a GeoJSON polygon.' : 'Draw a zone on the map.'}</span>`;
    $('#run').disabled = true;
    $('#run').textContent = 'Analyse';
    return;
  }
  try {
    const p = await postJSON('/api/plan', { ...req, allow_outside: true });
    if (seq !== S.planSeq) return;
    S.plan = p;
    renderPlan();
    drawGrid(p);
    if (S.zone?.kind === 'geojson' && S.map) showAoiOutline(p.aoi, true);
  } catch (e) {
    if (seq !== S.planSeq) return;
    S.plan = null;
    $('#plan').innerHTML = `<span style="color:var(--err)">${esc(e.message)}</span>`;
    $('#run').disabled = true;
  }
}

function renderPlan() {
  const p = S.plan;
  if (!p) return;
  const zone = p.aoi_area_ha != null ? `<div class="big">Zone: ${fmtArea(haToUnits(p.aoi_area_ha))}</div>` : '';
  const warn = p.warnings?.length ? `<ul>${p.warnings.map(w => `<li>${esc(w)}</li>`).join('')}</ul>` : '';
  $('#plan').innerHTML = `${zone}
    <div>Analysed image: ${Math.round(p.extent_m[0])} × ${Math.round(p.extent_m[1])} m (${fmtArea(haToUnits(p.extent_ha))})</div>
    <div>${p.nx} × ${p.ny} tiles = <b>${p.calls} Google Static Maps call${p.calls > 1 ? 's' : ''}</b> · about ${p.estimate_s} s</div>${warn}`;
  const ok = S.cfg.coordinates_enabled;
  $('#run').disabled = !ok;
  $('#run').textContent = ok ? `Analyse (${p.calls} Google call${p.calls > 1 ? 's' : ''})` : 'Needs a Google key on the server';
}

function clearGrid() {
  S.gridLines.forEach(o => o.setMap(null));
  S.gridLines = [];
}

function drawGrid(p) {
  if (!S.map || !p.grid) return;
  const g = p.grid, x1 = g.x0 + g.width_map_px, y1 = g.y0 + g.height_map_px;
  const [w, s, e, n] = p.bbox_wsen;
  S.gridLines.push(new google.maps.Rectangle({ map: S.map, bounds: { west: w, south: s, east: e, north: n }, ...STYLE.extent }));
  for (let i = 1; i < p.nx; i++) {
    const x = g.x0 + i * 640;
    S.gridLines.push(new google.maps.Polyline({ map: S.map, path: [worldToLatLng(x, g.y0, g.zoom), worldToLatLng(x, y1, g.zoom)], ...STYLE.grid }));
  }
  for (let j = 1; j < p.ny; j++) {
    const y = g.y0 + j * 600;
    S.gridLines.push(new google.maps.Polyline({ map: S.map, path: [worldToLatLng(g.x0, y, g.zoom), worldToLatLng(x1, y, g.zoom)], ...STYLE.grid }));
  }
  if (S.zone?.kind === 'geojson' || S.mode === 'point') {
    S.map.fitBounds({ west: w, south: s, east: e, north: n }, 40);
  }
}

// ------------------------------------------------------------------ run

async function run() {
  hideError();
  try {
    if (S.mode === 'upload') return await runUpload();
    const req = areaRequest();
    if (!req) return;
    let allow = false;
    if (S.plan && !S.plan.in_coverage) {
      if (!confirm('This area is outside greater Hyderabad, where the model was trained. Results will be indicative only. Continue?')) return;
      allow = true;
    }
    const r = await postJSON('/api/jobs', { ...req, allow_outside: allow });
    startPolling(r.job);
  } catch (e) {
    showError(e);
  }
}

async function runUpload() {
  const f = $('#upload-file').files[0];
  if (!f) { showError({ message: 'Choose an image file first.' }); return; }
  const fd = new FormData();
  fd.append('file', f);
  const scale = $('input[name=scale]:checked').value;
  if (scale === 'screenshot') { fd.append('zoom', $('#up-zoom').value); fd.append('dpr', $('#up-dpr').value); }
  if (scale === 'mpp') fd.append('mpp', $('#up-mpp').value);
  fd.append('fuse', String($('#opt-fuse').checked));
  fd.append('quality', $('#opt-quality').value);
  if (chosenModel()) fd.append('model', chosenModel());
  const r = await api('/api/jobs/upload', { method: 'POST', body: fd });
  startPolling(r.job);
}

function setBusy(on) {
  $('#step-progress').hidden = !on;
  $('#run').disabled = on || !S.plan && S.mode !== 'upload';
  $$('.panel input, .panel select, .panel textarea, .tabs button, .tool').forEach(el => {
    if (!el.closest('#step-result')) el.disabled = on;
  });
}

function startPolling(job) {
  clearResult();
  S.job = job;
  setBusy(true);
  updateProgress(job);
  addHistory(job);
  clearInterval(S.pollTimer);
  S.pollTimer = setInterval(poll, 1000);
}

async function poll() {
  try {
    const j = await api('/api/jobs/' + S.job.id);
    S.job = j;
    updateProgress(j);
    if (!['done', 'failed', 'cancelled'].includes(j.state)) return;
    clearInterval(S.pollTimer);
    setBusy(false);
    updateHistory(j);
    if (j.state === 'done') await showResult(j);
    else if (j.state === 'failed') showError(j.error);
    else showError({ message: 'The analysis was cancelled.' });
    renderPlan();
  } catch (e) {
    clearInterval(S.pollTimer);
    setBusy(false);
    showError(e);
  }
}

function updateProgress(j) {
  $('#bar').style.width = `${Math.round((j.progress || 0) * 100)}%`;
  const secs = j.started ? Math.round(Date.now() / 1000 - j.started) : 0;
  const what = j.state === 'queued' ? 'Waiting for the analysis ahead of it…' : (j.label || 'Working') + '…';
  $('#progress-label').textContent = `${what}${secs ? ` ${secs} s` : ''}`;
}

async function cancelJob() {
  if (!S.job) return;
  try { await api('/api/jobs/' + S.job.id, { method: 'DELETE' }); } catch (e) { showError(e); }
}

function showError(e) {
  const box = $('#error');
  box.innerHTML = esc(e?.message || String(e));
  box.hidden = false;
}
const hideError = () => { $('#error').hidden = true; };

// ------------------------------------------------------------------ results

function fileUrl(name, download = false) {
  return `/api/jobs/${S.jobId}/files/${name}${download ? '?download=1' : ''}`;
}

function clearResult() {
  S.overlays.forEach(o => o.ov.setMap(null));
  S.overlays = [];
  Object.values(S.urls).forEach(u => URL.revokeObjectURL(u));
  S.urls = {};
  S.canvases = {};
  S.result = null;
  S.jobId = null;
  $('#step-result').hidden = true;
  $('#readout').hidden = true;
  closeViewer();
  if (S.map) S.map.data.forEach(f => S.map.data.remove(f));
  hideError();
}

async function showResult(j) {
  const r = j.result;
  S.jobId = j.id;
  S.result = r;
  S.hidden.clear();
  S.version = 'fused';
  S.scope = r.stats.zone ? 'zone' : 'extent';
  $('#result-title').textContent = r.title || j.title || 'Result';
  const when = new Date((j.finished || Date.now() / 1000) * 1000).toLocaleString();
  const f = r.fusion;
  const src = f.active ? 'with ' + f.sources.map(s => ({ osm: 'OpenStreetMap', footprints: 'building footprints' }[s] || s)).join(' + ')
    + (f.osm?.osm_base ? ` (OSM data of ${f.osm.osm_base.slice(0, 10)})` : '') : 'model only';
  const mdl = r.model?.id ? ` · model ${r.model.id}` : '';
  $('#result-meta').textContent = `${when} · ${r.quality} quality${mdl} · ${r.elapsed_s} s · ${src}`;
  $('#result-notes').innerHTML = [...r.notes, ...r.warnings].map(n => `<li>${esc(n)}</li>`).join('');
  $('#lay-outside').closest('label').hidden = !r.stats.zone;
  $('#scope').hidden = !r.stats.zone;
  $('#version').hidden = !f.active;  // nothing to compare without map data
  $$('#version [data-ver]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ver === S.version)));
  renderToggles();
  renderStats();
  renderDownloads(r.files);
  $('#gold-entry').hidden = !S.cfg.gold?.enabled || !r.files.includes('imagery.jpg');
  $('#step-result').hidden = false;
  await loadCanvas(r.stats.zone ? 'classes_full.png' : 'classes.png');
  if (r.georeferenced && S.mapsOk) {
    const [w, s, e, n] = r.bbox_wsen;
    S.map.fitBounds({ west: w, south: s, east: e, north: n }, 30);
    if (r.aoi) showAoiOutline(r.aoi, false);
    await refreshLayers();
  } else {
    await openViewer();
  }
  $('#step-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showAoiOutline(geom, fit) {
  S.map.data.forEach(f => S.map.data.remove(f));
  const fs = S.map.data.addGeoJson({ type: 'Feature', geometry: geom, properties: {} });
  if (fit && fs.length) {
    const b = new google.maps.LatLngBounds();
    fs[0].getGeometry().forEachLatLng(ll => b.extend(ll));
    S.map.fitBounds(b, 40);
  }
}

function renderToggles() {
  $('#class-toggles').innerHTML = LEGEND_ORDER.map(n => className(n)).map(c =>
    `<label><input type="checkbox" data-cls="${c.id}" checked><i class="sw" style="background:${rgb(c.color)}"></i>${esc(c.label)}</label>`).join('');
  $$('#class-toggles input').forEach(i => i.onchange = () => {
    const id = +i.dataset.cls;
    if (i.checked) S.hidden.delete(id); else S.hidden.add(id);
    refreshLayers();
  });
}

function renderStats() {
  const r = S.result;
  if (!r) return;
  $$('#scope [data-scope]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.scope === S.scope)));
  const src = S.version === 'raw' && r.stats_model_only ? r.stats_model_only : r.stats;
  const st = (S.scope === 'zone' && src.zone) || src.extent;
  const rows = LEGEND_ORDER.map(n => {
    const v = st.classes[n], c = className(n);
    const badge = n === 'commercial' ? '<span class="badge" title="Commercial is the least reliable class (a shop-house looks like a house from above)">less reliable</span>' : '';
    return `<tr><td><i class="sw" style="background:${rgb(c.color)}"></i>${esc(c.label)}${badge}</td>
      <td class="num">${(v.share * 100).toFixed(1)}%</td><td class="num">${fmtArea(v.units)}</td>
      <td class="barcell"><div class="bar" style="width:${(v.share * 100).toFixed(1)}%;background:${rgb(c.color)}"></div></td></tr>`;
  }).join('');
  const which = !r.fusion.active ? 'Model output' : src === r.stats ? 'With map data' : 'Model only';
  $('#stats').innerHTML = `<caption class="muted small" style="text-align:left">${which} · ${S.scope === 'zone' && src.zone ? 'your zone' : 'whole image'}</caption>
    <tr><th>Class</th><th class="num">Share</th><th class="num">Area</th><th></th></tr>${rows}
    <tr class="total"><td>Total analysed</td><td></td><td class="num">${fmtArea(st.total.units)}</td><td></td></tr>`;
}

function renderDownloads(files) {
  const items = [['result.geojson', 'GeoJSON (zone)'], ['classes.tif', 'GeoTIFF'], ['stats.csv', 'CSV'],
                 ['overlay.png', 'Image with legend'], ['mask.png', 'Class mask PNG'], ['result.json', 'JSON']];
  $('#downloads').innerHTML = items.filter(([n]) => files.includes(n))
    .map(([n, label]) => `<a href="${fileUrl(n, true)}" download>${esc(label)}</a>`).join('');
}

async function loadCanvas(name) {
  if (S.canvases[name]) return S.canvases[name];
  const img = new Image();
  img.src = fileUrl(name);
  await img.decode();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  const c = { img, canvas, data: ctx.getImageData(0, 0, canvas.width, canvas.height) };
  S.canvases[name] = c;
  return c;
}

function classLayerName() {
  const outside = !$('#lay-outside').closest('label').hidden && $('#lay-outside').checked;
  if (S.version === 'raw') return 'raw.png';
  return outside && S.result.stats.zone ? 'classes_full.png' : 'classes.png';
}

async function classLayerUrl() {
  const name = classLayerName();
  const key = name + '|' + [...S.hidden].sort().join(',');
  if (S.urls[key]) return S.urls[key];
  if (!S.hidden.size) return fileUrl(name);
  const src = await loadCanvas(name);
  const hide = new Set([...S.hidden].map(id => colorKey(...classById(id).color)));
  const out = new ImageData(new Uint8ClampedArray(src.data.data), src.data.width, src.data.height);
  const d = out.data;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] && hide.has(colorKey(d[i], d[i + 1], d[i + 2]))) d[i + 3] = 0;
  }
  const c = document.createElement('canvas');
  c.width = out.width;
  c.height = out.height;
  c.getContext('2d').putImageData(out, 0, 0);
  const blob = await new Promise(res => c.toBlob(res, 'image/png'));
  return (S.urls[key] = URL.createObjectURL(blob));
}

async function refreshLayers() {
  $$('#version [data-ver]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.ver === S.version)));
  if (!S.result) return;
  renderStats();
  if (S.viewer) { await drawViewer(); return; }
  if (!S.mapsOk || !S.result.georeferenced) return;
  const [w, s, e, n] = S.result.bbox_wsen;
  const bounds = { west: w, south: s, east: e, north: n };
  const want = [];
  if ($('#lay-imagery').checked) want.push(['imagery', fileUrl('imagery.jpg'), 1]);
  want.push(['classes', await classLayerUrl(), +$('#opacity').value]);
  if ($('#lay-lowconf').checked) want.push(['lowconf', fileUrl('lowconf.png'), 1]);
  S.overlays.forEach(o => o.ov.setMap(null));
  S.overlays = want.map(([kind, url, opacity]) => ({
    kind, ov: new google.maps.GroundOverlay(url, bounds, { map: S.map, opacity, clickable: false }) }));
}

function classAt(c, x, y) {
  if (!c || x < 0 || y < 0 || x >= c.data.width || y >= c.data.height) return null;
  const i = (y * c.data.width + x) * 4, d = c.data.data;
  if (!d[i + 3]) return null;
  const k = colorKey(d[i], d[i + 1], d[i + 2]);
  return S.cfg.classes.find(cl => colorKey(...cl.color) === k) || null;
}

async function readout(x, y, where) {
  const fused = classAt(S.canvases[S.result.stats.zone ? 'classes_full.png' : 'classes.png'], x, y);
  if (!fused) { $('#readout').hidden = true; return; }
  let raw = null;
  if (S.result.fusion.active) raw = classAt(S.canvases['raw.png'] || await loadCanvas('raw.png'), x, y);
  const extra = raw && raw.id !== fused.id ? ` · model alone: ${esc(raw.label)}` : '';
  $('#readout').innerHTML = `<b>${esc(fused.label)}</b>${extra}${where ? '<br>' + where : ''}`;
  $('#readout').hidden = false;
}

function onMapMove(e) {
  const d = S.draw;
  if (d?.kind === 'rect' && d.anchor) d.preview.setBounds(boundsOf(d.anchor, e.latLng));
  else if (d?.kind === 'poly' && d.path.length) drawPolyLine(e.latLng);
  const r = S.result;
  if (!r?.georeferenced || d) return;
  const g = r.grid, [wx, wy] = latLngToWorld(e.latLng.lat(), e.latLng.lng(), g.zoom);
  readout(Math.floor((wx - g.x0) * g.scale), Math.floor((wy - g.y0) * g.scale),
          `${e.latLng.lat().toFixed(6)}, ${e.latLng.lng().toFixed(6)}`);
}

// ------------------------------------------------------------------ viewer
// for uploads without coordinates (or when the map is unavailable)

async function openViewer() {
  const [imagery] = await Promise.all([loadCanvas('imagery.jpg')]);
  const cv = $('#viewer-canvas');
  $('#viewer').hidden = false;
  const box = cv.getBoundingClientRect();
  const fit = Math.min(box.width / imagery.canvas.width, box.height / imagery.canvas.height);
  S.viewer = { zoom: fit, x: (box.width - imagery.canvas.width * fit) / 2, y: (box.height - imagery.canvas.height * fit) / 2 };
  cv.onwheel = ev => {
    ev.preventDefault();
    const v = S.viewer, f = Math.exp(-ev.deltaY * 0.0015), rx = ev.offsetX, ry = ev.offsetY;
    v.x = rx - (rx - v.x) * f; v.y = ry - (ry - v.y) * f; v.zoom *= f;
    drawViewer();
  };
  let drag = null;
  cv.onpointerdown = ev => { drag = [ev.clientX, ev.clientY]; cv.setPointerCapture(ev.pointerId); };
  cv.onpointerup = () => { drag = null; };
  cv.onpointermove = ev => {
    const v = S.viewer;
    if (drag) { v.x += ev.clientX - drag[0]; v.y += ev.clientY - drag[1]; drag = [ev.clientX, ev.clientY]; drawViewer(); return; }
    readout(Math.floor((ev.offsetX - v.x) / v.zoom), Math.floor((ev.offsetY - v.y) / v.zoom), '');
  };
  cv.onpointerleave = () => { $('#readout').hidden = true; };
  window.addEventListener('resize', drawViewer);
  await drawViewer();
}

async function drawViewer() {
  if (!S.viewer || !S.result) return;
  const cv = $('#viewer-canvas'), box = cv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  cv.width = Math.round(box.width * dpr);
  cv.height = Math.round(box.height * dpr);
  const ctx = cv.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
  const v = S.viewer, base = await loadCanvas('imagery.jpg');
  const w = base.canvas.width * v.zoom, h = base.canvas.height * v.zoom;
  ctx.drawImage(base.canvas, v.x, v.y, w, h);
  const url = await classLayerUrl();
  const layer = new Image();
  layer.src = url;
  await layer.decode();
  ctx.globalAlpha = +$('#opacity').value;
  ctx.drawImage(layer, v.x, v.y, w, h);
  ctx.globalAlpha = 1;
  if ($('#lay-lowconf').checked) ctx.drawImage((await loadCanvas('lowconf.png')).canvas, v.x, v.y, w, h);
}

function closeViewer() {
  S.viewer = null;
  $('#viewer').hidden = true;
}

// ------------------------------------------------------------------ history

function history() {
  try { return JSON.parse(sessionStorage.getItem('urbanseg.history') || '[]'); } catch { return []; }
}
function saveHistory(h) {
  try { sessionStorage.setItem('urbanseg.history', JSON.stringify(h.slice(0, 12))); } catch { /* ignore */ }
}
function addHistory(job) {
  saveHistory([{ id: job.id, title: job.title, t: Date.now(), state: job.state }, ...history().filter(h => h.id !== job.id)]);
  renderHistory();
}
function updateHistory(job) {
  saveHistory(history().map(h => h.id === job.id ? { ...h, state: job.state } : h));
  renderHistory();
}
function renderHistory() {
  const h = history();
  $('#step-history').hidden = !h.length;
  $('#history').innerHTML = h.map(x => `<li><button type="button" data-id="${esc(x.id)}">${esc(x.title || 'Analysis')}<br><span class="muted">${new Date(x.t).toLocaleTimeString()} · ${esc(x.state)}</span></button></li>`).join('');
  $$('#history button').forEach(b => b.onclick = async () => {
    try {
      const j = await api('/api/jobs/' + b.dataset.id);
      if (j.state === 'done') { clearResult(); await showResult(j); }
      else if (j.state === 'running' || j.state === 'queued') startPolling(j);
      else showError(j.error || { message: `That analysis ${j.state}.` });
    } catch (e) {
      showError({ message: e.status === 404 ? 'That result has expired (results are kept for an hour).' : e.message });
    }
  });
}
