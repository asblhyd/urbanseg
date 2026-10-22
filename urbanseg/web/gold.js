// Gold-set editor: correct a result's classes by hand. Saved samples are the
// gold set that urbanseg.gold scores models on (python -m urbanseg.gold evaluate).
// Labels travel as grey PNGs, class id * 60 (255 = not reviewed), so that a
// canvas's colour handling cannot turn one class into another.

const IGNORE = 255;
const SPREAD = 60;
const LEVELS = [0, SPREAD, 2 * SPREAD, 3 * SPREAD, 255];
const MAX_FILL = 4e6;  // pixels one click may fill
// paint keys: 1-4 follow the legend; 0 / U = unsure (not reviewed)
const KEY_CLASS = { 1: 'residential', 2: 'commercial', 3: 'free_land', 4: 'other' };

const $ = (sel, el = document) => el.querySelector(sel);
let ctx = null;  // { api, esc, showError, cfg }
const G = {
  W: 0, H: 0, labels: null, over: null, ovCtx: null, zoom: 1,
  cls: 1, tool: 'brush', size: 16, painting: false, last: null, dirty: null,
  undo: [], changed: false, jobId: null, sampleId: null, seed: 'model',
};
window.__gold = G;  // read-only handle for debugging and browser tests

export function initGold(c) {
  ctx = c;
  if (!c.cfg.gold?.enabled) return;
  $('#step-gold').hidden = false;
  buildBar();
  const view = $('#gold-view');
  const lab = $('#gold-lab');
  lab.addEventListener('pointerdown', onDown);
  lab.addEventListener('pointermove', onMove);
  window.addEventListener('pointerup', onUp);
  view.addEventListener('wheel', onWheel, { passive: false });
  document.addEventListener('keydown', onKey);
  $('#gold-close').onclick = close;
  $('#gold-save').onclick = () => save().catch(ctx.showError);
  $('#gold-undo').onclick = undo;
  $('#gold-gaps').onclick = fillGaps;
  $('#gold-fit').onclick = fit;
  $('#gold-size').oninput = e => { G.size = +e.target.value; showInfo(); };
  $('#gold-alpha').oninput = e => { $('#gold-lab').style.opacity = e.target.value; };
  renderGoldList();
}

function buildBar() {
  const order = ['residential', 'commercial', 'free_land', 'other'];
  const byName = n => ctx.cfg.classes.find(c => c.name === n);
  $('#gold-classes').innerHTML = order.map((n, i) => {
    const c = byName(n);
    return `<button type="button" class="tool" data-cls="${c.id}" title="${ctx.esc(c.label)} (${i + 1})">`
      + `<i class="sw" style="background:rgb(${c.color.join(',')})"></i>${ctx.esc(c.label)}</button>`;
  }).join('') + `<button type="button" class="tool" data-cls="${IGNORE}" title="Not sure: left out of the gold set (0)">`
    + '<i class="sw unsure"></i>Unsure</button>';
  $('#gold-classes').onclick = e => { const b = e.target.closest('button'); if (b) setClass(+b.dataset.cls); };
  $('#gold-tools').onclick = e => { const b = e.target.closest('button'); if (b) setTool(b.dataset.tool); };
  setClass(1);
  setTool('brush');
}

function setClass(id) {
  G.cls = id;
  for (const b of document.querySelectorAll('#gold-classes button')) b.setAttribute('aria-pressed', String(+b.dataset.cls === id));
}

function setTool(t) {
  G.tool = t;
  for (const b of document.querySelectorAll('#gold-tools button')) b.setAttribute('aria-pressed', String(b.dataset.tool === t));
  $('#gold-lab').style.cursor = t === 'pan' ? 'grab' : 'crosshair';
}

// ------------------------------------------------------------------ open

export async function openFromJob(jobId, title, seed = 'model') {
  G.jobId = jobId; G.sampleId = null; G.seed = seed;
  await open(`/api/jobs/${jobId}/files/imagery.jpg`, `/api/jobs/${jobId}/gold-seed.png?source=${seed}`,
    `Correcting “${title}” (starting from the ${seed === 'model' ? 'model alone' : 'result with map data'})`, '', true);
}

export async function openSample(id) {
  const s = await ctx.api(`/api/gold/${id}`);
  G.jobId = null; G.sampleId = id;
  await open(`/api/gold/${id}/image`, `/api/gold/${id}/labels.png`,
    `Gold sample ${s.title || id} · ${s.status} · seed ${s.seed || '?'}`, s.note || '', s.status !== 'draft');
}

async function open(imgUrl, labUrl, info, note, reviewed) {
  if (G.changed && !confirm('Discard the unsaved corrections?')) return;
  setStatus('Loading…');
  $('#gold-editor').hidden = false;
  const [img, lab] = await Promise.all([loadImage(imgUrl), loadImage(labUrl)]);
  if (img.naturalWidth !== lab.naturalWidth || img.naturalHeight !== lab.naturalHeight) {
    setStatus('The labels do not match the image size.');
    return;
  }
  const W = G.W = img.naturalWidth, H = G.H = img.naturalHeight;
  const ic = $('#gold-img'), lc = $('#gold-lab');
  ic.width = lc.width = W; ic.height = lc.height = H;
  ic.getContext('2d').drawImage(img, 0, 0);
  G.labels = decode(lab, W, H);
  G.ovCtx = lc.getContext('2d');
  G.over = G.ovCtx.createImageData(W, H);
  repaint(0, 0, W, H);
  G.undo = []; G.changed = false;
  $('#gold-note').value = note;
  $('#gold-reviewed').checked = reviewed;
  $('#gold-title').textContent = info;
  fit();
  setStatus('Paint the right class over mistakes (keys 1-4, 0 = unsure), then Save.');
}

function loadImage(url) {
  return new Promise((ok, fail) => {
    const im = new Image();
    im.onload = () => ok(im);
    im.onerror = () => fail(new Error('could not load ' + url));
    im.src = url;
  });
}

function decode(im, W, H) {
  const c = document.createElement('canvas');
  c.width = W; c.height = H;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(im, 0, 0);
  const d = cx.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  const ids = [0, 1, 2, 3, IGNORE];
  for (let i = 0; i < out.length; i++) {
    const v = d[i * 4];
    let best = 0;
    for (let k = 1; k < 5; k++) if (Math.abs(LEVELS[k] - v) < Math.abs(LEVELS[best] - v)) best = k;
    out[i] = ids[best];
  }
  return out;
}

// --------------------------------------------------------------- drawing

function repaint(x0, y0, x1, y1) {
  const { W, labels, over } = G;
  const col = ctx.cfg.classes.map(c => c.color);
  const d = over.data;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const i = y * W + x, v = labels[i], o = i * 4;
      if (v === IGNORE) {  // magenta checks: visible, clearly not a class
        const on = ((x >> 3) + (y >> 3)) & 1;
        d[o] = 255; d[o + 1] = 0; d[o + 2] = 255; d[o + 3] = on ? 150 : 40;
      } else {
        const c = col[v];
        d[o] = c[0]; d[o + 1] = c[1]; d[o + 2] = c[2]; d[o + 3] = 255;
      }
    }
  }
  G.ovCtx.putImageData(over, 0, 0, x0, y0, x1 - x0, y1 - y0);
}

function toImage(e) {
  const r = $('#gold-lab').getBoundingClientRect();
  return [(e.clientX - r.left) * G.W / r.width, (e.clientY - r.top) * G.H / r.height];
}

function snapshot() {
  G.undo.push(G.labels.slice());
  const keep = G.W * G.H > 8e6 ? 4 : 12;
  while (G.undo.length > keep) G.undo.shift();
}

function stamp(cx, cy) {
  const r = G.size / 2, { W, H, labels } = G;
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(W, Math.ceil(cx + r) + 1);
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(H, Math.ceil(cy + r) + 1);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const dx = x + 0.5 - cx, dy = y + 0.5 - cy;
      if (dx * dx + dy * dy <= r * r) labels[y * W + x] = G.cls;
    }
  }
  const d = G.dirty;
  G.dirty = d ? [Math.min(d[0], x0), Math.min(d[1], y0), Math.max(d[2], x1), Math.max(d[3], y1)] : [x0, y0, x1, y1];
}

function flush() {
  if (G.dirty) { repaint(...G.dirty); G.dirty = null; }
  G.raf = 0;
}

function fill(x, y) {
  const { W, H, labels } = G;
  x = Math.floor(x); y = Math.floor(y);
  if (x < 0 || y < 0 || x >= W || y >= H) return;
  const from = labels[y * W + x];
  if (from === G.cls) return;
  snapshot();
  const stack = [y * W + x];
  let n = 0, bx0 = x, by0 = y, bx1 = x, by1 = y;
  labels[y * W + x] = G.cls;
  while (stack.length) {
    const i = stack.pop(), px = i % W, py = (i - px) / W;
    if (++n > MAX_FILL) { setStatus('Fill stopped: that region is very large. Use the brush.'); break; }
    if (px < bx0) bx0 = px; if (px > bx1) bx1 = px; if (py < by0) by0 = py; if (py > by1) by1 = py;
    if (px > 0 && labels[i - 1] === from) { labels[i - 1] = G.cls; stack.push(i - 1); }
    if (px < W - 1 && labels[i + 1] === from) { labels[i + 1] = G.cls; stack.push(i + 1); }
    if (py > 0 && labels[i - W] === from) { labels[i - W] = G.cls; stack.push(i - W); }
    if (py < H - 1 && labels[i + W] === from) { labels[i + W] = G.cls; stack.push(i + W); }
  }
  repaint(bx0, by0, bx1 + 1, by1 + 1);
  G.changed = true;
}

function onDown(e) {
  if (!G.labels) return;
  if (G.tool === 'pan' || e.button === 1 || e.button === 2 || G.space) {
    G.pan = [e.clientX, e.clientY, $('#gold-view').scrollLeft, $('#gold-view').scrollTop];
    e.preventDefault();
    return;
  }
  if (e.button !== 0) return;
  const [x, y] = toImage(e);
  if (G.tool === 'fill') { fill(x, y); return; }
  snapshot();
  G.painting = true; G.last = [x, y]; G.changed = true;
  $('#gold-lab').setPointerCapture(e.pointerId);
  stamp(x, y);
  flush();
}

function onMove(e) {
  if (G.pan) {
    const v = $('#gold-view');
    v.scrollLeft = G.pan[2] - (e.clientX - G.pan[0]);
    v.scrollTop = G.pan[3] - (e.clientY - G.pan[1]);
    return;
  }
  if (!G.painting) return;
  const [x, y] = toImage(e), [lx, ly] = G.last;
  const step = Math.max(1, G.size / 4), n = Math.ceil(Math.hypot(x - lx, y - ly) / step);
  for (let k = 1; k <= n; k++) stamp(lx + (x - lx) * k / n, ly + (y - ly) * k / n);
  G.last = [x, y];
  if (!G.raf) G.raf = requestAnimationFrame(flush);
}

function onUp() {
  G.painting = false; G.pan = null;
  if (G.dirty) flush();
}

// every Unsure pixel takes the class of the nearest labelled one (4-neighbour
// breadth-first from all labelled pixels at once): auto-label seeds leave
// thin uncertain bands along every edge, which a reviewer can then check
function fillGaps() {
  const { W, H, labels } = G;
  let queue = new Int32Array(W * H), head = 0, tail = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] !== IGNORE) queue[tail++] = i;
  if (tail === 0 || tail === labels.length) { setStatus(tail ? 'Nothing is unsure.' : 'Nothing labelled to grow from.'); return; }
  snapshot();
  const out = labels;
  while (head < tail) {
    const i = queue[head++], x = i % W, v = out[i];
    if (x > 0 && out[i - 1] === IGNORE) { out[i - 1] = v; queue[tail++] = i - 1; }
    if (x < W - 1 && out[i + 1] === IGNORE) { out[i + 1] = v; queue[tail++] = i + 1; }
    if (i >= W && out[i - W] === IGNORE) { out[i - W] = v; queue[tail++] = i - W; }
    if (i < W * (H - 1) && out[i + W] === IGNORE) { out[i + W] = v; queue[tail++] = i + W; }
  }
  queue = null;
  repaint(0, 0, W, H);
  G.changed = true;
  setStatus('Unsure pixels now take their nearest class: check the edges, then Save.');
}

function undo() {
  const prev = G.undo.pop();
  if (!prev) return;
  G.labels = prev;
  repaint(0, 0, G.W, G.H);
  G.changed = true;
}

// ----------------------------------------------------------- zoom and pan

function setZoom(z, cx, cy) {
  const v = $('#gold-view');
  const old = G.zoom;
  G.zoom = Math.min(16, Math.max(0.05, z));
  for (const c of [$('#gold-img'), $('#gold-lab')]) {
    c.style.width = G.W * G.zoom + 'px';
    c.style.height = G.H * G.zoom + 'px';
  }
  $('#gold-stack').style.width = G.W * G.zoom + 'px';
  $('#gold-stack').style.height = G.H * G.zoom + 'px';
  if (cx !== undefined) {  // keep the point under the cursor in place
    const f = G.zoom / old;
    v.scrollLeft = (v.scrollLeft + cx) * f - cx;
    v.scrollTop = (v.scrollTop + cy) * f - cy;
  }
  showInfo();
}

function fit() {
  const v = $('#gold-view');
  setZoom(Math.min(v.clientWidth / G.W, v.clientHeight / G.H));
}

function onWheel(e) {
  if (!G.labels) return;
  e.preventDefault();
  const r = $('#gold-view').getBoundingClientRect();
  setZoom(G.zoom * Math.exp(-e.deltaY * 0.0015), e.clientX - r.left, e.clientY - r.top);
}

function onKey(e) {
  if ($('#gold-editor').hidden || e.target.matches('input, textarea, select')) return;
  if (e.key === ' ') { G.space = e.type === 'keydown'; e.preventDefault(); return; }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { undo(); e.preventDefault(); return; }
  const k = e.key.toLowerCase();
  if (KEY_CLASS[k]) setClass(ctx.cfg.classes.find(c => c.name === KEY_CLASS[k]).id);
  else if (k === '0' || k === 'u') setClass(IGNORE);
  else if (k === 'b') setTool('brush');
  else if (k === 'f') setTool('fill');
  else if (k === 'h') setTool('pan');
  else if (k === '[') { G.size = Math.max(2, G.size - 4); $('#gold-size').value = G.size; showInfo(); }
  else if (k === ']') { G.size = Math.min(120, G.size + 4); $('#gold-size').value = G.size; showInfo(); }
  else if (k === 'escape') close();
}
document.addEventListener('keyup', e => { if (e.key === ' ') G.space = false; });

function showInfo() {
  $('#gold-info').textContent = `${G.W}×${G.H} px · zoom ${Math.round(G.zoom * 100)}% · brush ${G.size} px`;
}

function setStatus(t) { $('#gold-status').textContent = t; }

function close() {
  if (G.changed && !confirm('Close without saving the corrections?')) return;
  $('#gold-editor').hidden = true;
  G.labels = G.over = null; G.undo = []; G.changed = false;
}

// ------------------------------------------------------------------ save

function encode() {
  const c = document.createElement('canvas');
  c.width = G.W; c.height = G.H;
  const cx = c.getContext('2d');
  const im = cx.createImageData(G.W, G.H), d = im.data, L = G.labels;
  for (let i = 0; i < L.length; i++) {
    const v = L[i] === IGNORE ? 255 : L[i] * SPREAD, o = i * 4;
    d[o] = d[o + 1] = d[o + 2] = v; d[o + 3] = 255;
  }
  cx.putImageData(im, 0, 0);
  return new Promise(ok => c.toBlob(ok, 'image/png'));
}

async function save() {
  if (!G.labels) return;
  setStatus('Saving…');
  const fd = new FormData();
  fd.append('labels', await encode(), 'labels.png');
  fd.append('status', $('#gold-reviewed').checked ? 'reviewed' : 'draft');
  fd.append('note', $('#gold-note').value);
  let s;
  if (G.sampleId) {
    s = await ctx.api(`/api/gold/${G.sampleId}`, { method: 'PUT', body: fd });
  } else {
    fd.append('job_id', G.jobId);
    fd.append('seed', G.seed);
    s = await ctx.api('/api/gold', { method: 'POST', body: fd });
    G.sampleId = s.id;
  }
  G.changed = false;
  setStatus(`Saved gold sample ${s.id} (${s.status}, ${Math.round(s.labelled_share * 100)}% labelled).`);
  renderGoldList();
}

// ------------------------------------------------------------------ list

export async function renderGoldList() {
  if (!ctx?.cfg.gold?.enabled) return;
  let r;
  try { r = await ctx.api('/api/gold'); } catch (e) { $('#gold-list').innerHTML = `<li class="muted small">${ctx.esc(e.message)}</li>`; return; }
  const n = r.samples.filter(s => s.status === 'reviewed').length;
  $('#gold-count').textContent = `${n} reviewed · ${r.samples.length - n} draft`;
  $('#gold-list').innerHTML = r.samples.slice(0, 30).map(s =>
    `<li><button type="button" data-id="${ctx.esc(s.id)}">${ctx.esc(s.title || s.id)}<br>`
    + `<span class="muted">${ctx.esc(s.status)} · ${Math.round(s.labelled_share * 100)}% labelled · `
    + `${ctx.esc((s.updated || '').slice(0, 16).replace('T', ' '))}</span></button></li>`).join('')
    || '<li class="muted small">No samples yet: open a result and choose “Correct”.</li>';
  $('#gold-list').onclick = e => { const b = e.target.closest('button'); if (b) openSample(b.dataset.id).catch(ctx.showError); };
}
