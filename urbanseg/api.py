"""FastAPI service: lat/long or uploaded image -> segmentation.

Run: uvicorn urbanseg.api:app --host 127.0.0.1 --port 8000
"""
import base64
import io
import json
import time
import traceback
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image

from . import config, infer, open_buildings

_model = None

MAX_UPLOAD_SIDE = 4096  # px; larger uploads are downscaled to keep CPU time sane
NO_KEY = ("coordinate mode needs a Google Maps API key: set GOOGLE_MAPS_KEY in "
          ".env and re-run ./run.sh (image upload works without one)")


def model():
    global _model
    if _model is None:
        _model = infer.load_model()
    return _model


@asynccontextmanager
async def lifespan(_app):
    model()  # load weights before /health reports ready
    yield


app = FastAPI(title="UrbanSeg", lifespan=lifespan)


def _png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _in_coverage(lat, lng):
    w, s, e, n = config.HYD_BBOX
    return s <= lat <= n and w <= lng <= e


def _result(img, classmap, meta, want_geojson, t0, note=None):
    stats = infer.area_stats(classmap, meta)
    fusion = meta.get("fusion", [])
    out = {
        "classes": config.CLASS_NAMES,
        "class_colors": {config.CLASS_NAMES[k]: v
                         for k, v in config.CLASS_COLORS.items()},
        "stats": stats,
        "fusion_active": bool(fusion),
        "fusion_sources": fusion,
        "elapsed_s": round(time.time() - t0, 1),
        "overlay_png_b64": base64.b64encode(
            _png_bytes(infer.render_overlay(img, classmap))).decode(),
    }
    if note:
        out["note"] = note
    if want_geojson:
        out["geojson"] = infer.to_geojson(classmap, meta)
    return out


@app.get("/health")
def health():
    return {"ok": True, "classes": config.CLASS_NAMES,
            "coordinates_enabled": bool(config.GOOGLE_MAPS_KEY),
            "footprint_index": open_buildings.PARQUET.exists()}


@app.get("/segment")
def segment_coords(lat: float = Query(..., ge=-85, le=85),
                   lng: float = Query(..., ge=-180, le=180),
                   grid: int = Query(2, ge=1, le=3), fuse: bool = True,
                   geojson: bool = False):
    t0 = time.time()
    if not config.GOOGLE_MAPS_KEY:
        raise HTTPException(503, NO_KEY)
    try:
        img, classmap, meta = infer.segment_latlng(
            model(), lat, lng, grid=grid, fuse=fuse)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(502, f"segmentation failed: {ex}") from ex
    note = None
    if not _in_coverage(lat, lng):
        note = ("location is outside greater Hyderabad, where the model was "
                "trained; expect lower accuracy")
    elif fuse and not meta.get("fusion"):
        note = ("OpenStreetMap was unreachable and no footprint index is "
                "built; model-only prediction")
    out = _result(img, classmap, meta, geojson, t0, note)
    out["center"] = [lat, lng]
    out["bbox_wsen"] = infer.imagery.bbox_latlng(meta)
    return out


@app.get("/segment.png")
def segment_png(lat: float = Query(..., ge=-85, le=85),
                lng: float = Query(..., ge=-180, le=180),
                grid: int = Query(2, ge=1, le=3), fuse: bool = True,
                legend: bool = True):
    if not config.GOOGLE_MAPS_KEY:
        raise HTTPException(503, NO_KEY)
    try:
        img, classmap, meta = infer.segment_latlng(
            model(), lat, lng, grid=grid, fuse=fuse)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(502, f"segmentation failed: {ex}") from ex
    overlay = infer.render_overlay(img, classmap)
    if legend:
        overlay = infer.add_legend(overlay, infer.area_stats(classmap, meta))
    return Response(_png_bytes(overlay), media_type="image/png")


@app.post("/segment-image")
async def segment_upload(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    try:
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as ex:  # noqa: BLE001
        raise HTTPException(400, f"not a readable image: {ex}") from ex
    if max(pil.size) > MAX_UPLOAD_SIDE:
        f = MAX_UPLOAD_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * f), int(pil.height * f)))
    img = np.asarray(pil)
    try:
        classmap, _probs, meta = infer.segment_image(model(), img)
    except Exception as ex:  # noqa: BLE001
        traceback.print_exc()
        raise HTTPException(502, f"segmentation failed: {ex}") from ex
    note = ("uploaded image has no geolocation: OSM/footprint fusion is off and "
            "area m² assumes ~0.28 m/px imagery (Google zoom 18). For best "
            "results upload satellite imagery at a similar scale.")
    return _result(img, classmap, meta, False, t0, note)


PAGE = """<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UrbanSeg</title>
<style>
  :root { --bg:#14161a; --card:#1e2128; --line:#2e323b; --txt:#e8eaee;
          --mut:#9aa1ad; --acc:#4f8ef7; }
  * { box-sizing:border-box; margin:0; }
  body { background:var(--bg); color:var(--txt);
         font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif; padding:28px; }
  .wrap { max-width:1100px; margin:0 auto; }
  h1 { font-size:22px; margin-bottom:4px; }
  .sub { color:var(--mut); margin-bottom:18px; }
  .legend { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px; }
  .legend span { display:inline-flex; align-items:center; gap:7px;
                 color:var(--mut); font-size:13px; }
  .sw { width:15px; height:15px; border-radius:3px; display:inline-block; }
  .cards { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
  @media (max-width:760px){ .cards{ grid-template-columns:1fr; } }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:18px; }
  .card h2 { font-size:15px; margin-bottom:12px; }
  label { display:block; font-size:12px; color:var(--mut); margin:10px 0 4px; }
  input,select { width:100%; padding:9px 10px; border-radius:7px;
    border:1px solid var(--line); background:#14161a; color:var(--txt);
    font-size:14px; }
  input[type=file] { padding:7px; }
  .row { display:flex; gap:10px; } .row > div { flex:1; }
  button { margin-top:14px; width:100%; padding:11px; border:0;
    border-radius:8px; background:var(--acc); color:#fff; font-size:14px;
    font-weight:600; cursor:pointer; }
  button:disabled { opacity:.5; cursor:wait; }
  .hint { font-size:12px; color:var(--mut); margin-top:8px; }
  #status { display:none; margin:18px 0; padding:12px 16px; border-radius:8px;
    background:var(--card); border:1px solid var(--line); align-items:center;
    gap:12px; }
  .spin { width:18px; height:18px; border:3px solid var(--line);
    border-top-color:var(--acc); border-radius:50%;
    animation:r 0.8s linear infinite; }
  @keyframes r { to { transform:rotate(360deg); } }
  #error { display:none; margin:18px 0; padding:12px 16px; border-radius:8px;
    background:#3a1d1d; border:1px solid #6e2c2c; color:#f2b8b8; }
  #result { display:none; margin-top:20px; }
  table { border-collapse:collapse; margin-bottom:14px; width:100%;
    max-width:560px; }
  td,th { padding:7px 12px; border-bottom:1px solid var(--line);
    text-align:left; font-size:14px; }
  th { color:var(--mut); font-weight:500; font-size:12px; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  #ovl { max-width:100%; border-radius:10px; border:1px solid var(--line); }
  .dl { display:inline-block; margin:0 12px 14px 0; color:var(--acc);
    font-size:13px; text-decoration:none; }
  footer { margin-top:34px; padding-top:14px; border-top:1px solid var(--line);
    font-size:12px; color:var(--mut); }
</style></head><body><div class="wrap">
<h1>UrbanSeg</h1>
<div class="sub">Residential / commercial / free-land segmentation of
Hyderabad satellite imagery</div>
<div class="legend" id="legend"></div>

<div class="cards">
  <div class="card">
    <h2>By coordinates</h2>
    <div class="row">
      <div><label>Latitude</label><input id="lat" value="17.4435"></div>
      <div><label>Longitude</label><input id="lng" value="78.3772"></div>
    </div>
    <label>Area size</label>
    <select id="grid">
      <option value="1">Small (~0.4 km)</option>
      <option value="2" selected>Medium (~0.7 km)</option>
      <option value="3">Large (~1.1 km)</option>
    </select>
    <button id="goCoords" onclick="segmentCoords()">Segment location</button>
    <div class="hint">Fetches live satellite imagery and fuses it with
    OpenStreetMap (and building footprints, if the index is built). Needs a
    Google Maps API key on the server. Takes 20&ndash;90 s.</div>
  </div>
  <div class="card">
    <h2>Upload satellite image</h2>
    <label>Image file (PNG/JPG)</label>
    <input type="file" id="file" accept="image/*">
    <button id="goUpload" onclick="segmentUpload()">Segment image</button>
    <div class="hint">Model-only (an uploaded image has no geolocation, so
    footprint fusion is off). Best at ~0.3 m/pixel &mdash; Google Maps
    zoom&nbsp;18 scale.</div>
  </div>
</div>

<div id="status"><div class="spin"></div><div id="statusTxt"></div></div>
<div id="error"></div>

<div id="result">
  <h2 style="font-size:15px;margin-bottom:10px">Result
    <span id="took" style="color:var(--mut);font-weight:400"></span></h2>
  <table id="stats"></table>
  <a class="dl" id="dlOverlay" download="segmentation_overlay.png">
    &#8595; overlay PNG</a>
  <a class="dl" id="dlGeo" download="segmentation.geojson"
     style="display:none">&#8595; GeoJSON</a>
  <div><img id="ovl"></div>
  <div class="hint" id="note"></div>
</div>

<footer>Imagery &copy; Google &middot; Map data &copy; OpenStreetMap
contributors (ODbL) &middot; Building footprints: Google Open Buildings
(CC BY 4.0) &middot; Trained for Hyderabad; results elsewhere are indicative
only. &middot; <a href="https://github.com/asblhyd/urbanseg"
style="color:var(--acc)">Source on GitHub</a></footer>

<script>
const META = __CLASS_META__;
const legend = document.getElementById('legend');
for (const [name, info] of Object.entries(META)) {
  legend.insertAdjacentHTML('beforeend',
    `<span><i class="sw" style="background:rgb(${info.color})"></i>${info.label}</span>`);
}
let timer = null;
function busy(on, msg) {
  document.getElementById('goCoords').disabled = on;
  document.getElementById('goUpload').disabled = on;
  const st = document.getElementById('status');
  st.style.display = on ? 'flex' : 'none';
  document.getElementById('error').style.display = 'none';
  if (on) {
    const t0 = Date.now();
    const txt = document.getElementById('statusTxt');
    txt.textContent = msg;
    clearInterval(timer);
    timer = setInterval(() => {
      txt.textContent = `${msg}  (${Math.round((Date.now()-t0)/1000)} s elapsed)`;
    }, 1000);
  } else clearInterval(timer);
}
function fail(msg) {
  busy(false);
  const e = document.getElementById('error');
  e.textContent = msg;
  e.style.display = 'block';
}
function render(d) {
  busy(false);
  document.getElementById('result').style.display = 'block';
  const srcNames = {osm: 'OpenStreetMap', footprints: 'building footprints'};
  document.getElementById('took').textContent = `— ${d.elapsed_s}s, ` +
    (d.fusion_active ? 'fused with ' + d.fusion_sources.map(k => srcNames[k]).join(' + ')
                     : 'model only');
  let rows = '<tr><th></th><th>Class</th><th>Share</th><th>Area</th></tr>';
  for (const [name, info] of Object.entries(META)) {
    const s = d.stats[name];
    rows += `<tr><td><i class="sw" style="background:rgb(${info.color})"></i></td>
      <td>${info.label}</td>
      <td class="num">${(s.share*100).toFixed(1)}%</td>
      <td class="num">${Math.round(s.area_m2).toLocaleString()} m&sup2;</td></tr>`;
  }
  document.getElementById('stats').innerHTML = rows;
  const src = 'data:image/png;base64,' + d.overlay_png_b64;
  document.getElementById('ovl').src = src;
  document.getElementById('dlOverlay').href = src;
  const g = document.getElementById('dlGeo');
  if (d.geojson) {
    g.href = URL.createObjectURL(new Blob([JSON.stringify(d.geojson)],
      {type:'application/geo+json'}));
    g.style.display = 'inline-block';
  } else g.style.display = 'none';
  document.getElementById('note').textContent = d.note || '';
  document.getElementById('result').scrollIntoView({behavior:'smooth'});
}
async function handle(resp) {
  if (!resp.ok) {
    let msg = `HTTP ${resp.status}`;
    try { msg += ': ' + (await resp.json()).detail; } catch {}
    throw new Error(msg);
  }
  return resp.json();
}
async function segmentCoords() {
  const lat = parseFloat(document.getElementById('lat').value);
  const lng = parseFloat(document.getElementById('lng').value);
  if (isNaN(lat) || isNaN(lng)) return fail('Enter numeric latitude and longitude.');
  const grid = document.getElementById('grid').value;
  busy(true, 'Fetching imagery, running segmentation and fusion…');
  try {
    render(await handle(await fetch(
      `/segment?lat=${lat}&lng=${lng}&grid=${grid}&geojson=true`)));
  } catch (e) { fail(e.message); }
}
async function segmentUpload() {
  const f = document.getElementById('file').files[0];
  if (!f) return fail('Choose an image file first.');
  const fd = new FormData(); fd.append('file', f);
  busy(true, 'Running segmentation on the uploaded image…');
  try {
    render(await handle(await fetch('/segment-image', {method:'POST', body:fd})));
  } catch (e) { fail(e.message); }
}
</script></div></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    meta = {
        "residential": {"color": "255,170,0", "label": "Residential building"},
        "commercial": {"color": "220,40,40", "label": "Commercial building"},
        "free_land": {"color": "80,200,120", "label": "Free land"},
        "other": {"color": "120,120,120", "label": "Roads / rail / water"},
    }
    return PAGE.replace("__CLASS_META__", json.dumps(meta))
