"""FastAPI service: the map web app (urbanseg/web) and a JSON API over the pipeline.

Run: uvicorn urbanseg.api:app --host 127.0.0.1 --port 8000

New API (used by the web app): /api/config, /api/locate, /api/plan,
/api/jobs (+ /upload, /{id}, /{id}/files/{name}). Every analysis runs as a
job on one worker thread (jobs.py); clients poll it for progress.
The original endpoints /segment, /segment.png and /segment-image keep their
request and response shapes for existing scripts; they wait for their job.

Security: no login. Hosts other than this machine are refused unless listed
in URBANSEG_ALLOWED_HOSTS; cross-site browser requests are refused, so other
web pages cannot spend Google calls through a running service. The Google key
is sent to a browser only for the map basemap: GOOGLE_MAPS_BROWSER_KEY when
set, else the server key, and that only when the page was opened as
localhost / 127.0.0.1 (set URBANSEG_SHARE_SERVER_KEY=0 to never share it).
"""
import base64
import io
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import numpy as np
from fastapi import FastAPI, Form, Query, Request, UploadFile
from fastapi import File as FormFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import aoi, config, gold, infer, jobs, open_buildings, pipeline, uploads

log = logging.getLogger(__name__)

VERSION = "1.2.0"
WEB = Path(__file__).resolve().parent / "web"
ALLOWED_HOSTS = [h.strip() for h in os.environ.get(
    "URBANSEG_ALLOWED_HOSTS", "127.0.0.1,localhost,::1").split(",") if h.strip()]
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
SHARE_SERVER_KEY = os.environ.get("URBANSEG_SHARE_SERVER_KEY", "1") != "0"
# the gold-set editor writes to config.GOLD_DIR: off with URBANSEG_GOLD=0
GOLD_ENABLED = os.environ.get("URBANSEG_GOLD", "1") != "0"
MAX_LABELS_MB = 64
LEGACY_TIMEOUT_S = float(os.environ.get("URBANSEG_LEGACY_TIMEOUT_S", "900"))
# rough CPU model time per Mpx for the plan's estimate (an M5 does ~1.1 s)
SEC_PER_MPX = float(os.environ.get("URBANSEG_SEC_PER_MPX", "2.5"))
NO_KEY = ("coordinate mode needs a Google Maps API key: set GOOGLE_MAPS_KEY in "
          ".env and re-run ./run.sh (image upload works without one)")
CLASS_LABELS = {"other": "Roads / rail / water", "residential": "Residential building",
                "commercial": "Commercial building", "free_land": "Free land"}
CLASS_HELP = {"other": "roads, rail, water bodies, drains",
              "residential": "houses, apartment blocks, hostels",
              "commercial": "shops, offices, industry, warehouses, schools, hospitals, "
                            "places of worship",
              "free_land": "anything unbuilt: bare or vacant ground, fields, parks, trees"}

_models = {}  # checkpoint path -> loaded model
_models_lock = threading.Lock()
manager = jobs.JobManager()


def model(model_id=None):
    """The loaded model of an id of config.MODELS (None: the default), kept for reuse."""
    _, path = infer.resolve_model(model_id)
    with _models_lock:
        if str(path) not in _models:
            _models[str(path)] = infer.load_model(path)
        return _models[str(path)]


def _model_ids():
    """{id: checkpoint present} for config.MODELS."""
    return {k: (config.MODELS_DIR / m["file"]).is_file() for k, m in config.MODELS.items()}


def _check_model(model_id):
    """model_id if a job may use it (None = the default); else ApiError.

    Only ids of config.MODELS: a checkpoint is unpickled when loaded, so a
    path from a request is never opened."""
    if model_id is None:
        return None
    present = _model_ids()
    if model_id not in present:
        raise ApiError(422, "unknown_model", f"model: no model {model_id!r}",
                       hint=f"one of {', '.join(present)}")
    if not present[model_id]:
        raise ApiError(422, "model_unavailable",
                       f"model {model_id!r} is not installed on this server "
                       f"(models/{config.MODELS[model_id]['file']})")
    return model_id


@asynccontextmanager
async def lifespan(_app):
    model()  # load weights before /health reports ready
    yield


app = FastAPI(title="UrbanSeg", version=VERSION, lifespan=lifespan)


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Refuse requests a browser makes on behalf of another site: an <img> or
    form on any web page could otherwise spend this key's Google calls."""
    site = request.headers.get("sec-fetch-site")
    if site and site not in ("same-origin", "none"):
        return _error(403, "cross_site", "requests from other web pages are not accepted")
    origin = request.headers.get("origin")
    if (request.method not in ("GET", "HEAD", "OPTIONS") and origin
            and urlsplit(origin).netloc != request.headers.get("host")):
        return _error(403, "cross_site", "requests from other web pages are not accepted")
    return await call_next(request)


# added last, so it runs first: unknown Host headers (DNS rebinding) are refused
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


def _error(status, code, message, hint=None, headers=None):
    body = {"error": {"code": code, "message": message}}
    if hint is not None:
        body["error"]["hint"] = hint
    return JSONResponse(body, status_code=status, headers=headers)


class ApiError(Exception):
    def __init__(self, status, code, message, hint=None, headers=None):
        super().__init__(message)
        self.status, self.code, self.message, self.hint, self.headers = (
            status, code, message, hint, headers)


@app.exception_handler(aoi.AOIError)
async def _aoi_error(_request, ex):
    return _error(422, ex.code, str(ex), ex.hint)


@app.exception_handler(RequestValidationError)
async def _validation_error(request, ex):
    if not request.url.path.startswith("/api/"):  # original endpoints: FastAPI's format
        return JSONResponse({"detail": jsonable_encoder(ex.errors())}, status_code=422)
    errs = ex.errors()
    first = errs[0] if errs else {}
    field = ".".join(str(p) for p in first.get("loc", ())[1:]) or "request"
    return _error(422, "invalid", f"{field}: {first.get('msg', 'invalid value')}")


# --------------------------------------------------------------------- web app

@app.get("/", include_in_schema=False)
def index():
    return FileResponse(WEB / "index.html", headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/health")
def health():
    return {"ok": True, "version": VERSION, "classes": config.CLASS_NAMES,
            "coordinates_enabled": bool(config.GOOGLE_MAPS_KEY),
            "footprint_index": open_buildings.PARQUET.exists(),
            "fusion_mode": infer.FUSION_MODE, "model": getattr(model(), "info", None),
            "jobs": manager.status()}


def _maps_key(request):
    """(key for the browser's map, None) or (None, why not)."""
    browser = os.environ.get("GOOGLE_MAPS_BROWSER_KEY", "")
    if browser:
        return browser, None
    if not config.GOOGLE_MAPS_KEY:
        return None, "no Google Maps key is configured"
    host = (request.url.hostname or "").strip("[]").lower()
    if SHARE_SERVER_KEY and host in LOCAL_HOSTS:
        return config.GOOGLE_MAPS_KEY, None
    return None, ("the map needs a browser key here: set GOOGLE_MAPS_BROWSER_KEY to a key "
                  "with the Maps JavaScript API, restricted to this site's address")


@app.get("/api/config")
def api_config(request: Request):
    key, why = _maps_key(request)
    sizes = {}
    for g, name in ((1, "Small"), (2, "Medium"), (3, "Large")):
        p = aoi.plan(lat=17.4, lng=78.48, grid=g)
        sizes[g] = {"name": name, "extent_m": p["extent_m"], "extent_ha": p["extent_ha"],
                    "calls": p["calls"]}
    return {
        "version": VERSION,
        "classes": [{"id": k, "name": n, "label": CLASS_LABELS[n], "help": CLASS_HELP[n],
                     "color": list(config.CLASS_COLORS[k])}
                    for k, n in enumerate(config.CLASS_NAMES)],
        "coverage_bbox": list(config.HYD_BBOX),
        "coordinates_enabled": bool(config.GOOGLE_MAPS_KEY),
        "footprint_index": open_buildings.PARQUET.exists(),
        "fusion_mode": infer.FUSION_MODE,
        "qualities": list(infer.QUALITIES),
        "models": [{"id": k, "label": config.MODELS[k]["label"],
                    "about": config.MODELS[k]["about"], "available": ok}
                   for k, ok in _model_ids().items()],
        "default_model": infer.resolve_model()[0],
        "gold": {"enabled": GOLD_ENABLED, "statuses": list(gold.STATUSES)},
        "grid_sizes": sizes,
        "units": ["m2", "ha", "acres", "sq_yd", "guntas"],
        "limits": {"max_tiles": aoi.MAX_TILES, "max_aoi_km2": aoi.MAX_AOI_KM2,
                   "max_upload_mb": uploads.MAX_UPLOAD_MB,
                   "max_native_mpx": uploads.MAX_NATIVE_MPX},
        "maps": {"key": key, "reason": why},
        "jobs": manager.status(),
    }


# ----------------------------------------------------------------- planning

# module-level: inside AreaRequest the field "aoi" shadows the module
MAX_LOCATION_CHARS, DEFAULT_MARGIN_M, MAX_MARGIN_M = (
    aoi.MAX_LOCATION_CHARS, aoi.DEFAULT_MARGIN_M, aoi.MAX_MARGIN_M)


class LocateRequest(BaseModel):
    text: str = Field(..., max_length=MAX_LOCATION_CHARS)


class AreaRequest(BaseModel):
    aoi: dict | list | None = None
    location: str | None = Field(None, max_length=MAX_LOCATION_CHARS)
    lat: float | None = None
    lng: float | None = None
    grid: int | None = Field(None, ge=1, le=10)
    margin_m: float = Field(DEFAULT_MARGIN_M, ge=0, le=MAX_MARGIN_M)
    allow_outside: bool = False
    fuse: bool = True
    quality: Literal["standard", "high"] = "standard"
    model: str | None = Field(None, max_length=40)
    title: str | None = Field(None, max_length=120)


@app.post("/api/locate")
def api_locate(req: LocateRequest):
    lat, lng = aoi.parse_latlng(req.text)
    w, s, e, n = config.HYD_BBOX
    return {"lat": lat, "lng": lng, "in_coverage": s <= lat <= n and w <= lng <= e}


def _plan(req):
    lat, lng = req.lat, req.lng
    if req.aoi is None and req.location:
        lat, lng = aoi.parse_latlng(req.location)
    if req.aoi is None and (lat is None or lng is None):
        raise aoi.AOIError("give a zone (aoi) or a location (location, or lat and lng)")
    if req.aoi is not None:
        p = aoi.plan(req.aoi, margin_m=req.margin_m, allow_outside=req.allow_outside)
    else:
        p = aoi.plan(lat=lat, lng=lng, grid=req.grid or 2, allow_outside=req.allow_outside)
    return p


def _estimate_s(p, fuse, quality):
    mpx = p["nx"] * config.USABLE_X * config.SCALE * p["ny"] * config.USABLE_Y * config.SCALE / 1e6
    model_s = mpx * SEC_PER_MPX * (8 if quality == "high" else 1)
    return round(2 + 0.4 * p["calls"] + model_s + ((6 + 0.4 * mpx) if fuse else 0))


def _public_plan(p, req):
    out = {k: v for k, v in p.items() if k != "meta"}
    out["estimate_s"] = _estimate_s(p, req.fuse, req.quality)
    # the web app draws the tile grid from these (Web-Mercator world px)
    out["grid"] = {k: p["meta"][k] for k in ("zoom", "scale", "x0", "y0",
                                             "width_map_px", "height_map_px")}
    return out


@app.post("/api/plan")
def api_plan(req: AreaRequest):
    return _public_plan(_plan(req), req)


# --------------------------------------------------------------------- jobs

def _submit(kind, title, fn):
    try:
        return manager.submit(kind, title, fn)
    except jobs.QueueFull as ex:
        raise ApiError(503, "busy", f"the server is busy ({ex}): try again shortly",
                       headers={"Retry-After": "30"}) from ex


def _area_title(req, p):
    if req.title:
        return req.title
    lat, lng = p["center"]
    where = f"{lat:.5f}, {lng:.5f}"
    if p.get("aoi"):
        return f"Zone of {p['aoi_area_ha']:.2f} ha near {where}"
    return f"{p['extent_m'][0]:.0f} x {p['extent_m'][1]:.0f} m around {where}"


def _area_job(req):
    if not config.GOOGLE_MAPS_KEY:
        raise ApiError(503, "no_key", NO_KEY)
    model_id = _check_model(req.model)
    p = _plan(req)
    title = _area_title(req, p)
    job = _submit("area", title, lambda j: pipeline.run_area(
        model(model_id), p, fuse=req.fuse, quality=req.quality, progress=j.progress,
        cancel=j.cancelled, title=title))
    return job, p


@app.post("/api/jobs", status_code=202)
def api_jobs(req: AreaRequest):
    job, p = _area_job(req)
    return {"job": job.public(), "plan": _public_plan(p, req)}


async def _read_upload(file):
    limit = int(uploads.MAX_UPLOAD_MB * 2**20)
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise ApiError(413, "too_large", f"the upload is over the {uploads.MAX_UPLOAD_MB:.0f} "
                       "MB limit")
    if not raw:
        raise ApiError(400, "unreadable", "the upload is empty")
    return raw


def _upload_job(raw, filename, *, mpp=None, zoom=None, dpr=None, fuse=True,
                quality="standard", title=None, model_id=None):
    title = title or "Uploaded image"
    model_id = _check_model(model_id)

    def run(job):
        t = time.time()
        job.progress(stage="decode", label=pipeline.STAGE_LABELS["decode"], fraction=0.02)
        up = uploads.load_upload(raw, filename, mpp=mpp, zoom=zoom, dpr=dpr)
        if job.cancelled():
            raise pipeline.Cancelled("cancelled")
        return pipeline.run_upload(model(model_id), up, fuse=fuse, quality=quality,
                                   progress=job.progress, cancel=job.cancelled, title=title,
                                   timings={"decode": time.time() - t})
    return _submit("upload", title, run)


@app.post("/api/jobs/upload", status_code=202)
async def api_jobs_upload(file: UploadFile = FormFile(...),
                          mpp: float | None = Form(None), zoom: float | None = Form(None),
                          dpr: float | None = Form(None), fuse: bool = Form(True),
                          quality: Literal["standard", "high"] = Form("standard"),
                          model: str | None = Form(None, max_length=40),
                          title: str | None = Form(None, max_length=120)):
    _check_model(model)  # before the upload is read
    raw = await _read_upload(file)
    name = re.sub(r"[^\w.\- ]", "", os.path.basename(file.filename or ""))[:80]
    job = _upload_job(raw, file.filename or "", mpp=mpp, zoom=zoom, dpr=dpr, fuse=fuse,
                      quality=quality, title=title or (f"Upload {name}" if name else None),
                      model_id=model)
    return {"job": job.public()}


def _job(job_id):
    job = manager.get(job_id) if re.fullmatch(r"[0-9a-f]{32}", job_id) else None
    if job is None:
        raise ApiError(404, "not_found", "no such analysis (results are kept for an hour)")
    return job


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    return _job(job_id).public()


@app.delete("/api/jobs/{job_id}")
def api_job_cancel(job_id: str):
    _job(job_id)
    return manager.cancel(job_id).public()


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "-", (text or "urbanseg").lower()).strip("-")[:60] or "urbanseg"


@app.get("/api/jobs/{job_id}/files/{name}")
def api_job_file(job_id: str, name: str, download: bool = False):
    job = _job(job_id)
    f = job.files.get(name) if job.state == "done" else None
    if f is None:
        raise ApiError(404, "not_found", "no such file for this analysis")
    headers = {"Cache-Control": "private, max-age=3600"}
    if download:
        stamp = time.strftime("%Y%m%d-%H%M", time.localtime(job.finished or time.time()))
        headers["Content-Disposition"] = (
            f'attachment; filename="urbanseg_{_slug(job.title)}_{stamp}_{name}"')
    return Response(f.bytes(), media_type=f.media_type, headers=headers)


# ------------------------------------------------------------------ gold set

def _gold(fn, *a, **k):
    """Run a urbanseg.gold call, its errors as API errors."""
    if not GOLD_ENABLED:
        raise ApiError(404, "gold_disabled", "the gold-set editor is off on this server")
    try:
        return fn(*a, **k)
    except gold.GoldError as ex:
        raise ApiError({"not_found": 404, "full": 507}.get(ex.code, 422), ex.code, str(ex)) from ex
    except OSError as ex:
        log.warning("gold set: %s", type(ex).__name__)
        raise ApiError(503, "gold_unavailable", "the gold set cannot be written on this "
                       "server (URBANSEG_GOLD_DIR)") from ex


def _done_job(job_id):
    job = _job(job_id)
    if job.state != "done":
        raise ApiError(409, "not_done", "the analysis has not finished")
    return job


def _job_labels(job, source):
    """Class ids of a finished job for the editor: the model alone or fused, 255 outside the zone."""
    mask = np.asarray(Image.open(io.BytesIO(job.files["mask.png"].bytes())))
    if source == "model" and "raw_mask.png" in job.files:
        raw = np.asarray(Image.open(io.BytesIO(job.files["raw_mask.png"].bytes())))
        mask = np.where(mask == gold.IGNORE, gold.IGNORE, raw)
    return np.where(mask < config.N_CLASSES, mask, gold.IGNORE).astype(np.uint8)


async def _read_labels(file):
    limit = MAX_LABELS_MB * 2**20
    raw = await file.read(limit + 1)
    if len(raw) > limit:
        raise ApiError(413, "too_large", f"the labels are over {MAX_LABELS_MB} MB")
    return raw


@app.get("/api/jobs/{job_id}/gold-seed.png")
def api_gold_seed(job_id: str, source: Literal["model", "fused"] = "model"):
    """A job's classes as editor labels (gold.encode_labels)."""
    job = _done_job(job_id)
    return Response(_gold(gold.encode_labels, _job_labels(job, source)), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/gold")
def api_gold_list():
    return {"samples": _gold(gold.samples)}


@app.post("/api/gold", status_code=201)
async def api_gold_create(job_id: str = Form(...), labels: UploadFile = FormFile(...),
                          seed: Literal["model", "fused"] = Form("model"),
                          status: Literal["draft", "reviewed"] = Form("reviewed"),
                          note: str = Form("", max_length=500)):
    """A gold sample from a finished analysis: its imagery as served, the labels as edited."""
    job = _done_job(job_id)
    raw = await _read_labels(labels)
    img_bytes = job.files["imagery.jpg"].bytes()
    img = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
    lab = _gold(gold.decode_labels, raw, img.shape[:2])
    s = job.summary
    mid = (s.get("model") or {}).get("id")
    info = {"title": job.title, "source": {"kind": "job", "job_kind": job.kind,
                                           "quality": s.get("quality")},
            "seed": f"{seed}:{mid or 'custom'}", "model": s.get("model"), "grid": s["grid"],
            "georeferenced": s["georeferenced"], "center": s["center"],
            "status": status, "note": note}
    return _gold(gold.create, img, lab, info, image_bytes=img_bytes, image_name="image.jpg")


@app.get("/api/gold/{sample_id}")
def api_gold_info(sample_id: str):
    return _gold(gold.info, sample_id)


@app.get("/api/gold/{sample_id}/image")
def api_gold_image(sample_id: str):
    p = _gold(gold.image_file, sample_id)
    return FileResponse(p, media_type="image/jpeg" if p.suffix == ".jpg" else "image/png",
                        headers={"Cache-Control": "no-store"})


@app.get("/api/gold/{sample_id}/labels.png")
def api_gold_labels(sample_id: str):
    _, lab, _ = _gold(gold.load, sample_id)
    return Response(gold.encode_labels(lab), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.put("/api/gold/{sample_id}")
async def api_gold_update(sample_id: str, labels: UploadFile = FormFile(...),
                          status: Literal["draft", "reviewed"] | None = Form(None),
                          note: str | None = Form(None, max_length=500)):
    _gold(gold.info, sample_id)  # 404 before the upload is read
    raw = await _read_labels(labels)
    return _gold(gold.update, sample_id, raw, status=status, note=note)


# ------------------------------------------------- original endpoints (kept)

def _wait(job):
    if not job.done.wait(LEGACY_TIMEOUT_S):
        manager.cancel(job.id)
        raise ApiError(504, "timeout", "the analysis took too long")
    if job.state != "done":
        err = job.error or {"status": 500, "message": "failed"}
        raise ApiError(err["status"] if err["status"] != 499 else 503, err.get("code", "failed"),
                       f"segmentation failed: {err['message']}")
    return job


def _legacy_body(job, want_geojson):
    s = job.summary
    st = s["stats"]["extent"]["classes"]
    out = {
        "classes": config.CLASS_NAMES,
        "class_colors": {config.CLASS_NAMES[k]: list(v) for k, v in config.CLASS_COLORS.items()},
        "stats": {n: {k: v[k] for k in ("pixels", "area_m2", "share")} for n, v in st.items()},
        "fusion_active": s["fusion"]["active"],
        "fusion_sources": s["fusion"]["sources"],
        "elapsed_s": s["elapsed_s"],
        "overlay_png_b64": base64.b64encode(job.files["overlay_plain.png"].bytes()).decode(),
    }
    notes = s["notes"] + s["warnings"]
    if notes:
        out["note"] = " ".join(notes)
    if want_geojson and "result.geojson" in job.files:
        out["geojson"] = json.loads(job.files["result.geojson"].bytes())
    return out


@app.exception_handler(ApiError)
async def _api_error(request, ex):
    """The new API's {"error": ...} body; the original endpoints keep {"detail"}."""
    if request.url.path.startswith("/api/"):
        return _error(ex.status, ex.code, ex.message, ex.hint, ex.headers)
    return JSONResponse({"detail": ex.message}, status_code=ex.status, headers=ex.headers)


def _legacy_area(lat, lng, grid, fuse):
    if not config.GOOGLE_MAPS_KEY:
        raise ApiError(503, "no_key", NO_KEY)
    req = AreaRequest(lat=lat, lng=lng, grid=grid, fuse=fuse, allow_outside=True)
    try:
        job, p = _area_job(req)
    except aoi.AOIError as ex:
        raise ApiError(422, ex.code, str(ex)) from ex
    return _wait(job), p


@app.get("/segment")
def segment_coords(lat: float = Query(..., ge=-85, le=85),
                   lng: float = Query(..., ge=-180, le=180),
                   grid: int = Query(2, ge=1, le=3), fuse: bool = True,
                   geojson: bool = False):
    job, p = _legacy_area(lat, lng, grid, fuse)
    out = _legacy_body(job, geojson)
    out["center"] = [lat, lng]
    out["bbox_wsen"] = p["bbox_wsen"]
    return out


@app.get("/segment.png")
def segment_png(lat: float = Query(..., ge=-85, le=85),
                lng: float = Query(..., ge=-180, le=180),
                grid: int = Query(2, ge=1, le=3), fuse: bool = True,
                legend: bool = True):
    job, _ = _legacy_area(lat, lng, grid, fuse)
    name = "overlay.png" if legend else "overlay_plain.png"
    return Response(job.files[name].bytes(), media_type="image/png")


@app.post("/segment-image")
async def segment_upload(file: UploadFile = FormFile(...)):
    raw = await _read_upload(file)
    job = _upload_job(raw, file.filename or "")
    # waiting blocks: off the event loop, so /health and polling stay responsive
    job = await run_in_threadpool(_wait, job)
    return _legacy_body(job, False)
