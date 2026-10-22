"""Google Static Maps satellite imagery: tile math, fetching, stitching.

All geometry is done in Web-Mercator "map pixels" at a fixed zoom
(world = 256 * 2**zoom map px). Images are SCALE x map px.

The API key travels in the request URL, so URLs are secrets: fetch errors
name the tile centre, a reason and the HTTP status, never the URL, and
redact() scrubs any other text before it is logged or returned.
"""
import html
import io
import json
import math
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from . import config

WORLD = lambda zoom: 256 * (2 ** zoom)

STATIC_MAPS_URL = "https://maps.googleapis.com/maps/api/staticmap"
# transient failures wait ~1, 2, 4 s between attempts; the jitter keeps
# parallel workers from retrying in lockstep
RETRY_BASE_S = 1.0
RETRY_MAX_S = 8.0
RETRY_JITTER = 0.25

# also apikey=, api_key=...: over-redacting a harmless value costs nothing
_KEY_PARAM = re.compile(r"(?i)(key=)[^\s&\"'<>#()\[\],;]+")
# no '<' inside a tag: `<[^>]*>` backtracks quadratically over a run of '<'
_TAG = re.compile(r"<[^<>]*>")
# Google answers 403 both for a bad key / disabled API / billing and for an
# exhausted quota; only its text tells them apart
_QUOTA_TEXT = re.compile(r"(?i)quota|limit|exceeded|too many")
# an error page's message comes first; a proxy's page can be megabytes
_ERROR_BODY_BYTES = 1 << 16
_LONG_WORD = 120  # longer "words" are blobs (base64 logos, minified CSS), not text


def redact(text):
    """str(text) with the configured key and every key=<value> replaced by ***."""
    text = str(text)
    if config.GOOGLE_MAPS_KEY:
        text = text.replace(config.GOOGLE_MAPS_KEY, "***")
    return _KEY_PARAM.sub(r"\1***", text)


def _shown(word):
    """A word of an error text as shown: request URLs and blobs elided."""
    if "staticmap" in word.lower():
        return "[request]"  # a request URL, which carries the key
    return word if len(word) <= _LONG_WORD else "[...]"


def _clean_text(text, limit=240):
    """One short printable line that is safe to show: no markup, request URL or key.

    It runs on whatever body an error response brings (a proxy's page, an
    error image), so every step is linear in len(text) and none can raise.
    """
    text = _TAG.sub(" ", str(text or ""))
    try:
        text = html.unescape(text)
    except ValueError:  # a decimal entity too long for int(): keep the entities
        pass
    text = " ".join(text.split())
    if not text.isprintable():  # control/format characters go before redact(),
        text = "".join(c for c in text if c.isprintable())  # so none can split a key
    return " ".join(map(_shown, redact(text).split()))[:limit]


class TileFetchError(RuntimeError):
    """A satellite tile could not be fetched.

    .reason is one of REASONS and .status the HTTP status (None when there
    was no response). The message holds only the tile centre, the reason,
    the status and a short redacted detail (Google's error text or the
    exception's type name), never the request URL or the key.
    """
    REASONS = ("network", "auth", "quota", "bad_request", "server", "bad_image",
               "cancelled")

    def __init__(self, reason, status=None, lat=None, lng=None, detail=""):
        if reason not in self.REASONS:
            raise ValueError(f"unknown TileFetchError reason {reason!r}")
        self.reason, self.status, self.lat, self.lng = reason, status, lat, lng
        self.detail = _clean_text(detail)
        msg = "tile fetch failed"
        if lat is not None and lng is not None:
            msg += f" at {lat:.5f},{lng:.5f}"
        msg += f": {reason}"
        if status is not None:
            msg += f" (HTTP {status})"
        super().__init__(msg + (f": {self.detail}" if self.detail else ""))

    def __reduce__(self):  # the default would re-call __init__ with the message
        return type(self), (self.reason, self.status, self.lat, self.lng, self.detail)


def latlng_to_world(lat, lng, zoom=config.ZOOM):
    w = WORLD(zoom)
    x = (lng + 180.0) / 360.0 * w
    s = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * w
    return x, y


def world_to_latlng(x, y, zoom=config.ZOOM):
    w = WORLD(zoom)
    lng = x / w * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / w))))
    return lat, lng


def meters_per_map_px(lat, zoom=config.ZOOM):
    return 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)


def _http_reason(status, text):
    if status == 429:
        return "quota"
    if status >= 500:
        return "server"
    if status in (401, 403):
        return "quota" if _QUOTA_TEXT.search(text or "") else "auth"
    return "bad_request"


def _error_text(r):
    """The start of a failed response's body as text, for the error message."""
    ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype and not ctype.startswith("text/") and not ctype.endswith(("json", "xml")):
        return f"non-text body ({ctype})"  # an error image, a binary blob
    body = r.content or b""
    try:
        text = body[:_ERROR_BODY_BYTES].decode(getattr(r, "encoding", None) or "utf-8",
                                               "replace")
    except LookupError:  # a charset Python does not know
        text = body[:_ERROR_BODY_BYTES].decode("utf-8", "replace")
    if len(body) > _ERROR_BODY_BYTES:  # the cut may split a word, even the key
        text = text[:max(map(text.rfind, " \t\n\r\f\v")) + 1]
    return text


def _http_failure(r):
    """(reason, status, detail) for a response that is not a tile.

    A function of its own so that no frame of _fetch_raw_tile, which error
    trackers capture, holds the raw body: Google's error page may echo the
    request URL and key.
    """
    text = _error_text(r)
    return _http_reason(r.status_code, text), r.status_code, _clean_text(text)


def _decode_tile(r):
    """(RGB image, None) for a valid tile response, else (None, why)."""
    ctype = r.headers.get("content-type", "")
    if not ctype.startswith("image"):
        return None, f"content-type {ctype.split(';')[0] or 'missing'}, not an image"
    try:
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as ex:  # noqa: BLE001
        return None, f"undecodable image ({type(ex).__name__})"
    side = config.TILE_MAP_PX * config.SCALE
    if img.size != (side, side):
        return None, f"image is {img.size[0]}x{img.size[1]} px, expected {side}x{side}"
    return img, None


def _backoff(attempt):
    """Seconds to wait before retry number `attempt` (1, 2, ...)."""
    delay = min(RETRY_MAX_S, RETRY_BASE_S * 2 ** (attempt - 1))
    return delay * (1 + random.uniform(0, RETRY_JITTER))


def _pause(seconds, cancel=None):
    """time.sleep that returns early once cancel() is true."""
    end = time.monotonic() + seconds
    while (left := end - time.monotonic()) > 0:
        if cancel is not None and cancel():
            return
        time.sleep(min(left, 0.1))


def _fetch_raw_tile(center_lat, center_lng, session=None, max_retries=4, *,
                    cancel=None, timeout=30):
    """One 640x640 (map px) satellite image, SCALE x resolution.

    Makes up to max_retries attempts: network errors, HTTP 429 and 5xx are
    retried with exponential backoff; any other HTTP error and an invalid
    image fail at once. cancel() is polled before every attempt. Raises
    TileFetchError, never an exception that carries the request URL.
    """
    params = {
        "center": f"{center_lat:.7f},{center_lng:.7f}",
        "zoom": str(config.ZOOM),
        "size": f"{config.TILE_MAP_PX}x{config.TILE_MAP_PX}",
        "scale": str(config.SCALE),
        "maptype": "satellite",
    }
    sess = session or requests
    where = {"lat": center_lat, "lng": center_lng}
    for attempt in range(max(1, max_retries)):
        if attempt:
            _pause(_backoff(attempt), cancel)
        if cancel is not None and cancel():
            raise TileFetchError("cancelled", **where)
        try:
            # the key joins the params only here so that no local variable
            # (captured by debuggers and error trackers) holds it
            r = sess.get(STATIC_MAPS_URL, timeout=timeout,
                         params={**params, "key": config.GOOGLE_MAPS_KEY})
        except Exception as ex:  # noqa: BLE001
            # requests' messages embed the URL with key=: keep only the type
            # name, and raise after the loop, outside this handler, so that
            # the exception is not even in the error's __context__
            failure = ("network", None, type(ex).__name__)
            continue
        if r.status_code == 200:
            img, why = _decode_tile(r)
            if img is None:
                raise TileFetchError("bad_image", 200, detail=why, **where)
            return img
        failure = _http_failure(r)
        if not (r.status_code == 429 or r.status_code >= 500):
            break  # bad key, API off, no billing, bad parameters: retrying won't help
    reason, status, detail = failure
    raise TileFetchError(reason, status, detail=detail, **where)


def _save_atomic(write, path):
    """write(tmp) and rename it onto path, so readers never see a partial file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def fetch_area(center_lat, center_lng, nx, ny, cache_dir: Path | None = None,
               name: str | None = None, sleep=0.05, *, max_workers=4,
               progress=None, cancel=None):
    """Fetch an nx x ny grid of tiles around a center; return (image, meta).

    meta georeferences the stitched image: img px -> world map px -> lat/lng.
    Up to max_workers threads fetch tiles, each with its own requests.Session
    and a `sleep` s pause after each request; every tile is pasted at its fixed
    grid position as it arrives, so the result does not depend on completion
    order. With cache_dir, tile (i, j) is read from or written to
    cache_dir/"{i}_{j}.png"; with cache_dir and name, meta.json and
    cache_dir.parent/"{name}_full.png" are written too (all writes atomic).

    progress(done, total) and cancel() are only ever called in the calling
    thread, so they may use objects bound to it (a sqlite job table, say):
    progress after each tile; cancel before each tile is handed to a worker
    and every 0.1 s while tiles download. Once cancel() returns True, or once
    a tile fails, no new request starts, retry backoffs are cut short,
    requests in flight finish, and the error is raised:
    TileFetchError(reason="cancelled") or the failing tile's TileFetchError.
    """
    s = config.SCALE
    ux, uy = config.USABLE_X, config.USABLE_Y
    cx, cy = latlng_to_world(center_lat, center_lng)
    x0 = cx - nx * ux / 2.0
    y0 = cy - ny * uy / 2.0
    meta = {
        "zoom": config.ZOOM, "scale": s, "x0": x0, "y0": y0,
        "width_map_px": nx * ux, "height_map_px": ny * uy,
        "center": [center_lat, center_lng], "nx": nx, "ny": ny,
    }

    def check_cancel():
        if cancel is not None and cancel():
            raise TileFetchError("cancelled", lat=center_lat, lng=center_lng)

    check_cancel()
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    local, sessions = threading.local(), []
    stop = threading.Event()  # set once the area fails: tiles in flight stop retrying

    def start_worker():
        local.session = requests.Session()
        sessions.append(local.session)

    def tile(i, j):
        cached = None if cache_dir is None else cache_dir / f"{i}_{j}.png"
        if cached is not None and cached.exists():
            return Image.open(cached).convert("RGB")
        tx = x0 + i * ux
        ty = y0 + j * uy
        c_lat, c_lng = world_to_latlng(tx + config.TILE_MAP_PX / 2.0,
                                       ty + config.TILE_MAP_PX / 2.0)
        raw = _fetch_raw_tile(c_lat, c_lng, session=local.session, cancel=stop.is_set)
        t = raw.crop((0, 0, ux * s, uy * s))
        if cached is not None:
            _save_atomic(lambda p: t.save(p, format="PNG"), cached)
        time.sleep(sleep)
        return t

    full = Image.new("RGB", (nx * ux * s, ny * uy * s))
    cells = ((i, j) for j in range(ny) for i in range(nx))
    workers = max(1, int(max_workers))
    total, done, running = nx * ny, 0, {}
    pool = ThreadPoolExecutor(workers, initializer=start_worker)
    try:
        while True:
            check_cancel()
            # tiles go to the workers only as they free up, right after a
            # cancel check, so a cancel stops every request not yet started
            while len(running) < workers and (cell := next(cells, None)) is not None:
                running[pool.submit(tile, *cell)] = cell
            if not running:
                break
            finished, _ = wait(running, timeout=0.1, return_when=FIRST_COMPLETED)
            for fut in finished:
                i, j = running.pop(fut)  # so the tile is freed once pasted
                full.paste(fut.result(), (i * ux * s, j * uy * s))
                done += 1
                if progress is not None:
                    progress(done, total)
    finally:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        for sess in sessions:
            sess.close()
    if cache_dir is not None and name is not None:
        _save_atomic(lambda p: p.write_text(json.dumps(meta)), cache_dir / "meta.json")
        _save_atomic(lambda p: full.save(p, format="PNG"),
                     cache_dir.parent / f"{name}_full.png")
    return full, meta


def bbox_latlng(meta):
    """(west, south, east, north) of a stitched area."""
    n_lat, w_lng = world_to_latlng(meta["x0"], meta["y0"], meta["zoom"])
    s_lat, e_lng = world_to_latlng(meta["x0"] + meta["width_map_px"],
                                   meta["y0"] + meta["height_map_px"], meta["zoom"])
    return (w_lng, s_lat, e_lng, n_lat)


def latlng_to_img_px(lat, lng, meta):
    x, y = latlng_to_world(lat, lng, meta["zoom"])
    return (x - meta["x0"]) * meta["scale"], (y - meta["y0"]) * meta["scale"]


def img_px_to_latlng(px, py, meta):
    return world_to_latlng(meta["x0"] + px / meta["scale"],
                           meta["y0"] + py / meta["scale"], meta["zoom"])


def area_tiles_dir(name):
    """Tiles folder of an area: data/tiles (v1 areas) or data_v2/tiles (config.AREAS_V2)."""
    d = config.TILES_DIR
    if not (d / name / "meta.json").exists() and any(a[0] == name for a in config.AREAS_V2):
        d = config.DATA_V2_DIR / "tiles"
    return d


def load_area(name):
    d = area_tiles_dir(name)
    meta = json.loads((d / name / "meta.json").read_text())
    img = Image.open(d / f"{name}_full.png").convert("RGB")
    return np.asarray(img), meta
