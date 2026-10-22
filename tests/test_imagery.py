"""imagery: key redaction, error bodies, retry policy, parallel stitching, progress, cancel.

Google is never called: requests.Session.get is replaced by a fake that
serves one synthetic tile per request centre, or the test scaffold's
network block answers with a real requests ConnectionError (whose message
embeds the URL and the key, which is exactly what must not escape).
"""
import base64
import functools
import hashlib
import io
import json
import pickle
import random
import shutil
import threading
import time
import traceback
import weakref

import numpy as np
import pytest
import requests
from PIL import Image

from urbanseg import config, imagery

SIDE = config.TILE_MAP_PX * config.SCALE


def _png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@functools.lru_cache(maxsize=None)
def tile_png(center):
    """A distinct, deterministic SIDE x SIDE tile for one request centre."""
    h = hashlib.sha1(center.encode()).digest()
    yy, xx = np.mgrid[0:SIDE, 0:SIDE]
    arr = np.stack([(xx + h[0]) % 256, (yy + h[1]) % 256,
                    (xx // 7 + yy // 5 + h[2]) % 256], -1).astype(np.uint8)
    arr[-config.CROP_BOTTOM_MAP_PX * config.SCALE:] = (255, 0, 255)  # watermark rows
    return _png(arr)


class FakeResponse:
    def __init__(self, status=200, content=b"", ctype="image/png", encoding=None):
        self.status_code, self.content, self.encoding = status, content, encoding
        self.headers = {"content-type": ctype}
        self.text = content.decode("utf-8", "replace")


def ok():
    return FakeResponse(200, tile_png("ok"))


def http(status, text="error", ctype="text/plain"):
    return FakeResponse(status, text.encode(), ctype)


class FakeGoogle:
    """Stands in for requests.Session.get: one synthetic tile per centre."""

    def __init__(self):
        self.calls, self.lock, self.jitter = [], threading.Lock(), 0.0
        self.respond = lambda params: FakeResponse(200, tile_png(params["center"]))

    def get(self, session, url, params=None, timeout=None, **kw):
        with self.lock:
            self.calls.append({"session": id(session), "thread": threading.get_ident(),
                               "url": url, **params})
        if self.jitter:
            time.sleep(random.uniform(0, self.jitter))  # shuffle completion order
        return self.respond(params)


@pytest.fixture
def google(monkeypatch, fake_key):
    fake = FakeGoogle()
    monkeypatch.setattr(requests.Session, "get",
                        lambda self, url, **kw: fake.get(self, url, **kw))
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.0)
    return fake


class Scripted:
    """A session whose get() plays back responses and exceptions in order."""

    def __init__(self, steps):
        self.steps, self.calls = list(steps), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        step = self.steps.pop(0)()
        if isinstance(step, BaseException):
            raise step
        return step


# ---- frozen copy of the sequential fetcher before parallel fetching -------

def _orig_fetch_raw_tile(center_lat, center_lng, session=None, max_retries=4):
    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{center_lat:.7f},{center_lng:.7f}",
        "zoom": str(config.ZOOM),
        "size": f"{config.TILE_MAP_PX}x{config.TILE_MAP_PX}",
        "scale": str(config.SCALE),
        "maptype": "satellite",
        "key": config.GOOGLE_MAPS_KEY,
    }
    sess = session or requests
    for attempt in range(max_retries):
        r = sess.get(url, params=params, timeout=30)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            if img.size == (config.TILE_MAP_PX * config.SCALE,) * 2:
                return img
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tile fetch failed at {center_lat},{center_lng}: HTTP {r.status_code}")


def orig_fetch_area(center_lat, center_lng, nx, ny, cache_dir=None, name=None, sleep=0.05):
    s = config.SCALE
    ux, uy = config.USABLE_X, config.USABLE_Y
    cx, cy = imagery.latlng_to_world(center_lat, center_lng)
    x0 = cx - nx * ux / 2.0
    y0 = cy - ny * uy / 2.0
    meta = {
        "zoom": config.ZOOM, "scale": s, "x0": x0, "y0": y0,
        "width_map_px": nx * ux, "height_map_px": ny * uy,
        "center": [center_lat, center_lng], "nx": nx, "ny": ny,
    }
    full = Image.new("RGB", (nx * ux * s, ny * uy * s))
    session = requests.Session()
    for j in range(ny):
        for i in range(nx):
            cached = None
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
                cached = cache_dir / f"{i}_{j}.png"
            if cached is not None and cached.exists():
                tile = Image.open(cached).convert("RGB")
            else:
                tx = x0 + i * ux
                ty = y0 + j * uy
                c_lat, c_lng = imagery.world_to_latlng(tx + config.TILE_MAP_PX / 2.0,
                                                       ty + config.TILE_MAP_PX / 2.0)
                raw = _orig_fetch_raw_tile(c_lat, c_lng, session=session)
                tile = raw.crop((0, 0, ux * s, uy * s))
                if cached is not None:
                    tile.save(cached)
                time.sleep(sleep)
            full.paste(tile, (i * ux * s, j * uy * s))
    if cache_dir is not None and name is not None:
        (cache_dir / "meta.json").write_text(json.dumps(meta))
        full.save(cache_dir.parent / f"{name}_full.png")
    return full, meta


# ---- the key never leaks --------------------------------------------------

def _library_locals(err):
    """repr of every local in the traceback's frames, as error trackers capture
    them, except the test's own frame (it holds the fake key on purpose)."""
    return "\n".join(f"{k}={v!r}" for frame, _ in traceback.walk_tb(err.__traceback__)
                     if frame.f_code.co_filename != __file__
                     for k, v in frame.f_locals.items())


def assert_no_secret(err, key):
    texts = [str(err), repr(err), "".join(traceback.format_exception(err)),
             _library_locals(err)]
    for text in texts:
        assert key not in text
        assert "googleapis" not in text and "staticmap?" not in text


def test_network_error_never_leaks_the_key(fake_key, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.0)
    # what the blocked network raises does carry the key: it must not escape
    with pytest.raises(Exception) as raw:
        requests.get(imagery.STATIC_MAPS_URL, params={"key": fake_key}, timeout=2)
    assert fake_key in str(raw.value)
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.4435, 78.3772, session=requests.Session())
    err = ei.value
    assert (err.reason, err.status) == ("network", None)
    assert err.__cause__ is None and err.__context__ is None
    assert str(err) == "tile fetch failed at 17.44350,78.37720: network: ConnectionError"
    assert_no_secret(err, fake_key)


def test_area_fetch_error_never_leaks_the_key(fake_key, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.0)
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery.fetch_area(17.4435, 78.3772, 2, 2, sleep=0)
    assert ei.value.reason == "network"
    assert_no_secret(ei.value, fake_key)


def test_google_error_text_is_shown_without_url_or_key(fake_key):
    page = ("<html><title>Error 404 (Not Found)!!1</title><p>The requested URL <code>"
            f"/maps/api/staticmap?center=1,2&amp;key={fake_key}</code> was not found "
            "on this server. That&#39;s all we know.</p></html>")
    sess = Scripted([lambda: http(404, page, "text/html")])
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=sess)
    assert ei.value.reason == "bad_request"
    assert "was not found on this server. That's all we know." in str(ei.value)
    assert_no_secret(ei.value, fake_key)

    text = f"The provided API key is invalid. (key={fake_key})"
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=Scripted([lambda: http(403, text)]))
    assert str(ei.value) == ("tile fetch failed at 17.44000,78.37000: auth (HTTP 403): "
                             "The provided API key is invalid. (key=***)")
    assert_no_secret(ei.value, fake_key)


def test_redact(fake_key):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=1,2&key={fake_key}&scale=2"
    assert imagery.redact(url) == ("https://maps.googleapis.com/maps/api/staticmap"
                                   "?center=1,2&key=***&scale=2")
    assert imagery.redact(f"bad key {fake_key}!") == "bad key ***!"
    assert (imagery.redact("?key=SOMEONE-ELSES&apikey=x&api_key=y")
            == "?key=***&apikey=***&api_key=***")
    assert imagery.redact(ValueError(f"KEY={fake_key}")) == "KEY=***"
    assert imagery.redact("center=17.4,78.3&zoom=18") == "center=17.4,78.3&zoom=18"


def test_redact_without_a_configured_key():
    assert config.GOOGLE_MAPS_KEY == ""
    assert imagery.redact("nothing to hide") == "nothing to hide"
    assert imagery.redact("a?key=x") == "a?key=***"


def test_tile_fetch_error_contract():
    err = imagery.TileFetchError("quota", 429, 17.4, 78.4, "Too Many Requests")
    assert isinstance(err, RuntimeError)
    assert str(err) == "tile fetch failed at 17.40000,78.40000: quota (HTTP 429): Too Many Requests"
    back = pickle.loads(pickle.dumps(err))
    assert (type(back), back.reason, back.status, str(back)) == (type(err), "quota", 429, str(err))
    assert str(imagery.TileFetchError("cancelled")) == "tile fetch failed: cancelled"
    with pytest.raises(ValueError):
        imagery.TileFetchError("nope")


def test_tile_fetch_error_cleans_any_detail_it_is_given(fake_key):
    url = f"https://maps.googleapis.com/maps/api/staticmap?center=1,2&key={fake_key}"
    err = imagery.TileFetchError("auth", 403, 17.4, 78.4,
                                 f"<b>Denied</b> for key={fake_key} ({fake_key}) at {url}")
    assert err.detail == "Denied for key=*** (***) at [request]"
    for text in (str(err), repr(err), str(pickle.loads(pickle.dumps(err)))):
        assert fake_key not in text and "googleapis" not in text


# ---- error bodies of any size or type ---------------------------------------

LOGO = base64.b64encode(bytes(range(256)) * 128).decode()  # one 43 KB "word"
PROXY_PAGE = (f"<html><head><style>.logo{{background:url(data:image/png;base64,{LOGO})}}"
              "</style></head><body><h1>503 Service Unavailable</h1></body></html>")


@pytest.mark.parametrize("body,detail", [
    ("a" * 64_000, "[...]"),
    ("<" * 64_000, "[...]"),
    ("&lt;" * 16_000, "[...]"),
    (PROXY_PAGE, "[...] 503 Service Unavailable"),
], ids=["no-whitespace", "unclosed-tags", "entities", "proxy-page"])
def test_large_error_bodies_are_cleaned_in_linear_time(monkeypatch, body, detail):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.0)
    sess = Scripted([lambda: http(503, body, "text/html")] * 2)
    t0 = time.perf_counter()
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=sess, max_retries=2)
    # quadratic patterns took ~10 s per attempt here, holding the GIL throughout
    assert time.perf_counter() - t0 < 1
    assert (ei.value.reason, ei.value.status, ei.value.detail) == ("server", 503, detail)


def test_only_the_start_of_a_huge_error_body_is_read():
    big = FakeResponse(502, b"Bad Gateway " + b"filler " * 1_000_000, "text/plain")
    assert len(imagery._error_text(big)) <= imagery._ERROR_BODY_BYTES
    t0 = time.perf_counter()
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=Scripted([lambda: big]), max_retries=1)
    assert time.perf_counter() - t0 < 0.5
    assert ei.value.detail.startswith("Bad Gateway filler filler")


def test_cut_error_body_never_shows_part_of_the_key(fake_key):
    # the cut at _ERROR_BODY_BYTES falls 10 characters into the key
    pad = imagery._ERROR_BODY_BYTES - len("Denied: ") - 10
    body = "Denied: " + " " * pad + fake_key + " was rejected"
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=Scripted([lambda: http(403, body)]))
    assert str(ei.value) == "tile fetch failed at 17.44000,78.37000: auth (HTTP 403): Denied:"


def test_binary_error_body_is_named_not_shown():
    png = _png(np.zeros((8, 8, 3), np.uint8))  # an error image, say
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37,
                                session=Scripted([lambda: FakeResponse(403, png, "image/png")]))
    assert str(ei.value) == ("tile fetch failed at 17.44000,78.37000: auth (HTTP 403): "
                             "non-text body (image/png)")


@pytest.mark.parametrize("response,detail", [
    (lambda: http(503, "Service\x00 Unavailable\x1b[0m"), "Service Unavailable[0m"),
    (lambda: http(503, "see \u202egnp.exe"), "see gnp.exe"),  # a right-to-left override
    (lambda: http(503, "bad &#" + "9" * 5000 + "; entity"), "bad [...] entity"),
    (lambda: FakeResponse(503, "Überlastet".encode(), "text/plain", encoding="x-bogus"),
     "Überlastet"),
], ids=["control-chars", "bidi-override", "huge-entity", "unknown-charset"])
def test_error_detail_is_printable_and_cleaning_never_raises(response, detail):
    with pytest.raises(imagery.TileFetchError) as ei:  # not ValueError, not LookupError
        imagery._fetch_raw_tile(17.44, 78.37, session=Scripted([response]), max_retries=1)
    assert ei.value.detail == detail and str(ei.value).isprintable()
    assert imagery.TileFetchError("server", detail="&#" + "9" * 5000).detail == "[...]"


# ---- retry and fail-fast policy -------------------------------------------

def _conn_error():
    return requests.exceptions.ConnectionError(
        "Max retries exceeded with url: /maps/api/staticmap?center=1,2&key=AIza-secret")


def _small_png():
    return FakeResponse(200, _png(np.zeros((512, 512, 3), np.uint8)))


MATRIX = {
    # steps (one per attempt) -> (reason, status) or None for success, attempts
    "ok": ([ok], None, 1),
    "network-then-ok": ([_conn_error, _conn_error, ok], None, 3),
    "timeout-then-ok": ([lambda: requests.exceptions.ReadTimeout("slow"), ok], None, 2),
    "5xx-then-ok": ([lambda: http(503), lambda: http(502), ok], None, 3),
    "429-then-ok": ([lambda: http(429), ok], None, 2),
    "network-always": ([_conn_error] * 4, ("network", None), 4),
    "500-always": ([lambda: http(500)] * 4, ("server", 500), 4),
    "429-always": ([lambda: http(429)] * 4, ("quota", 429), 4),
    "403-bad-key": ([lambda: http(403, "The provided API key is invalid.")], ("auth", 403), 1),
    "403-api-off": ([lambda: http(403, "This API project is not authorized to use this API.")],
                    ("auth", 403), 1),
    "403-billing": ([lambda: http(403, "You must enable Billing on the Google Cloud Project")],
                    ("auth", 403), 1),
    "403-quota": ([lambda: http(403, "You have exceeded your daily request quota for this API.")],
                  ("quota", 403), 1),
    "401": ([lambda: http(401)], ("auth", 401), 1),
    "400": ([lambda: http(400, "Invalid request. Invalid 'size' parameter.")],
            ("bad_request", 400), 1),
    "404": ([lambda: http(404)], ("bad_request", 404), 1),
    "network-then-403": ([_conn_error, lambda: http(403)], ("auth", 403), 2),
    "html-not-image": ([lambda: http(200, "<html>proxy login</html>", "text/html")],
                       ("bad_image", 200), 1),
    "wrong-size": ([_small_png], ("bad_image", 200), 1),
    "undecodable": ([lambda: FakeResponse(200, b"\x89PNG\r\n\x1a\n broken")],
                    ("bad_image", 200), 1),
}


@pytest.mark.parametrize("case", list(MATRIX))
def test_retry_and_fail_fast_matrix(case, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.0)
    steps, expect, attempts = MATRIX[case]
    sess = Scripted(steps)
    if expect is None:
        img = imagery._fetch_raw_tile(17.44, 78.37, session=sess)
        assert img.size == (SIDE, SIDE)
    else:
        with pytest.raises(imagery.TileFetchError) as ei:
            imagery._fetch_raw_tile(17.44, 78.37, session=sess)
        assert (ei.value.reason, ei.value.status) == expect
        assert "AIza-secret" not in str(ei.value)
    assert sess.calls == attempts


def test_backoff_is_exponential_with_jitter(monkeypatch):
    waits = []
    monkeypatch.setattr(imagery, "_pause", lambda s, cancel=None: waits.append(s))
    with pytest.raises(imagery.TileFetchError):
        imagery._fetch_raw_tile(17.44, 78.37, session=Scripted([lambda: http(503)] * 4))
    assert len(waits) == 3  # no pause after the last attempt
    for wait, base in zip(waits, (1.0, 2.0, 4.0)):
        assert base <= wait <= base * (1 + imagery.RETRY_JITTER)


# ---- parallel fetching ----------------------------------------------------

def _files(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


@pytest.mark.parametrize("nx,ny", [(3, 2), (2, 3), (1, 1)])
def test_parallel_fetch_is_identical_to_sequential(google, tmp_path, nx, ny):
    google.jitter = 0.02
    seq, par = tmp_path / "seq", tmp_path / "par"
    ref_img, ref_meta = orig_fetch_area(17.4435, 78.3772, nx, ny,
                                        cache_dir=seq / "area", name="area", sleep=0)
    ref_centers = sorted(c["center"] for c in google.calls)
    google.calls.clear()
    img, meta = imagery.fetch_area(17.4435, 78.3772, nx, ny, cache_dir=par / "area",
                                   name="area", sleep=0, max_workers=4)
    assert meta == ref_meta
    assert np.array_equal(np.asarray(img), np.asarray(ref_img))
    assert sorted(c["center"] for c in google.calls) == ref_centers
    # the same cache files, byte for byte, and no temp files left behind
    assert _files(par) == _files(seq) == sorted(
        [f"area/{i}_{j}.png" for i in range(nx) for j in range(ny)]
        + ["area/meta.json", "area_full.png"])
    for f in _files(seq):
        assert (par / f).read_bytes() == (seq / f).read_bytes(), f

    google.calls.clear()  # a second run is served from the cache alone
    img2, meta2 = imagery.fetch_area(17.4435, 78.3772, nx, ny, cache_dir=par / "area")
    assert google.calls == [] and meta2 == ref_meta
    assert np.array_equal(np.asarray(img2), np.asarray(ref_img))


def test_parallel_fetch_without_cache_matches_sequential(google):
    google.jitter = 0.01
    ref_img, ref_meta = orig_fetch_area(17.5453, 78.4854, 3, 3, sleep=0)
    img, meta = imagery.fetch_area(17.5453, 78.4854, 3, 3, sleep=0, max_workers=4)
    assert meta == ref_meta
    assert np.array_equal(np.asarray(img), np.asarray(ref_img))


def test_one_session_per_worker_and_all_closed(google, monkeypatch):
    closed, real_close = [], requests.Session.close

    def close(self):
        closed.append(id(self))
        real_close(self)

    monkeypatch.setattr(requests.Session, "close", close)
    google.jitter = 0.01
    imagery.fetch_area(17.44, 78.37, 3, 3, sleep=0, max_workers=3)
    threads = {}
    for c in google.calls:
        threads.setdefault(c["session"], set()).add(c["thread"])
    assert len(google.calls) == 9
    assert 1 <= len(threads) <= 3 and all(len(t) == 1 for t in threads.values())
    assert set(threads) <= set(closed)


def test_tiles_are_fetched_concurrently_by_max_workers(google):
    # the first four requests only return once all four are in flight: a
    # sequential fetch breaks the barrier and fails the area
    gate, real, lock = threading.Barrier(4, timeout=10), google.respond, threading.Lock()
    count = {"started": 0, "flying": 0, "peak": 0}

    def respond(params):
        with lock:
            count["started"] += 1
            count["flying"] += 1
            count["peak"] = max(count["peak"], count["flying"])
            first = count["started"] <= 4
        try:
            if first:
                gate.wait()
            return real(params)
        finally:
            with lock:
                count["flying"] -= 1

    google.respond = respond
    imagery.fetch_area(17.44, 78.37, 3, 2, sleep=0, max_workers=4)
    assert count["started"] == 6 and count["peak"] == 4  # never more than max_workers


def test_each_tile_is_freed_once_pasted(google, monkeypatch):
    crops, real_crop = [], Image.Image.crop

    def crop(self, box=None):
        out = real_crop(self, box)
        crops.append(weakref.ref(out))
        return out

    monkeypatch.setattr(Image.Image, "crop", crop)
    alive = []
    imagery.fetch_area(17.44, 78.37, 3, 3, sleep=0, max_workers=2,
                       progress=lambda d, t: alive.append(sum(r() is not None for r in crops)))
    assert len(crops) == 9 and len(alive) == 9
    assert max(alive) <= 2  # the tiles of 2 workers, not every tile fetched so far


@pytest.mark.data
def test_cached_dataset_tiles_restitch_to_the_dataset_image(google, tmp_path):
    name, lat, lng, nx, ny, _ = next(a for a in config.AREAS if a[0] == "kompally")
    src, dst = config.TILES_DIR / name, tmp_path / "tiles" / name
    dst.mkdir(parents=True)
    for i in range(nx):
        for j in range(ny):
            shutil.copyfile(src / f"{i}_{j}.png", dst / f"{i}_{j}.png")
    img, meta = imagery.fetch_area(lat, lng, nx, ny, cache_dir=dst, name=name)
    assert google.calls == []
    assert meta == json.loads((src / "meta.json").read_text())
    assert np.array_equal(np.asarray(img), imagery.load_area(name)[0])
    assert (dst / "meta.json").read_text() == (src / "meta.json").read_text()


# ---- progress and cancel --------------------------------------------------

def test_progress_reports_every_tile(google):
    seen = []
    imagery.fetch_area(17.44, 78.37, 2, 2, sleep=0, progress=lambda d, t: seen.append((d, t)))
    assert seen == [(1, 4), (2, 4), (3, 4), (4, 4)]


def test_cancel_before_start_makes_no_request(google):
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery.fetch_area(17.44, 78.37, 2, 2, cancel=lambda: True)
    assert ei.value.reason == "cancelled" and google.calls == []


def test_callbacks_run_only_in_the_calling_thread(google, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.05)  # backoffs that watch for a cancel
    me, failed_once = threading.get_ident(), set()

    def respond(params):  # every tile fails once, then arrives
        with google.lock:
            first = params["center"] not in failed_once
            failed_once.add(params["center"])
        return http(503) if first else FakeResponse(200, tile_png(params["center"]))

    def thread_bound(*args):  # like a job table on a sqlite connection made here
        if threading.get_ident() != me:
            raise RuntimeError("called from a worker thread")
        return False

    google.respond = respond
    imagery.fetch_area(17.44, 78.37, 2, 2, sleep=0, max_workers=4,
                       progress=thread_bound, cancel=thread_bound)
    assert len(google.calls) == 8


def test_cancel_mid_area_stops_new_requests(google):
    stop = threading.Event()

    def respond(params):
        stop.set()  # the user cancels while the first tile downloads
        return FakeResponse(200, tile_png(params["center"]))

    google.respond = respond
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery.fetch_area(17.44, 78.37, 3, 3, sleep=0, max_workers=1, cancel=stop.is_set)
    assert ei.value.reason == "cancelled"
    assert len(google.calls) == 1  # the request in flight finished; no new one started


def test_cancel_interrupts_retry_backoff(google, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 30.0)
    stop = threading.Event()

    def respond(params):
        stop.set()  # the user cancels while the first attempt fails
        return http(503)

    google.respond = respond
    t0 = time.monotonic()
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery._fetch_raw_tile(17.44, 78.37, session=requests.Session(), cancel=stop.is_set)
    assert ei.value.reason == "cancelled" and len(google.calls) == 1
    assert time.monotonic() - t0 < 2


def test_failing_tile_aborts_the_rest_of_the_area(google):
    google.respond = lambda params: http(403, "The provided API key is invalid.")
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery.fetch_area(17.44, 78.37, 3, 3, sleep=0, max_workers=1)
    assert (ei.value.reason, ei.value.status) == ("auth", 403)
    assert len(google.calls) <= 2  # not 9 tiles x 4 attempts


def test_failing_tile_stops_the_retries_of_tiles_in_flight(google, monkeypatch):
    monkeypatch.setattr(imagery, "RETRY_BASE_S", 0.5)  # 0.5 + 1 + 2 s of backoff
    seen = []

    def respond(params):
        with google.lock:
            seen.append(params["center"])
            first = len(seen) == 1
        if first:
            time.sleep(0.05)  # the other tile meanwhile fails into its backoff
            return http(403, "The provided API key is invalid.")
        return http(503)

    google.respond = respond
    t0 = time.monotonic()
    with pytest.raises(imagery.TileFetchError) as ei:
        imagery.fetch_area(17.44, 78.37, 2, 1, sleep=0, max_workers=2)
    assert ei.value.reason == "auth"
    assert time.monotonic() - t0 < 1.5 and len(google.calls) <= 3
