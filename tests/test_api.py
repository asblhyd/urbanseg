"""The HTTP API: security rules, key policy, planning, jobs, original endpoints.

The model and imagery are stubbed (fast); test_pipeline.py runs them for real.
"""
import base64
import io
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import conftest
from urbanseg import api, config, imagery, infer, jobs, osm_data, pipeline

LOCAL = "http://127.0.0.1"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api, "model", lambda model_id=None: None)  # fake predict needs none

    def fake_predict(model, img, quality="standard", progress=None, cancel=None, **kw):
        h, w = img.shape[:2]
        cm = np.full((h, w), 3, np.uint8)
        cm[: h // 2] = 1
        probs = np.zeros((4, h, w), np.float32)
        probs[1, : h // 2] = probs[3, h // 2:] = 0.9
        if progress:
            progress(1, 1)
        return cm, probs

    def osm_down(bbox, **kw):
        raise osm_data.OverpassError("unavailable", "test")

    monkeypatch.setattr(infer, "predict_quality", fake_predict)
    monkeypatch.setattr(osm_data, "get_live_osm", osm_down)
    api.manager = jobs.JobManager()  # a fresh queue per test
    monkeypatch.setattr(pipeline, "_osm_down_until", 0.0)  # no OSM backoff carried over
    monkeypatch.setattr(pipeline, "_osm_outages", 0)
    with TestClient(api.app, base_url=LOCAL) as c:
        yield c


@pytest.fixture
def fake_tiles(monkeypatch, fake_key):
    def fetch(lat, lng, nx, ny, *a, progress=None, cancel=None, **k):
        from urbanseg import aoi
        m = aoi.grid_meta(lat, lng, nx, ny)
        w, h = round(m["width_map_px"] * 2), round(m["height_map_px"] * 2)
        return Image.new("RGB", (w, h), (120, 110, 100)), m
    monkeypatch.setattr(imagery, "fetch_area", fetch)
    return fake_key


def _wait(client, job_id, timeout=20):
    t = time.time()
    while time.time() - t < timeout:
        j = client.get(f"/api/jobs/{job_id}").json()
        if j["state"] in ("done", "failed", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ security

def test_unknown_host_is_refused(client):
    r = client.get("/health", headers={"host": "evil.example"})
    assert r.status_code == 400


def test_cross_site_requests_are_refused(client):
    r = client.get("/health", headers={"sec-fetch-site": "cross-site"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "cross_site"
    r = client.get("/health", headers={"sec-fetch-site": "same-site"})  # another localhost port
    assert r.status_code == 403
    r = client.post("/api/locate", json={"text": "17.44, 78.37"},
                    headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get("/health", headers={"sec-fetch-site": "same-origin"}).status_code == 200
    assert client.post("/api/locate", json={"text": "17.44, 78.37"},
                       headers={"origin": LOCAL}).status_code == 200


def _req(host):
    return type("R", (), {"url": type("U", (), {"hostname": host})()})()


def test_maps_key_policy(client, monkeypatch, fake_key):
    # opened as 127.0.0.1 / localhost: the owner's own browser gets the .env key
    assert client.get("/api/config").json()["maps"]["key"] == fake_key
    assert api._maps_key(_req("localhost"))[0] == fake_key
    assert api._maps_key(_req("[::1]"))[0] == fake_key
    # any other host name (e.g. behind a reverse proxy) never sees the server key
    key, why = api._maps_key(_req("imageseg.dev.asbl.in"))
    assert key is None and "GOOGLE_MAPS_BROWSER_KEY" in why
    monkeypatch.setattr(api, "SHARE_SERVER_KEY", False)
    assert api._maps_key(_req("127.0.0.1"))[0] is None
    monkeypatch.setenv("GOOGLE_MAPS_BROWSER_KEY", "AIza-browser-key")
    assert api._maps_key(_req("imageseg.dev.asbl.in"))[0] == "AIza-browser-key"


def test_no_key_at_all(client):
    cfg = client.get("/api/config").json()
    assert cfg["maps"]["key"] is None and cfg["maps"]["reason"]


def test_config_and_static(client):
    cfg = client.get("/api/config").json()
    assert [c["name"] for c in cfg["classes"]] == config.CLASS_NAMES
    assert cfg["grid_sizes"]["2"]["calls"] == 4 and cfg["coordinates_enabled"] is False
    page = client.get("/")
    assert page.status_code == 200 and "/static/app.js" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/app.css").status_code == 200
    h = client.get("/health").json()
    assert h["ok"] and h["coordinates_enabled"] is False and "jobs" in h


def test_models_are_listed_and_chosen_by_id_only(client, fake_tiles, monkeypatch, tmp_path):
    monkeypatch.delenv("URBANSEG_MODEL", raising=False)
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    (tmp_path / "unet_r34.pt").write_bytes(b"")  # only v1 installed
    cfg = client.get("/api/config").json()
    assert {m["id"]: m["available"] for m in cfg["models"]} == {"v1": True, "v2": False}
    assert cfg["default_model"] == config.DEFAULT_MODEL
    area = {"lat": 17.44, "lng": 78.38, "grid": 1}
    # a checkpoint is unpickled when loaded: never a path from a request
    long_path = str(tmp_path / "unet_r34.pt")  # over the field's 40 characters
    assert client.post("/api/jobs", json={**area, "model": long_path}).status_code == 422
    for bad, code in (("v9", "unknown_model"), ("../models/unet_r34.pt", "unknown_model"),
                      ("v2", "model_unavailable")):
        r = client.post("/api/jobs", json={**area, "model": bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == code, bad
        r = client.post("/api/jobs/upload", files={"file": ("a.png", b"x", "image/png")},
                        data={"model": bad})
        assert r.status_code == 422 and r.json()["error"]["code"] == code, bad
    asked = []
    monkeypatch.setattr(api, "model", lambda model_id=None: asked.append(model_id))
    r = client.post("/api/jobs", json={**area, "model": "v1"})
    assert r.status_code == 202 and _wait(client, r.json()["job"]["id"])["state"] == "done"
    r = client.post("/api/jobs", json=area)
    assert r.status_code == 202 and _wait(client, r.json()["job"]["id"])["state"] == "done"
    assert asked == ["v1", None]  # None: the server's default


# ------------------------------------------------------------------ planning

def test_locate(client):
    r = client.post("/api/locate", json={"text": "17,4435; 78,3772"}).json()
    assert r == {"lat": 17.4435, "lng": 78.3772, "in_coverage": True}
    r = client.post("/api/locate", json={"text": "78.3772, 17.4435"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "swapped"
    assert r.json()["error"]["hint"] == {"lat": 17.4435, "lng": 78.3772}
    r = client.post("/api/locate", json={"text": "somewhere"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid"


def test_plan_point_and_zone(client):
    p = client.post("/api/plan", json={"lat": 17.4435, "lng": 78.3772, "grid": 2}).json()
    assert p["calls"] == 4 and "meta" not in p and p["grid"]["zoom"] == 18 and p["estimate_s"] > 0
    zone = [78.3770, 17.4430, 78.3785, 17.4442]  # ~2 ha
    p = client.post("/api/plan", json={"aoi": zone}).json()
    assert p["calls"] == 1 and p["aoi_area_ha"] == pytest.approx(2.1, abs=0.4)
    r = client.post("/api/plan", json={"aoi": [78.30, 17.40, 78.40, 17.48]})
    assert r.status_code == 422 and r.json()["error"]["code"] == "too_large"
    r = client.post("/api/plan", json={})
    assert r.status_code == 422
    r = client.post("/api/plan", json={"location": "17.44, 78.37", "grid": 99})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid"


def test_outside_coverage_needs_consent(client, fake_tiles):
    r = client.post("/api/jobs", json={"lat": 19.07, "lng": 72.87})
    assert r.status_code == 422 and r.json()["error"]["code"] == "outside_coverage"


# ---------------------------------------------------------------------- jobs

def test_area_job_without_key_is_refused(client):
    r = client.post("/api/jobs", json={"lat": 17.4435, "lng": 78.3772})
    assert r.status_code == 503 and r.json()["error"]["code"] == "no_key"


def test_area_job_runs_and_serves_files(client, fake_tiles):
    zone = [78.3770, 17.4430, 78.3785, 17.4442]
    r = client.post("/api/jobs", json={"aoi": zone, "title": "plot 7"})
    assert r.status_code == 202
    job = _wait(client, r.json()["job"]["id"])
    assert job["state"] == "done", job
    res = job["result"]
    assert res["title"] == "plot 7" and res["stats"]["zone"] is not None
    assert any("OpenStreetMap could not be reached" in n for n in res["notes"])
    png = client.get(f"/api/jobs/{job['id']}/files/classes.png")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    dl = client.get(f"/api/jobs/{job['id']}/files/stats.csv?download=1")
    assert "attachment" in dl.headers["content-disposition"] and "plot-7" in dl.headers["content-disposition"]
    assert client.get(f"/api/jobs/{job['id']}/files/../../etc/passwd").status_code == 404
    assert client.get(f"/api/jobs/{'0' * 32}").status_code == 404
    assert client.get("/api/jobs/not-a-job").status_code == 404


def test_imagery_failure_never_shows_the_key(client, fake_key, monkeypatch):
    def fetch(*a, **k):
        raise imagery.TileFetchError("network", lat=17.44, lng=78.37, detail="ConnectionError")
    monkeypatch.setattr(imagery, "fetch_area", fetch)
    r = client.post("/api/jobs", json={"lat": 17.4435, "lng": 78.3772})
    job = _wait(client, r.json()["job"]["id"])
    assert job["state"] == "failed" and job["error"]["code"] == "imagery_network"
    assert fake_key not in str(job) and "key=" not in str(job)
    legacy = client.get("/segment", params={"lat": 17.4435, "lng": 78.3772})
    assert legacy.status_code == 502 and fake_key not in legacy.text


def test_cancel_and_busy(client, fake_tiles, monkeypatch):
    gate = {"go": False}

    def slow_fetch(lat, lng, nx, ny, *a, progress=None, cancel=None, **k):
        while not gate["go"]:
            if cancel and cancel():
                raise imagery.TileFetchError("cancelled")
            time.sleep(0.01)
        raise imagery.TileFetchError("cancelled")

    monkeypatch.setattr(imagery, "fetch_area", slow_fetch)
    monkeypatch.setattr(jobs, "MAX_QUEUE", 1)
    first = client.post("/api/jobs", json={"lat": 17.4435, "lng": 78.3772}).json()["job"]
    time.sleep(0.2)  # first is running; one more may wait
    second = client.post("/api/jobs", json={"lat": 17.4435, "lng": 78.3772})
    assert second.status_code == 202
    third = client.post("/api/jobs", json={"lat": 17.4435, "lng": 78.3772})
    assert third.status_code == 503 and third.headers["retry-after"]
    assert client.delete(f"/api/jobs/{second.json()['job']['id']}").json()["state"] == "cancelled"
    client.delete(f"/api/jobs/{first['id']}")
    assert _wait(client, first["id"])["state"] == "cancelled"
    gate["go"] = True


def test_upload_job_png(client):
    buf = io.BytesIO()
    Image.fromarray(np.full((300, 400, 3), 128, np.uint8)).save(buf, format="PNG")
    r = client.post("/api/jobs/upload", files={"file": ("shot.png", buf.getvalue(), "image/png")},
                    data={"zoom": "17", "dpr": "1"})
    assert r.status_code == 202
    job = _wait(client, r.json()["job"]["id"])
    assert job["state"] == "done", job
    res = job["result"]
    assert res["upload"]["scale_source"] == "zoom" and not res["georeferenced"]
    # zoom 17 on a 1x screen is ~1.14 m/px: the 400x300 image covers ~0.16 km2
    assert res["stats"]["extent"]["total"]["area_m2"] == pytest.approx(
        400 * 300 * imagery.meters_per_map_px(17.4, 17) ** 2, rel=0.02)
    bad = client.post("/api/jobs/upload", files={"file": ("x.png", b"not an image", "image/png")})
    job = _wait(client, bad.json()["job"]["id"])
    assert job["state"] == "failed" and job["error"]["code"] == "unreadable"


# ---------------------------------------------------------- original endpoints

def test_original_segment_endpoint_shape(client, fake_tiles):
    r = client.get("/segment", params={"lat": 17.4435, "lng": 78.3772, "grid": 1, "geojson": "true"})
    assert r.status_code == 200, r.text
    d = r.json()
    assert set(d) >= {"classes", "class_colors", "stats", "fusion_active", "fusion_sources",
                      "elapsed_s", "overlay_png_b64", "center", "bbox_wsen", "geojson"}
    assert set(d["stats"]["residential"]) == {"pixels", "area_m2", "share"}
    Image.open(io.BytesIO(base64.b64decode(d["overlay_png_b64"]))).verify()
    png = client.get("/segment.png", params={"lat": 17.4435, "lng": 78.3772, "grid": 1})
    assert png.status_code == 200 and png.content[:8] == b"\x89PNG\r\n\x1a\n"
    r = client.get("/segment", params={"lat": 78.3772, "lng": 17.4435})
    assert r.status_code == 422  # a swapped pair is refused before any Google call


def test_original_upload_endpoint(client):
    buf = io.BytesIO()
    Image.fromarray(np.full((256, 256, 3), 90, np.uint8)).save(buf, format="PNG")
    r = client.post("/segment-image", files={"file": ("t.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["fusion_active"] is False and "scale unknown" in d["note"]
    assert conftest.FAKE_KEY not in r.text
