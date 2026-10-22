"""Gold set: label encoding, sample storage, seeding from areas, scoring, API."""
import io
import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from urbanseg import aoi, api, config, gold, infer, jobs, osm_data, pipeline

LOCAL = "http://127.0.0.1"


def _labels(h=40, w=60):
    lab = np.full((h, w), 3, np.uint8)
    lab[5:20, 5:30] = 1
    lab[25:35, 40:55] = 2
    lab[:, :3] = 0
    lab[30:, :10] = gold.IGNORE
    return lab


def test_labels_round_trip_and_survive_small_colour_shifts():
    lab = _labels()
    raw = gold.encode_labels(lab)
    assert (gold.decode_labels(raw, lab.shape) == lab).all()
    grey = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB")).astype(np.int16)
    shifted = np.clip(grey + 12, 0, 255).astype(np.uint8)  # e.g. a colour-managed canvas
    buf = io.BytesIO()
    Image.fromarray(shifted).save(buf, format="PNG")
    assert (gold.decode_labels(buf.getvalue()) == lab).all()


def test_decode_refuses_other_images_and_sizes():
    lab = _labels()
    with pytest.raises(gold.GoldError) as ei:
        gold.decode_labels(gold.encode_labels(lab), (10, 10))
    assert ei.value.code == "size_mismatch"
    buf = io.BytesIO()
    Image.fromarray(np.arange(256, dtype=np.uint8).reshape(16, 16)).save(buf, format="PNG")
    with pytest.raises(gold.GoldError, match="grey levels"):
        gold.decode_labels(buf.getvalue())
    with pytest.raises(gold.GoldError, match="readable PNG"):
        gold.decode_labels(b"not a png")


def _image(h=40, w=60):
    return np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)


def test_samples_are_created_listed_updated_and_loaded():
    s = gold.create(_image(), _labels(), {"title": "t", "seed": "model:v1"})
    assert s["status"] == "draft" and s["size"] == [60, 40] and gold.ID_RE.fullmatch(s["id"])
    assert s["labelled_share"] == pytest.approx(1 - 100 / 2400, abs=1e-4)
    assert [x["id"] for x in gold.samples()] == [s["id"]]
    lab = _labels()
    lab[:, 50:] = 1
    u = gold.update(s["id"], gold.encode_labels(lab), status="reviewed", note="checked")
    assert u["status"] == "reviewed" and u["note"] == "checked" and u["edits"] == 1
    img, back, meta = gold.load(s["id"])
    assert (back == lab).all() and (img == _image()).all() and meta["status"] == "reviewed"
    # nothing half-written is left behind
    assert sorted(p.name for p in config.GOLD_DIR.iterdir()) == [s["id"]]


def test_bad_ids_and_inputs_are_refused(monkeypatch):
    for bad in ("../etc", "g1_zz", "", None):
        with pytest.raises(gold.GoldError) as ei:
            gold.info(bad)
        assert ei.value.code == "not_found"
    with pytest.raises(gold.GoldError, match="differ in size"):
        gold.create(_image(), _labels(10, 10), {})
    s = gold.create(_image(), _labels(), {})
    with pytest.raises(gold.GoldError):
        gold.update(s["id"], gold.encode_labels(_labels()), status="final")
    monkeypatch.setattr(gold, "MAX_SAMPLES", 1)
    with pytest.raises(gold.GoldError) as ei:
        gold.create(_image(), _labels(), {})
    assert ei.value.code == "full"


def _v2_area(monkeypatch, name="goldarea"):
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    h, w = int(meta["height_map_px"] * 2), int(meta["width_map_px"] * 2)
    d = config.DATA_V2_DIR / "tiles" / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps(meta))
    img = np.random.default_rng(1).integers(0, 255, (h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(config.DATA_V2_DIR / "tiles" / f"{name}_full.png")
    lab = np.full((h, w), 3, np.uint8)
    lab[100:200, 100:200] = 1
    lab[150:160] = gold.IGNORE
    (config.DATA_V2_DIR / "masks_v2t").mkdir(parents=True)
    Image.fromarray(lab).save(config.DATA_V2_DIR / "masks_v2t" / f"{name}.png")
    monkeypatch.setattr(config, "AREAS_V2", [(name, 17.44, 78.38, 1, 1, "test2", "plots")])
    return img, lab, meta


def test_seed_area_takes_a_window_of_the_auto_labels(monkeypatch):
    img, lab, meta = _v2_area(monkeypatch)
    s = gold.seed_area("goldarea", 80, 90, 160)
    assert s["status"] == "draft" and s["seed"] == "labels:v2t"
    assert s["source"] == {"kind": "area", "area": "goldarea", "split": "test2",
                           "window": [80, 90, 240, 250]}
    g = s["grid"]
    assert (g["x0"], g["y0"]) == (meta["x0"] + 45, meta["y0"] + 40)
    im, back, _ = gold.load(s["id"])
    assert (im == img[80:240, 90:250]).all() and (back == lab[80:240, 90:250]).all()
    with pytest.raises(gold.GoldError, match="window"):
        gold.seed_area("goldarea", 10_000, 0, 160)


def test_evaluate_scores_reviewed_samples_only(monkeypatch):
    lab = _labels(64, 64)
    a = gold.create(_image(64, 64), lab, {"seed": "labels:v2t", "status": "reviewed"})
    gold.create(_image(64, 64), lab, {"seed": "model:v1"})  # a draft
    net = type("Net", (), {"info": {"id": "v1"}})()
    monkeypatch.setattr(infer, "load_model", lambda m, device=None: net)
    perfect = np.where(lab == gold.IGNORE, 3, lab)
    monkeypatch.setattr(infer, "predict", lambda model, img: (perfect, None))
    res = gold.evaluate_models(["v1"], out=lambda *_: None)["v1"]
    assert list(res["samples"]) == [a["id"]] and res["pooled"]["pixel_acc"] == 1.0
    assert list(res["by_seed"]) == ["labels:v2t"]
    assert len(gold.evaluate_models(["v1"], include_drafts=True, out=lambda *_: None)
               ["v1"]["samples"]) == 2
    for s in gold.samples():
        (config.GOLD_DIR / s["id"] / "sample.json").unlink()
    with pytest.raises(gold.GoldError, match="no reviewed"):
        gold.evaluate_models(["v1"])


# ------------------------------------------------------------------- API

@pytest.fixture
def client(monkeypatch, fake_key):
    monkeypatch.setattr(api, "model", lambda model_id=None: None)

    def fake_predict(model, img, quality="standard", progress=None, cancel=None, **kw):
        h, w = img.shape[:2]
        cm = np.full((h, w), 3, np.uint8)
        cm[: h // 2] = 1
        probs = np.zeros((4, h, w), np.float32)
        probs[1, : h // 2] = probs[3, h // 2:] = 0.9
        return cm, probs

    def fetch(lat, lng, nx, ny, *a, progress=None, cancel=None, **k):
        m = aoi.grid_meta(lat, lng, nx, ny)
        w, h = round(m["width_map_px"] * 2), round(m["height_map_px"] * 2)
        return Image.new("RGB", (w, h), (120, 110, 100)), m

    def osm_down(bbox, **kw):
        raise osm_data.OverpassError("unavailable", "test")

    from urbanseg import imagery
    monkeypatch.setattr(infer, "predict_quality", fake_predict)
    monkeypatch.setattr(imagery, "fetch_area", fetch)
    monkeypatch.setattr(osm_data, "get_live_osm", osm_down)
    monkeypatch.setattr(pipeline, "_osm_down_until", 0.0)
    monkeypatch.setattr(pipeline, "_osm_outages", 0)
    api.manager = jobs.JobManager()
    with TestClient(api.app, base_url=LOCAL) as c:
        yield c


def _done_job(client):
    r = client.post("/api/jobs", json={"lat": 17.44, "lng": 78.38, "grid": 1})
    assert r.status_code == 202
    jid = r.json()["job"]["id"]
    for _ in range(400):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] != "queued" and j["state"] != "running":
            break
        time.sleep(0.05)
    assert j["state"] == "done", j
    return jid, j


def test_api_correct_a_result_into_a_gold_sample(client):
    jid, j = _done_job(client)
    assert client.get("/api/config").json()["gold"]["enabled"] is True
    seed = client.get(f"/api/jobs/{jid}/gold-seed.png?source=model")
    assert seed.status_code == 200
    lab = gold.decode_labels(seed.content)
    w, h = j["result"]["image_size"]
    assert lab.shape == (h, w) and (lab[: h // 2] == 1).all() and (lab[h // 2:] == 3).all()
    lab[:20, :20] = 2  # the analyst's fix
    lab[-5:] = gold.IGNORE
    r = client.post("/api/gold", data={"job_id": jid, "seed": "model", "note": "first"},
                    files={"labels": ("labels.png", gold.encode_labels(lab), "image/png")})
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["status"] == "reviewed" and s["seed"] == "model:custom" and s["grid"]["zoom"] == 18
    assert s["source"]["kind"] == "job" and s["image"] == "image.jpg"
    assert [x["id"] for x in client.get("/api/gold").json()["samples"]] == [s["id"]]
    img = client.get(f"/api/gold/{s['id']}/image")
    assert img.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(img.content)).size == (w, h)
    back = gold.decode_labels(client.get(f"/api/gold/{s['id']}/labels.png").content)
    assert (back == lab).all()
    lab[-5:] = 3
    r = client.put(f"/api/gold/{s['id']}", data={"status": "draft", "note": "second"},
                   files={"labels": ("labels.png", gold.encode_labels(lab), "image/png")})
    assert r.status_code == 200 and r.json()["status"] == "draft" and r.json()["note"] == "second"
    assert client.get(f"/api/gold/{s['id']}").json()["labelled_share"] == 1.0


def test_api_gold_errors(client, monkeypatch):
    jid, j = _done_job(client)
    small = gold.encode_labels(np.zeros((8, 8), np.uint8))
    r = client.post("/api/gold", data={"job_id": jid},
                    files={"labels": ("l.png", small, "image/png")})
    assert r.status_code == 422 and r.json()["error"]["code"] == "size_mismatch"
    r = client.post("/api/gold", data={"job_id": "0" * 32},
                    files={"labels": ("l.png", small, "image/png")})
    assert r.status_code == 404
    assert client.get("/api/gold/g20260101000000_abcdef").status_code == 404
    assert client.get("/api/gold/..%2F..%2Fetc").status_code == 404
    r = client.put("/api/gold/g20260101000000_abcdef",
                   files={"labels": ("l.png", small, "image/png")})
    assert r.status_code == 404
    # cross-site writes are refused like every other POST
    r = client.post("/api/gold", data={"job_id": jid}, headers={"origin": "https://evil.example"},
                    files={"labels": ("l.png", small, "image/png")})
    assert r.status_code == 403
    monkeypatch.setattr(api, "GOLD_ENABLED", False)
    r = client.get("/api/gold")
    assert r.status_code == 404 and r.json()["error"]["code"] == "gold_disabled"
