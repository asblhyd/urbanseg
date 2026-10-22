"""pipeline.run_area / run_upload end to end on cached imagery (no network)."""
import hashlib
import io
import json

import numpy as np
import pytest
import rasterio
from PIL import Image
from rasterio.io import MemoryFile
from shapely.geometry import box, shape

import conftest
import helpers
from urbanseg import aoi, imagery, infer, osm_data, pipeline, uploads

NAME = "kondapur"
NODATA = infer.NODATA


@pytest.fixture
def cached_fetch(monkeypatch):
    """imagery.fetch_area served from a cached area (exact grid, no network):
    the requested grid must lie inside NAME's stitched image."""
    img, _, meta = helpers.load_area(NAME)

    def fetch(lat, lng, nx, ny, *a, progress=None, cancel=None, **k):
        m = aoi.grid_meta(lat, lng, nx, ny)
        s = m["scale"]
        x0, y0 = round((m["x0"] - meta["x0"]) * s), round((m["y0"] - meta["y0"]) * s)
        w, h = round(m["width_map_px"] * s), round(m["height_map_px"] * s)
        assert 0 <= x0 and 0 <= y0 and x0 + w <= img.shape[1] and y0 + h <= img.shape[0]
        if progress:
            progress(nx * ny, nx * ny)
        return Image.fromarray(img[y0:y0 + h, x0:x0 + w]), m

    monkeypatch.setattr(imagery, "fetch_area", fetch)
    monkeypatch.setattr(pipeline, "_osm_down_until", 0.0)  # no OSM backoff carried over
    monkeypatch.setattr(pipeline, "_osm_outages", 0)
    layers = helpers.osm_snapshot(NAME)
    monkeypatch.setattr(osm_data, "get_live_osm",
                        lambda bbox, **k: (layers, {"source": "cache", "osm_base": "2026-08-15T00:00:00Z",
                                                    "elements": 1}))
    return meta


def _zone(meta, frac=(0.40, 0.40, 0.55, 0.52)):
    """A rectangle (w, s, e, n) inside the cached area."""
    w, s, e, n = imagery.bbox_latlng(meta)
    fx0, fy0, fx1, fy1 = frac
    return [w + (e - w) * fx0, s + (n - s) * fy0, w + (e - w) * fx1, s + (n - s) * fy1]


@pytest.mark.data
@pytest.mark.model
def test_run_area_with_a_zone(model, cached_fetch):
    zone = _zone(cached_fetch)
    plan = aoi.plan(zone)
    assert plan["calls"] == 1  # a ~1.6 ha zone needs one tile, not four
    seen = []
    summary, files = pipeline.run_area(model, plan, progress=lambda **kw: seen.append(kw),
                                       title="test zone")
    # progress: monotonic, ends in render
    fr = [kw["fraction"] for kw in seen]
    assert fr == sorted(fr) and seen[-1]["stage"] == "render"
    # the checkpoint the fixture loaded (its id is None when it lies outside MODELS_DIR)
    assert summary["model"]["file"] == "unet_r34.pt"
    assert summary["model"]["sha256"] == hashlib.sha256(conftest.MODEL_PATH.read_bytes()).hexdigest()
    assert summary["fusion"]["active"] and "osm" in summary["fusion"]["sources"]
    mo = summary["stats_model_only"]["zone"]  # the model's own figures, same pixels
    assert mo["total"]["pixels"] == summary["stats"]["zone"]["total"]["pixels"]
    assert mo["classes"] != summary["stats"]["zone"]["classes"]
    # stats cover the zone, not the whole image
    zs, ex = summary["stats"]["zone"], summary["stats"]["extent"]
    zone_m2 = plan["aoi_area_ha"] * 1e4
    assert zs["total"]["area_m2"] == pytest.approx(zone_m2, rel=0.03)
    assert ex["total"]["area_m2"] > 3 * zs["total"]["area_m2"]
    assert set(zs["total"]["units"]) == {"m2", "ha", "acres", "sq_yd", "guntas"}
    # downloads exist and agree with the zone
    for name in ("classes.png", "raw.png", "lowconf.png", "imagery.jpg", "mask.png",
                 "stats.csv", "classes.tif", "result.geojson", "result.json", "classes_full.png"):
        assert name in files and files[name].bytes(), name
    mask = np.asarray(Image.open(io.BytesIO(files["mask.png"].bytes())))
    inside = mask != NODATA
    assert inside.sum() == zs["total"]["pixels"]
    with MemoryFile(files["classes.tif"].bytes()) as mf, mf.open() as src:
        assert src.crs.to_epsg() == 3857 and src.nodata == NODATA
        assert np.array_equal(src.read(1), mask)
        assert src.colormap(1)[1][:3] == (255, 170, 0)
    fc = json.loads(files["result.geojson"].bytes())
    zone_geom = box(*zone)
    assert fc["features"] and all(shape(f["geometry"]).buffer(-1e-9).within(zone_geom.buffer(1e-9))
                                  for f in fc["features"])
    assert {f["properties"]["class"] for f in fc["features"]} <= {"other", "residential",
                                                                   "commercial", "free_land"}
    assert json.loads(files["result.json"].bytes())["title"] == "test zone"
    # the legend overlay is built on demand and is a real PNG
    assert Image.open(io.BytesIO(files["overlay.png"].bytes())).size[0] > mask.shape[1]


@pytest.mark.data
@pytest.mark.model
def test_run_area_cancel_and_osm_outage(model, cached_fetch, monkeypatch):
    plan = aoi.plan(_zone(cached_fetch))
    with pytest.raises(pipeline.Cancelled):
        pipeline.run_area(model, plan, cancel=lambda: True)

    def down(bbox, **k):
        raise osm_data.OverpassError("timeout", "all mirrors timed out")

    monkeypatch.setattr(osm_data, "get_live_osm", down)
    summary, _ = pipeline.run_area(model, plan)
    assert "osm" not in summary["fusion"]["sources"]
    assert any("OpenStreetMap could not be reached" in n for n in summary["notes"])


@pytest.mark.data
@pytest.mark.model
def test_run_upload_georeferenced_and_plain(model, cached_fetch, tmp_path):
    img, _, meta = helpers.load_area(NAME)
    crop, _, m = helpers.crop(img, None, meta, 1200, 2000, 1300, 2100)
    tif = tmp_path / "crop.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=crop.shape[1], height=crop.shape[0],
                       count=3, dtype="uint8", crs="EPSG:3857",
                       transform=uploads.grid_transform(m)) as dst:
        dst.write(crop.transpose(2, 0, 1))
    up = uploads.load_upload(tif.read_bytes(), "crop.tif")
    summary, files = pipeline.run_upload(model, up)
    assert summary["georeferenced"] and summary["fusion"]["active"]
    assert "result.geojson" in files and "classes.tif" in files
    assert summary["stats"]["extent"]["total"]["area_m2"] == pytest.approx(
        infer.pixel_area_m2(m["center"][0]) * crop.shape[0] * crop.shape[1], rel=0.01)

    png = io.BytesIO()
    Image.fromarray(crop).save(png, format="PNG")
    up = uploads.load_upload(png.getvalue(), "crop.png", mpp=0.57)  # declared 2x coarser
    summary, files = pipeline.run_upload(model, up)
    assert not summary["georeferenced"] and not summary["fusion"]["active"]
    assert summary["bbox_wsen"] is None and "result.geojson" not in files
    # 0.57 m/px: 4x the ground area of the same pixels at the native scale
    assert summary["stats"]["extent"]["total"]["area_m2"] == pytest.approx(
        crop.shape[0] * crop.shape[1] * 0.57 ** 2, rel=0.02)
    assert any("model alone" in n for n in summary["notes"])


def test_lowconf_and_csv_helpers():
    probs = np.zeros((4, 20, 30), np.float32)
    probs[1] = 0.9
    probs[1, :, :10] = 0.4  # unsure in the left third
    probs[3, :, :10] = 0.35
    rgba = np.asarray(Image.open(io.BytesIO(pipeline.lowconf_png(probs))))
    assert rgba.shape == (20, 30, 4)
    assert rgba[:, :10, 3].any() and not rgba[:, 10:, 3].any()
    st = {"classes": {"residential": {"share": 1.0, "pixels": 4, "units": aoi.area_units(10)}},
          "total": {"pixels": 4, "area_m2": 10, "units": aoi.area_units(10)}}
    text = pipeline.stats_csv({"stats": {"zone": None, "extent": st}}).decode()
    assert text.splitlines()[0].startswith("scope,class,share,pixels,area_m2")
    assert "extent,residential" in text and "extent,total" in text


@pytest.mark.data
@pytest.mark.model
def test_osm_outage_backs_off(model, cached_fetch, monkeypatch):
    monkeypatch.setattr(pipeline, "_osm_down_until", 0.0)
    plan = aoi.plan(_zone(cached_fetch))
    calls = []

    def down(bbox, deadline_s=None, **k):
        calls.append(deadline_s)
        raise osm_data.OverpassError("timeout", "all mirrors timed out")

    monkeypatch.setattr(osm_data, "get_live_osm", down)
    monkeypatch.setattr(pipeline, "_osm_outages", 0)
    for _ in range(pipeline.OSM_BACKOFF_AFTER):  # outages in a row trip the breaker
        pipeline.run_area(model, plan)
    summary, _ = pipeline.run_area(model, plan)  # within the backoff: cache only
    assert calls == [pipeline.OSM_DEADLINE_S] * pipeline.OSM_BACKOFF_AFTER + [0]
    assert any("skipped" in n for n in summary["notes"])
