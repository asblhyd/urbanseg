"""fusion.fuse: roof mode keeps the colours on the imaged roofs; legacy mode is
infer.fuse_footprints with injected inputs.

Synthetic scenes are 240x240 px (~68 m) grids at zoom 18 / scale 2 with
footprints and OSM ways drawn in image px.
"""
import tracemalloc

import geopandas as gpd
import numpy as np
import pytest
from rasterio import features as rfeatures
from rasterio.transform import Affine
from scipy import ndimage as ndi
from shapely.geometry import LineString, Polygon
from shapely.ops import transform as shp_transform

import helpers
from urbanseg import config, fusion, imagery, infer, labels, open_buildings, osm_data

LAT, LNG = 17.44, 78.47
N = 240
BLOB = (slice(80, 160), slice(80, 160))
UNSURE = {0: 0.3, 2: 0.0, 3: 0.25}  # with p=0.45: building by plurality, P(building) 0.45


def grid_meta(h=N, w=N):
    s = config.SCALE
    cx, cy = imagery.latlng_to_world(LAT, LNG)
    return {"zoom": config.ZOOM, "scale": s, "x0": cx - w / s / 2, "y0": cy - h / s / 2,
            "width_map_px": w / s, "height_map_px": h / s, "center": [LAT, LNG], "nx": 1, "ny": 1}


def lnglat(meta, x, y):
    lat, lng = imagery.img_px_to_latlng(x, y, meta)
    return lng, lat


def rect(meta, x0, y0, x1, y1):
    """lng/lat polygon covering image px columns x0..x1-1 and rows y0..y1-1."""
    return Polygon([lnglat(meta, x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def hline(meta, y):
    return LineString([lnglat(meta, 0, y), lnglat(meta, N, y)])


def vline(meta, x):
    return LineString([lnglat(meta, x, 0), lnglat(meta, x, N)])


def vec(*p):
    """Class probabilities to assign to a block of probs[:, rows, cols]."""
    return np.array(p, np.float32)[:, None, None]


def footprints(*polys):
    return gpd.GeoDataFrame({"area_in_meters": [100.0] * len(polys),
                             "confidence": [0.9] * len(polys)}, geometry=list(polys), crs=4326)


def osm(roads=(), buildings=(), water=(), rails=(), waterways=()):
    layers = osm_data.parse_overpass({"elements": []})
    if roads:
        layers["roads"] = gpd.GeoDataFrame({"highway": [h for h, _ in roads]},
                                           geometry=[g for _, g in roads], crs=4326)
    if buildings:
        layers["buildings"] = gpd.GeoDataFrame(
            {"building": [t for t, _ in buildings], "levels": [""] * len(buildings)},
            geometry=[g for _, g in buildings], crs=4326)
    if water:
        layers["water"] = gpd.GeoDataFrame(geometry=list(water), crs=4326)
    if rails:
        layers["rails"] = gpd.GeoDataFrame(geometry=list(rails), crs=4326)
    if waterways:
        layers["waterways"] = gpd.GeoDataFrame({"waterway": [t for t, _ in waterways]},
                                               geometry=[g for _, g in waterways], crs=4326)
    return layers


def scene(klass=1, p=0.9, rest=None):
    """Free land with one model building blob (BLOB) of class `klass` at P = p.

    rest: probabilities of the other classes inside the blob (default: split evenly).
    """
    cm = np.full((N, N), 3, np.uint8)
    cm[BLOB] = klass
    probs = np.zeros((4, N, N), np.float32)
    probs[3], probs[0] = 0.9, 0.1
    blob = np.zeros((4, 1, 1), np.float32)
    blob[:] = (1 - p) / 3
    for k, v in (rest or {}).items():
        blob[k] = v
    blob[klass] = p
    probs[:, BLOB[0], BLOB[1]] = blob
    return cm, probs


def bld(a):
    return np.isin(a, (1, 2))


def far_footprint(meta):
    """A footprint in a corner: makes legacy run in footprint-index mode."""
    return footprints(rect(meta, 0, 0, 6, 6))


def test_roof_does_not_add_the_offset_strip_legacy_does():
    meta = grid_meta()
    cm, probs = scene()
    fp = footprints(rect(meta, 100, 90, 180, 170))  # ground footprint 20/10 px off the roof
    roof = fusion.fuse(cm, probs, dict(meta), layers=osm(), footprints=fp)
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=osm(), footprints=fp)
    assert not (bld(roof) & ~bld(cm)).any()
    assert (bld(roof) == bld(cm)).all()
    added = bld(legacy) & ~bld(cm)
    assert added[160:170, 100:180].all() and added[90:160, 160:180].all()


def test_roof_spreads_the_footprint_class_across_the_lean_within_one_blob():
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    cm[100:140, 20:220] = 1                # one long blob the model calls residential
    cm[150:170, 30:50] = 1                 # a separate blob 3 m below the footprint
    probs = np.zeros((4, N, N), np.float32)
    probs[3] = 1.0
    for sl in ((slice(100, 140), slice(20, 220)), (slice(150, 170), slice(30, 50))):
        probs[:, sl[0], sl[1]] = vec(0.05, 0.6, 0.3, 0.05)
    fp = footprints(rect(meta, 20, 100, 60, 140))
    shop = rect(meta, 25, 105, 55, 135)    # OSM says commercial
    layers = osm(buildings=[("commercial", shop)])
    m = dict(meta)
    out = fusion.fuse(cm, probs, m, layers=layers, footprints=fp)
    assert (out[100:140, 20:60] == 2).all()           # covered by the footprint
    lean_px = fusion.LEAN_M / (imagery.meters_per_map_px(LAT) / config.SCALE)
    assert (out[100:140, 60:60 + int(lean_px) - 1] == 2).all()  # same blob, within LEAN_M
    assert (out[100:140, 60 + int(lean_px) + 2:220] == 1).all()  # farther: the model's class
    assert (out[150:170, 30:50] == 1).all()           # other blob: not this building
    assert m["fusion_info"]["changed_px"]["class_change"] == int((out != cm).sum())
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=layers, footprints=fp)
    assert (legacy[100:140, 20:60] == 2).all() and (legacy[100:140, 60:220] == 1).all()
    # OSM outlines alone cover too few buildings to tell which one a pixel is
    osm_only = fusion.fuse(cm, probs, dict(meta), layers=layers, footprints=open_buildings.empty())
    assert (osm_only[105:135, 25:55] == 2).all()
    assert (osm_only[100:140, 60:220] == 1).all()


def test_lean_spread_takes_the_nearest_footprint_of_the_pixels_own_blob():
    """Other blobs' footprints decide nothing, even where they are nearer: in a
    blob's bounding box (a station in the corner of an L) or beside it."""
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    probs = np.zeros((4, N, N), np.float32)
    probs[3] = 1.0
    for rows, cols in ((slice(100, 200), slice(20, 40)), (slice(180, 200), slice(20, 140)),
                       (slice(100, 170), slice(60, 140)),  # the L and its station
                       (slice(20, 60), slice(20, 60)), (slice(20, 60), slice(62, 120))):
        cm[rows, cols] = 1
        probs[:, rows, cols] = vec(0.05, 0.6, 0.3, 0.05)
    fp = footprints(rect(meta, 20, 100, 40, 200), rect(meta, 60, 100, 140, 170),
                    rect(meta, 20, 20, 40, 60), rect(meta, 62, 20, 120, 60))
    layers = osm(buildings=[("commercial", rect(meta, 22, 102, 38, 198)),
                            ("train_station", rect(meta, 62, 102, 138, 168)),
                            ("commercial", rect(meta, 22, 22, 38, 58)),
                            ("residential", rect(meta, 64, 22, 118, 58))])
    out = fusion.fuse(cm, probs, dict(meta), layers=layers, footprints=fp)
    lean = int(fusion.LEAN_M / (imagery.meters_per_map_px(LAT) / config.SCALE))
    assert (out[180:200, 40:40 + lean - 1] == 2).all()   # the L's foot: its own footprint
    assert (out[180:200, 40 + lean + 1:140] == 1).all()  # beyond LEAN_M: the model's class
    assert (out[100:170, 60:140] == 0).all()
    assert (out[20:60, 20:60] == 2).all()                # blobs 2 px (0.6 m) apart
    assert (out[20:60, 62:120] == 1).all()


def test_road_through_a_confident_building_without_footprint_is_not_cut():
    meta = grid_meta()
    layers = osm(roads=[("residential", hline(meta, 120))])  # 6 m
    cm, probs = scene(p=0.95)
    out = fusion.fuse(cm, probs, dict(meta), layers=layers, footprints=open_buildings.empty())
    assert (out[BLOB] == 1).all()
    assert (out[115:125, 0:60] == 0).all()             # free land under the road -> other
    assert (out[100:105, 0:60] == 3).all()              # beside the road: untouched
    # legacy (with the footprint index) paints the full buffer over the roof
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=layers,
                         footprints=far_footprint(meta))
    assert (legacy[111:130, 80:160] == 0).all()


def test_road_cuts_only_unsure_uncovered_buildings_and_only_in_its_core():
    meta = grid_meta()
    layers = osm(roads=[("residential", hline(meta, 120))])
    cm, probs = scene(p=0.45, rest=UNSURE)  # building by plurality only
    m = dict(meta)
    out = fusion.fuse(cm, probs, m, layers=layers, footprints=open_buildings.empty())
    assert (out[116:124, 80:160] == 0).all()             # core: +-1.5 m of the centreline
    assert (out[110:113, 80:160] == 1).all() and (out[127:130, 80:160] == 1).all()
    assert m["fusion_info"]["changed_px"]["road_paint"] > 0
    covered = fusion.fuse(cm, probs, dict(meta), layers=layers,
                          footprints=footprints(rect(meta, 80, 80, 160, 160)))
    assert (covered[BLOB] == 1).all()                   # a footprint vouches for it


@pytest.mark.parametrize("kind", ["rail", "stream"])
def test_rails_and_streams_paint_8m_and_cut_unsure_buildings_only_in_the_core(kind):
    meta = grid_meta()
    line = hline(meta, 120)
    layers = osm(rails=[line]) if kind == "rail" else osm(waterways=[("stream", line)])
    cm, probs = scene(p=0.45, rest=UNSURE)
    out = fusion.fuse(cm, probs, dict(meta), layers=layers, footprints=open_buildings.empty())
    free = np.r_[0:60, 180:N]
    assert (out[107:133][:, free] == 0).all()            # 8 m = 28 px of free land
    assert (out[:105][:, free] == 3).all() and (out[135:][:, free] == 3).all()
    assert (out[114:126, 80:160] == 0).all()             # the central 4 m of the blob
    assert (out[100:112, 80:160] == 1).all() and (out[128:160, 80:160] == 1).all()
    cm, probs = scene(p=0.95)
    sure = fusion.fuse(cm, probs, dict(meta), layers=layers, footprints=open_buildings.empty())
    assert (sure[BLOB] == 1).all()


def test_roof_paints_ways_and_water_on_free_land_as_today():
    """Where the model sees no building, roof mode paints exactly what the
    footprint-index install paints today (labels.other_geoms_world): every
    highway value with config.ROAD_WIDTHS or the 5 m fallback, rails, waterways
    and water polygons (here not free land sure enough to be a dry bed)."""
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    probs = np.zeros((4, N, N), np.float32)
    probs[:] = vec(0.3, 0.15, 0.15, 0.4)
    roads = [(h, hline(meta, y)) for h, y in (("footway", 15), ("construction", 60),
                                              ("primary", 110), ("track", 190))]
    layers = osm(roads=roads, rails=[vline(meta, 30)], water=[rect(meta, 200, 200, 235, 235)],
                 waterways=[("stream", vline(meta, 100)), ("drain", vline(meta, 170))])
    kw = {"layers": layers, "footprints": far_footprint(meta)}
    roof = fusion.fuse(cm, probs, dict(meta), **kw)
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", **kw)
    assert ((roof == 0) == (legacy == 0)).all()
    assert 0.4 * N * N < (roof == 0).sum() < 0.75 * N * N


def test_dry_water_polygon_over_free_land_is_not_painted():
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    probs = np.zeros((4, N, N), np.float32)
    probs[3], probs[0] = 0.9, 0.1                       # confident free land: a dry tank bed
    probs[3, 140:200], probs[0, 140:200], probs[1, 140:200] = 0.4, 0.35, 0.25  # unsure
    cm[150:170, 150:170] = 1
    probs[:, 150:170, 150:170] = vec(0.05, 0.9, 0.0, 0.05)
    layers = osm(water=[rect(meta, 40, 40, 200, 200)])
    m = dict(meta)
    out = fusion.fuse(cm, probs, m, layers=layers, footprints=open_buildings.empty())
    assert (out[40:140, 40:200] == 3).all()
    assert (out[140:150, 40:200] == 0).all() and (out[170:200, 40:200] == 0).all()
    assert (out[150:170, 150:170] == 1).all()           # a model building is never water
    assert m["fusion_info"]["changed_px"]["water_paint"] == int((out == 0).sum())
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=layers,
                         footprints=open_buildings.empty())
    assert (legacy[40:140, 40:200] == 0).all()


def test_station_footprint_becomes_other():
    meta = grid_meta()
    cm, probs = scene(klass=2, p=0.8)
    station = rect(meta, 85, 85, 155, 155)
    layers = osm(buildings=[("train_station", station)])
    m = dict(meta)
    out = fusion.fuse(cm, probs, m, layers=layers, footprints=open_buildings.empty())
    assert (out[85:155, 85:155] == 0).all()
    assert m["fusion_info"]["changed_px"]["station"] > 0
    # with the footprint index the OSM tag reaches the Open Buildings footprint
    with_ob = fusion.fuse(cm, probs, dict(meta), layers=layers,
                          footprints=footprints(rect(meta, 80, 80, 160, 160)))
    assert (with_ob[BLOB] == 0).all()
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=layers,
                         footprints=open_buildings.empty())
    assert (legacy[85:155, 85:155] == 2).all()          # today: commercial


@pytest.mark.parametrize("mode", ["roof", "legacy"])
def test_footprints_the_model_barely_sees_do_not_vote(mode):
    meta = grid_meta()
    cm, probs = scene(p=0.6, rest={2: 0.3})             # the model: residential
    layers = osm(buildings=[("commercial", rect(meta, 100, 100, 140, 140))])
    stale = footprints(rect(meta, 0, 0, N, N))          # 11% model building: may be stale
    out = fusion.fuse(cm, probs, dict(meta), mode=mode, layers=layers, footprints=stale)
    assert (out == cm).all()
    seen = footprints(rect(meta, 40, 40, 200, 200))     # 25%: votes
    out = fusion.fuse(cm, probs, dict(meta), mode=mode, layers=layers, footprints=seen)
    assert (out[BLOB] == 2).all()


def test_vote_averages_only_the_model_building_pixels():
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    probs = np.zeros((4, N, N), np.float32)
    probs[3], probs[0] = 0.9, 0.1
    probs[:, 60:180, 60:180] = vec(0.0, 0.0, 0.49, 0.51)  # ground under the footprint
    cm[80:160, 80:120] = 1
    probs[:, 80:160, 80:120] = vec(0.05, 0.8, 0.15, 0.0)
    cm[80:160, 120:160] = 2
    probs[:, 80:160, 120:160] = vec(0.05, 0.35, 0.55, 0.05)
    fp = footprints(rect(meta, 60, 60, 180, 180))
    out = fusion.fuse(cm, probs, dict(meta), layers=osm(), footprints=fp)
    assert (out[BLOB] == 1).all()                       # the roof's mean: residential
    assert (out[~bld(cm)] == cm[~bld(cm)]).all()
    # today the mean runs over the whole footprint, ground included
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=osm(), footprints=fp)
    assert (legacy[60:180, 60:180] == 2).all()


@pytest.mark.parametrize("mode", ["roof", "legacy"])
@pytest.mark.parametrize("gap, expect", [(0.10, 2), (0.20, 1)])
def test_commercial_rules_decide_only_near_ties(mode, gap, expect):
    """An untagged footprint fronting a primary road, which the commercial rules
    call commercial, turns commercial only when |P(res) - P(com)| < 0.15."""
    meta = grid_meta()
    cm = np.full((N, N), 3, np.uint8)
    probs = np.zeros((4, N, N), np.float32)
    probs[3], probs[0] = 0.9, 0.1
    cm[60:100, 80:160] = 1
    probs[:, 60:100, 80:160] = vec(0.1, 0.45 + gap / 2, 0.45 - gap / 2, 0.0)
    fp = footprints(rect(meta, 80, 60, 160, 100))       # 11 m from the centreline
    layers = osm(roads=[("primary", hline(meta, 20))])
    assert labels.classify_footprints(labels.merge_footprints(fp, layers["buildings"]),
                                      layers).tolist() == [2]
    out = fusion.fuse(cm, probs, dict(meta), mode=mode, layers=layers, footprints=fp)
    assert (out[60:100, 80:160] == expect).all()


@pytest.mark.parametrize("mode", ["roof", "legacy"])
@pytest.mark.parametrize("index", [True, False], ids=["index", "osm-only"])
def test_residential_tag_is_decisive(mode, index):
    meta = grid_meta()
    cm, probs = scene(klass=2, p=0.6, rest={1: 0.3})    # the model: commercial
    layers = osm(buildings=[("apartments", rect(meta, 85, 85, 155, 155))])
    fp = footprints(rect(meta, 80, 80, 160, 160)) if index else open_buildings.empty()
    out = fusion.fuse(cm, probs, dict(meta), mode=mode, layers=layers, footprints=fp)
    assert (out[85:155, 85:155] == 1).all()


def test_nodata_does_not_count_against_a_footprint():
    """A footprint mostly outside an upload (NODATA) still votes on its valid part."""
    meta = grid_meta()
    cm, probs = scene(p=0.6, rest={2: 0.3})
    cm[80:160, 80:148] = fusion.NODATA                  # 85% of the footprint
    layers = osm(buildings=[("commercial", rect(meta, 82, 82, 158, 158))])
    out = fusion.fuse(cm, probs, dict(meta), layers=layers,
                      footprints=footprints(rect(meta, 80, 80, 160, 160)))
    assert (out[80:160, 148:160] == 2).all()
    assert (out[80:160, 80:148] == fusion.NODATA).all()


@pytest.mark.parametrize("mode, target", [("roof", "classify_footprints"),
                                          ("legacy", "other_geoms_world")])
def test_failure_inside_fusion_returns_the_input_and_records_the_reason(monkeypatch, mode, target):
    def boom(*a, **k):
        raise RuntimeError("bad geometry")
    monkeypatch.setattr(labels, target, boom)
    meta = grid_meta()
    cm, probs = scene()
    out = fusion.fuse(cm, probs, meta, mode=mode,
                      layers=osm(roads=[("residential", hline(meta, 120))]),
                      footprints=footprints(rect(meta, 80, 80, 160, 160)))
    assert out is cm
    assert meta["fusion"] == []
    info = meta["fusion_info"]
    assert info["error"] == "RuntimeError: bad geometry"
    assert info["sources"] == [] and info["changed_px"] == {} and info["mode"] == mode


def test_bad_inputs_never_raise():
    meta = grid_meta()
    cm, probs = scene()
    assert fusion.fuse(cm, probs[:, :10], meta, layers=osm()) is cm
    assert "sizes differ" in meta["fusion_info"]["error"]
    assert fusion.fuse(cm, probs, meta, mode="nope", layers=osm()) is cm
    assert "unknown fusion mode" in meta["fusion_info"]["error"]


@pytest.mark.parametrize("mode", ["roof", "legacy"])
def test_nodata_pixels_are_never_changed(mode):
    meta = grid_meta()
    cm, probs = scene(p=0.45, rest=UNSURE)
    cm[100:140, 0:200] = fusion.NODATA
    layers = osm(roads=[("residential", hline(meta, 120))], water=[rect(meta, 0, 0, 40, 240)])
    out = fusion.fuse(cm, probs, dict(meta), mode=mode, layers=layers,
                      footprints=footprints(rect(meta, 90, 90, 150, 150)))
    assert (out[100:140, 0:200] == fusion.NODATA).all()
    assert ((out == fusion.NODATA) == (cm == fusion.NODATA)).all()


def test_fusion_info_reports_sources_and_changes():
    meta = grid_meta()
    cm, probs = scene(p=0.45, rest=UNSURE)
    info_in = {"source": "cache", "query_version": "v3"}
    out = fusion.fuse(cm, probs, meta, layers=osm(roads=[("residential", hline(meta, 120))]),
                      footprints=footprints(rect(meta, 0, 0, 6, 6)), osm_info=info_in)
    info = meta["fusion_info"]
    assert set(info) == {"mode", "sources", "osm", "footprints_error", "changed_px", "error"}
    assert info["mode"] == "roof" and info["error"] is None and info["osm"] is info_in
    assert info["footprints_error"] is None
    assert meta["fusion"] == info["sources"] == ["footprints", "osm"]
    ch = info["changed_px"]
    assert set(ch) == {"class_change", "station", "road_paint", "water_paint", "total"}
    assert ch["total"] == int((out != cm).sum()) > 0
    assert ch["total"] == ch["class_change"] + ch["station"] + ch["road_paint"] + ch["water_paint"]
    # nothing to fuse with: model output unchanged, sources empty
    m = dict(meta)
    assert fusion.fuse(cm, probs, m, use_osm=False, footprints=open_buildings.empty()) is cm
    assert m["fusion"] == [] and m["fusion_info"]["changed_px"] == {"total": 0}
    assert m["fusion_info"]["osm"] is None


def test_live_osm_and_footprint_index_are_used_when_not_given(monkeypatch):
    meta = grid_meta()
    cm, probs = scene()
    fp = footprints(rect(meta, 80, 80, 160, 160))
    layers, info = osm(), {"source": "overpass", "query_version": "v3"}
    calls = []
    monkeypatch.setattr(osm_data, "get_live_osm",
                        lambda bbox, **k: calls.append(bbox) or (layers, info), raising=False)
    monkeypatch.setattr(open_buildings, "load_for_bbox", lambda bbox: calls.append(bbox) or fp)
    fusion.fuse(cm, probs, meta)
    assert calls == [imagery.bbox_latlng(meta)] * 2
    assert meta["fusion"] == ["footprints", "osm"] and meta["fusion_info"]["osm"] is info


def test_live_osm_fallback_fetches_without_a_cache_file(monkeypatch):
    """Until osm_data has get_live_osm: today's query, but no cache file keyed by
    a 24-bit bbox hash and written into config.OSM_DIR (the label snapshots)."""
    monkeypatch.delattr(osm_data, "get_live_osm", raising=False)
    calls = []

    def fetch(bbox, cache_path=None, **kw):
        calls.append((bbox, cache_path, kw))
        return {"elements": [], "osm3s": {"timestamp_osm_base": "2026-10-01T00:00:00Z"}}
    monkeypatch.setattr(osm_data, "fetch_overpass", fetch)
    meta = grid_meta()
    cm, probs = scene()
    fusion.fuse(cm, probs, meta, footprints=open_buildings.empty())
    assert calls == [(imagery.bbox_latlng(meta), None, fusion.LIVE_OVERPASS)]
    info = meta["fusion_info"]["osm"]
    assert info["source"] == "overpass" and info["cache_file"] is None
    assert info["query_version"] == "v2" and info["osm_base"] == "2026-10-01T00:00:00Z"
    assert meta["fusion"] == ["osm"] and not config.OSM_DIR.exists()


def test_osm_failure_fuses_footprints_only(monkeypatch):
    def down(bbox, **k):
        raise RuntimeError("overpass unavailable")
    monkeypatch.setattr(osm_data, "get_live_osm", down, raising=False)
    meta = grid_meta()
    cm, probs = scene()
    out = fusion.fuse(cm, probs, meta, footprints=footprints(rect(meta, 80, 80, 160, 160)))
    assert meta["fusion"] == ["footprints"]
    info = meta["fusion_info"]
    assert info["error"] is None and info["osm"]["source"] is None
    assert "overpass unavailable" in info["osm"]["error"]
    assert (bld(out) == bld(cm)).all()


def test_a_broken_footprint_index_is_reported(monkeypatch):
    def broken(bbox):
        raise OSError("truncated parquet")
    monkeypatch.setattr(open_buildings, "load_for_bbox", broken)
    meta = grid_meta()
    cm, probs = scene()
    fusion.fuse(cm, probs, meta, layers=osm())
    info = meta["fusion_info"]
    assert info["footprints_error"] == "OSError: truncated parquet"
    assert info["error"] is None and meta["fusion"] == info["sources"] == ["osm"]


def test_utm_zone_of_the_bbox_centre():
    assert fusion.utm_crs(config.HYD_BBOX) == 32644
    assert fusion.utm_crs((-0.2, 51.4, 0.0, 51.6)) == 32630
    assert fusion.utm_crs((151.1, -33.95, 151.3, -33.85)) == 32756
    assert fusion.utm_crs((179.9, 10, 180.0, 10.1)) == 32660


def test_roof_mode_peak_memory_stays_below_legacy():
    """The lean spread is solved blob by blob; a whole-image distance transform
    took roof mode to ~1.75x legacy's peak (636 vs 375 MB on a full area)."""
    if tracemalloc.is_tracing():
        pytest.skip("tracemalloc is already in use")
    n = 512
    meta = grid_meta(n, n)
    cm = np.full((n, n), 3, np.uint8)
    probs = np.zeros((4, n, n), np.float32)
    probs[3], probs[0] = 0.9, 0.1
    polys = []
    for y in range(8, n - 40, 64):
        for x in range(8, n - 40, 64):
            cm[y:y + 40, x:x + 40] = 1
            probs[:, y:y + 40, x:x + 40] = vec(0.05, 0.6, 0.3, 0.05)
            polys.append(rect(meta, x + 6, y + 6, x + 46, y + 46))  # leaning roofs
    kw = {"layers": osm(), "footprints": footprints(*polys)}
    peaks = {}
    for mode in ("legacy", "roof"):
        fusion.fuse(cm, probs, dict(meta), mode=mode, **kw)  # warm caches untraced
        m = dict(meta)
        tracemalloc.start()
        try:
            fusion.fuse(cm, probs, m, mode=mode, **kw)
            peaks[mode] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert m["fusion_info"]["error"] is None
    assert peaks["roof"] < peaks["legacy"]


_MAIN_LIVE_OVERPASS = {"timeout": 30, "rounds": 1, "pause": 0}
_MAIN_DECISIVE_TAGS = config.RES_BUILDING_TAGS | config.COM_BUILDING_TAGS


def _main_fuse_footprints(classmap, probs, meta, use_osm=True):
    """The code of infer.fuse_footprints before fusion existed (research commit
    bb91f89), verbatim apart from the two module constants it reads: the
    behaviour legacy mode must reproduce. Frozen here because infer is to
    delegate to fusion, which would make comparing with infer circular."""
    bbox = imagery.bbox_latlng(meta)
    meta["fusion"] = []
    try:
        fp = open_buildings.load_for_bbox(bbox)
    except Exception:  # noqa: BLE001
        fp = open_buildings.empty()
    if len(fp):
        meta["fusion"].append("footprints")
    layers = None
    if use_osm:
        try:
            layers = osm_data.get_area_osm(f"live_{hash(bbox) & 0xffffff:x}", bbox,
                                           **_MAIN_LIVE_OVERPASS)
            fp = labels.merge_footprints(fp, layers["buildings"])
            meta["fusion"].append("osm")
        except Exception:  # noqa: BLE001
            layers = None
    if not len(fp) and layers is None:
        return classmap
    s = meta["scale"]
    H, W = classmap.shape
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    zoom = meta["zoom"]
    world = lambda g: shp_transform(  # noqa: E731
        lambda x, y: labels._world_arrays(x, y, zoom), g)

    fp = fp.reset_index(drop=True)
    fp_ids = rfeatures.rasterize(
        [(world(g), i + 1) for i, g in enumerate(fp.geometry)],
        out_shape=(H, W), transform=tr, fill=0, dtype="int32")

    osm_only = "footprints" not in meta["fusion"]
    refined = classmap.copy()
    if layers is not None:
        geoms = labels.other_geoms_world(layers, zoom)
        if geoms:
            other = rfeatures.rasterize(
                [(g, 1) for g in geoms], out_shape=(H, W),
                transform=tr, fill=0, dtype="uint8").astype(bool)
            if osm_only:
                other &= ~np.isin(classmap, (1, 2))
            refined[other] = 0
    res_p, com_p = probs[1], probs[2]
    osm_cls = labels.classify_footprints(fp, layers) if layers is not None else None
    flat = fp_ids.ravel()
    order = np.argsort(flat, kind="stable")
    sorted_ids = flat[order]
    starts = np.searchsorted(sorted_ids, np.arange(1, len(fp) + 2))
    cm_flat, ref_flat = classmap.ravel(), refined.ravel()
    rp_flat, cp_flat = res_p.ravel(), com_p.ravel()
    for i in range(len(fp)):
        sel = order[starts[i]:starts[i + 1]]
        n = sel.size
        if n == 0:
            continue
        if osm_only and str(fp.iloc[i].get("tag") or "") not in _MAIN_DECISIVE_TAGS:
            continue
        bld_frac = float(np.isin(cm_flat[sel], (1, 2)).mean())
        if bld_frac < 0.20:
            continue  # model sees no building; footprint may be stale
        r, c = float(rp_flat[sel].mean()), float(cp_flat[sel].mean())
        klass = 1 if r >= c else 2
        if osm_cls is not None:
            tag = str(fp.iloc[i].get("tag") or "")
            if tag in config.RES_BUILDING_TAGS:
                klass = 1
            elif tag in config.COM_BUILDING_TAGS:
                klass = 2
            elif osm_cls[i] == 2 and abs(r - c) < 0.15:
                klass = 2
        ref_flat[sel] = klass
    return refined


def _fake_probs(mask, seed):
    """A stand-in model output: the labels moved a few px (like leaning roofs,
    so buildings spill onto road buffers) with smooth noise for ragged blobs.
    Every other label blob gets equal res/com logits, so footprints with near
    ties reach the commercial tie-break."""
    rng = np.random.default_rng(seed)
    moved = np.roll(mask, (6, -5), axis=(0, 1))
    logits = 2.0 * np.eye(4, dtype=np.float32)[moved].transpose(2, 0, 1)
    blobs, _ = ndi.label(np.isin(moved, (1, 2)))
    tied = blobs % 2 == 1
    logits[1][tied] = logits[2][tied] = 2.0
    noise = rng.normal(0, 5.0, logits.shape).astype(np.float32)
    logits += ndi.uniform_filter(noise, size=(1, 9, 9))
    e = np.exp(logits - logits.max(0))
    return (e / e.sum(0)).astype(np.float32)


@pytest.mark.data
@pytest.mark.parametrize("name, y0, x0", [("lbnagar", 1300, 1500), ("kompally", 2560, 2048)])
def test_legacy_equals_main_fuse_footprints_on_val_crops(monkeypatch, name, y0, x0):
    """Both crops hold untagged footprints the commercial rules mark, so the
    near-tie branch fires (lbnagar 4, kompally 15 footprints)."""
    img, mask, meta = helpers.load_area(name)
    img, mask, meta = helpers.crop(img, mask, meta, y0, y0 + 1024, x0, x0 + 1024)
    layers = helpers.osm_snapshot(name)
    ob = open_buildings.load_for_bbox(imagery.bbox_latlng(meta))
    assert len(ob) > 100
    probs = _fake_probs(mask, seed=len(name))
    cm = probs.argmax(0).astype(np.uint8)
    monkeypatch.setattr(osm_data, "get_area_osm", lambda *a, **k: layers)
    for fp, use_osm in ((ob, True), (open_buildings.empty(), True), (ob, False)):
        monkeypatch.setattr(open_buildings, "load_for_bbox", lambda bbox: fp)
        m_ref, m_new = dict(meta), dict(meta)
        ref = _main_fuse_footprints(cm, probs, m_ref, use_osm=use_osm)
        new = fusion.fuse(cm, probs, m_new, use_osm=use_osm, mode="legacy",
                          layers=layers, footprints=fp)
        assert m_new["fusion_info"]["error"] is None
        assert (new == ref).all()
        assert m_new["fusion"] == m_ref["fusion"]
        assert (new != cm).sum() > 1000
        roof = fusion.fuse(cm, probs, dict(meta), use_osm=use_osm, layers=layers, footprints=fp)
        assert not (bld(roof) & ~bld(cm)).any()


@pytest.mark.data
@pytest.mark.model
def test_roof_mode_on_a_real_prediction(model):
    img, mask, meta = helpers.load_area("lbnagar")
    img, mask, meta = helpers.crop(img, mask, meta, 1300, 1300 + 768, 1500, 1500 + 768)
    cm, probs = infer.predict(model, img)
    layers = helpers.osm_snapshot("lbnagar")
    ob = open_buildings.load_for_bbox(imagery.bbox_latlng(meta))
    legacy = fusion.fuse(cm, probs, dict(meta), mode="legacy", layers=layers, footprints=ob)
    m = dict(meta)
    roof = fusion.fuse(cm, probs, m, layers=layers, footprints=ob)
    assert m["fusion_info"]["error"] is None
    assert not (bld(roof) & ~bld(cm)).any()
    assert (bld(legacy) & ~bld(cm)).sum() > (bld(roof) & ~bld(cm)).sum()
    # roof mode removes far fewer model buildings than legacy road painting
    assert (bld(cm) & (roof == 0)).sum() < (bld(cm) & (legacy == 0)).sum() / 2
