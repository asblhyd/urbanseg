"""The aoi module: AOI parsing, tile planning, AOI masks/clipping, units, locations."""
import inspect
import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import shapely
from PIL import Image
from pyproj import CRS, Geod, Transformer
from shapely import affinity
from shapely.geometry import (LineString, MultiPolygon, Point, Polygon, box, mapping,
                              shape)

from urbanseg import aoi, config, imagery

GEOD = Geod(ellps="WGS84")
DMS = (17.4435, 78.3771944)  # 17°26'36.6"N 78°22'37.9"E
# the 2.0 ha Madhapur parcel of the diagnosis (aoi_clip_demo.py), in image px
MADHAPUR_PARCEL_PX = [(1480, 2080), (1990, 2060), (2060, 2470), (1700, 2560),
                      (1450, 2420)]


def _lnglat(geom_m, lat, lng):
    """A geometry drawn in metres on a local equal-area plane -> lon/lat."""
    crs = CRS.from_proj4(f"+proj=laea +lat_0={lat} +lon_0={lng} +ellps=WGS84 +units=m")
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    return shapely.transform(
        geom_m, lambda c: np.column_stack(t.transform(c[:, 0], c[:, 1])))


def _rect(lat, lng, w_m, h_m, angle=0.0):
    g = box(-w_m / 2, -h_m / 2, w_m / 2, h_m / 2)
    return _lnglat(affinity.rotate(g, angle, origin=(0, 0)), lat, lng)


def _random_aoi(rng, min_ha=0.5, max_ha=40.0):
    """A random AOI inside Hyderabad: rotated rectangle, star, holed or two-part."""
    lat, lng = rng.uniform(17.15, 17.65), rng.uniform(78.15, 78.75)
    area = 1e4 * math.exp(rng.uniform(math.log(min_ha), math.log(max_ha)))
    # never axis-aligned: an aligned edge makes the pixel-centre rule's error
    # coherent along it, which the analytic rectangle test covers instead
    angle = rng.uniform(5, 85) + 90 * rng.integers(4)
    kind = rng.choice(["rect", "star", "holed", "multi"])
    if kind == "star":
        n = int(rng.integers(5, 13))
        a = np.sort(rng.uniform(0, 2 * np.pi, n))
        r = rng.uniform(0.45, 1.0, n)
        g = Polygon(np.column_stack([r * np.cos(a), r * np.sin(a)]))
    else:
        asp = math.exp(rng.uniform(math.log(1 / 3), math.log(3)))
        g = box(-asp / 2, -0.5, asp / 2, 0.5)
        if kind == "holed":
            g = g.difference(box(-asp / 6, -0.15, asp / 6, 0.15))
        elif kind == "multi":
            g = MultiPolygon([affinity.translate(g, -asp * 0.6),
                              affinity.translate(g, asp * 0.6)])
    k = math.sqrt(area / g.area)
    g = affinity.rotate(affinity.scale(g, k, k, origin=(0, 0)), angle, origin=(0, 0))
    return _lnglat(g, lat, lng)


def _ground_mpp(lat, lng=78.0):
    """Reference WGS84 ground metres per map px at lat, (east-west, north-south),
    measured with pyproj over one map px centred there."""
    x, y = imagery.latlng_to_world(lat, lng)
    lat0, lng0 = imagery.world_to_latlng(x - 0.5, y - 0.5)
    lat1, lng1 = imagery.world_to_latlng(x + 0.5, y + 0.5)
    return GEOD.inv(lng0, lat, lng1, lat)[2], GEOD.inv(lng, lat0, lng, lat1)[2]


def _wgs84_px_area(meta):
    """Mean true (WGS84) ground area of one image pixel of an extent."""
    s = meta["scale"]
    n_px = meta["width_map_px"] * s * meta["height_map_px"] * s
    return aoi.geodesic_area_m2(box(*imagery.bbox_latlng(meta))) / n_px


def _shape_of(meta):
    s = meta["scale"]
    return int(round(meta["height_map_px"] * s)), int(round(meta["width_map_px"] * s))


def _area_meta(name):
    return json.loads((config.TILES_DIR / name / "meta.json").read_text())


def _classmap(name):
    """A val-area classmap: the precomputed release prediction when
    URBANSEG_PRED_DIR points at it, else the auto-label mask."""
    pred = os.environ.get("URBANSEG_PRED_DIR")
    p = Path(pred) / f"{name}_fused_osmonly.png" if pred else None
    if p is None or not p.is_file():
        p = config.MASKS_DIR / f"{name}.png"
    return np.asarray(Image.open(p)), _area_meta(name)


def _to_ll(meta):
    return lambda x, y: imagery.img_px_to_latlng(x, y, meta)[::-1]


def _code(fn, *a, **k):
    with pytest.raises(aoi.AOIError) as ei:
        fn(*a, **k)
    return ei.value


# ---------------------------------------------------------------- AOIError

def test_aoi_error_contract():
    err = aoi.AOIError("latitude and longitude look swapped", code="swapped",
                       hint={"lat": 17.4, "lng": 78.4})
    assert isinstance(err, ValueError) and err.code == "swapped"
    assert err.as_dict() == {"code": "swapped", "message": str(err),
                             "hint": {"lat": 17.4, "lng": 78.4}}
    assert aoi.AOIError("x").code == "invalid"
    with pytest.raises(ValueError):
        aoi.AOIError("x", code="nope")


# ---------------------------------------------------------------- parse_aoi

SQ = box(78.39, 17.445, 78.392, 17.447)


@pytest.mark.parametrize("obj", [
    mapping(SQ),
    {"type": "Feature", "properties": {"name": "plot"}, "geometry": mapping(SQ)},
    {"type": "FeatureCollection",
     "features": [{"type": "Feature", "geometry": mapping(SQ)}]},
    [78.39, 17.445, 78.392, 17.447],
    (78.39, 17.445, 78.392, 17.447),
    json.dumps(mapping(SQ)),
    SQ,
    {"type": "GeometryCollection", "geometries": [mapping(SQ)]},
])
def test_parse_aoi_accepts_every_input_form(obj):
    g = aoi.parse_aoi(obj)
    assert g.geom_type == "Polygon" and g.is_valid and g.exterior.is_ccw
    assert g.equals(SQ)


def test_parse_aoi_unions_all_polygons():
    a, b = box(78.39, 17.445, 78.392, 17.447), box(78.391, 17.446, 78.393, 17.448)
    far = box(78.40, 17.445, 78.401, 17.446)
    fc = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "geometry": mapping(a)},
        {"type": "Feature", "geometry": mapping(MultiPolygon([b, far]))}]}
    assert aoi.parse_aoi(fc).equals(shapely.union_all([a, b, far]))
    # overlapping parts of one (invalid) MultiPolygon are unioned, not cancelled
    g = aoi.parse_aoi(MultiPolygon([a, b]))
    assert g.is_valid and g.area == pytest.approx(shapely.union_all([a, b]).area)


@pytest.mark.parametrize("geom, kind", [
    (Point(78.39, 17.44), "Point"),
    (LineString([(78.39, 17.44), (78.40, 17.45)]), "LineString"),
    (shapely.MultiPoint([(78.39, 17.44), (78.40, 17.45)]), "Point"),
    (shapely.GeometryCollection([SQ, Point(78.395, 17.446)]), "Point"),
])
def test_parse_aoi_rejects_points_and_lines(geom, kind):
    for obj in (mapping(geom), {"type": "Feature", "geometry": mapping(geom)}):
        err = _code(aoi.parse_aoi, obj)
        assert err.code == "invalid" and kind in str(err) and "polygon" in str(err)


def test_parse_aoi_repairs_invalid_rings():
    bow = Polygon([(78.39, 17.44), (78.40, 17.45), (78.40, 17.44), (78.39, 17.45)])
    assert not bow.is_valid
    g = aoi.parse_aoi(mapping(bow))
    assert g.is_valid and g.geom_type == "MultiPolygon"  # both lobes kept
    assert g.area == pytest.approx(0.5 * 0.01 * 0.01)
    spike = Polygon([(78.39, 17.44), (78.40, 17.44), (78.40, 17.45), (78.395, 17.45),
                     (78.395, 17.46), (78.395, 17.45), (78.39, 17.45)])
    g = aoi.parse_aoi(spike)
    assert g.is_valid and g.geom_type == "Polygon" and g.area == pytest.approx(1e-4)


def test_parse_aoi_drops_z_and_orients_rings():
    cw = [(78.39, 17.445, 512.0), (78.39, 17.447, 512.0), (78.392, 17.447, 0.0),
          (78.392, 17.445, 0.0), (78.39, 17.445, 512.0)]
    g = aoi.parse_aoi({"type": "Polygon", "coordinates": [cw]})
    assert not g.has_z and g.exterior.is_ccw
    hole = [(78.3905, 17.4455), (78.3915, 17.4455), (78.3915, 17.4465),
            (78.3905, 17.4465)]
    holed = Polygon([c[:2] for c in cw], holes=[hole])
    g = aoi.parse_aoi(holed)
    assert g.exterior.is_ccw and not g.interiors[0].is_ccw  # RFC 7946 orientation
    outer = aoi.geodesic_area_m2(Polygon(holed.exterior))
    inner = aoi.geodesic_area_m2(Polygon(hole))
    assert aoi.geodesic_area_m2(g) == pytest.approx(outer - inner)


@pytest.mark.parametrize("obj", [
    None, 42, "not json", "[1, 2", {}, {"type": "FeatureCollection", "features": []},
    {"type": "Feature", "geometry": None}, {"type": "Polygon", "coordinates": []},
    {"type": "Polygon",
     "coordinates": [[[78.39, 17.44], [78.40, 17.45], [78.39, 17.44]]]},
    Polygon([(78.39, 17.44), (78.40, 17.44), (78.41, 17.44)]),  # zero area
    {"type": "Circle", "coordinates": [78.39, 17.44]},
    [78.39, 17.44, 78.40], [78.40, 17.44, 78.39, 17.45], [78.39, 17.45, 78.40, 17.44],
    ["w", "s", "e", "n"], [78.39, float("nan"), 78.40, 17.45],
    box(8_726_000, 1_970_000, 8_727_000, 1_971_000),  # EPSG:3857 metres, not lon/lat
    box(78.39, 85.5, 78.40, 86.0),
])
def test_parse_aoi_rejects_unusable_input(obj):
    assert _code(aoi.parse_aoi, obj).code == "invalid"


def test_parse_aoi_detects_swapped_coordinates():
    err = _code(aoi.parse_aoi, [17.445, 78.39, 17.447, 78.392])  # [s, w, n, e]
    assert err.code == "swapped"
    assert err.hint == {"bbox_wsen": [78.39, 17.445, 78.392, 17.447]}
    flipped = shapely.transform(_rect(17.4482, 78.3915, 120, 80, 30),
                                lambda c: c[:, ::-1].copy())
    assert _code(aoi.parse_aoi, mapping(flipped)).code == "swapped"
    # far from Hyderabad either way round: a valid (outside-coverage) AOI
    assert aoi.parse_aoi(box(77.20, 28.61, 77.21, 28.62)).is_valid


def test_parse_aoi_limits_the_area(monkeypatch):
    err = _code(aoi.parse_aoi, _rect(17.44, 78.40, 2500, 2000))  # 5 km2 > 4 km2
    assert err.code == "too_large" and "URBANSEG_MAX_AOI_KM2" in str(err)
    assert err.hint["area_km2"] == pytest.approx(5.0, rel=1e-3)
    monkeypatch.setattr(aoi, "MAX_AOI_KM2", 0.5)
    assert _code(aoi.parse_aoi, _rect(17.44, 78.40, 800, 700)).code == "too_large"
    assert aoi.parse_aoi(_rect(17.44, 78.40, 700, 700)).is_valid


def test_parse_aoi_rejects_non_finite_vertices():
    """A NaN vertex was dropped silently (halving this AOI); inf read as "empty"."""
    ring = "[[78.39, 17.44], [78.40, 17.44], [{}, 17.45], [78.39, 17.45], [78.39, 17.44]]"
    meta = aoi.grid_meta(17.445, 78.395, 1, 1)
    for bad in ("NaN", "Infinity", "-Infinity"):
        text = '{"type": "Polygon", "coordinates": [' + ring.format(bad) + "]}"
        # Python's json, and so Starlette's request.json(), accepts these literals
        for obj in (text, json.loads(text), shape(json.loads(text))):
            err = _code(aoi.parse_aoi, obj)
            assert err.code == "invalid" and "finite" in str(err), str(err)
            assert "finite" in str(_code(aoi.aoi_mask, obj, meta, (1200, 1280)))


def test_parse_aoi_checks_the_range_of_holes_too():
    """Bounds cover exteriors only: a hole at lat 1000 was "repaired" into a
    second part up there (NaN area, 4 calls), one at -1e308 into a new shell."""
    shell = list(box(78.39, 17.44, 78.40, 17.45).exterior.coords)
    for hole in ([(78.391, 1000.0), (78.392, 1000.0), (78.392, 1001.0)],
                 [(78.391, 17.441), (78.392, -1e308), (78.392, 17.442)],
                 [(500.0, 17.441), (501.0, 17.441), (501.0, 17.442)]):
        g = Polygon(shell, [hole])
        assert g.bounds == (78.39, 17.44, 78.40, 17.45)
        for fn in (aoi.parse_aoi, aoi.plan):
            err = _code(fn, g)
            assert err.code == "invalid" and "WGS84" in str(err), str(err)
        err = _code(aoi.parse_aoi, MultiPolygon([g, affinity.translate(SQ, 0.02)]))
        assert "WGS84" in str(err)


def test_parse_aoi_rejects_an_aoi_under_one_square_metre():
    """A double-click polygon was planned (and billed) but masked no pixel."""
    tiny = box(78.39, 17.44, 78.39 + 1e-6, 17.44 + 1e-6)  # 0.1 x 0.1 m
    for fn in (aoi.parse_aoi, aoi.plan):
        err = _code(fn, tiny)
        assert err.code == "invalid" and "at least 1 m2" in str(err), str(err)
    assert aoi.parse_aoi(_rect(17.4482, 78.3915, 1.5, 1.5)).is_valid  # 2.25 m2


def _star(n, seed=3):
    """A valid n-vertex star-shaped ring: the worst case of GEOS validation."""
    rng = np.random.default_rng(seed)
    a = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = 0.002 * rng.uniform(0.5, 1.0, n)
    return Polygon(np.column_stack([78.39 + r * np.cos(a), 17.44 + r * np.sin(a)]))


def test_parse_aoi_limits_vertices_and_parts(monkeypatch):
    """Validating a 100k-vertex ring took 3.6 s, once in each of plan, aoi_mask
    and clip_geojson; now input over the caps is refused before validation."""
    cap = aoi.MAX_AOI_VERTICES  # coordinates, each ring's closing one included
    meta = aoi.grid_meta(17.44, 78.39, 1, 1)
    big = _star(cap)  # cap + 1 coordinates
    empty_fc = {"type": "FeatureCollection", "features": []}
    for fn, args in ((aoi.parse_aoi, (mapping(big),)), (aoi.plan, (big,)),
                     (aoi.aoi_mask, (big, meta, (1200, 1280))),
                     (aoi.clip_geojson, (empty_fc, big))):
        err = _code(fn, *args)
        assert err.code == "invalid" and "URBANSEG_MAX_AOI_VERTICES" in str(err), str(err)
    assert aoi.parse_aoi(_star(cap - 1)).is_valid
    monkeypatch.setattr(aoi, "MAX_AOI_PARTS", 3)
    squares = [affinity.translate(SQ, 0.003 * i) for i in range(4)]
    fc = {"type": "FeatureCollection",
          "features": [{"type": "Feature", "geometry": mapping(g)} for g in squares]}
    # a FeatureCollection is refused on its feature count, before any geometry is built
    for obj, what in ((fc, "4 features"), (MultiPolygon(squares), "more than 3 parts")):
        err = _code(aoi.parse_aoi, obj)
        assert err.code == "invalid" and what in str(err), str(err)
        assert "URBANSEG_MAX_AOI_PARTS" in str(err)
    monkeypatch.setattr(aoi, "MAX_AOI_PARTS", 4)
    assert len(aoi.parse_aoi(fc).geoms) == 4


def _nested(kind, depth):
    obj = mapping(SQ)
    for _ in range(depth):
        obj = ({"type": kind, "features": [obj]} if kind == "FeatureCollection"
               else {"type": kind, "geometries": [obj]})
    return obj


@pytest.mark.parametrize("call", [
    lambda: aoi.parse_aoi("[" * 100_000 + "]" * 100_000),
    lambda: aoi.parse_aoi(_nested("FeatureCollection", 5000)),
    lambda: aoi.parse_aoi(_nested("GeometryCollection", 5000)),
    lambda: aoi.parse_aoi(_nested("FeatureCollection", 2)),  # RFC 7946: Features only
    lambda: aoi.parse_aoi({"type": "Feature",
                           "geometry": {"type": "Feature", "geometry": mapping(SQ)}}),
    lambda: aoi.parse_aoi([10 ** 400, 17.44, 78.40, 17.45]),
    lambda: aoi.plan(SQ, max_tiles="abc"),
    lambda: aoi.plan(SQ, max_tiles=float("inf")),
    lambda: aoi.plan(SQ, max_tiles=0),
    lambda: aoi.plan(SQ, max_tiles=True),
    lambda: aoi.plan(SQ, margin_m=1e308),
    lambda: aoi.plan(SQ, margin_m=10 ** 400),
    lambda: aoi.plan(SQ, margin_m=5000),
    lambda: aoi.plan(lat=17.44, lng=78.39, grid=10 ** 400),
    lambda: aoi.plan(lat=17.44, lng=78.39, grid=-10 ** 5000),
    lambda: aoi.plan(lat=10 ** 400, lng=78.39),
    lambda: aoi.parse_latlng("x" * 4000),
], ids=["deep-json", "deep-fc", "deep-gc", "fc-in-fc", "feature-in-feature",
        "huge-int-bbox", "max_tiles-text", "max_tiles-inf", "max_tiles-0",
        "max_tiles-bool", "margin-overflow", "margin-huge-int", "margin-5km",
        "grid-huge", "grid-unprintable", "lat-huge-int", "long-junk-text"])
def test_bad_input_raises_aoi_error_not_a_crash(call):
    """These raised RecursionError, ValueError or OverflowError (HTTP 500s)
    or were accepted; messages stay short whatever the input."""
    err = _code(call)
    assert err.code == "invalid" and len(str(err)) < 400, str(err)


def test_aoi_error_survives_pickling():
    """A process-pool job queue gets .code and .hint back intact."""
    import pickle

    err = aoi.AOIError("swap them", code="swapped", hint={"lat": 17.4, "lng": 78.4})
    back = pickle.loads(pickle.dumps(err))
    assert (str(back), back.code, back.hint) == ("swap them", "swapped",
                                                 {"lat": 17.4, "lng": 78.4})


def test_parse_aoi_keeps_all_the_area_a_self_overlapping_ring_encloses():
    """A pentagram drawn as one ring keeps its centre ("structure" repair); an
    even-odd ("linework") repair would cut the central pentagon out (-31%)."""
    lng, lat, big_r = 78.3915, 17.4482, 0.001

    def at(r, deg):
        a = math.radians(deg)
        return lng + r * math.cos(a), lat + r * math.sin(a)

    star = Polygon([at(big_r, 90 + 144 * k) for k in range(5)])
    small_r = big_r * math.cos(math.radians(72)) / math.cos(math.radians(36))
    outline = Polygon([at(small_r if k % 2 else big_r, 90 + 36 * k) for k in range(10)])
    assert not star.is_valid
    g = aoi.parse_aoi(star)
    assert g.geom_type == "Polygon" and g.area == pytest.approx(outline.area, rel=1e-9)


def test_geodesic_area_of_a_known_square():
    square = _rect(17.4482, 78.3915, 100, 100, 17)
    assert aoi.geodesic_area_m2(square) == pytest.approx(1e4, rel=1e-6)


def test_geodesic_area_does_not_depend_on_ring_orientation():
    outer, hole = _rect(17.4482, 78.3915, 200, 200), _rect(17.4482, 78.3915, 100, 50)
    ring_o, ring_h = list(outer.exterior.coords), list(hole.exterior.coords)
    for o in (ring_o, ring_o[::-1]):
        for h in (ring_h, ring_h[::-1]):  # GEOS keeps whatever orientation it is given
            area = aoi.geodesic_area_m2(Polygon(o, [h]))
            assert area == pytest.approx(35_000, rel=1e-6)
    a, b = _rect(17.4482, 78.3915, 100, 100), _rect(17.4482, 78.3945, 80, 60)
    mixed = MultiPolygon([Polygon(a.exterior.coords), Polygon(b.exterior.coords[::-1])])
    assert aoi.geodesic_area_m2(mixed) == pytest.approx(14_800, rel=1e-6)


# ---------------------------------------------------------------- grid_meta

def test_grid_meta_equals_fetch_area_meta(monkeypatch, fake_key):
    calls = []

    def blank_tile(lat, lng, *args, **kwargs):
        calls.append((lat, lng))
        return Image.new("RGB", (config.TILE_MAP_PX * config.SCALE,) * 2)

    monkeypatch.setattr(imagery, "_fetch_raw_tile", blank_tile)
    grids = [(17.4435, 78.3772, 1, 1), (17.4482, 78.3915, 2, 2), (17.39, 78.48, 3, 1),
             (17.5453, 78.4854, 2, 3), (-33.8688, 151.2093, 1, 2)]
    s = config.SCALE
    for lat, lng, nx, ny in grids:
        img, meta = imagery.fetch_area(lat, lng, nx, ny, sleep=0)
        assert aoi.grid_meta(lat, lng, nx, ny) == meta
        assert img.size == (nx * config.USABLE_X * s, ny * config.USABLE_Y * s)
    n = sum(nx * ny for *_, nx, ny in grids)
    assert len(calls) == n
    # what the API will fetch for a plan is exactly what the plan priced
    p = aoi.plan(_rect(17.4482, 78.3915, 500, 150, 10))
    _, meta = imagery.fetch_area(*p["center"], p["nx"], p["ny"], sleep=0)
    assert meta == p["meta"] and len(calls) == n + p["calls"]


@pytest.mark.data
def test_grid_meta_reproduces_every_training_grid():
    for name, lat, lng, nx, ny, _split in config.AREAS:
        assert aoi.grid_meta(lat, lng, nx, ny) == _area_meta(name), name


# ---------------------------------------------------------------- plan

@pytest.mark.parametrize("grid, extent_m, extent_ha", [  # WGS84 ground metres
    (1, [364.7, 339.8], 12.39), (2, [729.4, 679.7], 49.58),
    (3, [1094.1, 1019.5], 111.55)])
def test_plan_for_a_point_is_the_classic_square(grid, extent_m, extent_ha):
    lat, lng = 17.4482, 78.3915
    p = aoi.plan(lat=lat, lng=lng, grid=grid)
    assert (p["nx"], p["ny"], p["calls"]) == (grid, grid, grid * grid)
    assert p["center"] == [lat, lng] and p["meta"] == aoi.grid_meta(lat, lng, grid, grid)
    assert p["extent_m"] == extent_m and p["extent_ha"] == extent_ha
    assert p["bbox_wsen"] == list(imagery.bbox_latlng(p["meta"]))
    assert p["in_coverage"] and p["aoi_area_ha"] is None and p["aoi"] is None
    assert bool(p["warnings"]) == (grid * grid > aoi.WARN_TILES)


@pytest.mark.parametrize("kw", [
    {"lat": 17.4482, "lng": 78.3915, "grid": 1},
    {"lat": 17.4482, "lng": 78.3915, "grid": 3},
    {"aoi": _rect(17.31, 78.52, 500, 300, 15)},
    {"lat": -33.8688, "lng": 151.2093, "grid": 2},
    {"lat": 60.17, "lng": 24.94, "grid": 1}])
def test_plan_extent_is_in_wgs84_ground_metres(kw):
    """The extent was in nominal Web-Mercator metres, 0.53% more area than the
    WGS84 ground that area_stats counts."""
    p = aoi.plan(**kw)
    w, s, e, n = p["bbox_wsen"]
    lat, lng = p["center"]
    assert p["extent_m"][0] == pytest.approx(GEOD.inv(w, lat, e, lat)[2], abs=0.051)
    assert p["extent_m"][1] == pytest.approx(GEOD.inv(lng, s, lng, n)[2], abs=0.051)
    area_ha = aoi.geodesic_area_m2(box(w, s, e, n)) / 1e4
    assert p["extent_ha"] == pytest.approx(area_ha, abs=0.0051)


def test_plan_extent_equals_area_stats_total():
    from urbanseg import infer

    meta = aoi.grid_meta(17.4482, 78.3915, 1, 1)
    zeros = np.zeros(_shape_of(meta), np.uint8)
    nominal = zeros.size * (imagery.meters_per_map_px(17.4482) / config.SCALE) ** 2
    if infer.area_stats(zeros, meta)["other"]["area_m2"] == pytest.approx(nominal):
        pytest.skip("infer.area_stats still counts nominal Web-Mercator m2 "
                    "(package predict)")
    for grid in (1, 2, 3):
        p = aoi.plan(lat=17.4482, lng=78.3915, grid=grid)
        st = infer.area_stats(np.zeros(_shape_of(p["meta"]), np.uint8), p["meta"])
        total_ha = sum(v["area_m2"] for v in st.values()) / 1e4
        assert total_ha == pytest.approx(p["extent_ha"], abs=0.0051)


def test_plan_defaults_to_grid_2_like_segment_latlng():
    p = aoi.plan(lat=17.4482, lng=78.3915)
    assert p["meta"] == aoi.grid_meta(17.4482, 78.3915, 2, 2)


def test_one_hectare_parcel_needs_one_call_instead_of_four():
    lat, lng = 17.4482, 78.3915
    parcel = aoi.plan(_rect(lat, lng, 100, 100))
    medium = aoi.plan(lat=lat, lng=lng, grid=2)  # what the GUI fetched by default
    assert parcel["aoi_area_ha"] == pytest.approx(1.0, abs=1e-4)
    assert (parcel["calls"], parcel["extent_ha"]) == (1, 12.39)
    assert (medium["calls"], medium["extent_ha"]) == (4, 49.58)
    # any 1 ha plot, aspect 1:2 to 2:1 and any rotation, fits in a single tile
    rng = np.random.default_rng(1)
    for _ in range(200):
        asp = math.exp(rng.uniform(math.log(0.5), math.log(2)))
        w = math.sqrt(1e4 * asp)
        g = _rect(rng.uniform(17.15, 17.65), rng.uniform(78.15, 78.75), w, 1e4 / w,
                  rng.uniform(0, 180))
        assert aoi.plan(g)["calls"] == 1


def test_plan_sizes_nx_and_ny_independently():
    wide = aoi.plan(_rect(17.4482, 78.3915, 600, 120))
    tall = aoi.plan(_rect(17.4482, 78.3915, 100, 520))
    assert (wide["nx"], wide["ny"]) == (2, 1) and (tall["nx"], tall["ny"]) == (1, 2)
    assert wide["extent_m"] == [729.4, 339.8]


def test_plan_refuses_more_tiles_than_allowed(monkeypatch):
    big = _rect(17.4482, 78.3915, 1200, 1200)  # 4 x 4 tiles with the margin
    err = _code(aoi.plan, big)
    assert err.code == "too_large" and "split" in str(err)
    assert "URBANSEG_MAX_TILES" in str(err)
    assert err.hint == {"nx": 4, "ny": 4, "calls": 16, "max_tiles": 9}
    assert aoi.plan(big, max_tiles=16)["calls"] == 16
    assert _code(aoi.plan, lat=17.4482, lng=78.3915, grid=4).code == "too_large"
    monkeypatch.setattr(aoi, "MAX_TILES", 4)
    assert aoi.plan(lat=17.4482, lng=78.3915, grid=2)["calls"] == 4
    three_by_two = _rect(17.4482, 78.3915, 900, 400)
    assert _code(aoi.plan, three_by_two).code == "too_large"


def test_plan_coverage_and_warnings():
    outside = aoi.plan(lat=28.6139, lng=77.2090, grid=1)  # Delhi
    assert not outside["in_coverage"]
    assert "outside greater Hyderabad" in outside["warnings"][0]
    err = _code(aoi.plan, lat=28.6139, lng=77.2090, allow_outside=False)
    assert err.code == "outside_coverage"
    assert err.hint == {"coverage_bbox_wsen": list(config.HYD_BBOX)}
    north = config.HYD_BBOX[3]
    edge = aoi.plan(_rect(north, 78.40, 200, 100))  # straddles the northern limit
    assert not edge["in_coverage"] and "partly outside" in edge["warnings"][0]
    assert aoi.plan(_rect(17.4482, 78.3915, 100, 100))["warnings"] == []
    large = aoi.plan(lat=17.4482, lng=78.3915, grid=3)
    assert "9 Static Maps calls" in large["warnings"][0]
    # a margin below the model's weak outer band is allowed but flagged
    tight = aoi.plan(_rect(17.4482, 78.3915, 360, 100), margin_m=0)
    assert tight["calls"] == 1 and "image edge" in tight["warnings"][0]


@pytest.mark.parametrize("kw", [
    {"aoi": SQ, "lat": 17.44, "lng": 78.39}, {"aoi": SQ, "grid": 2}, {}, {"lat": 17.44},
    {"aoi": SQ, "margin_m": -5}, {"aoi": SQ, "margin_m": "wide"},
    {"lat": 17.44, "lng": 78.39, "grid": 0}, {"lat": 17.44, "lng": 78.39, "grid": 2.5},
    {"lat": 17.44, "lng": 78.39, "grid": "2"}, {"lat": "x", "lng": 78.39},
    {"lat": 95.0, "lng": 78.39}, {"lat": 17.44, "lng": 181.0},
])
def test_plan_rejects_bad_arguments(kw):
    assert _code(aoi.plan, **kw).code == "invalid"


def test_plan_does_not_add_a_tile_for_an_exact_fit():
    """An AOI exactly one tile wide and tall (margin 0) is one call even when
    float noise puts its span a hair over the tile."""
    lat, lng = 17.4482, 78.3915
    xw, yn = imagery.latlng_to_world(lat, lng)
    e = lng + config.USABLE_X * 360 / imagery.WORLD(config.ZOOM)
    while imagery.latlng_to_world(lat, e)[0] - xw < config.USABLE_X:
        e = float(np.nextafter(e, np.inf))
    s = imagery.world_to_latlng(xw, yn + config.USABLE_Y)[0]
    while imagery.latlng_to_world(s, lng)[1] - yn < config.USABLE_Y:
        s = float(np.nextafter(s, -np.inf))
    p = aoi.plan(box(lng, s, e, lat), margin_m=0)
    assert (p["nx"], p["ny"], p["calls"]) == (1, 1, 1)


def test_plan_rejects_a_swapped_point():
    err = _code(aoi.plan, lat=78.3915, lng=17.4482)
    assert err.code == "swapped" and err.hint == {"lat": 17.4482, "lng": 78.3915}


def test_plan_is_json_ready():
    keys = {"nx", "ny", "calls", "center", "extent_m", "extent_ha", "bbox_wsen",
            "aoi_area_ha", "in_coverage", "warnings", "meta"}
    for p in (aoi.plan(_rect(17.4482, 78.3915, 300, 200, 20)),
              aoi.plan(lat=17.4482, lng=78.3915)):
        back = json.loads(json.dumps(p))
        assert set(back) >= keys and back["meta"] == p["meta"]
    assert shape(json.loads(json.dumps(aoi.plan(SQ)["aoi"]))).equals(SQ)


def test_plan_covers_the_aoi_plus_margin_property():
    rng = np.random.default_rng(20261020)
    for _ in range(200):
        g = _random_aoi(rng, 0.05, 60)
        margin = float(rng.uniform(0, 60))
        p = aoi.plan(g, margin_m=margin, max_tiles=1000)
        meta, (nx, ny) = p["meta"], (p["nx"], p["ny"])
        assert p["calls"] == nx * ny and meta == aoi.grid_meta(*p["center"], nx, ny)
        w, s, e, n = g.bounds
        xw, yn = imagery.latlng_to_world(n, w)
        xe, ys = imagery.latlng_to_world(s, e)
        mx, my = _ground_mpp(p["center"][0])
        pad_x, pad_y = margin / mx, margin / my  # map px
        x1, y1 = meta["x0"] + meta["width_map_px"], meta["y0"] + meta["height_map_px"]
        # covers bbox + margin on every side, centred on the bbox ...
        assert meta["x0"] <= xw - pad_x + 1e-6 and x1 >= xe + pad_x - 1e-6
        assert meta["y0"] <= yn - pad_y + 1e-6 and y1 >= ys + pad_y - 1e-6
        assert (meta["x0"] + x1) / 2 == pytest.approx((xw + xe) / 2, abs=1e-6)
        assert (meta["y0"] + y1) / 2 == pytest.approx((yn + ys) / 2, abs=1e-6)
        # ... with the fewest tiles in each direction
        assert nx == 1 or (nx - 1) * config.USABLE_X < xe - xw + 2 * pad_x
        assert ny == 1 or (ny - 1) * config.USABLE_Y < ys - yn + 2 * pad_y
        # and the margin is margin_m WGS84 metres on the ground (the scale
        # drifts by <0.1% between the centre and the edge of a large plan)
        bw, bs, be, bn = p["bbox_wsen"]
        lat_c, lng_c = (s + n) / 2, (w + e) / 2
        for d in (GEOD.inv(bw, lat_c, w, lat_c)[2], GEOD.inv(e, lat_c, be, lat_c)[2],
                  GEOD.inv(lng_c, s, lng_c, bs)[2], GEOD.inv(lng_c, bn, lng_c, n)[2]):
            assert d >= 0.999 * margin - 1e-3


# ---------------------------------------------------------------- aoi_mask

def test_aoi_mask_matches_an_analytic_rectangle():
    meta = aoi.grid_meta(17.4482, 78.3915, 2, 1)
    H, W = _shape_of(meta)
    to_ll = _to_ll(meta)
    for x0, x1, y0, y1 in [(100.3, 400.7, 50.2, 300.9), (0.0, 1280.0, 10.4, 11.6),
                           (-50.0, 37.49, 1100.51, 1300.0),
                           (2000.25, 2559.75, 0.75, 1199.25)]:
        poly = Polygon([to_ll(x0, y0), to_ll(x1, y0), to_ll(x1, y1), to_ll(x0, y1)])
        m = aoi.aoi_mask(poly, meta, (H, W))
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        # pixel-centre rule: px i is inside when x0 <= i + 0.5 < x1
        want_c = [max(math.ceil(x0 - 0.5), 0), min(math.ceil(x1 - 0.5), W) - 1]
        want_r = [max(math.ceil(y0 - 0.5), 0), min(math.ceil(y1 - 0.5), H) - 1]
        for got, want in ((cols, want_c), (rows, want_r)):
            assert abs(got[0] - want[0]) <= 1 and abs(got[-1] - want[1]) <= 1
        assert [cols[0], cols[-1]] == want_c and [rows[0], rows[-1]] == want_r
        assert m.sum() == len(rows) * len(cols)  # a filled rectangle


def test_aoi_mask_holes_parts_and_inputs():
    meta = aoi.grid_meta(17.4482, 78.3915, 1, 1)
    H, W = _shape_of(meta)
    to_ll = _to_ll(meta)

    def sq(a, b):
        return [to_ll(a, a), to_ll(b, a), to_ll(b, b), to_ll(a, b)]

    g = MultiPolygon([Polygon(sq(100, 500), [sq(200, 300)]), Polygon(sq(900, 1000))])
    m = aoi.aoi_mask(g, meta, (H, W, 3))
    assert m.shape == (H, W) and m.dtype == bool
    assert m.sum() == 400 * 400 - 100 * 100 + 100 * 100
    assert not m[200:300, 200:300].any() and m[900:1000, 900:1000].all()
    assert (aoi.aoi_mask(mapping(g), meta, (H, W)) == m).all()  # GeoJSON works too
    off_image = box(78.50, 17.50, 78.51, 17.51)
    assert not aoi.aoi_mask(off_image, meta, (H, W)).any()


def test_aoi_mask_and_clip_geojson_check_shapely_input_too():
    """Shapely AOIs skipped parse_aoi's repair: clipping by a bow-tie raised
    GEOSException, and a line or point AOI gave a mask of line pixels."""
    meta = aoi.grid_meta(17.4482, 78.3915, 1, 1)
    hw = _shape_of(meta)
    bow = Polygon([(78.39, 17.447), (78.393, 17.45), (78.393, 17.447), (78.39, 17.45)])
    fc = {"type": "FeatureCollection",
          "features": [_feature(box(78.391, 17.446, 78.392, 17.4485))]}
    out = aoi.clip_geojson(fc, bow)
    assert len(out["features"]) == 1 and out == aoi.clip_geojson(fc, mapping(bow))
    m = aoi.aoi_mask(bow, meta, hw)
    assert m.any() and (m == aoi.aoi_mask(mapping(bow), meta, hw)).all()
    for g in (LineString([(78.39, 17.447), (78.393, 17.45)]), Point(78.3915, 17.4482),
              Polygon()):
        assert _code(aoi.aoi_mask, g, meta, hw).code == "invalid"
        assert _code(aoi.clip_geojson, fc, g).code == "invalid"


def test_aoi_mask_area_matches_the_geodesic_area_property():
    rng = np.random.default_rng(7)
    errs = []
    for _ in range(60):
        g = _random_aoi(rng, 0.5, 30)
        meta = aoi.plan(g, max_tiles=100)["meta"]
        m = aoi.aoi_mask(g, meta, _shape_of(meta))
        errs.append(m.sum() * _wgs84_px_area(meta) / aoi.geodesic_area_m2(g) - 1)
    errs = np.abs(errs)
    assert errs.max() < 0.01 and errs.mean() < 0.002, (errs.max(), errs.mean())


# ---------------------------------------------------------------- clip_geojson

def _feature(geom, klass="residential", **extra):
    return {"type": "Feature", "properties": {"class": klass, "area_m2": -1.0},
            "geometry": mapping(geom), **extra}


def test_clip_geojson_cuts_drops_and_recomputes_area():
    clip = _rect(17.4482, 78.3915, 200, 200)
    w, s, e, n = clip.bounds
    dx, dy = (e - w) / 10, (n - s) / 10
    inside = box(w + dx, s + dy, w + 2 * dx, s + 2 * dy)
    straddling = box(e - dx, s + dy, e + dx, s + 2 * dy)
    outside = box(e + dx, s, e + 2 * dx, s + dy)
    bow = Polygon([(w + 4 * dx, s + 4 * dy), (w + 6 * dx, s + 6 * dy),
                   (w + 6 * dx, s + 4 * dy), (w + 4 * dx, s + 6 * dy)])
    fc = {"type": "FeatureCollection", "features": [
        _feature(inside, id=7, bbox=[0, 0, 0, 0]), _feature(straddling, "commercial"),
        _feature(outside), _feature(bow, "free_land"), _feature(Point(w + dx, s + dx)),
        {"type": "Feature", "properties": {}, "geometry": None}]}
    before = json.dumps(fc)
    out = aoi.clip_geojson(fc, clip)
    assert json.dumps(fc) == before  # input untouched
    assert out["type"] == "FeatureCollection"
    classes = [f["properties"]["class"] for f in out["features"]]
    assert classes == ["residential", "commercial", "free_land"]
    f_in, f_cut, f_bow = out["features"]
    assert f_in["id"] == 7 and "bbox" not in f_in
    assert f_in["geometry"] == mapping(inside)  # entirely inside: coordinates kept
    for f in out["features"]:
        g = shape(f["geometry"])
        assert f["properties"]["area_m2"] == round(aoi.geodesic_area_m2(g), 1) > 0
        assert clip.buffer(1e-9).covers(g)
    assert f_cut["properties"]["area_m2"] == pytest.approx(
        aoi.geodesic_area_m2(straddling.intersection(clip)), abs=0.1)
    assert shape(f_bow["geometry"]).is_valid


def test_clip_geojson_never_returns_area_outside_the_aoi_property():
    rng = np.random.default_rng(11)
    for _ in range(40):
        g = _random_aoi(rng, 0.5, 20)
        (w, s, e, n), d = g.bounds, 0.0002  # ~20 m
        # building-like squares on a jittered grid around and across the AOI
        feats = []
        for x in np.arange(w - 3 * d, e + 3 * d, 1.5 * d):
            for y in np.arange(s - 3 * d, n + 3 * d, 1.5 * d):
                jx, jy = rng.uniform(0, 0.4 * d, 2)
                sq = box(x + jx, y + jy, x + jx + d, y + jy + d)
                feats.append(_feature(affinity.rotate(sq, rng.uniform(0, 90))))
        out = aoi.clip_geojson({"type": "FeatureCollection", "features": feats}, g)
        assert out["features"]
        tol = 0.1 * len(out["features"])  # area_m2 is rounded to 0.1 m2
        total = 0.0
        for f in out["features"]:
            cg = shape(f["geometry"])
            assert g.buffer(1e-9).covers(cg)
            assert f["properties"]["area_m2"] == round(aoi.geodesic_area_m2(cg), 1)
            total += f["properties"]["area_m2"]
        union = shapely.union_all([shape(f["geometry"]) for f in feats])
        assert total <= aoi.geodesic_area_m2(g) + tol
        # the rotated squares may overlap, so the sum can exceed their union
        assert total >= aoi.geodesic_area_m2(union.intersection(g)) - tol


@pytest.mark.data
def test_clip_geojson_agrees_with_the_aoi_mask_on_real_vectors():
    from urbanseg import infer

    cm, meta = _classmap("lbnagar")
    s = meta["scale"]
    y0, x0 = 1200, 1500
    crop, sub = cm[y0:y0 + 1200, x0:x0 + 1200], dict(meta)
    sub.update(x0=meta["x0"] + x0 / s, y0=meta["y0"] + y0 / s,
               width_map_px=600, height_map_px=600)
    sub["center"] = list(imagery.img_px_to_latlng(600, 600, sub))
    fc = infer.to_geojson(crop, sub, min_area_px=0, simplify_m=0)
    lat, lng = sub["center"]
    ring = [(-120, -90), (110, -110), (140, 60), (-30, 140), (-130, 40)]  # metres
    g = _lnglat(Polygon(ring), lat, lng)
    out = aoi.clip_geojson(fc, g)
    m = aoi.aoi_mask(g, sub, crop.shape)
    px = _wgs84_px_area(sub)
    a_aoi = aoi.geodesic_area_m2(g)
    for k, name in enumerate(config.CLASS_NAMES):
        feats = [f for f in out["features"] if f["properties"]["class"] == name]
        if not feats:
            continue
        vec = sum(f["properties"]["area_m2"] for f in feats)
        assert all(g.buffer(1e-9).covers(shape(f["geometry"])) for f in feats)
        assert abs(vec - (crop[m] == k).sum() * px) < 0.02 * a_aoi, name


# -------------------------------------------- AOI stats vs the diagnosis

def _parcel_mask(cm, meta, x0, y0, pw, ph):
    to_ll = _to_ll(meta)
    poly = Polygon([to_ll(x0, y0), to_ll(x0 + pw, y0), to_ll(x0 + pw, y0 + ph),
                    to_ll(x0, y0 + ph)])
    return aoi.aoi_mask(poly, meta, cm.shape)


def _parcels(cm, meta, seed=20261020):
    """s_checks.py's parcels: 1-10 ha rectangles, aspect 1:2-2:1, in image px."""
    rng = np.random.default_rng(seed)
    px_area = (imagery.meters_per_map_px(meta["center"][0]) / meta["scale"]) ** 2
    H, W = cm.shape
    for ha in (1, 2, 5, 10):
        for _ in range(5):
            asp = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
            pw = int(round(math.sqrt(ha * 1e4 / px_area * asp)))
            ph = int(round(ha * 1e4 / px_area / pw))
            cx = int(rng.integers(1280, W - 1280 + 1))
            cy = int(rng.integers(1200, H - 1200 + 1))
            yield cx - pw // 2, cy - ph // 2, pw, ph


@pytest.mark.data
@pytest.mark.parametrize("area", ["lbnagar", "kompally"])
def test_aoi_stats_equal_the_parcel_method(area):
    cm, meta = _classmap(area)
    for x0, y0, pw, ph in _parcels(cm, meta):
        m = _parcel_mask(cm, meta, x0, y0, pw, ph)
        parcel = cm[y0:y0 + ph, x0:x0 + pw]
        assert m.sum() == parcel.size and m[y0:y0 + ph, x0:x0 + pw].all()
        np.testing.assert_array_equal(np.bincount(cm[m], minlength=4),
                                      np.bincount(parcel.ravel(), minlength=4))


@pytest.mark.data
def test_area_stats_with_an_aoi_mask_equals_the_parcel_method():
    from urbanseg import infer

    if "mask" not in inspect.signature(infer.area_stats).parameters:
        pytest.skip("infer.area_stats has no mask= yet (package predict)")
    cm, meta = _classmap("lbnagar")
    for x0, y0, pw, ph in list(_parcels(cm, meta))[::4]:
        m = _parcel_mask(cm, meta, x0, y0, pw, ph)
        st = infer.area_stats(cm, meta, mask=m)
        counts = np.bincount(cm[y0:y0 + ph, x0:x0 + pw].ravel(), minlength=4)
        for k, name in enumerate(config.CLASS_NAMES):
            assert st[name]["pixels"] == counts[k]
            assert st[name]["share"] == pytest.approx(counts[k] / m.sum(), abs=1e-4)
        total = sum(v["area_m2"] for v in st.values())
        assert total == pytest.approx(pw * ph * _wgs84_px_area(meta), rel=0.01)


@pytest.mark.data
def test_madhapur_parcel_reproduces_the_diagnosis():
    pred = os.environ.get("URBANSEG_PRED_DIR")
    p = Path(pred or ".") / "madhapur_fused_osmonly.png"
    if not pred or not p.is_file():
        pytest.skip("set URBANSEG_PRED_DIR to the precomputed release predictions")
    cm, meta = np.asarray(Image.open(p)), _area_meta("madhapur")
    to_ll = _to_ll(meta)
    poly = Polygon([to_ll(x, y) for x, y in MADHAPUR_PARCEL_PX])
    m = aoi.aoi_mask(poly, meta, cm.shape)
    shares = np.round(np.bincount(cm[m], minlength=4) / m.sum() * 100, 1)
    assert shares.tolist() == [21.1, 47.0, 0.0, 31.9]  # the parcel itself
    cx, cy = (int(np.mean(v)) for v in zip(*MADHAPUR_PARCEL_PX))
    sq = cm[cy - 1200:cy + 1200, cx - 1280:cx + 1280]  # what the Medium square reported
    sq_shares = np.round(np.bincount(sq.ravel(), minlength=4) / sq.size * 100, 1)
    assert sq_shares.tolist() == [15.0, 57.6, 1.8, 25.6]
    p = aoi.plan(poly)
    assert p["calls"] == 1 and p["aoi_area_ha"] == pytest.approx(1.986, abs=1e-3)


# ---------------------------------------------------------------- area_units

def test_area_units():
    assert aoi.area_units(1e4) == {"m2": 10000.0, "ha": 1.0, "acres": 2.4711,
                                   "sq_yd": 11959.9, "guntas": 98.84}
    gunta = aoi.area_units(101.17141056)
    assert gunta["guntas"] == 1.0 and gunta["sq_yd"] == 121.0
    acre = aoi.area_units(4046.8564224)
    assert acre["acres"] == 1.0 and acre["guntas"] == 40.0 and acre["sq_yd"] == 4840.0
    zero = dict.fromkeys(("m2", "ha", "acres", "sq_yd", "guntas"), 0.0)
    assert aoi.area_units(0) == zero


# ---------------------------------------------------------------- parse_latlng

@pytest.mark.parametrize("text", [
    "17.4435, 78.3772", "17.4435 78.3772", "17.4435,78.3772", "17.4435;78.3772",
    " 17.4435 ,\t78.3772 ", "(17.4435, 78.3772)", "17.4435\n78.3772",
    "+17.4435, +78.3772", "17.4435, 78.3772",  # no-break space from web pages
    "17,4435 78,3772", "17,4435; 78,3772", "17,4435;78,3772", "17,4435, 78,3772",
    "17.4435° N, 78.3772° E", "17.4435N 78.3772E", "N17.4435 E78.3772",
    "17.4435°, 78.3772°",
])
def test_parse_latlng_decimal_degrees(text):
    assert aoi.parse_latlng(text) == (17.4435, 78.3772)


@pytest.mark.parametrize("text", [
    "17°26'36.6\"N 78°22'37.9\"E", "17°26'36.6\"N, 78°22'37.9\"E",
    "17° 26' 36.6\" N, 78° 22' 37.9\" E", "N 17° 26' 36.6\", E 78° 22' 37.9\"",
    "N17°26'36.6\" E78°22'37.9\"", "17°26′36.6″N 78°22′37.9″E",
    "17°26'36.6''N 78°22'37.9''E", "17º26’36.6”N 78º22’37.9”E",
    "17 26 36.6 N 78 22 37.9 E", "17°26'36,6\"N 78°22'37,9\"E",
    "17°26'36.6\"n 78°22'37.9\"e", "78°22'37.9\"E 17°26'36.6\"N",
    "17°26'36.6\" 78°22'37.9\"", "17°26'36.6\"N78°22'37.9\"E",
])
def test_parse_latlng_dms(text):
    assert aoi.parse_latlng(text) == pytest.approx(DMS, abs=1e-7)


def test_parse_latlng_signs_and_decimal_minutes():
    sydney = pytest.approx((-33.86881, 151.20931), abs=1e-5)
    hyd = pytest.approx((17.4435, 78.3772))
    assert aoi.parse_latlng("17°26.61'N 78°22.632'E") == hyd
    assert aoi.parse_latlng("N 17° 26.610 E 078° 22.632") == hyd  # GPS receivers
    assert aoi.parse_latlng("33°52'7.7\"S 151°12'33.5\"E") == sydney
    assert aoi.parse_latlng("-33°52'7.7\", 151°12'33.5\"") == sydney
    assert aoi.parse_latlng("40.7128, -74.0060") == (40.7128, -74.006)
    assert aoi.parse_latlng("40.7128, −74.0060") == (40.7128, -74.006)  # U+2212
    nyc = aoi.parse_latlng("40°42'46\"N 74°0'22\"W")
    assert nyc == pytest.approx((40.71278, -74.00611), abs=1e-5)


@pytest.mark.parametrize("url, want", [
    ("https://www.google.com/maps/@17.4435,78.3772,17z", (17.4435, 78.3772)),
    ("https://www.google.com/maps/@17.4435,78.3772,500m/data=!3m1!1e3",
     (17.4435, 78.3772)),
    ("https://maps.google.com/?q=17.4435,78.3772", (17.4435, 78.3772)),
    ("https://www.google.com/maps?q=17.4435%2C78.3772&z=18", (17.4435, 78.3772)),
    ("https://www.google.com/maps/search/?api=1&query=17.4435,78.3772",
     (17.4435, 78.3772)),
    ("https://maps.google.com/maps?ll=17.4435,78.3772&spn=0.01,0.01", (17.4435, 78.3772)),
    ("google.com/maps/@17.4435,78.3772,18z", (17.4435, 78.3772)),
    ("https://www.google.com/maps/search/17.4435,+78.3772?entry=tts", (17.4435, 78.3772)),
    # place links: the !3d!4d pin wins over the shifted /@ view centre
    ("https://www.google.com/maps/place/Cyber+Towers/@17.4504,78.3790,17z/data=!3m1"
     "!4b1!4m6!3m5!1s0x3bcb93d:0x1!8m2!3d17.4504!4d78.3809!16s%2Fg%2F1tf",
     (17.4504, 78.3809)),
    ("https://www.google.com/maps/place/17%C2%B026'36.6%22N+78%C2%B022'37.9%22E/"
     "@17.4435,78.3750,17z/", DMS),
    ("https://www.google.com/maps/place/17.4435,78.3772/@17.4435,78.3750,17z",
     (17.4435, 78.3772)),
    ("geo:17.4435,78.3772?z=18", (17.4435, 78.3772)),
])
def test_parse_latlng_google_maps_links(url, want):
    assert aoi.parse_latlng(url) == pytest.approx(want, abs=1e-7)


@pytest.mark.parametrize("url, want", [
    # directions with a single place: the place, never the /@ view centre
    ("https://www.google.com/maps/dir//17.4435,78.3772/@17.4350,78.3650,15z",
     (17.4435, 78.3772)),
    ("https://www.google.com/maps/dir//17%C2%B026'36.6%22N+78%C2%B022'37.9%22E/"
     "@17.4350,78.3650,15z/data=!4m2!4m1!3e0", DMS),
    ("https://www.google.com/maps/dir/?api=1&destination=17.4435,78.3772",
     (17.4435, 78.3772)),
    ("https://maps.google.com/maps?daddr=17.4435,78.3772&ll=17.40,78.40",
     (17.4435, 78.3772)),
    # q= is the pin, ll= only the legacy map centre, whichever comes first
    ("https://maps.google.com/maps?ll=17.40,78.40&q=17.4435,78.3772&z=14",
     (17.4435, 78.3772)),
    ("https://maps.google.com/maps?ll=17.40,78.40&q=17.4435,78.3772+(Cyber+Towers)",
     (17.4435, 78.3772)),
    ("geo:0,0?q=17.4435,78.3772(Cyber+Towers)", (17.4435, 78.3772)),
    ("https://www.google.com/maps/@?api=1&map_action=map&center=17.4435,78.3772&zoom=18",
     (17.4435, 78.3772)),
    # Apple Maps shares put the place in ll= and a label in q=
    ("https://maps.apple.com/?ll=17.4435,78.3772&q=Dropped%20Pin&t=m",
     (17.4435, 78.3772)),
])
def test_parse_latlng_links_resolve_to_the_place(url, want):
    assert aoi.parse_latlng(url) == pytest.approx(want, abs=1e-7)


@pytest.mark.parametrize("url, why", [
    # resolved to the /@ view centre, 1.6-3.5 km from any place named, and billed
    ("https://www.google.com/maps/dir/17.4000,78.4000/17.4435,78.3772/"
     "@17.4217,78.3886,14z/data=!3m1!4b1!4m2!4m1!3e0",
     "2 places (17.4, 78.4 to 17.4435, 78.3772)"),
    ("https://www.google.com/maps/dir/Ameerpet/Hitech+City/@17.44,78.41,13z/"
     "data=!3m1!4b1!4m2!4m1!3e0", "2 places ('Ameerpet' to 'Hitech City')"),
    ("https://www.google.com/maps/dir//Cyber+Towers/@17.45,78.38,17z/data=!4m8!4m7!1m0"
     "!1m5!1m1!1s0x3bcb93dc8c5d69df:0x19688beb557fa0ee!2m2!1d78.3809!2d17.4504",
     "'Cyber Towers' but no coordinates"),
    ("https://www.google.com/maps/dir/?api=1&origin=17.40,78.40"
     "&destination=17.4435,78.3772", "2 places (17.4, 78.4 to 17.4435, 78.3772)"),
    ("https://maps.google.com/maps?saddr=My+Location&daddr=17.4435,78.3772",
     "2 places ('My Location' to 17.4435, 78.3772)"),
    ("https://www.google.com/maps/dir/", "names no place"),
    # a search or a place given by name only: the view centre is not the place
    ("https://www.google.com/maps/search/hitech+city/@17.4435,78.3772,15z",
     "'hitech city' but no coordinates"),
    ("https://www.google.com/maps/place/Cyber+Towers/@17.4504,78.3790,17z",
     "'Cyber Towers' but no coordinates"),
    ("https://maps.google.com/maps?q=Cyber+Towers&ll=17.4504,78.3790",
     "'Cyber Towers' but no coordinates"),
])
def test_parse_latlng_rejects_links_without_exactly_one_place(url, why):
    err = _code(aoi.parse_latlng, url)
    assert err.code == "invalid" and why in str(err), str(err)


@pytest.mark.parametrize("text, why", [
    ("", "enter a location"), ("   ", "enter a location"),
    ("17,4435", "ambiguous"), ("-17,4435", "ambiguous"),
    ("17,4435,78,3772", "could not read"), ("17.4435, 78,3772", "could not read"),
    ("17.4435;78,3772", "could not read"), ("17.4435", "could not read"),
    ("17.4435 78.3772 18", "could not read"),
    ("lat 17.4435 lng 78.3772", "could not read"), ("1e3 78.37", "could not read"),
    ("17°26'36.6\"N", "found 1"), ("17°26'36.6\"N 18°22'37.9\"N", "two latitudes"),
    ("78°E 79°W", "two longitudes"), ("17°61'N 78°22'E", "below 60"),
    ("17°26'61\"N 78°22'E", "below 60"), ("17.5°26'N 78°E", "decimals"),
    ("-17°S 78°E", "minus sign"), ("26'36\"N 78°E", "could not read"),
    ("95, 78.3772", "within ±85"), ("17.4435, 181", "within ±180"),
    ("120, 17.44", "swapped?"),
    ("https://maps.app.goo.gl/AbCdEfGh", "short Google Maps links"),
    ("https://www.google.com/maps/place/Hitech+City", "no coordinates"),
    ("17.4435," + " " * 5000 + "78.3772", "too long"),
    ("17.4435" + " " * 4000 + "x", "could not read"),  # linear, not quadratic, regexes
])
def test_parse_latlng_rejects_ambiguous_or_unreadable_input(text, why):
    err = _code(aoi.parse_latlng, text)
    assert err.code == "invalid" and why in str(err), str(err)


def test_parse_latlng_rejects_non_text():
    for v in (None, 17.4435, (17.4435, 78.3772)):
        assert _code(aoi.parse_latlng, v).code == "invalid"


@pytest.mark.parametrize("text", [
    "78.3772, 17.4435", "78,3772; 17,4435", "78°22'37.9\" 17°26'36.6\"",
    "78.3772N 17.4435E", "https://maps.google.com/?q=78.3772,17.4435"])
def test_parse_latlng_detects_swapped_pairs(text):
    err = _code(aoi.parse_latlng, text)
    assert err.code == "swapped" and "Did you mean 17.4435, 78.37" in str(err)
    assert err.hint["lat"] == pytest.approx(17.4435)
    assert err.hint["lng"] == pytest.approx(78.377, abs=1e-3)


def test_parse_latlng_keeps_valid_points_outside_hyderabad():
    assert aoi.parse_latlng("28.6139, 77.2090") == (28.6139, 77.209)  # Delhi, not swapped
    assert not aoi.plan(lat=28.6139, lng=77.2090)["in_coverage"]


def test_the_inputs_that_went_to_the_wrong_place():
    """The GUI's parseFloat sent these 63.5 km / 2.4 km / 7,476 km away."""
    assert aoi.parse_latlng("17,4435 78,3772") == (17.4435, 78.3772)  # was 17, 78
    assert aoi.parse_latlng("17.4435, 78.3772") == (17.4435, 78.3772)  # pasted pair
    assert aoi.parse_latlng("17°26'36.6\"N 78°22'37.9\"E") == pytest.approx(DMS)
    assert _code(aoi.parse_latlng, "78.3772 17.4435").code == "swapped"  # was billed
