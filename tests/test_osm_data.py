"""osm_data: frozen v2 parsing, the v3 parser fixes, the validated live cache.

Overpass is never called: requests.Session.post (live path) and
requests.post (v2 path) are replaced by fakes, and a fake clock drives the
deadline tests (one uses real time: a read blocked until the deadline).
"""
import hashlib
import json
import math
import os
import pickle
import shutil
import threading
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import requests
from shapely.geometry import LineString, Point, Polygon

from urbanseg import config, imagery, osm_data


@pytest.fixture(autouse=True)
def _live_cache_in_tmp(monkeypatch, tmp_path):
    """The default live cache (<repo>/.cache/osm_live, or $URBANSEG_OSM_CACHE)
    must not collect test answers: they would be served to later runs."""
    monkeypatch.delenv("URBANSEG_OSM_CACHE", raising=False)
    monkeypatch.setattr(config, "ROOT", tmp_path / "repo")


# ---- frozen copy of the v2 query and parser the v1 labels were built with ----

V2_QUERY = """
[out:json][timeout:300];
(
  way["building"]({s},{w},{n},{e});
  way["landuse"]({s},{w},{n},{e});
  way["highway"]({s},{w},{n},{e});
  way["railway"]({s},{w},{n},{e});
  way["natural"="water"]({s},{w},{n},{e});
  way["natural"="wetland"]({s},{w},{n},{e});
  way["landuse"~"reservoir|basin"]({s},{w},{n},{e});
  relation["natural"="water"]({s},{w},{n},{e});
  relation["natural"="wetland"]({s},{w},{n},{e});
  relation["landuse"~"reservoir|basin"]({s},{w},{n},{e});
  way["leisure"~"park|garden|pitch|playground"]({s},{w},{n},{e});
  node["shop"]({s},{w},{n},{e});
  node["amenity"]({s},{w},{n},{e});
  node["office"]({s},{w},{n},{e});
  node["craft"]({s},{w},{n},{e});
  node["tourism"~"hotel|guest_house"]({s},{w},{n},{e});
);
out tags geom;
"""


def _v2_way_geom(el):
    pts = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
    if len(pts) < 2:
        return None
    closed = pts[0] == pts[-1]
    if closed and len(pts) >= 4:
        try:
            poly = Polygon(pts)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:  # noqa: BLE001
            return None
    return LineString(pts)


V2_WATER_WAY_TAGS = {"river", "canal", "stream", "drain", "ditch"}


def _v2_is_water_poly(tags):
    return (tags.get("natural") in ("water", "wetland")
            or tags.get("landuse") in ("reservoir", "basin"))


def v2_parse_overpass(data):
    buildings, landuse, roads, rails, water, greens, pois = [], [], [], [], [], [], []
    waterways = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "relation":
            if _v2_is_water_poly(tags):
                from shapely.ops import linemerge, polygonize, unary_union
                lines = []
                for m in el.get("members", []):
                    if (m.get("type") == "way" and m.get("role") in ("outer", "")
                            and m.get("geometry")):
                        pts = [(g["lon"], g["lat"]) for g in m["geometry"]]
                        if len(pts) >= 2:
                            lines.append(LineString(pts))
                if lines:
                    try:
                        for p in polygonize(linemerge(unary_union(lines))):
                            water.append({"geometry": p})
                    except Exception:  # noqa: BLE001
                        pass
            continue
        if el["type"] == "node":
            cat = ("shop" if "shop" in tags else
                   "office" if "office" in tags else
                   "amenity" if "amenity" in tags else
                   "craft" if "craft" in tags else
                   "tourism" if "tourism" in tags else None)
            if cat:
                pois.append({"geometry": Point(el["lon"], el["lat"]),
                             "cat": cat, "value": tags.get(cat, "")})
            continue
        if el["type"] != "way":
            continue
        geom = _v2_way_geom(el)
        if geom is None:
            continue
        if "building" in tags and geom.geom_type == "Polygon":
            buildings.append({"geometry": geom, "building": tags["building"],
                              "levels": tags.get("building:levels", "")})
        elif "landuse" in tags and geom.geom_type == "Polygon":
            landuse.append({"geometry": geom, "landuse": tags["landuse"]})
        elif "highway" in tags and geom.geom_type == "LineString":
            roads.append({"geometry": geom, "highway": tags["highway"]})
        elif "railway" in tags and geom.geom_type == "LineString":
            rails.append({"geometry": geom})
        elif _v2_is_water_poly(tags) and geom.geom_type == "Polygon":
            water.append({"geometry": geom})
        elif tags.get("waterway") in V2_WATER_WAY_TAGS and geom.geom_type == "LineString":
            waterways.append({"geometry": geom, "waterway": tags["waterway"]})
        elif "leisure" in tags and geom.geom_type == "Polygon":
            greens.append({"geometry": geom, "leisure": tags["leisure"]})

    def gdf(rows, cols):
        if not rows:
            return gpd.GeoDataFrame({c: [] for c in cols}, geometry=[], crs=4326)
        return gpd.GeoDataFrame(rows, crs=4326)

    return {
        "buildings": gdf(buildings, ["building", "levels"]),
        "landuse": gdf(landuse, ["landuse"]),
        "roads": gdf(roads, ["highway"]),
        "rails": gdf(rails, []),
        "water": gdf(water, []),
        "waterways": gdf(waterways, ["waterway"]),
        "greens": gdf(greens, ["leisure"]),
        "pois": gdf(pois, ["cat", "value"]),
    }


# ---- helpers ----------------------------------------------------------------

def assert_layers_equal(a, b, where=""):
    """Same layers, columns, dtypes, index, attribute values and geometry WKB."""
    assert list(a) == list(b), where
    for k in a:
        x, y, at = a[k], b[k], f"{where}:{k}"
        assert x.crs == y.crs, at
        assert list(x.columns) == list(y.columns), at
        pd.testing.assert_frame_equal(pd.DataFrame(x.drop(columns="geometry")),
                                      pd.DataFrame(y.drop(columns="geometry")),
                                      check_exact=True, obj=at)
        assert list(x.geometry.to_wkb()) == list(y.geometry.to_wkb()), at


def way(i, tags, coords):
    return {"type": "way", "id": i, "tags": tags,
            "geometry": [{"lat": y, "lon": x} for x, y in coords]}


def node(i, tags, x, y):
    return {"type": "node", "id": i, "lat": y, "lon": x, "tags": tags}


def ring(x, y, d=0.0004):
    return [(x, y), (x + d, y), (x + d, y + d), (x, y + d), (x, y)]


def line(x, y, d=0.0008):
    return [(x, y), (x + d, y + d / 2), (x + 2 * d, y)]


SAMPLE = {
    "version": 0.6, "generator": "fake Overpass",
    "osm3s": {"timestamp_osm_base": "2026-10-01T09:30:00Z"},
    "elements": [
        way(1, {"building": "apartments", "building:levels": "5"}, ring(78.361, 17.441)),
        way(2, {"building": "yes", "highway": "service"}, ring(78.362, 17.441)),
        way(3, {"building": "yes"},  # bow-tie: invalid, repaired by buffer(0)
            [(78.363, 17.441), (78.3634, 17.4414), (78.3634, 17.441), (78.363, 17.4414),
             (78.363, 17.441)]),
        way(10, {"highway": "residential"}, line(78.361, 17.443)),
        way(11, {"highway": "tertiary", "junction": "roundabout"}, ring(78.362, 17.443)),
        way(12, {"highway": "pedestrian", "area": "yes"}, ring(78.363, 17.443)),
        way(13, {"highway": "services"}, ring(78.364, 17.443)),
        way(20, {"railway": "rail"}, line(78.361, 17.445)),
        way(30, {"landuse": "reservoir"}, ring(78.361, 17.447)),
        way(31, {"natural": "water", "landuse": "recreation_ground"}, ring(78.362, 17.447)),
        way(32, {"landuse": "residential"}, ring(78.363, 17.447, 0.002)),
        way(33, {"natural": "wetland"}, ring(78.366, 17.447)),
        way(40, {"waterway": "drain"}, line(78.361, 17.449)),
        way(41, {"waterway": "canal"}, ring(78.363, 17.449)),
        way(50, {"leisure": "park"}, ring(78.361, 17.451)),
        way(60, {"highway": "footway"}, [(78.36, 17.45)]),  # a single point: skipped
        node(100, {"shop": "bakery", "name": "X"}, 78.3611, 17.4411),
        node(101, {"amenity": "bank"}, 78.3612, 17.4412),
        node(102, {"name": "no category"}, 78.3613, 17.4413),
        {"type": "relation", "id": 200, "tags": {"natural": "water", "type": "multipolygon"},
         "members": [{"type": "way", "ref": ref, "role": "outer",
                      "geometry": [{"lat": y, "lon": x} for x, y in half]}
                     for ref, half in ((8, ring(78.365, 17.452, 0.001)[:3]),
                                       (9, ring(78.365, 17.452, 0.001)[2:]))]},
        # what "out tags geom" really returns for a relation: tags, no members
        {"type": "relation", "id": 201, "tags": {"natural": "water"}},
    ],
}


# ---- v2 is frozen -------------------------------------------------------------

def test_v2_query_is_unchanged():
    assert osm_data.QUERY == V2_QUERY


WATERWAY_CLAUSE = '  way["waterway"~"^(river|canal|stream|drain|ditch)$"]({s},{w},{n},{e});'


def test_query_v3_is_v2_plus_waterways():
    v2 = V2_QUERY.splitlines()
    at = v2.index('  way["railway"]({s},{w},{n},{e});') + 1
    assert osm_data.QUERY_V3.splitlines() == v2[:at] + [WATERWAY_CLAUSE] + v2[at:]


def test_query_v4_is_v3_plus_water_relation_geometry():
    assert osm_data.QUERY_V4.startswith(osm_data.QUERY_V3)
    rels = osm_data.QUERY_V4[len(osm_data.QUERY_V3):].splitlines()
    want = [l for l in V2_QUERY.splitlines() if l.startswith("  relation[")]
    assert rels == ["("] + want + [");", "out body geom;"] and len(want) == 3
    assert osm_data.QUERY_WATER_RELATIONS.splitlines()[1:] == rels
    q = osm_data.QUERY_V4.format(w=78.36, s=17.44, e=78.37, n=17.45)
    assert q.count("[timeout:") == 1 and "{" not in q.replace("[out:json]", "")


# a lake as Overpass sends it for "out body geom": two outer ways that close a ring
LAKE = {"type": "relation", "id": 7, "tags": {"natural": "water", "type": "multipolygon"},
        "members": [
            {"type": "way", "ref": 1, "role": "outer", "geometry": [
                {"lat": 17.440, "lon": 78.360}, {"lat": 17.440, "lon": 78.364},
                {"lat": 17.444, "lon": 78.364}]},
            {"type": "way", "ref": 2, "role": "outer", "geometry": [
                {"lat": 17.444, "lon": 78.364}, {"lat": 17.444, "lon": 78.360},
                {"lat": 17.440, "lon": 78.360}]}]}


def test_water_relation_members_become_a_lake_and_bare_tags_do_not():
    bare = {k: v for k, v in LAKE.items() if k != "members"}  # what "out tags" sends
    for version in ("v2", "v3"):
        assert len(osm_data.parse_overpass({"elements": [bare]}, version=version)["water"]) == 0
        water = osm_data.parse_overpass({"elements": [bare, LAKE]}, version=version)["water"]
        assert len(water) == 1
        assert water.geometry.iloc[0].area == pytest.approx(0.004 * 0.004)


def _ring(*lonlat):
    return [{"lon": x, "lat": y} for x, y in lonlat]


def test_v3_assembles_single_way_lakes_islands_and_gapped_rings():
    square = _ring((78.360, 17.440), (78.364, 17.440), (78.364, 17.444), (78.360, 17.444),
                   (78.360, 17.440))
    island = _ring((78.361, 17.441), (78.362, 17.441), (78.362, 17.442), (78.361, 17.442),
                   (78.361, 17.441))
    single = {"type": "relation", "id": 8, "tags": {"natural": "water"}, "members": [
        {"type": "way", "role": "outer", "geometry": square},
        {"type": "way", "role": "inner", "geometry": island}]}
    # two halves whose ends miss each other by ~0.5 m, as OSM has Himayat Sagar
    gapped = {"type": "relation", "id": 9, "tags": {"natural": "water"}, "members": [
        {"type": "way", "role": "outer", "geometry": square[:3]},
        {"type": "way", "role": "outer", "geometry": _ring(
            (78.364004, 17.444003), (78.360, 17.444), (78.360004, 17.440003))}]}
    for rel, area in ((single, 0.004 ** 2 - 0.001 ** 2), (gapped, 0.004 ** 2)):
        water = osm_data.parse_overpass({"elements": [rel]}, version="v3")["water"]
        assert len(water) == 1 and water.geometry.iloc[0].area == pytest.approx(area, rel=1e-3)
        # v2 is frozen, and its relation handling drops both
        assert len(osm_data.parse_overpass({"elements": [rel]})["water"]) == 0


def test_live_answer_with_a_lake_relation_has_water(overpass, tmp_path):
    overpass.steps = [answer({**SAMPLE, "elements": SAMPLE["elements"] + [LAKE]})]
    layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    base = osm_data.parse_overpass(SAMPLE, version="v3")
    assert len(layers["water"]) == len(base["water"]) + 1
    call, = overpass.calls
    assert call["query"].rstrip().endswith("out body geom;")


def test_water_relations_snapshot_uses_its_own_query(monkeypatch, tmp_path):
    posts = []
    monkeypatch.setattr(requests, "post",
                        lambda url, **kw: posts.append(kw) or _Answer({"elements": [LAKE]}))
    path = tmp_path / "lake_water.json"
    got = osm_data.fetch_overpass((78.36, 17.44, 78.37, 17.45), cache_path=path, rounds=1,
                                  pause=0, query=osm_data.QUERY_WATER_RELATIONS)
    assert got == {"elements": [LAKE]} and json.loads(path.read_text()) == got
    assert posts[0]["data"]["data"] == osm_data.QUERY_WATER_RELATIONS.format(
        w=78.36, s=17.44, e=78.37, n=17.45)


def test_query_v3_formats_like_query():
    bbox = dict(w=78.36, s=17.44, e=78.37, n=17.45)
    q = osm_data.QUERY_V3.format(**bbox)  # needed a timeout= too, unlike QUERY
    assert q.replace(WATERWAY_CLAUSE.format(**bbox) + "\n", "") == V2_QUERY.format(**bbox)


def test_v2_parse_is_identical_on_a_synthetic_answer():
    assert_layers_equal(osm_data.parse_overpass(SAMPLE), v2_parse_overpass(SAMPLE))
    assert_layers_equal(osm_data.parse_overpass(SAMPLE, version="v2"), v2_parse_overpass(SAMPLE))
    assert_layers_equal(osm_data.parse_overpass({}), v2_parse_overpass({}))


def test_get_area_osm_reads_the_v2_snapshot_as_before():
    config.OSM_DIR.mkdir(parents=True)
    (config.OSM_DIR / "area_v2.json").write_text(json.dumps(SAMPLE))
    assert_layers_equal(osm_data.get_area_osm("area", (78.36, 17.44, 78.37, 17.45)),
                        v2_parse_overpass(SAMPLE))


def test_v2_fetch_writes_the_same_bytes_atomically(monkeypatch, tmp_path):
    class Answer:
        status_code = 200

        def json(self):
            return SAMPLE

    posts = []
    monkeypatch.setattr(requests, "post", lambda url, **kw: posts.append(kw) or Answer())
    path = tmp_path / "osm" / "area_v2.json"
    bbox = (78.36, 17.44, 78.37, 17.45)
    assert osm_data.fetch_overpass(bbox, cache_path=path, rounds=1, pause=0) == SAMPLE
    assert path.read_text() == json.dumps(SAMPLE)
    assert [p.name for p in path.parent.iterdir()] == ["area_v2.json"]  # no temp file left
    assert posts[0]["data"]["data"] == V2_QUERY.format(w=78.36, s=17.44, e=78.37, n=17.45)
    assert posts[0]["headers"] == osm_data.HEADERS
    # an existing file is returned as is: a label snapshot is never refetched
    path.write_text(json.dumps({"elements": [], "note": "snapshot"}))
    assert osm_data.fetch_overpass(bbox, cache_path=path)["note"] == "snapshot"
    assert len(posts) == 1


class _Answer:
    def __init__(self, data, status=200):
        self.status_code, self._data = status, data

    def json(self):
        return self._data


ABORTED = {"elements": SAMPLE["elements"][:1],
           "remark": 'runtime error: Query timed out in "query" at line 3 after 301 seconds.'}


def test_v2_fetch_never_keeps_an_aborted_answer(monkeypatch, tmp_path):
    answers = iter([_Answer(ABORTED), _Answer(SAMPLE)])
    monkeypatch.setattr(requests, "post", lambda url, **kw: next(answers))
    path = tmp_path / "area_v2.json"
    assert osm_data.fetch_overpass((78.36, 17.44, 78.37, 17.45), cache_path=path,
                                   rounds=1, pause=0) == SAMPLE
    assert json.loads(path.read_text()) == SAMPLE


def test_v2_fetch_hedges_a_slow_mirror(monkeypatch, tmp_path):
    first, second, third = osm_data.OVERPASS_URLS
    release, asked = threading.Event(), []

    def post(url, **kw):
        asked.append(url)
        if url == first:  # hangs until the test ends
            release.wait(10)
            return _Answer(SAMPLE)
        return _Answer(ABORTED) if url == second else _Answer(SAMPLE)

    monkeypatch.setattr(requests, "post", post)
    path = tmp_path / "area_v2.json"
    t0 = time.monotonic()
    try:
        assert osm_data.fetch_overpass((78.36, 17.44, 78.37, 17.45), cache_path=path,
                                       pause=0, hedge_s=0.2) == SAMPLE
    finally:
        release.set()
    assert time.monotonic() - t0 < 3 and asked == [first, second, third]
    assert path.read_text() == json.dumps(SAMPLE)


def test_v2_hedged_fetch_prefers_elements_and_reports_failures(monkeypatch, tmp_path):
    empty = {"elements": []}
    by_url = dict(zip(osm_data.OVERPASS_URLS, [_Answer(empty), _Answer(SAMPLE), _Answer(SAMPLE)]))
    monkeypatch.setattr(requests, "post", lambda url, **kw: by_url[url])
    got = osm_data.fetch_overpass((0, 0, 1, 1), pause=0, hedge_s=5)
    assert got == SAMPLE
    # every mirror empty: the empty answer is the answer
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Answer(empty))
    path = tmp_path / "empty_v2.json"
    assert osm_data.fetch_overpass((0, 0, 1, 1), cache_path=path, pause=0, hedge_s=5) == empty
    assert json.loads(path.read_text()) == empty
    # nothing usable: every mirror and round is named
    monkeypatch.setattr(requests, "post", lambda url, **kw: _Answer(None, status=429))
    with pytest.raises(RuntimeError, match="overpass failed") as ei:
        osm_data.fetch_overpass((0, 0, 1, 1), pause=0, hedge_s=5)
    assert str(ei.value).count("HTTP 429") == 2 * len(osm_data.OVERPASS_URLS)


def _snapshots():
    return sorted((config.DATA_DIR / "osm").glob("*_v2.json"))


@pytest.mark.data
def test_v2_parse_is_identical_on_every_label_snapshot():
    files = _snapshots()
    assert len(files) >= 26  # one per area, plus the old live caches
    for p in files:
        data = json.loads(p.read_text())
        assert_layers_equal(osm_data.parse_overpass(data), v2_parse_overpass(data), p.name)


@pytest.mark.data
def test_get_area_osm_serves_a_real_label_snapshot_unchanged():
    src = config.DATA_DIR / "osm" / "kompally_v2.json"
    config.OSM_DIR.mkdir(parents=True)
    shutil.copyfile(src, config.OSM_DIR / "kompally_v2.json")
    meta = json.loads((config.TILES_DIR / "kompally" / "meta.json").read_text())
    assert_layers_equal(osm_data.get_area_osm("kompally", imagery.bbox_latlng(meta)),
                        v2_parse_overpass(json.loads(src.read_text())))


# ---- v3 parser fixes ----------------------------------------------------------

def _both():
    return osm_data.parse_overpass(SAMPLE), osm_data.parse_overpass(SAMPLE, version="v3")


def test_v3_closed_highways_stay_lines_unless_they_are_areas():
    v2, v3 = _both()
    assert list(v2["roads"]["highway"]) == ["residential"]  # v2 dropped the roundabout
    assert list(v3["roads"]["highway"]) == ["residential", "tertiary"]
    roundabout = v3["roads"].geometry.iloc[1]
    assert roundabout.geom_type == "LineString" and roundabout.is_ring
    assert roundabout.equals(LineString(ring(78.362, 17.443)))
    # area=yes and highway=services rings are areas in both versions, never roads


def test_v3_tests_water_before_landuse():
    v2, v3 = _both()
    assert list(v2["landuse"]["landuse"]) == ["reservoir", "recreation_ground", "residential"]
    assert list(v3["landuse"]["landuse"]) == ["residential"]
    assert len(v2["water"]) == 2 and len(v3["water"]) == 4
    for got, want in zip(v3["water"].geometry,
                         [ring(78.361, 17.447), ring(78.362, 17.447), ring(78.366, 17.447)]):
        assert got.equals(Polygon(want))


def test_v3_parses_waterway_lines_even_when_closed():
    v2, v3 = _both()
    assert list(v2["waterways"]["waterway"]) == ["drain"]
    assert list(v3["waterways"]["waterway"]) == ["drain", "canal"]
    canal = v3["waterways"].geometry.iloc[1]
    assert canal.geom_type == "LineString" and canal.is_ring


def test_v3_leaves_the_other_layers_alone():
    v2, v3 = _both()
    for k in ("buildings", "rails", "greens", "pois"):
        assert_layers_equal({k: v3[k]}, {k: v2[k]}, "v3 vs v2")
    # a closed way tagged building and highway is still a building
    assert list(v3["buildings"]["building"][:2]) == ["apartments", "yes"]
    assert list(v3["pois"]["cat"]) == ["shop", "amenity"]


def test_unknown_parser_version_is_refused():
    with pytest.raises(ValueError):
        osm_data.parse_overpass(SAMPLE, version="v4")


def _closed_line_highway(el):
    t, g = el.get("tags", {}), el.get("geometry", [])
    return (el["type"] == "way" and "highway" in t and len(g) >= 4 and g[0] == g[-1]
            and t.get("area") != "yes" and t["highway"] not in ("services", "rest_area")
            and not {"building", "landuse", "natural", "leisure"} & set(t))


@pytest.mark.data
def test_v3_on_label_snapshots_only_adds_rings_and_moves_water():
    added = 0
    for p in _snapshots():
        data = json.loads(p.read_text())
        v2, v3 = v2_parse_overpass(data), osm_data.parse_overpass(data, version="v3")
        for k in ("buildings", "rails", "greens", "pois", "waterways"):
            assert_layers_equal({k: v3[k]}, {k: v2[k]}, p.name)
        rings = sum(map(_closed_line_highway, data["elements"]))
        assert len(v3["roads"]) == len(v2["roads"]) + rings, p.name
        assert len(v3["water"]) + len(v3["landuse"]) == len(v2["water"]) + len(v2["landuse"])
        added += rings
    assert added >= 26  # roundabouts and loop roads v2 dropped


# ---- live cache ---------------------------------------------------------------

BBOX = (78.3600001, 17.4400001, 78.3700002, 17.4500003)
INFO_KEYS = {"source", "mirror", "osm_base", "fetched_at", "elapsed_s", "cache_file",
             "query_version", "elements"}
PARTIAL = {**SAMPLE, "remark": 'runtime error: Query timed out in "query" at line 4 '
                               'after 25 seconds.'}
EMPTY = {"version": 0.6, "osm3s": SAMPLE["osm3s"], "elements": []}


class FakeAnswer:
    """A streamed requests.Response stand-in; chunk overrides the chunk size."""

    def __init__(self, status=200, body=b"", on_chunk=None, chunk=None):
        self.status_code, self.body, self.on_chunk = status, body, on_chunk
        self.chunk = chunk

    def iter_content(self, chunk_size=1):
        size = self.chunk or chunk_size
        for i in range(0, len(self.body), size):
            if self.on_chunk is not None:
                self.on_chunk()
            yield self.body[i:i + size]

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def answer(obj=SAMPLE, status=200, **kw):
    return FakeAnswer(status, json.dumps(obj).encode(), **kw)


class FakeOverpass:
    """Stands in for requests.Session.post; plays back one step per request."""

    def __init__(self):
        self.steps, self.calls = [], []

    def post(self, session, url, data=None, headers=None, timeout=None, stream=False, **kw):
        self.calls.append({"url": url, "query": data["data"], "headers": headers,
                           "timeout": timeout, "stream": stream})
        assert self.steps, "unexpected Overpass request"
        step = self.steps.pop(0)
        step = step() if callable(step) else step
        if isinstance(step, BaseException):
            raise step
        return step


@pytest.fixture
def overpass(monkeypatch):
    fake = FakeOverpass()
    monkeypatch.setattr(requests.Session, "post",
                        lambda self, url, **kw: fake.post(self, url, **kw))
    return fake


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    fake = FakeClock()
    monkeypatch.setattr(osm_data, "_clock", fake)
    return fake


def _age_cache(path, days):
    env = json.loads(path.read_text())
    then = datetime.now(timezone.utc) - timedelta(days=days)
    env["fetched_at"] = then.strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(env))


def test_live_round_trip(overpass, tmp_path):
    overpass.steps = [answer()]
    layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert set(info) == INFO_KEYS
    assert (info["source"], info["query_version"]) == ("overpass", "v4")
    assert info["mirror"] == osm_data.OVERPASS_URLS[0]
    assert info["osm_base"] == "2026-10-01T09:30:00Z"
    path = osm_data.live_cache_path(BBOX, tmp_path)
    assert info["cache_file"] == str(path)
    env = json.loads(path.read_text())
    assert set(env) == {"query_version", "bbox", "fetched_at", "osm_base", "mirror", "data"}
    assert env["query_version"] == "v4" and env["data"] == SAMPLE
    assert env["bbox"] == [78.36, 17.44, 78.37, 17.45]
    assert env["fetched_at"] == info["fetched_at"]
    assert info["elements"] == len(SAMPLE["elements"])
    assert_layers_equal(layers, osm_data.parse_overpass(SAMPLE, version="v3"))

    layers2, info2 = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert len(overpass.calls) == 1 and info2["source"] == "cache"
    same = ("mirror", "osm_base", "fetched_at", "cache_file", "query_version", "elements")
    assert {k: info2[k] for k in same} == {k: info[k] for k in same}
    assert_layers_equal(layers2, layers)


def test_answer_without_osm3s_has_no_osm_base(overpass, tmp_path):
    overpass.steps = [answer({**SAMPLE, "osm3s": "unexpected"})]
    layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert info["osm_base"] is None and len(layers["roads"]) == 2


def test_live_cache_key(tmp_path, monkeypatch):
    key = "v4" + "78.360000,17.440000,78.370000,17.450000"
    expect = tmp_path / f"live4_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"
    assert osm_data.live_cache_path(BBOX, tmp_path) == expect
    # one ulp of float noise (another machine's libm) no longer changes the file
    assert osm_data.live_cache_path([math.nextafter(v, 0) for v in BBOX], tmp_path) == expect
    assert osm_data.live_cache_path((BBOX[0] + 1e-5, *BBOX[1:]), tmp_path) != expect
    assert osm_data.live_cache_path(BBOX).parent == config.ROOT / ".cache" / "osm_live"
    monkeypatch.setenv("URBANSEG_OSM_CACHE", str(tmp_path / "volume"))
    assert osm_data.live_cache_path(BBOX) == tmp_path / "volume" / expect.name


def test_default_cache_dir_is_outside_data(overpass):
    # data/ holds the read-only label snapshots: a runtime cache goes elsewhere
    overpass.steps = [answer()]
    _, info = osm_data.get_live_osm(BBOX)
    path = Path(info["cache_file"])
    assert path.parent == config.ROOT / ".cache" / "osm_live" and path.is_file()
    assert not config.OSM_DIR.exists()
    assert config.DATA_DIR not in path.parents and config.ROOT / "data" not in path.parents


def test_truncated_cache_is_refetched(overpass, tmp_path):
    overpass.steps = [answer(), answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    path = osm_data.live_cache_path(BBOX, tmp_path)
    text = path.read_text()
    path.write_text(text[: len(text) // 2])  # what a crash mid-write used to leave
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert info["source"] == "overpass" and len(overpass.calls) == 2
    assert json.loads(path.read_text())["data"] == SAMPLE


MALFORMED = {"elements": [{"type": "node", "id": 1, "tags": {"shop": "x"}}]}  # no lat/lon


@pytest.mark.parametrize("tamper", ["bbox", "version", "no_elements", "remark", "not_a_dict",
                                    "fetched_at", "future", "empty", "unparsable"])
def test_invalid_cache_is_refetched(overpass, tmp_path, tamper):
    overpass.steps = [answer(), answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    path = osm_data.live_cache_path(BBOX, tmp_path)
    env = json.loads(path.read_text())
    if tamper == "bbox":  # another area's answer under this name
        env["bbox"] = [78.0, 17.0, 78.1, 17.1]
    elif tamper == "version":  # an answer of the v3 query: no lake relations
        env["query_version"] = "v3"
    elif tamper == "no_elements":
        del env["data"]["elements"]
    elif tamper == "remark":
        env["data"] = PARTIAL
    elif tamper == "not_a_dict":
        env = [env]
    elif tamper == "fetched_at":
        env["fetched_at"] = "yesterday"
    elif tamper == "future":  # a skewed clock: it would never expire, even with ttl_days=0
        then = datetime.now(timezone.utc) + timedelta(days=2)
        env["fetched_at"] = then.strftime("%Y-%m-%dT%H:%M:%SZ")
    elif tamper == "empty":  # written before empty answers stopped being cached
        env["data"] = EMPTY
    else:  # written by other code: parsing it raised KeyError for the whole TTL
        env["data"] = MALFORMED
    path.write_text(json.dumps(env))
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert info["source"] == "overpass" and len(overpass.calls) == 2
    assert json.loads(path.read_text())["data"] == SAMPLE


def test_cache_expires_after_ttl(overpass, tmp_path):
    overpass.steps = [answer(), answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    path = osm_data.live_cache_path(BBOX, tmp_path)
    _age_cache(path, 29)
    assert osm_data.get_live_osm(BBOX, cache_dir=tmp_path)[1]["source"] == "cache"
    _age_cache(path, 31)
    assert osm_data.get_live_osm(BBOX, cache_dir=tmp_path, ttl_days=None)[1]["source"] == "cache"
    assert osm_data.get_live_osm(BBOX, cache_dir=tmp_path, ttl_days=60)[1]["source"] == "cache"
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)  # default ttl_days=30
    assert info["source"] == "overpass" and len(overpass.calls) == 2
    assert osm_data.get_live_osm(BBOX, cache_dir=tmp_path)[1]["source"] == "cache"  # renewed


def test_partial_answer_is_never_cached(overpass, tmp_path):
    overpass.steps = [lambda: answer(PARTIAL)] * 3
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path / "cache")
    assert ei.value.reason == "partial" and len(overpass.calls) == 3
    assert "Query timed out" in str(ei.value)
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize("first", [PARTIAL, MALFORMED, {"elements": [{"id": 1}]}, EMPTY],
                         ids=["partial", "malformed", "untyped-element", "empty"])
def test_unusable_answer_falls_over_to_the_next_mirror(overpass, tmp_path, first):
    overpass.steps = [answer(first), answer()]
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert info["mirror"] == osm_data.OVERPASS_URLS[1]
    assert info["elements"] == len(SAMPLE["elements"])
    assert json.loads(Path(info["cache_file"]).read_text())["data"] == SAMPLE


def test_empty_answer_is_returned_but_never_cached(overpass, tmp_path):
    # an urban bbox is never empty: a lagging or broken mirror must not
    # blank OSM for the next 30 days
    overpass.steps = [lambda: answer(EMPTY)] * 6
    for _ in range(2):  # the second call asks again
        layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path / "cache")
        assert (info["source"], info["cache_file"], info["elements"]) == ("overpass", None, 0)
        assert info["mirror"] == osm_data.OVERPASS_URLS[0]
        assert info["osm_base"] == "2026-10-01T09:30:00Z"
        assert list(layers) == list(osm_data.parse_overpass({}))
        assert all(len(g) == 0 for g in layers.values())
    assert len(overpass.calls) == 6 and not (tmp_path / "cache").exists()


def test_cache_write_is_atomic(overpass, tmp_path, monkeypatch):
    moves, real_replace = [], os.replace

    def replace(src, dst):
        moves.append((Path(src), Path(dst)))
        assert json.loads(Path(src).read_text())["data"] == SAMPLE  # complete before the rename
        real_replace(src, dst)

    monkeypatch.setattr(osm_data.os, "replace", replace)
    overpass.steps = [answer()]
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    (src, dst), = moves
    assert dst == Path(info["cache_file"]) and src.parent == dst.parent and src != dst
    assert [p.name for p in tmp_path.iterdir()] == [dst.name]  # no temp file left


def test_failed_cache_write_keeps_the_answer_and_the_old_file(overpass, tmp_path, monkeypatch):
    overpass.steps = [answer(), answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    path = osm_data.live_cache_path(BBOX, tmp_path)
    _age_cache(path, 40)
    before = path.read_bytes()

    def disk_full(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(osm_data.os, "replace", disk_full)
    layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    assert info["source"] == "overpass" and info["cache_file"] is None
    assert_layers_equal(layers, osm_data.parse_overpass(SAMPLE, version="v3"))
    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_mirrors_are_tried_in_turn_within_the_deadline(overpass, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(osm_data, "LIVE_ATTEMPT_S", 30)  # the sequential-era cap
    def slow_failure():
        clock.t += 40  # the first mirror needs 40 s to fail
        return answer(status=504)

    overpass.steps = [slow_failure, answer()]
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    assert [c["url"] for c in overpass.calls] == osm_data.OVERPASS_URLS[:2]
    # (connect, read): the connection is a share of min(30 s, time left), not
    # extra time on top of it
    assert overpass.calls[0]["timeout"] == (5, 25)
    assert overpass.calls[1]["timeout"] == pytest.approx((5 / 3, 10 / 3))
    assert "[timeout:20]" in overpass.calls[0]["query"]  # the server gives up first
    assert "[timeout:2]" in overpass.calls[1]["query"]
    assert info["mirror"] == osm_data.OVERPASS_URLS[1] and info["elapsed_s"] == 40


def test_attempts_never_outlast_the_time_left(overpass, clock, tmp_path, monkeypatch):
    monkeypatch.setattr(osm_data, "LIVE_ATTEMPT_S", 30)  # the sequential-era cap
    spent = [12, 30, 2]  # fake seconds each mirror takes to fail

    def slow_failure():
        clock.t += spent.pop(0)
        return answer(status=503)

    overpass.steps = [slow_failure] * 3
    with pytest.raises(osm_data.OverpassError):
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    budgets = [sum(c["timeout"]) for c in overpass.calls]
    assert budgets == pytest.approx([30, 30, 3])  # min(30 s, 45 - time used)


def test_deadline_stops_trying_further_mirrors(overpass, clock, tmp_path):
    def slow_failure():
        clock.t += 30
        return answer(status=503)

    overpass.steps = [slow_failure, slow_failure]
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    assert ei.value.reason == "timeout" and len(overpass.calls) == 2
    assert "overpass-api.de: unavailable (HTTP 503)" in str(ei.value)


def test_cache_is_served_even_without_time_for_overpass(overpass, tmp_path):
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=0)
    assert ei.value.reason == "timeout" and overpass.calls == []
    overpass.steps = [answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=0)
    assert info["source"] == "cache" and len(overpass.calls) == 1


def test_answer_trickling_past_the_deadline_is_a_timeout(overpass, clock, tmp_path):
    body = json.dumps(SAMPLE).encode()

    def tick():
        clock.t += 20  # the deadline passes at the third of four chunks

    overpass.steps = [lambda: FakeAnswer(200, body, on_chunk=tick, chunk=len(body) // 4 + 1)]
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    assert ei.value.reason == "timeout" and len(overpass.calls) == 1
    assert "answer incomplete at the deadline" in str(ei.value)


def test_answer_complete_just_after_the_deadline_is_used(overpass, clock, tmp_path):
    body = json.dumps(SAMPLE).encode()
    chunks = math.ceil(len(body) / 1500)

    def tick():
        clock.t += 46 / chunks  # the last chunk arrives a little after the deadline

    overpass.steps = [lambda: answer(on_chunk=tick, chunk=1500)]
    layers, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    assert info["source"] == "overpass" and info["elapsed_s"] > 45
    assert_layers_equal(layers, osm_data.parse_overpass(SAMPLE, version="v3"))


class StalledAnswer(FakeAnswer):
    """Sends the start of an answer, then nothing until its socket is shut down
    (as urllib3's HTTPResponse.shutdown() does to a read in another thread)."""

    def __init__(self, obj=SAMPLE):
        super().__init__(200, json.dumps(obj).encode())
        self.cut = threading.Event()
        self.raw = types.SimpleNamespace(shutdown=self.cut.set)

    def iter_content(self, chunk_size=1):
        yield self.body[:100]
        if not self.cut.wait(10):  # each read is within the read timeout, so no error
            raise AssertionError("nothing ended the stalled read")
        raise requests.exceptions.ChunkedEncodingError("Connection broken: IncompleteRead")


def test_stalled_answer_is_cut_at_the_deadline(overpass, tmp_path):
    stalled = StalledAnswer()
    overpass.steps = [stalled]
    t0 = time.monotonic()
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=1.5)
    assert ei.value.reason == "timeout" and stalled.cut.is_set()
    assert time.monotonic() - t0 < 3  # not the 10 s the read would have blocked
    assert not osm_data.live_cache_path(BBOX, tmp_path).exists()


@pytest.mark.parametrize("step,reason", [
    (lambda: answer(status=503), "unavailable"),
    (lambda: answer(status=429), "unavailable"),
    (lambda: FakeAnswer(200, b"<html>The server is probably too busy</html>"), "bad_response"),
    (lambda: answer({"version": 0.6}), "bad_response"),
    # parsing these raised KeyError, after the answer had been cached
    (lambda: answer(MALFORMED), "bad_response"),
    (lambda: FakeAnswer(200, b"[" * 100_000), "bad_response"),  # was RecursionError
    (lambda: requests.exceptions.ConnectTimeout("connect timed out (proxy http://u:pw@p)"),
     "timeout"),
    (lambda: requests.exceptions.ConnectionError("refused (proxy http://u:pw@p)"),
     "unavailable"),
], ids=["503", "429", "html", "no-elements", "malformed-element", "deep-nesting",
        "connect-timeout", "connection-error"])
def test_failure_reasons(overpass, tmp_path, step, reason):
    overpass.steps = [step] * 3
    with pytest.raises(osm_data.OverpassError) as ei:
        osm_data.get_live_osm(BBOX, cache_dir=tmp_path / "cache")
    assert ei.value.reason == reason and len(overpass.calls) == 3
    # no query text and no exception text (proxy URLs can hold credentials)
    assert "way[" not in str(ei.value) and "pw@" not in str(ei.value)
    assert not (tmp_path / "cache").exists()


def test_request_keeps_the_user_agent_and_asks_for_waterways(overpass, tmp_path):
    overpass.steps = [answer()]
    osm_data.get_live_osm(BBOX, cache_dir=tmp_path)
    call, = overpass.calls
    assert call["headers"]["User-Agent"] == osm_data.HEADERS["User-Agent"]
    assert call["stream"] is True
    assert ('way["waterway"~"^(river|canal|stream|drain|ditch)$"]'
            '(17.440000,78.360000,17.450000,78.370000);') in call["query"]


def test_overpass_error_contract():
    err = osm_data.OverpassError("partial", "overpass-api.de: partial (runtime error)")
    assert isinstance(err, RuntimeError) and err.reason == "partial"
    assert str(err) == "overpass partial: overpass-api.de: partial (runtime error)"
    back = pickle.loads(pickle.dumps(err))
    assert (type(back), back.reason, str(back)) == (type(err), err.reason, str(err))
    with pytest.raises(ValueError):
        osm_data.OverpassError("nope")


def test_slow_mirror_is_hedged_by_the_next(overpass, tmp_path, monkeypatch):
    # an overloaded mirror holds its request; the next one is asked after
    # LIVE_HEDGE_S and its answer wins long before the first would finish
    monkeypatch.setattr(osm_data, "LIVE_HEDGE_S", 0.2)

    def overloaded():
        time.sleep(2.0)
        return answer(EMPTY)

    overpass.steps = [overloaded, answer()]
    t0 = time.monotonic()
    _, info = osm_data.get_live_osm(BBOX, cache_dir=tmp_path, deadline_s=45)
    assert info["mirror"] == osm_data.OVERPASS_URLS[1]
    assert time.monotonic() - t0 < 1.5
    assert [c["url"] for c in overpass.calls] == osm_data.OVERPASS_URLS[:2]
