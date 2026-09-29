"""OpenStreetMap data via Overpass: buildings, landuse, roads, water, POIs."""
import json
import time
from pathlib import Path

import geopandas as gpd
import requests
from shapely.geometry import LineString, Point, Polygon

from . import config

OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
# public Overpass instances reject or rate-limit requests without a
# meaningful User-Agent (406 / 429)
HEADERS = {"User-Agent": "UrbanSeg/1.0 (+https://github.com/asblhyd/urbanseg)"}

QUERY = """
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


def fetch_overpass(bbox, cache_path: Path | None = None, timeout=360, rounds=2,
                   pause=10):
    """bbox = (west, south, east, north). Returns overpass JSON dict.

    Tries each mirror `rounds` times, `timeout` s per attempt.
    """
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    w, s, e, n = bbox
    q = QUERY.format(w=w, s=s, e=e, n=n)
    last_err = None
    for url in OVERPASS_URLS * rounds:
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(data))
                return data
            last_err = f"HTTP {r.status_code}"
        except Exception as ex:  # noqa: BLE001
            last_err = str(ex)
        time.sleep(pause)
    raise RuntimeError(f"overpass failed for {bbox}: {last_err}")


def _way_geom(el):
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


WATER_WAY_TAGS = {"river", "canal", "stream", "drain", "ditch"}


def _is_water_poly(tags):
    return (tags.get("natural") in ("water", "wetland")
            or tags.get("landuse") in ("reservoir", "basin"))


def parse_overpass(data):
    """Split overpass JSON into typed GeoDataFrames (EPSG:4326)."""
    buildings, landuse, roads, rails, water, greens, pois = [], [], [], [], [], [], []
    waterways = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "relation":
            if _is_water_poly(tags):
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
        geom = _way_geom(el)
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
        elif _is_water_poly(tags) and geom.geom_type == "Polygon":
            water.append({"geometry": geom})
        elif tags.get("waterway") in WATER_WAY_TAGS and geom.geom_type == "LineString":
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


def get_area_osm(name, bbox, **fetch_kw):
    # v2: query includes water relations, wetlands, reservoirs, waterway lines
    cache = config.OSM_DIR / f"{name}_v2.json"
    return parse_overpass(fetch_overpass(bbox, cache_path=cache, **fetch_kw))
