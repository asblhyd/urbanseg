"""Google Open Buildings v3 footprints for the Hyderabad region.

Build the local index once with:  python -m urbanseg.open_buildings
"""
import sys

import geopandas as gpd
import pandas as pd
import requests
import s2sphere
from shapely import wkt as shapely_wkt

from . import config

BASE = "https://storage.googleapis.com/open-buildings-data/v3/polygons_s2_level_4_gzip"
PARQUET = config.OPEN_BUILDINGS_DIR / "hyderabad_buildings.parquet"


def s2_tokens_for_bbox(bbox):
    w, s, e, n = bbox
    tokens = set()
    for lat, lng in [(s, w), (s, e), (n, w), (n, e), ((s + n) / 2, (w + e) / 2)]:
        cell = s2sphere.CellId.from_lat_lng(
            s2sphere.LatLng.from_degrees(lat, lng)).parent(4)
        tokens.add(cell.to_token())
    return sorted(tokens)


def download_and_filter(bbox=config.HYD_BBOX, force=False):
    """Download S2 cell CSVs covering bbox, keep rows inside bbox, save parquet."""
    if PARQUET.exists() and not force:
        return PARQUET
    config.OPEN_BUILDINGS_DIR.mkdir(parents=True, exist_ok=True)
    w, s, e, n = bbox
    frames = []
    for token in s2_tokens_for_bbox(bbox):
        gz = config.OPEN_BUILDINGS_DIR / f"{token}_buildings.csv.gz"
        if not gz.exists():
            url = f"{BASE}/{token}_buildings.csv.gz"
            print(f"downloading {url}", flush=True)
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = gz.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                tmp.rename(gz)
        print(f"filtering {gz.name}", flush=True)
        for chunk in pd.read_csv(gz, chunksize=500_000,
                                 usecols=["latitude", "longitude", "area_in_meters",
                                          "confidence", "geometry"]):
            m = ((chunk["longitude"] >= w) & (chunk["longitude"] <= e) &
                 (chunk["latitude"] >= s) & (chunk["latitude"] <= n))
            if m.any():
                frames.append(chunk[m])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["confidence"] >= 0.65]
    # latitude-sorted small row groups let load_for_bbox skip most of the file
    df = df.sort_values("latitude").reset_index(drop=True)
    tmp = PARQUET.with_suffix(".tmp")
    df.to_parquet(tmp, row_group_size=50_000)
    tmp.rename(PARQUET)
    print(f"saved {len(df):,} footprints -> {PARQUET}", flush=True)
    return PARQUET


def empty():
    return gpd.GeoDataFrame({"area_in_meters": [], "confidence": []},
                            geometry=[], crs=4326)


def load_for_bbox(bbox):
    """GeoDataFrame of footprints intersecting bbox (EPSG:4326).

    Empty when the local index has not been built.
    """
    if not PARQUET.exists():
        return empty()
    w, s, e, n = bbox
    pad = 0.002
    df = pd.read_parquet(PARQUET, filters=[
        ("longitude", ">=", w - pad), ("longitude", "<=", e + pad),
        ("latitude", ">=", s - pad), ("latitude", "<=", n + pad)])
    if df.empty:
        return empty()
    geom = df["geometry"].map(shapely_wkt.loads)
    return gpd.GeoDataFrame(
        df[["area_in_meters", "confidence"]].reset_index(drop=True),
        geometry=list(geom), crs=4326)


if __name__ == "__main__":
    download_and_filter(force="--force" in sys.argv)
    if "--keep-raw" not in sys.argv:
        for gz in config.OPEN_BUILDINGS_DIR.glob("*_buildings.csv.gz"):
            gz.unlink()
            print(f"removed {gz.name}", flush=True)
