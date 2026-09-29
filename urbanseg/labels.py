"""Build 4-class ground-truth masks from Open Buildings + OSM for an area.

Classes: 0 other (road/rail/water), 1 residential, 2 commercial, 3 free land.
Painting order: free land base -> water/roads/rail -> buildings on top.
"""
import numpy as np
import geopandas as gpd
import pandas as pd
from PIL import Image
from rasterio import features as rfeatures
from rasterio.transform import Affine
from shapely.ops import transform as shp_transform

from . import config, imagery, open_buildings, osm_data

UTM = 32644  # UTM 44N covers Hyderabad
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}


def _world_arrays(lng, lat, zoom):
    import numpy as _np
    lng = _np.asarray(lng, dtype=float)
    lat = _np.asarray(lat, dtype=float)
    w = 256 * (2 ** zoom)
    x = (lng + 180.0) / 360.0 * w
    s = _np.sin(_np.radians(lat))
    y = (0.5 - _np.log((1 + s) / (1 - s)) / (4 * _np.pi)) * w
    return x, y


def merge_footprints(ob, osm_buildings):
    """Open Buildings footprints + OSM tag attach + OSM-only buildings."""
    ob = ob.copy()
    ob["tag"] = ""
    if len(osm_buildings) and len(ob):
        tagged = osm_buildings[osm_buildings["building"] != "yes"]
        if len(tagged):
            j = gpd.sjoin(ob, tagged[["building", "geometry"]],
                          how="left", predicate="intersects")
            j = j[~j.index.duplicated(keep="first")]
            ob["tag"] = j["building"].fillna("").values
    if len(osm_buildings):
        if len(ob):
            hit = gpd.sjoin(osm_buildings, ob[["geometry"]],
                            how="left", predicate="intersects")
            extra = osm_buildings.loc[hit[hit["index_right"].isna()].index.unique()]
        else:
            extra = osm_buildings
        if len(extra):
            extra = extra.rename(columns={"building": "tag"})[["tag", "geometry"]].copy()
            extra["area_in_meters"] = (
                extra.geometry.to_crs(UTM).area.values)
            extra["confidence"] = 1.0
            ob = pd.concat([ob, extra], ignore_index=True)
            ob = gpd.GeoDataFrame(ob, geometry="geometry", crs=4326)
    return ob


def classify_footprints(fp, layers):
    """Return int class (1 res / 2 com) per footprint row."""
    if not len(fp):
        return np.array([], dtype=int)
    fp = fp.reset_index(drop=True)
    fp_utm = fp.to_crs(UTM)
    cls = np.zeros(len(fp), dtype=int)

    tag = fp["tag"].fillna("").astype(str)
    cls[tag.isin(config.RES_BUILDING_TAGS).values] = 1
    cls[tag.isin(config.COM_BUILDING_TAGS).values] = 2

    # commercial-implying POIs on or immediately beside the building (8 m):
    # wider radii bleed red onto neighbours 2-3 rows deep
    poi_n = np.zeros(len(fp))
    pois = layers["pois"]
    if len(pois):
        keep = (pois["cat"].isin(["shop", "office", "craft", "tourism"])
                | ((pois["cat"] == "amenity")
                   & pois["value"].isin(config.AMENITY_COM)))
        pois = pois[keep]
    if len(pois):
        buf = gpd.GeoDataFrame({"i": np.arange(len(fp))},
                               geometry=fp_utm.geometry.buffer(8), crs=UTM)
        jj = gpd.sjoin(buf, pois.to_crs(UTM), predicate="intersects")
        cnt = jj.groupby("i").size()
        poi_n[cnt.index.values] = cnt.values

    # landuse under centroid
    lu_res = np.zeros(len(fp), dtype=bool)
    lu_com = np.zeros(len(fp), dtype=bool)
    lu = layers["landuse"]
    if len(lu):
        cent = gpd.GeoDataFrame({"i": np.arange(len(fp))},
                                geometry=fp.geometry.centroid, crs=4326)
        jj = gpd.sjoin(cent, lu, predicate="within")
        for i, lu_val in zip(jj["i"].values, jj["landuse"].values):
            if lu_val in config.RES_LANDUSE:
                lu_res[i] = True
            elif lu_val in config.COM_LANDUSE:
                lu_com[i] = True

    # road frontage: first row of buildings lining an arterial is almost
    # always shopfront commercial in Indian cities
    frontage = np.zeros(len(fp), dtype=bool)
    near_major = np.zeros(len(fp), dtype=bool)
    tert_frontage = np.zeros(len(fp), dtype=bool)
    roads = layers["roads"]
    if len(roads):
        mj = roads[roads["highway"].isin(MAJOR_ROADS)]
        if len(mj):
            mj_utm = mj.to_crs(UTM)
            frontage = fp_utm.geometry.intersects(
                mj_utm.buffer(16).union_all()).values
            near_major = fp_utm.geometry.intersects(
                mj_utm.buffer(30).union_all()).values
        tert = roads[roads["highway"] == "tertiary"]
        if len(tert):
            tert_frontage = fp_utm.geometry.intersects(
                tert.to_crs(UTM).buffer(18).union_all()).values

    # warehouses / sheds / campuses: large footprints are rarely single homes
    area = fp["area_in_meters"].fillna(0).values
    com_score = (1.5 * np.minimum(poi_n, 4)
                 + 2.0 * lu_com
                 + 1.5 * (area >= 1000)
                 + 1.0 * (area >= 3000)
                 + 2.0 * frontage
                 + 1.5 * (tert_frontage & (poi_n >= 1))
                 + 1.0 * (near_major & (area > 600)))
    res_score = 1.0 + 2.0 * lu_res
    untagged = cls == 0
    cls[untagged & (com_score > res_score + 0.5)] = 2
    cls[untagged & ~(com_score > res_score + 0.5)] = 1
    # transport infrastructure buildings belong to "other"
    cls[tag.isin(config.OTHER_BUILDING_TAGS).values] = 0
    return cls


def other_geoms_world(layers, zoom):
    """World-projected geometries for class 0 (roads/rail/water) from OSM."""
    def world(geoms):
        return [shp_transform(lambda x, y: _world_arrays(x, y, zoom), g)
                for g in geoms if g is not None and not g.is_empty]

    out = []
    if len(layers["water"]):
        out += world(layers["water"].geometry)
    roads = layers["roads"]
    if len(roads):
        widths = roads["highway"].map(config.ROAD_WIDTHS).fillna(5).values
        r_utm = roads.to_crs(UTM)
        buf = gpd.GeoSeries([g.buffer(wd / 2, cap_style=2)
                             for g, wd in zip(r_utm.geometry, widths)],
                            crs=UTM).to_crs(4326)
        out += world(buf)
    if len(layers["rails"]):
        r_utm = layers["rails"].to_crs(UTM)
        buf = gpd.GeoSeries([g.buffer(config.RAIL_WIDTH / 2)
                             for g in r_utm.geometry], crs=UTM).to_crs(4326)
        out += world(buf)
    ww = layers.get("waterways")
    if ww is not None and len(ww):
        widths = ww["waterway"].map(
            {"river": 25, "canal": 12, "stream": 8, "drain": 6, "ditch": 4}).fillna(6)
        w_utm = ww.to_crs(UTM)
        buf = gpd.GeoSeries([g.buffer(wd / 2, cap_style=2)
                             for g, wd in zip(w_utm.geometry, widths)],
                            crs=UTM).to_crs(4326)
        out += world(buf)
    return out


def build_mask(name, meta, layers=None):
    """Rasterize the ground-truth mask for a fetched area; saves mask + preview."""
    bbox = imagery.bbox_latlng(meta)
    if layers is None:
        layers = osm_data.get_area_osm(name, bbox)
    ob = open_buildings.load_for_bbox(bbox)
    fp = merge_footprints(ob, layers["buildings"])
    cls = classify_footprints(fp, layers)

    s = meta["scale"]
    H = meta["height_map_px"] * s
    W = meta["width_map_px"] * s
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    zoom = meta["zoom"]

    def world(geoms):
        return [shp_transform(lambda x, y: _world_arrays(x, y, zoom), g)
                for g in geoms if g is not None and not g.is_empty]

    mask = np.full((H, W), 3, dtype=np.uint8)  # free land base

    shapes = []
    if len(layers["water"]):
        shapes += [(g, 0) for g in world(layers["water"].geometry)]
    roads = layers["roads"]
    if len(roads):
        widths = roads["highway"].map(config.ROAD_WIDTHS).fillna(5).values
        r_utm = roads.to_crs(UTM)
        buf = [g.buffer(wd / 2, cap_style=2)
               for g, wd in zip(r_utm.geometry, widths)]
        buf = gpd.GeoSeries(buf, crs=UTM).to_crs(4326)
        shapes += [(g, 0) for g in world(buf)]
    if len(layers["rails"]):
        r_utm = layers["rails"].to_crs(UTM)
        buf = gpd.GeoSeries([g.buffer(config.RAIL_WIDTH / 2) for g in r_utm.geometry],
                            crs=UTM).to_crs(4326)
        shapes += [(g, 0) for g in world(buf)]
    ww = layers.get("waterways")
    if ww is not None and len(ww):
        widths = ww["waterway"].map(
            {"river": 25, "canal": 12, "stream": 8, "drain": 6, "ditch": 4}).fillna(6)
        w_utm = ww.to_crs(UTM)
        buf = gpd.GeoSeries([g.buffer(wd / 2, cap_style=2)
                             for g, wd in zip(w_utm.geometry, widths)],
                            crs=UTM).to_crs(4326)
        shapes += [(g, 0) for g in world(buf)]
    if len(fp):
        # +1.2 m dilation closes party-wall gaps in dense fabric that would
        # otherwise leak free-land pixels between touching buildings
        fp_dil = fp.geometry.to_crs(UTM).buffer(1.2).to_crs(4326)
        for klass in (0, 1, 2):
            sel = fp_dil[cls == klass]
            shapes += [(g, klass) for g in world(sel)]
    if shapes:
        rfeatures.rasterize(shapes, out=mask, transform=tr, default_value=0)

    config.MASKS_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask).save(config.MASKS_DIR / f"{name}.png")
    save_preview(name, mask)
    return mask, fp, cls


def save_preview(name, mask, alpha=0.45):
    img, _ = imagery.load_area(name)
    color = np.zeros_like(img)
    for k, c in config.CLASS_COLORS.items():
        color[mask == k] = c
    blend = (img * (1 - alpha) + color * alpha).astype(np.uint8)
    config.PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(blend).save(config.PREVIEW_DIR / f"{name}_overlay.png")
    return blend
