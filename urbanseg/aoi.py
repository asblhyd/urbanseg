"""Areas of interest: input parsing, tile planning, AOI masks and clipping.

A request names either an AOI (GeoJSON polygons or a [w, s, e, n] bbox) or a
point plus grid size. plan() turns it into the nx x ny tile grid that
imagery.fetch_area will fetch, with its cost and coverage checks, before any
billable call is made; aoi_mask() and clip_geojson() then restrict stats and
vectors to the AOI instead of the whole fetched rectangle. Geometries are
(lon, lat) on WGS84 as in GeoJSON; pixel maths uses imagery's Web-Mercator
grid, and lengths and areas are WGS84 ground metres.
"""
import json
import math
import operator
import os
import re
import reprlib
from urllib.parse import parse_qsl, unquote_plus, urlsplit

import numpy as np
import shapely
from pyproj import Geod
from rasterio import features as rfeatures
from rasterio.transform import Affine
from shapely.geometry import box, mapping
from shapely.geometry import shape as as_shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

from . import config, imagery, labels

# per-request limits; a larger zone has to be split into several requests
MAX_AOI_KM2 = float(os.environ.get("URBANSEG_MAX_AOI_KM2", "4"))
MIN_AOI_M2 = float(os.environ.get("URBANSEG_MIN_AOI_M2", "1"))  # ~12 image px
# GEOS validation is up to quadratic in a ring's vertices (40 ms at 10k, 3.6 s
# at 100k), and plan, aoi_mask and clip_geojson each validate the AOI
MAX_AOI_VERTICES = int(os.environ.get("URBANSEG_MAX_AOI_VERTICES", "10000"))
MAX_AOI_PARTS = int(os.environ.get("URBANSEG_MAX_AOI_PARTS", "1000"))
MAX_TILES = int(os.environ.get("URBANSEG_MAX_TILES", "9"))    # 3 x 3 = the old Large
WARN_TILES = int(os.environ.get("URBANSEG_WARN_TILES", "4"))  # more than the old Medium
DEFAULT_MARGIN_M = 20
MAX_MARGIN_M = 1000  # far more context than one 1024 px (~290 m) model window sees
# the model is ~2.6 pp less accurate in the outer 32 image px of an extent
# (0.700 vs 0.726 on val/test), so an AOI should not reach into that band
EDGE_PX = 32
MAX_LAT = 85.0  # Web-Mercator imagery limit, as in the API's query bounds

# m2 per unit and display decimals (~1 m2 resolution in every unit);
# 1 gunta = 121 sq yd, the unit of Hyderabad (Telangana) land records
UNITS = {"m2": (1.0, 1), "ha": (1e4, 4), "acres": (4046.8564224, 4),
         "sq_yd": (0.83612736, 1), "guntas": (101.17141056, 2)}

_GEOD = Geod(ellps="WGS84")
# echoes of user input in messages stay short, however long the input
_REPR = reprlib.Repr(maxlevel=2, maxdict=4, maxlist=6, maxtuple=6, maxstring=60,
                     maxother=60)


class AOIError(ValueError):
    """An area or location the service cannot analyse as given.

    .code is machine-readable: "invalid" (unreadable or impossible input),
    "too_large" (over MAX_AOI_KM2 or the tile limit), "outside_coverage"
    (outside config.HYD_BBOX when the caller did not allow that) or
    "swapped" (latitude and longitude reversed). .hint is None or a
    JSON-ready dict, e.g. the corrected point for "swapped".
    """
    CODES = ("invalid", "too_large", "outside_coverage", "swapped")

    def __init__(self, message, code="invalid", hint=None):
        if code not in self.CODES:
            raise ValueError(f"unknown AOIError code {code!r}")
        super().__init__(message)
        self.code, self.hint = code, hint

    def as_dict(self):
        return {"code": self.code, "message": str(self), "hint": self.hint}


# ---------------------------------------------------------------- geometry

def geodesic_area_m2(geom):
    """WGS84 geodesic area (m2) of a lon/lat (Multi)Polygon, holes subtracted."""
    # pyproj sums signed ring areas, so exteriors must be CCW and holes CW
    return abs(_GEOD.geometry_area_perimeter(shapely.orient_polygons(geom))[0])


def _ground_mpp(lat, zoom=config.ZOOM):
    """WGS84 ground metres per map px at lat: (east-west, north-south).

    imagery.meters_per_map_px is the east-west scale on a sphere of the
    equatorial radius a. The ellipsoid's prime-vertical and meridional radii
    N and M make the true scales N/a and M/a times that (+0.03% and -0.58% at
    Hyderabad); their product is the WGS84 pixel area infer.area_stats counts.
    """
    phi = math.radians(lat)
    w = 1 - _GEOD.es * math.sin(phi) ** 2
    k = 2 * math.pi * _GEOD.a * math.cos(phi) / imagery.WORLD(zoom)
    return k / math.sqrt(w), k * (1 - _GEOD.es) / w ** 1.5


def _parts(geom):
    """Non-empty single parts of a geometry, Multi*/collections flattened."""
    if hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _parts(g)
    elif not geom.is_empty:
        yield geom


def _polygonal(geom):
    """geom reduced to its polygons (overlays can add lines and points)."""
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom
    polys = [p for p in _parts(geom) if p.geom_type == "Polygon"]
    return shapely.union_all(polys) if polys else shapely.Polygon()


def _repair(geom):
    # "structure" keeps the area a ring encloses (a bow-tie keeps both lobes)
    # where the default "linework" reading would punch even-odd holes
    if geom.is_valid:
        return geom
    return shapely.make_valid(geom, method="structure", keep_collapsed=False)


def _hyd():
    return box(*config.HYD_BBOX)


def _in_hyd(lat, lng):
    w, s, e, n = config.HYD_BBOX
    return s <= lat <= n and w <= lng <= e


def _fmt(v):
    return f"{v:.7f}".rstrip("0").rstrip(".") if abs(v) < 1e7 else f"{v:.6g}"


def _show(v):
    """User input, shortened for an error message."""
    try:
        return _REPR.repr(v)
    except ValueError:  # an int with more digits than Python will print
        return f"<{type(v).__name__}>"


def _finite(v, name):
    try:
        v = float(v)
    except (TypeError, ValueError, OverflowError):
        v = math.nan
    if not math.isfinite(v):
        raise AOIError(f"{name} must be a finite number")
    return v


def _whole(v, name, hi=None):
    """v as an int from 1 to hi (no upper bound for None); bools are refused."""
    try:
        if isinstance(v, bool):
            raise TypeError
        v = operator.index(v)
    except TypeError:
        raise AOIError(f"{name} must be a whole number; got {_show(v)}") from None
    if v < 1 or (hi is not None and v > hi):
        raise AOIError(f"{name} must be at least 1{f' and at most {hi}' if hi else ''}; "
                       f"got {_show(v)}")
    return v


def _read(obj):
    """Any accepted AOI input -> one (unvalidated) shapely geometry."""
    if isinstance(obj, (str, bytes)):
        try:
            obj = json.loads(obj)
        except (ValueError, RecursionError):
            raise AOIError("AOI is not valid JSON: send GeoJSON or "
                           "[west, south, east, north]") from None
    if isinstance(obj, BaseGeometry):
        return obj
    if isinstance(obj, (list, tuple)):
        try:
            w, s, e, n = (float(v) for v in obj)
            ok = all(map(math.isfinite, (w, s, e, n)))
        except (TypeError, ValueError, OverflowError):
            ok = False
        if not ok:
            raise AOIError("a bbox must be four numbers [west, south, east, north]; "
                           f"got {_show(obj)}")
        if not (w < e and s < n):
            raise AOIError("a bbox must be [west, south, east, north] with west < "
                           f"east and south < north; got {_show(list(obj))}")
        return box(w, s, e, n)
    if not isinstance(obj, dict):
        raise AOIError(f"unsupported AOI of type {type(obj).__name__}: send GeoJSON "
                       "or [west, south, east, north]")
    # GeoJSON nests only FeatureCollection > Feature > geometry (RFC 7946), so
    # deeper input is refused here instead of recursing into it
    if obj.get("type") == "FeatureCollection":
        feats = obj.get("features") or []
        if not isinstance(feats, list) or not all(
                isinstance(f, dict) and f.get("type") != "FeatureCollection"
                for f in feats):
            raise AOIError("a FeatureCollection's features must be a list of Features")
        if len(feats) > MAX_AOI_PARTS:
            raise AOIError(f"the AOI has {len(feats):,} features; at most "
                           f"{MAX_AOI_PARTS:,} (URBANSEG_MAX_AOI_PARTS)")
        return shapely.GeometryCollection([_read(f) for f in feats])
    if obj.get("type") == "Feature":
        obj = obj.get("geometry")
        if not obj:
            raise AOIError("the AOI feature has no geometry")
        kind = obj.get("type") if isinstance(obj, dict) else None
        if kind is None or kind in ("Feature", "FeatureCollection"):
            raise AOIError("a Feature's geometry must be a GeoJSON geometry")
    try:
        return as_shape(obj)
    except RecursionError:
        raise AOIError("the AOI geometry is nested too deeply") from None
    except Exception as ex:  # noqa: BLE001 - shapely raises several types here
        raise AOIError(f"not a readable GeoJSON geometry: {str(ex)[:120]}") from None


def _geometry(obj):
    """Any accepted AOI input -> a valid 2D (Multi)Polygon with area, CCW exteriors.

    parse_aoi's checks of the input itself, without the request limits on
    coverage and size: aoi_mask and clip_geojson take every AOI through this.
    """
    if obj is None:
        raise AOIError("no AOI given")
    geom = shapely.force_2d(_read(obj))
    # cheap counts first: the costly validation below never sees huge input
    n_coords = shapely.get_num_coordinates(geom)
    if n_coords > MAX_AOI_VERTICES:
        raise AOIError(f"the AOI has {n_coords:,} vertices; at most "
                       f"{MAX_AOI_VERTICES:,} (URBANSEG_MAX_AOI_VERTICES): simplify it")
    parts = []
    for p in _parts(geom):
        parts.append(p)
        if len(parts) > MAX_AOI_PARTS:
            raise AOIError(f"the AOI has more than {MAX_AOI_PARTS:,} parts "
                           "(URBANSEG_MAX_AOI_PARTS): merge or split them")
    other = sorted({p.geom_type for p in parts} - {"Polygon"})
    if other:
        raise AOIError(f"an AOI must be a polygon, not {' / '.join(other)}: draw the "
                       "area as a polygon or rectangle, or give a point as lat/lng "
                       "with a grid size")
    if not parts:
        raise AOIError("the AOI is empty")
    # every vertex, holes included: GEOS bounds skip NaN and look at exteriors
    # only, and repair silently drops such vertices or turns a hole at lat
    # 1000 into a second polygon there
    xy = shapely.get_coordinates(geom)
    if not np.isfinite(xy).all():
        raise AOIError("AOI coordinates must be finite numbers")
    (w, s), (e, n) = xy.min(axis=0), xy.max(axis=0)
    if w < -180 or e > 180 or s < -MAX_LAT or n > MAX_LAT:
        raise AOIError("AOI coordinates must be WGS84 longitude, latitude (within "
                       f"±180, ±{MAX_LAT:g}); got bounds "
                       f"{', '.join(_fmt(v) for v in (w, s, e, n))}")
    if len(parts) == 1:
        geom = parts[0]
    # a valid (Multi)Polygon, e.g. a re-read plan()["aoi"], is kept as it is;
    # anything else is repaired part by part and unioned, so overlapping
    # polygons (several features, an invalid MultiPolygon) add up, not cancel
    if not (geom.geom_type in ("Polygon", "MultiPolygon") and geom.is_valid):
        geom = _polygonal(shapely.GeometryCollection([_repair(p) for p in parts]))
    if geom.is_empty or geom.area == 0:
        raise AOIError("the AOI has no area")
    return shapely.orient_polygons(geom)


def parse_aoi(obj):
    """Validated AOI as a shapely Polygon/MultiPolygon in (lon, lat).

    obj: a GeoJSON geometry, Feature or FeatureCollection (dict or JSON
    text; all polygons are unioned), a [west, south, east, north] bbox, or a
    shapely geometry. Invalid rings are repaired with make_valid (polygonal
    parts kept), Z values dropped, and exteriors oriented counter-clockwise.

    Raises AOIError: "invalid" (unreadable, points/lines, non-finite or
    out-of-range coordinates, no area, under MIN_AOI_M2, over
    MAX_AOI_VERTICES or MAX_AOI_PARTS), "swapped" (only the lat/lng-swapped
    AOI touches config.HYD_BBOX; .hint has its bbox) or "too_large" (over
    MAX_AOI_KM2). The limits come from env URBANSEG_MIN_AOI_M2,
    URBANSEG_MAX_AOI_VERTICES, URBANSEG_MAX_AOI_PARTS and URBANSEG_MAX_AOI_KM2.
    """
    geom = _geometry(obj)
    hyd = _hyd()
    if not geom.intersects(hyd):
        flipped = shapely.transform(geom, lambda c: c[:, [1, 0]])
        if flipped.intersects(hyd):
            raise AOIError("the AOI coordinates look like (latitude, longitude); "
                           "GeoJSON and bboxes are (longitude, latitude)",
                           code="swapped", hint={"bbox_wsen": list(flipped.bounds)})
    area = geodesic_area_m2(geom)
    if area < MIN_AOI_M2:  # e.g. a double-click: billable, but no pixel inside
        raise AOIError(f"the AOI covers only {area:.2g} m2; draw at least "
                       f"{MIN_AOI_M2:g} m2 (URBANSEG_MIN_AOI_M2)")
    if area > MAX_AOI_KM2 * 1e6:
        raise AOIError(f"the AOI covers {area / 1e6:.2f} km2; at most "
                       f"{MAX_AOI_KM2:g} km2 per request (URBANSEG_MAX_AOI_KM2): "
                       "split it into parts",
                       code="too_large",
                       hint={"area_km2": round(area / 1e6, 3), "max_km2": MAX_AOI_KM2})
    return geom


# ---------------------------------------------------------------- planning

def grid_meta(center_lat, center_lng, nx, ny):
    """The meta imagery.fetch_area returns for this grid, without fetching.

    Same arithmetic as fetch_area, so the two are equal (==): a request can
    be georeferenced, priced and rasterised before any tile is bought.
    """
    s = config.SCALE
    ux, uy = config.USABLE_X, config.USABLE_Y
    cx, cy = imagery.latlng_to_world(center_lat, center_lng)
    return {
        "zoom": config.ZOOM, "scale": s,
        "x0": cx - nx * ux / 2.0, "y0": cy - ny * uy / 2.0,
        "width_map_px": nx * ux, "height_map_px": ny * uy,
        "center": [center_lat, center_lng], "nx": nx, "ny": ny,
    }


def _check_point(lat, lng):
    """(lat, lng) as floats once range and swap checks pass."""
    lat, lng = _finite(lat, "latitude"), _finite(lng, "longitude")
    if abs(lat) > MAX_LAT or abs(lng) > 180:
        swap = abs(lng) <= MAX_LAT < abs(lat) <= 180
        raise AOIError(f"{_fmt(lat)}, {_fmt(lng)} is not a location the imagery covers: "
                       f"latitude must be within ±{MAX_LAT:g} and longitude within ±180"
                       + (" (are latitude and longitude swapped?)" if swap else ""))
    if not _in_hyd(lat, lng) and _in_hyd(lng, lat):
        raise AOIError(f"latitude and longitude look swapped: {_fmt(lat)}, {_fmt(lng)} "
                       f"is far from Hyderabad but {_fmt(lng)}, {_fmt(lat)} is inside "
                       f"it. Did you mean {_fmt(lng)}, {_fmt(lat)}?",
                       code="swapped", hint={"lat": lng, "lng": lat})
    return lat, lng


def plan(aoi=None, *, lat=None, lng=None, grid=None, margin_m=DEFAULT_MARGIN_M,
         max_tiles=None, allow_outside=True):
    """The tile grid, cost and checks of a request, before anything is fetched.

    With an AOI (anything parse_aoi accepts) the grid is the smallest nx x ny
    (chosen independently) whose usable extent, nx*640 x ny*600 map px,
    covers the AOI's bbox plus margin_m (0 to MAX_MARGIN_M ground metres) on
    every side, centred on that bbox. With lat/lng it is the classic grid x
    grid square around the point (grid 1-100, default 2; margin_m does not
    apply).

    Returns a JSON-ready dict: nx, ny, calls (nx*ny Static Maps requests),
    center [lat, lng] and meta (== imagery.fetch_area(*center, nx, ny)'s
    meta), extent_m [w, h] and extent_ha (WGS84 ground metres at the centre,
    so extent_ha is the area infer.area_stats counts over the whole extent),
    bbox_wsen, aoi_area_ha (WGS84 geodesic; None for a point), aoi (the
    validated AOI as a GeoJSON geometry; None for a point), in_coverage
    (AOI/point inside config.HYD_BBOX) and warnings (outside coverage, large
    request, AOI near the extent edge).

    Raises AOIError: "too_large" when calls > max_tiles (a whole number >= 1;
    None -> MAX_TILES, env URBANSEG_MAX_TILES, default 9); "outside_coverage"
    when not in coverage and allow_outside is False; parse_aoi's errors for
    the AOI; "invalid"/"swapped" for a bad point or arguments.
    """
    max_tiles = MAX_TILES if max_tiles is None else _whole(max_tiles, "max_tiles")
    geom = None
    if aoi is not None:
        if lat is not None or lng is not None or grid is not None:
            raise AOIError("give either an AOI or lat/lng with a grid size, not both")
        margin = _finite(DEFAULT_MARGIN_M if margin_m is None else margin_m, "margin_m")
        if not 0 <= margin <= MAX_MARGIN_M:
            raise AOIError(f"margin_m must be from 0 to {MAX_MARGIN_M:g} m; "
                           f"got {_show(margin_m)}")
        geom = parse_aoi(aoi)
        w, s, e, n = geom.bounds
        xw, yn = imagery.latlng_to_world(n, w)
        xe, ys = imagery.latlng_to_world(s, e)
        c_lat, c_lng = imagery.world_to_latlng((xw + xe) / 2, (yn + ys) / 2)
        mx, my = _ground_mpp(c_lat)
        # the epsilon keeps float noise from adding a tile to an exact fit
        nx = max(1, math.ceil((xe - xw + 2 * margin / mx) / config.USABLE_X - 1e-9))
        ny = max(1, math.ceil((ys - yn + 2 * margin / my) / config.USABLE_Y - 1e-9))
    elif lat is None or lng is None:
        raise AOIError("give an AOI, or both lat and lng")
    else:
        c_lat, c_lng = _check_point(lat, lng)
        nx = ny = _whole(2 if grid is None else grid, "grid", 100)
        mx, my = _ground_mpp(c_lat)

    calls = nx * ny
    if calls > max_tiles:
        tile = f"{config.USABLE_X * mx:.0f} x {config.USABLE_Y * my:.0f} m"
        if geom is not None:
            what = (f"the AOI spans {(xe - xw) * mx:.0f} x {(ys - yn) * my:.0f} m "
                    f"and with a {margin:g} m margin needs")
            fix = "Draw a smaller AOI or split it into parts."
        else:
            what, fix = f"grid {nx} needs", "Use a smaller grid."
        raise AOIError(f"{what} {nx} x {ny} = {calls} tiles; at most {max_tiles} are "
                       "allowed per request (URBANSEG_MAX_TILES; one tile covers "
                       f"{tile}). {fix}", code="too_large",
                       hint={"nx": nx, "ny": ny, "calls": calls, "max_tiles": max_tiles})

    meta = grid_meta(c_lat, c_lng, nx, ny)
    w_m, h_m = meta["width_map_px"] * mx, meta["height_map_px"] * my
    hyd = _hyd()
    in_cov = hyd.covers(geom) if geom is not None else _in_hyd(c_lat, c_lng)
    warnings = []
    if not in_cov:
        what = "the location is outside"
        if geom is not None:
            what = f"the AOI is {'partly ' if geom.intersects(hyd) else ''}outside"
        msg = (f"{what} greater Hyderabad, where the model was trained: "
               "expect lower accuracy")
        if not allow_outside:
            raise AOIError(msg, code="outside_coverage",
                           hint={"coverage_bbox_wsen": list(config.HYD_BBOX)})
        warnings.append(msg)
    if calls > WARN_TILES:
        warnings.append(f"large request: {nx} x {ny} tiles = {calls} Static Maps calls")
    if geom is not None:
        # map px between the AOI's bbox and the extent edge, per axis
        gx = (meta["width_map_px"] - (xe - xw)) / 2
        gy = (meta["height_map_px"] - (ys - yn)) / 2
        band = EDGE_PX / config.SCALE  # map px
        if min(gx, gy) < band:
            warnings.append(f"the AOI comes within {min(gx * mx, gy * my):.1f} m of the "
                            "image edge, where the model is less reliable; use "
                            f"margin_m >= {math.ceil(band * max(mx, my))}")
    return {
        "nx": nx, "ny": ny, "calls": calls,
        "center": [c_lat, c_lng],
        "extent_m": [round(w_m, 1), round(h_m, 1)],
        "extent_ha": round(w_m * h_m / 1e4, 2),
        "bbox_wsen": list(imagery.bbox_latlng(meta)),
        "aoi_area_ha": None if geom is None else round(geodesic_area_m2(geom) / 1e4, 4),
        "in_coverage": bool(in_cov),
        "warnings": warnings,
        "meta": meta,
        "aoi": None if geom is None else mapping(geom),
    }


# ---------------------------------------------------------------- clipping

def aoi_mask(geom, meta, shape):
    """Bool HxW mask of the image pixels whose centres lie inside the AOI.

    geom: a shapely geometry or anything parse_aoi accepts, checked and
    repaired as parse_aoi does but without its coverage and size limits
    (AOIError for points, lines or empty input); shape: the classmap's shape
    (extra dims ignored). Rasterised in world-map-px space with the Affine
    fusion and to_geojson use, so it lines up exactly with a classmap of
    that meta.
    """
    geom = _geometry(geom)
    shape = tuple(shape[:2])
    s, zoom = meta["scale"], meta["zoom"]
    tr = Affine(1.0 / s, 0, meta["x0"], 0, 1.0 / s, meta["y0"])
    world = shp_transform(lambda x, y: labels._world_arrays(x, y, zoom), geom)
    return rfeatures.rasterize([(world, 1)], out_shape=shape, transform=tr,
                               fill=0, dtype="uint8").astype(bool)


def clip_geojson(fc, geom):
    """A new FeatureCollection of fc's features cut to the AOI.

    geom is taken as aoi_mask takes it. Each feature geometry is intersected
    with the AOI (shapely); features left without area are dropped, as are
    non-polygonal ones. Properties and other members are kept, and "area_m2"
    is set to the WGS84 geodesic area of what remains. Features entirely
    inside keep their coordinates.
    """
    aoi = _geometry(geom)
    shapely.prepare(aoi)
    feats = []
    for f in fc.get("features") or ():
        if not f.get("geometry"):
            continue
        g = _repair(as_shape(f["geometry"]))
        if not aoi.intersects(g):
            continue
        g = _polygonal(g if aoi.covers(g) else g.intersection(aoi))
        if g.is_empty or g.area == 0:
            continue
        out = {k: v for k, v in f.items() if k not in ("geometry", "properties", "bbox")}
        out["properties"] = {**(f.get("properties") or {}),
                             "area_m2": round(geodesic_area_m2(g), 1)}
        out["geometry"] = mapping(g)
        feats.append(out)
    return {"type": "FeatureCollection", "features": feats}


def area_units(m2):
    """An area in m2 in every unit the GUI offers: m2, ha, acres, sq_yd, guntas.

    Rounded to about 1 m2 in each unit (0.1 m2 for m2 and sq_yd).
    """
    m2 = float(m2)
    return {u: round(m2 / f, d) for u, (f, d) in UNITS.items()}


# ---------------------------------------------------------------- locations

LATLNG_HELP = ("e.g. 17.4435, 78.3772 or 17,4435; 78,3772 or "
               "17°26'36.6\"N 78°22'37.9\"E or a Google Maps link")
MAX_LOCATION_CHARS = 4096  # long Google Maps links are ~1-2k characters
_NUM = r"[-+]?\d+(?:\.\d+)?"
# dot decimals separated by a comma, semicolon or spaces; or decimal commas
# separated by a semicolon, spaces or ", ". No two \s runs can meet, so a
# long run of blanks costs linear, not quadratic, backtracking.
_PAIRS = (re.compile(rf"({_NUM})(?:\s*[,;]\s*|\s+)({_NUM})"),
          re.compile(r"([-+]?\d+(?:,\d+)?)(?:\s*;\s*|\s*,\s+|\s+)([-+]?\d+(?:,\d+)?)"))
_URL = re.compile(r"(?i)^(?:https?://|www\.|geo:)|goo\.gl/|google\.[\w.]+/maps|"
                  r"!3d|/@|[?&](?:q|query|ll)=")
_PIN = re.compile(rf"!3d({_NUM})!4d({_NUM})")  # a place link's pin
_VIEW = re.compile(rf"/@({_NUM}),\s*({_NUM})")  # the map's view centre
_GEO = re.compile(rf"geo:({_NUM}),({_NUM})", re.I)
_LABELLED = re.compile(rf"(?:loc:)?\s*({_NUM})\s*,\s*({_NUM})\s*(?:\(.*)?", re.S)
_ROUTE = re.compile(r"(?:^|/)maps/dir(?:/|$)")
_ROUTE_KEYS = ("origin", "saddr", "waypoints", "destination", "daddr")  # route order
# number with its unit symbol | hemisphere | sign | separator | anything else
_DMS_TOKEN = re.compile(
    r"\s*(?:(\d+(?:[.,]\d+)?)\s*([°'\"]?)|([NSEW])|([-+])|([,;])|(\S))")
_TYPOS = str.maketrans({"″": '"', "”": '"', "“": '"', "′": "'", "’": "'", "‘": "'",
                        "´": "'", "`": "'", "º": "°", "˚": "°", "\u00a0": " ", "−": "-"})


def _dms_point(t, text):
    """(lat, lng) from degrees[-minutes[-seconds]] text, N/S/E/W optional."""
    def bad(why=""):
        return AOIError(f"could not read a location from {_show(text)}{why}; "
                        f"{LATLNG_HELP}")

    coords = [{"sign": 1, "hem": None, "vals": []}]

    def start():  # the next token begins a new coordinate
        if coords[-1]["vals"]:
            coords.append({"sign": 1, "hem": None, "vals": []})

    for num, unit, hem, sign, sep, junk in _DMS_TOKEN.findall(t):
        c = coords[-1]
        if junk:
            raise bad()
        if sep:
            start()
        elif sign:
            start()
            coords[-1]["sign"] = -1 if sign == "-" else 1
        elif hem and c["vals"] and c["hem"] is None:  # trailing: 17°26'N
            c["hem"] = hem
            start()
        elif hem:  # leading: N 17°26'
            start()
            if coords[-1]["hem"] is not None:
                raise bad()
            coords[-1]["hem"] = hem
        else:
            # slots 0/1/2 = degrees/minutes/seconds; a unit-less number takes
            # the next one ("17 26 36.6 N"), a repeated slot starts a new value
            last = c["vals"][-1][0] if c["vals"] else -1
            slot = "°'\"".index(unit) if unit else last + 1
            if slot <= last:
                start()
            coords[-1]["vals"].append((slot, float(num.replace(",", "."))))
    coords = [c for c in coords if c["vals"] or c["hem"] or c["sign"] < 0]
    if len(coords) != 2:
        raise bad(f" (expected 2 coordinates, found {len(coords)})")
    out = []
    for c in coords:
        v = dict(c["vals"])
        if not v or min(v) != 0 or max(v) > 2:
            raise bad()
        d, m, s = v[0], v.get(1, 0), v.get(2, 0)
        if m >= 60 or s >= 60:
            raise bad(" (minutes and seconds must be below 60)")
        if (len(v) > 1 and d != int(d)) or (2 in v and m != int(m)):
            raise bad(" (only the last of degrees, minutes, seconds may have decimals)")
        if c["hem"] and c["sign"] < 0:
            raise bad(" (use a minus sign or S/W, not both)")
        deg = d + m / 60 + s / 3600
        neg = c["sign"] < 0 or c["hem"] in ("S", "W")
        axis = {"N": "lat", "S": "lat", "E": "lng", "W": "lng"}.get(c["hem"])
        out.append((round(-deg if neg else deg, 7), axis))
    (v1, a1), (v2, a2) = out
    if a1 is not None and a1 == a2:
        both = "latitudes" if a1 == "lat" else "longitudes"
        raise AOIError(f"{_show(text)} gives two {both}; {LATLNG_HELP}")
    return (v2, v1) if a1 == "lng" or a2 == "lat" else (v1, v2)


def _text_point(t, text):
    t = t.translate(_TYPOS).replace("''", '"').strip("()[]{} \t\r\n")
    if re.fullmatch(r"[-+]?\d+,\d+", t):
        raise AOIError(f"{_show(text)} is ambiguous: one number with a decimal comma, "
                       f"or two whole numbers? Enter both coordinates, {LATLNG_HELP}")
    m = _PAIRS[0].fullmatch(t) or _PAIRS[1].fullmatch(t)
    if m:
        return tuple(float(v.replace(",", ".")) for v in m.groups())
    if re.search(r"[°'\"NSEWnsew]", t):
        return _dms_point(t.upper(), text)
    raise AOIError(f"could not read two coordinates from {_show(text)}: use a dot for "
                   f"decimals, or a semicolon or space between decimal-comma numbers; "
                   f"{LATLNG_HELP}")


def _link_point(text):
    """(lat, lng) when a place in a link is given as coordinates, else None."""
    m = _LABELLED.fullmatch(text)  # also q=lat,lng (label) and q=loc:lat,lng
    if m:
        return float(m[1]), float(m[2])
    try:
        return _text_point(text, text)
    except AOIError:
        return None  # a name, not coordinates


def _route_point(path, q):
    """The one place a directions link names; a route between several is ambiguous.

    path: the URL path after /maps/dir/; q: its lower-cased query parameters.
    """
    stops = []
    for seg in path.split("/"):
        if seg.startswith("@") or "=" in seg:  # the view centre, then data=...
            break
        stops.append(unquote_plus(seg))
    for k in _ROUTE_KEYS:
        stops += q.get(k, "").split("|")  # waypoints=a|b
    stops = [s.strip() for s in stops if s.strip()]  # an empty stop is left open
    points = [_link_point(s) for s in stops]
    if len(stops) == 1 and points[0]:
        return points[0]
    if len(stops) == 1:
        raise AOIError(f"this directions link names {_show(stops[0])} but no coordinates "
                       "for it: open the place in Google Maps and copy its link, or "
                       "right-click its pin and copy the 'lat, lng'")
    if not stops:
        raise AOIError(f"this directions link names no place; {LATLNG_HELP}")
    names = [f"{_fmt(p[0])}, {_fmt(p[1])}" if p else _show(s)
             for s, p in zip(stops, points)]
    raise AOIError(f"this directions link names {len(stops)} places "
                   f"({' to '.join(names[:4])}{' ...' if len(names) > 4 else ''}): "
                   f"enter the one to analyse on its own, {LATLNG_HELP}")


def _url_point(t):
    try:
        url = urlsplit(t)
    except ValueError:  # e.g. an unbalanced [ read as an IPv6 host
        raise AOIError(f"could not read this link; {LATLNG_HELP}") from None
    q = {}
    for k, v in parse_qsl(url.query):
        q.setdefault(k.lower(), v.strip())
    route = _ROUTE.search(url.path)
    if route or any(k in q for k in _ROUTE_KEYS):
        return _route_point(url.path[route.end():] if route else "", q)
    m = _PIN.search(unquote_plus(t))
    if m:
        return float(m[1]), float(m[2])
    if "maps.apple.com" in t.lower():  # Apple puts the place in ll= and a label in q=
        m = _LABELLED.fullmatch(q.get("ll") or q.get("coordinate") or "")
        if m:
            return float(m[1]), float(m[2])
    # the place as the link names it: /place/<x>, /search/<x>, q=<x>, query=<x>
    m = re.search(r"/(?:place|search)/([^/]+)", url.path)
    named = ([unquote_plus(m[1]).strip()] if m else []) + [
        q[k] for k in ("q", "query") if q.get(k)]
    for x in named:
        p = _link_point(x)
        if p:
            return p
    if named:  # by name only: the view centre can be km away, with the place off-centre
        raise AOIError(f"this link names {_show(named[0])} but no coordinates for it "
                       "(a link's map centre is not the place): open the place in "
                       "Google Maps and copy the link from the address bar, or "
                       "right-click its pin and copy the 'lat, lng'")
    # a bare map view (/@, the legacy ll=, center=) or geo: URI: where it looks
    m = (_VIEW.search(url.path)
         or _LABELLED.fullmatch(q.get("ll") or q.get("center") or "") or _GEO.match(t))
    if m:
        return float(m[1]), float(m[2])
    if "goo.gl" in t.lower():
        raise AOIError("short Google Maps links (maps.app.goo.gl/...) do not contain "
                       "the coordinates: open the link, then right-click the pin and "
                       "copy the 'lat, lng' line instead")
    raise AOIError(f"no coordinates found in this link; {LATLNG_HELP}")


def parse_latlng(text):
    """A typed or pasted location -> (lat, lng) in degrees.

    Accepts "17.4435, 78.3772" (comma, semicolon or space between), decimal
    commas ("17,4435 78,3772", "17,4435; 78,3772"), degrees-minutes-seconds
    such as 17°26'36.6"N 78°22'37.9"E (spaces, primes or '' allowed, N/S/E/W
    before or after, either order) and map links: the !3d..!4d.. pin, or
    coordinates in /place/, /search/, q= or query=; the one place of a
    directions link (/maps/dir/, origin= / destination=, saddr= / daddr=);
    for a link that names no place, the view centre (/@, ll=, center=,
    geo:); Apple Maps' ll=, which is its place.

    Raises AOIError: "invalid" for unreadable or ambiguous text and
    impossible coordinates. Ambiguous are a lone "17,4435" (one number or
    two?), a link that names a place or search without its coordinates (its
    view centre can be far from the place) and a directions link between
    several places. "swapped" when only the reversed pair lies in
    config.HYD_BBOX (the message has the corrected pair and
    .hint = {"lat", "lng"}).
    """
    if not isinstance(text, str) or not text.strip():
        raise AOIError(f"enter a location, {LATLNG_HELP}")
    if len(text) > MAX_LOCATION_CHARS:
        raise AOIError(f"the location text is too long ({len(text)} characters)")
    t = text.strip()
    lat, lng = _url_point(t) if _URL.search(t) else _text_point(t, t)
    return _check_point(lat, lng)
