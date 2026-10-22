"""OpenStreetMap data via Overpass: buildings, landuse, roads, water, POIs.

Two query/parser versions live side by side:
- v2 (QUERY, parse_overpass(data), get_area_osm) is frozen. The v1 training
  labels were rasterised from data/osm/<area>_v2.json through exactly this
  code, so it must keep returning the same layers.
- v3 (QUERY_V3, parse_overpass(data, version="v3")) adds waterways and the
  parser fixes below. Live fusion (get_live_osm) asks QUERY_V4, which is
  QUERY_V3 with the water relations' geometry, and parses it as v3; its
  cache is validated and expires.

Every query before QUERY_V4 ends in "out tags geom;", which gives relations
without their members: multipolygon lakes and reservoirs (Hussain Sagar,
Himayat Sagar) arrive as bare tags and never become water. The frozen
snapshots keep that gap; QUERY_WATER_RELATIONS fetches the missing
geometry for later label versions.
"""
import hashlib
import json
import os
import queue
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

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

# v3 = v2 + waterway lines (rivers, canals, nalas): the parser and the label
# code handle them, but v2 never asked for them. Formatted like QUERY; the
# live path lowers the server timeout per attempt (_live_query).
QUERY_V3 = """
[out:json][timeout:300];
(
  way["building"]({s},{w},{n},{e});
  way["landuse"]({s},{w},{n},{e});
  way["highway"]({s},{w},{n},{e});
  way["railway"]({s},{w},{n},{e});
  way["waterway"~"^(river|canal|stream|drain|ditch)$"]({s},{w},{n},{e});
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

# the water relations of QUERY / QUERY_V3, with members: "out tags" (above)
# leaves them out, "out body geom" sends each member way's full geometry
_WATER_RELATIONS = """(
  relation["natural"="water"]({s},{w},{n},{e});
  relation["natural"="wetland"]({s},{w},{n},{e});
  relation["landuse"~"reservoir|basin"]({s},{w},{n},{e});
);
out body geom;
"""
QUERY_WATER_RELATIONS = "[out:json][timeout:300];\n" + _WATER_RELATIONS
# v4 = v3 with the water relations' geometry (a second output statement)
QUERY_V4 = QUERY_V3 + _WATER_RELATIONS

LIVE_VERSION = "v4"   # the live query: cache key and envelope "query_version"
LIVE_PARSER = "v3"    # how its answers are parsed
# per mirror, and never more than the time left: hedged attempts run side by
# side, so a slow mirror (mail.ru took 12-25 s for one query) may use most of it
LIVE_ATTEMPT_S = float(os.environ.get("URBANSEG_OSM_ATTEMPT_S", "50"))
LIVE_CONNECT_S = 5  # within the attempt: a mirror that needs longer to connect is down
# ask the next mirror too when none has answered after this long (real s)
LIVE_HEDGE_S = float(os.environ.get("URBANSEG_OSM_HEDGE_S", "10"))
# Overpass reports a query it had to abort (timeout, out of memory) with
# HTTP 200 and a "remark"; the elements are then incomplete
_FAILED_REMARK = re.compile(r"(?i)error|timed? ?out|out of memory")
_clock = time.monotonic  # tests substitute a fake clock


class OverpassError(RuntimeError):
    """Live OSM could not be fetched; .reason is one of REASONS.

    The message names the mirrors tried and why each failed. It holds no
    exception text (a proxy URL may carry credentials) and no response data
    beyond Overpass's own short remark.
    """
    REASONS = ("timeout", "unavailable", "partial", "bad_response")

    def __init__(self, reason, detail=""):
        if reason not in self.REASONS:
            raise ValueError(f"unknown OverpassError reason {reason!r}")
        self.reason, self.detail = reason, detail
        super().__init__(f"overpass {reason}" + (f": {detail}" if detail else ""))

    def __reduce__(self):  # the default would re-call __init__ with the message
        return type(self), (self.reason, self.detail)


def _write_atomic(path, text):
    """Write via a temp file and os.replace, so a crash never leaves a partial file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(text)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _snapshot_answer(r):
    """A fetch_overpass response -> (answer, None) or (None, why it is unusable)."""
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    data = r.json()
    remark = data.get("remark") or ""
    if _FAILED_REMARK.search(remark):
        return None, f"partial: {remark[:100]}"
    return data, None


def fetch_overpass(bbox, cache_path: Path | None = None, timeout=360, rounds=2,
                   pause=10, hedge_s=None, query=QUERY):
    """bbox = (west, south, east, north). Returns overpass JSON dict.

    Tries each mirror `rounds` times, `timeout` s per attempt. This is the v2
    path: an existing cache file is returned as is, because it may be a
    training-label snapshot that must never be replaced by a newer answer.
    An answer Overpass aborted (HTTP 200 with an error remark) is never used.

    query: QUERY (the default) or QUERY_WATER_RELATIONS.

    hedge_s=None asks the mirrors in turn. Otherwise each round asks the next
    mirror too as soon as an attempt fails or hedge_s s pass without an
    answer, and the first answer with elements wins; an empty one is used
    only when no mirror of the round has more (as in _fetch_live).
    """
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    w, s, e, n = bbox
    q = query.format(w=w, s=s, e=e, n=n)

    def keep(data):
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(cache_path, json.dumps(data))
        return data

    last_err = None
    if hedge_s is None:
        for url in OVERPASS_URLS * rounds:
            try:
                r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=timeout)
                data, last_err = _snapshot_answer(r)
                if data is not None:
                    return keep(data)
            except Exception as ex:  # noqa: BLE001
                last_err = str(ex)
            time.sleep(pause)
        raise RuntimeError(f"overpass failed for {bbox}: {last_err}")

    results = queue.Queue()

    def ask(url):
        try:
            r = requests.post(url, data={"data": q}, headers=HEADERS, timeout=timeout)
            results.put((url, *_snapshot_answer(r)))
        except Exception as ex:  # noqa: BLE001
            results.put((url, None, str(ex)))

    errors = []
    for _ in range(rounds):
        todo, pending, empty = list(OVERPASS_URLS), 0, None
        while todo or pending:
            if todo:
                threading.Thread(target=ask, args=(todo.pop(0),), daemon=True,
                                 name="overpass").start()
                pending += 1
            try:  # past timeout (+ slack for a slow body) an attempt is given up
                url, data, err = results.get(timeout=hedge_s if todo else timeout + 30)
            except queue.Empty:
                if todo:
                    continue
                errors.append("no answer in time")
                break
            pending -= 1
            if data is not None and data.get("elements"):
                return keep(data)
            if data is not None:
                empty = empty or data
            else:
                errors.append(f"{urlparse(url).hostname}: {err}")
        if empty is not None:
            return keep(empty)
        time.sleep(pause)
    raise RuntimeError(f"overpass failed for {bbox}: {'; '.join(errors)}")


def _way_geom(el, as_line=False):
    pts = [(g["lon"], g["lat"]) for g in el.get("geometry", [])]
    if len(pts) < 2:
        return None
    closed = pts[0] == pts[-1]
    if closed and len(pts) >= 4 and not as_line:
        try:
            poly = Polygon(pts)
            return poly if poly.is_valid else poly.buffer(0)
        except Exception:  # noqa: BLE001
            return None
    return LineString(pts)


WATER_WAY_TAGS = {"river", "canal", "stream", "drain", "ditch"}
PARSER_VERSIONS = ("v2", "v3")
# highway values that are areas when closed, even without area=yes
AREA_HIGHWAYS = {"services", "rest_area"}
# a closed way with one of these keys is an area whatever else it carries
_AREA_KEYS = ("building", "landuse", "natural", "leisure")


def _is_water_poly(tags):
    return (tags.get("natural") in ("water", "wetland")
            or tags.get("landuse") in ("reservoir", "basin"))


def _linear_v3(tags):
    """v3: a way that stays a line when closed (roundabout, loop road, ring drain)."""
    if tags.get("area") == "yes" or any(k in tags for k in _AREA_KEYS):
        return False
    if "highway" in tags:
        return tags["highway"] not in AREA_HIGHWAYS
    return tags.get("waterway") in WATER_WAY_TAGS


# v3: water relation ways whose ends miss each other by less than this are
# joined (Himayat Sagar's outline has two 0.5 m gaps in OSM)
RING_SNAP_DEG = 2e-5  # ~2 m


def _snap_ends(lines, tol=RING_SNAP_DEG):
    """lines with every end moved onto the first end within tol of it."""
    reps = []
    def rep(pt):
        for q in reps:
            if abs(q[0] - pt[0]) < tol and abs(q[1] - pt[1]) < tol:
                return q
        reps.append(pt)
        return pt
    out = []
    for line in lines:
        cs = list(line.coords)
        cs[0], cs[-1] = rep(cs[0]), rep(cs[-1])
        out.append(LineString(cs))
    return out


def _relation_water_v3(el):
    """A water multipolygon's polygons: outer member ways noded together
    (one closed way, or several that join, also across gaps < RING_SNAP_DEG)
    and polygonized, minus the inner rings (islands)."""
    from shapely.ops import polygonize, unary_union

    def faces(roles):
        lines = []
        for m in el.get("members", []):
            if m.get("type") == "way" and m.get("role", "") in roles and m.get("geometry"):
                pts = [(g["lon"], g["lat"]) for g in m["geometry"]]
                if len(pts) >= 2:
                    lines.append(LineString(pts))
        return list(polygonize(unary_union(_snap_ends(lines)))) if lines else []

    islands = unary_union(faces(("inner",)))
    out = []
    for poly in faces(("outer", "")):
        if not islands.is_empty:
            poly = poly.difference(islands)
        out += [g for g in getattr(poly, "geoms", [poly]) if not g.is_empty]
    return out


def parse_overpass(data, version="v2"):
    """Split overpass JSON into typed GeoDataFrames (EPSG:4326).

    version="v2" (the default) is frozen: the v1 training labels were built
    from its output. version="v3" fixes three v2 gaps: closed highway ways
    (roundabouts, loop roads) stay LineStrings and reach "roads" unless they
    are areas (area=yes, services), where v2 made them polygons and dropped
    them; water/wetland/reservoir/basin polygons are tested before landuse,
    so a landuse=reservoir way is water, not landuse; closed waterway lines
    stay lines; water relations (only answers of QUERY_V4 or
    QUERY_WATER_RELATIONS carry their members) are assembled by
    _relation_water_v3, where v2 dropped any relation whose outline was a
    single way. Both versions return the same keys and columns.
    """
    if version not in PARSER_VERSIONS:
        raise ValueError(f"unknown OSM parser version {version!r}")
    v3 = version == "v3"
    buildings, landuse, roads, rails, water, greens, pois = [], [], [], [], [], [], []
    waterways = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "relation":
            if _is_water_poly(tags) and v3:
                water += [{"geometry": g} for g in _relation_water_v3(el)]
            elif _is_water_poly(tags):
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
        geom = _way_geom(el, as_line=v3 and _linear_v3(tags))
        if geom is None:
            continue
        if "building" in tags and geom.geom_type == "Polygon":
            buildings.append({"geometry": geom, "building": tags["building"],
                              "levels": tags.get("building:levels", "")})
        elif v3 and _is_water_poly(tags) and geom.geom_type == "Polygon":
            water.append({"geometry": geom})
        elif "landuse" in tags and geom.geom_type == "Polygon":
            landuse.append({"geometry": geom, "landuse": tags["landuse"]})
        elif "highway" in tags and geom.geom_type == "LineString":
            roads.append({"geometry": geom, "highway": tags["highway"]})
        elif "railway" in tags and geom.geom_type == "LineString":
            rails.append({"geometry": geom})
        elif _is_water_poly(tags) and geom.geom_type == "Polygon":  # v2 order
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
    # v2: query includes water relations, wetlands and reservoirs; waterway
    # lines are parsed but only QUERY_V3 asks for them
    cache = config.OSM_DIR / f"{name}_v2.json"
    return parse_overpass(fetch_overpass(bbox, cache_path=cache, **fetch_kw))


def _live_bbox(bbox):
    """bbox rounded to 6 decimals (~0.1 m): the identity of a live query."""
    return [round(float(v), 6) for v in bbox]


def live_cache_path(bbox, cache_dir=None):
    """<cache_dir>/live4_<16 hex of sha1("v4" + bbox rounded to 6 decimals)>.json.

    cache_dir defaults to $URBANSEG_OSM_CACHE, else <repo>/.cache/osm_live:
    a runtime cache stays out of data/, which holds the read-only label
    snapshots.
    """
    if cache_dir is None:
        env = os.environ.get("URBANSEG_OSM_CACHE")
        cache_dir = Path(env) if env else config.ROOT / ".cache" / "osm_live"
    key = LIVE_VERSION + ",".join(f"{v:.6f}" for v in _live_bbox(bbox))
    return Path(cache_dir) / f"live{LIVE_VERSION[1:]}_{hashlib.sha1(key.encode()).hexdigest()[:16]}.json"


def _live_query(box, read_timeout):
    """QUERY_V4 for box, with a server timeout below the client's read timeout,
    so that an overloaded server answers with a remark instead of keeping the
    client waiting."""
    w, s, e, n = (f"{v:.6f}" for v in box)
    server = max(1, int(0.8 * read_timeout))
    return re.sub(r"\[timeout:\d+\]", f"[timeout:{server}]",
                  QUERY_V4.format(w=w, s=s, e=e, n=n), count=1)


def _answer_problem(data):
    """None for a complete Overpass answer, else (reason, detail)."""
    if not isinstance(data, dict) or not isinstance(data.get("elements"), list):
        return "bad_response", "no elements in the answer"
    remark = " ".join(str(data.get("remark") or "").split())
    if _FAILED_REMARK.search(remark):
        return "partial", remark[:160]
    return None


def _parse_live(data):
    """(v3 layers, None) for a complete answer that parses, else (None, (reason, detail))."""
    problem = _answer_problem(data)
    if problem is None:
        try:
            return parse_overpass(data, version=LIVE_PARSER), None
        except Exception as ex:  # noqa: BLE001 - elements no Overpass sends: a broken mirror
            problem = ("bad_response", f"unparsable answer ({type(ex).__name__})")
    return None, problem


def _read_live_cache(path, box, ttl_days):
    """(envelope, layers) when the cache file is intact, for this query version
    and bbox, no older than ttl_days (None = no expiry) and not from the
    future, and holds a complete, non-empty answer that parses; else None."""
    try:
        env = json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):  # missing, unreadable or truncated
        return None
    if (not isinstance(env, dict) or env.get("query_version") != LIVE_VERSION
            or env.get("bbox") != box):
        return None
    try:
        age = datetime.now(timezone.utc) - datetime.fromisoformat(env["fetched_at"])
    except (KeyError, TypeError, ValueError):
        return None
    if age < timedelta(0) or (ttl_days is not None and age > timedelta(days=ttl_days)):
        return None
    # parsed here: a file written by other code may hold elements this parser rejects
    layers, problem = _parse_live(env.get("data"))
    if problem is not None or not env["data"]["elements"]:
        return None
    return env, layers


def _shutdown(response):
    """End a read of response blocked in another thread (urllib3 >= 2.3)."""
    try:
        response.raw.shutdown()
    except Exception:  # noqa: BLE001 - the read ended meanwhile, or an older urllib3
        pass


def _overpass_attempt(sess, url, query, timeout, deadline):
    """POST the query to one mirror -> (answer, None) or (None, (reason, detail)).

    timeout = (connect, read) s. A read timeout bounds each read, not the
    whole answer, so the deadline ends a slow answer: it is checked after
    every chunk, and a timer shuts the socket under a read still blocked
    then. An answer that is complete at the deadline is used all the same.
    """
    start, body = _clock(), bytearray()
    try:
        with sess.post(url, data={"data": query}, headers=HEADERS, stream=True,
                       timeout=timeout) as r:
            if r.status_code != 200:
                return None, ("unavailable", f"HTTP {r.status_code}")
            timer = threading.Timer(max(0.0, deadline - _clock()), _shutdown, (r,))
            timer.daemon = True
            timer.start()
            try:
                for chunk in r.iter_content(1 << 16):
                    body += chunk
                    if _clock() > deadline:
                        break
            finally:
                timer.cancel()
    except (requests.RequestException, OSError) as ex:
        if _clock() <= deadline:  # past it, the timer ended the read: the body decides
            # requests reports a read timeout while streaming as a ConnectionError
            if isinstance(ex, requests.Timeout) or _clock() - start >= timeout[1]:
                return None, ("timeout", f"no answer within {sum(timeout):.0f} s")
            return None, ("unavailable", type(ex).__name__)
    try:
        return json.loads(body), None  # an answer cut short never decodes
    except (ValueError, RecursionError):
        if _clock() > deadline:
            return None, ("timeout", "answer incomplete at the deadline")
        return None, ("bad_response", "answer is not JSON")


def _fetch_live(box, deadline):
    """The first usable v3 answer from the mirrors -> (answer, layers, url).

    Mirrors are hedged rather than queued: the first starts at once, the next
    one as soon as an attempt fails or, while none has answered, after
    LIVE_HEDGE_S of real time. Public instances are often overloaded (on
    2026-10-22 one answered a tiny query in 20 s, three not at all), and
    strictly in turn the first dead one used 30 of the 45 s. The first
    complete answer with elements wins; the others are abandoned. An answer
    without elements is used only when no mirror has more: for a bbox in
    Hyderabad it rather means a lagging or broken mirror.
    """
    results = queue.Queue()

    def attempt(url):
        left = deadline - _clock()
        t = min(LIVE_ATTEMPT_S, left)
        connect = min(LIVE_CONNECT_S, t / 3)  # a share of the attempt, not extra
        layers = None
        with requests.Session() as sess:
            data, failure = _overpass_attempt(sess, url, _live_query(box, t - connect),
                                              (connect, t - connect), deadline)
        if failure is None:
            layers, failure = _parse_live(data)
        results.put((url, data, layers, failure))

    failures, reason, empty = [], "timeout", None
    started = pending = 0
    hedge_at = time.monotonic()  # real time: _clock may be a test's fake
    while True:
        if (started < len(OVERPASS_URLS) and deadline - _clock() >= 1
                and (pending == 0 or time.monotonic() >= hedge_at)):
            threading.Thread(target=attempt, args=(OVERPASS_URLS[started],), daemon=True,
                             name="overpass").start()
            started, pending = started + 1, pending + 1
            hedge_at = time.monotonic() + LIVE_HEDGE_S
        if pending == 0:
            break
        try:
            url, data, layers, failure = results.get(timeout=0.05)
        except queue.Empty:
            continue
        pending -= 1
        host = urlparse(url).hostname
        if failure is None and data["elements"]:
            return data, layers, url
        if failure is None:
            empty = empty or (data, layers, url)
            failures.append(f"{host}: no elements")
        else:
            reason = failure[0]
            failures.append(f"{host}: {reason} ({failure[1]})")
    if empty is not None:
        return empty
    if len(failures) < len(OVERPASS_URLS):
        reason = "timeout"
        failures.append("deadline reached")
    raise OverpassError(reason, "; ".join(failures))


def get_live_osm(bbox, *, ttl_days=30, deadline_s=45, cache_dir=None):
    """Live OSM layers for bbox = (west, south, east, north) -> (layers, info).

    layers = parse_overpass(answer, version="v3"). The answer comes from
    live_cache_path(bbox, cache_dir) when that file decodes, matches the
    query version and the rounded bbox, is no older than ttl_days (None =
    no expiry) and holds a complete, non-empty answer that parses. Otherwise
    the mirrors are asked, hedged (see _fetch_live), until deadline_s is used
    up: each attempt gets min(LIVE_ATTEMPT_S, time left) to connect and start
    answering, and the deadline also ends an answer still arriving. A partial (error remark),
    unparsable or empty answer moves on to the next mirror. Only a complete,
    non-empty answer is cached, as an envelope {"query_version", "bbox",
    "fetched_at", "osm_base", "mirror", "data"} written atomically (a failed
    write costs the cache, not the answer); an empty one is returned, not
    cached, when no mirror has more.

    info = {"source": "cache"|"overpass", "mirror", "osm_base" (Overpass
    timestamp_osm_base), "fetched_at" (UTC ISO 8601), "elapsed_s",
    "cache_file" (None when not cached), "query_version": "v4", "elements"
    (how many the answer holds; 0 = no OSM data)}. Raises OverpassError
    with .reason "timeout", "unavailable", "partial" (the server aborted
    the query) or "bad_response" (not JSON, or elements the parser rejects).
    """
    t0 = _clock()
    box = _live_bbox(bbox)
    path = live_cache_path(box, cache_dir)
    hit = _read_live_cache(path, box, ttl_days)
    if hit is not None:
        (env, layers), source = hit, "cache"
    else:
        data, layers, mirror = _fetch_live(box, t0 + deadline_s)
        osm3s = data.get("osm3s")
        env = {"query_version": LIVE_VERSION, "bbox": box,
               "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "osm_base": (osm3s.get("timestamp_osm_base")
                            if isinstance(osm3s, dict) else None),
               "mirror": mirror, "data": data}
        source = "overpass"
        if not data["elements"]:  # never cached: the next call asks again
            path = None
        else:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                _write_atomic(path, json.dumps(env))
            except OSError:
                path = None
    info = {"source": source, "mirror": env.get("mirror"),
            "osm_base": env.get("osm_base"), "fetched_at": env.get("fetched_at"),
            "elapsed_s": round(_clock() - t0, 3),
            "cache_file": None if path is None else str(path),
            "query_version": LIVE_VERSION, "elements": len(env["data"]["elements"])}
    return layers, info
