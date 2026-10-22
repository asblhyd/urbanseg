"""labels_v2: per-building roof alignment and the v2 label rules."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import LineString, box

import helpers
from urbanseg import aoi, labels_v2, open_buildings, osm_data

IGNORE = labels_v2.IGNORE


def _scene(lean=(4, -6), tall=((10, -12),), n=(5, 6), seed=0):
    """Bright textured roofs on noisy ground; footprints offset from the roofs.

    lean (dx, dy) px is where each roof sits relative to its footprint (the
    shift align_footprints must find); the first len(tall) buildings lean more.
    """
    rng = np.random.default_rng(seed)
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    H, W = 1200, 1280
    img = rng.normal(100, 14, (H, W, 3)).clip(0, 255)
    rows = []
    k = 0
    for r in range(n[0]):
        for c in range(n[1]):
            y0, x0 = 120 + r * 190, 120 + c * 180
            h, w = 40 + 8 * (k % 3), 34 + 6 * (k % 4)
            dx, dy = tall[k] if k < len(tall) else lean
            ry0, rx0 = y0 + dy, x0 + dx
            roof = rng.normal(205, 10, (h, w, 3))
            img[ry0:ry0 + h, rx0:rx0 + w] = roof.clip(0, 255)
            rows.append({"geometry": labels_v2._from_px(box(x0, y0, x0 + w, y0 + h), meta),
                         "true_dx": dx, "true_dy": dy})
            k += 1
    fp = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    return img.astype(np.uint8), meta, fp


def test_alignment_recovers_a_common_lean_and_taller_buildings():
    img, meta, fp = _scene()
    out = labels_v2.align_footprints(img, meta, fp)
    err = np.hypot(out.shift_dx_px - out.true_dx, out.shift_dy_px - out.true_dy)
    assert (err <= 1).mean() >= 0.95, out[["shift_dx_px", "shift_dy_px", "true_dx", "true_dy"]]
    assert err[0] <= 1  # the tall one keeps its own, larger lean
    # the moved outline now covers its roof
    moved = labels_v2._to_px(out.geometry[5], meta)
    y, x = int(moved.centroid.y), int(moved.centroid.x)
    assert img[y, x].mean() > 170


def test_alignment_is_shift_consistent():
    # moving every input footprint by d must change the found shift by -d
    img, meta, fp = _scene(lean=(3, -5), tall=())
    base = labels_v2.align_footprints(img, meta, fp)
    d = (6, 4)
    moved = fp.copy()
    moved["geometry"] = [labels_v2._from_px(
        labels_v2.affinity.translate(labels_v2._to_px(g, meta), *d), meta) for g in fp.geometry]
    again = labels_v2.align_footprints(img, meta, moved)
    ddx = again.shift_dx_px.values - base.shift_dx_px.values
    ddy = again.shift_dy_px.values - base.shift_dy_px.values
    assert (np.hypot(ddx + d[0], ddy + d[1]) <= 1).mean() >= 0.95


def test_weak_evidence_falls_back_to_the_local_field():
    img, meta, fp = _scene(lean=(4, -6), tall=())
    # a footprint over featureless ground: no roof edges to find
    lone = labels_v2._from_px(box(1180, 1090, 1220, 1130), meta)
    fp = gpd.GeoDataFrame(list(fp.to_dict("records")) + [{"geometry": lone, "true_dx": 4, "true_dy": -6}],
                          geometry="geometry", crs=4326)
    img[1050:1200, 1150:1280] = 100  # flat
    out = labels_v2.align_footprints(img, meta, fp)
    assert out.shift_source.iloc[-1] == "field"
    assert (out.shift_dx_px.iloc[-1], out.shift_dy_px.iloc[-1]) == pytest.approx((4, -6), abs=1)


def test_empty_and_border_footprints():
    img, meta, fp = _scene()
    empty = labels_v2.align_footprints(img, meta, fp.iloc[:0])
    assert len(empty) == 0 and "shift_dx_px" in empty
    edge = labels_v2._from_px(box(0, 0, 30, 30), meta)  # cannot be searched
    one = gpd.GeoDataFrame([{"geometry": edge}], geometry="geometry", crs=4326)
    out = labels_v2.align_footprints(img, meta, one)
    assert out.shift_source.iloc[0] == "none" and out.shift_dx_px.iloc[0] == 0


def _empty_layers():
    return osm_data.parse_overpass({}, version="v3")


def test_mask_rules_on_a_synthetic_scene():
    img, meta, fp = _scene(lean=(0, 0), tall=())
    fp = fp.drop(columns=["true_dx", "true_dy"])
    fp["area_in_meters"], fp["confidence"] = 120.0, 0.9
    mask, aligned = labels_v2.build_mask_v2(img, meta, _empty_layers(), fp)
    assert mask.dtype == np.uint8 and set(np.unique(mask)) <= {1, 2, 3, IGNORE}
    # buildings are not enlarged: label building area ~ outline area (minus edge band)
    outline_px = sum(labels_v2._to_px(g, meta).area for g in aligned.geometry)
    bld = np.isin(mask, (1, 2)).sum()
    band = (mask == IGNORE).sum()
    assert 0.6 * outline_px < bld < outline_px and bld + band > outline_px
    # the band straddles each edge: 2 px inside and outside
    g = labels_v2._to_px(aligned.geometry[0], meta)
    x0, y0, x1, y1 = (int(round(v)) for v in g.bounds)
    ym = (y0 + y1) // 2
    assert mask[ym, x0 - 3] == 3 and mask[ym, x0 - 1] == IGNORE and mask[ym, x0 + 1] == IGNORE
    assert mask[ym, x0 + 3] in (1, 2)


def test_gap_between_touching_buildings_is_closed_not_the_outer_edge():
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    img = np.full((1200, 1280, 3), 100, np.uint8)
    a = labels_v2._from_px(box(400, 400, 460, 460), meta)
    b = labels_v2._from_px(box(463, 400, 523, 460), meta)  # 3 px (0.9 m) party-wall gap
    fp = gpd.GeoDataFrame([{"geometry": a}, {"geometry": b}], geometry="geometry", crs=4326)
    fp["area_in_meters"], fp["confidence"] = 300.0, 0.9
    mask, _ = labels_v2.build_mask_v2(img, meta, _empty_layers(), fp)
    assert np.isin(mask[430, 460:463], (1, 2)).all()  # the gap is building
    assert mask[430, 395] == 3                         # outside stays free land


@pytest.mark.data
def test_real_area_crop_alignment_moves_footprints_north_like_the_diagnosis():
    # kondapur: the diagnosis measured the footprints ~2.6 m south of the roofs
    # (per-area median shift (2, -9) px)
    img, _, meta = helpers.load_area("kondapur")
    img, _, m = helpers.crop(img, None, meta, 600, 1800, 1300, 2500)
    from urbanseg import imagery
    fp = open_buildings.load_for_bbox(imagery.bbox_latlng(m))
    assert len(fp) > 200
    out, det = labels_v2.align_footprints(img, m, fp, return_details=True)
    # only footprints inside the crop have a score map (load_for_bbox pads ~220 m)
    inside = np.isfinite(det["raw"][:, 2])
    own = (out.shift_source == "own").values
    assert inside.sum() > 150 and own[inside].mean() > 0.3
    assert np.median(out.shift_dy_px.values[inside]) < -3  # northward


def _footprints(*px_boxes, meta):
    fp = gpd.GeoDataFrame([{"geometry": labels_v2._from_px(box(*b), meta)} for b in px_boxes],
                          geometry="geometry", crs=4326)
    fp["area_in_meters"], fp["confidence"] = 300.0, 0.9
    return fp


def test_strip_swept_to_the_roof_is_ignore_not_free_land():
    img, meta, fp = _scene(lean=(0, -10), tall=(), n=(2, 2))
    fp = fp.drop(columns=["true_dx", "true_dy"])
    fp["area_in_meters"], fp["confidence"] = 120.0, 0.9
    mask, aligned = labels_v2.build_mask_v2(img, meta, _empty_layers(), fp)
    r = aligned.iloc[0]
    assert (r.shift_dx_px, r.shift_dy_px) == pytest.approx((0, -10), abs=1)
    x0, _, x1, roof_bottom = labels_v2._to_px(r.geometry, meta).bounds
    xm, yb = int((x0 + x1) / 2), int(round(roof_bottom))
    assert (mask[yb + 3:yb + 8, xm] == IGNORE).all()  # the wall strip, past the edge band
    assert mask[yb + 14, xm] == 3                       # past the footprint: free land
    assert mask[yb - 10, xm] in (1, 2)                 # the roof itself


def test_road_band_never_cuts_into_a_building():
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    img = np.full((1200, 1280, 3), 100, np.uint8)  # no evidence: footprints stay put
    fp = _footprints((400, 400, 460, 460), meta=meta)
    a, b = labels_v2._from_px(LineString([(300, 465), (560, 465)]), meta).coords
    way = {"type": "way", "id": 1, "tags": {"highway": "residential"},
           "geometry": [{"lon": a[0], "lat": a[1]}, {"lon": b[0], "lat": b[1]}]}
    layers = osm_data.parse_overpass({"elements": [way]}, version="v3")
    mask, _ = labels_v2.build_mask_v2(img, meta, layers, fp)
    # the 6 m road's buffer starts ~10 px north of its line, inside the building:
    # its edge band (+-3 px around y~455) stays building
    assert np.isin(mask[450:456, 430], (1, 2)).all()
    assert (mask[458:462, 430] == IGNORE).all()   # the building's own edge band (rows 458-461)
    assert mask[468, 430] == 0                    # road beyond it
    assert mask[468, 350] == 0 and mask[455, 350] == IGNORE  # road band away from buildings


def test_teacher_ignore_marks_bright_unlabelled_roofs_only():
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    H = W = 400
    img = np.full((H, W, 3), 110, np.uint8)
    mask = np.full((H, W), 3, np.uint8)
    p = np.zeros((H, W), np.float32)
    mask[20:100, 20:100], img[20:100, 20:100] = 1, 200                  # labelled roofs: the luma scale
    img[150:200, 150:200], p[150:200, 150:200] = 205, 0.95              # missed roof
    img[150:200, 250:300], p[150:200, 250:300] = 60, 0.95               # shadowed lane
    img[250:300, 150:200], p[250:300, 150:200] = (60, 170, 60), 0.95    # trees
    img[250:300, 250:300], p[250:300, 250:300] = 205, 0.6               # model unsure
    img[330:336, 330:336], p[330:336, 330:336] = 205, 0.95              # ~3 m2 speck
    mask[340:390, 20:70], img[340:390, 20:70], p[340:390, 20:70] = 0, 205, 0.95  # a road
    ig = labels_v2.teacher_ignore(img, mask, p, meta)
    assert ig[160:190, 160:190].all()
    for sl in [np.s_[150:200, 250:300], np.s_[250:300, 150:200], np.s_[250:300, 250:300],
               np.s_[325:340, 325:340], np.s_[340:390, 20:70], np.s_[20:100, 20:100]]:
        assert not ig[sl].any()
    # without roofs to measure, a fixed floor
    assert labels_v2.teacher_ignore(img, np.where(mask == 1, 3, mask), p, meta)[160:190, 160:190].all()


def test_water_is_other_where_it_looks_wet_smooth_or_rippled():
    meta = aoi.grid_meta(17.44, 78.38, 1, 1)
    rng = np.random.default_rng(0)
    img = np.full((400, 400, 3), 140, np.float32)
    patches = {(60, 60): ((30, 50, 60), 3),      # calm water
               (60, 220): ((44, 72, 72), 25),    # rippled water (Hussain Sagar)
               (220, 60): ((125, 100, 75), 20),  # dry tank bed
               (220, 220): ((40, 62, 35), 20)}   # trees
    for (y, x), (rgb, sd) in patches.items():
        img[y:y + 120, x:x + 120] = np.array(rgb) + rng.normal(0, sd, (120, 120, 1))
    img = img.clip(0, 255).astype(np.uint8)
    ring = labels_v2._from_px(box(40, 40, 360, 360), meta).exterior.coords
    lake = {"type": "way", "id": 3, "tags": {"natural": "water"},
            "geometry": [{"lon": x, "lat": y} for x, y in ring]}
    layers = osm_data.parse_overpass({"elements": [lake]}, version="v3")
    mask, _ = labels_v2.build_mask_v2(img, meta, layers, open_buildings.empty())
    assert mask[120, 120] == 0 and mask[120, 280] == 0
    assert mask[280, 120] == IGNORE and mask[280, 280] == IGNORE
    assert mask[20, 20] == 3  # outside the lake
