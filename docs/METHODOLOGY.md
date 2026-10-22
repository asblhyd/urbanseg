# How UrbanSeg was built

This document describes the method: how the training data was assembled, how
the model was trained and evaluated, and what the serving pipeline does on top
of the model. The repository ships the code and both checkpoints (v2, the
default since 1.2, and v1 of releases 1.0-1.1). It does not
ship imagery, labels or map extracts. Everything here can be regenerated
with the scripts in this repo and your own API key.

## 1. The task

Given a satellite view of part of Hyderabad, label every pixel as one of:

| id | class         | covers                                                                  |
| ---- | --------------- | ------------------------------------------------------------------------- |
| 0  | `other`       | roads, rail, water bodies, drains                                       |
| 1  | `residential` | houses, apartment blocks, hostels                                       |
| 2  | `commercial`  | shops, offices, industry, warehouses, schools, hospitals, places of worship |
| 3  | `free_land`   | anything unbuilt: bare earth, scrub, fields, parks, vegetation          |

"Commercial" deliberately includes institutional and industrial buildings. For
land-use questions the useful split is "people live here" versus "people
work, shop or study here".

## 2. Imagery

- Source: Google Maps Static API, `maptype=satellite`, zoom 18, `scale=2`.
  That is about 0.57 m per map pixel at Hyderabad's latitude, or roughly
  **0.28 m per image pixel**.
- Each request returns a 640 × 640 map-pixel tile (1280 × 1280 image pixels).
  The bottom 40 map pixels carry the provider watermark. We crop them off so
  tiles can be stitched without seams, and credit the provider in the UI and
  in the rendered legend instead.
- Tiles are laid out on an exact Web-Mercator grid around a centre point and
  stitched, so every stitched image carries a precise pixel → lat/lng
  transform (`urbanseg/imagery.py`). The transform is exact (all 104 tile
  seams of the 26 areas are continuous to the pixel), so vector data can be
  burned onto the same grid. The vectors themselves are not aligned with the
  imagery, though: see "Known limitations of these labels" below.

### Areas

We picked 26 localities, each a 3 × 3 tile grid (about 1.1 × 1.0 km,
roughly 30 km² in total). They
were chosen to cover the city's three very different textures:

- **Residential-heavy:** KPHB, Nizampet, Miyapur, Uppal, Dilsukhnagar,
  Vanasthalipuram, Alwal, Manikonda, Old City, Banjara Hills
- **Commercial / industrial:** Hitech City, Gachibowli Financial District,
  Ameerpet, Abids, SD Road, Jeedimetla, Balanagar
- **Outskirts / free land:** Kokapet, Adibatla, Shamshabad, Ghatkesar

Split: 21 training areas, 2 validation areas (LB Nagar, Kompally) used only
for choosing the checkpoint, and 3 held-out test areas (Madhapur, Begumpet,
Kondapur) that were never used for any training decision. The exact centres
are in `urbanseg/config.py` (`AREAS`).

### Version-2 areas (1.2)

The 26 areas are almost all dense, which is where the 1.1 model was weakest
outside them. 21 more 3 × 3 areas (`AREAS_V2`, built by
`scripts/build_dataset_v2.py` into `data_v2/`; 189 Static Maps requests)
cover what they miss: reservoirs and lakes (Himayat Sagar, Shamirpet),
plotted layouts and villages (Tukkuguda, Ibrahimpatnam, Hayathnagar, Kollur,
Bachupally, a village east of Ghatkesar), industrial outskirts (Patancheru;
two sites west and north of the city turned out to be mostly farmland with
long sheds), cantonment (Trimulgherry, Sainikpuri), old dense fabric
(Tolichowki) and high-rise outskirts (Tellapur, Narsingi).

16 of them are training areas. Five are held out as the split `test2`, one
per new texture and at least 2 km from every training area: Hussain Sagar
(lake), Jubilee Hills (villas), Chandrayangutta (old dense), Medchal
(plots), Cherlapally (industry).

## 3. Labels, without hand tracing

Tracing lakhs of buildings by hand wasn't practical, so the ground truth is
generated from two open datasets and a set of rules (`urbanseg/labels.py`).

**Footprints.** Google Open Buildings v3 polygons for greater Hyderabad
(bbox 78.10–78.80 E, 17.10–17.70 N), keeping confidence ≥ 0.65. That
comes to about 2.8 million footprints. OpenStreetMap buildings are merged in:
OSM tags are attached to overlapping Open Buildings footprints, and OSM-only
buildings are added.

**Residential vs commercial.** A footprint with a decisive OSM `building=*`
tag (e.g. `apartments`, `house` vs `office`, `retail`, `industrial`) takes
that class. Every other footprint is scored:

| signal                                                      | commercial score |
| ------------------------------------------------------------- | ------------------ |
| shop / office / craft / hotel / commercial-amenity POIs within 8 m | +1.5 each (max 4) |
| centroid inside commercial / industrial / institutional landuse | +2.0 |
| footprint ≥ 1,000 m²                                        | +1.5             |
| footprint ≥ 3,000 m²                                        | +1.0             |
| touches a 16 m buffer of a motorway / trunk / primary / secondary road | +2.0 |
| touches an 18 m buffer of a tertiary road **and** has a POI | +1.5             |
| within 30 m of a major road **and** larger than 600 m²      | +1.0             |

The residential score is 1.0, plus 2.0 if the centroid sits in residential
landuse. A building is commercial when its commercial score exceeds the
residential score by more than 0.5. The road-frontage rules matter a lot in
Indian cities: the first row of buildings along an arterial road is almost
always shopfronts, even when the upper floors are homes. The POI radius is
kept tight (8 m) because wider radii bled "commercial" two or three rows deep
into residential lanes. Station and transport buildings go to `other`.

**Rasterising.** Masks are painted in a fixed order, and later layers win:

1. everything starts as `free_land`
2. water polygons, plus OSM roads, rail and waterways buffered to realistic
   widths (e.g. primary road 16 m, residential street 6 m, rail 8 m,
   canal 12 m; see `ROAD_WIDTHS` in `config.py`) → `other`
3. building footprints, dilated by 1.2 m to close the party-wall gaps in
   dense fabric → `residential` / `commercial`

**Audit.** Every area's mask was rendered side by side with the imagery
(`scripts/qc_render.py`) and reviewed visually in two rounds, and the rules
above were tuned from that review. Areas whose labels stayed noisy were
down-weighted during training (`AREA_WEIGHTS`; Nizampet was dropped
entirely). `scripts/testfix_render.py` and `scripts/apply_corrections.py`
support box-level manual corrections of the test masks. They write separate
`*_gt.png` files, which `dataset.load_split` uses only when asked
(`use_corrected=True` / `--use-corrected`); none exist yet.

### Known limitations of these labels

Measured in a review of the 1.0 release (October 2026):

- **Buildings are enlarged.** The 1.2 m dilation is 21.5-22.6% of all label
  building pixels, and buildings are painted last, over 13-32% of the road
  and water buffers. A model trained on them draws buildings 1.25-1.54x the
  area of their footprints, mostly as a halo of up to 4 m.
- **Footprints are not on the roofs.** Open Buildings and OSM outlines mark
  ground positions, while Google's mosaic shows each roof displaced by the
  building's lean: 0.6-7.6 m per area (median 2.2 m), up to about 10 m in
  single tiles, larger for taller buildings, and in a direction that changes
  between Google's captures, even inside one tile.
- **Roads have fixed widths** per OSM class and centrelines that can be
  metres off; footpaths and tracks get 5 m; some dry tank beds are mapped
  as water.
- **New buildings are missing** where Open Buildings is older than the
  imagery.
- On the held-out areas about a third (27-42%) of the measured disagreement
  between model and labels is label error, so agreement with these labels
  is capped near 88-92%.

The stored masks are kept unchanged as version v1. Later versions live in
`data_v2/masks_<version>/` and never overwrite v1.

### Version-2 labels (1.2)

`urbanseg/labels_v2.py` builds new labels for all 47 areas from the same
footprints and OpenStreetMap snapshots (`build_dataset_v2.py teacher` and
`masks_v2`). Where the rules cannot be sure, the pixel becomes IGNORE
(255), which carries no loss in training and is not scored in evaluation:

- **Buildings on their roofs.** Each footprint is moved onto its roof as
  imaged: its outline is matched against the image's edges and its
  inside/outside contrast over ±32 px (±9 m). Strong matches form a local
  field, the score-weighted median shift within 120 m, because a capture's
  lean is coherent. Each building then takes its best shift within ±8 px of
  that field when its own evidence is good enough, else the field itself.
  11-53% of an area's buildings align on their own evidence. Per-area
  median shifts are 0.4-3.3 m where there are many buildings, and the field
  changes direction where Google's mosaic switches capture.
- **No enlargement.** Outlines are used as they are. Only party-wall gaps
  narrower than about 2.4 m are closed, and 2 px (0.57 m) on each side of
  every building edge is IGNORE.
- **Walls.** The strip an outline sweeps on its way from the footprint to
  the roof (the walls a leaning building shows, or ground of unknown class)
  is IGNORE, not free land.
- **Roads** keep their per-class widths, but paths and tracks get 2 m and
  3 px on each side of every road buffer's edge is IGNORE. Buildings win:
  no road or water band reaches into a building.
- **Water** is `other` only where it looks wet (dark, and smooth or
  blue-green, since a rippled lake is not smooth); the rest of a water
  polygon is IGNORE (dry tank beds).
- **Lakes.** Every OpenStreetMap query of 1.0-1.1 ended in `out tags geom`,
  which drops relation members, so multipolygon lakes and reservoirs (Hussain
  Sagar, Himayat Sagar, and lakes in Begumpet and Nizampet) never became
  water. The v2 labels fetch those relations' geometry once
  (`data_v2/osm/<area>_water.json`), and live fusion asks for it too (query
  v4). The frozen v1 snapshots keep the gap.

Label version **v2t** adds one rule. Open Buildings (its release has nothing
below 0.65 confidence) misses 6-22% of the roof area in the dense areas
checked: KPHB 9%, Kondapur 6%, Madhapur 22%, the last mostly tall buildings
leaning beyond the search range. Microsoft's footprints would add under
0.6 ha per area. v2t therefore marks free land IGNORE where the 1.1 model is
at least 85% sure it is a building, the pixel is as bright as the area's
labelled roofs (shadowed lanes and setbacks stay free land, so the model is
still taught not to spill into them) and not green, in blobs of at least
15 m². It never turns anything into a building. In dense areas 13-21% of the
pixels end up IGNORE.

`data_v2/masks_v1/` also holds the new areas labelled with the v1 rules, for
comparison; those rules miss the lakes above.

## 4. Model and training

The v1 model (releases 1.0-1.1, `models/unet_r34.pt`):

- **Architecture:** U-Net with a ResNet-34 encoder
  (`segmentation-models-pytorch`), ImageNet-initialised, 4 output classes.
- **Inputs:** random 512 × 512 crops, sampled from training areas in
  proportion to their label-quality weight. Augmentation: random 90°
  rotations, horizontal flips, ±20% gain and ±20 brightness jitter.
- **Loss:** cross-entropy (class weights 1.0 / 1.0 / 2.0 / 0.8, so the
  rarer commercial class counts double) plus multiclass Dice.
- **Optimiser:** AdamW, lr 3e-4, weight decay 1e-4, cosine schedule,
  batch 12, about 1,400 crops per epoch (117 batches), 22 epochs, on
  Python 3.14.4.
- **Selection:** the checkpoint with the best validation pixel accuracy is
  kept (`models/unet_r34.pt`).
- **Cost:** about four hours on a 96-core CPU machine. No GPU was used.

### The v2 model (1.2)

Same architecture, trained afresh from ImageNet weights on the v2t labels of
all 36 training areas (the 20 kept v1 areas with their weights, the 16 new
ones at 1.0):

- **Augmentation v2:** the v1 flips and rotations plus scale jitter
  (0.8-1.25×), per-channel gain (±8%), saturation (0.75-1.25), blur (30% of
  crops, sigma 0.3-1.2 px) and JPEG re-compression (30%, quality 55-95), for
  screenshots and other imagery.
- **Class-aware crops:** a quarter of the crops are centred near commercial
  pixels, the rarest class.
- 40 epochs × 2,000 fresh crops (batch 12), AdamW, lr 3e-4, cosine
  schedule, bfloat16 autocast, about 2 hours on an Apple M5 GPU:
  `python -m urbanseg.train 40 2000 --out im_v2t_s0 --lr 3e-4 --label-version
  v2t --new-areas --aug v2 --class-aware 0.25 --amp bf16 --device mps`.
- **Selection:** best validation mIoU on v2t labels with the training
  overlap masked: epoch 33, mIoU 0.605 on validation crops. A fine-tune of
  the 1.1 model on the same data (15 epochs, lr 1e-4) reached 0.597 and was
  not chosen. It was 0.8 points better on `test2` pixel accuracy, but
  checkpoints are chosen on validation only.

The run's recorded provenance says `dirty`: it started while unrelated
edits (API, evaluation, help text) were uncommitted. Its training code was
that of the commit it names.

### Earlier: the 1.0 fine-tune

A later fine-tuning pass didn't beat that checkpoint on validation.
Validation accuracy plateaued around epoch 10 while training loss kept
falling, and this was read as label noise. A sampler bug explains much of
it: the crop seeds did not change between epochs (they came from
`torch.initial_seed()`, which persistent DataLoader workers keep), so every
epoch and the fine-tune replayed the same 1,302 distinct crops with the
same augmentation, and 35% of the labelled training pixels were never seen.
The shipped model scores 84.5% on its own crops and 74.9% on fresh crops of
the same areas. Four more epochs on fresh crops improved the held-out test
areas by 0.6-1.4 points of pixel accuracy (commercial IoU 0.22 → 0.30). The
bug is fixed in 1.1 (`dataset.EpochSampler`); the shipped weights are still
those of the 1.0 run.

## 5. Inference and fusion

`urbanseg/infer.py`:

1. **Sliding window.** 1024 px windows with a 768 px stride. Each window's
   softmax is weighted by a Gaussian (sigma = a quarter of the window) and
   the sum divided by the summed weights; plain averaging (1.0) left
   straight class seams along window edges. *High* quality averages eight
   flips and rotations (about 1.5 points better, 8× slower); the CLI uses it
   by default.
2. **Fusion** (`urbanseg/fusion.py`; for imagery with a location: coordinate
   requests and georeferenced uploads). OSM is fetched live from Overpass
   (cached for 30 days) and Open Buildings footprints come from the local
   index if it is built. The default `roof` mode keeps the colours where the
   imagery shows the roofs:
   - The model decides where buildings are. Footprints decide what each
     building is: a decisive OSM tag, else the model's mean residential vs
     commercial probability over the building pixels the footprint covers,
     with the commercial rules above breaking near-ties (gap < 0.15). That
     class spreads to the rest of the model's building within 8 m, covering
     the lean. A footprint the model barely sees as building (< 20%) is
     ignored: it may be stale. Station and transport footprints become
     `other`.
   - OSM roads, rail and waterways are painted as `other` where the model
     sees no building, and through a building only in the central half of
     the road and only where the model is unsure it is a building
     (P < 0.5). OSM water is painted only where the model does not see free
     land (dry tank beds stay free land).
   - `URBANSEG_FUSION_MODE=legacy` restores the 1.0 rules: whole footprints
     are snapped, and with the footprint index all OSM buffers are painted
     over everything. That pasted ground-referenced outlines beside leaning
     roofs (18-54% of the building area it added was facade or base) and cut
     0.2-1.0 ha of real roofs per area with road buffers.
3. **Outputs:** class layers, per-class area (WGS84 ellipsoid) for the zone
   and the whole image, GeoJSON polygons in WGS84 with `area_m2`, a GeoTIFF
   of the class raster (EPSG:3857), CSV statistics.

## 6. Evaluation

All figures are pixel agreement with the **auto-generated labels** described
above, on the full stitched images of held-out areas. No label was traced by
hand.

### Release 1.2

Model only (`python -m urbanseg.evaluate --split test|test2 --model v1|v2
--label-version v2t --no-fuse [--tta 8]`):

| split / labels | model | quality | pixel acc. | mIoU | residential | commercial | free land | other | building BF | building area ratio |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test / v2t | v1 | standard | 76.5% | 0.511 | 0.666 | 0.258 | 0.674 | 0.448 | 0.305 | 1.14 |
| test / v2t | **v2** | standard | **80.8%** | **0.569** | 0.724 | 0.330 | 0.740 | 0.480 | 0.482 | 1.03 |
| test / v2t | v1 | high | 78.3% | 0.541 | 0.691 | 0.311 | 0.695 | 0.466 | 0.329 | 1.14 |
| test / v2t | **v2** | high | **81.6%** | **0.582** | 0.734 | 0.352 | 0.748 | 0.492 | 0.489 | 1.03 |
| test2 / v2t | v1 | standard | 76.3% | 0.528 | 0.641 | 0.215 | 0.645 | 0.612 | 0.342 | 1.09 |
| test2 / v2t | **v2** | standard | **80.8%** | **0.605** | 0.700 | 0.345 | 0.706 | 0.668 | 0.487 | 1.01 |
| test2 / v2t | v1 | high | 78.9% | 0.563 | 0.669 | 0.255 | 0.680 | 0.649 | 0.362 | 1.09 |
| test2 / v2t | **v2** | high | **82.6%** | **0.627** | 0.707 | 0.361 | 0.729 | 0.712 | 0.495 | 1.01 |

Against the plain v2 labels (without the teacher rule), standard / high:
test 73.9% / 75.8% (v1) and 78.4% / 79.1% (v2); test2 74.9% / 77.5% (v1)
and 79.5% / 81.3% (v2). Against the v1 labels on the test areas: 70.6% /
72.1% (v1) and 71.8% / 72.4% (v2); v2 is better there too although those
labels are 1.2 m wider than the buildings. On `test2` the v1 rules label the
lakes as free land, so v1 labels are not used there. Every held-out area
improved, by 2.7 (Medchal) to 6.3 (Jubilee Hills) points against v2t.

What the new columns mean:

- **Building area ratio**: predicted / labelled building pixels on scored
  pixels. Above 1 the model paints more building than the labels: v1's
  1.14 was the colour spilling past roofs. v2's labels and model keep to the
  roofs.
- **Building BF** in 1.1 was undefined on v2 labels, whose every edge lies
  in an IGNORE band. Since 1.2, IGNORE thinner than about 9 px takes the
  nearest label for edges only (`evaluate.edge_bands`); wider IGNORE and v1
  labels score as before, so the 1.1 figures below still hold.
- **v2t favours v1 slightly**: its rule leaves out pixels where the v1 model
  said building and the labels said free land, so v1's mistakes there go
  unscored. v2's lead holds on the plain v2 labels as well.

### Release 1.1

Model only, release 1.1 (`python -m urbanseg.evaluate --split test --no-fuse
--tta 1` / `--tta 8`), scored once for the release:

| quality | pixel acc. | mIoU | residential IoU | commercial IoU | free-land IoU | other IoU | within 1 m | building / other / free |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| standard (web default) | 70.6% | 0.450 | 0.623 | 0.228 | 0.572 | 0.378 | 78.9% | 73.6% |
| high (8 passes; CLI default) | 72.1% | 0.473 | 0.643 | 0.272 | 0.587 | 0.389 | 80.5% | 74.8% |

"Within 1 m" counts a pixel as correct when its label class occurs within
1 m (4 px). The last column merges residential and commercial into one
building class. Building boundary F1 at 1 m is 0.334 (standard) and 0.356
(high). Release 1.0 scored 70.5% / 0.448 single-pass with plain window
averaging, and 71.5% with its 4-pass TTA.

How to read these numbers:

- **They understate the model.** About a third (27-42%) of the measured
  disagreement is label error (see "Known limitations of these labels"),
  and agreement with these labels tops out near 88-92%. Gains beyond about
  85% measured this way would mostly be fitting label noise.
- **No accuracy is reported for fusion.** Fusion reuses the OpenStreetMap
  and Open Buildings vectors the labels were drawn from. With 1.0's rules
  every true road/water pixel was "predicted" correctly by construction, and
  at least 91% of the reported gain from fusion (to 75.2% with OSM, 78.4%
  with the footprint index, 79.3% with TTA) was copied geometry. Those
  figures measure agreement with the vectors, not with the ground.
  `evaluate.py` now reports them as `agreement_with_vectors`, next to a
  `fusion_untouched` score on the pixels no vector touches. The roof-aligned
  fusion of 1.1 deliberately departs from the vectors where the imagery
  disagrees, so against these labels it scores lower, and that says nothing
  about whether it is right.
- **1.0's OSM-only rule was chosen on the test split**, against this
  document's own rule. Chosen on validation, its test figure would have been
  about 73.9%, not 75.2%.
- **Validation is contaminated and noisy.** LB Nagar overlaps the training
  area Vanasthalipuram by 0.24 km² (21.6% of LB Nagar; `--mask-overlap`
  ignores it). Epoch-to-epoch spread on the plateau is about ±0.5 points,
  and the test areas lie 0.4-0.8 km from training areas.
- **"Not overfitted" does not follow from the training areas.** They score
  about 76% overall, but that mixes seen and unseen pixels: 84.5% on the
  crops the model trained on, 74.9% on fresh crops (see section 4).
- **Commercial is the hard class.** A shop-house looks almost exactly like a
  house from above, so most of the commercial signal comes from context:
  roads, shop listings, footprint size.

### The gold set

The figures above measure agreement with labels that are themselves wrong
in places. The web app's correction editor (README: *Correct results*)
turns results into hand-checked **gold samples** (`urbanseg/gold.py`,
stored in `data_v2/gold/`, never trained on). `python -m urbanseg.gold
seed-area` seeds drafts from windows of the held-out areas, started from
their v2t labels so that no model's output anchors the review, and
`python -m urbanseg.gold evaluate --model v1 v2` scores models on the
reviewed samples, also per seed. Future figures will be reported against
it, with model-only and fused results compared on the same pixels.

## 7. Reproducing it

You need Python 3.12, a Google Maps Platform key with the Maps Static API
enabled, about 10 GB of disk, and patience. The dataset build makes
26 × 9 = 234 Static Maps requests.

Before you fetch imagery for training, check the Google Maps Platform Terms of
Service. You are responsible for how you use the data you download.

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # put your GOOGLE_MAPS_KEY in it

python scripts/build_dataset.py ob      # Open Buildings: 4.8 GB download → ~370 MB parquet
python scripts/build_dataset.py tiles   # imagery for all areas
python scripts/build_dataset.py osm     # OSM extracts via Overpass
python scripts/build_dataset.py masks   # rasterised labels
python scripts/qc_render.py             # side-by-side panels to eyeball the labels

python -m urbanseg.train 22 1200 --out my_run   # epochs, crops per epoch -> models/runs/my_run/
python -m urbanseg.evaluate --split val --model models/runs/my_run/best.pt --mask-overlap
python -m urbanseg.evaluate --split test --no-fuse --tta 8   # once, at the end
```

The version-2 data and model (needs the steps above first, and
`models/unet_r34.pt` as the v2t teacher):

```bash
python scripts/build_dataset_v2.py tiles      # 21 new areas: 189 Static Maps requests
python scripts/build_dataset_v2.py osm        # their OSM snapshots
python scripts/build_dataset_v2.py masks      # v1 rules, for comparison
python scripts/build_dataset_v2.py teacher    # the 1.1 model's P(building), all 47 areas
python scripts/build_dataset_v2.py masks_v2   # v2 and v2t labels, all 47 areas
python -m urbanseg.train 40 2000 --out my_v2 --lr 3e-4 --label-version v2t \
    --new-areas --aug v2 --class-aware 0.25          # add --amp bf16 --device mps on a Mac
python -m urbanseg.evaluate --split val --model models/runs/my_v2/best.pt \
    --label-version v2t --mask-overlap --no-fuse
```

Everything lands under `data/`, `data_v2/`, `models/` and `reports/`. These are
git-ignored, so none of it gets committed by accident. Training never
overwrites `models/unet_r34.pt` or an earlier run, and evaluation writes a
new `reports/eval_runs/<id>/` each time. To add areas without touching the
existing ones, pass their names: `python scripts/build_dataset.py tiles
<name> ...` (a run without names rebuilds every area).

## 8. Using it for another city

The model has only seen Hyderabad. Elsewhere in India it will still produce
something, but expect lower accuracy. To adapt it:

1. Point `HYD_BBOX` in `config.py` at your city and rebuild the footprint
   index (`python -m urbanseg.open_buildings --force`). The script works out
   which Open Buildings S2 cells cover the box.
2. Change the UTM zone in `labels.py` (`UTM = 32644` is zone 44N).
3. Replace `AREAS` with localities from your city, rebuild the dataset, and
   fine-tune from the shipped checkpoint:
   `python -m urbanseg.train 10 1200 --init models/unet_r34.pt --lr 1e-4 --out mycity`.
