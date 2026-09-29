# How UrbanSeg was built

This document describes the method: how the training data was assembled, how
the model was trained and evaluated, and what the serving pipeline does on top
of the model. The repository ships the code and the final checkpoint. It does not
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
  transform (`urbanseg/imagery.py`). That transform is what lets us burn
  vector data into pixel-aligned masks.

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
`*_gt.png` files that are used only for evaluation, never for training.

## 4. Model and training

- **Architecture:** U-Net with a ResNet-34 encoder
  (`segmentation-models-pytorch`), ImageNet-initialised, 4 output classes.
- **Inputs:** random 512 × 512 crops, sampled from training areas in
  proportion to their label-quality weight. Augmentation: random 90°
  rotations, horizontal flips, ±20% gain and ±20 brightness jitter.
- **Loss:** cross-entropy (class weights 1.0 / 1.0 / 2.0 / 0.8, so the
  rarer commercial class counts double) plus multiclass Dice.
- **Optimiser:** AdamW, lr 3e-4, weight decay 1e-4, cosine schedule,
  batch 12, 1,200 crops per epoch, 22 epochs.
- **Selection:** the checkpoint with the best validation pixel accuracy is
  kept (`models/unet_r34.pt`).
- **Cost:** about four hours on a 96-core CPU machine. No GPU was used.

A later fine-tuning pass didn't beat that checkpoint on validation. Validation
accuracy plateaued around epoch 10 while training loss kept falling, which
told us the remaining error was mostly label noise rather than model capacity.

## 5. Inference and fusion

`urbanseg/infer.py`:

1. **Sliding window.** 1024 px windows with a 768 px stride. Softmax
   probabilities are averaged where windows overlap. The CLI can
   additionally average over four flips/rotations (test-time augmentation,
   about 1 point better, 4× slower). The web service skips TTA to keep
   response times reasonable.
2. **Fusion** (coordinate requests only, since an uploaded image has no location):
   - OSM roads, rail and water for the view are fetched live from Overpass
     and painted as `other`.
   - Each known building footprint (OSM, plus Open Buildings if the local
     index is built) is snapped to a single class. If fewer than 20% of its
     pixels are predicted as building, it is left alone: the footprint may be
     stale, or the building demolished. Otherwise the whole footprint takes
     residential or commercial by the model's mean probability inside it. A
     decisive OSM tag overrides the model, and the commercial rules above
     break near-ties (probability gap < 0.15).
   - Without the Open Buildings index, the fusion is more conservative. See
     the note under Evaluation.
3. **Outputs:** colour overlay, per-class area in m², and GeoJSON polygons
   in WGS84.

## 6. Evaluation

Metrics are computed on the full stitched images of the held-out areas
against the auto-generated labels described above (`python -m
urbanseg.evaluate test`).

Web service (single pass, no TTA):

| held-out test (3 areas)        | pixel acc. | mIoU  | residential IoU | commercial IoU | free-land IoU | other IoU |
| -------------------------------- | -----------: | ------: | ----------------: | ---------------: | --------------: | ----------: |
| model only (uploads)           | 70.5%      | 0.448 | 0.622           | 0.222          | 0.571         | 0.376     |
| model + OSM (default)          | 75.2%      | 0.524 | 0.622           | 0.234          | 0.628         | 0.613     |
| model + OSM + Open Buildings   | 78.4%      | 0.577 | 0.667           | 0.314          | 0.653         | 0.674     |

With test-time augmentation (`python -m urbanseg.evaluate test --tta`, and
the CLI default):

| held-out test (3 areas)        | pixel acc. | mIoU  | residential IoU | commercial IoU | free-land IoU | other IoU |
| -------------------------------- | -----------: | ------: | ----------------: | ---------------: | --------------: | ----------: |
| model only                     | 71.5%      | 0.460 | 0.637           | 0.243          | 0.582         | 0.378     |
| model + OSM + Open Buildings   | 79.3%      | 0.597 | 0.677           | 0.358          | 0.664         | 0.687     |

Validation (2 areas, TTA): 73.0% model only, 80.1% with full fusion. Training
areas: 75.9% model only, so the model is not badly overfitted.

The OSM-only row reflects a deliberate choice. OSM building outlines in
Hyderabad cover well under half the building area that Open Buildings does,
and snapping to all of them, or painting road buffers over buildings the model
has found, scored *worse* on this split (71.9%). So without the footprint index,
roads and water are painted only where the model sees no building, and only
OSM outlines with a decisive residential/commercial tag are snapped.

Two caveats:

- **The labels are noisy.** The labels come from open data, so in places the
  model finds real buildings that a stale footprint database calls empty land,
  and it is marked wrong for being right. Treat these numbers as a lower bound.
- **Commercial is the hard class.** A shop-house looks almost exactly like
  a house from above. Most of the commercial signal comes from context: roads,
  POIs, footprint size. That's why fusion helps it most.

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

python -m urbanseg.train 22 1200        # epochs, crops per epoch
python -m urbanseg.evaluate test        # writes reports/eval_test.json
```

Everything lands under `data/`, `models/` and `reports/`. These are
git-ignored, so none of it gets committed by accident.

## 8. Using it for another city

The model has only seen Hyderabad. Elsewhere in India it will still produce
something, but expect lower accuracy. To adapt it:

1. Point `HYD_BBOX` in `config.py` at your city and rebuild the footprint
   index (`python -m urbanseg.open_buildings --force`). The script works out
   which Open Buildings S2 cells cover the box.
2. Change the UTM zone in `labels.py` (`UTM = 32644` is zone 44N).
3. Replace `AREAS` with localities from your city, rebuild the dataset, and
   fine-tune from the shipped checkpoint:
   `python -m urbanseg.train 10 1200 models/unet_r34.pt 1e-4 unet_r34_mycity.pt`.
