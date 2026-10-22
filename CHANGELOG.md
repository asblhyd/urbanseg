# Changelog

## 1.2.0 (2026-10-22)

A new model, trained on labels that put buildings on their roofs and on 16
more areas, is the default. The 1.1 model stays available.

### Model v2

- **Colours keep to the roofs.** Against roof-aligned labels v1 painted
  1.14× the labelled building area; v2 paints 1.03×. Building edges within
  1 m (boundary F1) rose from 0.31 to 0.48.
- **More accurate everywhere it was tested.** Model only, standard quality:
  80.8% on the test areas (v1: 76.5%) and 80.8% on five new held-out areas
  (v1: 76.3%); high quality 81.6% and 82.6%. Every held-out area improved,
  commercial IoU most (0.26 → 0.33 on test, 0.22 → 0.35 on the new areas).
  Against the old v1 labels v2 is also ahead: 71.8% vs 70.6%.
- **Lakes, layouts, villages, industry, cantonment, high-rise outskirts**:
  21 new areas (16 for training, 5 held out) cover what the original,
  mostly dense areas missed.
- Trained with scale, colour, blur and JPEG augmentation (screenshots), a
  share of crops centred on commercial buildings, and fresh crops every
  epoch.
- **Choose the model**: in the web app (Review → Model), with `model` in
  API jobs, `scripts/segment.py --model v1`, or `URBANSEG_MODEL`. Results
  name the model and its checksum.

### Gold set

- **Correct results by hand** in the web app (*Correct…* under a result):
  brush, fill, Unsure, undo, zoom. Saved corrections form a gold set of
  hand-checked labels, kept apart from the training data, and
  `python -m urbanseg.gold evaluate --model v1 v2` scores models on it.
  Drafts can be seeded from held-out areas with their automatic labels.
  `URBANSEG_GOLD=0` turns the editor off on a public server.

### Fixes

- **Lakes were not water.** Every OpenStreetMap query ended in
  `out tags geom;`, which drops the outlines of multipolygon water bodies
  (Hussain Sagar, Himayat Sagar and other lakes mapped as multipolygons), so fusion never painted them
  as water and the v1 labels called them free land. Live fusion now fetches
  them (cached answers from earlier versions are not reused); lake outlines
  that OSM splits into broken pieces are joined, and islands cut out.
- Overpass answers that the server aborted are no longer stored; dataset
  builds ask the mirrors side by side.
- Evaluation: building boundary F1 works with labels that leave edges
  unscored, and a building area ratio shows how far buildings spill.

### Data and training

- Version-2 labels (`urbanseg/labels_v2.py`): footprints moved onto their
  roofs per building, no widening, uncertain edges, walls and dry water left
  unscored; v2t also leaves out roofs Open Buildings misses.
  `scripts/build_dataset_v2.py` rebuilds them; the v1 data is unchanged.
- Training: `--new-areas`, `--label-version v2t`, `--aug v2`,
  `--class-aware`, `--amp bf16`, `--accum`.

## 1.1.0 (2026-10-22)

A diagnosis of 1.0 found why results "went out of zone" and how far the
published accuracy could be trusted; this release fixes what can be fixed
without retraining. The model weights are unchanged.

### Results stay in your zone

- **Zones.** Draw a rectangle or polygon, import a GeoJSON polygon, or give a
  point and a size. Statistics and exports cover the zone only; before, they
  always covered a fixed square of 0.4-1.1 km around the point (for a 1 ha
  plot the residential share was off by 15 points on median). The fewest
  imagery tiles are fetched: a 1-2 ha plot needs 1 Static Maps request, not 4.
- **Colours stay on the roofs.** New fusion (`URBANSEG_FUSION_MODE=roof`, the
  default): building footprints decide residential vs commercial but no longer
  paint ground-referenced outlines over the image, which added facade and base
  strips beside leaning roofs (18-54% of the building area fusion added). Roads
  no longer cut roofs the model is sure of, dry lake beds mapped as water stay
  free land, and stations are no longer coloured commercial. `legacy` restores
  the 1.0 rules.
- **Uploads at the right scale.** GeoTIFF / JPEG 2000 files with coordinates
  are placed on the map grid (map-data fusion, exact areas). Screenshots are
  scaled from their Google Maps zoom and screen type, other images from their
  metres per pixel. The silent downscale of images over 4096 px (which
  misreported their area, e.g. a 0.09 km² drone image as 1.36 km²) is gone;
  16-bit imagery is stretched instead of turning white or black; oversized and
  malformed files are refused with a reason.
- **No straight cuts at window edges**: the model's overlapping windows are
  blended with Gaussian weights instead of plain averages.
- **Coordinates**: decimal commas, degrees-minutes-seconds and Google Maps
  links are understood; swapped latitude/longitude and places outside greater
  Hyderabad are caught before any Google request.

### Web app

A new map-based app replaces the two-form page: choose the area on Google
Maps, review the tiles, request count and time, run with progress and cancel,
then explore: class toggles, opacity, model-only vs with map data,
low-confidence hatching, the analysed imagery, a hover readout, statistics in
m², ha, acres, sq yd or guntas, and downloads (GeoJSON, GeoTIFF, CSV, PNG,
class mask, JSON). Uploads without coordinates open in an image viewer.

### API

- New JSON API: `/api/config`, `/api/locate`, `/api/plan`, `/api/jobs`,
  `/api/jobs/upload`, `/api/jobs/{id}` (status, `DELETE` to cancel) and
  `/api/jobs/{id}/files/{name}`. `/segment`, `/segment.png`, `/segment-image`
  and `/health` keep their request and response shapes; their GeoJSON now has
  all four classes and `area_m2`.
- Runs are jobs on one inference worker with a short queue (busy → 503 with
  `Retry-After`). An upload no longer freezes the server.

### Security

- **Fixed:** a network failure while fetching imagery returned the Google key
  in the error message and printed it in the log.
- Requests with an unknown `Host` (DNS rebinding) and requests made by other
  web pages through the browser are refused, so a site you visit cannot spend
  your Google quota. `URBANSEG_ALLOWED_HOSTS` lists extra host names.
- The server key is sent to a browser (for the map) only when the app is
  opened as localhost; `GOOGLE_MAPS_BROWSER_KEY` sets a separate browser key.
- Error messages never echo internal details.

### Reliability

- Imagery tiles are fetched in parallel. Network errors, 429 and 5xx are
  retried with backoff; a bad key or missing billing fails at once with
  Google's reason (before, it was retried four times and the reason was lost).
- OpenStreetMap: the Overpass mirrors are hedged (the next one is asked when
  one fails or is slow), the fetch runs alongside the imagery and the model,
  answers are cached for 30 days in the `urbanseg-cache` volume and validated
  when read, partial answers are never cached, and after repeated outages OSM
  is skipped for two minutes instead of each run waiting it out.
- A corrupt cache file no longer disables OSM fusion for an area forever.

### Outputs

- Areas use the WGS84 ellipsoid (1.0 was 0.55% high).
- *High* quality: 8-way test-time augmentation, about +1.5 points, 8× slower.
- GeoJSON for all four classes with `area_m2`, clipped to the zone; GeoTIFF
  of the class raster with a colour table; CSV statistics in five units.
- Command line: `--aoi`, `--bbox`, `--location`; `--mpp`, `--zoom`, `--dpr`
  for images; `--quality`, `--fusion-mode`; writes the same files.

### Training and evaluation (maintainers)

- **Sampler bug fixed.** Every epoch of the 1.0 training run, and of the
  fine-tune, replayed the same 1,302 crops with the same augmentation, so 35%
  of the labelled pixels were never seen. Crops are now drawn per (seed,
  epoch, index). Four epochs on fresh crops already beat the shipped model on
  the test areas (+0.6 to +1.4 points, commercial IoU 0.22 → 0.30); a full
  retrain is planned.
- `train.py` never overwrites `models/unet_r34.pt` or an existing run; runs
  go to `models/runs/<name>/` with their config and provenance.
- Hand-corrected `*_gt.png` masks are used only with `--use-corrected`.
- `evaluate.py` writes to `reports/eval_runs/<id>/` and never over earlier
  reports. It reports model-only metrics, boundary-tolerant accuracy, a
  three-class score and building boundary F1. Fused scores are labelled
  `agreement_with_vectors` (they reuse the label sources) and come with a
  `fusion_untouched` score. Optional masking of the validation area that
  overlaps a training area (LB Nagar / Vanasthalipuram, 21.6%).

### Known limitations

- The model was trained on labels whose buildings are enlarged by 1.2 m and
  sit 1-8 m off leaning roofs, so it still draws buildings about 25-55% larger
  than their footprints, mostly as a halo of up to 4 m. Fixing this needs new,
  versioned labels and a retrain.
- There is no hand-checked test set yet; accuracy figures are agreement with
  automatically generated labels (see the README).
