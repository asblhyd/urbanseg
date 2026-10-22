# UrbanSeg

UrbanSeg looks at satellite imagery of Hyderabad and colours in what's there:
**residential buildings**, **commercial buildings**, **free land**, and
**roads / rail / water**. Draw your zone on a map (or give a point, a GeoJSON
polygon, or upload your own image) and you get the share and area of each
class inside that zone, a map overlay, and files you can open in QGIS
(GeoJSON, GeoTIFF) or a spreadsheet (CSV).

It runs on your own machine in Docker. One script builds it, starts it, and
serves the web app at **<http://127.0.0.1:5555>**.

| colour | class | what it means |
| --- | --- | --- |
| orange | residential | houses, apartment blocks, hostels |
| red | commercial | shops, offices, industry, schools, hospitals, places of worship |
| green | free land | anything unbuilt: open or vacant ground, fields, parks, trees |
| grey | other | roads, rail, water |

This repository contains **code and a trained model only**. No imagery, labels
or map extracts are included. How the training data was built, and how to
rebuild it, is in [docs/METHODOLOGY.md](docs/METHODOLOGY.md). What changed in
each version is in [CHANGELOG.md](CHANGELOG.md).

---

## What you need

- **Docker**, any of: Docker Desktop (macOS, Windows), Docker Engine on
  Linux (including the Ubuntu snap), or Colima / Rancher Desktop on macOS.
  Check it works with `docker run --rm hello-world`. On Windows, run the
  commands below in WSL 2 or Git Bash.
- **Git with Git LFS.** The model weights (98 MB) are stored with
  [Git LFS](https://git-lfs.com). Install it with `sudo apt install git-lfs`,
  `brew install git-lfs`, or the installer from the website.
- **About 5 GB of free disk, and at least 4 GB of memory for Docker.** One
  analysis uses 1.5-4 GB while the model runs. Docker Desktop and Colima run a
  VM with its own limit: Docker Desktop → Settings → Resources → Memory, or
  `colima start --memory 4`. `run.sh` checks this. More CPU cores make each
  run faster; no GPU is needed.
- **A Google Maps Platform API key** with two APIs enabled in the
  [Google Cloud console](https://console.cloud.google.com/google/maps-apis):
  - **Maps Static API**: the satellite imagery that is analysed. Without it
    you can still upload your own images.
  - **Maps JavaScript API**: the interactive map you draw your zone on.
    Without it the app still works, with coordinates, GeoJSON zones and
    uploads, but without a map.

  A zone needs 1 to 9 Static Maps requests (a 1-2 ha plot needs one); the app
  shows the count before anything is fetched. See
  [Keys and security](#keys-and-security) before you put the service on a
  network.

## Run it: step by step

**Step 1. Turn on Git LFS.** You only do this once per machine.

```bash
git lfs install
```

**Step 2. Clone the repository.**

```bash
git clone https://github.com/asblhyd/urbanseg.git
cd urbanseg
```

If you cloned before installing Git LFS, run `git lfs pull` now to fetch the
real weights. `run.sh` will tell you if you forgot. Use `git clone` rather
than GitHub's "Download ZIP"; the ZIP may contain only an LFS placeholder.

**Step 3. Add your Google Maps key.**

```bash
cp .env.example .env
```

Open `.env` and put your key after `GOOGLE_MAPS_KEY=`. You can also skip this
step: `run.sh` asks for the key the first time and saves it to `.env`.
`.env` is git-ignored, so your key won't end up in a commit.

**Step 4. Start it.**

```bash
./run.sh
```

(`bash run.sh` and `sh run.sh` work too.) The script checks the model file
and its checksum, reads your key, builds the Docker image, starts a container
called `urbanseg` on `127.0.0.1:5555`, and waits until it is healthy. The
first build downloads about 1.5 GB of Python packages and takes several
minutes; after that `./run.sh` is back up in well under a minute.

**Step 5. Open <http://127.0.0.1:5555> in your browser.**

The container restarts on its own whenever Docker restarts, until you run
`./run.sh stop`.

## Using the web app

### 1. Choose the area

- **Draw zone** (recommended). Use *Go to* to jump to a place: paste
  coordinates in almost any form (`17.4435, 78.3772`, `17,4435; 78,3772`,
  degrees-minutes-seconds) or a Google Maps link. Then pick *Rectangle*
  (click two opposite corners) or *Polygon* (click each corner, double-click
  to finish) and drag the white handles to adjust.
- **Point**: click the map to set a centre and pick a size (Small ~0.4 km,
  Medium ~0.7 km, Large ~1.1 km across).
- **Import**: load or paste a GeoJSON polygon, e.g. a plot boundary
  exported from QGIS.
- **Upload**: analyse an image you already have (see
  [Your own images](#your-own-images)).

Swapped latitude/longitude and other typos are caught before any Google
request is made.

### 2. Review

Before you run, the app draws the imagery tiles it will fetch on the map and
tells you the zone's area, the analysed extent, the number of Google Static
Maps requests and an estimated time. Options:

- **Use map data**: refine the model's result with OpenStreetMap (roads,
  water, building tags) and, if you built it, the building-footprint index.
- **Quality**: *High* averages 8 rotated/flipped passes. It is about 8× slower
  and about 1.5 points more accurate.
- **Model**: *v2 (October 2026)*, the default, or *v1 (August 2026)*, the
  model of releases 1.0-1.1, to compare. The result line names the model
  used, and `result.json` records its checksum.

### 3. Run

A progress bar shows each stage (imagery, model, OpenStreetMap, fusion). You
can cancel at any time.

### 4. Explore and export

- **Statistics for your zone**, or for the whole analysed image, in m²,
  hectares, acres, square yards or guntas.
- **Layers**: opacity; each class on or off; *Model only* vs *With map
  data*; hatching where the model is unsure; the analysed imagery itself;
  results outside your zone.
- **Hover** over the map to read the class at any point (and what the model
  alone said, where map data changed it).
- **Downloads**: GeoJSON of the zone (one polygon per region, with
  `area_m2`), GeoTIFF of the class raster (EPSG:3857, with a colour table),
  CSV of the statistics, the overlay with a legend, the class-id mask, and
  the full result as JSON.

Colours sit on the roofs as Google's imagery shows them. Tall buildings lean
in satellite imagery, so for them an exported outline can be 1-8 m from the
building's ground position.

Results are kept in memory for an hour; *This session* in the side panel
lists your recent runs.

### 5. Correct results: the gold set

The accuracy figures below compare the model with labels that were
generated automatically, and those labels have errors of their own. To
measure the real accuracy, correct results by hand:

1. Under a result, choose where to start (*the model alone* is the fairest
   start) and click **Correct…**.
2. Paint over mistakes with the class buttons (keys 1-4), mark parts you
   cannot decide as **Unsure** (key 0), **Fill** a whole region of one
   class with a click (F), pan with the right mouse button or by holding
   space, zoom with the wheel, undo with Ctrl/Cmd+Z.
3. Tick *I checked all of it* and **Save**. The sample joins the **Gold
   set** list in the side panel, where you can reopen and edit it.

Samples keep the analysed imagery and your labels in `data_v2/gold/` (a
Docker volume when run with `./run.sh`); Unsure pixels are never scored. To
score models on them:

```bash
python -m urbanseg.gold evaluate --model v1 v2
```

`python -m urbanseg.gold seed-area <area> --window Y0 X0 SIZE` makes a draft
from a training-data area, started from its automatic labels instead of a
model's output; *Fill unsure from neighbours* then closes the uncertain
bands those labels leave along edges, for you to check.

### Your own images

The model was trained on imagery at about **0.29 m per pixel** and only works
near that scale, so the app needs to know your image's scale:

| what you upload | choose | what happens |
| --- | --- | --- |
| GeoTIFF / orthophoto / JPEG 2000 with coordinates | *The file has coordinates* | placed on the map, resampled to the model's scale, fused with map data, exact areas |
| Google Maps screenshot | *Google Maps screenshot*, with the zoom level and your screen type | resampled to the model's scale; exact areas |
| drone or other photo without coordinates | *I know its resolution* (metres per pixel) | resampled; exact areas |
| anything else | *I don't know* | analysed as if at 0.29 m/px; areas are approximate |

Images coarser than about 0.5 m/px get a warning (small buildings become
unreliable); coarser than 2.3 m/px is refused. Up to 200 MB and about 6.5 km²
of ground per image.

## Everyday commands

| command | what it does |
| --- | --- |
| `./run.sh` | build if anything changed, then (re)start the service |
| `./run.sh stop` | stop and remove the container (image, footprint index and OSM cache are kept) |
| `./run.sh logs` | follow the service logs (Ctrl-C to leave) |
| `PORT=6000 ./run.sh` | serve on `127.0.0.1:6000` instead |
| `./run.sh --with-footprints` | also build the optional footprint index (below) |
| `git pull && ./run.sh` | update to the latest version |

To remove everything: `./run.sh stop`, then `docker rmi urbanseg:latest` and
`docker volume rm urbanseg-footprints urbanseg-cache`.

## Optional: the building-footprint index

Coordinate requests are refined with live OpenStreetMap data. OSM building
coverage in Hyderabad is patchy, though. For better building classes you can
build a local index of
[Google Open Buildings](https://sites.research.google/open-buildings/)
footprints for greater Hyderabad:

```bash
./run.sh --with-footprints
```

This is a one-time job. It downloads a 4.8 GB public file from Google, keeps
the ~2.8 million footprints inside greater Hyderabad (about 370 MB), and
deletes the download. While it runs it needs about 6 GB of free disk and 4 GB
of memory for Docker, and takes roughly 10-40 minutes. The index is kept in a
Docker volume (`urbanseg-footprints`).

## How accurate is it?

Read these numbers with care. They are pixel agreement with **labels that
were generated automatically** from Google Open Buildings and OpenStreetMap,
on Hyderabad localities the model never saw: the three test areas (Madhapur,
Begumpet, Kondapur) and five more added in 1.2 (Hussain Sagar, Jubilee
Hills, Chandrayangutta, Medchal, Cherlapally). Nobody traced these labels by
hand; the [gold set](#5-correct-results-the-gold-set) is how that changes.

Model only, against the roof-aligned v2t labels (see
[METHODOLOGY](docs/METHODOLOGY.md#version-2-labels-12)):

| model | test areas | 5 new areas | building area vs labels |
| --- | ---: | ---: | ---: |
| **v2**, standard quality | **80.8%** (mIoU 0.57) | **80.8%** (0.60) | 1.03× |
| **v2**, high quality | **81.6%** (0.58) | **82.6%** (0.63) | 1.03× |
| v1, standard quality | 76.5% (0.51) | 76.3% (0.53) | 1.14× |
| v1, high quality | 78.3% (0.54) | 78.9% (0.56) | 1.14× |

Against the original v1 labels, on the test areas: v2 71.8% (high 72.4%),
v1 70.6% (high 72.1%). v2 scores higher there too, although those labels
widen every building by 1.2 m and v2 no longer does.

- **"Building area vs labels"** is how much building the model paints per
  labelled building pixel. v1's 1.14× was the colour spilling past roofs;
  v2 keeps to the roofs. Building edges within 1 m (boundary F1) rose from
  0.31 to 0.48.
- **Every one of the eight held-out areas improved**, by 2.7 to 6.3 points;
  commercial IoU went from 0.26 to 0.33 (test) and 0.22 to 0.35 (new
  areas), and lakes are now recognised as water.
- **The labels still have errors.** Open Buildings misses 6-22% of the roof
  area in dense localities; v2t leaves such places out of scoring rather than
  calling them free land, so the figures are not the true accuracy. The gold
  set is.
- **No "with map data" figure is given on purpose.** Fusion reuses the same
  OpenStreetMap and Open Buildings data the labels were drawn from, so
  scoring it against those labels mostly measures how much it copied. Earlier
  versions reported 75-79% that way.
- **Residential and free land are the strong classes. Commercial is the
  weakest.** A shop-house looks like a house from above, so most of its
  signal comes from context (roads, shop listings, size).

Hand-checked figures will come from the gold set: review results (or
drafts seeded from the held-out areas with `python -m urbanseg.gold
seed-area`) in the editor, then `python -m urbanseg.gold evaluate --model v1
v2` scores both models on what has been reviewed. [docs/METHODOLOGY.md](docs/METHODOLOGY.md) has
the details, including what is known to limit accuracy today.

**It is trained on Hyderabad.** It runs elsewhere, but the app asks you to
confirm, and results outside greater Hyderabad are indicative only.

## Where your data goes

- **Zones and coordinates** you analyse are sent to Google (to fetch the
  imagery) and to public [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
  servers (OSM data for that box). OSM answers are cached for 30 days in the
  `urbanseg-cache` Docker volume. Nothing else leaves your machine.
- **Uploaded images** are processed inside the container and never sent
  anywhere. Files over 1 MB are spooled to a temporary file in the container
  while they are received. Results, including those of uploads, are kept in
  memory for an hour and are never written to disk.
- The map in the web app is loaded from Google, as on any site that embeds
  Google Maps.

## Keys and security

The service has **no login** and listens on `127.0.0.1` only, on purpose. It
refuses requests whose `Host` is not this machine, and requests that another
web page tries to make through your browser, so a malicious site cannot
spend your Google quota.

The map in the browser needs a key the browser can see. On your own machine
the app reuses your `GOOGLE_MAPS_KEY` for it, but only when you open the app
as `localhost` or `127.0.0.1`. For anything else:

- **To use it from your laptop when it runs on a server**, tunnel over SSH.
  It then behaves as if local:
  ```bash
  ssh -N -L 5555:127.0.0.1:5555 you@your-server
  # now open http://127.0.0.1:5555 on your laptop
  ```
- **To serve it on a network**, put it behind a reverse proxy with
  authentication and HTTPS, and set:
  - `URBANSEG_ALLOWED_HOSTS=your.host.name`
  - `GOOGLE_MAPS_BROWSER_KEY` to a second key that has only the Maps
    JavaScript API and is restricted to your site's address (HTTP referrer).
    The server key is then never sent to browsers.
  - Restrict the server key to the Maps Static API and your server's IP
    address, and set a daily quota in the Google Cloud console.

## Configuration

All optional, set in `.env` or the environment:

| variable | default | what it does |
| --- | --- | --- |
| `GOOGLE_MAPS_KEY` | (none) | server key: Maps Static API (imagery) |
| `GOOGLE_MAPS_BROWSER_KEY` | (none) | separate browser key for the map (Maps JavaScript API) |
| `URBANSEG_ALLOWED_HOSTS` | `127.0.0.1,localhost,::1` | host names the service answers to (`*` = any) |
| `URBANSEG_SHARE_SERVER_KEY` | `1` | `0` never sends the server key to a browser |
| `URBANSEG_FUSION_MODE` | `roof` | `roof` keeps colours on the roofs; `legacy` = v1.0 rules |
| `URBANSEG_MAX_TILES` | `9` | most Static Maps requests one run may make |
| `URBANSEG_MAX_AOI_KM2` | `4` | largest zone |
| `URBANSEG_MAX_UPLOAD_MB` | `200` | largest upload |
| `URBANSEG_MAX_NATIVE_MPX` | `80` | largest upload after resampling (~6.5 km²) |
| `URBANSEG_MAX_QUEUE` | `4` | runs that may wait while one runs |
| `URBANSEG_JOB_TTL_S` | `3600` | how long results are kept |
| `URBANSEG_OSM_DEADLINE_S` | `60` | how long to wait for OpenStreetMap |
| `URBANSEG_DEVICE` | `cpu` | `mps` / `auto` to use an Apple GPU outside Docker |
| `URBANSEG_MODEL` | `v2` | default model: `v2` or `v1` (or a checkpoint path, outside the API) |
| `URBANSEG_GOLD` | `1` | `0` turns the gold-set editor off (anyone who can open the app can save samples) |
| `URBANSEG_GOLD_DIR` | `data_v2/gold` | where gold samples are stored |

## HTTP API

The web app is a page over a JSON API; interactive docs are at
<http://127.0.0.1:5555/docs>. Analyses run as jobs:

```bash
# plan a zone (no Google request): tiles, request count, extent, warnings
curl -X POST localhost:5555/api/plan -H 'Content-Type: application/json' \
     -d '{"aoi": [78.3905, 17.4475, 78.3925, 17.4490]}'

# run it: returns {"job": {"id": ...}}; poll until "state" is "done"
curl -X POST localhost:5555/api/jobs -H 'Content-Type: application/json' \
     -d '{"aoi": [78.3905, 17.4475, 78.3925, 17.4490], "quality": "standard"}'
curl localhost:5555/api/jobs/<id>
curl -o zone.geojson localhost:5555/api/jobs/<id>/files/result.geojson

# an image: file plus its scale (mpp, or zoom and dpr for a screenshot)
curl -F file=@ortho.tif localhost:5555/api/jobs/upload
```

`aoi` is a GeoJSON geometry, Feature or FeatureCollection, or a
`[west, south, east, north]` box; instead of `aoi` you can send `lat`, `lng`
and `grid` (tiles per side), or a `location` text. Files of a finished job
include `classes.png`, `raw.png`, `lowconf.png`, `imagery.jpg`, `mask.png`,
`classes.tif`, `result.geojson`, `stats.csv`, `overlay.png` and
`result.json`.

The original endpoints still work as before for existing scripts:

```bash
curl "http://127.0.0.1:5555/segment?lat=17.4435&lng=78.3772&grid=2&geojson=true"
curl -o overlay.png "http://127.0.0.1:5555/segment.png?lat=17.4435&lng=78.3772"
curl -F file=@my_tile.png http://127.0.0.1:5555/segment-image
curl http://127.0.0.1:5555/health
```

## Command line

The command-line tool writes the overlay, class mask, GeoTIFF, GeoJSON and
statistics to a folder. It defaults to *High* quality.

```bash
python scripts/segment.py --bbox 78.3905,17.4475,78.3925,17.4490
python scripts/segment.py --aoi plot.geojson --out outputs/plot7
python scripts/segment.py --location "17.4435, 78.3772" --grid 2
python scripts/segment.py --image ortho.tif
python scripts/segment.py --image shot.png --zoom 18 --dpr 2
python scripts/segment.py --location "17.4435, 78.3772" --model v1   # the older model
```

`python scripts/segment.py --help` lists every option.

## Troubleshooting

**"models/unet_r34.pt is a Git LFS pointer"**: the weights weren't downloaded.
Run `git lfs install && git lfs pull`.

**"cannot reach the Docker daemon"**: start Docker Desktop, or on Linux run
`sudo systemctl start docker`. If your user isn't in the `docker` group,
`run.sh` falls back to `sudo docker` and may ask for your password.

**"could not start the container … port already in use"**: something else is
on 5555. Use `PORT=5556 ./run.sh`.

**"Map unavailable"**: the key has no Maps JavaScript API. Enable it in the
Google Cloud console, or set `GOOGLE_MAPS_BROWSER_KEY`. Everything except
drawing on the map still works.

**"Google rejected the API key"**: the Maps Static API isn't enabled for it,
or billing isn't set up on the Google Cloud project.

**"OpenStreetMap could not be reached"**: the public Overpass servers are
often overloaded. The result is still produced, without roads, water and
building tags. Try again in a few minutes. Once an area has worked, its OSM
data is cached for 30 days.

**"ran out of memory" / "was killed"**: give Docker at least 4 GB (Docker
Desktop → Settings → Resources → Memory, or `colima start --memory 4`).

**Docker installed as a snap (Ubuntu)**: nothing to do; `run.sh` detects it.
If Docker misreads the build files, force the other way of sending them with
`BUILD_CONTEXT=stream ./run.sh` (or `BUILD_CONTEXT=dir ./run.sh`).

**Apple Silicon / ARM**: Docker builds a native ARM64 image; every dependency
ships a prebuilt ARM64 wheel.

## Running without Docker

With Python 3.12:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn urbanseg.api:app --host 127.0.0.1 --port 5555
```

With uv, add `--index-strategy unsafe-best-match` to `uv pip install`. On an
Apple Silicon Mac, `URBANSEG_DEVICE=mps` runs the model about 5× faster.

## Development

```bash
pip install pytest httpx
pytest                 # unit tests; data- and model-dependent tests skip without data/
pytest -m slow         # also the full-area runs
```

The tests block the network, never use a real key, and fail if anything under
`data/` or `models/` changes.

## What's in the repository

```text
run.sh                 the one script you need
Dockerfile             image definition used by run.sh
requirements.txt       pinned Python dependencies (CPU PyTorch)
.env.example           template for your Google Maps key
models/unet_r34_v2.pt  v2 weights, the default (Git LFS) + .sha256 checksum
models/unet_r34.pt     v1 weights (Git LFS) + .sha256 checksum
urbanseg/              the package
  api.py               FastAPI service and JSON API
  web/                 the map web app (plain HTML/CSS/JS, no build step)
  pipeline.py          one analysis: imagery, model, map data, outputs
  jobs.py              background jobs, one inference at a time
  aoi.py               zones, tile planning, clipping, coordinate parsing
  uploads.py           uploaded images: scale, georeference, limits
  infer.py             sliding-window inference and outputs
  fusion.py            refinement with footprints and OpenStreetMap
  imagery.py           Static Maps tile maths, fetching and stitching
  osm_data.py          OpenStreetMap via Overpass (cached)
  open_buildings.py    Google Open Buildings index (build + lookup)
  labels.py            training-label generation rules (v1)
  labels_v2.py         roof-aligned v2 labels
  gold.py              the hand-checked gold set: storage, seeding, scoring
  dataset.py, train.py, evaluate.py   training and evaluation
  config.py            classes, colours, training areas, rule vocabularies
scripts/               command-line tool and dataset / label-QC tools
tests/                 pytest suite
docs/METHODOLOGY.md    how the data, labels and model were produced
```

## Credits and licence

UrbanSeg is released under the [Apache License 2.0](LICENSE).

At runtime it uses data from third parties, each under its own terms:
satellite imagery and the interactive map © Google (Google Maps Platform
Terms of Service); map data © OpenStreetMap contributors, under the ODbL;
and, if you build the index, building footprints from Google Open Buildings,
under CC BY 4.0 and ODbL. See [NOTICE](NOTICE).
