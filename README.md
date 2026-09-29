# UrbanSeg

UrbanSeg looks at a satellite image of Hyderabad and colours in what's there:
**residential buildings**, **commercial buildings**, **free land**, and
**roads / rail / water**. Give it a latitude and longitude and it fetches the
imagery for you. Or upload a satellite picture you already have. You get back
an overlay, the area of each class in square metres, and a GeoJSON file you can
open in QGIS.

It runs on your own machine in Docker. One script builds it, starts it, and
serves a small web app at **<http://127.0.0.1:5555>**.

| colour | class | what it means |
| --- | --- | --- |
| orange | residential | houses, apartment blocks |
| red | commercial | shops, offices, industry, schools, hospitals |
| green | free land | open or vacant ground, fields, parks, greenery |
| grey | other | roads, rail, water |

This repository contains **code and a trained model only**. No imagery, labels
or map extracts are included. If you want to know how the training data was
built and how to rebuild it yourself, read
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

---

## What you need

- **Docker**, any of: Docker Desktop (macOS, Windows), Docker Engine on
  Linux (including the Ubuntu snap), or Colima / Rancher Desktop on macOS.
  Check it works with `docker run --rm hello-world`. On Windows, run the
  commands below in WSL 2 or Git Bash.
- **Git with Git LFS.** The model weights (98 MB) are stored with
  [Git LFS](https://git-lfs.com). Install it with `sudo apt install git-lfs`,
  `brew install git-lfs`, or the installer from the website.
- **About 5 GB of free disk, and at least 3 GB of memory for Docker** (4 GB
  if you build the footprint index). On Linux that's simply your RAM. Docker
  Desktop and Colima run a VM with its own limit: Docker Desktop → Settings →
  Resources → Memory, or `colima start --memory 4` (Colima defaults to 2 GB).
  `run.sh` checks this and tells you if it's too low. More CPU cores make
  each request faster; no GPU is needed.
- **A Google Maps API key** *(optional)*, with the **Maps Static API**
  enabled. It is only needed for the "By coordinates" mode. Without a key you can
  still upload images. You can create one in the
  [Google Cloud console](https://console.cloud.google.com/google/maps-apis).
  Each coordinate request makes 1 to 9 Static Maps calls (4 at the default size).

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
than GitHub's "Download ZIP"; the ZIP may contain only an LFS placeholder
instead of the model.

**Step 3. Add your Google Maps key (optional).**

```bash
cp .env.example .env
```

Open `.env` and put your key after `GOOGLE_MAPS_KEY=`. You can also skip this
step: `run.sh` asks for the key the first time and saves it to `.env` for you.
`.env` is git-ignored, so your key won't end up in a commit.

**Step 4. Start it.**

```bash
chmod +x run.sh
./run.sh
```

(`bash run.sh` and `sh run.sh` work too.)

The script does this:

1. checks that the model file is the real weights, not an LFS placeholder, and that its checksum matches;
2. reads your Maps key from `.env`, or asks for it;
3. builds the Docker image;
4. starts a container called `urbanseg` on `127.0.0.1:5555`;
5. waits until the service is healthy and prints its status.

The first build downloads about 1.5 GB of Python packages (mostly PyTorch)
and takes several minutes. After that it's cached, and `./run.sh` is back up
in well under a minute.

When it's done you'll see:

```text
UrbanSeg is running:  http://127.0.0.1:5555

  By coordinates   : enabled
  Footprint index  : not built (optional: ./run.sh --with-footprints)
```

**Step 5. Open <http://127.0.0.1:5555> in your browser.**

The container restarts on its own whenever Docker restarts, until you run
`./run.sh stop`.

## Using the web app

### By coordinates (recommended)

1. In Google Maps, right-click the spot you care about and click the numbers
   at the top of the menu (e.g. `17.4435, 78.3772`) to copy them.
2. Paste the first number into *Latitude* and the second into *Longitude*.
3. Pick an area size: Small is about 0.4 km across, Medium about 0.7 km, Large about 1.1 km.
4. Click **Segment location**. It fetches fresh imagery, runs the model,
   and cleans the result up with OpenStreetMap roads, water and building
   outlines. Expect 20–90 seconds, depending on area size and CPU.

### Upload a satellite image

Choose a PNG or JPG and click **Segment image**. An uploaded image has no
location attached, so this path is the model on its own, with no map data to help.
It works best on imagery at a similar scale to Google Maps zoom 18
(about 0.3 m per pixel). The area figures assume that scale.

### The result

You get the overlay, a table with each class's share and area
in m², and download links for the overlay PNG and, for coordinate requests,
a GeoJSON of the polygons.

## Everyday commands

| command | what it does |
| --- | --- |
| `./run.sh` | build if anything changed, then (re)start the service |
| `./run.sh stop` | stop and remove the container (the image and footprint index are kept) |
| `./run.sh logs` | follow the service logs (Ctrl-C to leave) |
| `PORT=6000 ./run.sh` | serve on `127.0.0.1:6000` instead |
| `./run.sh --with-footprints` | also build the optional footprint index (below) |
| `git pull && ./run.sh` | update to the latest version |

To remove everything: `./run.sh stop`, then
`docker rmi urbanseg:latest` and `docker volume rm urbanseg-footprints`.

## Optional: the building-footprint index

By default, coordinate requests are refined with live OpenStreetMap data.
OSM building coverage in Hyderabad is patchy, though. For better building
outlines you can build a local index of
[Google Open Buildings](https://sites.research.google/open-buildings/)
footprints for greater Hyderabad:

```bash
./run.sh --with-footprints
```

This is a one-time job. It downloads a 4.8 GB public file from Google,
keeps the ~2.8 million footprints inside greater Hyderabad (about 370 MB),
and deletes the download. While it runs it needs about 6 GB of free disk and
4 GB of memory for Docker. It takes roughly 10–40 minutes depending on your
connection and CPU (about 11 minutes on a 2-core cloud VM). The index is kept
in a Docker volume (`urbanseg-footprints`), so later runs of `./run.sh` pick
it up automatically.

## How accurate is it?

Measured on three Hyderabad localities that the model never saw during
training (Madhapur, Begumpet, Kondapur):

| mode | pixel accuracy | mean IoU |
| --- | ---: | ---: |
| uploaded image (model only) | 70.5% | 0.45 |
| coordinates, default (model + OpenStreetMap) | 75.2% | 0.52 |
| coordinates + footprint index | 78.4% | 0.58 |

These are the numbers for the web app. The command-line tool also averages over
flips and rotations by default, which adds about a point (71.5% model only,
79.3% with the footprint index) but takes four times as long.

Residential and free land are the strong classes. Commercial is the hardest,
because a shop-house looks just like a house from above. The ground truth
was generated from open data rather than traced by hand, so it carries some
noise of its own. [docs/METHODOLOGY.md](docs/METHODOLOGY.md) covers how the
labels were made, how the model was trained, and all the per-class numbers.

**It is trained on Hyderabad.** It will run anywhere, and the app warns you
when you're outside greater Hyderabad, but elsewhere the results are
indicative only.

## Where your data goes

- **Coordinates** you enter are sent to Google (to fetch the imagery) and to
  a public [Overpass API](https://wiki.openstreetmap.org/wiki/Overpass_API)
  server (to fetch OSM map data for that box). Nothing else leaves your machine.
- **Uploaded images** are processed in memory inside the container and are
  not saved or sent anywhere.
- The service listens on `127.0.0.1` only and has no login. See the next
  section before you expose it to other machines.

## Using it from another machine

There is no authentication, so the port is bound to localhost on purpose. If
UrbanSeg runs on a server and you want to use it from your laptop, tunnel over
SSH instead of opening the port:

```bash
ssh -N -L 5555:127.0.0.1:5555 you@your-server
# now open http://127.0.0.1:5555 on your laptop
```

If you really do want it on your network, put it behind a reverse proxy
with authentication.

## HTTP API

The web app is a thin page over a small JSON API. You can call it directly:

```bash
# segment a location (JSON: stats, base64 overlay, optional GeoJSON)
curl "http://127.0.0.1:5555/segment?lat=17.4435&lng=78.3772&grid=2&geojson=true"

# the overlay with a legend, as a PNG
curl -o overlay.png "http://127.0.0.1:5555/segment.png?lat=17.4435&lng=78.3772"

# segment an image you already have
curl -F file=@my_tile.png http://127.0.0.1:5555/segment-image

# is it up, and what's enabled?
curl http://127.0.0.1:5555/health
```

`grid` is the number of tiles per side, from 1 to 3 (default 2). Add
`fuse=false` to get the raw model output. Interactive docs are at
<http://127.0.0.1:5555/docs>.

## Troubleshooting

**"models/unet_r34.pt is a Git LFS pointer"**: the weights weren't downloaded.
Run `git lfs install && git lfs pull`.

**"cannot reach the Docker daemon"**: start Docker Desktop, or on Linux run
`sudo systemctl start docker`. If your user isn't in the `docker` group,
`run.sh` falls back to `sudo docker` and may ask for your password.

**"could not start the container … port already in use"**: something else is
on 5555. Use `PORT=5556 ./run.sh`.

**"By coordinates: disabled"**: no key was found. Add `GOOGLE_MAPS_KEY=...` to
`.env` and run `./run.sh` again.

**Coordinate requests fail with HTTP 502 "tile fetch failed"**: the key
is wrong, the Maps Static API isn't enabled for it, or billing isn't set up
on the Google Cloud project.

**The result says "model only" for a coordinate request**: the public Overpass
servers were busy or unreachable, so there was no map data to fuse. Try again
in a minute.

**"ran out of memory" / "was killed" / requests fail with "Failed to fetch"**:
Docker's VM is too small. Give it at least 3 GB (4 GB for the footprint
index): Docker Desktop → Settings → Resources → Memory, or
`colima start --memory 4`. Then run `./run.sh` again.

**Docker installed as a snap (Ubuntu)**: nothing to do. The snap can't read
files outside your home folder, so `run.sh` detects it and streams the build
files to Docker instead.

**The build can't find files, or Docker misreads the build files**: force the
other way of sending them with `BUILD_CONTEXT=stream ./run.sh` (or
`BUILD_CONTEXT=dir ./run.sh`).

**Apple Silicon / ARM**: Docker builds a native ARM64 image, and every
dependency (PyTorch included) ships a prebuilt ARM64 wheel, so nothing is
compiled. End-to-end runs have been done on x86-64 Linux.

## Running without Docker

If you'd rather use Python directly (3.12 recommended):

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
uvicorn urbanseg.api:app --host 127.0.0.1 --port 5555
```

There is also a command-line tool that writes the overlay, mask, stats and
GeoJSON to a folder:

```bash
python scripts/segment.py --lat 17.4435 --lng 78.3772 --out outputs/hitech
python scripts/segment.py --image my_tile.png
```

## What's in the repository

```text
run.sh                 the one script you need
Dockerfile             image definition used by run.sh
requirements.txt       pinned Python dependencies (CPU PyTorch)
.env.example           template for your Google Maps key
models/unet_r34.pt     trained weights (Git LFS) + .sha256 checksum
urbanseg/              the package
  api.py               FastAPI service and web page
  infer.py             sliding-window inference, map-data fusion, outputs
  imagery.py           Static Maps tile maths, fetching and stitching
  osm_data.py          OpenStreetMap via Overpass
  open_buildings.py    Google Open Buildings index (build + lookup)
  labels.py            training-label generation rules
  dataset.py, train.py, evaluate.py   training and evaluation
  config.py            classes, colours, training areas, rule vocabularies
scripts/               CLI inference and dataset / label-QC tools
docs/METHODOLOGY.md    how the data, labels and model were produced
```

## Credits and licence

UrbanSeg is released under the [Apache License 2.0](LICENSE).

At runtime it uses data from third parties, each under its own terms:
satellite imagery © Google (Maps Static API, subject to the Google Maps
Platform Terms of Service); map data © OpenStreetMap contributors, under the
ODbL; and, if you build the index, building footprints from Google Open
Buildings, under CC BY 4.0 and ODbL. See [NOTICE](NOTICE).
