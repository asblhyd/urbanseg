#!/usr/bin/env bash
# UrbanSeg launcher: builds the Docker image and serves the web app on
# http://127.0.0.1:5555 (reachable from this machine only).
#
#   ./run.sh                    build (cached after the first time) and start
#   ./run.sh --with-footprints  same, plus build the optional footprint index
#                               (one-time ~4.8 GB download, Google Open Buildings)
#   ./run.sh stop               stop and remove the container
#   ./run.sh logs               follow the service logs
#
# PORT=6000 ./run.sh serves on a different local port.
# BUILD_CONTEXT=dir|stream overrides how the build files are sent to Docker.

# `sh run.sh` runs dash on Debian/Ubuntu; everything below needs bash
if [ -z "${BASH_VERSION:-}" ]; then exec bash "$0" "$@"; fi

set -euo pipefail

cd "$(dirname "$0")"

IMAGE="urbanseg:latest"
NAME="urbanseg"
VOLUME="urbanseg-footprints"
MODEL="models/unet_r34.pt"
HOST="127.0.0.1"
PORT="${PORT:-5555}"
NEED_GB_SERVICE=3      # 'Large' requests peak at ~1.5 GB, plus VM/engine overhead
NEED_GB_FOOTPRINTS=4   # measured peak 2.4 GB while filtering the Open Buildings download

# Git Bash on Windows rewrites "name:/path" arguments such as -v volume:/app
export MSYS_NO_PATHCONV=1

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die() { printf '\nerror: %s\n' "$*" >&2; exit 1; }
mem_hint() {
  echo "Docker Desktop: Settings > Resources > Memory. Colima: colima start --memory $1."
}

# --- Docker ---------------------------------------------------------------
command -v docker >/dev/null 2>&1 \
  || die "Docker is not installed. Install Docker Desktop (macOS/Windows) or Docker Engine (Linux): https://docs.docker.com/get-docker/"
if docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif [ "$(uname -s)" = Linux ] && command -v sudo >/dev/null 2>&1 && sudo docker info >/dev/null 2>&1; then
  # not in the docker group: go through sudo but keep the Maps key
  DOCKER=(sudo --preserve-env=GOOGLE_MAPS_KEY docker)
else
  die "cannot reach the Docker daemon. Start Docker Desktop (or Colima, or 'sudo systemctl start docker' on Linux), wait until it is running, then re-run."
fi

WITH_FOOTPRINTS=0
case "${1:-}" in
  "") ;;
  --with-footprints) WITH_FOOTPRINTS=1 ;;
  stop)
    if "${DOCKER[@]}" container inspect "$NAME" >/dev/null 2>&1; then
      "${DOCKER[@]}" rm -f "$NAME" >/dev/null
      echo "stopped"
    else
      echo "not running"
    fi
    exit 0 ;;
  logs) exec "${DOCKER[@]}" logs -f "$NAME" ;;
  -h|--help) sed -n '2,12s/^# \{0,1\}//p' "$0"; exit 0 ;;
  *) die "unknown option '$1' (see ./run.sh --help)" ;;
esac

# memory available to containers (the VM size on Docker Desktop / Colima)
mem_bytes="$("${DOCKER[@]}" info --format '{{.MemTotal}}' 2>/dev/null || true)"
case "$mem_bytes" in ''|*[!0-9]*) mem_bytes=0 ;; esac
mem_gb=$(( (mem_bytes + 500000000) / 1000000000 ))

# --- model weights (tracked with Git LFS) ----------------------------------
say "Checking model weights"
[ -f "$MODEL" ] || die "$MODEL not found. Clone the repo with Git LFS installed (see README)."
if head -c 200 "$MODEL" | grep -aq "git-lfs.github.com/spec"; then
  die "$MODEL is a Git LFS pointer, not the weights. Run: git lfs install && git lfs pull"
fi
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum --check --status "$MODEL.sha256" || die "$MODEL does not match its checksum. Run: git lfs pull"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 --check --status "$MODEL.sha256" || die "$MODEL does not match its checksum. Run: git lfs pull"
fi
echo "ok"

# --- Google Maps key (optional) -------------------------------------------
say "Google Maps API key"
if [ -z "${GOOGLE_MAPS_KEY:-}" ] && [ -f .env ]; then
  GOOGLE_MAPS_KEY="$(sed -n 's/^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}GOOGLE_MAPS_KEY[[:space:]]*=//p' .env \
    | tail -n 1 | tr -d "\"' \r")"
fi
if [ -z "${GOOGLE_MAPS_KEY:-}" ] && [ -t 0 ]; then
  echo "Only needed for the 'By coordinates' mode. Press Enter to skip; image upload works without it."
  read -r -s -p "Paste key: " GOOGLE_MAPS_KEY
  echo
  if [ -n "$GOOGLE_MAPS_KEY" ]; then
    { grep -v '^[[:space:]]*\(export[[:space:]]\{1,\}\)\{0,1\}GOOGLE_MAPS_KEY[[:space:]]*=' .env 2>/dev/null || true
      printf 'GOOGLE_MAPS_KEY=%s\n' "$GOOGLE_MAPS_KEY"; } > .env.tmp
    mv .env.tmp .env
    chmod 600 .env
    echo "saved to .env (git-ignored)"
  fi
fi
export GOOGLE_MAPS_KEY="${GOOGLE_MAPS_KEY:-}"
[ -n "$GOOGLE_MAPS_KEY" ] && echo "found" || echo "none: coordinate mode will be disabled"

# --- image ----------------------------------------------------------------
# Docker Desktop, Docker Engine, Colima etc. build straight from this folder
# (.dockerignore keeps the context to what the image needs). Snap-packaged
# Docker cannot read files outside $HOME, so there the files are streamed as a
# gzipped tarball instead; gzip makes Docker treat the stream as a context
# whichever tar produced it (macOS bsdtar output alone is misread as a Dockerfile).
if [ -z "${BUILD_CONTEXT:-}" ]; then
  BUILD_CONTEXT=dir
  docker_bin="$(command -v docker)"
  resolved="$(readlink -f "$docker_bin" 2>/dev/null || echo "$docker_bin")"
  case "$docker_bin:$resolved" in
    /snap/*|*:/snap/*|*:/usr/bin/snap) BUILD_CONTEXT=stream ;;
  esac
fi
say "Building the Docker image (first time: ~1.5 GB download, a few minutes; cached after)"
case "$BUILD_CONTEXT" in
  dir)
    "${DOCKER[@]}" build -t "$IMAGE" . ;;
  stream)
    COPYFILE_DISABLE=1 tar -czf - --exclude='__pycache__' \
        Dockerfile requirements.txt urbanseg "$MODEL" \
      | "${DOCKER[@]}" build -t "$IMAGE" - ;;
  *)
    die "BUILD_CONTEXT must be 'dir' or 'stream', not '$BUILD_CONTEXT'" ;;
esac || die "the Docker build failed (output above). Network errors: just re-run, finished steps are cached. If Docker could not read the files, try: BUILD_CONTEXT=stream ./run.sh"

# --- optional footprint index -----------------------------------------------
in_volume() {
  "${DOCKER[@]}" run --rm --no-healthcheck \
    -v "$VOLUME:/app/data/open_buildings" "$IMAGE" "$@"
}
if [ "$WITH_FOOTPRINTS" = 1 ]; then
  if in_volume test -f data/open_buildings/hyderabad_buildings.parquet; then
    say "Footprint index already built"
  else
    if [ "$mem_gb" -gt 0 ] && [ "$mem_gb" -lt "$NEED_GB_FOOTPRINTS" ]; then
      die "building the footprint index needs about $NEED_GB_FOOTPRINTS GB of memory for Docker, which has ${mem_gb} GB. $(mem_hint "$NEED_GB_FOOTPRINTS") Then re-run ./run.sh --with-footprints. (The app itself works without the index: ./run.sh)"
    fi
    say "Building the footprint index (one-time: ~4.8 GB download, needs ~6 GB disk; 10-40 min)"
    rc=0
    in_volume python -m urbanseg.open_buildings || rc=$?
    if [ "$rc" = 137 ]; then
      die "the footprint build was killed, almost certainly for lack of memory. Give Docker at least $NEED_GB_FOOTPRINTS GB ($(mem_hint "$NEED_GB_FOOTPRINTS")) and re-run ./run.sh --with-footprints; a finished download is reused."
    elif [ "$rc" != 0 ]; then
      die "the footprint build failed (output above). Re-run ./run.sh --with-footprints to retry; a finished download is reused. The app works without the index: ./run.sh"
    fi
  fi
fi

if [ "$mem_gb" -gt 0 ] && [ "$mem_gb" -lt "$NEED_GB_SERVICE" ]; then
  warn "Docker has ${mem_gb} GB of memory. A 'Large' request uses about 1.5 GB, so it may fail below $NEED_GB_SERVICE GB. To raise it: $(mem_hint 4)"
fi

# --- start ----------------------------------------------------------------
say "Starting UrbanSeg"
"${DOCKER[@]}" rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER[@]}" run -d --name "$NAME" \
    -p "$HOST:$PORT:8000" \
    -e GOOGLE_MAPS_KEY \
    -v "$VOLUME:/app/data/open_buildings" \
    --restart unless-stopped \
    "$IMAGE" >/dev/null \
  || die "could not start the container. Is port $PORT already in use? Try: PORT=5556 ./run.sh"

printf 'waiting for the service'
health=""
for _ in $(seq 1 90); do
  state="$("${DOCKER[@]}" container inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo gone)"
  if [ "$state" != running ]; then
    echo
    "${DOCKER[@]}" logs --tail 30 "$NAME" 2>&1 || true
    oom="$("${DOCKER[@]}" container inspect -f '{{.State.OOMKilled}}' "$NAME" 2>/dev/null || true)"
    [ "$oom" = true ] && die "the container ran out of memory while starting. $(mem_hint 4)"
    die "the container stopped during startup (last log lines above)"
  fi
  if health="$("${DOCKER[@]}" exec "$NAME" python -c \
      "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read().decode())" \
      2>/dev/null)"; then
    break
  fi
  printf '.'
  sleep 2
done
echo
[ -n "$health" ] || die "the service did not come up within 3 minutes. See: ./run.sh logs"

case "$health" in *'"coordinates_enabled":true'*) coords="enabled" ;;
  *) coords="disabled (no Google Maps key; image upload still works)" ;; esac
case "$health" in *'"footprint_index":true'*) fps="built" ;;
  *) fps="not built (optional: ./run.sh --with-footprints)" ;; esac

cat <<EOF

UrbanSeg is running:  http://$HOST:$PORT

  By coordinates   : $coords
  Footprint index  : $fps

  Logs: ./run.sh logs      Stop: ./run.sh stop
EOF
