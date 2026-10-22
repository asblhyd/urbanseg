FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# rasterio's wheel links against system libexpat, absent from slim images
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY urbanseg/ urbanseg/
# both models stay selectable (URBANSEG_MODEL, or per run in the web app)
COPY models/unet_r34.pt models/unet_r34_v2.pt models/

# .cache holds live OpenStreetMap answers, data/open_buildings the optional
# footprint index and data_v2/gold the hand-checked gold set; run.sh mounts
# them as named volumes so they survive restarts
RUN useradd --create-home --uid 10001 app \
    && mkdir -p data/osm data/open_buildings data_v2/gold .cache/osm_live \
    && chown -R app:app data data_v2 .cache
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

# single worker: each request already uses every core, and the model is memory-heavy
CMD ["uvicorn", "urbanseg.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
