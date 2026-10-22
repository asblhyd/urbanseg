"""Shared test setup.

For every test:
- no network: sockets may only reach loopback, DNS lookups of other hosts fail;
- no real API key: config.GOOGLE_MAPS_KEY is blank (use the ``fake_key`` fixture);
- protected data stays read-only: every write location in config points into
  the test's tmp_path, and the session fails if anything under data/ or
  models/ was modified while the tests ran.

Tests that need the local research data are marked ``data`` (and ``model``
for the trained checkpoint) and are skipped when it is not available, e.g. in
CI for the open-source release. A git worktree has no data/ of its own, so
point ``URBANSEG_TEST_DATA_ROOT`` at the main checkout when testing there.
"""
import os
import socket
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _data_root():
    env = os.environ.get("URBANSEG_TEST_DATA_ROOT")
    if env:
        return Path(env)
    for cand in (REPO, *REPO.parents):
        if (cand / "data" / "tiles").is_dir():
            return cand
    return None


DATA_ROOT = _data_root()
DATA = DATA_ROOT / "data" if DATA_ROOT else None
HAVE_DATA = bool(DATA and (DATA / "tiles").is_dir() and (DATA / "masks").is_dir())
MODEL_PATH = DATA_ROOT / "models" / "unet_r34.pt" if DATA_ROOT else None
HAVE_MODEL = bool(MODEL_PATH and MODEL_PATH.is_file())
FAKE_KEY = "AIzaTEST-not-a-real-key-0000000000000"
_SESSION_START = time.time()


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "data" in item.keywords and not HAVE_DATA:
            item.add_marker(pytest.mark.skip(reason="research data/ not available"))
        if "model" in item.keywords and not HAVE_MODEL:
            item.add_marker(pytest.mark.skip(reason="trained checkpoint not available"))


def _is_local(host):
    return isinstance(host, str) and (
        host.startswith("127.") or host in ("localhost", "::1", "0.0.0.0", ""))


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def connect(self, address):
        if self.family == getattr(socket, "AF_UNIX", object()) or _is_local(
                address[0] if isinstance(address, tuple) else address):
            return real_connect(self, address)
        raise RuntimeError(f"network access is blocked in tests: {address!r}")

    def getaddrinfo(host, *a, **k):
        if _is_local(host if isinstance(host, str) else str(host)):
            return real_getaddrinfo(host, *a, **k)
        raise socket.gaierror(f"DNS lookup blocked in tests: {host!r}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    from urbanseg import config, open_buildings

    monkeypatch.setattr(config, "GOOGLE_MAPS_KEY", "")
    if HAVE_DATA:  # read-only inputs
        monkeypatch.setattr(config, "DATA_DIR", DATA)
        monkeypatch.setattr(config, "TILES_DIR", DATA / "tiles")
        monkeypatch.setattr(config, "MASKS_DIR", DATA / "masks")
        monkeypatch.setattr(config, "OPEN_BUILDINGS_DIR", DATA / "open_buildings")
        monkeypatch.setattr(open_buildings, "PARQUET",
                            DATA / "open_buildings" / "hyderabad_buildings.parquet")
    else:
        monkeypatch.setattr(open_buildings, "PARQUET", tmp_path / "no_index.parquet")
    if HAVE_MODEL:
        monkeypatch.setattr(config, "MODELS_DIR", MODEL_PATH.parent)
    # data_v2/ (new areas, later label versions) is read-only like data/;
    # tests that need it point DATA_V2_DIR at it themselves
    monkeypatch.setattr(config, "DATA_V2_DIR", tmp_path / "data_v2")
    monkeypatch.setattr(config, "GOLD_DIR", tmp_path / "gold")
    # everything the code may write goes to tmp (the env var would win over
    # config.OSM_DIR for the live OSM cache)
    monkeypatch.setenv("URBANSEG_OSM_CACHE", str(tmp_path / "osm_live"))
    monkeypatch.setattr(config, "OSM_DIR", tmp_path / "osm")
    monkeypatch.setattr(config, "PREVIEW_DIR", tmp_path / "preview")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    yield


@pytest.fixture
def fake_key(monkeypatch):
    """A dummy Google key; the network stays blocked."""
    from urbanseg import config
    monkeypatch.setattr(config, "GOOGLE_MAPS_KEY", FAKE_KEY)
    return FAKE_KEY


@pytest.fixture(scope="session")
def model():
    if not HAVE_MODEL:
        pytest.skip("trained checkpoint not available")
    import torch
    torch.set_num_threads(int(os.environ.get("URBANSEG_TEST_THREADS", "3")))
    from urbanseg import infer
    return infer.load_model(MODEL_PATH)


def _modified_since(root, t0):
    out = []
    for p in root.rglob("*"):
        try:
            if p.stat().st_mtime > t0:
                out.append(p)
        except FileNotFoundError:
            pass
    return out


def pytest_sessionfinish(session, exitstatus):
    if not DATA_ROOT:
        return
    roots = [r for r in (DATA_ROOT / "data", DATA_ROOT / "data_v2", DATA_ROOT / "models")
             if r.exists()]
    bad = [p for r in roots for p in _modified_since(r, _SESSION_START)]
    if bad:
        session.exitstatus = 1
        print("\nERROR: the test run modified protected files:\n  "
              + "\n  ".join(str(p) for p in bad[:20]))
