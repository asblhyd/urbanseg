"""The guarantees conftest.py promises every other test."""
import socket

import pytest
import requests

import conftest
from urbanseg import config


def test_network_is_blocked():
    with pytest.raises((RuntimeError, OSError, requests.exceptions.ConnectionError)):
        requests.get("https://maps.googleapis.com/maps/api/staticmap", timeout=2)
    with pytest.raises(OSError):
        socket.getaddrinfo("example.com", 443)


def test_real_key_is_never_visible(fake_key):
    assert config.GOOGLE_MAPS_KEY == conftest.FAKE_KEY


def test_write_paths_point_into_tmp(tmp_path):
    for p in (config.OSM_DIR, config.PREVIEW_DIR, config.REPORTS_DIR):
        assert str(p).startswith(str(tmp_path))


@pytest.mark.data
def test_read_paths_point_at_research_data():
    assert (config.TILES_DIR / "madhapur" / "meta.json").is_file()
    assert (config.MASKS_DIR / "madhapur.png").is_file()
