"""Unit tests for RemoteAccessor.procedural_download / load orchestration logic.

Uses FakeGEEAccessor (see conftest.py), which reuses the real, already unit-tested
tiling logic from GEEMosaicAccessor but fakes the actual "download" — so these tests
exercise dedup, retry, and error-handling behavior fully offline.
"""

import numpy as np
import pytest
from conftest import patch_value
from odc.geo.geobox import GeoBox


def test_load_downloads_and_places_all_patches(make_fake_accessor):
    accessor = make_fake_accessor()
    accessor.create()

    ds = accessor.load(accessor.extent, persist=True)

    assert sorted(accessor.download_log) == ["0-0", "0-1", "1-0", "1-1"]
    # top-left quadrant (row 0, col 0) should carry patch "0-0"'s value everywhere
    assert np.all(ds.value.values[:10, :10] == patch_value("0-0"))
    assert np.all(ds.value.values[:10, 10:] == patch_value("0-1"))
    assert np.all(ds.value.values[10:, :10] == patch_value("1-0"))
    assert np.all(ds.value.values[10:, 10:] == patch_value("1-1"))


def test_procedural_download_skips_already_loaded_patches(make_fake_accessor):
    accessor = make_fake_accessor()
    accessor.create()

    accessor.procedural_download(accessor.extent, None)
    assert len(accessor.download_log) == 4

    # Requesting the same extent again must not re-download anything
    accessor.procedural_download(accessor.extent, None)
    assert len(accessor.download_log) == 4


def test_procedural_download_only_downloads_missing_patches(make_fake_accessor):
    accessor = make_fake_accessor()
    accessor.create()

    # Only download the top-left tile first
    top_left = GeoBox.from_bbox((-1, 0, 0, 1), crs="EPSG:4326", resolution=0.1).extent
    accessor.procedural_download(top_left, None)
    assert accessor.download_log == ["0-0"]

    # Now request the full extent: only the 3 missing tiles should be downloaded
    accessor.procedural_download(accessor.extent, None)
    assert accessor.download_log == ["0-0", "0-1", "1-0", "1-1"]
    assert set(accessor.loaded_patches()) == {"0-0", "0-1", "1-0", "1-1"}


def test_procedural_download_raises_when_no_adjacent_patches(make_fake_accessor):
    accessor = make_fake_accessor()
    accessor.create()

    far_away = GeoBox.from_bbox((50, 50, 51, 51), crs="EPSG:4326", resolution=0.1).extent

    with pytest.raises(ValueError, match="No adjacent patches found"):
        accessor.procedural_download(far_away, None)

    assert accessor.download_log == []


def test_procedural_download_retries_and_succeeds(make_fake_accessor):
    accessor = make_fake_accessor(fail_plan={"0-0": 2})
    accessor.create()

    accessor.procedural_download(accessor.extent, None)

    assert accessor.download_log.count("0-0") == 3  # 2 failures + 1 success
    assert set(accessor.loaded_patches()) == {"0-0", "0-1", "1-0", "1-1"}


def test_procedural_download_raises_after_exhausting_retries(make_fake_accessor):
    accessor = make_fake_accessor(fail_plan={"0-0": 5})
    accessor.create()

    with pytest.raises(RuntimeError, match="tries to download the tile failed"):
        accessor.procedural_download(accessor.extent, None)

    # "0-0" is first in the (simple, sequential) backend's processing order, so it
    # exhausts all 5 retries and aborts before any other patch is attempted.
    assert accessor.download_log == ["0-0"] * 5
    assert accessor.loaded_patches() == []
