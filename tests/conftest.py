"""Shared test fixtures for smart-geocubes tests."""

import itertools
from typing import ClassVar

import icechunk
import odc.geo.xr
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.gee import GEEMosaicAccessor
from smart_geocubes.core.patches import PatchIndex


def patch_value(patch_id: str) -> float:
    """Compute the deterministic fill value for a non-temporal FakeGEEAccessor patch id.

    Args:
        patch_id (str): A non-temporal patch id, e.g. "1-2".

    Returns:
        float: The fill value, e.g. "1-2" -> 102.0.

    """
    row, col = (int(p) for p in patch_id.split("-"))
    return float(row * 100 + col)


class FakeGEEAccessor(GEEMosaicAccessor):
    """A GEEMosaicAccessor with a fully synthetic, offline `download_patch`.

    Reuses the real (and separately unit-tested, see test_gee_accessor.py) tiling
    logic from GEEMosaicAccessor.adjacent_patches, so tests exercise realistic patch
    ids and geoboxes. Only the actual "download" is faked: each patch is filled with
    a value derived deterministically from its id (see `patch_value`), and downloads
    can be scripted to fail a number of times before succeeding via `fail_plan`.
    """

    extent: GeoBox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.1)  # 20x20px
    temporal_extent = None
    chunk_size = 10  # -> 2x2 grid of patches: "0-0", "0-1", "1-0", "1-1"
    channels: ClassVar[list] = ["value"]
    _channels_meta: ClassVar[dict] = {"value": {}}
    _channels_encoding: ClassVar[dict] = {"value": {"dtype": "float32"}}

    def __init__(self, *args, fail_plan: dict[str, int] | None = None, **kwargs):
        """Initialize the fake accessor.

        Args:
            fail_plan (dict[str, int] | None): Maps a patch id to the number of times
                its download should raise before succeeding.
            *args: Forwarded to RemoteAccessor.
            **kwargs: Forwarded to RemoteAccessor.

        """
        self.download_log: list[str] = []
        self._fail_plan = dict(fail_plan or {})
        super().__init__(*args, **kwargs)

    def download_patch(self, idx: PatchIndex) -> xr.Dataset:
        self.download_log.append(idx.id)
        remaining = self._fail_plan.get(idx.id, 0)
        if remaining > 0:
            self._fail_plan[idx.id] = remaining - 1
            raise RuntimeError(f"Simulated failure for patch {idx.id} ({remaining} left)")
        geobox = idx.item.geobox
        value = patch_value(idx.id)
        return xr.Dataset({"value": odc.geo.xr.xr_zeros(geobox, dtype="float32", always_yx=True) + value})

    def current_state(self):
        return None

    def visualize_state(self, ax=None):
        pass


@pytest.fixture
def make_fake_accessor(tmp_path):
    """Build a factory for FakeGEEAccessors, each backed by a fresh local icechunk store.

    Can be called multiple times within a single test; each call gets its own store.

    Returns:
        Callable[..., FakeGEEAccessor]: A factory taking `backend` and `fail_plan`.

    """
    counter = itertools.count()

    def _make(backend: str = "simple", fail_plan: dict[str, int] | None = None) -> FakeGEEAccessor:
        storage = icechunk.local_filesystem_storage(str(tmp_path / f"fake_{next(counter)}.icechunk"))
        return FakeGEEAccessor(storage, backend=backend, fail_plan=fail_plan)

    return _make
