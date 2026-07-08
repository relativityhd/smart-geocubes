import shutil
from typing import ClassVar

import icechunk
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from odc.geo.geobox import GeoBox

from smart_geocubes.core import RemoteAccessor


class AccessorDegree(RemoteAccessor):
    extent: GeoBox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.0001)
    temporal_extent: None = None
    chunk_size: int = 100
    channels: ClassVar[list] = ["red", "green", "blue"]
    _channels_meta: ClassVar[dict] = {"red": {"nodata": 0}, "green": {"nodata": 0}, "blue": {"nodata": 0}}
    _channels_encoding: ClassVar[dict] = {
        "red": {"dtype": "uint16"},
        "green": {"dtype": "uint16"},
        "blue": {"dtype": "uint16"},
    }

    def adjacent_patches(self, roi, toi):
        pass

    def download_patch(self, idx):
        pass

    def current_state(self):
        pass

    def visualize_state(self, ax=None):
        pass


class AccessorMeter(RemoteAccessor):
    extent: GeoBox = GeoBox.from_bbox((-10000, -10000, 10000, 10000), crs="EPSG:3857", resolution=1)
    temporal_extent: pd.DatetimeIndex = pd.date_range("2020-01-01", periods=3, freq="D")
    chunk_size: int = 100
    channels: ClassVar[list] = ["red", "green", "blue"]
    _channels_meta: ClassVar[dict] = {"red": {"nodata": 0}, "green": {"nodata": 0}, "blue": {"nodata": 0}}
    _channels_encoding: ClassVar[dict] = {
        "red": {"dtype": "uint16"},
        "green": {"dtype": "uint16"},
        "blue": {"dtype": "uint16"},
    }

    def adjacent_patches(self, roi, toi):
        pass

    def download_patch(self, idx):
        pass

    def current_state(self):
        pass

    def visualize_state(self, ax=None):
        pass


def test_create_datacube():
    for accessor_cls in (AccessorDegree, AccessorMeter):
        storage = icechunk.local_filesystem_storage("test.zarr")
        accessor = accessor_cls(storage)

        try:
            accessor.create()

            ds = accessor.open_xarray()
            # ds = xr.open_zarr(storage, chunks={}, consolidated=False)
            print(ds.sizes)
            if isinstance(accessor, AccessorMeter):
                assert ds.sizes == {"x": 20000, "y": 20000, "time": 3}
                assert ds.chunksizes == {
                    "x": tuple([100] * 200),
                    "y": tuple([100] * 200),
                    "time": (1, 1, 1),
                }
            elif isinstance(accessor, AccessorDegree):
                assert ds.sizes == {"x": 20000, "y": 20000}
                assert ds.chunksizes == {
                    "x": tuple([100] * 200),
                    "y": tuple([100] * 200),
                }
            assert "red" in ds
            assert "green" in ds
            assert "blue" in ds
            assert "x" in ds.coords
            assert "y" in ds.coords

            assert ds.red.attrs["nodata"] == 0
            assert ds.green.attrs["nodata"] == 0
            assert ds.blue.attrs["nodata"] == 0

            # Coordinates are pixel centers (offset by half a resolution from the extent's
            # edges), and y is descending (north-up convention: the first row is the
            # northernmost/highest y). assert_allclose (not assert_array_equal) because
            # np.arange's float accumulation drifts slightly from the affine-based values
            # odc.geo actually produces.
            if isinstance(accessor, AccessorMeter):
                assert_allclose(
                    ds.coords["x"].values,
                    np.arange(-10000, 10000, 1) + 0.5,
                )
                assert_allclose(
                    ds.coords["y"].values,
                    np.arange(10000, -10000, -1) - 0.5,
                )
            elif isinstance(accessor, AccessorDegree):
                assert_allclose(
                    ds.coords["x"].values,
                    np.arange(-1, 1, 0.0001) + 0.0001 / 2,
                )
                assert_allclose(
                    ds.coords["y"].values,
                    np.arange(1, -1, -0.0001) - 0.0001 / 2,
                )
        finally:
            if "ds" in locals():
                del ds
            shutil.rmtree("test.zarr", ignore_errors=True)


class AccessorTiny(RemoteAccessor):
    """A minimal 4x4px accessor, used only to pin the pixel-center coordinate convention."""

    extent: GeoBox = GeoBox.from_bbox((0, 0, 4, 4), crs="EPSG:3857", resolution=1)
    temporal_extent: None = None
    chunk_size: int = 4
    channels: ClassVar[list] = ["red"]
    _channels_meta: ClassVar[dict] = {"red": {}}
    _channels_encoding: ClassVar[dict] = {"red": {"dtype": "uint16"}}

    def adjacent_patches(self, roi, toi):
        pass

    def download_patch(self, idx):
        pass

    def current_state(self):
        pass

    def visualize_state(self, ax=None):
        pass


def test_datacube_coordinates_are_pixel_centers():
    # Regression test: the coordinate assertions in test_create_datacube above used to
    # assume grid-edge coordinates (e.g. x starting exactly at -1.0) and an ascending y
    # axis, neither of which matches odc.geo's actual convention (pixel centers, y
    # descending). That bug was masked for a while by an unrelated `isinstance` bug that
    # made the assertions unreachable. This test pins the convention down explicitly with
    # a small, easy-to-verify-by-eye grid.
    storage = icechunk.local_filesystem_storage("test_tiny.zarr")
    accessor = AccessorTiny(storage)
    try:
        accessor.create()
        ds = accessor.open_xarray()
        assert_allclose(ds.coords["x"].values, [0.5, 1.5, 2.5, 3.5])
        assert_allclose(ds.coords["y"].values, [3.5, 2.5, 1.5, 0.5])
    finally:
        if "ds" in locals():
            del ds
        shutil.rmtree("test_tiny.zarr", ignore_errors=True)


def test_create_datacube_exists():
    storage = icechunk.local_filesystem_storage("test.zarr")
    accessor = AccessorDegree(storage)
    try:
        accessor.create()
        with pytest.raises(FileExistsError):
            accessor.create()
    finally:
        shutil.rmtree("test.zarr", ignore_errors=True)


def test_create_datacube_overwrite():
    storage = icechunk.local_filesystem_storage("test.zarr")
    accessor = AccessorDegree(storage)
    try:
        accessor.create()
        accessor.create(overwrite=True)
    finally:
        shutil.rmtree("test.zarr", ignore_errors=True)
