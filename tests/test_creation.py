import os
from typing import ClassVar

import icechunk
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors import RemoteAccessor


class AccessorDegree(RemoteAccessor):
    extent: GeoBox = GeoBox.from_bbox([-1, -1, 1, 1], crs="EPSG:4326", resolution=0.0001)
    chunk_size: int = 100
    channels: ClassVar[list] = ["red", "green", "blue"]
    channels_meta: ClassVar[dict] = {"red": {"nodata": 0}, "green": {"nodata": 0}, "blue": {"nodata": 0}}
    channels_encoding: ClassVar[dict] = {
        "red": {"dtype": "uint16"},
        "green": {"dtype": "uint16"},
        "blue": {"dtype": "uint16"},
    }

    def procedural_download(self, geobox):
        pass

    def current_state(self):
        pass

    def visualize_state(self, ax=None):
        pass


class AccessorMeter(RemoteAccessor):
    extent: GeoBox = GeoBox.from_bbox([-10000, -10000, 10000, 10000], crs="EPSG:3857", resolution=1)
    chunk_size: int = 100
    channels: ClassVar[list] = ["red", "green", "blue"]
    channels_meta: ClassVar[dict] = {"red": {"nodata": 0}, "green": {"nodata": 0}, "blue": {"nodata": 0}}
    channels_encoding: ClassVar[dict] = {
        "red": {"dtype": "uint16"},
        "green": {"dtype": "uint16"},
        "blue": {"dtype": "uint16"},
    }

    def procedural_download(self, geobox):
        pass

    def current_state(self):
        pass

    def visualize_state(self, ax=None):
        pass


def test_create_datacube():
    for accessor_cls in [AccessorDegree, AccessorMeter]:
        storage = icechunk.local_filesystem_storage("test.zarr")
        accessor = accessor_cls(storage)

        accessor.create()

        ds = xr.open_zarr("test.zarr", chunks={}, consolidated=False)
        try:
            print(ds.sizes)
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

            if isinstance(accessor_cls, AccessorMeter):
                assert_array_equal(
                    ds.coords["x"].values,
                    np.arange(-10000, 10000, 1),
                )
                assert_array_equal(
                    ds.coords["y"].values,
                    np.arange(-10000, 10000, 1),
                )
            elif isinstance(accessor_cls, AccessorDegree):
                assert_array_equal(
                    ds.coords["x"].values,
                    np.arange(-1, 1, 0.0001),
                )
                assert_array_equal(
                    ds.coords["y"].values,
                    np.arange(-1, 1, 0.0001),
                )
        finally:
            del ds
            os.system("rm -rf test.zarr")


def test_create_datacube_exists():
    storage = icechunk.local_filesystem_storage("test.zarr")
    accessor = AccessorDegree(storage)
    try:
        accessor.create()
        with pytest.raises(FileExistsError):
            accessor.create()
    finally:
        os.system("rm -rf test.zarr")


def test_create_datacube_overwrite():
    storage = icechunk.local_filesystem_storage("test.zarr")
    accessor = AccessorDegree(storage)
    try:
        accessor.create()
        accessor.create(overwrite=True)
    finally:
        os.system("rm -rf test.zarr")
