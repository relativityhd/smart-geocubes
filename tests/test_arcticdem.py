import multiprocessing as mp
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import icechunk
from numpy.testing import assert_almost_equal
from odc.geo.geobox import GeoBox

import smart_geocubes


def test_arcticdem32m_download():
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
        accessor = smart_geocubes.ArcticDEM32m(storage)
        adem = accessor.load(geobox, create=True)
        assert adem.dem.mean() == 102.0299
        assert adem.dem.min() == 46.34375
        assert adem.dem.max() == 483.83594
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["x"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["x"].values / 1_000_000,
            decimal=4,
        )
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["y"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["y"].values / 1_000_000,
            decimal=4,
        )
    finally:
        del adem
        os.system("rm -rf arcticdem_32m.zarr")


def test_arcticdem2m_download():
    try:
        geobox = GeoBox.from_bbox((150, 65, 150.1, 65.1), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("arcticdem_2m.zarr")
        accessor = smart_geocubes.ArcticDEM2m(storage)
        adem = accessor.load(geobox, create=True)
        print(adem.dem.mean(), adem.dem.min(), adem.dem.max())
        assert adem.dem.mean() == 203.03644
        assert adem.dem.min() == 149.7421
        assert adem.dem.max() == 285.5547
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["x"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["x"].values / 1_000_000,
            decimal=4,
        )
        assert_almost_equal(
            adem.odc.geobox.center_pixel.coords["y"].values / 1_000_000,
            geobox.to_crs("EPSG:3413").center_pixel.coords["y"].values / 1_000_000,
            decimal=4,
        )
    finally:
        del adem
        os.system("rm -rf arcticdem_2m.zarr")


Stats = namedtuple("Stats", ["mean", "min", "max"])


def test_arcticdem_download_threaded():
    try:
        storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
        accessor = smart_geocubes.ArcticDEM32m(storage)
        accessor.create(overwrite=True)

        def _task(i, geobox: GeoBox) -> Stats:
            adem = accessor.load(geobox, concurrency_mode="threading")
            return i, Stats(adem.dem.mean(), adem.dem.min(), adem.dem.max())

        geoboxes = [
            GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000)),
            GeoBox.from_bbox((150.5, 65, 151.5, 65.5), shape=(1000, 1000)),
            GeoBox.from_bbox((151, 65, 152, 65.5), shape=(1000, 1000)),
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(_task, list(range(3)), geoboxes))

        for i, result in results:
            if i != 0:
                continue
            assert result.mean == 102.0299
            assert result.min == 46.34375
            assert result.max == 483.83594
    finally:
        os.system("rm -rf arcticdem_32m.zarr")


def _mp_task(i, geobox: GeoBox) -> tuple[int, Stats]:
    storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
    accessor = smart_geocubes.ArcticDEM32m(storage)
    adem = accessor.load(geobox)
    return i, (adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())


def test_arcticdem_download_blocking_processes():
    try:
        mp.set_start_method("forkserver")
        storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
        accessor = smart_geocubes.ArcticDEM32m(storage)
        accessor.create(overwrite=True)

        geoboxes = [
            GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000)),
            GeoBox.from_bbox((150.5, 65, 151.5, 65.5), shape=(1000, 1000)),
            GeoBox.from_bbox((151, 65, 152, 65.5), shape=(1000, 1000)),
        ]

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(_mp_task, list(range(3)), geoboxes))

        for i, result in results:
            if i != 0:
                continue
            result = Stats(*result)
            assert result.mean == 102.0299
            assert result.min == 46.34375
            assert result.max == 483.83594
    finally:
        os.system("rm -rf arcticdem_32m.zarr")
