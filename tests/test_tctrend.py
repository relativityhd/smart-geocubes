import logging
import os
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import ee
import icechunk
from odc.geo.geobox import GeoBox
from pytest import approx

import smart_geocubes

# Setup logging
logger = logging.getLogger("smart_geocubes")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


def test_tctrend2019_download():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("tctrend2019.zarr")
        accessor = smart_geocubes.TCTrend2019(storage, backend="threaded")
        tc = accessor.load(geobox, create=True)
        print(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item())
        assert tc.TCB_slope.mean().item() == approx(133.7227277914448)
        assert tc.TCB_slope.min().item() == approx(0)
        assert tc.TCB_slope.max().item() == approx(255)
        assert tc.TCW_slope.mean().item() == approx(127.70839598200597)
        assert tc.TCW_slope.min().item() == approx(0)
        assert tc.TCW_slope.max().item() == approx(255)
        assert tc.TCG_slope.mean().item() == approx(133.26384158176066)
        assert tc.TCG_slope.min().item() == approx(0)
        assert tc.TCG_slope.max().item() == approx(255)
    finally:
        if "tc" in locals():
            del tc
        os.system("rm -rf tctrend2019.zarr")


def test_tctrend2020_download():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("tctrend2020.zarr")
        accessor = smart_geocubes.TCTrend2020(storage, backend="threaded")
        tc = accessor.load(geobox, create=True)
        print(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item())
        assert tc.TCB_slope.mean().item() == approx(134.90014447669557)
        assert tc.TCB_slope.min().item() == approx(0)
        assert tc.TCB_slope.max().item() == approx(255)
        assert tc.TCW_slope.mean().item() == approx(123.24139237727539)
        assert tc.TCW_slope.min().item() == approx(0)
        assert tc.TCW_slope.max().item() == approx(255)
        assert tc.TCG_slope.mean().item() == approx(130.60853312670162)
        assert tc.TCG_slope.min().item() == approx(0)
        assert tc.TCG_slope.max().item() == approx(255)
    finally:
        if "tc" in locals():
            del tc
        os.system("rm -rf tctrend2020.zarr")


def test_tctrend2022_download():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("tctrend2022.zarr")
        accessor = smart_geocubes.TCTrend2022(storage, backend="threaded")
        tc = accessor.load(geobox, create=True)
        print(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item())
        assert tc.TCB_slope.mean().item() == approx(135.21512058744273)
        assert tc.TCB_slope.min().item() == approx(0)
        assert tc.TCB_slope.max().item() == approx(255)
        assert tc.TCW_slope.mean().item() == approx(120.57892658359583)
        assert tc.TCW_slope.min().item() == approx(0)
        assert tc.TCW_slope.max().item() == approx(255)
        assert tc.TCG_slope.mean().item() == approx(128.1437808089185)
        assert tc.TCG_slope.min().item() == approx(0)
        assert tc.TCG_slope.max().item() == approx(255)
    finally:
        if "tc" in locals():
            del tc
        os.system("rm -rf tctrend2022.zarr")


def test_tctrend2024_download():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("tctrend2024.zarr")
        accessor = smart_geocubes.TCTrend2024(storage, backend="threaded")
        tc = accessor.load(geobox, create=True)
        print(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item())
        assert tc.TCB_slope.mean().item() == approx(129.70483792940837)
        assert tc.TCB_slope.min().item() == approx(0)
        assert tc.TCB_slope.max().item() == approx(255)
        assert tc.TCW_slope.mean().item() == approx(127.15408459182393)
        assert tc.TCW_slope.min().item() == approx(0)
        assert tc.TCW_slope.max().item() == approx(255)
        assert tc.TCG_slope.mean().item() == approx(129.2719793031435)
        assert tc.TCG_slope.min().item() == approx(0)
        assert tc.TCG_slope.max().item() == approx(255)
    finally:
        if "tc" in locals():
            del tc
        os.system("rm -rf tctrend2024.zarr")


Stats = namedtuple("Stats", ["mean", "min", "max"])


def test_tctrend2024_download_threaded():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        storage = icechunk.local_filesystem_storage("tctrend2024.zarr")
        accessor = smart_geocubes.TCTrend2024(storage, backend="threaded")
        accessor.create(overwrite=True)

        def _task(i, geobox: GeoBox) -> tuple[int, tuple[Stats, Stats, Stats]]:
            tc = accessor.load(geobox)
            return i, (
                Stats(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item()),
                Stats(tc.TCW_slope.mean().item(), tc.TCW_slope.min().item(), tc.TCW_slope.max().item()),
                Stats(tc.TCG_slope.mean().item(), tc.TCG_slope.min().item(), tc.TCG_slope.max().item()),
            )

        geoboxes = [
            GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000)),
            GeoBox.from_bbox((151, 65, 152, 66), shape=(1000, 1000)),
            GeoBox.from_bbox((152, 65, 153, 66), shape=(1000, 1000)),
        ]

        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(_task, list(range(3)), geoboxes))

        for i, (tcb, tcw, tcg) in results:
            if i != 0:
                continue
            assert tcb.mean == approx(129.70483792940837)
            assert tcb.min == approx(0)
            assert tcb.max == approx(255)
            assert tcw.mean == approx(127.15408459182393)
            assert tcw.min == approx(0)
            assert tcw.max == approx(255)
            assert tcg.mean == approx(129.2719793031435)
            assert tcg.min == approx(0)
            assert tcg.max == approx(255)
    finally:
        os.system("rm -rf tctrend2024.zarr")


def _mp_task(i, geobox: GeoBox) -> tuple[int, tuple[Stats, Stats, Stats]]:
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    storage = icechunk.local_filesystem_storage("tctrend2024.zarr")
    accessor = smart_geocubes.TCTrend2024(storage, backend="simple")
    tc = accessor.load(geobox)
    return i, (
        Stats(tc.TCB_slope.mean().item(), tc.TCB_slope.min().item(), tc.TCB_slope.max().item()),
        Stats(tc.TCW_slope.mean().item(), tc.TCW_slope.min().item(), tc.TCW_slope.max().item()),
        Stats(tc.TCG_slope.mean().item(), tc.TCG_slope.min().item(), tc.TCG_slope.max().item()),
    )


def test_tctrend2024_download_blocking_processes():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        # This test fails with the "spawn" method, however, it's not possible to set the start method from a test
        # So this test stays broken for now
        # mp.set_start_method("forkserver")
        storage = icechunk.local_filesystem_storage("tctrend2024.zarr")
        accessor = smart_geocubes.TCTrend2024(storage)
        accessor.create(overwrite=True)

        geoboxes = [
            GeoBox.from_bbox((150, 65, 151, 66), shape=(1000, 1000)),
            GeoBox.from_bbox((151, 65, 152, 66), shape=(1000, 1000)),
            GeoBox.from_bbox((152, 65, 153, 66), shape=(1000, 1000)),
        ]

        with ProcessPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(_mp_task, list(range(3)), geoboxes))

        for i, (tcb, tcw, tcg) in results:
            if i != 0:
                continue
            tcb, tcw, tcg = Stats(*tcb), Stats(*tcw), Stats(*tcg)
            assert tcb.mean == approx(129.70483792940837)
            assert tcb.min == approx(0)
            assert tcb.max == approx(255)
            assert tcw.mean == approx(127.15408459182393)
            assert tcw.min == approx(0)
            assert tcw.max == approx(255)
            assert tcg.mean == approx(129.2719793031435)
            assert tcg.min == approx(0)
            assert tcg.max == approx(255)
    finally:
        os.system("rm -rf tctrend2024.zarr")
