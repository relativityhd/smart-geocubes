import logging
import shutil
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from types import SimpleNamespace

import geopandas as gpd
import icechunk
import pytest
from numpy.testing import assert_almost_equal
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pytest import approx
from shapely.geometry import box

import smart_geocubes
from smart_geocubes.datasets.arcticdem import (
    ArcticDEMABC,
    LazyStacPatchIndex,
    _get_stac_url,
)

# Setup logging
logger = logging.getLogger("smart_geocubes")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)


# --- _get_stac_url: pure URL construction, no I/O -----------------------------


@pytest.mark.parametrize(
    "dem_id,res,expected",
    [
        (
            "36_24_32m_v4.1",
            "32m",
            "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-32m/items/36_24_32m_v4.1",
        ),
        (
            "12_08_10m_v4.1",
            "10m",
            "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-10m/items/12_08_10m_v4.1",
        ),
        (
            "45_11_2m_v4.1",
            "2m",
            "https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-2m/items/45_11_2m_v4.1",
        ),
    ],
)
def test_get_stac_url(dem_id, res, expected):
    assert _get_stac_url(dem_id, res) == expected


# --- ArcticDEMABC.adjacent_patches: boundary-mocked local parquet extent lookup ---


@pytest.fixture
def fake_arcticdem_accessor(tmp_path):
    gdf = gpd.GeoDataFrame(
        {"dem_id": ["36_24_32m_v4.1", "37_24_32m_v4.1"]},
        geometry=[box(150, 65, 151, 66), box(151, 65, 152, 66)],
        crs="EPSG:4326",
    )
    gdf.to_parquet(tmp_path / "ArcticDEM_Mosaic_Index_v4_1_32m.parquet")

    return SimpleNamespace(
        assert_created=lambda: None,
        extent=GeoBox.from_bbox((-100_000, -100_000, 100_000, 100_000), crs="epsg:3413", resolution=32),
        _aux_dir=tmp_path,
    )


def test_adjacent_patches_geobox_roi_returns_intersecting_tile_only(fake_arcticdem_accessor):
    roi = GeoBox.from_bbox((150, 65, 150.5, 65.5), crs="EPSG:4326", resolution=0.01)

    patches = ArcticDEMABC.adjacent_patches(fake_arcticdem_accessor, roi, None)

    assert [p.id for p in patches] == ["36_24_32m_v4.1"]
    assert isinstance(patches[0], LazyStacPatchIndex)
    assert patches[0].stac_file == _get_stac_url("36_24_32m_v4.1", "32m")


def test_adjacent_patches_geometry_roi_with_no_intersection_returns_empty_list(fake_arcticdem_accessor):
    roi = Geometry(box(0, 0, 1, 1), crs="EPSG:4326")  # far from either fixture tile

    patches = ArcticDEMABC.adjacent_patches(fake_arcticdem_accessor, roi, None)

    assert patches == []


def test_adjacent_patches_geodataframe_roi_returns_all_intersecting_tiles(fake_arcticdem_accessor):
    roi = gpd.GeoDataFrame(geometry=[box(150, 65, 152, 66)], crs="EPSG:4326")

    patches = ArcticDEMABC.adjacent_patches(fake_arcticdem_accessor, roi, None)

    assert sorted(p.id for p in patches) == ["36_24_32m_v4.1", "37_24_32m_v4.1"]


def test_adjacent_patches_invalid_roi_type_raises(fake_arcticdem_accessor):
    with pytest.raises(ValueError, match="roi must be a GeoBox or a GeoDataFrame"):
        # Deliberately passing a type outside the declared union to exercise the
        # runtime catch-all branch; ty correctly flags this as a type error too.
        ArcticDEMABC.adjacent_patches(fake_arcticdem_accessor, "not-a-roi", None)  # ty:ignore[invalid-argument-type]


# --- download tests: exercise remote STAC data access and multi-process behavior --
# Marked as integration tests so they are skipped by default in CI.


@pytest.mark.integration
def test_arcticdem32m_download():
    try:
        geobox = GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
        accessor = smart_geocubes.ArcticDEM32m(storage, backend="threaded")
        adem = accessor.load(geobox, create=True)
        print(adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())
        assert adem.dem.mean().item() == approx(102.10579)
        assert adem.dem.min().item() == approx(46.429688)
        assert adem.dem.max().item() == approx(483.83594)
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
        if "adem" in locals():
            del adem
        shutil.rmtree("arcticdem_32m.zarr", ignore_errors=True)


@pytest.mark.integration
def test_arcticdem2m_download():
    try:
        geobox = GeoBox.from_bbox((150, 65, 150.1, 65.1), shape=(1000, 1000))
        storage = icechunk.local_filesystem_storage("arcticdem_2m.zarr")
        accessor = smart_geocubes.ArcticDEM2m(storage, backend="threaded")
        adem = accessor.load(geobox, create=True)
        print(adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())
        assert adem.dem.mean().item() == approx(203.03644)
        assert adem.dem.min().item() == approx(149.7421)
        assert adem.dem.max().item() == approx(285.5547)
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
        if "adem" in locals():
            del adem
        shutil.rmtree("arcticdem_2m.zarr", ignore_errors=True)


Stats = namedtuple("Stats", ["mean", "min", "max"])


@pytest.mark.integration
def test_arcticdem_download_threaded():
    try:
        storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
        accessor = smart_geocubes.ArcticDEM32m(storage, backend="threaded")
        accessor.create(overwrite=True)

        def _task(i, geobox: GeoBox) -> tuple[int, Stats]:
            adem = accessor.load(geobox)
            return i, Stats(adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())

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
            assert result.mean == approx(102.10579)
            assert result.min == approx(46.429688)
            assert result.max == approx(483.83594)
    finally:
        shutil.rmtree("arcticdem_32m.zarr", ignore_errors=True)


def _mp_task(i, geobox: GeoBox) -> tuple[int, Stats]:
    storage = icechunk.local_filesystem_storage("arcticdem_32m.zarr")
    accessor = smart_geocubes.ArcticDEM32m(storage, backend="simple")
    adem = accessor.load(geobox)
    return i, Stats(adem.dem.mean().item(), adem.dem.min().item(), adem.dem.max().item())


@pytest.mark.integration
def test_arcticdem_download_blocking_processes():
    try:
        # This test fails with the "spawn" method, however, it's not possible to set the start method from a test
        # So this test stays broken for now
        # mp.set_start_method("forkserver")
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
            assert result.mean == approx(102.10579)
            assert result.min == approx(46.429688)
            assert result.max == approx(483.83594)
    finally:
        shutil.rmtree("arcticdem_32m.zarr", ignore_errors=True)
