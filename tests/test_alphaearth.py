import logging
import os
import shutil

import ee
import icechunk
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox
from pytest import approx

import smart_geocubes

# Setup logging
logger = logging.getLogger("smart_geocubes")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# ~5km x 5km at 65N. Small enough to keep the AOI within a single tile of the datacube's
# 3600px chunk grid (chunk_size=3600 @ ~10m resolution ~ 36km per tile), so only a single
# patch is downloaded per requested year.
AOI = GeoBox.from_bbox((150, 65, 150.1063, 65.0449), shape=(500, 500))


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GEE_PROJECT") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
    reason="GEE credentials not configured; skip integration tests",
)
def test_alphaearth_download():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        storage = icechunk.local_filesystem_storage("alphaearth.zarr")
        accessor = smart_geocubes.AlphaEarthEmbeddings(storage, backend="threaded")
        ds = accessor.load(AOI, toi="2020-01-01", create=True)

        # A single timestamp selection collapses "time" from a dimension to a scalar coordinate.
        assert "time" not in ds.dims
        assert pd.Timestamp(ds.time.item()) == pd.Timestamp("2020-01-01")

        print(ds.A00.mean().item(), ds.A00.min().item(), ds.A00.max().item())
        assert ds.A00.mean().item() == approx(-0.061940282583236694)
        assert ds.A00.min().item() == approx(-0.1663360297679901)
        assert ds.A00.max().item() == approx(-0.004982699174433947)
        print(ds.A01.mean().item(), ds.A01.min().item(), ds.A01.max().item())
        assert ds.A01.mean().item() == approx(-0.17543981969356537)
        assert ds.A01.min().item() == approx(-0.29287198185920715)
        assert ds.A01.max().item() == approx(-0.11374086886644363)
        print(ds.A63.mean().item(), ds.A63.min().item(), ds.A63.max().item())
        assert ds.A63.mean().item() == approx(0.08527452498674393)
        assert ds.A63.min().item() == approx(-0.010396001860499382)
        assert ds.A63.max().item() == approx(0.2069357931613922)
    finally:
        if "ds" in locals():
            del ds
        shutil.rmtree("alphaearth.zarr", ignore_errors=True)


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GEE_PROJECT") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
    reason="GEE credentials not configured; skip integration tests",
)
def test_alphaearth_download_different_year():
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    try:
        storage = icechunk.local_filesystem_storage("alphaearth.zarr")
        accessor = smart_geocubes.AlphaEarthEmbeddings(storage, backend="threaded")

        ds_2020 = accessor.load(AOI, toi="2020-01-01", create=True)
        mean_2020 = ds_2020.A00.mean().item()
        del ds_2020

        ds_2022 = accessor.load(AOI, toi="2022-01-01")
        assert pd.Timestamp(ds_2022.time.item()) == pd.Timestamp("2022-01-01")
        mean_2022 = ds_2022.A00.mean().item()

        assert mean_2020 == approx(-0.061940282583236694)
        assert mean_2022 == approx(-0.08845065534114838)

        # The two years are stored as distinct patches, keyed by their own temporal index
        state = accessor.current_state()
        assert state is not None
        assert set(state["time"]) == {
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2022-01-01"),
        }
    finally:
        if "ds_2022" in locals():
            del ds_2022
        shutil.rmtree("alphaearth.zarr", ignore_errors=True)


def test_alphaearth_visualize_state_not_implemented(tmp_path):
    storage = icechunk.local_filesystem_storage(str(tmp_path / "alphaearth.zarr"))
    accessor = smart_geocubes.AlphaEarthEmbeddings(storage)
    accessor.create()

    with pytest.raises(NotImplementedError):
        accessor.visualize_state()
