import os

import ee
import pytest
from odc.geo.geobox import GeoBox

import smart_geocubes

# All tests in this module hit remote services (Google Earth Engine or the ArcticDEM STAC
# API) and are marked as integration tests. The GEE-backed tests additionally skip if
# credentials are not configured; the ArcticDEM STAC API is public and needs no credentials.


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GEE_PROJECT") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
    reason="GEE credentials not configured; skip integration tests",
)
def test_utm01_tcvis(tmp_path):
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    accessor = smart_geocubes.TCTrend(tmp_path / "tcvis.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((-180.0, 67.5, -179.5, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


@pytest.mark.integration
@pytest.mark.skipif(
    not (os.getenv("GEE_PROJECT") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
    reason="GEE credentials not configured; skip integration tests",
)
def test_utm60_tcvis(tmp_path):
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    accessor = smart_geocubes.TCTrend(tmp_path / "tcvis.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((179.5, 67.5, 180.0, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


@pytest.mark.integration
def test_utm01_arcticdem(tmp_path):
    accessor = smart_geocubes.ArcticDEM10m(tmp_path / "arcticdem.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((-180.0, 67.5, -179.5, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


@pytest.mark.integration
def test_utm60_arcticdem(tmp_path):
    accessor = smart_geocubes.ArcticDEM10m(tmp_path / "arcticdem.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((179.5, 67.5, 180, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)
