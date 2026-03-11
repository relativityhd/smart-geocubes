import os

import ee
from odc.geo.geobox import GeoBox

import smart_geocubes


def test_utm01_tcvis(tmp_path):
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    accessor = smart_geocubes.TCTrend(tmp_path / "tcvis.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((-180.0, 67.5, -179.5, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


def test_utm60_tcvis(tmp_path):
    ee.Initialize(project=os.getenv("GEE_PROJECT"))
    accessor = smart_geocubes.TCTrend(tmp_path / "tcvis.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((179.5, 67.5, 180.0, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


def test_utm01_arcticdem(tmp_path):
    accessor = smart_geocubes.ArcticDEM10m(tmp_path / "arcticdem.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((-180.0, 67.5, -179.5, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)


def test_utm60_arcticdem(tmp_path):
    accessor = smart_geocubes.ArcticDEM10m(tmp_path / "arcticdem.icechunk")
    accessor.create(overwrite=False)

    aoi = GeoBox.from_bbox((179.5, 67.5, 180, 67.8), crs=4326, resolution=0.1)

    accessor.procedural_download(aoi, None)
