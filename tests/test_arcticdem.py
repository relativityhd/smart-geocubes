import os

from numpy.testing import assert_almost_equal
from odc.geo.geobox import GeoBox

from smart_geocubes.datasets.arcticdem import ArcticDEM32m


def test_arcticdem_download():
    geobox = GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000))
    accessor = ArcticDEM32m("arcticdem_32m.zarr")
    adem = accessor.load(geobox, create=True)
    try:
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
