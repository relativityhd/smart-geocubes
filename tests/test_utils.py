"""Unit tests for smart_geocubes.core.utils.

These are pure formatting/version-check helpers, no I/O involved.
"""

import collections

import pytest
from affine import Affine
from odc.geo.crs import CRS
from odc.geo.geobox import GeoBox

from smart_geocubes.core import utils

_VersionInfo = collections.namedtuple("_VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"])


# --- _check_python_version -------------------------------------------------------


@pytest.fixture
def fake_version(monkeypatch):
    def _set(major: int, minor: int):
        monkeypatch.setattr(utils.sys, "version_info", _VersionInfo(major, minor, 0, "final", 0))

    return _set


def test_check_python_version_minor_below_required(fake_version):
    fake_version(3, 12)
    assert utils._check_python_version(3, 13) is False


def test_check_python_version_minor_equal_required(fake_version):
    fake_version(3, 12)
    assert utils._check_python_version(3, 12) is True


def test_check_python_version_minor_above_required(fake_version):
    fake_version(3, 12)
    assert utils._check_python_version(3, 11) is True


def test_check_python_version_major_below_required(fake_version):
    fake_version(2, 7)
    assert utils._check_python_version(3, 0) is False


def test_check_python_version_major_above_required(fake_version):
    fake_version(4, 0)
    assert utils._check_python_version(3, 13) is True


# --- _geobox_repr ------------------------------------------------------------------


def test_geobox_repr_degrees_crs():
    gb = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    assert utils._geobox_repr(gb) == "GeoBox(Shape2d(x=4, y=4), 0.50000° x 0.50000° @ [-1.0, 1.0] in EPSG:4326)"


def test_geobox_repr_meter_crs():
    gb = GeoBox.from_bbox((-1000, -1000, 1000, 1000), crs="EPSG:3857", resolution=100)
    assert utils._geobox_repr(gb) == "GeoBox(Shape2d(x=20, y=20), 100.0m x 100.0m @ [-1000.0, 1000.0] in EPSG:3857)"


def test_geobox_repr_no_crs():
    gb = GeoBox((10, 10), Affine.translation(0, 0) * Affine.scale(1, -1), crs=None)
    assert utils._geobox_repr(gb) == "GeoBox(Shape2d(x=10, y=10), 1.0 x 1.0 @ [0.0, 0.0] in Unknown CRS)"


def test_geobox_repr_non_epsg_crs():
    crs = CRS("+proj=stere +lat_0=90 +lon_0=-45")
    gb = GeoBox.from_bbox((-1000, -1000, 1000, 1000), crs=crs, resolution=10)
    assert "Non-EPSG CRS" in utils._geobox_repr(gb)


# --- _geometry_repr -----------------------------------------------------------------


def test_geometry_repr_degrees_crs():
    gb = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.5)
    assert utils._geometry_repr(gb.extent) == "Geometry((-0.00000°, -0.00000°) in EPSG:4326)"


def test_geometry_repr_meter_crs():
    gb = GeoBox.from_bbox((-1000, -1000, 1000, 1000), crs="EPSG:3857", resolution=100)
    assert utils._geometry_repr(gb.extent) == "Geometry((-0.0m, -0.0m) in EPSG:3857)"


def test_geometry_repr_no_crs():
    import shapely.geometry as sg
    from odc.geo.geom import Geometry

    geom = Geometry(sg.box(0, 0, 1, 1), crs=None)
    assert utils._geometry_repr(geom) == "Geometry((0.5, 0.5) in Unknown CRS)"


def test_geometry_repr_non_epsg_crs():
    crs = CRS("+proj=stere +lat_0=90 +lon_0=-45")
    gb = GeoBox.from_bbox((-1000, -1000, 1000, 1000), crs=crs, resolution=10)
    assert "Non-EPSG CRS" in utils._geometry_repr(gb.extent)
