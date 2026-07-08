"""Unit tests for GEEMosaicAccessor.adjacent_patches.

Unlike `download_patch`, `adjacent_patches` never touches Google Earth Engine or the
network at all — it is pure geobox tiling (via odc.geo's GeoboxTiles) plus a
GeoDataFrame spatial join. So these tests build small accessor subclasses and
construct instances with `object.__new__`, bypassing `RemoteAccessor.__init__`
entirely (which would otherwise require an icechunk store for no benefit here,
since `adjacent_patches` never touches `self.repo`/`self.storage`/`self.backend`).
"""

from typing import ClassVar

import geopandas as gpd
import pandas as pd
import pytest
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from pytest import approx
from shapely.geometry import box

from smart_geocubes.accessors.gee import GEEMosaicAccessor


class NonTemporalAcc(GEEMosaicAccessor):
    """20x20px extent, chunk_size=10 -> a 2x2 grid of patches: "0-0", "0-1", "1-0", "1-1"."""

    extent: GeoBox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.1)
    temporal_extent = None
    chunk_size = 10
    channels: ClassVar[list] = ["value"]
    _channels_meta: ClassVar[dict] = {}
    _channels_encoding: ClassVar[dict] = {}

    def download_patch(self, idx):
        raise NotImplementedError

    def current_state(self):
        return None

    def visualize_state(self, ax=None):
        pass


class TemporalAcc(NonTemporalAcc):
    """Same spatial grid as NonTemporalAcc, with 3 yearly time steps."""

    temporal_extent = pd.date_range("2020-01-01", periods=3, freq="YS")


class GlobalAcc(GEEMosaicAccessor):
    """A 360x20px global extent, chunk_size=10 -> 36 columns x 2 rows, for antimeridian tests."""

    extent: GeoBox = GeoBox.from_bbox((-180, -10, 180, 10), crs="EPSG:4326", resolution=1)
    temporal_extent = None
    chunk_size = 10
    channels: ClassVar[list] = ["value"]
    _channels_meta: ClassVar[dict] = {}
    _channels_encoding: ClassVar[dict] = {}

    def download_patch(self, idx):
        raise NotImplementedError

    def current_state(self):
        return None

    def visualize_state(self, ax=None):
        pass


def _make(cls):
    return object.__new__(cls)


# --- ROI type dispatch ------------------------------------------------------------


def test_geobox_roi_returns_all_tiles():
    acc = _make(NonTemporalAcc)
    patches = acc.adjacent_patches(acc.extent, None)
    assert sorted(p.id for p in patches) == ["0-0", "0-1", "1-0", "1-1"]


def test_geometry_roi_returns_only_intersecting_tile():
    acc = _make(NonTemporalAcc)
    geom = Geometry(box(-0.9, 0.5, -0.8, 0.6), crs="EPSG:4326")  # fully inside tile "0-0"
    patches = acc.adjacent_patches(geom, None)
    assert [p.id for p in patches] == ["0-0"]


def test_geodataframe_roi_dedups_overlapping_geometries_in_same_tile():
    acc = _make(NonTemporalAcc)
    gdf = gpd.GeoDataFrame(
        geometry=[box(-0.9, 0.5, -0.8, 0.6), box(-0.95, 0.55, -0.85, 0.65)],
        crs="EPSG:4326",
    )
    patches = acc.adjacent_patches(gdf, None)
    assert [p.id for p in patches] == ["0-0"]


def test_geodataframe_roi_returns_distinct_tiles_for_separate_geometries():
    acc = _make(NonTemporalAcc)
    gdf = gpd.GeoDataFrame(
        geometry=[box(-0.9, 0.5, -0.8, 0.6), box(0.5, -0.9, 0.6, -0.8)],  # tiles "0-0" and "1-1"
        crs="EPSG:4326",
    )
    patches = acc.adjacent_patches(gdf, None)
    assert sorted(p.id for p in patches) == ["0-0", "1-1"]


def test_invalid_roi_type_raises():
    acc = _make(NonTemporalAcc)
    with pytest.raises(ValueError, match="Invalid ROI type"):
        acc.adjacent_patches("not-a-roi", None)


# --- PatchIndex contents ------------------------------------------------------------


def test_patch_index_geobox_and_geometry_match_the_tile():
    acc = _make(NonTemporalAcc)
    geom = Geometry(box(-0.9, 0.5, -0.8, 0.6), crs="EPSG:4326")
    (patch,) = acc.adjacent_patches(geom, None)

    assert patch.geometry.crs == "EPSG:4326"
    assert patch.time is None
    assert patch.item.time is None
    assert patch.item.geobox.shape.x == 10
    assert patch.item.geobox.shape.y == 10
    # tile "0-0" is the top-left quadrant: x in [-1, 0), y in (0, 1]
    assert patch.item.geobox.affine.c == approx(-1.0)  # x origin
    assert patch.item.geobox.affine.f == approx(1.0)  # y origin


# --- antimeridian ------------------------------------------------------------------


def test_west_antimeridian_edge_resolves_to_leftmost_column():
    acc = _make(GlobalAcc)
    west_edge = GeoBox.from_bbox((-180, -5, -179, 5), crs="EPSG:4326", resolution=1).extent
    patches = acc.adjacent_patches(west_edge, None)
    assert {p.id for p in patches} == {"0-0", "1-0"}
    assert all(p.item.geobox.affine.c == approx(-180.0) for p in patches)


def test_east_antimeridian_edge_resolves_to_rightmost_column():
    acc = _make(GlobalAcc)
    east_edge = GeoBox.from_bbox((179, -5, 180, 5), crs="EPSG:4326", resolution=1).extent
    patches = acc.adjacent_patches(east_edge, None)
    assert {p.id for p in patches} == {"0-35", "1-35"}
    assert all(p.item.geobox.affine.c == approx(170.0) for p in patches)


# --- temporal indexing --------------------------------------------------------------


def test_temporal_toi_none_returns_every_timestep_x_every_tile():
    acc = _make(TemporalAcc)
    patches = acc.adjacent_patches(acc.extent, None)
    assert len(patches) == 4 * 3  # 4 spatial tiles x 3 years
    assert {p.time for p in patches} == set(acc.temporal_extent)


def test_temporal_single_toi_returns_only_that_timestep():
    acc = _make(TemporalAcc)
    patches = acc.adjacent_patches(acc.extent, "2020-01-01")
    assert len(patches) == 4
    assert {p.time for p in patches} == {pd.Timestamp("2020-01-01")}
    assert {p.item.time for p in patches} == {pd.Timestamp("2020-01-01")}
    assert sorted(p.id.split("-", 1)[1] for p in patches) == ["0-0", "0-1", "1-0", "1-1"]


def test_toi_on_non_temporal_accessor_raises():
    acc = _make(NonTemporalAcc)
    with pytest.raises(ValueError, match="not temporal"):
        acc.adjacent_patches(acc.extent, "2020-01-01")
