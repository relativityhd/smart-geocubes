"""Unit tests for smart_geocubes.accessors.stac.

`correct_bounds` is pure and tested against small synthetic datasets. The rest of
this module boundary-mocks `pystac_client.Client.open` with canned `pystac.Item`
fixtures, so `adjacent_patches`/`current_state` are exercised without ever hitting
a real STAC API.
"""

from datetime import UTC, datetime
from typing import ClassVar

import odc.geo.xr
import pandas as pd
import pystac
import pystac_client
import pytest
import xarray as xr
from odc.geo.geobox import GeoBox
from shapely.geometry import box, mapping

from smart_geocubes.accessors.stac import STACAccessor, correct_bounds


def _make_tile(bbox: tuple[float, float, float, float], resolution: float = 1) -> xr.Dataset:
    gb = GeoBox.from_bbox(bbox, crs="EPSG:3857", resolution=resolution)
    return xr.Dataset({"red": odc.geo.xr.xr_zeros(gb, dtype="uint8")})


def test_correct_bounds_tile_fully_within_geobox_is_unchanged():
    tile = _make_tile((0, 0, 10, 10))
    zgeobox = GeoBox.from_bbox((-5, -5, 20, 20), crs="EPSG:3857", resolution=1)

    out = correct_bounds(tile, zgeobox)

    assert out.sizes == tile.sizes
    assert out.equals(tile)


def test_correct_bounds_crops_tile_extending_beyond_geobox():
    tile = _make_tile((0, 0, 10, 10))
    zgeobox = GeoBox.from_bbox((0, 0, 6, 6), crs="EPSG:3857", resolution=1)

    out = correct_bounds(tile, zgeobox)

    assert out.sizes == {"y": 6, "x": 6}


def test_correct_bounds_tile_outside_geobox_raises():
    tile = _make_tile((100, 100, 110, 110))
    zgeobox = GeoBox.from_bbox((0, 0, 6, 6), crs="EPSG:3857", resolution=1)

    with pytest.raises(ValueError, match="out of bounds"):
        correct_bounds(tile, zgeobox)


# --- boundary-mocked pystac_client -------------------------------------------------


def _make_item(
    item_id: str = "fake-item",
    dt: datetime | None = datetime(2020, 1, 1, tzinfo=UTC),
    start: datetime | None = None,
    end: datetime | None = None,
) -> pystac.Item:
    # pystac.Item itself requires either `datetime` or both `start_datetime` and
    # `end_datetime`; `dt` defaults to a fixed value so callers that don't care
    # about temporal fields (e.g. non-temporal-accessor tests) don't need to think
    # about this pystac-level constraint.
    geom = mapping(box(-1, -1, 1, 1))
    return pystac.Item(
        id=item_id,
        geometry=geom,
        bbox=[-1, -1, 1, 1],
        datetime=dt,
        properties={},
        start_datetime=start,
        end_datetime=end,
    )


class _FakeSearch:
    def __init__(self, items, item_collection=None):
        self._items = items
        self._item_collection = item_collection

    def items(self):
        return iter(self._items)

    def item_collection_as_dict(self):
        return self._item_collection


class _FakeCatalog:
    def __init__(self, items=None, item_collection=None):
        self.items = items or []
        self.item_collection = item_collection
        self.search_calls: list[dict] = []

    def search(self, **kwargs):
        self.search_calls.append(kwargs)
        return _FakeSearch(self.items, self.item_collection)


@pytest.fixture
def fake_catalog(monkeypatch):
    catalog = _FakeCatalog()
    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda url: catalog))
    return catalog


class NonTemporalSTAC(STACAccessor):
    extent: GeoBox = GeoBox.from_bbox((-1, -1, 1, 1), crs="EPSG:4326", resolution=0.1)
    temporal_extent = None
    chunk_size = 10
    channels: ClassVar[list] = ["value"]
    _channels_meta: ClassVar[dict] = {}
    _channels_encoding: ClassVar[dict] = {}
    stac_api_url = "https://example.com/stac"
    collection = "fake-collection"

    def download_patch(self, idx):
        raise NotImplementedError

    def current_state(self):
        return None

    def visualize_state(self, ax=None):
        pass


class TemporalSTAC(NonTemporalSTAC):
    temporal_extent = pd.date_range("2020-01-01", periods=3, freq="YS")


def _make(cls):
    return object.__new__(cls)


def test_adjacent_patches_non_temporal_builds_patch_index_from_items(fake_catalog):
    fake_catalog.items = [_make_item("item-a"), _make_item("item-b")]
    acc = _make(NonTemporalSTAC)

    patches = acc.adjacent_patches(acc.extent, None)

    assert sorted(p.id for p in patches) == ["item-a", "item-b"]
    assert all(p.time is None for p in patches)
    assert all(p.geometry.crs == "EPSG:4326" for p in patches)


def test_adjacent_patches_temporal_uses_item_datetime(fake_catalog):
    dt = datetime(2020, 6, 1, tzinfo=UTC)
    fake_catalog.items = [_make_item("item-a", dt=dt)]
    acc = _make(TemporalSTAC)

    (patch,) = acc.adjacent_patches(acc.extent, None)

    assert patch.time == dt


def test_adjacent_patches_temporal_item_without_datetime_uses_start_end_range(fake_catalog):
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 12, 31, tzinfo=UTC)
    fake_catalog.items = [_make_item("item-a", dt=None, start=start, end=end)]
    acc = _make(TemporalSTAC)

    (patch,) = acc.adjacent_patches(acc.extent, None)

    assert patch.time == (start, end)


def test_adjacent_patches_temporal_item_missing_temporal_extent_raises(fake_catalog):
    # pystac.Item's own constructor requires either `datetime` or both start/end,
    # so simulate the malformed state the source code defends against (an item with
    # none of the three) by clearing the properties after construction.
    start = datetime(2020, 1, 1, tzinfo=UTC)
    end = datetime(2020, 12, 31, tzinfo=UTC)
    item = _make_item("item-a", dt=None, start=start, end=end)
    del item.properties["start_datetime"]
    del item.properties["end_datetime"]
    fake_catalog.items = [item]
    acc = _make(TemporalSTAC)

    with pytest.raises(AssertionError, match="has no temporal extent"):
        acc.adjacent_patches(acc.extent, None)


def test_adjacent_patches_geodataframe_roi_passed_through_unchanged(fake_catalog):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(geometry=[box(-1, -1, 1, 1)], crs="EPSG:4326")
    acc = _make(NonTemporalSTAC)

    acc.adjacent_patches(gdf, None)

    assert fake_catalog.search_calls[0]["intersects"] is gdf


def test_adjacent_patches_geobox_roi_converted_to_geom(fake_catalog):
    acc = _make(NonTemporalSTAC)

    acc.adjacent_patches(acc.extent, None)

    intersects = fake_catalog.search_calls[0]["intersects"]
    assert not hasattr(intersects, "crs")  # reprojected down to a plain shapely geometry


def test_adjacent_patches_invalid_roi_type_raises(fake_catalog):
    acc = _make(NonTemporalSTAC)
    with pytest.raises(ValueError, match="Invalid ROI type"):
        acc.adjacent_patches("not-a-roi", None)


def test_adjacent_patches_toi_forwarded_to_search_as_datetime_range(fake_catalog):
    acc = _make(TemporalSTAC)

    acc.adjacent_patches(acc.extent, slice("2020-01-01", "2020-06-01"))

    assert fake_catalog.search_calls[0]["datetime"] == ("2020-01-01", "2020-06-01")


def test_adjacent_patches_collection_and_url_are_used(fake_catalog, monkeypatch):
    opened_urls = []
    monkeypatch.setattr(pystac_client.Client, "open", staticmethod(lambda url: opened_urls.append(url) or fake_catalog))
    acc = _make(NonTemporalSTAC)

    acc.adjacent_patches(acc.extent, None)

    assert opened_urls == ["https://example.com/stac"]
    assert fake_catalog.search_calls[0]["collections"] == ["fake-collection"]


# --- current_state -------------------------------------------------------------------


def _fake_self(**overrides):
    from types import SimpleNamespace

    defaults = {
        "created": True,
        "loaded_patches": lambda: ["item-a", "item-b"],
        "stac_api_url": "https://example.com/stac",
        "collection": "fake-collection",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_current_state_returns_none_when_not_created():
    fake_self = _fake_self(created=False)
    assert STACAccessor.current_state(fake_self) is None


def test_current_state_returns_none_when_no_loaded_patches():
    fake_self = _fake_self(loaded_patches=lambda: [])
    assert STACAccessor.current_state(fake_self) is None


def test_current_state_builds_geodataframe_from_search_results(fake_catalog):
    fake_catalog.item_collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": mapping(box(-1, -1, 1, 1)), "properties": {"id": "item-a"}},
            {"type": "Feature", "geometry": mapping(box(1, 1, 2, 2)), "properties": {"id": "item-b"}},
        ],
    }
    fake_self = _fake_self()

    gdf = STACAccessor.current_state(fake_self)

    assert gdf is not None
    assert sorted(gdf["id"]) == ["item-a", "item-b"]
    assert fake_catalog.search_calls[0]["ids"] == ["item-a", "item-b"]
