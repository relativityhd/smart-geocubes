"""Unit tests for the pure index (de-)serialization logic of GEEMosaicAccessor.

`_stringify_index`/`_parse_index` only touch `self.is_temporal`, so they are
exercised here with a minimal duck-typed stand-in instead of a fully constructed
accessor (which would require an icechunk store). No network access, no `ee`/`xee`
import required since GEEMosaicAccessor only imports those lazily inside
`download_patch`.
"""

from types import SimpleNamespace
from typing import cast

import pytest

from smart_geocubes.accessors.gee import GEEMosaicAccessor

NON_TEMPORAL = cast(GEEMosaicAccessor, SimpleNamespace(is_temporal=False))
TEMPORAL = cast(GEEMosaicAccessor, SimpleNamespace(is_temporal=True))


def test_stringify_index_non_temporal():
    assert GEEMosaicAccessor._stringify_index(NON_TEMPORAL, (3, 7)) == "3-7"


def test_stringify_index_temporal():
    assert GEEMosaicAccessor._stringify_index(TEMPORAL, (3, 7), 2) == "2-3-7"


def test_parse_index_non_temporal():
    spatial_idx, time_idx = GEEMosaicAccessor._parse_index(NON_TEMPORAL, "3-7")
    assert spatial_idx == (3, 7)
    assert time_idx is None


def test_parse_index_temporal():
    spatial_idx, time_idx = GEEMosaicAccessor._parse_index(TEMPORAL, "2-3-7")
    assert spatial_idx == (3, 7)
    assert time_idx == 2


@pytest.mark.parametrize("spatial_idx", [(0, 0), (12, 345)])
def test_stringify_parse_roundtrip_non_temporal(spatial_idx):
    # Note: spatial indices are always non-negative (they come from GeoboxTiles grid
    # positions). Round-tripping a negative index would break, since "-" is used both
    # as the field separator and as a sign character, but that input never occurs.
    idx_str = GEEMosaicAccessor._stringify_index(NON_TEMPORAL, spatial_idx)
    assert GEEMosaicAccessor._parse_index(NON_TEMPORAL, idx_str) == (spatial_idx, None)


@pytest.mark.parametrize("spatial_idx,time_idx", [((0, 0), 0), ((12, 345), 99)])
def test_stringify_parse_roundtrip_temporal(spatial_idx, time_idx):
    idx_str = GEEMosaicAccessor._stringify_index(TEMPORAL, spatial_idx, time_idx)
    assert GEEMosaicAccessor._parse_index(TEMPORAL, idx_str) == (spatial_idx, time_idx)


def test_parse_index_non_temporal_string_on_temporal_accessor_raises():
    with pytest.raises(AssertionError, match="Non-temporal index"):
        GEEMosaicAccessor._parse_index(TEMPORAL, "3-7")


def test_parse_index_temporal_string_on_non_temporal_accessor_raises():
    with pytest.raises(AssertionError, match="Temporal index"):
        GEEMosaicAccessor._parse_index(NON_TEMPORAL, "2-3-7")


def test_parse_index_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid index string"):
        GEEMosaicAccessor._parse_index(NON_TEMPORAL, "3-7-8-9")
