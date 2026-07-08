"""Unit tests for the time-of-interest (TOI) utilities.

These are pure functions with no I/O, so they run fully offline.
"""

from datetime import datetime

import pandas as pd
import pytest

from smart_geocubes.core.toi import _repr_toi, extract_toi_range, normalize_toi

EXTENT = pd.date_range("2020-01-01", periods=5, freq="D")  # 2020-01-01 .. 2020-01-05


# --- normalize_toi -------------------------------------------------------------


def test_normalize_toi_none_returns_full_extent():
    result = normalize_toi(EXTENT, None)
    pd.testing.assert_index_equal(result, EXTENT)


def test_normalize_toi_strips_time_of_day_from_extent():
    extent_with_time = pd.DatetimeIndex(["2020-01-01 08:00", "2020-01-02 20:00"])
    result = normalize_toi(extent_with_time, None)
    assert list(result) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]


def test_normalize_toi_single_string_exact_match():
    result = normalize_toi(EXTENT, "2020-01-03")
    assert list(result) == [pd.Timestamp("2020-01-03")]


def test_normalize_toi_single_datetime_nearest_match():
    # 2020-01-03 18:00 is closer to 2020-01-04 00:00 than to 2020-01-03 00:00
    result = normalize_toi(EXTENT, datetime(2020, 1, 3, 18, 0))
    assert list(result) == [pd.Timestamp("2020-01-04")]


def test_normalize_toi_single_timestamp_snaps_to_nearest_available_day():
    # ty infers pd.Timestamp(...) as `Timestamp | NaTType`, and NaTType isn't part of
    # the TOI union, even though this call can never actually produce a NaT.
    result = normalize_toi(EXTENT, pd.Timestamp("2020-01-10"))  # ty:ignore[invalid-argument-type]
    assert list(result) == [pd.Timestamp("2020-01-05")]


def test_normalize_toi_list_of_dates():
    result = normalize_toi(EXTENT, ["2020-01-01", "2020-01-05"])
    assert list(result) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-05")]


def test_normalize_toi_slice_within_extent():
    result = normalize_toi(EXTENT, slice("2020-01-02", "2020-01-04"))
    assert list(result) == [
        pd.Timestamp("2020-01-02"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-04"),
    ]


def test_normalize_toi_slice_outside_extent_raises():
    with pytest.raises(ValueError, match="not found in temporal extent"):
        normalize_toi(EXTENT, slice("2021-01-01", "2021-01-02"))


def test_normalize_toi_backfill_method():
    # "backfill": snap to the next available date at or after the requested one
    result = normalize_toi(EXTENT, "2019-12-31", method="backfill")
    assert list(result) == [pd.Timestamp("2020-01-01")]


def test_normalize_toi_pad_method():
    # "pad": snap to the previous available date at or before the requested one
    result = normalize_toi(EXTENT, "2020-01-10", method="pad")
    assert list(result) == [pd.Timestamp("2020-01-05")]


# --- extract_toi_range -----------------------------------------------------------


def test_extract_toi_range_none():
    assert extract_toi_range(None) is None


def test_extract_toi_range_string():
    assert extract_toi_range("2020-01-01") == "2020-01-01"


def test_extract_toi_range_datetime():
    dt = datetime(2020, 1, 1)
    assert extract_toi_range(dt) == dt


def test_extract_toi_range_timestamp_returned_unchanged():
    # pd.Timestamp is a subclass of datetime.datetime, so it is caught by the
    # `isinstance(toi, str | datetime)` branch before the dedicated `pd.Timestamp`
    # branch is reached. The Timestamp is returned as-is, never converted via
    # `.to_pydatetime()`. Pinning this down so a future refactor doesn't change it
    # by accident without a deliberate decision.
    ts = pd.Timestamp("2020-01-01")
    result = extract_toi_range(ts)  # ty:ignore[invalid-argument-type]
    assert result is ts


def test_extract_toi_range_slice():
    s = slice("2020-01-01", "2020-01-05")
    assert extract_toi_range(s) == ("2020-01-01", "2020-01-05")


def test_extract_toi_range_invalid_type_raises():
    with pytest.raises(ValueError, match="Cannot extract range"):
        extract_toi_range(["2020-01-01", "2020-01-02"])


# --- _repr_toi ---------------------------------------------------------------------


def test_repr_toi_none():
    assert _repr_toi(None) == "None"


def test_repr_toi_datetime_index():
    assert _repr_toi(EXTENT) == "pd.DatetimeIndex[5]"


def test_repr_toi_list():
    assert _repr_toi(["2020-01-01", "2020-01-02"]) == "list[2]"


def test_repr_toi_slice():
    assert _repr_toi(slice("2020-01-01", "2020-01-05")) == "[2020-01-01, 2020-01-05]"


def test_repr_toi_timestamp():
    ts = pd.Timestamp("2020-01-01")
    assert _repr_toi(ts) == str(ts)  # ty:ignore[invalid-argument-type]


def test_repr_toi_invalid_type_raises():
    # Deliberately passing a type outside the TOI union to exercise the runtime
    # catch-all branch; ty correctly flags this as a type error too.
    with pytest.raises(ValueError, match="Invalid type for toi"):
        _repr_toi(123)  # ty:ignore[invalid-argument-type]
