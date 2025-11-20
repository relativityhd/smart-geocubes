"""Time of Interest (TOI) utilities."""

from datetime import datetime

import pandas as pd

type TOI = pd.DatetimeIndex | list[pd.Timestamp | str | datetime] | slice | pd.Timestamp | str | datetime | None


def normalize_toi(extent: pd.DatetimeIndex, toi: TOI, method="nearest") -> pd.DatetimeIndex:
    """Normalize the time of interest (toi) to the given temporal extent.

    Args:
        extent (pd.DatetimeIndex): The temporal extent to normalize against.
        toi (TOI): The time of interest to normalize.
        method (str, optional): The method to use for normalization. Defaults to "nearest".
            Other options are "pad", "backfill", "ffill", "bfill".

    Returns:
        pd.DatetimeIndex: The normalized time of interest.

    Raises:
        ValueError: If the time of interest is not found in the temporal extent.

    """
    # Normalize the extent
    extent = extent.normalize()

    if toi is None:
        return extent

    if isinstance(toi, str | datetime | pd.Timestamp):
        idxr = extent.get_indexer([toi], method=method)
    elif isinstance(toi, slice):
        idxr = extent.slice_indexer(toi.start, toi.stop)
    elif isinstance(toi, list):
        idxr = extent.get_indexer(toi, method=method)

    toi_norm = extent[idxr]
    if len(toi_norm) == 0:
        raise ValueError(f"Time {toi} not found in temporal extent.")
    return toi_norm


def extract_toi_range(toi: TOI) -> str | datetime | tuple[str, str] | tuple[datetime | datetime] | None:
    """Extract the datetime range or a specific datetime from the time of interest (toi).

    Args:
        toi (TOI): The time of interest.

    Returns:
        str | datetime | tuple[str, str] | tuple[datetime, datetime] | None: The extracted datetime or datetime range.

    Raises:
        ValueError: If the time of interest is of an invalid type.

    """
    if toi is None or isinstance(toi, str | datetime):
        return toi
    elif isinstance(toi, pd.Timestamp):
        return toi.to_pydatetime()
    elif isinstance(toi, slice):
        return toi.start, toi.stop
    else:
        raise ValueError(f"Cannot extract range from toi of type {type(toi)}.")


def _repr_toi(toi: TOI) -> str:
    """Get a string representation of the time of interest.

    Args:
        toi (TOI): The time of interest.

    Returns:
        str: The string representation of the time of interest.

    Raises:
        ValueError: If the time of interest is of an invalid type.

    """
    if isinstance(toi, pd.DatetimeIndex):
        return f"pd.DatetimeIndex[{len(toi)}]"
    if isinstance(toi, list):
        return f"list[{len(toi)}]"
    if isinstance(toi, slice):
        return f"[{toi.start}, {toi.stop}]"
    if isinstance(toi, pd.Timestamp | str | datetime):
        return str(toi)
    if toi is None:
        return "None"
    raise ValueError(f"Invalid type for toi: {type(toi)}")
