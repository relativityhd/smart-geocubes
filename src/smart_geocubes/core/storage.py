"""Local zarr-storage related functions."""

import logging
from typing import TypedDict

import numpy as np
import pandas as pd
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)


class CoordEncoding(TypedDict):
    """TypedDict for the encoding of regularly spaced coordinates."""

    compressors: list[BytesBytesCodec]
    filters: list[BytesBytesCodec]


def optimize_coord_encoding(values: np.ndarray, dx: int) -> CoordEncoding:
    """Optimize zarr encoding of regularly spaced coordinates.

    Taken from https://github.com/earth-mover/serverless-datacube-demo/blob/a15423b9734898f52468bebc441e29ccf3789410/src/lib.py#L280

    Args:
        values (np.ndarray): The coordinates to encode
        dx (int): The spacing between the coordinates

    Returns:
        CoordEncoding: A dictionary containing the zarr compressors and filters to use for encoding the coordinates.

    """
    compressor = BloscCodec()
    return {"compressors": [compressor]}


def optimize_temporal_encoding(temporal_extent: pd.DatetimeIndex) -> dict:
    """Optimize the encoding of temporal data.

    Args:
        temporal_extent (pd.DatetimeIndex): The temporal extent to encode.

    Returns:
        dict: A dictionary containing the zarr compressors and filters to use for encoding the temporal data.

    """
    compressor = BloscCodec()
    return {"compressors": [compressor]}
