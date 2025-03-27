"""Local zarr-storage related functions."""

import logging
import warnings
from typing import TypedDict

import numpy as np
from numcodecs.zarr3 import Delta, FixedScaleOffset
from zarr.abc.codec import BytesBytesCodec
from zarr.codecs import BloscCodec

logger = logging.getLogger(__name__)

zarr3blosc_warning = "Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations."  # noqa: E501
zarr3consolidated_warning = "Consolidated metadata is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future."  # noqa: E501
warnings.filterwarnings("ignore", category=UserWarning, message=zarr3blosc_warning)
warnings.filterwarnings("ignore", category=UserWarning, message=zarr3consolidated_warning)


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
    dx_all = np.diff(values)
    # dx = dx_all[0]
    np.testing.assert_allclose(dx_all, dx), "must be regularly spaced"

    offset_codec = FixedScaleOffset(offset=values[0], scale=1 / dx, dtype=values.dtype, astype="<i8")
    delta_codec = Delta(dtype="<i8", astype="<i2")
    compressor = BloscCodec()

    # Since the update to zarr 3., we can't test the encoding and decoding in a simple maner anymore
    # because they use async operations etc.
    # enc0 = offset_codec.encode(values)
    # everything should be offset by 1 at this point
    # np.testing.assert_equal(np.unique(np.diff(enc0)), [1])
    # enc1 = delta_codec.encode(enc0)
    # now we should be able to compress the shit out of this
    # enc2 = compressor.encode(enc1)
    # decoded = offset_codec.decode(delta_codec.decode(compressor.decode(enc2)))
    # will produce numerical precision differences
    # np.testing.assert_equal(values, decoded)
    # np.testing.assert_allclose(values, decoded)

    return {"compressors": [compressor], "filters": [offset_codec, delta_codec]}
