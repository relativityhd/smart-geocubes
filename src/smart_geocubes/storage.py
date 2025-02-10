"""Local zarr-storage related functions."""

import logging
import time
import warnings
from typing import TypedDict

import numpy as np
import odc.geo
import odc.geo.xr
import xarray as xr
import zarr
import zarr.codecs
import zarr.storage
from numcodecs.abc import Codec
from numcodecs.zarr3 import Blosc, Delta, FixedScaleOffset
from odc.geo.geobox import GeoBox

logger = logging.getLogger(__name__)

zarr3blosc_warning = "Numcodecs codecs are not in the Zarr version 3 specification and may not be supported by other zarr implementations."  # noqa: E501
zarr3consolidated_warning = "Consolidated metadata is currently not part in the Zarr format 3 specification. It may not be supported by other zarr implementations and may change in the future."  # noqa: E501
warnings.filterwarnings("ignore", category=UserWarning, message=zarr3blosc_warning)
warnings.filterwarnings("ignore", category=UserWarning, message=zarr3consolidated_warning)


class CoordEncoding(TypedDict):
    """TypedDict for the encoding of regularly spaced coordinates."""

    compressors: list[Codec]
    filters: list[Codec]


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
    compressor = Blosc(cname="zstd")

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


def create_empty_datacube(
    title: str,
    storage: zarr.storage.StoreLike,
    geobox: GeoBox,
    chunk_size: int,
    data_vars: list,
    meta: dict,
    var_encoding: dict,
):
    """Create an empty datacube from a GeoBox covering the complete extent of the geobox's CRS.

    Args:
        title (str): The title of the dataset
        storage (zarr.storage.StoreLike): The zarr storage object where the datacube will be saved.
        geobox (GeoBox): The geobox for which the dataset should be created.
        chunk_size (int): The size of a single chunk in pixels.
        data_vars (list): List of data variables which should be created
        meta (dict): Metadata for the data variables. Key must be in data_vars.
        var_encoding (dict): Encoding for the data variables. Key must be in data_vars.

    Raises:
        ValueError: If the geobox's CRS does not have x and y coordinates. EPSG:4326 is the only supported lat-lon CRS.

    """
    tick_fstart = time.perf_counter()
    logger.debug(
        f"Creating an empty zarr datacube '{title}' with the variables"
        f"{data_vars} at a {geobox.resolution=} (epsg:{geobox.crs.epsg}) and {chunk_size=} to {storage=}"
    )

    ds = xr.Dataset(
        {
            name: odc.geo.xr.xr_zeros(geobox, chunks=-1, dtype=var_encoding[name].get("dtype", "float32"))
            for name in data_vars
        },
        attrs={"title": title, "loaded_tiles": []},
    )

    # Add metadata
    for name, meta in meta.items():
        ds[name].attrs.update(meta)

    # Turn lat-lon into x-y
    if geobox.crs.epsg == 4326:
        nmap = {"longitude": "x", "latitude": "y"}
        ds = ds.rename(nmap)

    if "x" not in ds.coords or "y" not in ds.coords:
        raise ValueError(f"The dataset must have x and y coordinates, but got {ds.coords.keys()}")

    # Get the encoding for the coordinates, variables and spatial reference
    coords_encoding = {
        "x": {"chunks": ds.x.shape, **optimize_coord_encoding(ds.x.values, geobox.resolution.x)},
        "y": {"chunks": ds.y.shape, **optimize_coord_encoding(ds.y.values, geobox.resolution.y)},
    }
    var_encoding = {
        name: {"chunks": (chunk_size, chunk_size), "compressors": [Blosc(cname="zstd")], **var_encoding[name]}
        for name in data_vars
    }
    encoding = {
        "spatial_ref": {"chunks": None, "dtype": "int32"},
        **coords_encoding,
        **var_encoding,
    }
    logger.debug(f"Datacube {encoding=}")

    ds.to_zarr(
        storage,
        encoding=encoding,
        compute=False,
    )

    tick_fend = time.perf_counter()
    logger.debug(f"Empty datacube created in {tick_fend - tick_fstart:.2f} seconds")
