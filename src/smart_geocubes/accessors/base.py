"""Base class for remote accessors."""

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import xarray as xr
import zarr
import zarr.storage
from odc.geo.geobox import GeoBox
from zarr.core.sync import sync
from zarr.storage._common import make_store_path

from smart_geocubes.storage import create_empty_datacube

if TYPE_CHECKING:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError:
        pass

logger = logging.getLogger(__name__)

# Lock for downloading the data
# This will block other processes from downloading or processing the data, until the current download is finished.
# This may result in halting a tile-process for a while, even if it's data was already downloaded.
download_lock = mp.Lock()


class RemoteAccessor(ABC):
    """Base class for remote accessors."""

    extent: GeoBox
    chunk_size: int
    data_vars: list
    data_vars_meta: dict
    data_vars_encoding: dict

    @classmethod
    @abstractmethod
    def procedural_download(cls, storage: zarr.storage.StoreLike, geobox: GeoBox, data_vars: list[str] | None = None):
        """Download the data for the given geobox.

        Must be implemented by the Accessor.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage to download the data to.
            geobox (GeoBox): The reference geobox to download the data for.
            data_vars (list, optional): The data variables of the datacube.
                If set, overwrites the data variables set by the Dataset.
                Only used at first datacube creation. Defaults to None.

        """

    @classmethod
    @abstractmethod
    def extend(cls, storage: zarr.storage.StoreLike) -> "gpd.GeoDataFrame":
        """Get the extend, hence the already downloaded tiles, of the datacube.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage where the datacube is located.

        Returns:
            gpd.GeoDataFrame: The extend of the datacube

        """

    @classmethod
    @abstractmethod
    def visualize_extend(cls, storage: zarr.storage.StoreLike) -> "plt.Figure":
        """Visulize the extend, hence the already downloaded and filled data, of the datacube.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage where the datacube is located.

        Returns:
            plt.Figure: The figure with the visualization

        """

    @classmethod
    def load_like(
        cls,
        ref: xr.Dataset | xr.DataArray,
        cube_store: zarr.storage.StoreLike,
        buffer: int = 0,
        persist: bool = True,
        *,
        dataset_title: str | None = None,
        cube_extent: GeoBox | None = None,
        chunk_size: int | None = None,
        data_vars: list | None = None,
        data_vars_meta: dict | None = None,
        data_vars_encoding: dict | None = None,
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            ref (xr.Dataset | xr.DataArray): The reference dataarray or dataset to load the data for.
            cube_store (zarr.storage.StoreLike): The zarr storage to load the data from.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            dataset_title (str, optional): The title of the dataset. If set, overwrites the default title.
                Only used at first datacube creation. Defaults to None.
            cube_extent (GeoBox, optional): The extent of the datacube.
                If set, overwrites the default extent set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            chunk_size (int, optional): The chunk size of the datacube.
                If set, overwrites the default chunk size set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars (list, optional): The data variables of the datacube.
                If set, overwrites the data variables set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars_meta (dict, optional): The metadata of the data variables.
                If set, overwrites the metadata set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars_encoding (dict, optional): The encoding of the data variables.
                If set, overwrites the encoding set by the Dataset.
                Only used at first datacube creation. Defaults to None.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        return cls.load(
            ref.geobox,
            cube_store,
            buffer,
            persist,
            dataset_title=dataset_title,
            cube_extent=cube_extent,
            chunk_size=chunk_size,
            data_vars=data_vars,
            data_vars_meta=data_vars_meta,
            data_vars_encoding=data_vars_encoding,
        )

    @classmethod
    def load(
        cls,
        geobox: GeoBox,
        cube_store: zarr.storage.StoreLike,
        buffer: int = 0,
        persist: bool = True,
        *,
        dataset_title: str | None = None,
        cube_extent: GeoBox | None = None,
        chunk_size: int | None = None,
        data_vars: list | None = None,
        data_vars_meta: dict | None = None,
        data_vars_encoding: dict | None = None,
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            geobox (GeoBox): The reference geobox to load the data for.
            cube_store (zarr.storage.StoreLike): The zarr storage to load the data from.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            dataset_title (str, optional): The title of the dataset. If set, overwrites the default title.
                Only used at first datacube creation. Defaults to None.
            cube_extent (GeoBox, optional): The extent of the datacube.
                If set, overwrites the default extent set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            chunk_size (int, optional): The chunk size of the datacube.
                If set, overwrites the default chunk size set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars (list, optional): The data variables of the datacube.
                If set, overwrites the data variables set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars_meta (dict, optional): The metadata of the data variables.
                If set, overwrites the metadata set by the Dataset.
                Only used at first datacube creation. Defaults to None.
            data_vars_encoding (dict, optional): The encoding of the data variables.
                If set, overwrites the encoding set by the Dataset.
                Only used at first datacube creation. Defaults to None.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        tick_fstart = time.perf_counter()

        logger.debug(f"Found a reference resolution of {geobox.resolution.x}m")

        dataset_title = dataset_title or cls.__name__
        cube_extent = cube_extent or cls.extent
        chunk_size = chunk_size or cls.chunk_size
        data_vars = data_vars or cls.data_vars
        data_vars_meta = data_vars_meta or cls.data_vars_meta
        data_vars_encoding = data_vars_encoding or cls.data_vars_encoding

        # Check if the zarr data already exists
        cube_storage_path = sync(make_store_path(cube_store, mode="r"))
        if sync(cube_storage_path.is_empty()):
            logger.debug(f"Creating a new zarr datacube with {cube_storage_path=}")
            create_empty_datacube(
                dataset_title,
                cube_store,
                cube_extent,
                chunk_size,
                data_vars,
                data_vars_meta,
                data_vars_encoding,
            )

        # Download the adjacent tiles (if necessary)
        reference_geobox = geobox.to_crs(cube_extent.crs, resolution=cube_extent.resolution.x).pad(buffer)
        with download_lock:
            cls.procedural_download(cube_store, reference_geobox, data_vars)

        # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
        chunks = None if persist else "auto"
        xrcube = xr.open_zarr(cube_store, mask_and_scale=False, chunks=chunks).set_coords("spatial_ref")

        # Get an AOI slice of the datacube
        xrcube_aoi = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)

        # The following code would load the lazy zarr data from disk into memory
        if persist:
            tick_sload = time.perf_counter()
            xrcube_aoi = xrcube_aoi.load()
            tick_eload = time.perf_counter()
            logger.debug(f"{dataset_title} AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

        tused = time.perf_counter() - tick_fstart
        logger.debug(f"{dataset_title} tile {'loaded' if persist else 'lazy-opened'} in {tused:.2f} seconds")
        return xrcube_aoi
