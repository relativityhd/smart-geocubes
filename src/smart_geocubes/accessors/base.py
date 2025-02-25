"""Base class for remote accessors."""

import logging
import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

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

    # These class variables (not object properties) are meant to set by a specific dataset accesor
    extent: GeoBox
    chunk_size: int
    channels: ClassVar[list]
    channels_meta: ClassVar[dict]
    channels_encoding: ClassVar[dict]

    def __init__(
        self,
        storage: zarr.storage.StoreLike,
        extent: GeoBox | None = None,
        chunk_size: int | None = None,
        channels: list | None = None,
        channels_meta: dict | None = None,
        channels_encoding: dict | None = None,
    ):
        """Initialize base class for remote accessors.

        Args:
            storage (zarr.storage.StoreLike): Where the datacube is stored.

        """
        self.storage = storage
        # We overwrite optionally the dataset attributes with user defined settings
        if extent is not None:
            self.extent = extent
        if chunk_size is not None:
            self.chunk_size = chunk_size
        if channels is not None:
            self.channels = channels
        if channels_meta is not None:
            self.channels_meta
        if channels_encoding is not None:
            self.channels_encoding

        # TODO: store the settings variables in zarr metadata and validate here

    def create(self, overwrite: bool = False):
        """Create an empty zarr datacube on disk.

        Args:
            overwrite (bool, optional): Allowing overwriting an existing datacube. Defaults to False.

        Raises:
            FileExistsError: If a datacube already exists at location

        """
        # Check if the zarr data already exists
        cube_storage_path = sync(make_store_path(self.storage, mode="r"))
        if not overwrite and not sync(cube_storage_path.is_empty()):
            raise FileExistsError(f"Cannot create a new  datacube. {cube_storage_path} ({self.storage}) is not empty!")

        logger.debug(f"Creating a new zarr datacube with {cube_storage_path=}")
        create_empty_datacube(
            title=type(self).__name__,
            storage=self.storage,
            geobox=self.extent,
            chunk_size=self.chunk_size,
            data_vars=self.channels,
            meta=self.channels_meta,
            var_encoding=self.channels_encoding,
            overwrite=overwrite,
        )

    @abstractmethod
    def procedural_download(self, geobox: GeoBox):
        """Download the data for the given geobox.

        Must be implemented by the Accessor.

        Args:
            geobox (GeoBox): The reference geobox to download the data for.

        """

    @abstractmethod
    def current_state(self) -> "gpd.GeoDataFrame | None":
        """Get info about currently stored tiles / chunk.

        Returns:
            gpd.GeoDataFrame | None: Tile or Chunk info.

        """

    @abstractmethod
    def visualize_state(self, ax: "plt.Axes | None" = None) -> "plt.Figure | plt.Axes":
        """Visulize currently stored tiles / chunk.

        Args:
            ax (plt.Axes | None, optional): The axes drawn to. If None, will create a new figure and axes.
                Defaults to None.

        Returns:
            plt.Figure | plt.Axes: The figure with the visualization

        """

    def load_like(
        self,
        ref: xr.Dataset | xr.DataArray,
        buffer: int = 0,
        persist: bool = True,
        create: bool = False,
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            ref (xr.Dataset | xr.DataArray): The reference dataarray or dataset to load the data for.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                Defaults to False.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        return self.load(geobox=ref.geobox, buffer=buffer, persist=persist, create=create)

    def load(self, geobox: GeoBox, buffer: int = 0, persist: bool = True, create: bool = False) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            geobox (GeoBox): The reference geobox to load the data for.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                Defaults to False.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        tick_fstart = time.perf_counter()

        logger.debug(f"Found a reference resolution of {geobox.resolution.x}m")

        # Check if the zarr data already exists
        if create:
            cube_storage_path = sync(make_store_path(self.storage, mode="r"))
            if sync(cube_storage_path.is_empty()):
                self.create(overwrite=True)  # We can savely overwrite here, since the storage path is empty

        # Download the adjacent tiles (if necessary)
        reference_geobox = geobox.to_crs(self.extent.crs, resolution=self.extent.resolution.x).pad(buffer)
        with download_lock:
            self.procedural_download(reference_geobox)

        # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
        chunks = None if persist else "auto"
        xrcube = xr.open_zarr(self.storage, mask_and_scale=False, chunks=chunks).set_coords("spatial_ref")

        # Get an AOI slice of the datacube
        xrcube_aoi = xrcube.odc.crop(reference_geobox.extent, apply_mask=False)

        # The following code would load the lazy zarr data from disk into memory
        if persist:
            tick_sload = time.perf_counter()
            xrcube_aoi = xrcube_aoi.load()
            tick_eload = time.perf_counter()
            logger.debug(f"{type(self).__name__} AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

        tused = time.perf_counter() - tick_fstart
        logger.debug(f"{type(self).__name__} tile {'loaded' if persist else 'lazy-opened'} in {tused:.2f} seconds")
        return xrcube_aoi
