"""Base class for remote accessors."""

import logging
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple

import icechunk
import odc.geo
import odc.geo.xr
import xarray as xr
import zarr
from numcodecs.zarr3 import Blosc
from odc.geo.geobox import GeoBox
from zarr.core.sync import sync

from smart_geocubes.storage import optimize_coord_encoding

if TYPE_CHECKING:
    try:
        import geopandas as gpd
        import matplotlib.pyplot as plt
    except ImportError:
        pass

logger = logging.getLogger(__name__)


ConcurrencyModes = Literal["blocking", "threading"]  # Potential future modes: "dask", "process", "server"


class TileWrapper(NamedTuple):
    """Wrapper for a tile with an id."""

    id: str
    item: Any


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
        storage: icechunk.Storage,
        title: str | None = None,
        extent: GeoBox | None = None,
        chunk_size: int | None = None,
        channels: list | None = None,
        channels_meta: dict | None = None,
        channels_encoding: dict | None = None,
    ):
        """Initialize base class for remote accessors.

        Args:
            storage (icechunk.Storage): The icechunk storage of the datacube.
            ### Storage specific
            The following arguments will overwrite the dataset defaults
            and are only necessary for the creation of a new datacube.

            title (str | None, optional): The title of the datacube. Defaults to None.
            extent (GeoBox | None, optional): The extent of the datacube. Defaults to None.
            chunk_size (int | None, optional): The chunk size of the datacube. Defaults to None.
            channels (list | None, optional): The channels of the datacube. Defaults to None.
            channels_meta (dict | None, optional): The channels meta of the datacube. Defaults to None.
            channels_encoding (dict | None, optional): The channels encoding of the datacube. Defaults to None.

        Raises:
            ValueError: If the storage is not an icechunk.Storage.

        """
        if not isinstance(storage, icechunk.Storage):
            raise ValueError(f"Expected an icechunk.Storage, but got {type(storage)}")
        self.storage = storage
        logger.debug(f"Using storage {storage=}")
        self.repo = icechunk.Repository.open_or_create(storage)  # Will create a "main" branch
        logger.debug(f"Using repository {self.repo=}")

        # We overwrite optionally the dataset attributes with user defined settings
        self.title = title or type(self).__name__
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
            ValueError: If the extent does not have x and y coordinates.

        """
        tick_fstart = time.perf_counter()
        # Check if the zarr data already exists
        session = self.repo.writable_session("main")
        cube_is_empty = sync(session.store.is_empty(""))
        if not overwrite and not cube_is_empty:
            logger.debug(f"Unable to create a new datacube. {overwrite=} {cube_is_empty=} {session.store=}")
            raise FileExistsError(f"Cannot create a new  datacube. {session.store=} is not empty!")

        logger.debug(
            f"Creating an empty zarr datacube '{self.title}' with the variables"
            f" {self.channels} at a {self.extent.resolution=} (epsg:{self.extent.crs.epsg})"
            f" and {self.chunk_size=} to {session.store=}"
        )

        ds = xr.Dataset(
            {
                name: odc.geo.xr.xr_zeros(
                    self.extent, chunks=-1, dtype=self.channels_encoding[name].get("dtype", "float32")
                )
                for name in self.channels
            },
            attrs={"title": self.title, "loaded_tiles": []},
        )

        # Add metadata
        for name, meta in self.channels_meta.items():
            ds[name].attrs.update(meta)

        # Turn lat-lon into x-y
        if self.extent.crs.epsg == 4326:
            nmap = {"longitude": "x", "latitude": "y"}
            ds = ds.rename(nmap)

        if "x" not in ds.coords or "y" not in ds.coords:
            raise ValueError(f"The dataset must have x and y coordinates, but got {ds.coords.keys()}")

        # Get the encoding for the coordinates, variables and spatial reference
        coords_encoding = {
            "x": {"chunks": ds.x.shape, **optimize_coord_encoding(ds.x.values, self.extent.resolution.x)},
            "y": {"chunks": ds.y.shape, **optimize_coord_encoding(ds.y.values, self.extent.resolution.y)},
        }
        var_encoding = {
            name: {
                "chunks": (self.chunk_size, self.chunk_size),
                "compressors": [Blosc(cname="zstd")],
                **self.channels_encoding[name],
            }
            for name in self.channels
        }
        encoding = {
            "spatial_ref": {"chunks": None, "dtype": "int32"},
            **coords_encoding,
            **var_encoding,
        }
        logger.debug(f"Datacube {encoding=}")

        ds.to_zarr(
            session.store,
            encoding=encoding,
            compute=False,
            consolidated=False,
            zarr_format=3,
            mode="w" if overwrite else "w-",
        )

        commit = session.commit("Initialize empty datacube")

        tick_fend = time.perf_counter()
        logger.debug(f"Empty datacube {commit=} created in {tick_fend - tick_fstart:.2f} seconds")

    def load_like(
        self,
        ref: xr.Dataset | xr.DataArray,
        buffer: int = 0,
        persist: bool = True,
        create: bool = False,
        concurrency_mode: ConcurrencyModes = "blocking",
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            ref (xr.Dataset | xr.DataArray): The reference dataarray or dataset to load the data for.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                This is not recommended, because it can have side effects in a multi-process environment.
                Defaults to False.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        return self.load(geobox=ref.geobox, buffer=buffer, persist=persist, create=create)

    def load(
        self,
        geobox: GeoBox,
        buffer: int = 0,
        persist: bool = True,
        create: bool = False,
        concurrency_mode: ConcurrencyModes = "blocking",
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            geobox (GeoBox): The reference geobox to load the data for.
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                This is not recommended, because it can have side effects in a multi-process environment.
                Defaults to False.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        tick_fstart = time.perf_counter()

        logger.debug(f"Found a reference resolution of {geobox.resolution.x}m")

        # Create the datacube if it does not exist
        if create:
            try:
                self.create(overwrite=False)
            except FileExistsError:  # We are okay if the datacube already exists
                pass

        # Download the adjacent tiles (if necessary)
        reference_geobox = geobox.to_crs(self.extent.crs, resolution=self.extent.resolution.x).pad(buffer)
        self.procedural_download(reference_geobox, concurrency_mode=concurrency_mode)

        # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
        session = self.repo.readonly_session("main")
        chunks = None if persist else "auto"
        xrcube = xr.open_zarr(
            session.store,
            mask_and_scale=False,
            chunks=chunks,
            consolidated=False,
        ).set_coords("spatial_ref")

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

    @abstractmethod
    def adjacent_tiles(self, geobox: GeoBox) -> list[TileWrapper]:
        """Get the adjacent tiles for the given geobox.

        Must be implemented by the Accessor.

        Args:
            geobox (GeoBox): The reference geobox to get the adjacent tiles for.

        Returns:
            list[TileWrapper]: The adjacent tile(-id)s for the given geobox.

        """

    @abstractmethod
    def download_tile(self, zcube: zarr.Array, tile: TileWrapper):
        """Download the data for the given tile.

        Must be implemented by the Accessor.

        Args:
            zcube (zarr.Array): The datacube to write the tile data to.
            tile (TileWrapper): The reference tile to download the data for.

        """

    # TODO: move or delete -> think about type hints
    @abstractmethod
    def procedural_download(self, geobox: GeoBox, concurrency_mode: ConcurrencyModes = "blocking"):
        """Download the data for the given geobox.

        Must be implemented by the Concurrency-Mixins.

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
