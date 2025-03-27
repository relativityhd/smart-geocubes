"""Base class for remote accessors."""

import logging
import sys
import time
from abc import ABC, abstractmethod
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, TypedDict, Unpack

import geopandas as gpd
import icechunk
import odc.geo
import odc.geo.xr
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox, Resolution
from zarr.codecs import BloscCodec
from zarr.core.sync import sync

from smart_geocubes.concurrency import AlreadyDownloadedError
from smart_geocubes.storage import optimize_coord_encoding

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass

logger = logging.getLogger(__name__)


ConcurrencyModes = Literal["blocking", "threading"]  # Potential future modes: "dask", "process", "server"


def _check_python_version(min_major: int, min_minor: int) -> bool:
    """Check the current Python version against a minimum required version.

    Args:
        min_major (int): The minimum required major version.
        min_minor (int): The minimum required minor version.

    Returns:
        bool: True if the current Python version is greater than or equal to the minimum required

    """
    return sys.version_info.major > min_major or (
        sys.version_info.major == min_major and sys.version_info.minor >= min_minor
    )


class LoadParams(TypedDict):
    """TypedDict for the load function parameters."""

    buffer: int
    persist: bool
    create: bool
    concurrency_mode: ConcurrencyModes


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
        storage: icechunk.Storage | Path | str,
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
            ### Creation specific
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
        if isinstance(storage, (str | Path)):
            storage = storage if isinstance(storage, str) else str(storage.resolve())
            storage = icechunk.local_filesystem_storage(storage)
        if not isinstance(storage, icechunk.Storage):
            raise ValueError(f"Expected an icechunk.Storage, but got {type(storage)}")
        self.storage = storage
        logger.debug(f"Using storage {storage=}")
        # TODO: disable "create" if running in a multi-process environment
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

        # TODO: store the settings variables in metadata and validate here

        # The TypeVar used by the ThreadingHandler was added in 3.12
        # The Shutdown method of the queue was added in 3.13
        # Hence, we don't want to import the module unless Python 3.13 is installed
        if _check_python_version(3, 13):
            from smart_geocubes.concurrency.threading import ThreadingHandler

            self.threading_handler = ThreadingHandler(self._threading_download)

    # TODO: is this equal to self.extent?
    @cached_property
    def geobox(self) -> GeoBox:
        """Turn a zarr datacube into a GeoBox.

        Returns:
            GeoBox: The GeoBox created from the zarr datacube.

        """
        session = self.repo.readonly_session("main")
        zcube = zarr.open(store=session.store, mode="r")
        res = Resolution(zcube["x"].attrs.get("resolution"), zcube["y"].attrs.get("resolution"))
        return GeoBox.from_bbox(
            (zcube["x"][0], zcube["y"][-1], zcube["x"][-1], zcube["y"][0]),
            resolution=res,
            crs=zcube["spatial_ref"].attrs["crs_wkt"],
        )

    @property
    def created(self) -> bool:
        """Check if the datacube already exists in the storage.

        Returns:
            bool: True if the datacube already exists in the storage.

        """
        session = self.repo.readonly_session("main")
        return not sync(session.store.is_empty(""))

    def assert_created(self):
        """Assert that the datacube exists in the storage.

        Raises:
            FileNotFoundError: If the datacube does not exist.

        """
        if not self.created:
            msg = f"Datacube {self.title} does not exist."
            " Please use the `create` method or pass `create=True` to `load`."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def create(self, overwrite: bool = False):
        """Create an empty datacube and write it to the store.

        Args:
            overwrite (bool, optional): Allowing overwriting an existing datacube. Defaults to False.

        Raises:
            FileExistsError: If a datacube already exists at location

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
                    self.extent,
                    chunks=-1,
                    dtype=self.channels_encoding[name].get("dtype", "float32"),
                    always_yx=True,
                )
                for name in self.channels
            },
            attrs={"title": self.title, "loaded_tiles": []},
        )

        # Add metadata
        for name, meta in self.channels_meta.items():
            ds[name].attrs.update(meta)

        # Get the encoding for the coordinates, variables and spatial reference
        coords_encoding = {
            "x": {"chunks": ds.x.shape, **optimize_coord_encoding(ds.x.values, self.extent.resolution.x)},
            "y": {"chunks": ds.y.shape, **optimize_coord_encoding(ds.y.values, self.extent.resolution.y)},
        }
        var_encoding = {
            name: {
                "chunks": (self.chunk_size, self.chunk_size),
                "compressors": [BloscCodec(clevel=9)],
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

        self.post_create()

    def post_create(self):
        """Post create actions. Can be overwritten by the dataset accessor."""
        pass

    def load_like(
        self,
        ref: xr.Dataset | xr.DataArray,
        **kwargs: Unpack[LoadParams],
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            ref (xr.Dataset | xr.DataArray): The reference dataarray or dataset to load the data for.
            **kwargs: The load parameters (buffer, persist, create, concurrency_mode).
            ### Kwargs:
            buffer (int, optional): The buffer around the projected geobox in pixels. Defaults to 0.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                This is not recommended, because it can have side effects in a multi-process environment.
                Defaults to False.
            concurrency_mode (ConcurrencyModes, optional): The concurrency mode for the download.
                Defaults to "blocking".

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        return self.load(geobox=ref.geobox, **kwargs)

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
            concurrency_mode (ConcurrencyModes, optional): The concurrency mode for the download.
                Defaults to "blocking".

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        tick_fstart = time.perf_counter()

        logger.debug(f"{geobox=}: {geobox.resolution.x}m original resolution")

        # Create the datacube if it does not exist
        if create:
            try:
                self.create(overwrite=False)
            except FileExistsError:  # We are okay if the datacube already exists
                pass
        else:
            # Check if the datacube exists
            self.assert_created()

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
            logger.debug(f"{geobox=}: {self.title} AOI loaded from disk in {tick_eload - tick_sload:.2f} seconds")

        tused = time.perf_counter() - tick_fstart
        logger.debug(f"{geobox=}: {self.title} tile {'loaded' if persist else 'lazy-opened'} in {tused:.2f} seconds")
        return xrcube_aoi

    def procedural_download(self, geobox: GeoBox, concurrency_mode: ConcurrencyModes = "blocking"):
        """Download the data for the given geobox.

        Note:
            The "threading" concurrency mode requires Python 3.13 or higher.

        Args:
            geobox (GeoBox): The reference geobox to download the data for.
            concurrency_mode (ConcurrencyModes, optional): The concurrency mode for the download.
                Defaults to "blocking".

        Raises:
            ValueError: If an unknown concurrency mode is provided.

        """
        self.assert_created()
        if concurrency_mode == "blocking":
            self.procedural_download_blocking(geobox)
        elif concurrency_mode == "threading":
            self.procedural_download_threading(geobox)
        else:
            raise ValueError(f"Unknown concurrency mode {concurrency_mode}")

    def procedural_download_blocking(self, geobox: GeoBox, tries: int = 5):
        """Download tiles procedurally in blocking mode.

        Warning:
            This method is meant for single-process use, but can (in theory) be used in a multi-process environment.
            However, in a multi-process environment it can happen that multiple processes try to write concurrently,
            which results in a conflict.
            In such cases, the download will be retried until it succeeds or the number of maximum-tries is reached.

        Args:
            accessor (RemoteAccessor): The accessor.
            geobox (GeoBox): The geobox of the aoi to download.
            tries (int, optional): Number of maximum tries. Defaults to 5.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            ValueError: If no tries are left.

        """
        if tries == 0:
            logger.warning("No tries left, skipping download")
            raise ValueError("Unable to commit, no tries left.")

        tick_fstart = time.perf_counter()
        adjacent_tiles = self.adjacent_tiles(geobox)
        if not adjacent_tiles:
            logger.error(f"{geobox=}: No adjacent tiles found: {adjacent_tiles=}")
            raise ValueError("No adjacent tiles found - is the provided geobox corrent?")

        session = self.repo.writable_session("main")
        zcube = zarr.open(store=session.store, mode="r+")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])
        new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
        logger.debug(f"{geobox=}:  {len(adjacent_tiles)=} & {len(loaded_tiles)=} -> {len(new_tiles)=} to download")
        if not new_tiles:
            return

        for tile in new_tiles:
            logger.debug(f"{tile.id=}: Start downloading")
            tick_dstart = time.perf_counter()
            self.download_tile(zcube, tile)
            tick_dend = time.perf_counter()
            logger.debug(f"{tile.id=}: Done downloading in {tick_dend - tick_dstart:.2f} seconds")
            loaded_tiles.append(tile.id)
            zcube.attrs["loaded_tiles"] = loaded_tiles

        try:
            # session.rebase(icechunk.ConflictDetector())
            session.commit(f"Procedurally downloaded tiles {[tile.id for tile in new_tiles]} in blocking mode")
        # Currently not possible, because attrs will always result in a conflict
        # except icechunk.RebaseFailedError as e:
        #     logger.warning(f"Rebase failed: {e}")
        #     logger.debug(f"Retrying download with {tries - 1} tries left")
        #     self.procedural_download_blocking(geobox, tries=tries - 1)
        except icechunk.ConflictError as e:
            logger.warning(f"Icechunk session is out of sync: {e}")
            logger.debug(f"Retrying download with {tries - 1} tries left")
            self.procedural_download_blocking(geobox, tries=tries - 1)

        tick_fend = time.perf_counter()
        logger.info(f"Downloaded {len(new_tiles)} tiles in {tick_fend - tick_fstart:.2f} seconds")

    def _threading_download(self, tile: TileWrapper):
        tick_fstart = time.perf_counter()
        session = self.repo.writable_session("main")
        zcube = zarr.open(store=session.store, mode="r+")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])

        if tile.id in loaded_tiles:
            logger.debug(f"{tile.id=} Already loaded")
            raise AlreadyDownloadedError

        logger.debug(f"{tile.id=}: Start downloading")
        tick_dstart = time.perf_counter()
        self.download_tile(zcube, tile)
        tick_dend = time.perf_counter()
        logger.debug(f"{tile.id=}: Done downloading in {tick_dend - tick_dstart:.2f} seconds")
        loaded_tiles.append(tile.id)
        zcube.attrs["loaded_tiles"] = loaded_tiles
        # session.rebase(icechunk.ConflictDetector())
        session.commit(f"Procedurally downloaded {tile.id=} in threading mode")
        tick_fend = time.perf_counter()
        logger.info(f"Downloaded one new tile in {tick_fend - tick_fstart:.2f} seconds")

    def procedural_download_threading(self, geobox: GeoBox):
        """Download tiles procedurally in threading mode.

        Note:
            This method ensures that only a single download is running at a time.
            It uses a SetQueue to prevent duplicate downloads.
            The threading mode requires Python 3.13 or higher.

        Args:
            accessor (RemoteAccessor): The accessor.
            geobox (GeoBox): The geobox of the aoi to download.
            tries (int, optional): Number of maximum tries. Defaults to 5.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            RuntimeError: If the Python version is lower than 3.13.

        """
        if not _check_python_version(3, 13):
            raise RuntimeError("Threading mode requires Python 3.13 or higher")
        with self.threading_handler:
            adjacent_tiles = self.adjacent_tiles(geobox)
            if not adjacent_tiles:
                logger.error(f"{geobox=}: No adjacent tiles found: {adjacent_tiles=}")
                raise ValueError("No adjacent tiles found - is the provided geobox corrent?")

            # Wait until all new_items are loaded
            prev_len = None
            while True:
                session = self.repo.readonly_session("main")
                zcube = zarr.open(store=session.store, mode="r")
                loaded_tiles = zcube.attrs.get("loaded_tiles", [])
                new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
                done_tiles = [tile for tile in adjacent_tiles if tile.id in loaded_tiles]
                if not new_tiles:
                    break
                if prev_len != len(new_tiles):
                    logger.debug(
                        f"{geobox=}: {len(done_tiles)} of {len(adjacent_tiles)} downloaded."
                        f" Missing: {[t.id for t in new_tiles]} Done: {[t.id for t in done_tiles]}"
                    )
                for tile in new_tiles:
                    self.threading_handler._queue.put(tile)
                prev_len = len(new_tiles)
                time.sleep(5)

    def open_zarr(self) -> zarr.Group:
        """Open the zarr datacube in read-only mode.

        Returns:
            zarr.Group: The zarr datacube.

        """
        self.assert_created()
        session = self.repo.readonly_session("main")
        zcube = zarr.open(store=session.store, mode="r")
        return zcube

    def open_xarray(self) -> xr.Dataset:
        """Open the xarray datacube in read-only mode.

        Returns:
            xr.Dataset: The xarray datacube.

        """
        self.assert_created()
        session = self.repo.readonly_session("main")
        xcube = xr.open_zarr(session.store, mask_and_scale=False, consolidated=False).set_coords("spatial_ref")
        return xcube

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

    @abstractmethod
    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles / chunk.

        Must be implemented by the Accessor.

        Returns:
            gpd.GeoDataFrame | None: Tile or Chunk info.

        """

    @abstractmethod
    def visualize_state(self, ax: "plt.Axes | None" = None) -> "plt.Figure | plt.Axes":
        """Visulize currently stored tiles / chunk.

        Must be implemented by the DatasetAccessor.

        Args:
            ax (plt.Axes | None, optional): The axes drawn to. If None, will create a new figure and axes.
                Defaults to None.

        Returns:
            plt.Figure | plt.Axes: The figure with the visualization

        """
