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
import numpy as np
import odc.geo
import odc.geo.xr
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox, Resolution
from stopuhr import StopUhr
from zarr.codecs import BloscCodec
from zarr.core.sync import sync

from smart_geocubes._storage import optimize_coord_encoding
from smart_geocubes.exceptions import AlreadyDownloadedError

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


def _geobox_repr(geobox: GeoBox) -> str:
    """Get a better string representation of a geobox.

    Args:
        geobox (GeoBox): The geobox to represent.

    Returns:
        str: The string representation of the geobox.

    """
    crs = f"EPSG:{geobox.crs.epsg}" if geobox.crs.epsg else "Non-EPSG CRS"

    return f"GeoBox({geobox.shape}, Anchor[{geobox.affine.c} - {geobox.affine.f}], {crs})"


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
    """Base class for remote accessors.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the underlaying zarr array. Should be equal to the extent geobox.
            However, this property is used to find the target index of the downloaded data, so better save than sorry.
        created (bool): True if the datacube already exists in the storage.

    """

    # These class variables (not object properties) are meant to set by a specific dataset accesor
    extent: GeoBox
    chunk_size: int
    channels: ClassVar[list]
    _channels_meta: ClassVar[dict]
    _channels_encoding: ClassVar[dict]

    def __init__(
        self,
        storage: icechunk.Storage | Path | str,
        create_icechunk_storage: bool = True,
    ):
        """Initialize base class for remote accessors.

        !!! warning

            In a multiprocessing environment, it is strongly recommended to not set `create_icechunk_storage=False`.

        Args:
            storage (icechunk.Storage): The icechunk storage of the datacube.
            create_icechunk_storage (bool, optional): If an icechunk repository should be created at provided storage
                if no exists.
                This should be disabled in a multiprocessing environment.
                Defaults to True.

        Raises:
            ValueError: If the storage is not an icechunk.Storage.

        """
        # Title is used for logging, debugging and as a default name for the datacube
        self.title = self.__class__.__name__

        if isinstance(storage, (str | Path)):
            storage = storage if isinstance(storage, str) else str(storage.resolve())
            storage = icechunk.local_filesystem_storage(storage)
        if not isinstance(storage, icechunk.Storage):
            raise ValueError(f"Expected an icechunk.Storage, but got {type(storage)}")
        self.storage = storage
        logger.debug(f"Using storage {storage=}")
        if create_icechunk_storage:
            self.repo = icechunk.Repository.open_or_create(storage)  # Will create a "main" branch
        else:
            self.repo = icechunk.Repository.open(storage)
        logger.debug(f"Using repository {self.repo=}")

        # The benchmarking timer for this accessor
        self.stopuhr = StopUhr(logger.debug)

        # The TypeVar used by the ThreadingHandler was added in 3.12
        # The Shutdown method of the queue was added in 3.13
        # Hence, we don't want to import the module unless Python 3.13 is installed
        if _check_python_version(3, 13):
            from smart_geocubes._concurrency.threading import ThreadingHandler

            self._threading_handler = ThreadingHandler(self._threading_download)

        self.post_init()

    def __repr__(self):  # noqa: D105
        return f"{self.title}({_geobox_repr(self.extent)}, {self.channels})"

    def __str__(self):  # noqa: D105
        return self.__repr__()

    def post_init(self):
        """Post init actions. Can be overwritten by the dataset accessor."""
        pass

    def log_benchmark_summary(self):
        """Log the benchmark summary."""
        self.stopuhr.summary()

    @cached_property
    def zgeobox(self) -> GeoBox:  # noqa: D102
        session = self.repo.readonly_session("main")
        zcube = zarr.open(store=session.store, mode="r")
        res = Resolution(zcube["x"].attrs.get("resolution"), zcube["y"].attrs.get("resolution"))
        return GeoBox.from_bbox(
            (zcube["x"][0], zcube["y"][-1], zcube["x"][-1], zcube["y"][0]),
            resolution=res,
            crs=zcube["spatial_ref"].attrs["crs_wkt"],
        )

    @property
    def created(self) -> bool:  # noqa: D102
        session = self.repo.readonly_session("main")
        return not sync(session.store.is_empty(""))

    @property
    def loaded_tiles(self) -> list[str]:  # noqa: D102
        session = self.repo.readonly_session("main")
        zcube = zarr.open(store=session.store, mode="r")
        return zcube.attrs["loaded_tiles"].copy()

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

    def create(self, overwrite: bool = False, exists_ok: bool = False):
        """Create an empty datacube and write it to the store.

        Args:
            overwrite (bool, optional): Allowing overwriting an existing datacube.
                Has no effect if exists_ok is True. Defaults to False.
            exists_ok (bool, optional): Do not raise an error if the datacube already exists.

        Raises:
            FileExistsError: If a datacube already exists at location and exists_ok is False.

        """
        if exists_ok and self.created:
            logger.debug("Datacube was already created.")
            return

        with self.stopuhr("Empty datacube creation"):
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
                        dtype=self._channels_encoding[name].get("dtype", "float32"),
                        always_yx=True,
                    )
                    for name in self.channels
                },
                attrs={"title": self.title, "loaded_tiles": []},
            )

            # Add metadata
            for name, meta in self._channels_meta.items():
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
                    **self._channels_encoding[name],
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
            logger.debug(f"Datacube created: {commit=}")

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

        Keyword Args:
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
        return self.load(_geobox_repr(ref.geobox), **kwargs)

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
        with self.stopuhr(f"{_geobox_repr(geobox)}: {self.title} tile {'loading' if persist else 'lazy-loading'}"):
            logger.debug(f"{_geobox_repr(geobox)}: {geobox.resolution} original resolution")

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
                with self.stopuhr(f"{_geobox_repr(geobox)}: {self.title} AOI loading from disk"):
                    xrcube_aoi = xrcube_aoi.load()
        return xrcube_aoi

    def download(self, roi: GeoBox | gpd.GeoDataFrame):
        """Download the data for the given region of interest which can be provided either as GeoBox or GeoDataFrame.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe to download the data for.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            ValueError: If no tries are left.

        """
        roi_repr = _geobox_repr(roi) if isinstance(roi, GeoBox) else "GeoDataframe"
        with self.stopuhr(f"Download of {roi_repr}"):
            adjacent_tiles = self.adjacent_tiles(roi)
            if not adjacent_tiles:
                logger.error(f"No adjacent tiles found: {adjacent_tiles=}")
                raise ValueError("No adjacent tiles found - is the provided geobox corrent?")

            session = self.repo.readonly_session("main")
            zcube = zarr.open(store=session.store, mode="r")
            loaded_tiles = zcube.attrs.get("loaded_tiles", [])
            new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
            logger.debug(f"{len(adjacent_tiles)=} & {len(loaded_tiles)=} -> {len(new_tiles)=} to download")
            if not new_tiles:
                return

            for tile in new_tiles:
                with self.stopuhr(f"{tile.id=}: Downloading one new tile in blocking mode"):
                    logger.debug(f"{tile.id=}: Start downloading")
                    tiledata = self.download_tile(tile)

                # Try to write the data to file until a limit is reached
                limit = 100
                for i in range(limit):
                    try:
                        self._write_tile_to_zarr(tiledata, tile)
                        break
                    except icechunk.ConflictError as conflict_error:
                        logger.debug(f"{tile.id=}: {conflict_error=} at retry {i}/{limit}")
                else:
                    logger.error(
                        f"{tile.id=}: {limit} tries to write the tile failed. "
                        "Please check if the datacube is already created and not empty."
                    )
                    raise ValueError(f"{tile.id=}: {limit} tries to write the tile failed.")

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
            raise ValueError("Threading mode is currently disabled. Use 'blocking' instead.")
            # self.procedural_download_threading(geobox)
        else:
            raise ValueError(f"Unknown concurrency mode {concurrency_mode}")

    def procedural_download_blocking(self, geobox: GeoBox):
        """Download tiles procedurally in blocking mode.

        Warning:
            This method is meant for single-process use, but can (in theory) be used in a multi-process environment.
            However, in a multi-process environment it can happen that multiple processes try to write concurrently,
            which results in a conflict.
            In such cases, the download will be retried until it succeeds or the number of maximum-tries is reached.

        Args:
            geobox (GeoBox): The geobox of the aoi to download.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            ValueError: If no tries are left.

        """
        with self.stopuhr(f"{_geobox_repr(geobox)}: Procedural download in blocking mode"):
            adjacent_tiles = self.adjacent_tiles(geobox)
            if not adjacent_tiles:
                logger.error(f"{_geobox_repr(geobox)}: No adjacent tiles found: {adjacent_tiles=}")
                raise ValueError("No adjacent tiles found - is the provided geobox corrent?")

            session = self.repo.readonly_session("main")
            zcube = zarr.open(store=session.store, mode="r")
            loaded_tiles = zcube.attrs.get("loaded_tiles", [])
            new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
            logger.debug(
                f"{_geobox_repr(geobox)}:  {len(adjacent_tiles)=} & {len(loaded_tiles)=}"
                f" -> {len(new_tiles)=} to download"
            )
            if not new_tiles:
                return

            for tile in new_tiles:
                with self.stopuhr(f"{tile.id=}: Downloading one new tile in blocking mode"):
                    logger.debug(f"{tile.id=}: Start downloading")
                    tiledata = self.download_tile(tile)

                # Try to write the data to file until a limit is reached
                limit = 100
                for i in range(limit):
                    try:
                        self._write_tile_to_zarr(tiledata, tile)
                        break
                    except icechunk.ConflictError as conflict_error:
                        logger.debug(f"{tile.id=}: {conflict_error=} at retry {i}/{limit}")
                else:
                    logger.error(
                        f"{tile.id=}: {limit} tries to write the tile failed. "
                        "Please check if the datacube is already created and not empty."
                    )
                    raise ValueError(f"{tile.id=}: {limit} tries to write the tile failed.")

    def _write_tile_to_zarr(self, tiledata: xr.Dataset, tile: TileWrapper):
        with self.stopuhr(f"{tile.id=}: Writing tile to zarr"):
            session = self.repo.writable_session("main")
            zcube = zarr.open(store=session.store, mode="r+")
            loaded_tiles = zcube.attrs.get("loaded_tiles", [])

            # Check if the tile was already written from another process, then do nothing
            if tile.id in loaded_tiles:
                logger.debug(f"{tile.id=}: Already loaded")
                return

            # Get the slice of the datacube where the tile will be written
            logger.debug(
                f"{tile.id=}: {tiledata.sizes=} {tiledata.x[0].item()=} {tiledata.y[0].item()=}"
                f" {zcube['x'][0]=} {zcube['y'][0]=}"
            )
            target_slice = self.zgeobox.overlap_roi(tiledata.odc.geobox)
            logger.debug(f"{tile.id=}: Writing to {target_slice=}")

            for channel in self.channels:
                raw_data = tiledata[channel].values
                # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
                # Replace these nan borders with existing data if there is any
                raw_data = np.where(~np.isnan(raw_data), raw_data, zcube[channel][target_slice])
                zcube[channel][target_slice] = raw_data

            loaded_tiles.append(tile.id)
            zcube.attrs["loaded_tiles"] = loaded_tiles
            session.commit(f"Write tile {tile.id}")

    def _threading_download(self, tile: TileWrapper):
        session = self.repo.writable_session("main")
        zcube = zarr.open(store=session.store, mode="r+")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])

        if tile.id in loaded_tiles:
            logger.debug(f"{tile.id=} Already loaded")
            raise AlreadyDownloadedError

        with self.stopuhr(f"{tile.id=}: Downloading one new tile in threading mode"):
            logger.debug(f"{tile.id=}: Start downloading")
            tiledata = self.download_tile(zcube, tile)

        self._write_tile_to_zarr(tiledata, tile)

        loaded_tiles.append(tile.id)
        zcube.attrs["loaded_tiles"] = loaded_tiles
        # session.rebase(icechunk.ConflictDetector())
        commit = session.commit(f"Procedurally downloaded {tile.id=} in threading mode")
        logger.debug(f"{tile.id=}: {commit=}")

    def procedural_download_threading(self, geobox: GeoBox):
        """Download tiles procedurally in threading mode.

        Note:
            This method ensures that only a single download is running at a time.
            It uses a SetQueue to prevent duplicate downloads.
            The threading mode requires Python 3.13 or higher.

        Args:
            geobox (GeoBox): The geobox of the aoi to download.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            RuntimeError: If the Python version is lower than 3.13.

        """
        if not _check_python_version(3, 13):
            raise RuntimeError("Threading mode requires Python 3.13 or higher")
        with self._threading_handler:
            adjacent_tiles = self.adjacent_tiles(geobox)
            if not adjacent_tiles:
                logger.error(f"{_geobox_repr(geobox)}: No adjacent tiles found: {adjacent_tiles=}")
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
                        f"{_geobox_repr(geobox)}: {len(done_tiles)} of {len(adjacent_tiles)} downloaded."
                        f" Missing: {[t.id for t in new_tiles]} Done: {[t.id for t in done_tiles]}"
                    )
                for tile in new_tiles:
                    self._threading_handler._queue.put(tile)
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
    def adjacent_tiles(self, roi: GeoBox | gpd.GeoDataFrame) -> list[TileWrapper]:
        """Get the adjacent tiles for the given geobox.

        Must be implemented by the Accessor.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe

        Returns:
            list[TileWrapper]: The adjacent tile(-id)s for the given geobox.

        """

    @abstractmethod
    def download_tile(self, tile: TileWrapper) -> xr.Dataset:
        """Download the data for the given tile.

        Must be implemented by the Accessor.

        Args:
            tile (TileWrapper): The reference tile to download the data for.

        Returns:
            xr.Dataset: The downloaded tile data.

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
