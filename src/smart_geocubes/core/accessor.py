"""Base class for remote accessors."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, TypedDict, Unpack

import geopandas as gpd
import icechunk
import odc.geo.xr
import pandas as pd
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry
from stopuhr import Chronometer
from zarr.codecs import BloscCodec
from zarr.core.sync import sync

from smart_geocubes.backends import SimpleBackend, ThreadedBackend
from smart_geocubes.core.patches import PatchIndex
from smart_geocubes.core.storage import optimize_coord_encoding, optimize_temporal_encoding
from smart_geocubes.core.toi import TOI, _repr_toi
from smart_geocubes.core.utils import _check_python_version, _geobox_repr, _geometry_repr

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class LoadParams(TypedDict):
    """TypedDict for the load function parameters."""

    buffer: int
    persist: bool
    create: bool


class RemoteAccessor(ABC):
    """Base class for remote accessors.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (Chronometer): The benchmarking timer from the stopuhr library.
        created (bool): True if the datacube already exists in the storage.

    """

    # These class variables (not object properties) are meant to set by a specific dataset accesor
    extent: GeoBox
    temporal_extent: pd.DatetimeIndex | None
    chunk_size: int
    channels: ClassVar[list]
    _channels_meta: ClassVar[dict]
    _channels_encoding: ClassVar[dict]

    def __init__(
        self,
        storage: icechunk.Storage | Path | str,
        create_icechunk_storage: bool = True,
        backend: Literal["threaded", "simple"] = "threaded",
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
            backend (Literal["threaded", "simple"], optional): The backend to use for downloading data.
                Currently, only "threaded" is supported. Defaults to "threaded".

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
        self.stopuhr = Chronometer(logger.debug)

        if backend == "threaded":
            if not _check_python_version(3, 13):
                raise NotImplementedError(
                    "The 'threaded' backend is only fully supported in Python 3.13 and above."
                    " Please consider using the 'simple' backend in a multiprocessing environment"
                    " or upgrade your Python version."
                )
            self.backend = ThreadedBackend(self.repo, self.download_patch)
        elif backend == "simple":
            self.backend = SimpleBackend(self.repo, self.download_patch)
        else:
            raise ValueError(f"Unknown backend {backend}")

        self.post_init()

    def post_init(self):
        """Post init actions. Can be overwritten by the dataset accessor."""
        pass

    def __repr__(self):  # noqa: D105
        return f"{self.title}({_geobox_repr(self.extent)}, {self.channels})"

    def __str__(self):  # noqa: D105
        return self.__repr__()

    @property
    def is_temporal(self) -> bool:
        """Check if the datacube has a temporal dimension.

        Returns:
            bool: True if the datacube has a temporal dimension.

        """
        return self.temporal_extent is not None

    @property
    def created(self) -> bool:
        """Check if the datacube already exists in the storage.

        Returns:
            bool: True if the datacube already exists in the storage.

        """
        return self.backend.created

    def loaded_patches(self) -> list[str]:
        """Get the ids of already (down-)loaded patches.

        Returns:
            list[str]: A list of already loaded patch ids.

        """
        session = self.repo.readonly_session("main")
        zcube = zarr.open(store=session.store, mode="r")
        return zcube.attrs.get("loaded_patches", []).copy()

    def assert_created(self):
        """Assert that the datacube exists in the storage."""
        self.backend.assert_created()

    def assert_temporal_cube(self):
        """Assert that the datacube has a temporal dimension.

        Raises:
            ValueError: If the datacube has no temporal dimension.

        """
        if self.temporal_extent is None:
            msg = f"Datacube {self.title} has no temporal dimension."
            logger.error(msg)
            raise ValueError(msg)

    def open_zarr(self) -> zarr.Group:
        """Open the zarr datacube in read-only mode.

        Returns:
            zarr.Group: The zarr datacube.

        """
        return self.backend.open_zarr()

    def open_xarray(self) -> xr.Dataset:
        """Open the xarray datacube in read-only mode.

        Returns:
            xr.Dataset: The xarray datacube.

        """
        return self.backend.open_xarray()

    def log_benchmark_summary(self):
        """Log the benchmark summary."""
        self.stopuhr.summary()

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
                attrs={"title": self.title, "loaded_patches": []},
            )

            # Expand to temporal dimension if defined
            if self.temporal_extent is not None:
                ds = ds.expand_dims(time=self.temporal_extent)

            # Add metadata
            for name, meta in self._channels_meta.items():
                ds[name].attrs.update(meta)

            # Get the encoding for the coordinates, variables and spatial reference
            coords_encoding = {
                "x": {"chunks": ds.x.shape, **optimize_coord_encoding(ds.x.values, self.extent.resolution.x)},
                "y": {"chunks": ds.y.shape, **optimize_coord_encoding(ds.y.values, self.extent.resolution.y)},
            }
            if self.temporal_extent is not None:
                coords_encoding["time"] = {"chunks": ds.time.shape, **optimize_temporal_encoding(self.temporal_extent)}
            chunks = (
                (1, self.chunk_size, self.chunk_size)
                if self.temporal_extent is not None
                else (self.chunk_size, self.chunk_size)
            )
            var_encoding = {
                name: {
                    "chunks": chunks,
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

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        toi = None
        if "time" in ref.coords and self.temporal_extent is not None:
            toi = ref.get_index("time")
        return self.load(ref.geobox, toi=toi, **kwargs)

    def load(
        self,
        aoi: Geometry | GeoBox,
        toi: TOI = None,
        persist: bool = True,
        create: bool = False,
    ) -> xr.Dataset:
        """Load the data for the given geobox.

        Args:
            aoi (Geometry | GeoBox): The reference geometry to load the data for. If a Geobox is provided,
                it will use the extent of the geobox.
            toi (TOI): The temporal slice to load. Defaults to None.
            persist (bool, optional): If the data should be persisted in memory.
                If not, this will return a Dask backed Dataset. Defaults to True.
            create (bool, optional): Create a new zarr array at defined storage if it not exists.
                This is not recommended, because it can have side effects in a multi-process environment.
                Defaults to False.

        Returns:
            xr.Dataset: The loaded dataset in the same resolution and extent like the geobox.

        """
        if toi is not None:
            self.assert_temporal_cube()

        if isinstance(aoi, GeoBox):
            aoi = aoi.extent

        with self.stopuhr(f"{_geometry_repr(aoi)}: {self.title} tile {'loading' if persist else 'lazy-loading'}"):
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
            aligned_aoi = aoi.to_crs(self.extent.crs)
            with self.stopuhr(f"{_geometry_repr(aoi)}: Procedural download in blocking mode"):
                self.procedural_download(aligned_aoi, toi)

            # Load the datacube and set the spatial_ref since it is set as a coordinate within the zarr format
            session = self.repo.readonly_session("main")
            chunks = None if persist else "auto"
            xrcube = xr.open_zarr(
                session.store,
                mask_and_scale=False,
                chunks=chunks,
                consolidated=False,
            ).set_coords("spatial_ref")

            # Get temporal slice if time is provided
            if toi is not None:
                xrcube = xrcube.sel(time=toi)

            # Get an AOI slice of the datacube
            xrcube_aoi = xrcube.odc.crop(aligned_aoi, apply_mask=False)

            # The following code would load the lazy zarr data from disk into memory
            if persist:
                with self.stopuhr(f"{_geometry_repr(aoi)}: {self.title} AOI loading from disk"):
                    xrcube_aoi = xrcube_aoi.load()
        return xrcube_aoi

    def procedural_download(self, aoi: Geometry | GeoBox, toi: TOI):
        """Download tiles procedurally.

        Warning:
            This method is meant for single-process use, but can (in theory) be used in a multi-process environment.
            However, in a multi-process environment it can happen that multiple processes try to write concurrently,
            which results in a conflict.
            In such cases, the download will be retried until it succeeds or the number of maximum-tries is reached.

        Args:
            aoi (Geometry | GeoBox): The geometry of the aoi to download. If a Geobox is provided,
                it will use the extent of the geobox.
            toi (TOI): The time of interest to download.

        Raises:
            ValueError: If no adjacent tiles are found. This can happen if the geobox is out of the dataset bounds.
            ValueError: If not all downloads were successful.

        """
        if isinstance(aoi, GeoBox):
            aoi = aoi.extent

        adjacent_patches = self.adjacent_patches(aoi, toi)
        # interest-string
        soi = f"{_geometry_repr(aoi)}" + (f" @ {_repr_toi(toi)}" if toi is not None else "")
        if not adjacent_patches:
            logger.error(f"{soi}: No adjacent patches found: {adjacent_patches=}")
            raise ValueError("No adjacent patches found - is the provided aoi and toi correct?")

        loaded_patches = self.loaded_patches()

        new_patches = [patch for patch in adjacent_patches if patch.id not in loaded_patches]

        logger.debug(f"{soi}:  {len(adjacent_patches)=} & {len(loaded_patches)=} -> {len(new_patches)=} to download")
        if not new_patches:
            return

        # This raises Errors if anything goes wrong -> we want to propagate
        self.backend.submit(new_patches)

    @abstractmethod
    def adjacent_patches(self, roi: Geometry | GeoBox | gpd.GeoDataFrame, toi: TOI) -> list[PatchIndex]:
        """Get the adjacent patches for the given geobox.

        Must be implemented by the Accessor.

        Args:
            roi (Geometry | GeoBox | gpd.GeoDataFrame): The reference geometry, geobox or reference geodataframe
            toi (TOI): The time of interest to download.

        Returns:
            list[PatchIndex]: The adjacent patch(-id)s for the given geobox.

        """

    @abstractmethod
    def download_patch(self, idx: PatchIndex) -> xr.Dataset:
        """Download the data for the given patch.

        Must be implemented by the Accessor.

        Args:
            idx (PatchIndex): The reference patch to download the data for.

        Returns:
            xr.Dataset: The downloaded patch data.

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
