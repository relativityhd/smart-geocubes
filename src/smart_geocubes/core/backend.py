"""Write specific backends."""

import abc
import logging
from collections.abc import Callable

import icechunk
import numpy as np
import xarray as xr
import zarr
from zarr.core.sync import sync

from smart_geocubes.core.patches import PatchIndex
from smart_geocubes.core.utils import _log_xcube_stats

logger = logging.getLogger(__name__)


class DownloadBackend(abc.ABC):
    """Base class for download backends."""

    def __init__(self, repo: icechunk.Repository, f: Callable[[PatchIndex], xr.Dataset]):
        """Initialize the ThreadedBackend.

        Args:
            repo (icechunk.Repository): The icechunk repository.
            f (callable[[PatchIndex], xr.Dataset]): A function that takes a PatchIndex and returns an xr.Dataset.
                This should be implemented by the specific source backend.

        """
        self.repo = repo
        self.download_from_source = f

    @property
    def created(self) -> bool:
        """Check if the datacube already exists in the storage.

        Returns:
            bool: True if the datacube already exists in the storage.

        """
        return not sync(self.repo.readonly_session("main").store.is_empty(""))

    def assert_created(self, session: icechunk.Session | None = None):
        """Assert that the datacube exists in the storage.

        Raises:
            FileNotFoundError: If the datacube does not exist.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        if sync(session.store.is_empty("")):
            msg = "Datacube does not exist."
            " Please use the `create` method or pass `create=True` to `load`."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def open_zarr(self, session: icechunk.Session | None = None) -> zarr.Group:
        """Open the zarr datacube in read-only mode.

        Returns:
            zarr.Group: The zarr datacube.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        self.assert_created(session)
        zcube = zarr.open(store=session.store, mode="r")
        return zcube

    def open_xarray(self, session: icechunk.Session | None = None) -> xr.Dataset:
        """Open the xarray datacube in read-only mode.

        Returns:
            xr.Dataset: The xarray datacube.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        self.assert_created(session)
        xcube = xr.open_zarr(session.store, mask_and_scale=False, consolidated=False).set_coords("spatial_ref")
        return xcube

    def loaded_patches(self, session: icechunk.Session | None = None) -> list[str]:
        """Get a list of all loaded patch ids.

        Returns:
            list[str]: A list of all loaded patch ids.

        """
        zcube = self.open_zarr(session)
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])
        return loaded_tiles

    def _get_target_slice(
        self, patch: xr.Dataset, session: icechunk.Session | None = None
    ) -> tuple[slice | list[int], slice, slice] | tuple[slice, slice]:
        xcube = self.open_xarray(session)
        _log_xcube_stats(xcube, "Target xcube")

        spatial_target = xcube.odc.geobox.overlap_roi(patch.odc.geobox)

        if "time" not in xcube.dims:
            logger.debug(f"Geocube is not temporal - writing to {spatial_target=}.")
            return spatial_target

        assert "time" in patch.dims, "Geocube is temporal - patches need a time dimension."
        assert len(patch.time) >= 1, "Patch must have at least one time step."

        extent = xcube.get_index("time").normalize()
        temporal_target = extent.get_indexer(patch.time.values, method="nearest")

        logger.debug(f"Writing to temporal {temporal_target=} and spatial {spatial_target=}.")

        return (temporal_target, *spatial_target)

    def _write_patch_variable(self, zcube: zarr.Group, data: np.ndarray, var: str, target: tuple):
        # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
        # Replace these nan borders with existing data if there is any
        mask = np.isnan(data)
        if np.any(mask):
            existing_data = zcube[var][target]
            data[mask] = existing_data[mask]
        zcube[var][target] = data

    def _download_from_source_with_retries(self, idx: PatchIndex) -> xr.Dataset:
        # Also handles download errors properly
        logger.debug(f"Downloading patch {idx.id}...")
        last_exception: Exception | None = None
        for t in range(5):
            try:
                patch = self.download_from_source(idx)
                break
            except (KeyboardInterrupt, SystemError, SystemExit) as e:
                raise e
            except Exception as e:
                logger.error(f"{idx.id=}: {e=} at download retry {t}/5")
                last_exception = e
        else:
            logger.error(
                f"{idx.id} failed to download after 5 tries. Please check your internet connection and data access."
            )
            raise ValueError(f"{idx.id=}: 5 tries to download the tile failed. ") from last_exception
        patch.attrs["patch_id"] = idx.id
        return patch

    def close(self) -> bool:
        """Close the backend.

        Returns:
            bool: True if the backend was closed successfully, False otherwise.

        """
        # Base implementation does nothing
        return True

    @abc.abstractmethod
    def submit(self, idx: PatchIndex | list[PatchIndex]):
        """Submit a patch download request to the backend.

        Args:
            idx (PatchIndex | list[PatchIndex]): The index or multiple indices of the patch(es) to download.

        """
        pass
