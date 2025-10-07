"""Write specific backends."""

import logging
from concurrent.futures import wait

import xarray as xr
import zarr

from smart_geocubes.core.backend import DownloadBackend
from smart_geocubes.core.patches import PatchIndex

logger = logging.getLogger(__name__)


class SimpleBackend(DownloadBackend):
    """Simple, blocking backend for downloading patches."""

    def _write_patch(self, patch: xr.Dataset):
        patch_id = str(patch.attrs["patch_id"])

        session = self.repo.writable_session("main")
        zcube = zarr.open(session.store, mode="r+")

        loaded_patches = self.loaded_patches(session)
        if patch_id in loaded_patches:
            logger.debug(f"Patch {patch_id} already written, skipping.")
            return

        target = self._get_target_slice(patch)

        futures = {
            self.writing_pool.submit(self._write_patch_variable, zcube, patch[var].data, var, target): var
            for var in patch.data_vars
        }
        _, failed = wait(futures)
        if len(failed) > 0:
            logger.error(f"Writing patch {patch_id} failed for variables {[futures[f] for f in failed]}.")
            raise RuntimeError(f"Writing patch {patch_id} failed.")

        loaded_patches.append(patch_id)
        zcube.attrs["loaded_patches"] = loaded_patches
        session.commit(f"Write patch {patch_id}")
        logger.info(f"Patch {patch_id} written successfully.")

        # Update session after change
        self.session = self.repo.readonly_session("main")

    def submit(self, idx: PatchIndex | list[PatchIndex]):
        """Submit a patch download request to the backend.

        Args:
            idx (PatchIndex | list[PatchIndex]): The index or multiple indices of the patch(es) to download.

        """
        if isinstance(idx, PatchIndex):
            idx = [idx]
        for i in idx:
            patch = self._download_from_source_with_retries(i)
            self._write_patch(patch)
