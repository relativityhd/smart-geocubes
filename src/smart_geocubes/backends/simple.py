"""Write specific backends."""

import logging

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

        for var in patch.data_vars:
            self._write_patch_variable(zcube, patch[var].data, var, target, patch_id)

        loaded_patches.append(patch_id)
        zcube.attrs["loaded_patches"] = loaded_patches
        session.commit(f"Write patch {patch_id}")
        logger.info(f"Patch {patch_id} written successfully.")

    def submit(self, idx: PatchIndex | list[PatchIndex]):
        """Submit a patch download request to the backend.

        Args:
            idx (PatchIndex | list[PatchIndex]): The index or multiple indices of the patch(es) to download.

        """
        if isinstance(idx, PatchIndex):
            idx = [idx]
        for i in idx:
            self._log_event("start_download", i.id)
            patch = self._download_from_source_with_retries(i)
            self._log_event("end_download", i.id)
            self._log_event("start_write", i.id)
            self._write_patch(patch)
            self._log_event("end_write", i.id)
