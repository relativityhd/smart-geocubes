"""Write specific backends."""

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, wait
from queue import Queue
from threading import Thread

import icechunk
import xarray as xr
import zarr

from smart_geocubes.core.backend import DownloadBackend
from smart_geocubes.core.patches import PatchIndex

logger = logging.getLogger(__name__)


class ThreadedBackend(DownloadBackend):
    """Threaded backend for downloading patches."""

    def __init__(self, repo: icechunk.Repository, f: Callable[[PatchIndex], xr.Dataset], concurrent_downloads: int = 2):
        """Initialize the ThreadedBackend.

        Args:
            repo (icechunk.Repository): The icechunk repository.
            f (callable[[PatchIndex], xr.Dataset]): A function that takes a PatchIndex and returns an xr.Dataset.
                This should be implemented by the specific source backend.
            concurrent_downloads (int, optional): The number of concurrent downloads. Defaults to 2.

        """
        super().__init__(repo, f)

        self.download_pool = ThreadPoolExecutor(max_workers=concurrent_downloads)

        # The writer allows for asynchronous download and writing
        # The writing_pool allows for concurrent writes within the writer thread
        # The write_queue is blocking with a maxsize of 2 to prevent too many successful downloads
        # from piling up in memory
        self.writer = Thread(target=self._writer, daemon=True, name="WriterThread")
        self.write_queue: Queue[xr.Dataset] = Queue(maxsize=2)
        self.writing_pool = ThreadPoolExecutor(max_workers=4)
        self.writer.start()

    def close(self) -> bool:
        """Close the backend.

        Returns:
            bool: True if the backend was closed successfully, False otherwise.

        """
        logger.debug("Closing Backend...")

        self.download_pool.shutdown(wait=False, cancel_futures=True)
        logger.debug("Download pool shut down.")

        self.write_queue.shutdown(immediate=True)
        logger.debug("Write queue shut down.")

        self.writer.join()
        logger.debug("Writer thread joined.")

        self.writing_pool.shutdown(wait=True, cancel_futures=False)
        logger.debug("Writing pool shut down.")

        logger.info("Backend closed.")
        return True

    def _writer(self):
        while True:
            # This basically ensures that only a single patch is written at a time
            # The concurrency happens inside the write_patch function, where the variables are written in parallel
            patch = self.write_queue.get()
            if patch is None:
                break

            patch_id = patch.attrs.get("patch_id", "unknown")

            logger.debug(f"Writing patch {patch_id}...")
            for t in range(100):
                try:
                    self._write_patch(patch)
                    break
                except icechunk.ConflictError as conflict_error:
                    logger.debug(f"{patch_id=}: {conflict_error=} at write retry {t}/100")
            else:
                logger.error(
                    f"{patch_id=}: 100 tries to write the tile failed. "
                    "Please check if the datacube is already created and not empty."
                )
                raise ValueError(f"{patch_id=}: 100 tries to write the tile failed.")

            self.write_queue.task_done()

    def _write_patch(self, patch: xr.Dataset):
        """Write a downloaded patch to the repository.

        Args:
            patch (xr.Dataset): The downloaded patch.

        Raises:
            RuntimeError: If writing the patch fails.

        """
        patch_id = str(patch.attrs["patch_id"])

        session = self.repo.writable_session("main")
        zcube = zarr.open(session.store, mode="r+")

        loaded_patches = self.loaded_patches(session)
        if patch_id in loaded_patches:
            logger.debug(f"Patch {patch_id} already written, skipping.")
            return

        target = self._get_target_slice(patch, session)

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

    def _download_in_pool(self, idx: PatchIndex):
        # Wrapping this to be able to pass it into the queue, which is blocking
        # This prevents too many successful downloads from piling up in memory
        # because the queue has a maxsize of 2
        patch = self._download_from_source_with_retries(idx)
        self.write_queue.put(patch)

    def submit(self, idx: PatchIndex | list[PatchIndex]):
        """Submit a patch download request to the backend.

        Args:
            idx (PatchIndex | list[PatchIndex]): The index or multiple indices of the patch(es) to download.

        Raises:
            RuntimeError: If the writer thread is not alive or if downloading failed for any patches.

        """
        if isinstance(idx, PatchIndex):
            idx = [idx]

        # Check if the writer thread is still alive
        if not self.writer.is_alive():
            raise RuntimeError("Writer thread is not alive. This happens if the writer thread crashes.")

        futures = [self.download_pool.submit(self._download_in_pool, i) for i in idx]

        _, failed = wait(futures)
        if len(failed) > 0:
            raise RuntimeError(f"Downloading failed for {len(failed)} patches.")

        # Check if the queue is still alive
        if self.write_queue.is_shutdown:
            raise RuntimeError(
                "Write queue is not alive. This happens if the writer thread crashes or the backend is closed."
            )
        self.write_queue.all_tasks_done.wait()
