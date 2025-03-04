import logging
import queue
import threading
import time
from dataclasses import dataclass

import icechunk
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor, TileWrapper

logger = logging.getLogger(__name__)


class SetQueue(queue.Queue):
    def _init(self, maxsize):
        self.queue: list[TileWrapper] = []

    def _put(self, item: TileWrapper):
        if item.id not in [t.id for t in self.queue]:
            self.queue.append(item)

    def _get(self):
        return self.queue.pop(0)


@dataclass
class Downloader:
    queue: SetQueue

    def loop(self):
        while True:
            try:
                tile = self.queue.get()
            except queue.ShutDown:
                break
            self.download_tile(tile)


class ThreadingMixin(RemoteAccessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._queue = SetQueue()
        self._dthread: threading.Thread | None = None

    def __del__(self):
        self._queue.shutdown()
        with self._lock:
            if self._dthread is not None:
                self._dthread.join(timeout=3)

    def _download_loop(self):
        while True:
            try:
                tile = self._queue.get()
            except queue.ShutDown:
                break

            session = self.repo.writable_session("main")
            zcube = zarr.open(store=session.store, mode="r+")
            loaded_tiles = zcube.attrs.get("loaded_tiles", [])

            if tile.id in loaded_tiles:
                logger.debug(f"Tile {tile.id} already loaded")
                continue

            logger.debug(f"Downloading {tile.id}")
            tick_dstart = time.perf_counter()
            self.download_tile(zcube, tile)
            tick_dend = time.perf_counter()
            logger.debug(f"Downloaded {tile.id} in {tick_dend - tick_dstart:.2f} seconds")
            loaded_tiles.append(tile.id)
            zcube.attrs["loaded_tiles"] = loaded_tiles

            try:
                session.rebase(icechunk.ConflictDetector())
                session.commit(f"Procedurally downloaded {tile.id=} in threading mode")
            # Currently not possible, because attrs will always result in a conflict
            # except icechunk.RebaseFailedError as e:
            #     logger.warning(f"Rebase failed: {e}")
            #     logger.debug("Retrying download")
            #     self._queue.put(tile)
            except icechunk.ConflictError as e:
                logger.warning(f"Icechunk session is out of sync: {e}")
                logger.debug("Retrying download")
                self._queue.put(tile)

    def procedural_download_threading(self, geobox: GeoBox):
        with self._lock:
            if self._dthread is None:
                self._dthread = threading.Thread(target=self._download_loop, daemon=True)
                self._dthread.start()

        adjacent_tiles = self.adjacent_tiles(geobox)
        if not adjacent_tiles:
            logger.error("No adjacent tiles found")
            raise ValueError("No adjacent tiles found - is the provided geobox corrent?")
        logger.debug(f"Found {len(adjacent_tiles)} adjacent tiles: {[t.id for t in adjacent_tiles]}")

        # Wait until all new_items are loaded
        prev_len = None
        while True:
            session = self.repo.readonly_session("main")
            zcube = zarr.open(store=session.store, mode="r")
            loaded_tiles = zcube.attrs.get("loaded_tiles", [])
            new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
            if prev_len is None:
                logger.debug(f"Found {len(new_tiles)} new tiles to download: {[t.id for t in new_tiles]}")
            if not new_tiles:
                break
            if prev_len is not None and prev_len != len(new_tiles):
                logger.debug(f"{len(new_tiles)} new tiles left to download: {[t.id for t in new_tiles]}")
            for tile in new_tiles:
                self._queue.put(tile)
            prev_len = len(new_tiles)
            time.sleep(5)

        # If no more items are in queue, shut it down
        with self._lock:
            if self._queue.empty():
                self._queue.shutdown()
                self._dthread.join()
                self._dthread = None
