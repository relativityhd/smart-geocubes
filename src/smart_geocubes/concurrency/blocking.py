import logging
import time

import icechunk
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor

logger = logging.getLogger(__name__)


class BlockingMixin(RemoteAccessor):
    def procedural_download_blocking(self, geobox: GeoBox, tries: int = 5):
        if tries == 0:
            logger.warning("No tries left, skipping download")
            raise ValueError("Unable to commit, no tries left.")

        tick_fstart = time.perf_counter()
        adjacent_tiles = self.adjacent_tiles(geobox)
        if not adjacent_tiles:
            logger.error("No adjacent tiles found")
            raise ValueError("No adjacent tiles found - is the provided geobox corrent?")
        logger.debug(f"Found {len(adjacent_tiles)=} adjacent tiles")

        session = self.repo.writable_session("main")
        zcube = zarr.open(store=session.store, mode="r+")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])
        logger.debug(f"{len(loaded_tiles)=} tiles already loaded")
        new_tiles = [tile for tile in adjacent_tiles if tile.id not in loaded_tiles]
        logger.debug(f"Found {len(new_tiles)} new tiles to download")
        if not new_tiles:
            return

        for tile in new_tiles:
            logger.debug(f"Downloading {tile.id}")
            tick_dstart = time.perf_counter()
            self.download_tile(zcube, tile)
            tick_dend = time.perf_counter()
            logger.debug(f"Downloaded {tile.id} in {tick_dend - tick_dstart:.2f} seconds")
            loaded_tiles.append(tile.id)
            zcube.attrs["loaded_tiles"] = loaded_tiles

        try:
            session.rebase(icechunk.ConflictDetector())
            session.commit(f"Procedurally downloaded tiles {[tile.id for tile in new_tiles]} in blocking mode")
        # Currently not possible, because attrs will always result in a conflict
        # except icechunk.RebaseFailedError as e:
        #     logger.warning(f"Rebase failed: {e}")
        #     logger.debug(f"Retrying download with {tries - 1} tries left")
        #     self.procedural_download_blocking(geobox, tries=tries - 1)
        except icechunk.ConflictError as e:
            logger.warning(f"Icechunk session is after rebase still out of sync: {e}")
            logger.debug(f"Retrying download with {tries - 1} tries left")
            self.procedural_download_blocking(geobox, tries=tries - 1)

        tick_fend = time.perf_counter()
        logger.debug(f"Downloaded {len(new_tiles)} tiles in {tick_fend - tick_fstart:.2f} seconds")
