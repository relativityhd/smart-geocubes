"""STAC Accessor for Smart Geocubes."""

import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor

if TYPE_CHECKING:
    try:
        import geopandas as gpd
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class STACAccessor(RemoteAccessor):
    """Accessor for STAC data."""

    stac_api_url: str
    collection: str

    def procedural_download(self, geobox: GeoBox, progress_tracker: Any | None = None):
        """Download the data via STAC for the given geobox.

        Args:
            geobox (GeoBox): The reference geobox to download the data for.
            progress_tracker (Any | None, optional): A tqdm or rich progress bar / function to it.

        """
        import pystac_client
        from odc.stac import stac_load

        tick_fstart = time.perf_counter()

        cube = xr.open_zarr(self.storage, mask_and_scale=False)

        catalog = pystac_client.Client.open(self.stac_api_url)
        search = catalog.search(collections=[self.collection], intersects=geobox.to_crs("EPSG:4326").extent.geom)
        items = list(search.items())

        # Check if zarr data already contains the data via the attrs
        loaded_tiles: list[str] = cube.attrs.get("loaded_tiles", []).copy()

        # Collect all tiles which should be downloaded
        new_tiles = [item for item in items if item.id not in loaded_tiles]

        if not len(new_tiles):
            logger.debug("No new tiles to download")
            return
        logger.debug(f"Found {len(new_tiles)} new tiles: {[tile.id for tile in new_tiles]}")

        # Downlaod without dask
        # We load each item one-by-one to reduce the memory footprint
        za = zarr.open(self.storage)
        for item in new_tiles:
            tick_downloads = time.perf_counter()
            tile = stac_load([item], bands=self.channels, chunks=None, progress=progress_tracker)

            # TODO: Allow for multi-temporal datacubes
            tile = tile.max("time")

            # Get the slice of the datacube where the tile will be written
            x_start_idx = int((tile.x[0] - cube.x[0]) // tile.x.attrs["resolution"])
            y_start_idx = int((tile.y[0] - cube.y[0]) // tile.y.attrs["resolution"])
            target_slice = {
                "x": slice(x_start_idx, x_start_idx + tile.sizes["x"]),
                "y": slice(y_start_idx, y_start_idx + tile.sizes["y"]),
            }
            target_slice = tuple(target_slice.values())

            for channel in self.channels:
                raw_data = tile[channel].values
                # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
                # Replace these nan borders with existing data if there is any
                raw_data = np.where(~np.isnan(raw_data), raw_data, za[channel][target_slice])
                za[channel][target_slice] = raw_data
            loaded_tiles.append(item.id)
            za.attrs["loaded_tiles"] = loaded_tiles
            # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
            zarr.consolidate_metadata(self.storage)
            tick_downloade = time.perf_counter()
            logger.debug(f"Downloaded and written {item.id=} to datacube in {tick_downloade - tick_downloads:.2f}s")

        tick_fend = time.perf_counter()
        logger.info(f"Procedural download of {len(new_tiles)} tiles completed in {tick_fend - tick_fstart:.2f} seconds")

    def current_state(self) -> "gpd.GeoDataFrame | None":
        """Get info about currently stored tiles.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage where the datacube is located.

        Returns:
            gpd.GeoDataFrame: Tile info from pystac. None if datacube is empty.


        """
        import geopandas as gpd
        import pystac_client

        za = zarr.open(self.storage, mode="r")
        loaded_tiles = za.attrs["loaded_tiles"]

        if len(loaded_tiles) == 0:
            return None

        catalog = pystac_client.Client.open(self.stac_api_url)
        search = catalog.search(collections=[self.collection], ids=loaded_tiles)
        stac_json = search.item_collection_as_dict()

        gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")
        return gdf
