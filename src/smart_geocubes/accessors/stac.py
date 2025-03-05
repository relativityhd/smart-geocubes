"""STAC Accessor for Smart Geocubes."""

import logging
from typing import TYPE_CHECKING

import numpy as np
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor, TileWrapper
from smart_geocubes.storage import TargetSlice

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

    def adjacent_tiles(self, geobox: GeoBox) -> list[TileWrapper]:
        """Get adjacent tiles from a STAC API.

        Args:
            geobox (GeoBox): The geobox for which to get adjacent tiles.

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        """
        import pystac_client

        catalog = pystac_client.Client.open(self.stac_api_url)
        search = catalog.search(collections=[self.collection], intersects=geobox.to_crs("EPSG:4326").extent.geom)
        items = list(search.items())
        return [TileWrapper(item.id, item) for item in items]

    def download_tile(self, zcube: zarr.Array, stac_tile: TileWrapper):
        """Download a tile from a STAC API and write it to a zarr datacube.

        Args:
            zcube (zarr.Array): The zarr datacube to write the tile to.
            stac_tile (TileWrapper): The tile to download and write.

        """
        from odc.stac import stac_load

        tile = stac_load([stac_tile.item], bands=self.channels, chunks=None, progress=None)

        # TODO: Allow for multi-temporal datacubes
        tile = tile.max("time")

        # Get the slice of the datacube where the tile will be written
        x_start_idx = int((tile.x[0] - zcube["x"][0]) // tile.x.attrs["resolution"])
        y_start_idx = int((tile.y[0] - zcube["y"][0]) // tile.y.attrs["resolution"])
        target_slice = TargetSlice(
            x=slice(x_start_idx, x_start_idx + tile.sizes["x"]),
            y=slice(y_start_idx, y_start_idx + tile.sizes["y"]),
        )
        logger.debug(f"Writing {stac_tile.id=} to {target_slice=}")

        for channel in self.channels:
            raw_data = tile[channel].values
            # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
            # Replace these nan borders with existing data if there is any
            raw_data = np.where(~np.isnan(raw_data), raw_data, zcube[channel][target_slice])
            zcube[channel][target_slice] = raw_data

    def current_state(self) -> "gpd.GeoDataFrame | None":
        """Get info about currently stored tiles.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage where the datacube is located.

        Returns:
            gpd.GeoDataFrame: Tile info from pystac. None if datacube is empty.


        """
        import geopandas as gpd
        import pystac_client

        session = self.repo.readonly_session("main")
        zcube = zarr.open(session.store, mode="r")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])

        if len(loaded_tiles) == 0:
            return None

        catalog = pystac_client.Client.open(self.stac_api_url)
        search = catalog.search(collections=[self.collection], ids=loaded_tiles)
        stac_json = search.item_collection_as_dict()

        gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")
        return gdf
