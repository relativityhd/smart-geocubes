"""STAC Accessor for Smart Geocubes."""

import logging

import geopandas as gpd
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor, TileWrapper

logger = logging.getLogger(__name__)


def correct_bounds(tile: xr.Dataset, zgeobox: GeoBox) -> xr.Dataset:
    """Correct the bounds of a tile to fit within a GeoBox.

    Args:
        tile (xr.Dataset): The tile to correct.
        zgeobox (GeoBox): The GeoBox to correct the tile to.

    Raises:
        ValueError: If the tile is out of the geobox's bounds.

    Returns:
        xr.Dataset: The corrected tile.

    """
    yslice, xslice = tile.odc.geobox.overlap_roi(zgeobox)
    yslice_is_valid = yslice.start >= 0 and yslice.start < yslice.stop and yslice.stop <= tile.sizes["y"]
    xslice_is_valid = xslice.start >= 0 and xslice.start < xslice.stop and xslice.stop <= tile.sizes["x"]
    if not yslice_is_valid or not xslice_is_valid:
        logger.error(f"Tile is out of bounds! {yslice=} {xslice=} {tile.sizes=} {zgeobox=}")
        raise ValueError("Tile is out of bounds!")
    if yslice.start != 0 or xslice.start != 0 or yslice.stop != tile.sizes["y"] or xslice.stop != tile.sizes["x"]:
        logger.warning(
            f"Correcting tile bounds. This is an indicator that the datacube extent is to narrow."
            f" This will crop the tile to fit the datacube. {yslice=} {xslice=} {tile.sizes=} {zgeobox=}"
        )
        tile = tile.isel(x=xslice, y=yslice)
    return tile


class STACAccessor(RemoteAccessor):
    """Accessor for STAC data.

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

    stac_api_url: str
    collection: str

    def adjacent_tiles(self, roi: GeoBox | gpd.GeoDataFrame) -> list[TileWrapper]:
        """Get adjacent tiles from a STAC API.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        """
        import pystac_client

        catalog = pystac_client.Client.open(self.stac_api_url)
        geom = roi if isinstance(roi, gpd.GeoDataFrame) else roi.to_crs("EPSG:4326").extent.geom
        search = catalog.search(collections=[self.collection], intersects=geom)
        items = list(search.items())
        return [TileWrapper(item.id, item) for item in items]

    def download_tile(self, tile: TileWrapper) -> xr.Dataset:
        """Download a tile from a STAC API and write it to a zarr datacube.

        Args:
            tile (TileWrapper): The tile to download and write.

        Returns:
            xr.Dataset: The downloaded tile data.

        """
        from odc.stac import stac_load

        tiledata = stac_load([tile.item], bands=self.channels, chunks=None, progress=None)

        # TODO: Allow for multi-temporal datacubes
        tiledata = tiledata.max("time")

        return tiledata

    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles.

        Returns:
            gpd.GeoDataFrame: Tile info from pystac. None if datacube is empty.


        """
        import geopandas as gpd
        import pystac_client

        if not self.created:
            return None

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
