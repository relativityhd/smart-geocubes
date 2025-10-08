"""STAC Accessor for Smart Geocubes."""

import logging
from typing import TYPE_CHECKING

import geopandas as gpd
import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

from smart_geocubes.core import TOI, PatchIndex, RemoteAccessor, extract_toi_range

if TYPE_CHECKING:
    from pystac import Item

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
    # TODO: do the same for time dimension
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

    def adjacent_patches(self, roi: GeoBox | gpd.GeoDataFrame, toi: TOI | None) -> list[PatchIndex["Item"]]:
        """Get the adjacent patches for the given geobox.

        Must be implemented by the Accessor.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe
            toi (TOI): The time of interest to download.

        Returns:
            list[PatchIndex[Item]]: The adjacent patch(-id)s for the given geobox.

        """
        import pystac_client

        if self.is_temporal:
            toi = extract_toi_range(self.temporal_extent, toi)

        catalog = pystac_client.Client.open(self.stac_api_url)
        geom = roi if isinstance(roi, gpd.GeoDataFrame) else roi.to_crs("EPSG:4326").extent.geom
        search = catalog.search(collections=[self.collection], intersects=geom, datetime=toi)
        items = list(search.items())

        patch_idxs = []
        for item in items:
            geom = Geometry(item.geometry, crs="EPSG:4326")
            if self.is_temporal:
                if item.datetime is not None:
                    idx = PatchIndex(item.id, geom, item.datetime, item)
                else:
                    idx = PatchIndex(
                        item.id, geom, (item.common_metadata.start_datetime, item.common_metadata.end_datetime), item
                    )
            else:
                idx = PatchIndex(item.id, geom, None, item)
            patch_idxs.append(idx)
        return patch_idxs

    def download_patch(self, idx: PatchIndex["Item"]) -> xr.Dataset:
        """Download the data for the given patch.

        Must be implemented by the Accessor.

        Args:
            idx (PatchIndex[Item]): The reference patch to download the data for.

        Returns:
            xr.Dataset: The downloaded patch data.

        """
        from odc.stac import stac_load

        patch = stac_load([idx.item], bands=self.channels, chunks=None, progress=None)

        # Do a mosaic if multiple items are returned for non-temporal data
        if "time" in patch.dims and self.temporal_extent is None:
            patch = patch.max("time")

        return patch

    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles.

        Returns:
            gpd.GeoDataFrame: Tile info from pystac. None if datacube is empty.

        """
        import geopandas as gpd
        import pystac_client

        if not self.created:
            return None

        loaded_patches = self.loaded_patches()

        if len(loaded_patches) == 0:
            return None

        catalog = pystac_client.Client.open(self.stac_api_url)
        search = catalog.search(collections=[self.collection], ids=loaded_patches)
        stac_json = search.item_collection_as_dict()

        gdf = gpd.GeoDataFrame.from_features(stac_json, "epsg:4326")
        return gdf
