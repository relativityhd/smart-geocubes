"""Google Earth Engine Accessor for Smart Geocubes."""

import logging
import warnings

import geopandas as gpd
import numpy as np
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox, GeoboxTiles

from smart_geocubes.accessors.base import RemoteAccessor, TileWrapper

logger = logging.getLogger(__name__)

EE_WARN_MSG = "Unable to retrieve 'system:time_start' values from an ImageCollection due to: No 'system:time_start' values found in the 'ImageCollection'."  # noqa: E501


def _tileidx_to_id(tileidx: tuple[int, int]) -> str:
    return "-".join(str(i) for i in tileidx)


def _id_to_tileidx(tileid: str) -> tuple[int, int]:
    return tuple(int(i) for i in tileid.split("-"))


class GEEAccessor(RemoteAccessor):
    """Accessor for Google Earth Engine data."""

    collection: str

    def adjacent_tiles(self, geobox: GeoBox) -> list[TileWrapper]:
        """Get adjacent tiles from Google Earth Engine.

        Args:
            geobox (GeoBox): The geobox for which to get adjacent tiles.

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        """
        tiles = GeoboxTiles(self.extent, (self.chunk_size, self.chunk_size))
        return [TileWrapper(_tileidx_to_id(idx), tiles[idx]) for idx in tiles.tiles(geobox.extent)]

    def download_tile(self, zcube: zarr.Group, geobox_tile: TileWrapper):
        """Download a tile from Google Earth Engine.

        Args:
            zcube (zarr.Group): The zarr datacube to download the tile to.
            geobox_tile (TileWrapper): The tile to download.

        """
        import ee
        import rioxarray  # noqa: F401
        import xee  # noqa: F401

        # Note: This is a little bit weird: First we create an own grid which overlaps to the chunks
        # of the zarr array. Then we create a mosaic of the data and clip it to a single chunk.
        # We could load the images from the collection directly instead of creating a mosaic.
        # However, this would require more testing and probably results a lot of manual computation
        # of slices etc. like in the stac variant. So for now, we just use the mosaic.
        logging.getLogger("urllib3.connectionpool").disabled = True
        geom = ee.Geometry.Rectangle(geobox_tile.item.geographic_extent.boundingbox)
        ee_img = ee.ImageCollection(self.collection).mosaic().clip(geom)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
            tile = xr.open_dataset(
                ee_img,
                engine="ee",
                geometry=geom,
                crs=f"epsg:{self.extent.crs.to_epsg()}",
                scale=self.extent.resolution.x,
            )

        # TODO: Allow for multi-temporal datacubes and lat/lon coordinates
        tile = tile.max("time").rename({"lon": "x", "lat": "y"}).transpose("y", "x")

        # Download the data
        tile.load()
        logging.getLogger("urllib3.connectionpool").disabled = False

        # Flip y-axis, because convention is x in positive direction and y in negative, but gee use positive for both
        tile = tile.isel(y=slice(None, None, -1))

        # For some reason xee does not always set the crs
        tile = tile.odc.assign_crs(self.extent.crs)

        # Recrop the data to the geobox_tile, since gee does not always return the exact extent
        tile = tile.odc.crop(geobox_tile.item.extent)

        # Save original min-max values for each band for clipping later
        clip_values = {
            band: (tile[band].min().values.item(), tile[band].max().values.item()) for band in tile.data_vars
        }

        # Interpolate missing values (there are very few, so we actually can interpolate them)
        tile.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        for band in tile.data_vars:
            tile[band] = tile[band].rio.write_nodata(np.nan).rio.interpolate_na()

        # Convert to uint8
        for band in tile.data_vars:
            band_min, band_max = clip_values[band]
            tile[band] = tile[band].clip(band_min, band_max, keep_attrs=True).astype("uint8").rio.write_nodata(None)

        # Get the slice of the datacube where the tile will be written
        zgeobox = self.geobox
        logger.debug(
            f"{geobox_tile.id=}: {tile.sizes=} {tile.x[0].item()=} {tile.y[0].item()=}"
            f" {zcube['x'][0]=} {zcube['y'][0]=}"
        )
        target_slice = zgeobox.overlap_roi(tile.odc.geobox)

        logger.debug(f"tile.id={geobox_tile.id}: Writing to {target_slice=}")

        for channel in self.channels:
            raw_data = tile[channel].values
            zcube[channel][target_slice] = raw_data

    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage where the datacube is located.

        Returns:
            gpd.GeoDataFrame: Tiles from odc.geo.GeoboxTiles. None if datacube is empty.

        """
        import geopandas as gpd

        if not self.created:
            return None

        session = self.repo.readonly_session("main")
        zcube = zarr.open(session.store, mode="r")
        loaded_tiles = zcube.attrs.get("loaded_tiles", [])

        if len(loaded_tiles) == 0:
            return None

        tiles = GeoboxTiles(self.extent, (self.chunk_size, self.chunk_size))
        loaded_tiles = [{"geometry": tiles[_id_to_tileidx(tid)].extent.geom, "id": tid} for tid in loaded_tiles]
        gdf = gpd.GeoDataFrame(loaded_tiles, crs=self.extent.crs.to_wkt())
        return gdf
