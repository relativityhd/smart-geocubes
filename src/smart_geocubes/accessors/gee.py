"""Google Earth Engine Accessor for Smart Geocubes."""

import logging
import warnings
from functools import cached_property

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
    """Accessor for Google Earth Engine data.

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

    collection: str

    @cached_property
    def _tile_geometries(self) -> gpd.GeoDataFrame:
        data = [
            {"idx": idx, "geometry": self._extent_tiles[idx].boundingbox.polygon.geom}
            for idx in self._extent_tiles._all_tiles()
        ]
        return gpd.GeoDataFrame(data, crs=self.extent.crs.to_wkt())

    @cached_property
    def _extent_tiles(self) -> GeoboxTiles:
        return GeoboxTiles(self.extent, (self.chunk_size, self.chunk_size))

    def adjacent_tiles(self, roi: GeoBox | gpd.GeoDataFrame) -> list[TileWrapper]:
        """Get adjacent tiles from Google Earth Engine.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        """
        if isinstance(roi, gpd.GeoDataFrame):
            adjacent_geometries = (
                gpd.sjoin(self._tile_geometries, roi.to_crs(self.extent.crs.wkt), how="inner", predicate="intersects")
                .reset_index()
                .drop_duplicates(subset="index", keep="first")
                .set_index("index")
            )
            return [TileWrapper(_tileidx_to_id(idx), self._extent_tiles[idx]) for idx in adjacent_geometries["idx"]]

        elif isinstance(roi, GeoBox):
            return [
                TileWrapper(_tileidx_to_id(idx), self._extent_tiles[idx])
                for idx in self._extent_tiles.tiles(roi.extent)
            ]

    def download_tile(self, tile: TileWrapper) -> xr.Dataset:
        """Download a tile from Google Earth Engine.

        Args:
            tile (TileWrapper): The tile to download.

        Returns:
            xr.Dataset: The downloaded tile data.

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
        geom = ee.Geometry.Rectangle(tile.item.geographic_extent.boundingbox)
        ee_img = ee.ImageCollection(self.collection).mosaic().clip(geom)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
            tiledata = xr.open_dataset(
                ee_img,
                engine="ee",
                geometry=geom,
                crs=f"epsg:{self.extent.crs.to_epsg()}",
                scale=self.extent.resolution.x,
            )

        # TODO: Allow for multi-temporal datacubes and lat/lon coordinates
        tiledata = tiledata.max("time").rename({"lon": "x", "lat": "y"}).transpose("y", "x")

        # Download the data
        tiledata.load()
        logging.getLogger("urllib3.connectionpool").disabled = False

        # Flip y-axis, because convention is x in positive direction and y in negative, but gee use positive for both
        tiledata = tiledata.isel(y=slice(None, None, -1))

        # For some reason xee does not always set the crs
        tiledata = tiledata.odc.assign_crs(self.extent.crs)

        # Recrop the data to the tile, since gee does not always return the exact extent
        tiledata = tiledata.odc.crop(tile.item.extent)

        # Save original min-max values for each band for clipping later
        clip_values = {
            band: (tiledata[band].min().values.item(), tiledata[band].max().values.item())
            for band in tiledata.data_vars
        }

        # Interpolate missing values (there are very few, so we actually can interpolate them)
        tiledata.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        for band in tiledata.data_vars:
            tiledata[band] = tiledata[band].rio.write_nodata(np.nan).rio.interpolate_na()

        # Convert to uint8
        for band in tiledata.data_vars:
            band_min, band_max = clip_values[band]
            tiledata[band] = (
                tiledata[band].clip(band_min, band_max, keep_attrs=True).astype("uint8").rio.write_nodata(None)
            )

        return tiledata

    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles.

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
