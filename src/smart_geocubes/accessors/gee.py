"""Google Earth Engine Accessor for Smart Geocubes."""

import logging
import warnings
from functools import cached_property

import geopandas as gpd
import pandas as pd
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


def _timetileidx_to_id(timeidx: int, tileidx: tuple[int, int]) -> str:
    return f"{timeidx}-" + "-".join(str(i) for i in tileidx)


def _id_to_timetileidx(tid: str) -> tuple[int, int, int]:
    return tuple(int(i) for i in tid.split("-"))


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

    def adjacent_tiles(self, roi: GeoBox | gpd.GeoDataFrame, time: str | slice | None) -> list[TileWrapper]:
        """Get adjacent tiles from Google Earth Engine.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe
            time (str | slice | None): The reference time or time slice

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        Raises:
            ValueError: If the ROI type is invalid.
            ValueError: If the time is not found in the temporal extent.
            ValueError: If a time slice is provided, but the datacube has no temporal extent.

        """
        if time is not None and self.temporal_extent is None:
            raise ValueError("Temporal extent is not defined for this datacube.")
        if isinstance(roi, gpd.GeoDataFrame):
            adjacent_geometries = (
                gpd.sjoin(self._tile_geometries, roi.to_crs(self.extent.crs.wkt), how="inner", predicate="intersects")
                .reset_index()
                .drop_duplicates(subset="index", keep="first")
                .set_index("index")
            )
            geom_idx = list(adjacent_geometries["idx"])

        elif isinstance(roi, GeoBox):
            geom_idx = list(self._extent_tiles.tiles(roi.extent))
        else:
            raise ValueError("Invalid ROI type.")

        if self.temporal_extent is None:
            return [TileWrapper(_tileidx_to_id(idx), self._extent_tiles[idx]) for idx in geom_idx]

        # Now datacube is temporal
        if time is None:
            return [
                TileWrapper(_timetileidx_to_id(ti, idx), self._extent_tiles[idx])
                for ti in self.temporal_extent.strftime("%Y%m%d%H%M%S").tolist()
                for idx in geom_idx
            ]
        elif isinstance(time, slice):
            idxr = self.temporal_extent.slice_indexer(time.start, time.stop)
            if len(idxr) == 0:
                raise ValueError(f"Time slice {time} not found in temporal extent.")
            time_indices = self.temporal_extent[idxr].strftime("%Y%m%d%H%M%S").tolist()
            return [
                TileWrapper(_timetileidx_to_id(ti, idx), self._extent_tiles[idx])
                for ti in time_indices
                for idx in geom_idx
            ]
        else:
            idxr = self.temporal_extent.get_indexer([time])
            if len(idxr) == 0:
                raise ValueError(f"Time {time} not found in temporal extent.")
            time_index = self.temporal_extent[idxr].strftime("%Y%m%d%H%M%S").tolist()[0]
            return [TileWrapper(_timetileidx_to_id(time_index, idx), self._extent_tiles[idx]) for idx in geom_idx]

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
        ee_col = ee.ImageCollection(self.collection)
        if self.temporal_extent is not None:
            timeidx, _, _ = _id_to_timetileidx(tile.id)
            time = pd.to_datetime(timeidx, format="%Y%m%d%H%M%S")
            ee_col = ee_col.filterDate(time)
        ee_img = ee_col.mosaic().clip(geom)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
            tiledata = xr.open_dataset(
                ee_img,
                engine="ee",
                geometry=geom,
                crs=f"epsg:{self.extent.crs.to_epsg()}",
                scale=self.extent.resolution.x,
            )

        # Do a mosaic if time axis are returned for non-temporal data
        if "time" in tiledata.dims and self.temporal_extent is None:
            tiledata = tiledata.max("time")

        tiledata = tiledata.rename({"lon": "x", "lat": "y"})
        if "time" in tiledata.dims:
            tiledata["time"] = [time]
            tiledata = tiledata.transpose("time", "y", "x")
        else:
            tiledata = tiledata.transpose("y", "x")

        # Download the data
        tiledata_datatype = type(next(iter(tiledata.data_vars.values()))._variable._data)
        logger.debug(f"{tile.id=}: Trigger GEE download ({tiledata_datatype=})")
        tiledata.load()
        tiledata_datatype = type(next(iter(tiledata.data_vars.values()))._variable._data)
        logger.debug(f"{tile.id=}: Finished GEE download ({tiledata_datatype=})")
        logging.getLogger("urllib3.connectionpool").disabled = False

        # Flip y-axis, because convention is x in positive direction and y in negative, but gee use positive for both
        tiledata = tiledata.isel(y=slice(None, None, -1))

        # For some reason xee does not always set the crs
        tiledata = tiledata.odc.assign_crs(self.extent.crs)

        # Recrop the data to the tile, since gee does not always return the exact extent
        tiledata = tiledata.odc.crop(tile.item.extent)

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
        if self.temporal_extent is not None:
            loaded_tiles = [
                {"geometry": tiles[_id_to_timetileidx(tid)[1:]].extent.geom, "id": tid} for tid in loaded_tiles
            ]
        else:
            loaded_tiles = [{"geometry": tiles[_id_to_tileidx(tid)].extent.geom, "id": tid} for tid in loaded_tiles]
        gdf = gpd.GeoDataFrame(loaded_tiles, crs=self.extent.crs.to_wkt())
        return gdf
