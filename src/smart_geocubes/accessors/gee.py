"""Google Earth Engine Accessor for Smart Geocubes."""

import logging
import warnings
from dataclasses import dataclass
from functools import cached_property

import geopandas as gpd
import pandas as pd
import xarray as xr
from odc.geo.geobox import GeoBox, GeoboxTiles

from smart_geocubes.core import TOI, PatchIndex, RemoteAccessor, normalize_toi

logger = logging.getLogger(__name__)

EE_WARN_MSG = "Unable to retrieve 'system:time_start' values from an ImageCollection due to: No 'system:time_start' values found in the 'ImageCollection'."  # noqa: E501


@dataclass
class Item:
    """Simple wrapper over what accessor returns as item."""

    geobox: GeoBox
    time: pd.Timestamp | None = None


class GEEMosaicAccessor(RemoteAccessor):
    """Accessor for Google Earth Engine data using mosaics.

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

    def _stringify_index(self, spatial_idx: tuple[int, int], time_idx: int | None = None) -> str:
        s = "-".join(str(i) for i in spatial_idx)
        if time_idx is not None:
            s = f"{time_idx}-{s}"
        return s

    def _parse_index(self, idx: str) -> tuple[tuple[int, int], int | None]:
        # Returns spatial_idx, temporal_idx
        parts = idx.split("-")
        if len(parts) == 2:
            assert not self.is_temporal, "Non-temporal index provided for temporal datacube."
            return (int(parts[0]), int(parts[1])), None
        elif len(parts) == 3:
            assert self.is_temporal, "Temporal index provided for non-temporal datacube."
            return (int(parts[1]), int(parts[2])), int(parts[0])
        else:
            raise ValueError(f"Invalid index string: {idx}")

    def adjacent_patches(self, roi: GeoBox | gpd.GeoDataFrame, toi: TOI) -> list[PatchIndex[Item]]:
        """Get the adjacent patches for the given geobox.

        Must be implemented by the Accessor.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe
            toi (TOI): The time of interest to download.

        Returns:
            list[PatchIndex[Item]]: The adjacent patch(-id)s for the given geobox.

        Raises:
            ValueError: If the ROI type is invalid.
            ValueError: If the datacube is not temporal, but a time of interest is provided.

        """
        if toi is not None and not self.is_temporal:
            raise ValueError("Datacube is not temporal, but time of interest is provided.")

        if isinstance(roi, gpd.GeoDataFrame):
            adjacent_geometries = (
                gpd.sjoin(self._tile_geometries, roi.to_crs(self.extent.crs.wkt), how="inner", predicate="intersects")
                .reset_index()
                .drop_duplicates(subset="index", keep="first")
                .set_index("index")
            )
            spatial_idxs: list[tuple[int, int]] = list(adjacent_geometries["idx"])
        elif isinstance(roi, GeoBox):
            spatial_idxs: list[tuple[int, int]] = list(self._extent_tiles.tiles(roi.extent))
        else:
            raise ValueError("Invalid ROI type.")

        if not self.is_temporal:
            return [
                PatchIndex(
                    self._stringify_index(spatial_idx),
                    self._extent_tiles[spatial_idx].geographic_extent,
                    None,
                    Item(self._extent_tiles[spatial_idx], None),
                )
                for spatial_idx in spatial_idxs
            ]

        # Now datacube is temporal
        toi = normalize_toi(self.temporal_extent, toi)
        patch_idxs = []
        for time in toi:
            time_idx = self.temporal_extent.get_loc(time)
            assert isinstance(time_idx, int), "Non-Unique temporal extents are not supported!"
            for spatial_idx in spatial_idxs:
                patch_idxs.append(
                    PatchIndex(
                        self._stringify_index(spatial_idx, time_idx),
                        self._extent_tiles[spatial_idx].geographic_extent,
                        time,
                        Item(self._extent_tiles[spatial_idx], time),
                    )
                )
        return patch_idxs

    def download_patch(self, idx: PatchIndex[Item]) -> xr.Dataset:
        """Download the data for the given patch.

        Must be implemented by the Accessor.

        Args:
            idx (PatchIndex[Item]): The reference patch to download the data for.

        Returns:
            xr.Dataset: The downloaded patch data.

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

        ee_col = ee.ImageCollection(self.collection)
        if self.is_temporal:
            ee_col = ee_col.filterDate(idx.item.time)
        geom = ee.Geometry.Rectangle(idx.item.geobox.geographic_extent.boundingbox)
        ee_img = ee_col.mosaic().clip(geom)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=EE_WARN_MSG)
            patch = xr.open_dataset(
                ee_img,
                engine="ee",
                geometry=geom,
                crs=f"epsg:{self.extent.crs.to_epsg()}",
                scale=self.extent.resolution.x,
            )

        # Do a mosaic if time axis are returned for non-temporal data
        if "time" in patch.dims and not self.is_temporal:
            patch = patch.max("time")

        patch = patch.rename({"lon": "x", "lat": "y"})
        if "time" in patch.dims:
            patch["time"] = [idx.item.time]
            patch = patch.transpose("time", "y", "x")
        else:
            patch = patch.transpose("y", "x")

        # Download the data
        logger.debug(f"{idx.id=}: Trigger GEE download)")
        patch.load()
        logger.debug(f"{idx.id=}: Finished GEE download")
        logging.getLogger("urllib3.connectionpool").disabled = False

        # Flip y-axis, because convention is x in positive direction and y in negative, but gee use positive for both
        patch = patch.isel(y=slice(None, None, -1))

        # For some reason xee does not always set the crs
        patch = patch.odc.assign_crs(self.extent.crs)

        # Recrop the data to the tile, since gee does not always return the exact extent
        patch = patch.odc.crop(idx.item.geobox.extent)

        return patch

    def current_state(self) -> gpd.GeoDataFrame | None:
        """Get info about currently stored tiles.

        Returns:
            gpd.GeoDataFrame: Tiles from odc.geo.GeoboxTiles. None if datacube is empty.

        """
        import geopandas as gpd

        if not self.created:
            return None

        loaded_patches = self.loaded_patches()

        if len(loaded_patches) == 0:
            return None

        patch_infos = []
        for pid in loaded_patches:
            spatial_idx, temporal_idx = self._parse_index(pid)
            geometry = self._extent_tiles[spatial_idx].extent.geom
            if self.is_temporal:
                time = self.temporal_extent[temporal_idx]
                patch_infos.append({"geometry": geometry, "id": pid, "time": time})
            else:
                patch_infos.append({"geometry": geometry, "id": pid})

        gdf = gpd.GeoDataFrame(patch_infos, crs=self.extent.crs.to_wkt())
        return gdf
