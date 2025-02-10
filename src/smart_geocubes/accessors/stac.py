"""STAC Accessor for Smart Geocubes."""

import logging
import time

import pystac
import pystac_client
import xarray as xr
import zarr
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor

logger = logging.getLogger(__name__)


class STACAccessor(RemoteAccessor):
    """Accessor for STAC data."""

    stac_api_url: str
    collection: str

    @classmethod
    def procedural_download(cls, storage: zarr.storage.StoreLike, geobox: GeoBox, data_vars: list[str] | None = None):
        """Download the data via STAC for the given geobox.

        Args:
            storage (zarr.storage.StoreLike): The zarr storage to download the data to.
            geobox (GeoBox): The reference geobox to download the data for.
            data_vars (list, optional): The data variables of the datacube.
                If set, overwrites the data variables set by the Dataset.
                Only used at first datacube creation. Defaults to None.

        """
        tick_fstart = time.perf_counter()

        data_vars = data_vars or cls.data_vars
        cube = xr.open_zarr(storage, mask_and_scale=False)

        catalog = pystac_client.Client.open(cls.stac_api_url)
        search = catalog.search(collections=[cls.collection], intersects=geobox.to_crs("EPSG:4326").extent.geom)
        items = list(search.items())

        # Check if zarr data already contains the data via the attrs
        loaded_tiles: list[str] = cube.attrs.get("loaded_tiles", []).copy()

        # Collect all tiles which should be downloaded
        new_tiles = [item for item in items if item.id not in loaded_tiles]

        if not len(new_tiles):
            logger.debug("No new tiles to download")
            return
        logger.debug(f"Found {len(new_tiles)} new tiles: {[tile.id for tile in new_tiles]}")

        # Collect the stac items
        items = pystac.ItemCollection(new_tiles)

        assert abs(cls.extent.resolution.x) == abs(cls.extent.resolution.y), (
            "Unequal x and y resolution is not supported yet"
        )
        resolution = abs(cls.extent.resolution.x)

        # Read the metadata and calculate the target slice
        # TODO: There should be a way to calculate the target slice without downloading the metadata
        # However, this is fine for now, since the overhead is very small and the resulting code very clear

        # This does not download the data into memory, since chunks=-1 will create a dask array
        # We need the coordinate information to calculate the target slice and the needed chunking for the real loading
        ds = xr.open_dataset(items, bands=data_vars, engine="stac", resolution=resolution, crs="3413", chunks=-1)

        # Get the slice of the datacube where the tile will be written
        x_start_idx = int((ds.x[0] - cube.x[0]) // ds.x.attrs["resolution"])
        y_start_idx = int((ds.y[0] - cube.y[0]) // ds.y.attrs["resolution"])
        target_slice = {
            "x": slice(x_start_idx, x_start_idx + ds.sizes["x"]),
            "y": slice(y_start_idx, y_start_idx + ds.sizes["y"]),
        }

        cube_aoi = cube.isel(target_slice).drop_vars("spatial_ref")

        # Now open the data for real, but still as dask array, hence the download occurs later
        ds = xr.open_dataset(
            items,
            bands=data_vars,
            engine="stac",
            resolution=resolution,
            crs=geobox.crs.to_epsg(),
            chunks=dict(cube_aoi.chunks),
        ).drop_vars("spatial_ref")
        if "time" not in data_vars:
            ds = ds.max("time")

        # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
        # Replace these nan borders with existing data if there is any
        ds = ds.fillna(cube_aoi)

        # Write the data to the datacube, we manually aligned the chunks, hence we can do safe_chunks=False
        tick_downloads = time.perf_counter()
        ds.to_zarr(storage, region=target_slice, safe_chunks=False)
        tick_downloade = time.perf_counter()
        logger.debug(f"Downloaded and written data to datacube in {tick_downloade - tick_downloads:.2f}s")

        # Update loaded_tiles (with native zarr, since xarray does not support this yet)
        loaded_tiles.extend([tile.id for tile in new_tiles])
        za = zarr.open(storage)
        za.attrs["loaded_tiles"] = loaded_tiles
        # Xarray default behaviour is to read the consolidated metadata, hence, we must update it
        zarr.consolidate_metadata(storage)

        tick_fend = time.perf_counter()
        logger.info(f"Procedural download of {len(new_tiles)} tiles completed in {tick_fend - tick_fstart:.2f} seconds")
