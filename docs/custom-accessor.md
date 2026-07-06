# Custom Datasets

A new dataset for existing source accessors (currently Google Earth Engine and `STAC`) can be added quite easily by inheriting one and fill in the required class variables:

- `extent` (`odc.geo.GeoBox`): The extent of the datacube represented by a GeoBox.
- `temporal_extent` (`pd.DatetimeIndex`): The temporal extent of the datacube.
- `chunk_size` (`int`): The chunk size of the datacube.
- `channels` (`list`): The channels of the datacube.

Further for the `GEEMosaicAccessor`:

- `collection` (`str`): The GEE collection of the dataset

and for the `STACAccessor`:

- `stac_api_url` (`str`): The URL to the STAC
- `collection` (`str`): The collection of the dataset

This could look like this:

```py
# Google Earth Engine
class AlphaEarthEmbeddings(GEEMosaicAccessor):
    # The collection is needed for the GEE Accessor, NOT for the STAC Accessor
    collection = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    # The rest is common for all accessors
    extent = GeoBox.from_bbox((-180, -90, 180, 90), "epsg:4326", resolution=8.98315284e-05)  # res ~ 10m at equator
    temporal_extent = pd.date_range("2017-01-01", "2024-01-01", freq="YS")
    chunk_size = 3600
    channels: ClassVar[list] = [f"A{i:02d}" for i in range(64)]
```

To enrich the persisted `Zarr` datacube with metadata attributes you can add a dict of dicts to `_channels_meta` and hardcode the encoding with `_channels_encoding`.
You can use the existing datasets as examples on how to use these attributes.

## Custom Source

As currently only Google Earth Engine and `STAC` based endpoints are supported, other sources like `VirtualiZarr` arrays or Microsofts Planetary Computer need a custom accessor.
To write such an accessor, inherit the `RemoteAccessor` ABC and implement the following functions:

- `adjacent_patches(self, roi: Geometry | GeoBox | gpd.GeoDataFrame, toi: TOI) -> list[PatchIndex]`: Checks which patches / tiles intersect with the region of interest
- `download_patch(self, idx: PatchIndex) -> xr.Dataset`: Downloads a single pathc / tile and returns it as an Xarray `Dataset`
- `current_state(self) -> gpd.GeoDataFrame | None`: Get the current state of the persisted `Zarr` datacube in form of a geopandas `GeoDataFrame` where each row represents an already downloaded patch / tile.
- `def visualize_state(self, ax: "plt.Axes | None" = None) -> "plt.Figure | plt.Axes"`: Visualizes above state

!!! note "Patches and Chunks"
    While **chunks** represent subset arrays of the persisted `Zarr` stores, **patches** represent the native subsets of the data's source, e.g. a single file of a mosaic.
    For sources like Google Earth Engine mosaics or `VirtualiZarr`, where the complete array is virtualized into a single continous array, patches can be defined arbitrary.
    E.g. the `GEEMosaicAccessor` uses `odc.geo.GeoboxTiles` which map the chunks of the persisted `Zarr` store, which allows for a much simpler and more efficient implementation.

If you really need to implement such an accessor I recommend to have a look at the two existing implementations and contact me via email or GitHub issue.
