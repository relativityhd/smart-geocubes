# Goals for 0.1.0

- Large Refactoring
- Better test coverage: Split tests into unit (small & no download) and integration tests, create test datasets

## Kind of Cubes

- Spatial-only cubes, e.g. ArcticDEM
- Spatial-only cubes with multiple CRSs, example MISSING
- Spatio-temporal cubes, e.g. TCTrend
- Spatio-temporal cubes with multiple CRSs, e.g. AlphaEarth, Sentinel-2

## Workflows

- On-demand data loading
- Pre-download data from an AOI
- Offline only data loading
- Check current state & visualize it

## Current Workflow

load -> procedural_download -> procedural_download_blocking -> download_tile -> _write_tile_to_zarr

## Suggested refactor

Convert GEE to STAC?
Use pystac objects as internal API?

### New workflow

load -> procedural_download -> backend.download_patch -> backend.write_patch

### Update dependencies

- zarr >= 3.2 (wait for https://github.com/zarr-developers/zarr-python/issues/3493 to be resolved)
- icechunk >= 1.1

### New naming scheme

- `Chunk`: Arrays as they are stored within the Zarr store
- `Patch`: Arrays as they are stored in the source data store, e.g. GEE or STAC
- `ROI`: Region of Interest (only spatial), this is what the user passes as indexing / querying arguments
- `TOI`: Time of Interest (only temporal), this is what the user passes as indexing / querying arguments
- `Tile`: Unprojected data arrays for a given ROI
- `Cube`: The zarr store containing all downloaded data

### New indexing / querying API

Use `odc.geo` Geobox for spatial queries, use `pandas` datetime indexing for temporal queries.

- `roi: GeoBox` for spatial queries
- `toi: Into<list[datetime]>` for temporal queries (anything that is supported by pandas indexing)

### New `Patch` indexing system

This should only be used internally and only by advanced users (me).
This system is the foundation of the internal logic glueing everything together.
A `PatchIndex` should have the following properties:

- A unique ID
- A `GeoBox` defining its spatial extent and resolution (and CRS)
- A time range (start, end) or time stamp defining its temporal extent

### Introduction of backends

Backends should handle all actions performed on the data store, hence downloading and writing patches.
They should also handle distributed computing.
A backend must implement the following interface:

`download_patch(self, patch: PatchIndex)`
`write_patch(self, patch: PatchIndex, patch: xr.DataSet)`

Backends which should be implemented:

- GEEBackend
- STACBackend

### New / Updated Core functions

- `patches(self) -> List[PatchIndex]`: Returns all available patches (TODO: some collections, e.g. Sentinel-2 have billions of patches, do this lazy?)
- `adjacent_patches(self, geobox: GeoBox)`

### New / Updated API

- `load(self, roi: GeoBox, toi: )`
