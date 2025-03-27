# Smart Geocubes

A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray, zarr and icechunk.

## Development

Install for everything:

```sh
uv sync --all-extras --all-groups
```

### Roadmap

Features:

- [ ] STAC: Add dask download as optional
- [x] STAC: Make the progress-bar optional
- [ ] Overall: Add support for temporal axis
- [ ] Overall: Add support for 3D data
- [ ] Overall: Add support for 4D data
- [X] GEE Accessor
- [ ] Widen support for lat-lon data
- [ ] Support different x-y resolutions
- [ ] True threaded mode: multiple threads for downloading, one thread for writing, multiple for loading

Datasets:

- [X] ArcticDEM: increase readspeed by using extent files
- [X] TCTrend Dataset
- [ ] S2 Dataset
- [ ] Landsat Dataset

Admin:

- [X] Use StopUhr to measure performance
- [ ] Write documentation (sphinx or mkdocs)
- [ ] Add GitHub Action
- [ ] Publish to PyPy
