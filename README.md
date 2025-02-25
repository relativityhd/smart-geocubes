# Smart Geocubes

A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray and zarr.

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
- [ ] GEE Accessor
- [ ] Widen support for lat-lon data

Datasets:

- [ ] TCVIS Dataset
- [ ] S2 Dataset
- [ ] Landsat Dataset

Admin:

- [ ] Write documentation (sphinx or mkdocs)
- [ ] Add GitHub Action
- [ ] Publish to PyPy
