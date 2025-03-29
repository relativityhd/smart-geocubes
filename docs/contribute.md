# Contribute

You are welcome do add a new pre-defined dataset.
For other features, please open an issue on GitHub.

## Roadmap

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
- [X] Predownload data
  - [X] Interface with geobox
  - [X] Interface with geopandas: find all intersecting tiles between geopandas and the cubes extent

Datasets:

- [X] ArcticDEM: increase readspeed by using extent files
- [X] TCTrend Dataset
- [ ] S2 Dataset
- [ ] Landsat Dataset

Admin:

- [X] Use StopUhr to measure performance
- [X] Write documentation (sphinx or mkdocs)
- [X] Add GitHub Action
- [X] Publish to PyPy
- [ ] Replace all execptions with custom exceptions
- [x] Further flatten the public facing API
- [ ] Replace TileWrapper NamedTuple with a dataclass
- [x] Make concurrency and storage module private
