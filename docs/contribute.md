# Contribute

You are welcome do add a new pre-defined dataset.
For other features, please open an issue on GitHub.

## Roadmap

Features:

- [ ] STAC: Add dask download as optional
- [x] Overall: Add support for temporal axis
- [ ] Overall: Add support for UTM-Zones
- [ ] Overall: Add support for 3D data
- [ ] Overall: Add support for 4D data
- [x] GEE Accessor
- [x] Widen support for lat-lon data
- [ ] Support different x-y resolutions
- [x] True threaded mode: multiple threads for downloading, one thread for writing, multiple for loading
- [x] Predownload data
  - [x] Interface with geobox
  - [x] Interface with geopandas: find all intersecting tiles between geopandas and the cubes extent

Datasets:

- [x] ArcticDEM: increase readspeed by using extent files
- [x] TCTrend Dataset
- [ ] S2 Dataset
- [ ] Landsat Dataset

Admin:

- [x] Use StopUhr to measure performance
- [x] Write documentation (sphinx or mkdocs)
- [x] Add GitHub Action
- [x] Publish to PyPy
- [ ] Replace all execptions with custom exceptions
- [x] Further flatten the public facing API
- [x] Replace TileWrapper NamedTuple with a dataclass
- [x] Make concurrency and storage module private
