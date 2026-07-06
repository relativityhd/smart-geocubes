# Contribute

You are welcome to add a new pre-defined dataset.
For bugs, feature requests, pull requests, and support questions, please open an issue on GitHub.

If you want to contribute code, please describe the problem you want to solve and the dataset or workflow it affects in a GitHub issue.
That makes it easier to align the contribution with the project roadmap.

Please read this page carefully and have a look at the `CODE_OF_CONDUCT.md` before contributing.

## Dev-Setup

Please work on a branch of a fork of this repository in order to create a pull-request with your changes later.
If you unsure how this works, please read [GitHub's guide for Open-Source contribution](https://opensource.guide/how-to-contribute/#how-to-submit-a-contribution) and [GitHub's documentation](https://docs.github.com/en/pull-requests).

Once you have cloned your fork, install all dependencies and dev-dependencies with `uv`:

```sh
uv sync --all-groups --all-extras
```

## Running tests

As of now, there are some tests implemented which actually download data from the remote-locations (STAC, GEE) and test the values against expected ones.
The download may take some time, up to several hours, depending on the internet connection.
Thus, it is not recommended to run all tests at once.

```sh
uv run pytest  # This may take some time!
```

### Google Earth Engine (GEE) — credentials & running integration tests

Some datasets and integration tests use Google Earth Engine (GEE). To run those tests or use the GEE accessors you must install the `gee` extras and authenticate with Earth Engine.

1. Authenticate with Earth Engine:

    Interactive (recommended for local development):

    ```bash
    earthengine authenticate
    ```

2. Running tests:

    - Unit tests (run in CI):

    ```bash
    pytest -q -m "not integration"
    ```

    - Integration tests (requires GEE authentication and `GEE_PROJECT` if used by tests):

    ```bash
    export GEE_PROJECT=your-project-id
    pytest -q -m integration
    ```

If you do not have GEE credentials, integration tests are intentionally marked with the `integration` marker and will be skipped by default in CI.

## Roadmap

Features:

- [ ] STAC: Add dask download as optional
- [x] Overall: Add support for temporal axis
- [ ] Overall: Add support for UTM-Zones
- [ ] Overall: Add support for 3D data
- [ ] Overall: Add support for 4D data
- [ ] Overall: Harden support for antimeridian
- [x] GEE Accessor
- [x] Widen support for lat-lon data
- [ ] Support different x-y resolutions
- [x] True threaded mode: multiple threads for downloading, one thread for writing, multiple for loading
- [x] Predownload data
  - [x] Interface with geobox
  - [x] Interface with geopandas: find all intersecting tiles between geopandas and the cubes extent
- [ ] Replace all execptions with custom exceptions

Datasets:

- [x] ArcticDEM: increase readspeed by using extent files
- [x] TCTrend Dataset
- [ ] S2 Dataset
- [ ] Landsat Dataset
