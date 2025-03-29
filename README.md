# Smart Geocubes

A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray, zarr and icechunk.

> The concept of this package is heavily inspired by [EarthMovers implementation of serverless datacube generation](https://earthmover.io/blog/serverless-datacube-pipeline).

## Quickstart

Install the package with `uv` or `pip`:

```sh
pip install smart-geocubes
```

```sh
uv add smart-geocubes
```

Open data for your region of interest:

```python
import smart_geocubes
from odc.geo.geobox import GeoBox

accessor = smart_geocubes.ArcticDEM32m("datacubes/arcticdem_32m.icechunk")

roi = GeoBox.from_bbox((150, 65, 151, 65.5), shape=(1000, 1000), crs="EPSG:4326")

arcticdem_at_roi = accessor.load(roi, create=True)
```

## Out of the box included datasets

| Dataset              | Quickuse                      | Source                                   | Link                                                                         |
| -------------------- | ----------------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| ArcticDEM Mosaic 2m  | `smart_geocubes.ArcticDEM2m`  | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                               |
| ArcticDEM Mosaic 10m | `smart_geocubes.ArcticDEM10m` | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                               |
| ArcticDEM Mosaic 32m | `smart_geocubes.ArcticDEM32m` | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                               |
| Tasseled Cap Tren    | `smart_geocubes.TCTrend`      | Google Earth Engine                      | [AWI](https://apgc.awi.de/dataset/pan-arctic-vis-landscape-change-2003-2022) |

## Implemented Remote Accessors

| Accessor                        | Description                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `smart_geocubes.accessors.STAC` | Accessor for the STAC API, which allows to download data from a STAC API.                 |
| `smart_geocubes.accessors.GEE`  | Accessor for Google Earth Engine, which allows to download data from Google Earth Engine. |

## What is the purpose of this package?

This package solves a specific problem that most people who work with Earth observation data don't need to worry about.
When you're creating new data from existing data (for example, doing image segmentation with machine learning on Sentinel-2 images), people usually:

1. Download all the data
2. Run the algorithms and data science on it
3. Delete the data afterwards

This "batched-processing" works great if you have a big computer with lots of storage space, like a cluster.

But if you're working on a smaller computer (like a laptop with a few hundred GB of storage and 16GB of RAM), this approach creates problems.
It makes it really hard to test and improve your programs because you don't have enough space.
Using frameworks like Ray for processing is also tricky with this approach.
They work better with "concurrent-processing": when each step of your processing pipeline can be done for each elements separately instead expecting to run a single step for all your data at once.
Plus, if you only need to look at certain areas but don't know which ones ahead of time, downloading everything is wasteful.

So instead, this package downloads the data only when you need it. But downloading the same thing over and over is inefficient. That's why we save (or "cache") the data on your computer's hard drive in form of zarr datacubes.
We call this way of working "procedural download" because you download pieces as you need them.

Therefore, this package does handle:

1. The download "on-demand" (or "procedural download") of the data
2. The caching of the data on your computer's hard drive
3. The loading of the data into memory for regions specified by the user
4. Making everything thread-safe, so you can run on any scaling framework you like.

> **Danger!**
> On linux systems it is necessary to the the multiprocessing start method to `spawn` or `forkserver`.
> Read more about this [here](https://icechunk.io/en/latest/icechunk-python/parallel/#uncooperative-distributed-writes), [here](https://github.com/earth-mover/icechunk/discussions/802) and [here](https://docs.pola.rs/user-guide/misc/multiprocessing/).

The approach itself is already implemented in one of the pipelines we develop at the AWI, you can read more about [their docs](https://awi-response.github.io/darts-nextgen/latest/dev/auxiliary/#procedural-download).

> This won't help if your computer doesn't have fast storage space available - like if you're working on a cloud-cluster that can't save files locally.

## Contribute

Please read the [contribution guidelines](docs/contribute.md) for more information on how to contribute to this project.
