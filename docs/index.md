---
hide:
  - navigation
---

# Smart-Geocubes

A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray, zarr and icechunk.

Earth-observation workflows often need to repeatedly inspect and transform large raster datasets without downloading everything up front.
That becomes expensive on laptops, awkward for iterative analysis, and hard to scale cleanly when processing regions of interest one tile at a time.

Smart-Geocubes is built for geospatial researchers, data engineers, and applied scientists who need reproducible, on-demand access to STAC- and Google Earth Engine-backed raster products.
It turns remote sources into cached, chunked datacubes so you can load only the areas you need, reuse what was already downloaded, and keep the workflow compatible with threaded or distributed processing.

!!! abstract "Inspiration"
    The concept of this package is heavily inspired by [EarthMovers implementation of serverless datacube generation](https://earthmover.io/blog/serverless-datacube-pipeline).

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

## What's next?

<div class="grid cards" markdown>

-   :material-lightbulb-on:{ .lg .middle } __Getting Started__

    ---

    Learn how to use `smart-geocubes` with the quickstart notebook.

    [:octicons-arrow-right-24: Get Started](examples/quickstart.ipynb)

-   :material-cloud-download:{ .lg .middle } __Write custom Dataset Accessors__

    ---

    Read how `smart-geocubes` to add a new dataset or even a custom remote accessor.

    [:octicons-arrow-right-24: Add your own dataset](custom-accessor.md)

-   :material-open-source-initiative:{ .lg .middle } __Contribute__

    ---

    Learn about what I plan to do with this package and how you can help.

    [:octicons-arrow-right-24: Contribute & Roadmap](contribute.md)

-   :material-api:{ .lg .middle } __API Reference__

    ---

    View the API reference of the components.

    [:octicons-arrow-right-24: Reference](reference/smart_geocubes)

</div>


## Out of the box included datasets

| Dataset                          | Quickuse                              | Source                                   | Link / Notes                                                                                                             |
| -------------------------------- | ------------------------------------- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| ArcticDEM Mosaic 2m              | `smart_geocubes.ArcticDEM2m`          | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                                                                           |
| ArcticDEM Mosaic 10m             | `smart_geocubes.ArcticDEM10m`         | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                                                                           |
| ArcticDEM Mosaic 32m             | `smart_geocubes.ArcticDEM32m`         | [STAC](https://stac.pgc.umn.edu/api/v1/) | [PGC](https://www.pgc.umn.edu/data/arcticdem/)                                                                           |
| Tasseled Cap Trends 2019         | `smart_geocubes.TCTrend2019`          | Google Earth Engine                      | [AWI](https://apgc.awi.de/dataset/pan-arctic-vis-landscape-change-2003-2022)                                             |
| Tasseled Cap Trends 2020         | `smart_geocubes.TCTrend2020`          | Google Earth Engine                      | [AWI](https://apgc.awi.de/dataset/pan-arctic-vis-landscape-change-2003-2022)                                             |
| Tasseled Cap Trends 2022         | `smart_geocubes.TCTrend2022`          | Google Earth Engine                      | [AWI](https://apgc.awi.de/dataset/pan-arctic-vis-landscape-change-2003-2022)                                             |
| Tasseled Cap Trends 2024         | `smart_geocubes.TCTrend2024`          | Google Earth Engine                      | [AWI](https://apgc.awi.de/dataset/pan-arctic-vis-landscape-change-2003-2022)                                             |
| AlphaEarth Satellite Embeddings* | `smart_geocubes.AlphaEarthEmbeddings` | Google Earth Engine                      | [EE](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?hl=de#description) |

*: Note that the original embeddings are stored in their respective UTM-Zones, but Smart-Geocubes reprojects them into EPSG:4326 to create a single Datacube. This may change in the future, if UTM-Zones are properly supported.

## Implemented Remote Accessors

| Accessor                        | Description                                                                               |
| ------------------------------- | ----------------------------------------------------------------------------------------- |
| `smart_geocubes.accessors.STAC` | Accessor for the STAC API, which allows to download data from a STAC API.                 |
| `smart_geocubes.accessors.GEE`  | Accessor for Google Earth Engine, which allows to download data from Google Earth Engine. |

## What is the purpose of this package?

This package solves a specific problem that Earth-observation practitioners run into when they need repeated, tile-based access to large raster datasets.
When you're creating new data from existing data (for example, doing image segmentation with machine learning on Sentinel-2 images), people usually:

1. Download all the data
2. Run the algorithms and data science on it
3. Delete the data afterwards

This "batched-processing" works great if you have a big computer with lots of storage space, like a cluster.

But if you're working on a smaller computer (like a laptop with a few hundred GB of storage and 16GB of RAM), this approach creates problems.
It makes it hard to test and improve your programs because you don't have enough space.
Using frameworks like Ray for processing is also tricky with this approach.
They work better with "concurrent-processing": when each step of your processing pipeline can be done for each element separately instead of expecting to run a single step for all your data at once.
Plus, if you only need to look at certain areas but don't know which ones ahead of time, downloading everything is wasteful.

So instead, this package downloads the data only when you need it. But downloading the same thing over and over is inefficient. That's why we save (or "cache") the data on your computer's hard drive in form of zarr datacubes.
We call this way of working "procedural download" because you download pieces as you need them.

Therefore, this package does handle:

1. The download "on-demand" (or "procedural download") of the data
2. The caching of the data on your computer's hard drive
3. The loading of the data into memory for regions specified by the user
4. Making everything thread-safe, so you can run on any scaling framework you like.

!!! danger "Multiprocessing"
    On linux systems it is necessary to the the multiprocessing start method to `spawn` or `forkserver`.
    Read more about this in [icechunk's documentation](https://icechunk.io/en/latest/icechunk-python/parallel/#uncooperative-distributed-writes), [a discussion on icechunk's GitHub repository](https://github.com/earth-mover/icechunk/discussions/802) and [in Polars documentation](https://docs.pola.rs/user-guide/misc/multiprocessing/).

The approach itself is already implemented in one of the pipelines we develop at the AWI, you can read more about [their docs](https://awi-response.github.io/darts-nextgen/latest/dev/auxiliary/#procedural-download).

!!! note "Cloud computing"
  
    This library won't help if your computer doesn't have fast storage space available - like if you're working on a cloud-cluster that can't save files locally.
