"""Smart-Geocubes: A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray and zarr."""  # noqa: E501

import importlib.metadata

from smart_geocubes import accessors as accessors
from smart_geocubes.datasets.alphaearth import AlphaEarthEmbeddings as AlphaEarthEmbeddings
from smart_geocubes.datasets.arcticdem import ArcticDEM2m as ArcticDEM2m
from smart_geocubes.datasets.arcticdem import ArcticDEM10m as ArcticDEM10m
from smart_geocubes.datasets.arcticdem import ArcticDEM32m as ArcticDEM32m
from smart_geocubes.datasets.tctrend import TCTrend as TCTrend
from smart_geocubes.datasets.tctrend import TCTrend2019 as TCTrend2019
from smart_geocubes.datasets.tctrend import TCTrend2020 as TCTrend2020
from smart_geocubes.datasets.tctrend import TCTrend2022 as TCTrend2022

try:
    __version__ = importlib.metadata.version("smart-geocubes")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "AlphaEarthEmbeddings",
    "ArcticDEM2m",
    "ArcticDEM10m",
    "ArcticDEM32m",
    "TCTrend",
    "TCTrend2019",
    "TCTrend2020",
    "TCTrend2022",
    "accessors",
]
