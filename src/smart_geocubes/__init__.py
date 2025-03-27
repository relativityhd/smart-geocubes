"""Smart-Geocubes: A high-performance library for intelligent loading and caching of remote geospatial raster data, built with xarray and zarr."""  # noqa: E501

import importlib.metadata

try:
    __version__ = importlib.metadata.version("smart-geocubes")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"
