"""Utility functions for smart geocubes."""

import logging
import sys

import xarray as xr
from odc.geo.geobox import GeoBox
from odc.geo.geom import Geometry

logger = logging.getLogger(__name__)


def _check_python_version(min_major: int, min_minor: int) -> bool:
    """Check the current Python version against a minimum required version.

    Args:
        min_major (int): The minimum required major version.
        min_minor (int): The minimum required minor version.

    Returns:
        bool: True if the current Python version is greater than or equal to the minimum required

    """
    return sys.version_info.major > min_major or (
        sys.version_info.major == min_major and sys.version_info.minor >= min_minor
    )


def _geobox_repr(geobox: GeoBox) -> str:
    crs = f"EPSG:{geobox.crs.epsg}" if geobox.crs.epsg else "Non-EPSG CRS"

    is_degrees = geobox.crs.units[0].startswith("degree")
    is_meter = geobox.crs.units[0].startswith("metre") or geobox.crs.units[0].startswith("meter")

    if is_degrees:
        res = f"{abs(geobox.affine.a):.5f}° x {abs(geobox.affine.e):.5f}°"
    elif is_meter:
        res = f"{abs(geobox.affine.a):.1f}m x {abs(geobox.affine.e):.1f}m"
    else:
        res = f"{abs(geobox.affine.a)} x {abs(geobox.affine.e)}"

    return f"GeoBox({geobox.shape}, {res} @ [{geobox.affine.c}, {geobox.affine.f}] in {crs})"


def _geometry_repr(geometry: Geometry) -> str:
    crs = f"EPSG:{geometry.crs.epsg}" if geometry.crs.epsg else "Non-EPSG CRS"

    is_degrees = geometry.crs.units[0].startswith("degree")
    is_meter = geometry.crs.units[0].startswith("metre") or geometry.crs.units[0].startswith("meter")

    x, y = geometry.centroid.xy
    x, y = x[0], y[0]
    if is_meter:
        center_str = f"({x:.1f}m, {y:.1f}m)"
    elif is_degrees:
        center_str = f"({x:.5f}°, {y:.5f}°)"
    else:
        center_str = f"({x}, {y})"
    return f"Geometry({center_str} in {crs})"


def _log_xcube_stats(xcube: xr.Dataset, prefix: str = "xcube"):
    is_degrees = xcube.odc.geobox.crs.units[0].startswith("degree")
    is_meter = xcube.odc.geobox.crs.units[0].startswith("metre") or xcube.odc.geobox.crs.units[0].startswith("meter")

    x_extent = xcube.x[0].item(), xcube.x[1].item()
    y_extent = xcube.y[0].item(), xcube.y[1].item()
    x_res = abs(xcube.x[1] - xcube.x[0]).item()
    y_res = abs(xcube.y[1] - xcube.y[0]).item()
    if is_degrees:
        x_extent = f"{x_extent[0]:.5f}° - {x_extent[1]:.5f}°"
        y_extent = f"{y_extent[0]:.5f}° - {y_extent[1]:.5f}°"
        x_res = f"{x_res:.5f}°"
        y_res = f"{y_res:.5f}°"
    elif is_meter:
        x_extent = f"{x_extent[0]:.1f}m - {x_extent[1]:.1f}m"
        y_extent = f"{y_extent[0]:.1f}m - {y_extent[1]:.1f}m"
        x_res = f"{x_res:.1f}m"
        y_res = f"{y_res:.1f}m"
    else:
        x_extent = f"{x_extent[0]} - {x_extent[1]}"
        y_extent = f"{y_extent[0]} - {y_extent[1]}"
        x_res = str(x_res)
        y_res = str(y_res)

    logger.debug(f"{prefix}.sizes={xcube.sizes}")
    logger.debug(f"{prefix} X Extent: {x_extent} ({x_res})")
    logger.debug(f"{prefix} Y Extent: {y_extent} ({y_res})")

    if "time" in xcube.dims:
        time_extent = xcube.time[0].item(), xcube.time[-1].item()
        logger.debug(f"{prefix} Time Extent: {time_extent[0]} - {time_extent[1]}")
