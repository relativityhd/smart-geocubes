"""Predefined accessor for ArcticDEM 32m, 10m and 2m data."""

from typing import ClassVar

from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.stac import STACAccessor


class ArcticDEM32m(STACAccessor):
    """Accessor for ArcticDEM 32m data."""

    stac_api_url = "https://stac.pgc.umn.edu/api/v1/"
    collection = "arcticdem-mosaics-v4.1-32m"

    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=32)
    chunk_size = 3600
    data_vars: ClassVar[list] = ["dem", "datamask"]
    data_vars_meta: ClassVar[dict] = {
        "dem": {
            "long_name": "Digital Elevation Model",
            "data_source": "ArcticDEM",
            "units": "m",
            "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
        },
        "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
    }
    data_vars_encoding: ClassVar[dict] = {
        "dem": {"dtype": "float32"},
        "datamask": {"dtype": "bool"},
    }


class ArcticDEM10m(STACAccessor):
    """Accessor for ArcticDEM 10m data."""

    stac_api_url = "https://stac.pgc.umn.edu/api/v1/"
    collection = "arcticdem-mosaics-v4.1-10m"

    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=10)
    chunk_size = 3600
    data_vars: ClassVar[list] = ["dem", "datamask"]
    data_vars_meta: ClassVar[dict] = {
        "dem": {
            "long_name": "Digital Elevation Model",
            "data_source": "ArcticDEM",
            "units": "m",
            "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
        },
        "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
    }
    data_vars_encoding: ClassVar[dict] = {
        "dem": {"dtype": "float32"},
        "datamask": {"dtype": "bool"},
    }


class ArcticDEM2m(STACAccessor):
    """Accessor for ArcticDEM 2m data."""

    stac_api_url = "https://stac.pgc.umn.edu/api/v1/"
    collection = "arcticdem-mosaics-v4.1-2m"

    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=2)
    chunk_size = 3600
    data_vars: ClassVar[list] = ["dem", "datamask"]
    data_vars_meta: ClassVar[dict] = {
        "dem": {
            "long_name": "Digital Elevation Model",
            "data_source": "ArcticDEM",
            "units": "m",
            "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
        },
        "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
    }
    data_vars_encoding: ClassVar[dict] = {
        "dem": {"dtype": "float32"},
        "datamask": {"dtype": "bool"},
    }
