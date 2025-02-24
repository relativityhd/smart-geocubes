"""STAC Accessor for Smart Geocubes."""

import logging
from typing import TYPE_CHECKING

from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.base import RemoteAccessor

if TYPE_CHECKING:
    try:
        import geopandas as gpd
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class GEEAccessor(RemoteAccessor):
    """Accessor for Google Earth Engine data."""

    def procedural_download(self, geobox: GeoBox, data_vars: list[str] = None):
        """Download the data via GEE for the given geobox.

        Args:
            geobox (GeoBox): The reference geobox to download the data for.
            data_vars (list, optional): The data variables of the datacube.
                If set, overwrites the data variables set by the Dataset.
                Only used at first datacube creation. Defaults to None.

        """
        pass
