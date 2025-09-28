"""Predefined accessor for TCTrend data."""

from typing import TYPE_CHECKING, ClassVar

import pandas as pd
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.gee import GEEAccessor

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass


class AlphaEarthEmbeddings(GEEAccessor):
    """Accessor for AlphaEarth Embeddings data.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the zarr array. Should be equal to the extent geobox.
        created (bool): True if the datacube already exists in the storage.

    """

    collection = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
    # res ~ 10m at equator
    extent = GeoBox.from_bbox((-180, -90, 180, 90), "epsg:4326", resolution=8.98315284e-05)
    temporal_extent = pd.date_range("2017-01-01", "2024-01-01", freq="YS")
    chunk_size = 3600
    channels: ClassVar[list] = [f"A{i:02d}" for i in range(64)]
    _channels_meta: ClassVar[dict] = {
        f"A{i:02d}": {
            "long_name": f"AlphaEarth Embedding Band {i}",
            "data_source": "ee:GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        }
        for i in range(64)
    }
    _channels_encoding: ClassVar[dict] = {f"A{i:02d}": {"dtype": "float32"} for i in range(64)}

    def visualize_state(self, ax: "plt.Axes | None" = None) -> "plt.Figure | plt.Axes":
        """Visulize the extend, hence the already downloaded and filled data, of the datacube.

        Args:
            ax (plt.Axes | None): The axes drawn to. If None, will create a new figure and axes.

        Returns:
            plt.Figure | plt.Axes: The figure with the visualization if no axes was provided, else the axes.

        Raises:
            ValueError: If the datacube is empty

        """
        raise NotImplementedError("Visualization not implemented yet for AlphaEarth Embeddings datacube.")
