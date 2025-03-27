"""Predefined accessor for TCTrend data."""

from typing import TYPE_CHECKING, ClassVar

from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.gee import GEEAccessor

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass


class TCTrend(GEEAccessor):
    """Accessor for TCTrend data."""

    collection = "users/ingmarnitze/TCTrend_SR_2000-2019_TCVIS"
    extent = GeoBox.from_bbox((-180, -90, 180, 90), "epsg:4326", resolution=0.00026949458523585647)
    chunk_size = 3600
    channels: ClassVar[list] = ["TCB_slope", "TCG_slope", "TCW_slope"]
    channels_meta: ClassVar[dict] = {
        "TCB_slope": {
            "long_name": "Tasseled Cap Brightness Trend",
            "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
        },
        "TCG_slope": {
            "long_name": "Tasseled Cap Greenness Trend",
            "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
        },
        "TCW_slope": {
            "long_name": "Tasseled Cap Wetness Trend",
            "data_source": "ee:ingmarnitze/TCTrend_SR_2000-2019_TCVIS",
        },
    }
    channels_encoding: ClassVar[dict] = {
        "TCB_slope": {"dtype": "uint8"},
        "TCG_slope": {"dtype": "uint8"},
        "TCW_slope": {"dtype": "uint8"},
    }

    def visualize_state(self, ax: "plt.Axes | None" = None) -> "plt.Figure | plt.Axes":
        """Visulize the extend, hence the already downloaded and filled data, of the datacube.

        Args:
            ax (plt.Axes | None): The axes drawn to. If None, will create a new figure and axes.

        Returns:
            plt.Figure | plt.Axes: The figure with the visualization if no axes was provided, else the axes.

        Raises:
            ValueError: If the datacube is empty

        """
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        tile_info = self.current_state()

        if tile_info is None:
            raise ValueError("Datacube is not created or loaded yet. Can't visualize!")

        # Define the projection
        projection = ccrs.PlateCarree()

        # Create a figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": projection})

        # Set the extent to show the whole world
        ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())

        # Add features
        ax.add_feature(cfeature.LAND, zorder=0, edgecolor="black", facecolor="white")
        ax.add_feature(cfeature.OCEAN, zorder=0, facecolor="lightgrey")
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.LAKES, alpha=0.5)
        ax.add_feature(cfeature.RIVERS)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False

        tile_info.plot(
            "id",
            ax=ax,
            transform=ccrs.PlateCarree(),
            edgecolor="black",
            categorical=True,
            aspect="equal",
            alpha=0.5,
        )

        if fig is not None:
            return fig
        else:
            return ax
