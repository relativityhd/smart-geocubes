"""Predefined accessor for ArcticDEM 32m, 10m and 2m data."""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from odc.geo.geobox import GeoBox

from smart_geocubes.accessors.stac import STACAccessor

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass


class ArcticDEMABC(STACAccessor):
    """ABC for Arcticdem data."""

    stac_api_url = "https://stac.pgc.umn.edu/api/v1/"
    chunk_size = 3600
    channels: ClassVar[list] = ["dem", "datamask"]
    channels_meta: ClassVar[dict] = {
        "dem": {
            "long_name": "Digital Elevation Model",
            "data_source": "ArcticDEM",
            "units": "m",
            "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
        },
        "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
    }
    channels_encoding: ClassVar[dict] = {
        "dem": {"dtype": "float32"},
        "datamask": {"dtype": "bool"},
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
        import matplotlib.path as mpath
        import matplotlib.pyplot as plt

        tile_info = self.current_state()

        if tile_info is None:
            raise ValueError("Datacube is not loaded yet. Can't visualize!")

        # Define the projection
        projection = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

        # Create a figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": projection})

        # Set the extent to focus on the North Pole
        ax.set_extent([-180, 180, 60, 90], crs=ccrs.PlateCarree())

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

        # Compute a circle in axes coordinates, which we can use as a boundary
        # for the map. We can pan/zoom as much as we like - the boundary will be
        # permanently circular.
        theta = np.linspace(0, 2 * np.pi, 100)
        center, radius = [0.5, 0.5], 0.5
        verts = np.vstack([np.sin(theta), np.cos(theta)]).T
        circle = mpath.Path(verts * radius + center)

        ax.set_boundary(circle, transform=ax.transAxes)

        tile_info.plot(
            "title",
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


class ArcticDEM32m(ArcticDEMABC):
    """Accessor for ArcticDEM 32m data."""

    collection = "arcticdem-mosaics-v4.1-32m"
    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=32)


class ArcticDEM10m(ArcticDEMABC):
    """Accessor for ArcticDEM 10m data."""

    collection = "arcticdem-mosaics-v4.1-10m"
    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=10)


class ArcticDEM2m(ArcticDEMABC):
    """Accessor for ArcticDEM 2m data."""

    collection = "arcticdem-mosaics-v4.1-2m"
    extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=2)
