"""Predefined accessor for ArcticDEM 32m, 10m and 2m data."""

import io
import logging
import os
import zipfile
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import geopandas as gpd
import numpy as np
from odc.geo.geobox import GeoBox
from stopuhr import stopuhr

from smart_geocubes.accessors.base import TileWrapper
from smart_geocubes.accessors.stac import STACAccessor

if TYPE_CHECKING:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        pass

logger = logging.getLogger(__name__)


class LazyStacTileWrapper:
    """Lazy wrapper for a TileWrapper containing a STAC Item.

    This is necessary since the download function of the STAC accessor expects a
    TileWrapper object containing a pystac.Item.

    However, creating such a pystac Item always fetches the metadata from the STAC API.
    For just loading the ArcticDEM data, we don't need this pystac Item.
    Hence, we create it lazily when it is actually needed.
    """

    def __init__(self, tile_id: str, stac_file: str):  # noqa: D107
        self.id = tile_id
        self.stac_file = stac_file

    def __iter__(self):  # noqa: D105
        return iter([self.id, self.item])

    @cached_property
    def item(self):  # noqa: D102
        import pystac

        return pystac.Item.from_file(self.stac_file)


def _download_arcticdem_extent(save_dir: Path):
    """Download the ArcticDEM mosaic extent info from the provided URL and extracts it to the specified directory.

    Args:
        save_dir (Path): The directory where the extracted data will be saved.

    Example:
        ```python
        from darts_acquisition.arcticdem.datacube import download_arcticdem_extent

        save_dir = Path("data/arcticdem")
        download_arcticdem_extent(save_dir)
        ```

        Resulting in the following directory structure:

        ```sh
        $ tree data/arcticdem
        data/arcticdem
        ├── ArcticDEM_Mosaic_Index_v4_1_2m.parquet
        ├── ArcticDEM_Mosaic_Index_v4_1_10m.parquet
        └── ArcticDEM_Mosaic_Index_v4_1_32m.parquet
        ```

    """
    import requests

    with stopuhr("Downloading the ArcticDEM mosaic extent", printer=logger.debug):
        url = "https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/indexes/ArcticDEM_Mosaic_Index_latest_gpqt.zip"
        logger.debug(f"Downloading the arcticdem mosaic extent from {url} to {save_dir.resolve()}")
        response = requests.get(url)

        # Get the downloaded data as a byte string
        data = response.content
        logger.debug(f"Downloaded {len(data)} bytes")

    with stopuhr("Extracting the ArcticDEM mosaic extent", printer=logger.debug):
        # Create a bytesIO object
        with io.BytesIO(data) as buffer:
            # Create a zipfile.ZipFile object and extract the files to a directory
            save_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(buffer, "r") as zip_ref:
                # Get the name of the zipfile (the parent directory)
                zip_name = zip_ref.namelist()[0].split("/")[0]

                # Extract the files to the specified directory
                zip_ref.extractall(save_dir)

        # Move the extracted files to the parent directory
        extracted_dir = save_dir / zip_name
        for file in extracted_dir.iterdir():
            file.rename(save_dir / file.name)

        # Remove the empty directory
        extracted_dir.rmdir()

    logger.info(f"Download and extraction of the arcticdem mosiac extent from {url} to {save_dir.resolve()} completed")


def _get_stac_url(dem_id: str, res: str) -> str:
    """Convert the dem_id to a STAC URL.

    Args:
        dem_id (str): The dem_id of the ArcticDEM data. E.g. "36_24_32m_v4.1"
        res (str): The resolution of the ArcticDEM data. E.g. "32m", "10m", "2m"

    Returns:
        str: The STAC URL of the ArcticDEM data.

    """
    return f"https://stac.pgc.umn.edu/api/v1/collections/arcticdem-mosaics-v4.1-{res}/items/{dem_id}"


class ArcticDEMABC(STACAccessor):
    """ABC for Arcticdem data.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the underlaying zarr array. Should be equal to the extent geobox.
            However, this property is used to find the target index of the downloaded data, so better save than sorry.
        created (bool): True if the datacube already exists in the storage.

    """

    stac_api_url = "https://stac.pgc.umn.edu/api/v1/"
    chunk_size = 3600
    channels: ClassVar[list] = ["dem", "datamask"]
    _channels_meta: ClassVar[dict] = {
        "dem": {
            "long_name": "Digital Elevation Model",
            "data_source": "ArcticDEM",
            "units": "m",
            "description": "Digital Elevation Model, elevation resolution is cropped to ~1cm",
        },
        "datamask": {"long_name": "Data Mask", "source": "ArcticDEM"},
    }
    _channels_encoding: ClassVar[dict] = {
        "dem": {"dtype": "float32"},
        "datamask": {"dtype": "bool"},
    }

    @cached_property
    def _aux_dir(self) -> Path:
        save_dir = os.environ.get("SMART_GEOCUBES_AUX", None)
        save_dir = Path(save_dir) if save_dir else Path(__file__).parent.parent / "data"
        save_dir.mkdir(exist_ok=True)
        return save_dir

    def post_init(self):
        """Check if the ArcticDEM mosaic extent info is already downloaded and downlaod if not."""
        required_files = [self._aux_dir / f"ArcticDEM_Mosaic_Index_v4_1_{res}.parquet" for res in ["2m", "10m", "32m"]]
        if not all(file.exists() for file in required_files):
            _download_arcticdem_extent(self._aux_dir)

    def post_create(self):
        """Download the ArcticDEM mosaic extent info and store it in the datacube."""
        _download_arcticdem_extent(self._aux_dir)

    def adjacent_tiles(self, roi: GeoBox | gpd.GeoDataFrame) -> list[TileWrapper]:
        """Get adjacent tiles from a STAC API.

        Overwrite the default implementation from the STAC accessor
        to use pre-downloaded extent files instead of querying the STAC API.
        This results in a faster loading time, but requires the extent files to be downloaded beforehand.
        This is done in the post_create step.

        Args:
            roi (GeoBox | gpd.GeoDataFrame): The reference geobox or reference geodataframe

        Returns:
            list[TileWrapper]: List of adjacent tiles, wrapped in own datastructure for easier processing.

        Raises:
            ValueError: If the roi is not a GeoBox or a GeoDataFrame.

        """
        # Assumes that the extent files are already present and the datacube is already created
        self.assert_created()

        resolution = f"{int(self.extent.resolution.x)}m"
        extent_info = gpd.read_parquet(self._aux_dir / f"ArcticDEM_Mosaic_Index_v4_1_{resolution}.parquet")
        if isinstance(roi, gpd.GeoDataFrame):
            adjacent_tiles = (
                gpd.sjoin(
                    extent_info,
                    roi[["geometry"]].to_crs(self.extent.crs.wkt),
                    how="inner",
                    predicate="intersects",
                )
                .reset_index()
                .drop_duplicates(subset="index", keep="first", ignore_index=True)
            )
        elif isinstance(roi, GeoBox):
            adjacent_tiles = extent_info.loc[extent_info.intersects(roi.boundingbox.polygon.geom)].copy()
        else:
            raise ValueError("roi must be a GeoBox or a GeoDataFrame")
        if adjacent_tiles.empty:
            return []
        return [
            LazyStacTileWrapper(tile.dem_id, _get_stac_url(tile.dem_id, resolution))
            for tile in adjacent_tiles.itertuples()
        ]

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
            raise ValueError("Datacube is not created or loaded yet. Can't visualize!")

        # Define the projection
        projection = ccrs.Stereographic(central_latitude=90, central_longitude=-45, true_scale_latitude=70)

        # Create a figure
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": projection})

        # Set the extent to focus on the North Pole
        ax.set_extent([-180, 180, 50, 90], crs=ccrs.PlateCarree())

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
    """Accessor for ArcticDEM 32m data.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the underlaying zarr array. Should be equal to the extent geobox.
            However, this property is used to find the target index of the downloaded data, so better save than sorry.
        created (bool): True if the datacube already exists in the storage.

    """

    collection = "arcticdem-mosaics-v4.1-32m"
    # extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=32)
    extent: GeoBox = GeoBox.from_bbox((-4000096, -3400096, 3400096, 4100096), "epsg:3413", resolution=32)


class ArcticDEM10m(ArcticDEMABC):
    """Accessor for ArcticDEM 10m data.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the underlaying zarr array. Should be equal to the extent geobox.
            However, this property is used to find the target index of the downloaded data, so better save than sorry.
        created (bool): True if the datacube already exists in the storage.

    """

    collection = "arcticdem-mosaics-v4.1-10m"
    # extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=10)
    extent: GeoBox = GeoBox.from_bbox((-4000100, -3400100, 3400100, 4100100), "epsg:3413", resolution=10)


class ArcticDEM2m(ArcticDEMABC):
    """Accessor for ArcticDEM 2m data.

    Attributes:
        extent (GeoBox): The extent of the datacube represented by a GeoBox.
        chunk_size (int): The chunk size of the datacube.
        channels (list): The channels of the datacube.
        storage (icechunk.Storage): The icechunk storage.
        repo (icechunk.Repository): The icechunk repository.
        title (str): The title of the datacube.
        stopuhr (StopUhr): The benchmarking timer from the stopuhr library.
        zgeobox (GeoBox): The geobox of the underlaying zarr array. Should be equal to the extent geobox.
            However, this property is used to find the target index of the downloaded data, so better save than sorry.
        created (bool): True if the datacube already exists in the storage.

    """

    collection = "arcticdem-mosaics-v4.1-2m"
    # extent: GeoBox = GeoBox.from_bbox((-3314693.24, -3314693.24, 3314693.24, 3314693.24), "epsg:3413", resolution=2)
    extent: GeoBox = GeoBox.from_bbox((-4000100, -3400100, 3400100, 4100100), "epsg:3413", resolution=2)
