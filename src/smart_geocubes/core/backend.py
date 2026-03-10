"""Write specific backends."""

import abc
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import TypedDict

import icechunk
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from zarr.core.sync import sync

from smart_geocubes.core.patches import PatchIndex
from smart_geocubes.core.utils import _log_xcube_stats

logger = logging.getLogger(__name__)


class _Event(TypedDict):
    action: str
    patch_id: str
    time: datetime
    thread_id: int
    thread_name: str


class DownloadBackend(abc.ABC):
    """Base class for download backends."""

    def __init__(self, repo: icechunk.Repository, f: Callable[[PatchIndex], xr.Dataset]):
        """Initialize the ThreadedBackend.

        Args:
            repo (icechunk.Repository): The icechunk repository.
            f (callable[[PatchIndex], xr.Dataset]): A function that takes a PatchIndex and returns an xr.Dataset.
                This should be implemented by the specific source backend.

        """
        self.repo = repo
        self.download_from_source = f

        # For debugging purposes
        self._events: list[_Event] = []

    @property
    def created(self) -> bool:
        """Check if the datacube already exists in the storage.

        Returns:
            bool: True if the datacube already exists in the storage.

        """
        return not sync(self.repo.readonly_session("main").store.is_empty(""))

    def _log_event(self, action: str, patch_id: str):
        self._events.append(
            {
                "action": action,
                "patch_id": patch_id,
                "time": datetime.now(),
                "thread_id": threading.get_ident(),
                "thread_name": threading.current_thread().name,
            }
        )

    def _get_events(self) -> pd.DataFrame:
        if len(self._events) == 0:
            logger.info("No events to show")
            return pd.DataFrame()

        events = pd.DataFrame(self._events)

        # Convert the event to a clearner format for analysis
        # cols: patch_id, action, time_start, time_end, duration, thread_id
        start_events = events[events["action"].str.startswith("start")].copy()
        end_events = events[events["action"].str.startswith("end")].copy()
        start_events["action"] = start_events["action"].str.replace("start_", "")
        end_events["action"] = end_events["action"].str.replace("end_", "")
        merged = start_events.merge(
            end_events,
            on=["patch_id", "action", "thread_id", "thread_name"],
            suffixes=("_start", "_end"),
        )
        merged["duration"] = merged["time_end"] - merged["time_start"]

        for col in ("action", "patch_id", "time_start", "time_end", "thread_id", "duration", "thread_name"):
            assert col in merged.columns, f"Missing required column '{col}' in events; something broke..."

        return merged

    def _plot_events(self):
        """Plot thread activity as a Gantt-like chart.

        The plot shows threads on the y-axis and time on the x-axis, with bars
        colored by action type (download vs write). Each bar represents the
        interval between `time_start` and `time_end` for a given `patch_id` and
        `thread_id`.

        Returns:
            tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or None: (fig, ax)
                when there is data to plot, otherwise None.

        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        events = self._get_events()
        if events.empty:
            logger.info("No events to plot")
            return None

        for col in ("action", "patch_id", "time_start", "time_end", "thread_name", "duration"):
            assert col in events.columns, f"Missing required column '{col}' in events; cannot plot."

        df = events.copy()

        # Remove retry suffixes (e.g., "_try0", "_try68") while preserving variable names
        # This keeps "write_dem", "write_datamask" distinct but collapses "write_dem_try0" to "write_dem"
        df["action_type"] = df["action"].str.replace(r"_try\d+$", "", regex=True)

        # Map threads to y positions
        threads = sorted(df["thread_name"].dropna().unique())
        thread_y = {t: i for i, t in enumerate(threads)}

        # Convert times to matplotlib internal date format
        # If end is missing, show it up to now so ongoing work is visible
        df["_start_num"] = mdates.date2num(df["time_start"])
        df["_end_num"] = mdates.date2num(df["time_end"].fillna(pd.Timestamp.now()))
        df["width"] = df["_end_num"] - df["_start_num"]

        # Create color mapping: download gets one color, all write_* actions get variations of another
        unique_actions = df["action_type"].unique()
        palette = {}

        # Color for download
        if "download" in unique_actions:
            palette["download"] = "#1f77b4"

        # Colors for write actions - use different shades/colors for different variables
        write_actions = [a for a in unique_actions if a.startswith("write")]
        if write_actions:
            # Use a color cycle for different write actions
            write_colors = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
            for i, action in enumerate(sorted(write_actions)):
                palette[action] = write_colors[i % len(write_colors)]

        fig, ax = plt.subplots(figsize=(12, max(3, len(threads) * 0.6)))

        for _, row in df.iterrows():
            y = thread_y.get(row["thread_name"], len(threads))
            start = row["_start_num"]
            width = max(row["width"], 1e-6)
            color = palette.get(row["action_type"], palette["download"])
            ax.barh(y, width, left=start, height=0.6, align="center", color=color, edgecolor="k", alpha=0.9)

            # annotate with patch id if the bar is wide enough
            patch_label = str(row.get("patch_id", ""))
            if patch_label:
                text_x = start + 0.01 * width
                ax.text(text_x, y, patch_label, va="center", ha="left", color="white", fontsize=6, clip_on=True)

        # Y ticks
        y_ticks = [thread_y[t] for t in threads]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(t) for t in threads])
        ax.set_ylabel("thread_id")

        # X axis as dates
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        fig.autofmt_xdate()

        # Legend for actions - show all unique action types
        legend_actions = sorted(palette.keys())
        handles = [plt.Rectangle((0, 0), 1, 1, color=palette[k]) for k in legend_actions]
        ax.legend(handles, legend_actions, title="action", bbox_to_anchor=(1.02, 1), loc="upper left")

        ax.set_title("Thread activity timeline")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        return fig, ax

    def assert_created(self, session: icechunk.Session | None = None):
        """Assert that the datacube exists in the storage.

        Raises:
            FileNotFoundError: If the datacube does not exist.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        if sync(session.store.is_empty("")):
            msg = "Datacube does not exist."
            " Please use the `create` method or pass `create=True` to `load`."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def open_zarr(self, session: icechunk.Session | None = None) -> zarr.Group:
        """Open the zarr datacube in read-only mode.

        Returns:
            zarr.Group: The zarr datacube.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        self.assert_created(session)
        zcube = zarr.open(store=session.store, mode="r")
        return zcube

    def open_xarray(self, session: icechunk.Session | None = None) -> xr.Dataset:
        """Open the xarray datacube in read-only mode.

        Returns:
            xr.Dataset: The xarray datacube.

        """
        if session is None:
            session = self.repo.readonly_session("main")
        self.assert_created(session)
        xcube = xr.open_zarr(session.store, mask_and_scale=False, consolidated=False).set_coords("spatial_ref")
        return xcube

    def loaded_patches(self, session: icechunk.Session | None = None) -> list[str]:
        """Get a list of all loaded patch ids.

        Returns:
            list[str]: A list of all loaded patch ids.

        """
        zcube = self.open_zarr(session)
        loaded_patches = zcube.attrs.get("loaded_patches", [])
        return loaded_patches

    def _get_target_slice(
        self, patch: xr.Dataset, session: icechunk.Session | None = None
    ) -> tuple[slice | list[int], slice, slice] | tuple[slice, slice]:
        xcube = self.open_xarray(session)
        _log_xcube_stats(xcube, "Target xcube")

        spatial_target = xcube.odc.geobox.overlap_roi(patch.odc.geobox)

        if "time" not in xcube.dims:
            logger.debug(f"Geocube is not temporal - writing to {spatial_target=}.")
            return spatial_target

        assert "time" in patch.dims, "Geocube is temporal - patches need a time dimension."
        assert len(patch.time) >= 1, "Patch must have at least one time step."

        extent = xcube.get_index("time").normalize()
        temporal_target = extent.get_indexer(patch.time.values, method="nearest")

        logger.debug(f"Writing to temporal {temporal_target=} and spatial {spatial_target=}.")

        return (temporal_target, *spatial_target)

    def _write_patch_variable(self, zcube: zarr.Group, data: np.ndarray, var: str, target: tuple, patch_id: str):
        # Sometimes the data downloaded from stac has nan-borders, which would overwrite existing data
        # Replace these nan borders with existing data if there is any
        self._log_event(f"start_write_{var}", patch_id)
        mask = np.isnan(data)
        if np.any(mask):
            existing_data = zcube[var][target]
            data[mask] = existing_data[mask]
        zcube[var][target] = data
        self._log_event(f"end_write_{var}", patch_id)

    def _download_from_source_with_retries(self, idx: PatchIndex) -> xr.Dataset:
        # Also handles download errors properly
        logger.debug(f"Downloading patch {idx.id}...")
        last_exception: Exception | None = None
        for t in range(5):
            try:
                patch = self.download_from_source(idx)
                break
            except (KeyboardInterrupt, SystemError, SystemExit) as e:
                raise e
            except Exception as e:
                logger.error(f"{idx.id=}: {e=} at download retry {t}/5")
                last_exception = e
        else:
            logger.error(
                f"{idx.id} failed to download after 5 tries. Please check your internet connection and data access."
            )
            raise ValueError(f"{idx.id=}: 5 tries to download the tile failed. ") from last_exception
        patch.attrs["patch_id"] = idx.id
        return patch

    def close(self) -> bool:
        """Close the backend.

        Returns:
            bool: True if the backend was closed successfully, False otherwise.

        """
        # Base implementation does nothing
        return True

    @abc.abstractmethod
    def submit(self, idx: PatchIndex | list[PatchIndex]):
        """Submit a patch download request to the backend.

        Args:
            idx (PatchIndex | list[PatchIndex]): The index or multiple indices of the patch(es) to download.

        """
        pass
