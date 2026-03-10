"""Metadata for a single patch in a data cube."""

from dataclasses import dataclass
from datetime import datetime
from typing import Generic, TypeVar

from odc.geo.geom import Geometry

PatchItem = TypeVar("PatchItem")


@dataclass
class PatchIndex(Generic[PatchItem]):
    """Metadata for a single patch in a data cube."""

    """Unique identifier for the patch."""
    id: str

    # TODO: check weather geobox and time should be deleted?
    """The Geometry defining the spatial extent in EPSG:4326."""
    geometry: Geometry

    """The time of the patch, if applicable."""
    time: tuple[str, str] | tuple[datetime, datetime] | str | datetime | None = None

    """An associated item or metadata object useful for other operations."""
    item: PatchItem | None = None
