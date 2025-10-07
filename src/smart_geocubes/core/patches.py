from dataclasses import dataclass

from odc.geo.geobox import GeoBox


@dataclass
class PatchIndex:
    """Metadata for a single patch in a data cube."""

    """Unique identifier for the patch."""
    id: str

    """The GeoBox defining the spatial extent and resolution of the patch."""
    geobox: GeoBox

    """The time range (start, end) or time stamp defining the temporal extent of the patch."""
    time_range: tuple[str, str] | None = None
