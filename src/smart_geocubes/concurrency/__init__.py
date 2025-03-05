"""Concurrency support for remote accessors."""


class AlreadyDownloadedError(Exception):
    """Exception to raise when a tile is already downloaded."""

    pass
