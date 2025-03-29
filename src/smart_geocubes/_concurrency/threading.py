"""Threading handler for remote accessors."""

import logging
import queue
import threading
from collections.abc import Callable

import icechunk

from smart_geocubes.exceptions import AlreadyDownloadedError

logger = logging.getLogger(__name__)


class SetQueue[T](queue.Queue):
    """Set-like queue.

    Actually uses a list which checks for duplicates before adding an item.
    """

    def _init(self, maxsize):
        self.queue: list[T] = []

    def _put(self, item: T):
        if item.id not in [t.id for t in self.queue]:
            self.queue.append(item)

    def _get(self):
        return self.queue.pop(0)


def _threading_download_loop[T](f: Callable[[T], None], _queue: SetQueue[T]):
    while True:
        try:
            tile = _queue.get()
        except queue.ShutDown:
            break

        try:
            f(tile)
        except icechunk.ConflictError as e:
            logger.warning(f"Icechunk session is out of sync: {e}")
            logger.debug("Retrying download")
            _queue.put(tile)
        except AlreadyDownloadedError:
            continue


class ThreadingHandler[T]:
    """A threading handler for downloading tiles."""

    def __init__(self, f: Callable[[T], None]):
        """Create a new threading handler.

        Args:
            f (Callable[[T], None]): The function to call for each tile.

        """
        self._lock = threading.Lock()
        self._queue: SetQueue[T] = SetQueue()
        self._dthread: threading.Thread | None = None
        self.f = f

    def __del__(self):
        """Stop the download thread when the object is deleted."""
        self._queue.shutdown()
        with self._lock:
            if self._dthread is not None:
                self._dthread.join(timeout=3)

    def __enter__(self):
        """Start the download thread when used as context manager."""
        self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the download thread when used as context manager."""
        self.stop()

    def start(self):
        """Start the download thread, if not exists yet."""
        with self._lock:
            if self._dthread is None:
                self._dthread = threading.Thread(
                    target=_threading_download_loop, daemon=True, args=(self.f, self._queue)
                )
                self._dthread.start()

    def stop(self):
        """Stop the download thread."""
        with self._lock:
            if self._queue.empty():
                self._queue.shutdown(immediate=True)
                if self._dthread is not None:
                    self._dthread.join()
                    self._dthread = None
                self._queue = SetQueue()
