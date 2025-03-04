from typing import Literal

from smart_geocubes.accessors.base import ConcurrencyModes
from smart_geocubes.concurrency.blocking import BlockingMixin
from smart_geocubes.concurrency.threading import ThreadingMixin as ThreadingMixin


# This is a mixin class which combines all the concurrency mixins
class ConcurrentRemoteAccessor(BlockingMixin, ThreadingMixin):
    def procedural_download(self, geobox, concurrency_mode: ConcurrencyModes = "blocking"):
        if concurrency_mode == "blocking":
            self.procedural_download_blocking(geobox)
        elif concurrency_mode == "threading":
            self.procedural_download_threading(geobox)
        else:
            raise ValueError(f"Unknown concurrency mode {concurrency_mode}")
