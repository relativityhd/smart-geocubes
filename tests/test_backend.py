"""Unit tests for the DownloadBackend orchestration logic.

Covers retries, the NaN-border write behavior, and ThreadedBackend's concurrent
write path and crash handling.

`_download_from_source_with_retries` only ever calls the injected `f`, so it is
tested directly against a bare SimpleBackend with `repo=None` — no icechunk store
needed. The write-path tests use FakeGEEAccessor (see conftest.py) for a real,
local-filesystem icechunk store.
"""

import threading
from typing import cast

import icechunk
import numpy as np
import odc.geo.xr
import pytest
import xarray as xr
import zarr
from numpy.testing import assert_array_equal

from smart_geocubes.backends.simple import SimpleBackend
from smart_geocubes.core.patches import PatchIndex


def _make_idx(id_: str = "p1") -> PatchIndex:
    # `geometry` is unused by the code under test here; ty otherwise wants a real
    # `Geometry` since PatchIndex's field isn't declared optional.
    return PatchIndex(id_, geometry=None, time=None, item=None)  # ty:ignore[invalid-argument-type]


def _make_backend(f) -> SimpleBackend:
    # `repo` is unused by `_download_from_source_with_retries`, which is all these
    # retry tests exercise; ty otherwise wants a real `icechunk.Repository`.
    return SimpleBackend(repo=None, f=f)  # ty:ignore[invalid-argument-type]


def _open_writable_zcube(session: icechunk.Session) -> zarr.Group:
    zcube = zarr.open(store=session.store, mode="r+")
    assert isinstance(zcube, zarr.Group), "Expected the zarr store to be a zarr Group."
    return zcube


def _array(zcube: zarr.Group, name: str) -> zarr.Array:
    arr = zcube[name]
    assert isinstance(arr, zarr.Array), f"Expected {name!r} to be a zarr Array."
    return arr


def _read_all(arr: zarr.Array) -> np.ndarray:
    # zarr's Array.__getitem__ is typed to return a large union of possible scalar
    # and array-like results; a plain full-array slice always yields a real ndarray
    # though. The codebase works around the same zarr-typeshed limitation elsewhere,
    # see e.g. `_write_patch_variable` in src/smart_geocubes/core/backend.py.
    return cast(np.ndarray, arr[:])


# --- _download_from_source_with_retries -------------------------------------------


def test_retries_succeeds_on_first_try():
    calls = []

    def f(idx):
        calls.append(idx.id)
        return xr.Dataset()

    backend = _make_backend(f)
    result = backend._download_from_source_with_retries(_make_idx("p1"))

    assert calls == ["p1"]
    assert result.attrs["patch_id"] == "p1"


def test_retries_succeeds_after_transient_failures():
    calls = []

    def f(idx):
        calls.append(idx.id)
        if len(calls) < 3:
            raise RuntimeError("transient")
        return xr.Dataset()

    backend = _make_backend(f)
    result = backend._download_from_source_with_retries(_make_idx("p1"))

    assert len(calls) == 3
    assert result.attrs["patch_id"] == "p1"


def test_retries_raises_after_five_failures_with_cause():
    calls = []

    def f(idx):
        calls.append(idx.id)
        raise RuntimeError("permanent")

    backend = _make_backend(f)
    with pytest.raises(RuntimeError, match="5 tries to download the tile failed") as excinfo:
        backend._download_from_source_with_retries(_make_idx("p1"))

    assert len(calls) == 5
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert str(excinfo.value.__cause__) == "permanent"


def test_retries_propagates_keyboard_interrupt_immediately():
    calls = []

    def f(idx):
        calls.append(idx.id)
        raise KeyboardInterrupt()

    backend = _make_backend(f)
    with pytest.raises(KeyboardInterrupt):
        backend._download_from_source_with_retries(_make_idx("p1"))

    assert len(calls) == 1  # not retried


# --- _write_patch_variable: NaN-border preservation ---------------------------------


def test_write_patch_variable_preserves_existing_data_under_nan_borders(make_fake_accessor):
    accessor = make_fake_accessor()
    accessor.create()

    session = accessor.repo.writable_session("main")
    zcube = _open_writable_zcube(session)
    shape = _array(zcube, "value").shape

    baseline = np.full(shape, 7.0, dtype="float32")
    _array(zcube, "value")[:] = baseline
    session.commit("seed baseline")

    session = accessor.repo.writable_session("main")
    zcube = _open_writable_zcube(session)
    new_data = np.full(shape, 42.0, dtype="float32")
    new_data[0, :] = np.nan  # top row: simulate a tile with a NaN border
    new_data[:, 0] = np.nan  # left column: simulate a tile with a NaN border

    target = (slice(0, shape[0]), slice(0, shape[1]))
    accessor.backend._write_patch_variable(zcube, new_data, "value", target, "test-patch")

    result = _read_all(_array(zcube, "value"))
    assert_array_equal(result[0, :], 7.0)  # NaN border: baseline preserved
    assert_array_equal(result[:, 0], 7.0)  # NaN border: baseline preserved
    assert_array_equal(result[1:, 1:], 42.0)  # interior: overwritten


# --- ThreadedBackend: crash handling and lifecycle ----------------------------------


def test_threaded_backend_write_failure_surfaces_and_does_not_mark_loaded(make_fake_accessor):
    accessor = make_fake_accessor(backend="threaded")
    accessor.create()

    # A patch with an unknown data variable: writing it fails inside the writing_pool.
    bad_var = odc.geo.xr.xr_zeros(accessor.extent, dtype="float32", always_yx=True)
    bad_patch = xr.Dataset({"not_a_real_channel": bad_var})
    bad_patch.attrs["patch_id"] = "bad-patch"

    with pytest.raises(RuntimeError, match="failed"):
        accessor.backend._write_patch(bad_patch)

    assert accessor.loaded_patches() == []


def test_threaded_backend_close_shuts_down_cleanly(make_fake_accessor, monkeypatch):
    # close() shuts the write queue down to unblock the writer thread's pending
    # get(); _writer() must catch the resulting queue.ShutDown itself rather than
    # let it escape as an unhandled exception in that thread (regression test for
    # a bug where this wasn't the case).
    thread_exceptions = []
    monkeypatch.setattr(threading, "excepthook", thread_exceptions.append)

    accessor = make_fake_accessor(backend="threaded")
    accessor.create()
    accessor.procedural_download(accessor.extent, None)

    assert accessor.backend.close() is True
    assert not accessor.backend.writer.is_alive()
    assert accessor.backend.write_queue.is_shutdown
    assert thread_exceptions == []
