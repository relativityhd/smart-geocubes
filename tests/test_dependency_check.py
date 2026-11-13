"""Tests for the optional-dependency (extras) check of the accessors."""

import importlib.util

import pytest

import smart_geocubes
from smart_geocubes.core import utils

# (accessor class, extra, a module that belongs to that extra)
CASES = [
    (smart_geocubes.ArcticDEM32m, "stac", "pystac_client"),
    (smart_geocubes.AlphaEarthEmbeddings, "gee", "ee"),
    (smart_geocubes.TCTrend2024, "gee", "xee"),
]


@pytest.fixture
def hide_module(monkeypatch):
    # Make `find_spec` report the given module as not installed, leaving all others untouched
    def _hide(name: str):
        real_find_spec = importlib.util.find_spec

        def fake_find_spec(module, *args, **kwargs):
            if module == name:
                return None
            return real_find_spec(module, *args, **kwargs)

        monkeypatch.setattr(utils.importlib.util, "find_spec", fake_find_spec)

    return _hide


@pytest.mark.parametrize(("accessor_cls", "extra", "module"), CASES)
def test_missing_extra_raises_before_side_effects(tmp_path, hide_module, accessor_cls, extra, module):
    hide_module(module)
    store = tmp_path / "cube.icechunk"

    with pytest.raises(ModuleNotFoundError, match=rf"requires the '{extra}' extra") as excinfo:
        accessor_cls(store, backend="simple")

    assert excinfo.value.name == module
    assert accessor_cls.__name__ in str(excinfo.value)
    assert f"smart-geocubes[{extra}]" in str(excinfo.value)
    # The check must run before the icechunk repository is created
    assert not store.exists()


@pytest.mark.parametrize(("accessor_cls", "extra", "module"), CASES)
def test_installed_extra_creates_accessor(tmp_path, accessor_cls, extra, module):
    # Sanity counterpart to the test above: with all extras installed, construction works and does
    # create the repository, so the `not store.exists()` assertion above is actually meaningful.
    store = tmp_path / "cube.icechunk"
    accessor_cls(store, backend="simple")
    assert store.exists()


def test_require_extra_lists_all_missing_modules():
    with pytest.raises(ModuleNotFoundError) as excinfo:
        utils._require_extra("stac", ["pystac", "not_a_module_a", "not_a_module_b"], "SomeAccessor")

    msg = str(excinfo.value)
    assert "missing: not_a_module_a, not_a_module_b" in msg
    assert "pystac," not in msg
    assert excinfo.value.name == "not_a_module_a"


def test_require_extra_passes_when_installed():
    utils._require_extra("stac", ["pystac"], "SomeAccessor")
