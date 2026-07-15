# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

## v0.1.6

- Added a comprehensive unit test suite covering accessors, backends, time-of-interest, and utilities, along with a CI workflow to run it.
- Fixed `STACAccessor` crashing with an `UnboundLocalError` when downloading without a time-of-interest, and removed a duplicate, redundant STAC search call.
- Fixed `ThreadedBackend` silently swallowing exceptions from failed patch writes instead of raising them, and fixed its writer thread crashing instead of shutting down cleanly when the write queue closes.
- Added backend documentation.
- Fixed `ty` errors in the quickstart notebook.

## v0.1.5

- Fixed leftover `ty` type errors.
- Added architecture and custom-accessor docs, a Code of Conduct, and this Changelog.
- Expanded test coverage for antimeridian handling, ArcticDEM, and TCTrend.

## v0.1.4

- Added GeoDataFrame support to `procedural_download`, including GeoDataFrame-aware interest logging.

## v0.1.3

- Fixed ArcticDEM tests.
- Tightened type checks with `ty` and refreshed the release with broader backend, dataset, and test updates.

## v0.1.2

- Updated the quickstart and contribute docs, including new backend Gantt figures.
- Improved backend handling and fixed the patch-id bookkeeping for loaded patches.

## v0.1.1

- Fixed the lockfile after the v0.1.0 release.

## v0.1.0

- Major refactor to the new core layer, introducing dedicated accessor, backend, storage, patch, and time-of-interest modules.
- Split execution into simple and threaded backends and moved dataset accessors onto the new geometry-based interfaces.
- Updated dependencies, tests, and release scaffolding to match the refactor.

## v0.0.10

- Added temporal datacube support and the initial AlphaEarth dataset plumbing.
- Fixed temporal bugs and relaxed dependency constraints.

## v0.0.9

- Added TCTrend 2020 and 2022 coverage.
- Added the project citation metadata.

## v0.0.8

- Improved download flow by fetching auxiliary data after initialization and adding pyarrow as a dependency.
- Refined download and geobox representations and fixed ArcticDEM adjacent tile bookkeeping.
- This release is tagged as both v0.0.8 and v0.0.7.

## v0.0.6

- Fixed CRS parsing.

## v0.0.5

- Added easier access to loaded tiles.

## v0.0.4

- Fixed an outdated argument in the GEE accessor.

## v0.0.3

- Added download support for GeoPandas dataframes.
- Fixed falsey parsing of the 2m ArcticDEM STAC URL and updated related docs and tests.

## v0.0.2

- Revamped the documentation and made download behavior more efficient.
- Added the docs site assets, quickstart notebook, and updated project presentation files.

## v0.0.1

- Added the API reference and GitHub workflow for version publishing.
- Established the initial docs, package metadata, and core accessor/dataset structure.
