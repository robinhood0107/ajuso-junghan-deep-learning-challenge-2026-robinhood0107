# Changelog

All notable changes to this project are documented in this file.

## [0.1.1.0] - 2026-08-17

### Changed

- Require all five fixed-base OOF folds to share one source, configuration, B0, and GPU-device provenance before qualification.
- Revalidate every fold's run files, manifest, and execution inputs before a base selection can be frozen or used for holdout evaluation.
- Keep legacy base OOF v1 evidence readable while requiring regenerated v2 evidence for any new selection freeze.
