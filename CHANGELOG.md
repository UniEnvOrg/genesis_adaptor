# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1b1 (2026-07-22)

### Added

- Tianji Marvin dual-arm robot with Wuji dexterous hands
- GSO and YCB object sets with hub-based pools, labels, and filtering
- Environment map manager for HDRI lighting randomization, plus new tabletop tasks
- Various new sensors and a restructured keypoint tracker node
- Working pick-and-place reference policy
- Deterministic seeding across worlds, tasks, scene managers, and reference policies: optional `seed` constructor arguments, per-node generators reseeded from lifecycle seeds, reproducible YCB/GSO pool sampling via caller-provided generators, and seeded mesh grasp-candidate sampling through trimesh's `sample_surface(seed=...)`

### Changed

- Collection assets are now downloaded on demand from Hugging Face (`UniEnvOrg/unienv_genesis_collection_assets`) instead of being shipped with the package
- Updated asset acquiring for the Franka FR3 + Robotiq 2F85 robot
- Updated Contact-GraspNet policy; reference policy speedups
- Interface changes and a concrete wrapper implementation
- Removed redundant lifecycle priorities
- Updated Tianji Marvin + Wuji hand default rest pose

### Fixed

- Device-mismatch errors in RNG call sites (`randperm`/`randint`) when running under a CUDA default device

## 0.0.1a1 (2026-06-08)

- Initial release