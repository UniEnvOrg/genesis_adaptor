# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- **Amazon Berkeley Objects (ABO)** rigid object family: all 7,953 modelled Amazon.com products (CC BY 4.0 — Amazon.com relicensed ABO from CC BY-NC 4.0 in mid-2023) as `ABOObject` / `get_abo_catalog` / `list_abo_objects` / `make_abo_pool` under `objects/abo/<3dmodel_id>/`, with `scripts/prepare_abo_assets.py` as the packager (downloads the models CSV + listings shards, joins `3dmodel_id`, bakes the Y-up → Z-up rotation into `model.obj` so the trimesh bounding box and the placed frame agree, renames the OBJ texture to `texture.png` and rewrites the `.mtl`/`.obj` references, generates the broad → detailed language annotations from `item_name`/`bullet_point`, writes the provenance `README.md` + verbatim CC BY 4.0 `LICENSE` and prints the `hf upload` line). Collision comes from Genesis's own `convexify` at load time (the dataset ships no decomposition); `box_gripper_graspable` is baked into `scripts/data/abo_graspable.json` — computed inline from the converted meshes and merged by every run, so an incremental `--objects`/`--limit` run extends the bake and the full run completes it (`scripts/compute_graspable.py abo` re-bakes it from a prepared tree).

### Changed

- Relaxed the `genesis-world` pin to `>=1.4` (verified 1.4.3 in-tree: no ABD/numerics change — the pinned regression probe was byte-identical source, bit-identical results). Genesis 1.4.3 no longer installs the compute device as torch's *ambient* default from `gs.init()` (upstream #3431), so tensors built without an explicit `device` now land on CPU on a GPU run. Device audit of the collection: `base_object`/`room_builder`/both clutter managers/tasks/`EntityGroup` already pin or inherit devices correctly and were left alone; the one genuine hazard fixed is `TableBuilder`, whose `size`/`height`/anchor-pose tensors inherited the ambient default and feed `PickPlaceTask.sample_target_placement` (CUDA `xy` · CPU `placement_half_xy` device-mismatch on a GPU world) — they are now pinned to `world.device` like every other world-frame quantity. One follow-on hardening from the same change: the offline FK walker installs genesis's own `gs.EPS` fallback (`gs.init(precision="32")`'s exact value) before calling `gu.polar`, whose numba kernel reads `gs.EPS` as a compile-time global — `EPS` is `None` until `gs.init`, and whether that crashes (`float64 > none` `TypingError`) or not used to depend on numba's *disk cache* being warm from a previous genesis install (a reinstall invalidates it, which is how the 1.4.3 upgrade surfaced this). Both compat patches verified still required and still applying under 1.4.3; the full scoped suite, the robot/controller suites and the batched/CUDA suites (725 tests) pass under 1.4.3, and a CUDA sanity-gate smoke (`refrigerator_fridge` through `_gate_worker.py` with `genesis_compat.install()`) loads + gates + renders on the 4090 with offline-FK vs `vAABB` Δ ≤ 6e-8 m.

### Added

- Three new USD asset families in the articulated rooms pack, following the ArtVIP integration: **MolmoSpaces** (Ai2-THOR subset, CC BY 4.0 — the packaged stages carry the mandatory joint-body0/body1 swap), **Lightwheel SimReady** open bundle (CC BY-NC 4.0, maintainer-accepted — packaged stages carry the OmniPBR MDL → UsdPreviewSurface material rewrite) and **NVIDIA SimReady Warehouse** rigid props (CC BY 4.0, `RigidBodyAPI` authored at packaging, metersPerUnit normalised to 1.0). Each ships a descriptor (`MolmoSpacesFurniture` / `LightwheelFurniture` / `SimReadyWarehouseProp`) with catalog/list/pool helpers, a `scripts/prepare_*_assets.py` packaging pipeline with a Genesis sanity gate, offline converter tests and GPU smokes; the shared pxr converters live in `unienv_genesis_collection/utils/usd_conversions.py` (joint-body swap + MDL → UsdPreviewSurface) and `RoomBuilder._resolve_descriptor` resolves the `molmospaces` / `lightwheel` / `simready_warehouse` sources.

### Fixed

- The USD offline-FK walker raised `KeyError` for stages whose welded links carry `FixedJoint` prims (the reference-`qpos` lookup ran before the moving-joint branch); fixed-joint welds now skip the displacement lookup and keep their relative-offset composition.
- The offline USD FK walker (`usd_link_meshes_at_config`) mirrored Genesis's USD build in the two places that still diverged from it: **frames are now RIGID** — each link's authored world transform with its linear scale polar-decomposed out (`compute_gs_transform`/`extract_scale`), the child's q = 0 pose its authored rigid relative pose `inv(Q_parent) @ Q_child`, and the DOF displaced about the **body1-side anchor/axis** with `localPos1`/`localRot1` mapped through the body-relationship target prim's authored world matrix — instead of composing joint-frame chains inside scale-carrying ancestors (SUNDVIK ×1000, table_1/drawers_chest ×0.03937, door001's non-uniform scope) and substituting `R(localRot0)` for the authored link rotation. Recovers ArtVIP SUNDVIK_wardrobe…172cm, table_1, drawers_chest_of_drawers_5, floor_lamp002, desk_lamp009, standing_fan_big_electric_fan_8, door001, toolbox_clamshell__tool_box_8, built-in_oven_oven_7/8 at the gate (verified live, closed/opened Δ ≤ 2 mm vs 0.03–0.33 m tolerances). A metersPerUnit ≠ 1 stage cannot lose or leak units through the split (a positive uniform factor lands in the scale, never in the rigid part) and `link_local=True` now returns the rigid runtime link frame — the same frame `get_link_pose` maps back to the world — with the authored scale riding in the geometry, so part-grasp candidates no longer double-apply it.
- The offline USD FK walker now applies Genesis's *visibility gate*: drawables whose computed visibility resolves to `invisible` are excluded from both the visual box and the collision box (`genesis/utils/usd/usd_geometry.parse_prim_geoms`'s `is_visible`), and a link whose drawables are all invisible is omitted (no Genesis geometry). Recovers Lightwheel CoffeeMachine039 + Stovetop002/005/006/009/015/022/026 and ArtVIP clock001 / small_instrument_centrifuge001, whose authored-invisible `Sites/*` burner/place/coffee gprims used to widen the offline box.
- The ArtVIP packaging fixups gained two repairs (applied to the packaged tree in place for the affected ids): microwave_2's collider `/root/E_base_plate_1/P_d713ed04dd5c91` had its triangle winding **reversed** — a watertight shell wound inside-out makes the SDF wall-thickness probe fire every ray outward (`IndexError` at `get_wall_thickness`); the predicate reproduces that exact failure offline (welded-watertight + consistent winding + negative signed volume + **zero-hit probe**) so the 40+ benign negative-volume colliders (open/convexified or locally-flipped loads, e.g. refrigerator_fridge) are never touched; and world-welded **orphan joints** (empty `body0` on a non-root body — oven_1's door hinge, table_dressing_table004's mirror weld, which split the stage into 2 entities) are **re-rooted onto the main articulation root** with `Body0Rel` re-pointed and `localPos0`/`localRot0` re-expressed so the joint frame keeps its authored world placement (oven_1/microwave_2 now gate at Δ ≤ 1e-5 m, table_dressing_table004 at 2.3e-3 vs 0.216 tol).
- A sanity-gate batch no longer dies because one object id is unknown: the per-object descriptor construction is now inside the gate's drop guard in all four family gates (ArtVIP/Lightwheel/MolmoSpaces/SimReady), so an id missing from the packaged catalog (which lists only survivors) is recorded as that object's drop record instead of killing the whole worker — and with it every other object's record.
- The Lightwheel Locomotion packaging dropped 4 of 8 scenes: the MDL → UsdPreviewSurface rewrite re-authored material graphs living in *nested payload layers* into the root layer without re-anchoring their texture paths (so the packaged tree demanded `textures/…` beside the root instead of inside the payload directory — MudGround/SnowGround/GravelGround/KitchenRoom); textures are now resolved against the authoring layer and rewritten relative to the root layer, prefixed-but-shipped texture paths are repaired (KitchenRoom's `3d66Model-19913701-files-024.JPG` vs the shipped `19913701-files-024.JPG`), and the static family's offline bbox walks the point-accurate world bounds of the geometry `add_stage` instantiates (Grass lost 0.14 m in Z, the 2-storey staircase 20 m in X/Y to the `BBoxCache` extent-box proxy). The descriptors no longer demand the warn-and-recorded UE material records, which re-triggered a Hub download during the sanity gate (Apartment).

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