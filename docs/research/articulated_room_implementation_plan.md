# Implementation Plan: Procedurally Generated Rooms with Articulated Furniture

**Target repo:** `genesis_adaptor` (`unienv_genesis_collection/`, `unienv_genesis/`, `scripts/`)
**Asset sources:** RoboCasa (CC BY 4.0, MJCF) · ReplicaCAD (CC BY 4.0, URDF+GLB)
**Genesis version pinned:** 1.1.2 — all APIs below were verified against the installed package, not guessed.

This plan was written after reading the actual code. References below are to real symbols:

- `unienv_genesis_collection/objects/base_object.py` — `ManipulatableObject` (ABC, `Generic[EntityT]`), `make_bbox`, `GraspCandidates`, `_local_to_world_candidates`
- `objects/mesh_object.py` — `MeshObject` (trimesh lazy load, antipodal grasp sampling)
- `objects/hub_objects/{hub_asset_object,gso_object,ycb_object,catalog}.py` — `HubAssetMeshObject`, `xml_file_references`, `ObjectInfo`, `load_catalog`, `GSOObject.load_into_scene` (`gs.morphs.MJCF`), `YCBObject.load_into_scene` (`gs.morphs.URDF`)
- `scenes/table_builder.py` — `TableBuilder(WorldNode)`, `reload_priorities = {100}`, `EntityGroup`, seedable `random.Random`
- `tasks/tabletop/pickplace.py` — `ObjectRelocateTask` lifecycle convention (priorities = `max(manager.priorities)+1`, `reload` loads at `_PARK_POS`, `reset` calls `set_pos_3d`, reward/termination in `post_environment_step`)
- `unienv_genesis/nodes/entities/entity_controllers/base.py:27` — `resolve_joint_indexes(entity, joint_indexes=None, joint_names=None)`: stateless name→DOF-index resolution off the built entity (the pattern this plan adopts for furniture)
- `unienv_genesis/nodes/entities/robot_node.py:223` — `_apply_rest` resolving `RobotKeyframe.joint_names` via `resolve_joint_indexes` then `entity.set_dofs_position(values, dofs_idx_local=resolved)`
- `scripts/prepare_{gso,ycb}_assets.py` — repackage → upload pattern

---

## Design decision log

- **Joint registry: considered and rejected.** An earlier draft of this plan proposed a per-asset *semantic joint registry* (`JointInfo` records with `q_min`/`q_max`/`child_link`/grasp parts) baked into `catalog.json` at packaging time. The user reviewed the existing robot actor and rejected it: robots in this codebase maintain **no registry** — they read joint names off the built entity at runtime via the stateless helper `resolve_joint_indexes` (`entity_controllers/base.py:27`, ~20 lines: builds `{j.name: j.dofs_idx_local for j in entity.joints}` on the fly, errors listing available names) and apply keyframes by name through it (`robot_node.py:243`, `_apply_rest`). **This plan adopts the same pattern:** joint and link addressing on `ArticulatedObject` is **by name, resolved at runtime against the built entity**; nothing per-asset is hardcoded or baked into `catalog.json`.
- **Open fraction derives limits at runtime.** `q_min`/`q_max` are not baked per-asset; they are read from the entity via Genesis `entity.get_dofs_limit(dofs_idx_local)` (returns `(lower, upper)`, batched) inside the open-fraction helper (§1.2).
- **Union bbox is offline-FK.** The multi-link reference-configuration AABB is computed offline (no Genesis scene) via `genesis.ext.urdfpy` FK for URDF, with a runtime entity-based cross-check as a packaging-time gate (§1.3). Descriptors stay offline-safe; the runtime entity is *never* required for `get_local_bounding_box()`.
- **Part grasps resolve links by name at runtime.** `list_part_grasping_candidates` takes a link *name* (optionally through a caller-passed alias dict like `{"handle": "<link_name>"}` — a plain runtime argument, not baked metadata) and resolves the link index off the built entity, mirroring `resolve_joint_indexes`.

---

## 0. Gap analysis (what exists vs. what is missing)

| Capability | Status today | Needed for articulated furniture |
|---|---|---|
| Load articulated URDF/MJCF | ✅ works (`gs.morphs.URDF/MJCF` accept articulated files; `robot_node.py` proves it) | reuse |
| Base-link placement | ✅ `set_pos_3d` / `entity.set_pos` | furniture: place **once at load** via morph `pos`/`quat` |
| `get_local_bounding_box` | ❌ single visual mesh (`MeshObject._load_trimesh`) | union over **all links** at a reference joint configuration |
| Grasp candidates | ❌ bbox/antipodal on one mesh, object frame | **part-aware**: candidates on a named link (e.g. a handle link), transformed by that link's pose |
| Joint name→DOF resolution | ✅ `resolve_joint_indexes` (entity controllers) | reuse for furniture (see §1.2 import note) |
| Open/close fraction semantics | ❌ (robots don't need it) | thin helper on top of `resolve_joint_indexes` + `get_dofs_limit` |
| Per-link poses | ❌ in object descriptors | `entity.get_links_pos/get_links_quat(links_idx_local, envs_idx, relative=True)` |
| Rooms | ❌ table only | `RoomBuilder` WorldNode + layout sources |
| Asset pipelines | ✅ GSO/YCB | `prepare_robocasa_assets.py`, `prepare_replicacad_assets.py` |

### Verified Genesis 1.1.2 facts the design builds on

- `gs.JOINT_TYPE`: `FIXED=0, REVOLUTE=1, PRISMATIC=2, SPHERICAL=3, FREE=4` (exported top-level).
- `RigidJoint` exposes `.name`, `.idx_local`, `.type`, `.n_dofs`, `.dofs_idx_local`, `.link` (child `RigidLink`, which has `.name`/`.idx_local`). `entity.joints`, `entity.links` are plain lists; `entity.get_joint(name=...)` raises with the list of available names.
- Joint state: `set_dofs_position(position, dofs_idx_local, envs_idx, zero_velocity=…)`, `get_dofs_position(...)` — batched (`(n_dofs,)` or `(B, n_dofs)`); `get_dofs_limit(dofs_idx_local, envs_idx)` returns `(lower_limit, upper_limit)` of shape `(n_dofs,)` or `(B, n_dofs)`. DoF state is **always batched per env**, even for fixed-base entities — per-env drawer positions in batched worlds work with no morph options.
- Base link: `get_pos()/get_quat()/set_pos()/set_quat()` operate on the base link; with `fixed=True` (URDF) the base is a 0-DoF weld but articulated joints still move.
- **MJCF has no `fixed=` morph flag** — the XML decides. A jointless root body is welded to the world (exactly the fixed-GSO `model.xml` trick in `gso_object.py`). So fixed RoboCasa fixtures = MJCF *without* `<freejoint/>`; a movable variant (if ever wanted) = injected `<freejoint/>`, mirroring `prepare_gso_assets.py`.
- `genesis/utils/mjcf.py` **absolutizes** `compiler` `assetdir`/`meshdir`/`texturedir` relative to the MJCF file and resolves `<include>` recursively — a repackaged self-contained MJCF dir loads regardless of original `meshdir` layout.
- URDF parsing goes through **`genesis.ext.urdfpy`** (vendored; plain `pip install urdfpy` is *not* present in the env). `genesis.ext.urdfpy.URDF.link_fk` is available → offline forward kinematics for the reference-configuration union bbox without instantiating Genesis.
- `merge_fixed_links=True` is the URDF default; `ycb_object.py` sets it `False` with the comment that "Nyx currently cannot export URDF entities after Genesis merges fixed links". Articulated furniture must also pass `merge_fixed_links=False` (or `links_to_keep`) so named handle/door links survive for part-aware queries.
- `entity.get_vAABB(envs_idx)` → `(2,3)` / `(B,2,3)` world AABB aggregated from visual geoms — a cheap **runtime** cross-check for the offline union bbox (requires `n_vgeoms > 0`).
- `batch_fixed_verts`: batches fixed-geometry vertex buffers per env so `set_pos` can take env-specific poses on fixed entities; significant memory cost. For rooms this is the enabling switch for **per-env layout diversity** in batched worlds.
- MJCF anisotropic scale is rejected (isotropic only) — same constraint GSO already encodes.
- `coacd` and `trimesh 4.12` are installed in the env; `pygltflib` present (GLB meshes load).

---

## 1. `ArticulatedObject` abstraction

### 1.1 Placement in the class hierarchy

**Decision: sibling to `MeshObject` under `ManipulatableObject`, not a subclass of it.**

Rationale:
- `MeshObject` is built around *one* `mesh_file` (`self.mesh_file`, `_load_trimesh`) used for bbox + grasp sampling. Articulated furniture has one visual mesh **per link**; subclassing would leave a meaningless `mesh_file` and force overriding nearly every method.
- `HubAssetMeshObject` additionally bakes in `mesh_file_name`/`visual_texture_file_name` — same problem. Its *machinery* (`_ensure_assets`, `_required_files`, `xml_file_references`) is reused in a thin new hub class (§1.4), not by inheriting `MeshObject` geometry.
- `get_local_bounding_box` is overridden to return the **multi-link union AABB at the reference configuration**, in the base-link local frame. The base class contract (`get_bounding_box(entity)` transforms the local AABB by the base-link pose, `base_object.py:343`) then keeps working verbatim — the docstring's "reflect runtime state (e.g. articulated joints)" hook is honored by the *reference-configuration* convention plus explicit per-link queries for the dynamic part.
- The bbox-grasp default (`list_box_gripper_grasping_candidates_local`, `base_object.py:590`) is **kept as a whole-object fallback** (uses the union bbox), but the primary grasp path for furniture is the new part-aware API (§1.2).
- **No joint registry.** Consistent with the robot actor, the descriptor stores *no* joint metadata. The only constructor state beyond `ManipulatableObject`'s is the per-link visual meshes (loaded lazily for the offline bbox) and an optional `reference_qpos` mapping joint names → positions (defaults to all-zero, i.e. the asset's authored closed pose; joint names here are only FK dict keys, not a registry).

### 1.2 New module: `unienv_genesis_collection/objects/articulated_object.py`

```python
class ArticulationError(RuntimeError): ...


# ---- stateless runtime resolvers (mirror entity_controllers/base.py:27) ----

def resolve_joint_dofs(entity, joint_name: str) -> List[int]:
    """dofs_idx_local for a named joint, resolved off the built entity.

    Same on-the-fly map as `resolve_joint_indexes`
    (`{j.name: j.dofs_idx_local for j in entity.joints}`); single-joint
    convenience raising ArticulationError with the available joint names.
    """

def resolve_link_index(entity, link_name: str) -> int:
    """idx_local for a named link, resolved off the built entity.

    Link-level analogue of `resolve_joint_indexes` (robots never needed it):
    `{l.name: l.idx_local for l in entity.links}`; errors list available
    link names.
    """


class ArticulatedObject(ManipulatableObject[EntityT]):
    """Descriptor for articulated furniture (cabinets, drawers, appliances).

    - fixed by default (furniture is placed once at load; `set_pos_3d` is
      overridden only to default `random_quaternion=False` and document the
      `batch_fixed_verts=True` requirement for fixed entities in batched
      worlds).
    - `get_local_bounding_box()` returns the union AABB over all links at the
      reference configuration (joints at `reference_qpos`, default all-zero),
      in the base-link frame. Computed OFFLINE (§1.3) — no Genesis scene
      required, matching the descriptor contract used by tasks
      (pickplace.py:141 calls it right after reload).
    - Joint/link addressing is BY NAME against the built entity via the
      stateless resolvers above — no registry, no baked metadata.
    """

    def __init__(
        self,
        name: str = "articulated_object",
        fixed: bool = True,
        reference_qpos: Optional[Mapping[str, float]] = None,  # FK keys only
        language_annotations: Optional[Sequence[str]] = None,
    ): ...

    # ---- per-link geometry (offline; subclass provides meshes) ----
    def _load_link_meshes_at_reference(self) -> Dict[str, "trimesh.Trimesh"]:
        """link name -> visual mesh transformed into the base-link frame at
        the reference configuration. Implemented by the URDF and MJCF hub
        classes via their respective FK helpers (§1.3)."""

    def get_link_local_bounding_box(self, link_name) -> BoxSpace
        # trimesh bbox of that link's mesh in the BASE frame at reference
        # configuration (cached per link name).
    def get_local_bounding_box(self) -> BoxSpace
        # union of get_link_local_bounding_box over all links (cached).

    # ---- joint state (runtime, entity-backed; thin pass-throughs by name) ----
    def get_dofs_position(self, entity, joint_name, envs_idx=None) -> torch.Tensor
        # resolve_joint_dofs(entity, joint_name) -> entity.get_dofs_position(...)
    def set_dofs_position(self, entity, position, joint_name,
                          envs_idx=None, zero_velocity=True) -> None

    # ---- open-fraction helper (the one thing robots don't provide) ----
    def get_open_fraction(self, entity, joint_name, envs_idx=None) -> torch.Tensor
        """(q - q_min) / (q_max - q_min) clamped to [0, 1].

        q_min/q_max come from `entity.get_dofs_limit(dofs_idx_local)` at
        runtime — NOT from baked metadata. Hinge (rad) vs slide (m) is
        uniform in fraction space. Raises ArticulationError if the joint's
        limits are ±inf (unbounded joint has no meaningful fraction).
        """
    def set_open_fraction(self, entity, joint_name, fraction,
                          envs_idx=None, zero_velocity=True) -> None
        # fraction clamped to [0, 1] -> q = q_min + f * (q_max - q_min),
        # applied via set_dofs_position by name.

    # ---- per-link runtime pose ----
    def get_link_pose(self, entity, link_name, envs_idx=None
                      ) -> Tuple[torch.Tensor, torch.Tensor]
        # resolve_link_index + entity.get_links_pos/get_links_quat(
        #     links_idx_local, relative=True)

    # ---- part-aware grasps ----
    def list_part_grasping_candidates(
        self, entity, link_name: str, gripper_width: float,
        gripper_depth: float = 0.01,
        part_aliases: Optional[Mapping[str, str]] = None,  # e.g. {"handle": "<link_name>"}
        rng=None,
    ) -> GraspCandidates
        """World-frame candidates on a named link.

        `link_name` is resolved against the entity at runtime (optionally
        through the caller-passed `part_aliases` dict — a plain runtime
        argument, NOT baked per-asset metadata). Candidates are the bbox
        enumeration of that link's mesh (same algorithm as
        ManipulatableObject.list_box_gripper_grasping_candidates_local,
        run on the link-local bbox; cached per (link, width, depth)), then
        transformed by the link's CURRENT pose via get_link_pose — so they
        track a drawer as it slides. Unknown names raise ArticulationError
        listing available links (fail loud; callers wanting whole-object
        grasps use the inherited bbox method).
        """

    # ---- placement ----
    def set_pos_3d(self, entity, positions, random_quaternion=False, rng=None,
                   envs_idx=None) -> None
        # delegates to super(); only the default differs (furniture keeps its
        # authored yaw).
```

**Reconciliation with the existing API** (explicit):

- `get_bounding_box(entity)`, `get_footprint*`, `is_box_gripper_graspable`, `list_box_gripper_grasping_candidates*` — inherited unchanged; they consume the overridden `get_local_bounding_box()`. Batched handling (`(B, 3)` poses) flows through unchanged.
- The `_local_grasp_cache` pattern (dict keyed by gripper params) is reused per-link: cache key gains the link name.
- `language_annotations`, `name`, catalog conventions — unchanged; **`catalog.json` gains no articulation block** (see decision log).

**Import note:** `resolve_joint_indexes` lives in `unienv_genesis/nodes/entities/entity_controllers/base.py`, importable from `unienv_genesis_collection` (the dependency direction is already collection→unienv_genesis — e.g. `hub_asset_object.py` imports `unienv_genesis.world` for typing, `mesh_object.py` imports `genesis.engine.entities`). The module either (a) imports it directly, or (b) if layering review objects, inlines the ~10-line name-map logic in `resolve_joint_dofs`/`resolve_link_index` above. Decide at implementation time; both keep behavior identical.

### 1.3 Offline reference-configuration union bbox

**Decision: compute offline via FK, never from the runtime entity.**

The union bbox must be available **without a Genesis scene** (descriptors are offline-safe; `get_local_bounding_box` is called by tasks before any step — see `pickplace.py:141`). Since the articulation catalog block is gone, there is no baked bbox either — offline FK at descriptor-load time:

- **URDF (ReplicaCAD):** parse with `genesis.ext.urdfpy.URDF.load`, call `robot.link_fk(cfg=reference_qpos)` → per-link 4×4 transforms at the reference configuration; load each link's visual mesh with `trimesh.load(force="mesh")` (GLB supported), transform vertices, take per-link and global min/max → `make_bbox`. Implemented as `urdf_link_meshes_at_config(urdf_file, reference_qpos) -> Dict[str, trimesh.Trimesh]` in the new module.
- **MJCF (RoboCasa):** Genesis does not vendored an MJCF FK API, so implement a minimal FK walker in the module (`mjcf_link_meshes_at_config`): parse with `xml.etree.ElementTree` (already the project's convention via `xml_file_references`), walk `worldbody` nesting applying `pos`/`quat`/`euler`, apply hinge (`<joint type="hinge" axis=…>`) rotation and slide translation at the reference q, gather `<geom type="mesh" mesh=…>` + `<asset><mesh file=…>`. RoboCasa fixtures use simple hinge/slide chains, so this stays small (~150 lines).
- **Runtime cross-check (packaging-time gate, not a data source):** the prepare pipelines (§2) compare each repackaged asset's offline union bbox against `entity.get_vAABB()` (transformed to base frame) at q=0 within tolerance. This catches FK-parser bugs per asset while Genesis is available, so parser gaps are found *before assets ship* — no catalog fallback needed. The same check runs as a test helper (`assert_bbox_matches_entity`) in the smoke tests.

### 1.4 Hub-backed concrete classes

New subpackage `unienv_genesis_collection/objects/hub_objects/articulated/`:

```
hub_objects/articulated/
  __init__.py
  hub_articulated_object.py     # HubAssetArticulatedObject
  robocasa_fixture.py           # RoboCasaFixture  (MJCF)
  replicacad_furniture.py       # ReplicaCADFurniture (URDF)
```

```python
class HubAssetArticulatedObject(ArticulatedObject):
    """ArticulatedObject backed by a per-object directory on the Hub.

    Reimplements the HubAssetMeshObject machinery (asset_relative_dir /
    _required_files / _ensure_assets / ensure_assets) because the parent
    class's single-mesh assumptions don't apply. `xml_file_references` is
    reused as-is for both `file=` (MJCF) and `filename=` (URDF).
    """
    def __init__(self, asset_relative_dir, description_file_name, **kw): ...
    def _required_files(self) -> List[Path]
        # description file + every mesh/texture it references (recursive for
        # MJCF includes), mirroring GSOObject._required_files.
```

```python
class RoboCasaFixture(HubAssetArticulatedObject):
    """RoboCasa kitchen fixture (cabinet/drawer/appliance) via gs.morphs.MJCF.

    Assets: objects/robocasa/<fixture_id>/ with model.xml (jointless root =
    welded base, fixed) — mirrors GSOObject's fixed path.  Joint/link names
    are whatever the MJCF declares; they are discovered at runtime off the
    built entity (resolve_joint_dofs/resolve_link_index), never baked.
    """
    def _load_link_meshes_at_reference(self): ...  # mjcf_link_meshes_at_config
    def load_into_scene(self, world, pos, quat, name_override=None,
                        batch_fixed_verts=False):
        # gs.morphs.MJCF(file=..., scale=..., pos, quat, align=False,
        #                batch_fixed_verts=..., requires_jac_and_IK=False)
        # — identical options to GSOObject.load_into_scene.
```

```python
class ReplicaCADFurniture(HubAssetArticulatedObject):
    """ReplicaCAD articulated furniture via gs.morphs.URDF.

    Loads with fixed=self.fixed, align=False, merge_fixed_links=False (named
    door/drawer links must survive for part queries — same rationale as the
    YCB comment about link merging), batch_fixed_verts=batch_fixed_verts.
    """
    def _load_link_meshes_at_reference(self): ...  # urdf_link_meshes_at_config
```

Each module mirrors the GSO/YCB file layout: `<DATASET>_ASSETS_ROOT`, `<DATASET>_CATALOG_PATH`, `get_<ds>_catalog()`, `list_<ds>_objects(tags=...)`, `make_<ds>_pool(...)`. Catalog entries carry only the existing fields (`language_annotations`, `tags`, optional `standing_quaternion`) — tags gain values like `"furniture"`, `"articulated"`, `"kitchen"` for pool filtering; **no joint/part metadata**.

---

## 2. Asset ingestion pipelines

Both follow the proven `prepare_{gso,ycb}_assets.py` shape: clone/download upstream → repackage into self-contained per-object dirs → write `catalog.json` + `LICENSE` + provenance `README.md` → print the `hf upload UniEnvOrg/unienv_genesis_collection_assets … --repo-type dataset` command.

### 2.1 `scripts/prepare_robocasa_assets.py`

- **Source:** `python -m robocasa.scripts.download_kitchen_assets` output, or HF `robocasa/robocasa-assets`; script takes `--source` like the existing ones.
- **Selection:** fixture registry subset — `cabinet_hinge`, `cabinet_single`, `drawer`, `panel`, handles, `dishwasher`, `oven`, `fridge` (×3), `microwave`, `stove`, `sink`, counters. Parameterized `--fixtures`.
- **Repackaging (the real work):** RoboCasa fixture MJCFs reference shared texture/mesh registries across the tree. Per fixture:
  1. Parse the fixture MJCF; resolve every `file=` reference (via `xml_file_references`-equivalent logic) *plus* recursively resolved `<include>`s (Genesis does this at load; the script must do it at packaging to know what to copy).
  2. Copy referenced meshes/textures into `objects/robocasa/<fixture_id>/` and rewrite refs to directory-local names (same rewrite step as `prepare_ycb_assets.py`).
  3. Ensure the root body has **no freejoint** (fixed furniture; the welded-root convention).
  4. Sanity-load in Genesis in a throwaway scene: (a) verify the loader accepts it, (b) verify the descriptor's offline MJCF FK union bbox matches `entity.get_vAABB()` at q=0 within tolerance (§1.3 cross-check) — catches FK-parser gaps per asset, (c) print the discovered joint/link names to the packaging log (informational only — nothing per-asset is written).
- **Collision:** RoboCasa fixtures ship authored collision geoms in MJCF — used as-is, no V-HACD/CoACD step. If any fixture lacks collision geoms, run `coacd` (installed) per link mesh and emit `collision_*.stl` + MJCF refs, mirroring the YCB pipeline.
- **Kitchen arena:** walls/floor/counter shell exported as `scenes/rooms/robocasa_kitchen/<variant>/` (static MJCF, visual + collision) — feeds `RoomBuilder` (§3).

### 2.2 `scripts/prepare_replicacad_assets.py`

- **Source:** HF `ai-habitat/ReplicaCAD_dataset` (132 MB) or `python -m habitat_sim.utils.datasets_download --uids replica_cad`.
- **Furniture:** the 6+ articulated URDFs (fridge, cabinets, drawers, etc.) → `objects/replicacad/<item_id>/` with URDF + GLB meshes copied directory-local; URDF `filename=` refs rewritten. Authored convex collision parts are already referenced by the URDFs → used as-is. Same Genesis sanity-load step (loader acceptance + offline-vs-`get_vAABB` bbox cross-check + joint/link discovery log).
- **The 90+ rigid objects:** optional `--include-rigid`; they map onto the *existing* `YCBObject`-style path (single-mesh `MeshObject`/`HubAssetMeshObject`) — stretch goal, not required for rooms.
- **Stage shells:** 6 stage GLBs → `scenes/rooms/replicacad/<stage_id>/` (loaded directly as `gs.morphs.Mesh(fixed=True)` by `RoomBuilder`).
- **Arrangements:** the 84 `scene_instance.json` files are small → ship verbatim under `scenes/rooms/replicacad/arrangements/`; a converter translates each into the `RoomLayoutSpec` JSON the `RoomBuilder` consumes (§3.2), resolving object ids to catalog ids and dropping non-furniture entries (or mapping them to rigid descriptors).

### 2.3 Shared pipeline utilities

Extract shared scaffolding into `scripts/_asset_pipeline_utils.py` only if it stays simpler than copy-paste — default: copy the pattern (consistency with existing scripts beats DRY here).

---

## 3. Room / procedural generation

### 3.1 `unienv_genesis_collection/scenes/room_builder.py`

Follows `TableBuilder` (`table_builder.py`) conventions: `WorldNode`, static `create_*` method, seedable `random.Random`, batch-aware tensors.

```python
@dataclass(frozen=True)
class FurniturePlacement:
    object_id: str                 # catalog id, e.g. "cabinet_hinge_0"
    source: str                    # "robocasa" | "replicacad"
    pos: Tuple[float, float, float]
    yaw: float                     # rooms are yaw-only (furniture stands upright)
    initial_open: Mapping[str, float] = frozenset()  # joint_name -> fraction
                                                     # (names resolved at runtime)

@dataclass(frozen=True)
class RoomLayoutSpec:
    shell_id: Optional[str]        # scenes/rooms/<shell_id>, None = procedural walls
    room_size: Tuple[float, float, float]  # (width, depth, wall_height)
    furniture: Tuple[FurniturePlacement, ...]

class RoomBuilder(WorldNode):
    """Builds a room: shell (walls/floor) + fixed articulated furniture.

    reload_priorities = {110}   # before TableBuilder(100) / robots(50) /
                                # clutter(25); rooms are the outermost structure.

    Modes (mutually exclusive):
      - layout=RoomLayoutSpec(...)                    explicit
      - arrangement="replicacad/<scene_instance>"     baked ReplicaCAD arrangement
      - layout_yaml="robocasa/<layout>.yaml"          RoboCasa kitchen layout,
                                                      resolved to a spec at
                                                      prepare time (preferred)
                                                      or at runtime (PyYAML)
      - procedural=True + room_size + furniture_pool  sampled layout (§3.3)
    """

    def __init__(self, world, name="room", *, layout=None, arrangement=None,
                 layout_yaml=None, procedural=False,
                 room_size=(6.0, 5.0, 2.7),
                 furniture_pool: Optional[Sequence[HubAssetArticulatedObject]] = None,
                 n_furniture: Tuple[int, int] = (3, 6),
                 wall_clearance: float = 0.05,
                 batch_fixed_verts: Optional[bool] = None,  # default: world.batch_size is not None
                 seed: Optional[int] = None): ...

    # TableBuilder-style helpers, room frame = origin at floor center:
    def xy_to_world(self, local_xy) -> torch.Tensor
    def placement_world_pose(self, local_pos, local_quat=None)

    # Lifecycle
    def reload(self, *, priority=0, seed=None, mask=None, **kwargs)
        # resolves the spec (deterministic given seed), then loads:
        #   shell entity (gs.morphs.Mesh/MJCF, fixed=True)
        #   each furniture descriptor -> load_into_scene(world, pos, quat,
        #       batch_fixed_verts=self._batch_fixed_verts)
        # Furniture entities stored: self.furniture_entities: Dict[str, RigidEntity]
        #                            self.furniture: Dict[str, ArticulatedObject]

    # Task-facing API
    def get_receptacles(self) -> Dict[str, Dict[str, BoxSpace]]
        # per furniture instance: named interior/surface regions in world
        # frame, derived from descriptor bboxes (drawer interior ≈ bbox of
        # the moving link at open configuration shrunk by wall thickness;
        # counter top ≈ top slab of union bbox). Link names are looked up
        # against the built entity at call time.
    def get_interaction_points(
        self, link_aliases: Optional[Mapping[str, str]] = None,
    ) -> Dict[str, Dict[str, GraspCandidates]]
        # per furniture instance: part grasp candidates; thin wrapper over
        # descriptor.list_part_grasping_candidates (aliases passed through).
    def sample_free_floor_positions(self, n, rng) -> torch.Tensor
        # rejection-sample inside room rect minus furniture footprints
        # (reuses get_footprint per descriptor) — for robot spawn / clutter.
```

### 3.2 Placement of fixed articulated entities

- Furniture is loaded **once** in `reload` via the descriptor's `load_into_scene(world, pos=..., quat=yaw_to_quat(yaw))` — the morph pose carries the placement (no post-load `set_pos` needed unbatched). `yaw_to_quat` already exists in `table_builder.py:60`.
- `fixed=True` everywhere: URDF via morph flag; MJCF via jointless root (§0).
- **Batched worlds:** with `batch_fixed_verts=True` (defaulted from `world.batch_size is not None`, exactly the check `pickplace.py:132` performs), per-env *layout diversity* is possible: `reset` may call `descriptor.set_pos_3d(entity, positions, envs_idx=mask)` on fixed entities. Without it, all envs share one layout — documented limitation, same trade-off Genesis imposes on `TableBuilder` (which passes `batch_fixed_verts=False` in `reload`, `table_builder.py:348`).
- Initial joint states (`initial_open` fractions, keyed by joint **name**) applied in `reset` via `descriptor.set_open_fraction(entity, joint_name, f)` — names resolved at runtime through `resolve_joint_dofs` + limits from `get_dofs_limit`; DoF state is per-env even for fixed entities, so **per-env articulation randomisation is free** (no `batch_fixed_verts` needed for that). Unknown joint names in a layout spec raise with the available-names list at reset time — exactly the robot keyframe behavior.

### 3.3 Procedural layout mode

1. Sample `n ~ Uniform(n_furniture)` furniture descriptors from `furniture_pool` (seeded `random.Random`, reseeded in `reload(seed=…)` — same pattern as `TableBuilder._rng`).
2. Wall-following placement: sample a wall, position furniture flush against it (`wall_clearance`), yaw facing inward; fall back to interior grid spots. Reject overlaps using `descriptor.get_footprint_radius()` (cheap circular check — the same primitive `ClutteredTableManager` uses) with per-item padding.
3. RoboCasa-style realism presets: `procedural_style="kitchen"` biases pools (counter row along one wall, appliances adjacent) — a small rules layer on top of the sampler, kept optional.

### 3.4 Loading ReplicaCAD arrangements / RoboCasa layouts

Both are baked to `RoomLayoutSpec` JSON at **prepare time** (preferred: no runtime dependency on habitat/PyYAML, deterministic unit tests) and resolved at runtime by id: `RoomBuilder(arrangement="replicacad/apt_0")` → `get_asset_path("scenes", "rooms", "replicacad", "arrangements", "apt_0.json")`.

---

## 4. Task integration

New `unienv_genesis_collection/tasks/articulated/open_fixture.py`, modeled directly on `ObjectRelocateTask` (`pickplace.py`), using the runtime resolver pattern end-to-end:

```python
class OpenArticulatedJointTask(WorldNode):
    """Open/close a named joint of a room furniture instance to a target fraction.

    Priority convention identical to pickplace.py: defaults sit at
    max(room_builder.priorities)+1 so the task resets after the room rebuilds.

    The joint is addressed BY NAME: DOF indices are resolved at reset/step
    time through the descriptor's runtime resolvers (resolve_joint_dofs,
    mirroring entity_controllers/base.py:27), and the open fraction derives
    q_min/q_max from entity.get_dofs_limit at runtime — no registry.

    Observation (DictSpace, batch-aware shapes via get_batch_aware_shape):
      joint_position      ([B],)   current open fraction in [0, 1]
      target_fraction     ([B],)   goal fraction
      handle_position     ([B], 3) world pos of the grasp link's bbox center
    Reward:  -|current_fraction - target_fraction| (dense)
    Termination: |.| < success_threshold
    get_info(): furniture id, joint name, language annotations (mirrors
                pickplace.py's get_info contract).
    """

    def __init__(self, world, room_builder: RoomBuilder,
                 furniture_key: str, joint_name: str,
                 grasp_link_name: Optional[str] = None,   # for obs/policy hook
                 target_fraction_range=(0.6, 1.0),
                 success_threshold=0.05, seed=None, **priorities): ...

    def reset(self, *, priority=0, seed=None, mask=None, **kwargs):
        # sample target fraction per env; randomize INITIAL joint state per
        # env via descriptor.set_open_fraction(entity, joint_name, f,
        # envs_idx=mask) — same masked-reset semantics as pickplace.py:237-247.

    def post_environment_step(self, dt, *, priority=0):
        # fraction = descriptor.get_open_fraction(entity, joint_name)

    # Reference-policy hook: expose part grasp candidates so an existing
    # GraspPolicy/GotoPolicy (unienv_genesis_refpolicy) can reach the handle:
    def get_handle_grasp_candidates(self) -> GraspCandidates
        # descriptor.list_part_grasping_candidates(
        #     entity, grasp_link_name or <joint's child link name>, ...)
```

This is the canonical "open top drawer of cabinet X" task: `RoomBuilder` provides `furniture`/`furniture_entities`, the task addresses one `(furniture_key, joint_name)` pair resolved at runtime, and everything is batch/mask consistent with existing tasks.

---

## 5. File/module layout summary

```
unienv_genesis_collection/
  objects/
    articulated_object.py                    # ArticulatedObject, ArticulationError,
                                             # resolve_joint_dofs / resolve_link_index,
                                             # urdf/mjcf FK helpers
    hub_objects/
      articulated/
        __init__.py
        hub_articulated_object.py            # HubAssetArticulatedObject
        robocasa_fixture.py                  # RoboCasaFixture + catalog/list/pool helpers
        replicacad_furniture.py              # ReplicaCADFurniture + catalog/list/pool helpers
  scenes/
    room_builder.py                          # RoomBuilder, RoomLayoutSpec, FurniturePlacement
  tasks/
    articulated/
      __init__.py
      open_fixture.py                        # OpenArticulatedJointTask
scripts/
  prepare_robocasa_assets.py
  prepare_replicacad_assets.py
tests/
  test_articulated_object.py                 # offline: FK bbox, resolvers, open fraction
  test_robocasa_fixture.py                   # load smoke (like test_gso_object.py)
  test_replicacad_furniture.py
  test_room_builder.py / _batched.py         # layouts, procedural seeding, masks
  test_open_fixture_task.py
docs/research/articulated_room_implementation_plan.md  # this document
assets (Hub):
  objects/robocasa/<fixture_id>/...          # MJCF + meshes + textures, catalog.json
  objects/replicacad/<item_id>/...           # URDF + GLB, catalog.json
  scenes/rooms/{robocasa_kitchen,replicacad}/...  # shells + arrangements/*.json
```

(`catalog.py` is **unchanged** — no `articulation` field.)

---

## 6. Staged plan (effort in dev-days, rough)

| Stage | Deliverable | Depends on | Effort |
|---|---|---|---|
| **S1** | `articulated_object.py`: `ArticulatedObject`, URDF FK union bbox (via `genesis.ext.urdfpy`), runtime name resolvers, open-fraction helper (`get_dofs_limit`-based), part grasps; offline unit tests | — | 2–3 |
| **S2** | `prepare_replicacad_assets.py` + `ReplicaCADFurniture`; smoke tests on 2–3 items (incl. bbox cross-check vs `get_vAABB`) | S1 | 2–3 |
| **S3** | `scenes/room_builder.py`: explicit-spec mode + procedural sampler + receptacle/interaction API; tests incl. batched | S1 | 3–4 |
| **S4** | `prepare_robocasa_assets.py` + `RoboCasaFixture`; MJCF FK walker + pipeline bbox cross-check; smoke tests on cabinet + drawer + one appliance | S1 | 3–4 |
| **S5** | ReplicaCAD arrangements → `RoomLayoutSpec` baking; `RoomBuilder(arrangement=…)`; RoboCasa layout YAML baking | S2, S3, S4 | 2 |
| **S6** | `tasks/articulated/open_fixture.py` + masked-reset tests; example script wiring RoomBuilder + robot + task | S3, S4 | 2–3 |
| **S7** | Kitchen-shell assets (`scenes/rooms/…`), procedural "kitchen" style preset, docs + Hub upload of all assets | S4, S5 | 2–3 |

Total ≈ 16–22 dev-days (dropping the registry/catalog-baking trims ~1–2 days off S1 and the pipelines). S1 unblocks everything; S2/S4 are independent (URDF vs MJCF) and parallelizable.

---

## 7. Risks and unknowns (with mitigations)

1. **RoboCasa MJCF fidelity in Genesis.** Genesis's MJCF parser absolutizes compiler dirs and handles includes (verified), but the `MJCF` morph docstring states the file is processed "as if describing a single entity" and scene-level options are ignored; fixtures using MuJoCo-only features (equality constraints, tendons, friction latches) may simulate differently. *Mitigation:* S4's sanity-load step loads every repackaged fixture in Genesis and asserts the joint set is non-empty and the offline FK bbox matches `get_vAABB` within tolerance; failures drop the fixture with a warning rather than failing the pipeline. Verify cabinet hinge + drawer first (cheapest signal).
2. **Batched fixed articulated entities.** DoF state is per-env (verified), so per-env articulation is safe. Per-env *base poses* of fixed entities require `batch_fixed_verts=True` with "significantly increasing memory" (Genesis docstring). Rooms hold several large fixed meshes → memory could blow up at large batch sizes. *Mitigation:* default per-env layout diversity off (shared layout, per-env joint randomisation only); document the memory trade-off; add a batched test at modest B (e.g. 4–8) before claiming support.
3. **`merge_fixed_links` interaction.** Must be `False` (or `links_to_keep=[handle links]`) for part queries, matching the YCB rationale. *Risk:* ReplicaCAD URDFs with many fixed-joint sublinks increase link count/solver cost. *Mitigation:* measure per-item link counts in S2; use `links_to_keep` with only the grasp-target links preserved if counts hurt.
4. **GLB visual meshes + textures.** ReplicaCAD uses GLB; trimesh/Genesis load it, but texture fidelity through `gs.morphs.URDF` (`prioritize_urdf_material=False` default) needs a visual check in S2. *Mitigation:* render one frame per item in the pipeline smoke test.
5. **Runtime MJCF FK parser (§1.3).** May miss MJCF features (defaults classes, asset transforms). *Mitigation:* the prepare pipelines run the offline-FK-vs-`get_vAABB` cross-check per asset while Genesis is available, so parser gaps are caught at packaging time and fixed (or the fixture dropped) before assets ship. There is no silent-wrong-bbox path.
6. **Receptacle precision.** Drawer-interior regions derived from bboxes are approximations. ReplicaCAD ships authored receptacle metadata — the prepare script may carry it into the arrangement/spec files when present and prefer it over the bbox heuristic. Unknown until assets are inspected in S2.
7. **Joint-name stability.** Addressing is by name against the built entity; if Genesis's loaders mangle/namespace names (e.g. MJCF name collisions), layout specs using `initial_open` names would break. *Mitigation:* the pipeline's discovery log records the exact built-entity names; `resolve_joint_dofs` errors list available names (same UX as `resolve_joint_indexes`); name-normalization is a deliberate non-feature (kept identical to the robot pattern).
8. **License hygiene.** Both sources are CC BY 4.0 — attribution files (`LICENSE`, provenance `README.md`, same convention as GSO/YCB pipelines) are generated by the prepare scripts; no blockers found.

---

## 8. Explicit non-goals (for this plan)

- Articulated *movable* furniture (chairs with casters etc.) — the freejoint variant exists in the design but is not built.
- ReplicaCAD's 90+ rigid objects (stretch goal in S2 only).
- Mobile-base navigation tasks (room builder exposes `sample_free_floor_positions` as the hook, nothing more).
- Photorealistic domain randomisation of shells beyond what the assets ship.
- Any per-asset joint/part metadata baked into code or `catalog.json` (rejected — see decision log).

---

### Key design takeaways (TL;DR)

1. **`ArticulatedObject` is a sibling of `MeshObject`, not a child** — the single-`mesh_file` assumption in `MeshObject`/`HubAssetMeshObject` doesn't survive articulation; the hub machinery (`_ensure_assets`, `xml_file_references`) is reused in a new `HubAssetArticulatedObject`.
2. **No joint registry — the robot pattern wins.** Joint/link addressing is by name, resolved statelessly off the built entity (`resolve_joint_dofs`/`resolve_link_index` mirroring `entity_controllers/base.py:27`); `catalog.json` stays untouched.
3. **Union bbox at a reference configuration** (offline FK via `genesis.ext.urdfpy` for URDF, small MJCF walker for RoboCasa) keeps the entire `ManipulatableObject` contract (`get_bounding_box`, footprints, bbox-grasp fallback) working untouched; a `get_vAABB` cross-check in the pipelines catches FK-parser gaps at packaging time.
4. **Open fraction = `resolve_joint_dofs` + `get_dofs_limit` at runtime** — the only semantics added beyond what robots already have; part-aware grasps on named links ride on `get_links_pos/quat` so they track moving drawers.
5. **Fixed-by-default furniture**: URDF `fixed=True`, MJCF jointless root (no `fixed=` flag exists on MJCF morphs — the welded-root GSO trick). DoF state is per-env in batched worlds for free; per-env *layouts* need `batch_fixed_verts=True` and its memory cost.
6. **Watch out:** `merge_fixed_links=False` (or `links_to_keep`) is mandatory for handle/door part queries, and RoboCasa MJCF fixtures need a Genesis sanity-load gate in the pipeline since Genesis ignores MuJoCo scene-level options/equality constraints.
