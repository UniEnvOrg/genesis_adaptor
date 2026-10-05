# Articulated furniture & rooms

`genesis_adaptor` can build **rooms furnished with fixed articulated furniture**
(kitchen cabinets, drawers, doors, appliances) and drive a **single named joint
of one furniture instance** — the "open the top drawer of cabinet X" family of
manipulation objectives — with the same node/priority lifecycle as every other
scene (see the [README](../README.md)).

This page covers the three building blocks, the packaged assets and their
licenses, the room-layout sources, and how to regenerate the assets.

| Piece | Module | Role |
|---|---|---|
| `ArticulatedObject` | `unienv_genesis_collection/objects/articulated_object.py` | Descriptor contract: offline-FK union / per-link bboxes, `load_into_scene`, runtime joint & link **name resolution** (`resolve_joint_dofs`, `resolve_link_index`), open-fraction helpers (`get_open_fraction` / `set_open_fraction`, limits from `entity.get_dofs_limit()`), part grasp candidates. |
| `RoboCasaFixture`, `ReplicaCADFurniture`, `ArtVIPFurniture`, `MolmoSpacesFurniture`, `LightwheelFurniture`, `LightwheelEnv` (static environments), `SimReadyWarehouseProp` (`make_*_pool`, `list_*_objects`, `get_*_catalog`) | `.../objects/hub_objects/articulated/` (`LightwheelEnv` sits in `.../objects/hub_objects/`) | Catalog-backed descriptors for the packaged furniture (MJCF via `gs.morphs.MJCF`; ReplicaCAD URDFs via `gs.morphs.URDF` with `merge_fixed_links=False`; the USD families via `gs.morphs.USD` + `scene.add_stage` with clean-name translation of the full-path USD joint/link names — `SimReadyWarehouseProp` is the rigid, joint-less member). |
| `RoomBuilder` | `unienv_genesis_collection/scenes/room_builder.py` | Builds the room shell plus every fixed furniture instance, and exposes the task-facing API (`furniture` / `furniture_entities`, `get_receptacles`, `get_interaction_points`, `sample_free_floor_positions`, `placement_world_pose`, `set_furniture_pose_3d`). |
| `OpenArticulatedJointTask` | `unienv_genesis_collection/tasks/articulated/open_fixture.py` | Binds to a `(furniture_key, joint_name)` pair of a `RoomBuilder` room and provides a dense fraction-tracking reward with a randomized goal. |

## Minimal usage

```python
import genesis as gs
from genesis.constants import backend as gs_backend

gs.init(backend=gs_backend.gpu)

from unienv_interface.world import WorldEnv
from unienv_genesis import GenesisWorld
from unienv_genesis_collection.robots.franka_fr3_robotiq_2f85 import FrankaFR3WithRobotiq2F85
from unienv_genesis_collection.scenes.room_builder import RoomBuilder
from unienv_genesis_collection.tasks.articulated import OpenArticulatedJointTask

world = GenesisWorld(batch_size=4)  # batched: 4 parallel environments
room = RoomBuilder(world, name="room", layout_yaml="robocasa_kitchen/open_kitchen")
robot = FrankaFR3WithRobotiq2F85(world, name="franka", pos=(-0.73, 0.57, 0.0))
task = OpenArticulatedJointTask(
    world,
    room_builder=room,
    furniture_key="cabinet_hinge",          # key from the resolved layout
    joint_name="rightdoorhinge",            # MJCF joint name, resolved at runtime
    grasp_link_name="cabinet_hinge_right_door_handle",
)

env = WorldEnv(world, [room, robot, task], render_mode=None)
_, obs, info = env.reset(seed=0, reload=True)
print(obs["open_fixture_task"])
# joint_position:   ([B],)   current open fraction
# target_fraction:  ([B],)   per-env goal fraction
# handle_position:  ([B], 3) world position of the handle link's bbox centre
```

The room can also come from a **procedural** layout or a ReplicaCAD arrangement:

```python
from unienv_genesis_collection.objects import (
    make_robocasa_pool, make_replicacad_pool, make_artvip_pool,
)

procedural = RoomBuilder(
    world, procedural=True, n_furniture=(3, 5), seed=0,
    furniture_pool=make_robocasa_pool(tags="kitchen", n=6),
    # or: furniture_pool=make_replicacad_pool(tags="articulated", n=6),
    # or: furniture_pool=make_artvip_pool(tags="kitchen", n=6),   # ArtVIP USD
)
replicacad = RoomBuilder(world, arrangement="replicacad/apt_0")   # baked arrangement
```

`examples/room_open_drawer_fr3.py` is the complete end-to-end example (room +
Franka FR3 + `OpenArticulatedJointTask` + a scripted reach towards the handle):

```bash
python examples/room_open_drawer_fr3.py --backend cuda --steps 90             # + rendered frames
python examples/room_open_drawer_fr3.py --backend cpu --n-envs 4 --no-render  # headless, batched
```

## The task contract

`OpenArticulatedJointTask` owns no entities — `RoomBuilder.reload` loads the
furniture once, and the task binds to it by **name** (its priorities default to
one above the room's, so it binds and resets after the room rebuilt):

* **observation** — `joint_position` / `target_fraction` (one scalar per env)
  and `handle_position` (the joint's child link, or `grasp_link_name`, as
  reported on the built entity).
* **reward** — `-|current_fraction - target_fraction|`, dense, in `[-1, 0]`.
* **termination** — `|current - target| < success_threshold`.
* **reset** — the goal is sampled from `target_fraction_range` (default
  `(0.6, 1.0)`) and the joint starts **strictly below its own goal**, so an
  episode always begins outside the success band.  Masked resets
  (`env.reset(mask=...)`) re-randomize only the selected envs.
* `get_info()` reports the furniture key / catalog id / joint / handle link and
  the catalog's language annotations.

The fraction is the **normalised joint position** between the runtime limits,
`(q - q_min) / (q_max - q_min)`.  `0 = closed, 1 = open` therefore holds only
when the asset authors its open pose at the upper limit (see the caveats below).

## Assets

| Source | Packaged under | Loader | License |
|---|---|---|---|
| **RoboCasa** kitchen fixtures (`cabinet_hinge`, `drawer`, `stove_simple_gas`, …) and the kitchen arena shells (`open_kitchen`, `galley`) | `assets/objects/robocasa/`, `assets/scenes/rooms/robocasa_kitchen/` | MJCF (`model.xml`; jointless root ⇒ welded/fixed) | [robocasa.ai](https://robocasa.ai) — **CC BY 4.0** (attribution shipped as `objects/robocasa/LICENSE`) |
| **ReplicaCAD** articulated furniture (fridge, cabinets, doors, …) and stage shells / arrangements | `assets/objects/replicacad/`, `assets/scenes/rooms/replicacad/` | URDF + GLB | ⚠️ **inconsistent upstream license** — see below |
| **ArtVIP** articulated USD furniture (`refrigerator_fridge_4`, `cabinet_1`, …) | `assets/objects/artvip/` | USD (`<object_id>.usd` + `resource/**` sidecars; `gs.morphs.USD` via `scene.add_stage`) | Card states **Apache-2.0**, but the snapshot **commits no LICENSE file** — see below; the `Ikea_furniture`/`Medical_equipment` subsets are excluded |
| **MolmoSpaces** (Ai2-THOR) articulated USD furniture (`fridge_26`, `ikeacabinet_1`, …) | `assets/objects/molmospaces/` | USD (`<object_id>.usda` + `Textures/` sidecar; `gs.morphs.USD` via `scene.add_stage`; the packaged stages carry the mandatory joint-body0/body1 swap) | **CC BY 4.0** — only the `isaac/objects/thor` subset is packaged (the Objaverse subset, ODC-BY with per-object NC entries, is never fetched) |
| **Lightwheel** SimReady open bundle articulated USD assets (`lightwheel_Microwave043`, `lightwheel_Stovetop002`, …) | `assets/objects/lightwheel/` | USD (`<object_id>.usd` + `texture/` sidecar; packaged stages carry the MDL → UsdPreviewSurface material rewrite) | ⚠️ **CC BY-NC 4.0 — non-commercial** (accepted by the maintainer) — see below |
| **Lightwheel** SimReady open bundle **static environments** (`lightwheel_env_Grass`, `lightwheel_env_Apartment`, …) | `assets/objects/lightwheel_env/` | USD scene stages (`<object_id>.usd` + reference closure; `gs.morphs.USD` via `scene.add_stage`; **no joints** — terrains/rooms) | ⚠️ **CC BY-NC 4.0 — non-commercial** (same bundle as the articulated subset) — see below |
| **NVIDIA SimReady Warehouse** rigid props (`prop_cardbox_c1`, `prop_trolley_wheel_base`, …) | `assets/objects/simready_warehouse/` | USD (`<object_id>.usd` — the upstream `<Asset>_physics.usd` overlay; `RigidBodyAPI` authored at packaging; **zero joints** — rigid props) | Card tags **CC BY 4.0** but the archive **commits no LICENSE file** — attribution shipped (see below) |

> **ReplicaCAD license caveat.**  The upstream ReplicaCAD dataset is
> inconsistent: the Hugging Face dataset card (front matter and prose) declares
> **CC BY 4.0**, while the `LICENSE.txt` shipped in the snapshot states
> **CC BY-NC 4.0** with a non-commercial clause.  Treat the *more restrictive*
> license (**CC BY-NC 4.0, no commercial use**) as governing and attribute
> "ReplicaCAD dataset, © Meta/Facebook Reality Labs (CC BY-NC 4.0)" when using
> these models.  The upstream `LICENSE.txt` ships verbatim next to the assets
> (`assets/objects/replicacad/LICENSE`) and the provenance README repeats this
> note.

> **ArtVIP license caveat.**  The ArtVIP dataset card declares **Apache-2.0**,
> but the snapshot commits **no LICENSE file** — re-check the dataset card
> before relying on the grant and attribute "ArtVIP dataset, © X-Humanoid
> (Apache-2.0)" (attribution shipped as `assets/objects/artvip/LICENSE`).  All
> categories — including the `Ikea_furniture` (37) and `Medical_equipment` (15)
> upstream subsets, whose products sit directly under the category dir
> (``<category>/<product>/model_*.usd``) — are packaged under that caveat.

> **MolmoSpaces license.**  The MolmoSpaces dataset card licenses the
> Objaverse subset **ODC-BY 1.0** and *all other subsets* **CC BY 4.0**.  Only
> the `isaac/objects/thor` USD subset is packaged — CC BY 4.0 wholesale — and
> the Objaverse subset is never fetched, so nothing non-CC-BY-4.0 enters the
> pack.  Attribute "MolmoSpaces THOR objects, Model(s) by the Allen Institute
> for AI (Ai2), CC BY 4.0" (shipped as `assets/objects/molmospaces/LICENSE`).

> **Lightwheel license caveat — non-commercial.**  The Lightwheel open bundle
> (`LightwheelAI/Lightwheel-simready-asset`) is **CC BY-NC 4.0** ("Free for
> non-commercial use") and requires the `lightwheel_{asset_name}` naming
> format for attribution — hence the `lightwheel_`-prefixed object ids.  The
> non-commercial clause is **accepted by the collection maintainer** (the
> same treatment as the ReplicaCAD caveat above): the bundle's Drive archive
> ships **no LICENSE file** (only the GitHub repo carries one), so the pack
> always writes the canonical CC BY-NC 4.0 Creative Commons legal-code text
> (`assets/objects/lightwheel/LICENSE` and, for the static environments,
> `assets/objects/lightwheel_env/LICENSE`) and the provenance README repeats
> the warning.  Consumers with commercial
> intentions must license the commercial library from
> [simready.com](https://simready.com) instead of using this pack.  The
> static environment family (`objects/lightwheel_env/`, the bundle's
> `Locomotion/` subset minus `KitchenRoom`) carries the **same** license;
> `lightwheel_KitchenRoom_RSS` is articulated and ships in
> `objects/lightwheel/`.

> **SimReady Warehouse license caveat.**  The dataset card tags **CC BY 4.0**
> but the archive commits **no LICENSE file** — re-check the dataset card
> before relying on the grant and attribute "Physical AI SimReady Warehouse
> Dataset, © NVIDIA Corporation (CC BY 4.0)" (attribution shipped as
> `assets/objects/simready_warehouse/LICENSE`).

### ArtVIP (USD) notes

ArtVIP stages load through `scene.add_stage` (the composite-stage loader):
every per-object stage splits into exactly **one** `RigidEntity`
(`load_into_scene` asserts this), and `add_stage` takes no entity `name` —
`ArtVIPFurniture(name_override=…)` calls warn-and-ignore, so
`RoomBuilder.reload`'s name override has no effect for this family (the
builder keys its own dicts; ownership is unchanged).

* **Clean joint/link names.**  Genesis addresses USD joints/links by their
  full prim path — e.g. the runtime joint name
  `/root/E_door1_1/RevoluteJoint_skbx_25_right` — while the task-facing
  name is the *leaf* prim token: `RevoluteJoint_skbx_25_right`.  The
  descriptor keeps the offline clean-name translation (`"/"`-prefixed
  names pass through unchanged); both spellings address the same joint, and
  unknown clean names raise `ArticulationError` listing the leaf names.
  Duplicate leaves receive the parent-chain suffix
  (`E_door1/hinge`); *packaging drops objects whose raw leaves collide*,
  so shipped assets have unique leaf names and the suffix form never
  appears in the wild.
* **Box-proxy offline FK.**  The offline (no-Genesis) FK walker builds each
  link's geometry as the box around its subtree bounds; the reference-config
  union bbox therefore matches the runtime `get_vAABB()` up to the
  box-vs-mesh silhouette deltas (a few cm on non-convex overhangs) — which
  the packaging gate bounds via abs/rel tolerances, per-object deltas in
  `packaging_log.json`.  Part-grasp candidates enumerate in the same
  link-local boxes; treat them as coarse envelopes (the URDF/MJCF helpers
  carry real per-vertex meshes — USD refinement would be a follow-up
  sampling step over the UsdGeom meshes).
* **Placement/frames.**  USD stages are *not* re-centred onto their root
  body: the entity frame = the stage's world frame (elevated by the morph
  `pos`), so the offline FK bbox (same frame) matches `get_vAABB()` without
  an extra offset.  ArtVIP objects are Z-up, so no `standing_quaternion` is
  authored in the generated catalog.

### MolmoSpaces (USD) notes

MolmoSpaces stages load through the same `scene.add_stage` contract as the
ArtVIP family (single entity asserted, `name_override` warned-and-ignored,
clean leaf joint/link names).  Two packaging adaptations are **built into
the shipped stages**:

* **Joint-body0/body1 swap (mandatory).**  MolmoSpaces authors every
  UsdPhysics joint with `body0` = child / `body1` = parent (the MJCF order) —
  the reverse of Genesis's loader contract.  As shipped, 12/18 probed
  variants *fail to load* (`Link … has multiple parents`) and the remaining
  6 load with the movable part in the base link (the oven swings around the
  door).  Every packaged stage is rewritten by
  `unienv_genesis_collection.utils.usd_conversions.swap_usd_joint_bodies`
  (relationship targets **and** the per-side `localPos0/1`/`localRot0/1`
  frames) during packaging; the sanity gate verifies the swapped stage in
  Genesis (single entity, correct DOFs, closed + halfway-opened bbox).
* **The `<Name>_mesh` variant is packaged** — convex-hull `Mesh` colliders
  (e.g. 14 colliders for Fridge_26 instead of the base variant's 281
  primitive colliders).  Materials are already UsdPreviewSurface (no MDL in
  the thor USD subset).

### Lightwheel (USD) notes

* **Materials — the mandatory adaptation.**  The bundle authors OmniPBR MDL,
  which usd-core cannot read (Genesis would lose the materials silently).
  Packaging rewrites every graph into native `UsdPreviewSurface` +
  `UsdUVTexture` (`unienv_genesis_collection.utils.usd_conversions.convert_mdl_to_preview_surface`,
  shared with the Warehouse family) — verified: 4/4 Microwave043 materials
  parse as color/roughness/metallic image textures after the rewrite.
* **No joint-body swap** — the bundle authors `body0` = parent / `body1` =
  child, already Genesis's order.
* **Base fixity varies upstream**: some assets weld their base
  (`physics:kinematicEnabled` on the base body, e.g. Stovetop002); the
  descriptor's `fixed` morph flag overrides root fixity at load.
* **CC BY-NC 4.0** — see the license caveat above.

### Lightwheel static environments (USD, `Locomotion/` subset) notes

* **Separate family, not furniture.**  `LightwheelEnv` (`objects/lightwheel_env/`)
  loads the bundle's jointless scene stages — 7 terrains/rooms/a staircase —
  with `gs.morphs.USD` + `scene.add_stage`.  The articulated tree's descriptor
  asserts a single articulated entity, exposes joint APIs and its gate
  requires moving joints; a terrain violates all three (and
  `usd_link_meshes_at_config` returns `{}` for a jointless
  `RigidBodyAPI`-less stage, breaking the articulated geometry path outright).
* **Routing is explicit.**  `KitchenRoom_RSS.usd` is genuinely articulated
  (35 revolute + 23 prismatic joints, 131 rigid bodies) and ships through the
  *articulated* pipeline as `lightwheel_KitchenRoom_RSS`; the other seven go to
  `lightwheel_env_<Category>` (`2-StoryStaircase` →
  `lightwheel_env_2_story_staircase`).  Root layers come from the measured
  `LOCOMOTION_ROOTS` map (`Apartment` → `scene_04.usd`; `KitchenRoom.usd` is
  unusable — an absolute escaping payload plus missing sublayers;
  `Ground_stone.usd` is a payload of `SLATEGround.usd`, not a root).
* **Closure-copy packaging.**  Only the root layer plus its
  `_usd_required_files` closure ships, so upstream junk (`E/`, `.thumbs`,
  `.DS_Store`, `Table049/`, `ClothDemo_002/`, the S3-mirror dir) is excluded by
  construction while referenced payload dirs are kept.  Apartment's 20
  upstream UE material records (`Game/.../MI_leaf_32_1.MI_leaf_32_1`) are
  recorded as `unresolvable_refs` in `packaging_log.json` instead of failing
  the pack; missing `.usd` layers/textures stay fatal.
* **Geometry** comes from a cached pxr `BBoxCache` union (trimesh cannot read
  USD; the largest stage is 439 MB / 1764 meshes).  The static gate asserts no
  joints, collision present, and offline-vs-loaded `get_vAABB()` agreement.
* **CC BY-NC 4.0** — see the license caveat above.

### SimReady Warehouse (rigid props) notes

* **Zero articulation upstream** — the dataset README states "USDPhysics
  Colliders applied, but no Rigid Bodies are applied".  Packaging applies
  the README's Isaac recipe offline: `UsdPhysics.RigidBodyAPI` on the
  object's root payload prim, which makes `scene.add_stage` collapse the
  prop into exactly **one** rigid entity (per-prim `CollisionAPI`
  colliders would parse as one entity each, and `ArticulationRootAPI`
  alone yields none).  `SimReadyWarehouseProp` is therefore a joint-less
  member of the articulated family: `list_joint_names() → []`, the union
  bbox is the whole-prop box, and the entity is welded by default
  (`fixed=False` loads it as a free dynamic rigid body).
* **Materials are rewritten** from OmniPBR MDL into UsdPreviewSurface (the
  shared converter); the local `materials/*.mdl` recipes are dropped, their
  textures ship.
* **Units normalised to metersPerUnit = 1.0** (upstream mixes 0.01 /
  0.0099999… / 1.0): the factor folds into the root prim's linear map, so
  the world size is unchanged and Genesis's `_meter_scale` correction
  becomes a no-op — the offline FK matches the runtime frame exactly.
* **Assemblies** (`--categories assembly`, 103 CSV rows) are packaged with
  their payload-closure prop directories at the upstream subtree paths (the
  payload arcs reach `../../general/<Prop>/…`); the packaged root is a thin
  sublayer of the preserved assembly stage, and the single-body collapse
  ships the arrangement as one welded rigid body.  A snapshot that misses
  any payload'd prop fails the closure verification and the assembly is
  dropped with a warning.

Baked room layouts (see below) live in:

* `assets/scenes/rooms/robocasa_kitchen/layouts/<style>.json` —
  `RoomBuilder(layout_yaml="robocasa_kitchen/open_kitchen")`.
* `assets/scenes/rooms/replicacad/arrangements/<name>.json` (baked from the
  upstream `*.scene_instance.json`) — `RoomBuilder(arrangement="replicacad/apt_0")`.

A baked spec is a plain `RoomLayoutSpec` JSON: `shell_id`, `room_size` and one
entry per furniture instance (`object_id`, `source`, room-frame `pos` / `yaw`,
optional `initial_open` joint fractions).  The room frame has its **origin at
the floor centre**, `z` up; ReplicaCAD placements are converted from the stage's
Y-up glTF frame at bake time.

## Regenerating / uploading the assets

The collection assets are **not** shipped in the Python package — they live in
the `UniEnvOrg/unienv_genesis_collection_assets` Hub dataset and are downloaded
on demand.  To regenerate them:

```bash
# 1. RoboCasa fixtures + kitchen shells (fixtures.zip / textures.zip from the
#    Hub, or a local robocasa/models/assets download via --source)
python scripts/prepare_robocasa_assets.py --output-dir /tmp/robocasa_assets

# 2. ReplicaCAD furniture + stage shells + arrangement inputs
python scripts/prepare_replicacad_assets.py --output-dir /tmp/replicacad_assets

# 2b. ArtVIP USD furniture (X-Humanoid/ArtVIP; categories from
#     --categories, defaulting to major_appliances + small_furniture; the
#     Ikea/Medical subsets are refused while their terms are unconfirmed).
#     Thumbnail renders land in /tmp/artvip_sanity (--render-dir).
python scripts/prepare_artvip_assets.py --output-dir /tmp/artvip_assets

# 2c. MolmoSpaces (Ai2-THOR) USD furniture (allenai/molmospaces; a ~1.1 GB
#     thor shard is downloaded and extracted, or pass --source at an
#     extracted object tree).  The joint-body swap runs at packaging.
#     Thumbnail renders land in /tmp/molmospaces_sanity.
python scripts/prepare_molmospaces_assets.py --output-dir /tmp/molmo_assets

# 2d. Lightwheel SimReady open bundle (the 3.41 GB anonymous Google Drive
#     bundle is downloaded, or pass --source at an extracted
#     Lightwheel_OpenSource tree).  CC BY-NC 4.0 — see the caveat above.
#     Thumbnail renders land in /tmp/lightwheel_sanity.
python scripts/prepare_lightwheel_assets.py --output-dir /tmp/lw_assets

# 2d-bis. The bundle's Locomotion subset (--subset locomotion|all): the 7
#     static environments into objects/lightwheel_env/ (static gate) plus
#     lightwheel_KitchenRoom_RSS into objects/lightwheel/ (articulated gate).
python scripts/prepare_lightwheel_assets.py --subset locomotion \
    --output-dir /tmp/lw_assets

# 2e. NVIDIA SimReady Warehouse rigid props (nvidia/PhysicalAI-SimReady-
#     Warehouse-01; --categories prop|handmanip|assembly over the 753-row
#     CSV catalog, plus --objects for a subset).  Thumbnail renders land
#     in /tmp/simready_warehouse_sanity.
python scripts/prepare_simready_warehouse_assets.py --output-dir /tmp/wh_assets

# 3. Bake the room layouts (RoomLayoutSpec JSON) from the packaged inputs; the
#    specs land in the same assets root, so they upload with the trees above.
#    Use --families robocasa for the RoboCasa fixtures.
python scripts/bake_room_layouts.py \
    --source /tmp/robocasa_assets \
    --output-dir /tmp/robocasa_assets \
    --families robocasa
```

The prepare scripts run a Genesis sanity gate (loader acceptance, joint set,
offline-FK bbox vs. `entity.get_vAABB()`, plus per-item standing-pose
thumbnail renders — skipped with a warning on CPU-only machines, where the
Rasterizer camera is unavailable) and print the exact `hf upload` commands for
their trees.  The ArtVIP/MolmoSpaces/Lightwheel gates additionally cross-check
a *halfway-opened* configuration (per-joint midpoints) and drop objects whose
upstream leaf joint/link names collide; the Warehouse gate skips the
moving-joint requirement (rigid props) but keeps the single-entity assert,
the bbox cross-check and the thumbnails.  The thumbnails land in a scratch
dir (`--render-dir`, default `/tmp/robocasa_sanity` / `/tmp/replicacad_sanity`
/ `/tmp/artvip_sanity` / `/tmp/molmospaces_sanity` / `/tmp/lightwheel_sanity`
/ `/tmp/simready_warehouse_sanity`; `--no-render` disables them) for
eyeballing — they are **not** uploaded.  **Uploading is left to you** — it
needs write access to the org; nothing is uploaded automatically.

## Caveats

* **Fraction direction follows the asset.**  RoboCasa's `drawer` fixture
  authors its slide closed at the *upper* limit (`range [-0.6, 0]`), so its
  fraction runs backwards (0 = fully extended); the cabinet's `rightdoorhinge`
  (`[0, 3]` rad) has the natural direction, while `leftdoorhinge` (`[-3, 0]`)
  is mirrored.  Pick `target_fraction_range` accordingly.
* **Hinge ranges can overshoot the fixture body.**  `cabinet_hinge`'s doors
  range up to 3 rad, but above ~0.6 of the fraction the door penetrates the
  cabinet/counter geometry and the solver pushes it back (the joint drifts on
  its own).  Sample goals inside a physically clean band (the example uses
  `(0.4, 0.6)`).
* **Grasp candidates are link-frame correct.**  `RoomBuilder.get_interaction_points`
  / `OpenArticulatedJointTask.get_handle_grasp_candidates` enumerate an
  **offline per-link bbox in the link's own local frame** and transform it by
  the link's *current* world pose, so candidates track a swinging/sliding part
  with no offset — even for links away from the base-link origin (the RoboCasa
  drawer's `drawer_door_handle`, the cabinet doors' handles).  Note that the
  other bbox helper, `get_link_local_bounding_box`, is documented in the
  *base-link* frame (it feeds the union bbox); do **not** combine it with the
  link pose for grasp candidates — that double-transform was an earlier bug,
  now pinned by a regression test (`tests/test_articulated_object.py`).
  `handle_position` from the task observation (measured by rigid attachment to
  the link pose) remains a reliable world-frame reference for reaching.
* **Camera lighting.**  The example's `--render` cameras use the classic
  rasterizer, whose lighting comes from `gs.options.VisOptions` — the default
  directional light is blocked by the room walls, so pass a vertical light /
  ambient (as `examples/room_open_drawer_fr3.py` does).  Nyx cameras
  (`backend="nyx"`, GPU) ignore `VisOptions` and use the world's env map.
