# Survey: Articulated Furniture / Room Asset Sources for Genesis (UniEnvDev)

**Scope:** sources of **procedurally generated rooms with articulated furniture** (cabinets w/ doors, sliding drawers, chairs, tables, shelves, fridges, dishwashers), usable by Genesis, with effort estimates to repackage into this project's per-object folder format. Every claim was verified by fetching the actual repo/dataset page (URLs cited).

## 0. The target format (verified from this repo)

| Fact | Evidence |
|---|---|
| Assets on HF **dataset** repo `UniEnvOrg/unienv_genesis_collection_assets`, lazy download | `asset_path.py` (`REPO_ID`, `REPO_TYPE`, `_hub_kwargs`) |
| Whole per-object dir downloaded as a subtree | `hub_asset_object.py`: `download_assets(allow_patterns=f"{asset_relative_dir}/**")` |
| Any descriptor works if meshes/textures are **directory-local** | `xml_file_references()` parses `file=` (MJCF) **and** `filename=` (URDF) |
| MJCF via `gs.morphs.MJCF`, URDF via `gs.morphs.URDF` | `gso_object.py`, `ycb_object.py::load_into_scene` |
| Metadata = sibling `catalog.json` (`language_annotations`, `tags`, `standing_quaternion`) | `hub_objects/catalog.py` |
| Existing layout | `objects/gso/<id>/` (1000), `objects/ycb/<id>/` (81), `scenes/env_maps`, `scenes/table` — **a `scenes/` convention already exists** |
| Genesis loaders | `genesis/utils/mjcf.py`, `urdf.py`, `usd/` |

**Implication:** any self-contained MJCF/URDF drops in with near-zero code change; collision must already be in the descriptor (MJCF `mesh file=`, URDF `<collision><mesh filename=>`) or be generated (CoACD/V-HACD). Rooms slot into `scenes/<room_id>/`.

## 1. PartNet-Mobility (SAPIEN) — best *coverage*, worst *access*
- **~2K articulated objects** (cabinets single/double-door, drawers, chairs, tables, fridges, dishwashers, ovens, storage).
- **Format:** `model.obj`+`.mtl`+texture, `mobility_vhacd.urdf`, V-HACD `_collision_*.obj`, `semantics.txt`, `meta.json`. URDF-native.
- **License:** **gated, non-commercial research/education only** — gate requires name/affiliation/purpose and an institutional email; *"Researcher shall use the Database only for non-commercial research and educational purposes"*; builds on ShapeNet ToU.
- **Access:** `huggingface.co/datasets/sapien-sim/PartNetMobility` (gated **manual**); `github.com/haosulab/SAPIEN`; ungated per-model zips at `https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset/<model_id>.zip`; unlabeled mirrors (`LocalWorldModels/partnet-mobility-urdf-parts`, `SeaLab/partnet-mobility-updated`, `yuchen0187/partnet-mobility`).
- **Quality:** V-HACD precomputed; inconsistent units/origins/textures per model.
- **Rooms:** ❌. **Effort:** Low–Medium (same shape as the existing YCB pipeline).

## 2. AI2-THOR / ProcTHOR / Objaverse (objathor / Holodeck)
- 120 iTHOR rooms, 89 RoboTHOR apartments, 10 ArchitecTHOR houses, **12,000 ProcTHOR houses**. Articulated in the Unity runtime, but **exports are static GLB**.
- **Access:** `hssd/ai2thor-hab` (gated "share your contact information"), `haosulab/AI2THOR` (`ai2thor.zip`).
- **License:** code Apache-2.0; ⚠️ **no asset-specific license file** (`NOTICE`/`THIRD-PARTY.md` → 404) and the HF export is gated → mesh redistribution unclarified.
- **Holodeck** (`allenai/Holodeck`, Apache-2.0) generates rooms from Objaverse via `allenai/objathor` (Apache-2.0).
- **Rooms:** ✅ richest procedural room generator. **Effort:** High (must author joints yourself).

## 3. BEHAVIOR-1K / iGibson
- ❌ **Blocked.** Assets "purchased for the BEHAVIOR-1K dataset construction… The **encrypted** dataset distributed with OmniGibson includes processed versions" (TurboSquid, non-redistributable). iGibson: models *"are free to use within iGibson 2.0… **encrypted** and can only be used within iGibson 2.0"*; code MIT, dataset via licence form + key.

## 4. 3D-FRONT / 3D-FUTURE
- **18,968 rooms** + **13,151 textured furniture objects**. Scene JSON (floorplan + furniture poses) rendered by `3D-FRONT-FUTURE/3D-FRONT-ToolBox` (`json2obj.py`).
- **License:** academic/non-commercial. ⚠️ **Official Tianchi pages are now dead**; only HF mirrors work (`huanngzh/3D-Front` cc-by-nc-4.0, `JIAQI-CHEN/3D-Front`, `liaolw/3D-FUTURE`).
- **Rooms:** ✅ excellent. **No articulation.**

## 5. HSSD / Habitat + ReplicaCAD

**5a. ReplicaCAD — the most license-clean articulated source.** 84/105 arrangements; **90+ objects with convex collision geometry + mass/friction/restitution + receptacle metadata**; **6+ URDF assets defining articulated furniture and door properties**; 6 stage shells (walls/floor) + `scene_instance.json`. **License: ✅ CC BY 4.0** ("All materials are licensed under the CC BY 4.0 Public License"). Access: `aihabitat.org/datasets/replica_cad/`, HF `ai-habitat/ReplicaCAD_dataset` (132 MB), `python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset`, ManiSkill `haosulab/ReplicaCAD`. **Rooms:** ✅.

**5b. HSSD** — 211 scenes, 18k+ models; **v0.3.0 added "2000+ articulated assets in the `urdf/` directory"** (drawers, cabinets, appliances) with receptacle meshes + **202 articulated scenes (161 public)** + PARTNR. Format: `urdf/<hash>/<hash>.urdf` + `<name>_door01.glb` + `<name>_shelf_receptacle_mesh.glb` + `.ao_config.json`. **License: ⚠️ CC BY-NC 4.0**. Access: HF `hssd/hssd-hab` (92k downloads), `hssd/hssd-models`, `hssd/hssd-scenes`.

## 6. Simulator-bundled assets

| Source | Articulated cabinets/drawers? | Format | License | Access |
|---|---|---|---|---|
| **RoboCasa** | ✅✅ **best-in-class**: `cabinet_hinge.xml`, `cabinet_housing.xml`, `cabinet_open.xml`, `cabinet_single.xml`, `drawer.xml`, `panel.xml`, handles, counters, **dishwasher/oven/fridge×3/microwave/stove/sink/blender/toaster** registry, `empty_kitchen_arena.xml`, kitchen layout YAMLs | **MJCF** + YAML layouts | ✅ **Assets CC BY 4.0**, code MIT | `python -m robocasa.scripts.download_kitchen_assets` (~10 GB) / HF `robocasa/robocasa-assets` / `haosulab/RoboCasa` |
| **ManiSkill 3** | ⬜ bundles RoboCasa/ReplicaCAD/AI2THOR + PartNet-Mobility objects | SAPIEN scene datasets | code Apache-2.0, **assets CC BY-NC 4.0** | `python -m mani_skill.utils.download_asset --list "scene"` |
| robosuite | ⬜ door hinges only | MJCF | MIT | `github.com/ARISE-Initiative/robosuite` |
| RLBench | ⬜ few tasks, CoppeliaSim `.ttm` | — | — | `github.com/stepjam/RLBench` |
| MetaWorld | ❌ rigid objects on a table | MJCF | MIT | `github.com/Farama-Foundation/Metaworld` |

## 7. Objaverse / Objaverse-XL
800K / 10M+ objects; **ODC-By 1.0** dataset license with per-object CC license fields (`'license': 'by'`). **❌ essentially no articulation**; I could **not verify any official PartNet-Mobility or articulated subset**. Practical use is indirect (RoboCasa's `objaverse.zip` clutter, Holodeck room filling).

## 8. Part-level / articulated datasets

| Source | Contents | Format | License | Access |
|---|---|---|---|---|
| **GAPartNet** | 8,489 parts on 1,166 objects, 27 categories, 9 GAPart classes; built on PartNet-Mobility | meshes + part annotations | ❌ NC; repo has **no LICENSE** | Google Form + ToS (`pku-epic.github.io/GAPartNet`); HF `jirufengyu/gapartnet` |
| **ArtVIP** | **476 digital-twin articulated objects**, 6 digital-twin + 6 interactive scenes, modular interaction types (Damping cabinet, Magnetic dishwasher/fridge), affordances | **USD** | ✅ **Apache-2.0** | HF `X-Humanoid/ArtVIP` (9.6k dl), arXiv 2506.04941 |
| **projectsim (Kaedim)** | procedurally generated articulated storage/appliances/wardrobes/desks; per-link mass+inertia from mesh integrals, physics materials, "MuJoCo-certified joints", friction latches | **MJCF** + USD + blend, 4K textures | ✅ **CC BY 4.0** (auto-gate) | HF `projectsim/articulated-storage-and-appliances`, `-wardrobes`, `-desks`, … |
| OPD | openable-part detection benchmark | detection only | MIT | `github.com/3dlg-hcvc/OPD` |
| URDFormer | image→URDF generator | URDF | MIT | `github.com/WEIRDLabUW/urdformer` |
| AKB-48 | ⚠️ **no live download channel verified** | — | — | — |

## 9. HF-hosted sim-ready assets

⭐ **`Genesis-Intelligence/assets`** (40k dl, card says **MIT**): **`fridge/fridge.urdf` + `fridge0015_door01.glb`/`e02` + shelf/top receptacle GLBs**, **`microwave/microwave.urdf` + door handle**, `Lightwheel_Kitchen/` (**Dishwasher054, Kitchen_Cabinet001/002, Kitchen_TopCabinet, Refrigerator001, Door012, Microwave017, RangeHood015** as `.usd`), `usd/Lightwheel_Kitchen001`, `usd/chair_array.usd`. ⚠️ The `_door01.glb` / `_receptacle_mesh.glb` naming matches HSSD exactly → provenance caveat. Also: `LocalWorldModels/partnet-mobility-urdf-parts` (unlabeled), `UniEnvOrg/unienv_genesis_collection_assets`.

## 10. Comparison table

| # | Source | Articulated coverage | Format | Genesis fit | License | Download | Rooms? | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **RoboCasa** | ⭑⭑⭑ cabinets+drawers+appliances | **MJCF** | ⭑⭑⭑ | ✅ CC BY 4.0 | ⭑⭑⭑ HF + CLI | ✅ kitchens | **Low** |
| 2 | **ReplicaCAD** | ⭑ (6+ URDF items) | URDF+GLB+JSON | ⭑⭑⭑ | ✅ CC BY 4.0 | ⭑⭑⭑ 132 MB | ✅ 84 arrangements | **Low** |
| 3 | **PartNet-Mobility** | ⭑⭑⭑ ~2K objects | URDF+V-HACD | ⭑⭑⭑ | ❌ gated NC | ⭑ gated | ❌ | Low–Med |
| 4 | **HSSD v0.3** | ⭑⭑⭑ 2000+ assets | URDF+GLB | ⭑⭑⭑ | ⚠️ CC BY-NC 4.0 | ⭑⭑⭑ | ✅ 161 scenes | Low |
| 5 | **ArtVIP** | ⭑⭑ 476 objects | USD | ⭑ | ✅ Apache-2.0 | ⭑⭑⭑ | ✅ 6+6 scenes | Med–High |
| 6 | **projectsim** | ⭑⭑ procedural furniture | **MJCF**+USD | ⭑⭑⭑ | ✅ CC BY 4.0 | ⭑⭑ | ❌ | Low |
| 7 | ManiSkill 3 | ⬜ bundler | SAPIEN | ⭑⭑⭑ | ⚠️ NC assets | ⭑⭑⭑ | ✅ | Low |
| 8 | Genesis assets | ⭑ fridge/microwave | URDF+USD | ⭑⭑⭑ | MIT ⚠️ | ⭑⭑⭑ | ⬜ | Very low |
| 9 | GAPartNet | ⭑⭑ 1,166 objects | annotations | ⭑⭑ | ❌ NC | ⭑ form | ❌ | Med |
| 10 | 3D-FRONT | ❌ rigid | JSON+OBJ | ⭑⭑ | ❌ academic | ⭑ dead pages | ✅ 18,968 | Med |
| 11 | AI2-THOR/ProcTHOR | ❌ rigid | GLB+JSON | ⭑ | unclear | ⭑⭑ gated | ✅ 12k+ | High |
| 12 | BEHAVIOR/iGibson | ⭑⭑⭑ **encrypted** | USD | ⭑ | ❌ commercial | ⭑ form | ✅ | **Blocked** |
| 13 | Objaverse | ❌ | GLB/USDZ | ⭑ | ODC-By | ⭑⭑⭑ | ❌ | High |

## 11. Recommendation — top 3

**🥇 #1 RoboCasa fixtures + kitchen layouts** (`robocasa.ai`, `github.com/robocasa/robocasa`, HF `robocasa/robocasa-assets`)
(a) unmatched kitchen articulation — hinged cabinet doors, sliding drawers, dishwasher/oven/fridge×3/microwave/stove/sink; (b) **MJCF already** → straight into `gs.morphs.MJCF`, zero joint authoring; (c) ✅ **Assets CC BY 4.0**; (d) ✅ HF zips + `mani_skill.utils.download_asset RoboCasa` + CLI. **Rooms:** ✅ walls/floor/counters + YAML layouts (120 → 2500). **Effort: Low.**

**🥈 #2 ReplicaCAD** (with HSSD as superset if NC is OK) (`aihabitat.org/datasets/replica_cad/`, HF `ai-habitat/ReplicaCAD_dataset`)
(c) ✅ **CC BY 4.0** — cleanest permissive licence among truly articulated, room-level sources; (a) 6+ URDF articulated items + 90+ objects with convex collision + mass/friction; HSSD v0.3 gives **2000+ articulated assets / 161 articulated scenes** if CC BY-NC is acceptable; (b) URDF + authored colliders → `gs.morphs.URDF`; (d) ✅ 132 MB, one-line CLI. **Effort: Low.**

**🥉 #3 PartNet-Mobility** (`huggingface.co/datasets/sapien-sim/PartNetMobility`, per-model zips on `storage1.ucsd.edu`)
(a) ✅ best **breadth** (~2K objects: cabinets, drawers, chairs, tables, fridges, dishwashers); (b) ✅ URDF + V-HACD precomputed → `prepare_ycb_assets.py` is a near-perfect template; (c) ❌ gated **non-commercial** (also taints GAPartNet + all mirrors); (d) ⚠️ manual gate w/ institutional email. **Rooms:** ❌ — compose yourself. Take only if the project is academic/non-commercial.

**Recommended combination:** RoboCasa (kitchen rooms + MJCF cabinets/drawers, CC BY 4.0) + ReplicaCAD (apartment rooms + URDF furniture, CC BY 4.0) + projectsim/ArtVIP (procedural extra articulation, CC BY 4.0/Apache-2.0) + Genesis `Genesis-Intelligence/assets` (Genesis-native `fridge.urdf`, `microwave.urdf`). Add HSSD + PartNet-Mobility only if non-commercial use is acceptable. This gives permissive licensing end-to-end, native MJCF/URDF, and both full rooms and single articulated objects with no USD authoring.

## 12. Caveats / unverified
1. **AI2-THOR asset licence** — Apache-2.0 code, but no asset-specific licence file and the HF export is gated.
2. **`Genesis-Intelligence/assets` MIT vs HSSD-derived geometry** — naming matches HSSD's articulated URDF layout exactly; confirm before redistributing.
3. **3D-FRONT official access** — canonical Tianchi pages dead; only HF mirrors remain.
4. **Objaverse articulated subset** — none verified; ODC-By covers metadata, objects carry their own licence.
5. **AKB-48** — no live project page, GitHub repo or HF dataset verified.
6. **Genesis USD support** — `genesis/utils/mjcf.py`, `urdf.py`, `usd/` exist; I did not verify completeness of USD→Genesis conversion, so USD-only sources (ArtVIP, Lightwheel) should assume a conversion step.

## 13. Amazon Berkeley Objects (ABO) — *added to the repo (rigid objects)*

**Status: implemented** — `objects/abo/<3dmodel_id>/` on the assets repo, descriptor `ABOObject` / `make_abo_pool` (`unienv_genesis_collection/objects/hub_objects/abo_object.py`), packager `scripts/prepare_abo_assets.py`, catalog + `box_gripper_graspable` bake (`scripts/data/abo_graspable.json`, `scripts/compute_graspable.py abo`).

- **Contents:** **7,953 modelled household products** (full ABO 3D coverage, no category filter) with embedded PBR materials — furniture, lighting, appliances, decor, toys, kitchenware. Single mesh, non-watertight scans (~24k verts / ~42k faces average, 2k–4k textures).
- **Access:** public S3 `https://amazon-berkeley-objects.s3.amazonaws.com/` — models at `3dmodels/original/<path>` (the `path` column of `3dmodels/metadata/3dmodels.csv.gz`; the shard letter is **not** derivable from the id), listings at `listings/metadata/listings_{0..9,a..f}.json.gz` (16 shards, ~85 MB; join key `3dmodel_id`, prefer the first `en_US` listing).
- **Format:** glTF 2.0 binary (GLB), **Y-up**, metric (metres), products standing on the `Y=0` plane. The packager bakes the Y-up → Z-up rotation into `model.obj` (the same `(X, Y, Z) → (X, -Z, Y)` mapping Genesis applies to Y-up formats) so the trimesh bbox path, the mesh frame and the placed frame agree; the OBJ keeps the base-colour texture as `texture.png` and drops the normal / metallic-roughness maps.
- **Collision:** no decomposition ships with the dataset → Genesis's own `convexify` (CoACD) builds it at load time from the mesh (`gs.morphs.Mesh`). A V-HACD/CoACD precompute step is deliberately deferred.
- **Licence: ✅ CC BY 4.0** — ABO was **relicensed by Amazon.com from CC BY-NC 4.0 to CC BY 4.0 in mid-2023**; the older CC BY-NC file is no longer served (404) and any note listing ABO as non-commercial (e.g. through third-party mirrors or older dataset cards) is stale. `objects/abo/LICENSE` is the bucket's verbatim `LICENSE-CC-BY-4.0.txt`; attribution = Amazon.com (data/images/models) plus the dataset builders (Guillaumin, Dideriksen, Deng, Arora, Collins, Malik — arXiv:2110.06199), recorded in `objects/abo/README.md`.
- **Size:** ~154 GB of GLBs at source; the prepared tree ships OBJ + one texture per object (~110 GB estimated) and is downloaded **lazily per object** through the existing `asset_path` machinery.
