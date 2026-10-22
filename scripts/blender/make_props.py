"""Generate the task props (dustpan, hand brush, shelf, paper balls).

Run headless with Blender (tested with 5.2)::

    blender -b --factory-startup --python scripts/blender/make_props.py

Then upload them with the other collection assets (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \
        unienv_genesis_collection/assets/objects/props objects/props --repo-type dataset

:mod:`unienv_genesis_collection.objects.props` loads them.

Writes ``unienv_genesis_collection/assets/objects/props/<asset>/``: one visual OBJ (+ MTL)
per asset (Nyx renders only a link's first ``<visual>``, so the brush's wood and
bristles share one multi-material OBJ), one convex collision OBJ per collision
piece, and a single-link URDF tying them together. Collision pieces are the un-bevelled convex hulls the
visuals are built from, so physics never depends on mesh decomposition.

Units are metres, +Z up. Wood textures are CC0 (Poly Haven ``oak_veneer_01``,
``cherry_veneer``), stored next to this script in ``textures/``.
"""

from __future__ import annotations

import math
import random
import shutil
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector, noise

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "unienv_genesis_collection" / "assets" / "objects" / "props"
TEXTURES = Path(__file__).resolve().parent / "textures"


# ----------------------------------------------------------------------------
# Mesh helpers
# ----------------------------------------------------------------------------


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def hull_bmesh(points) -> bmesh.types.BMesh:
    bm = bmesh.new()
    for p in points:
        bm.verts.new(Vector(p))
    result = bmesh.ops.convex_hull(bm, input=list(bm.verts), use_existing_faces=False)
    junk = {g for g in result["geom_interior"] + result["geom_unused"] if isinstance(g, bmesh.types.BMVert)}
    junk |= {v for v in bm.verts if not v.link_faces}
    if junk:
        bmesh.ops.delete(bm, geom=list(junk), context="VERTS")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def box_points(lo, hi):
    return [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]


def object_from_bmesh(name: str, bm: bmesh.types.BMesh) -> bpy.types.Object:
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def apply_modifiers(obj: bpy.types.Object) -> None:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(depsgraph)
    mesh = bpy.data.meshes.new_from_object(evaluated)
    obj.modifiers.clear()
    old = obj.data
    obj.data = mesh
    bpy.data.meshes.remove(old)


def bevel(obj: bpy.types.Object, width: float, segments: int = 3) -> None:
    mod = obj.modifiers.new("bevel", "BEVEL")
    mod.width = width
    mod.segments = segments
    mod.limit_method = "ANGLE"
    mod.angle_limit = math.radians(30)
    apply_modifiers(obj)


def drill(obj: bpy.types.Object, center, radius: float, axis: str = "Z", depth: float = 0.1) -> None:
    """Boolean-subtract a cylinder (visual-only holes, e.g. hanging holes)."""
    bm = bmesh.new()
    rot = {"Z": Matrix.Identity(4), "Y": Matrix.Rotation(math.pi / 2, 4, "X"), "X": Matrix.Rotation(math.pi / 2, 4, "Y")}[axis]
    bmesh.ops.create_cone(
        bm, cap_ends=True, segments=32, radius1=radius, radius2=radius, depth=depth,
        matrix=Matrix.Translation(Vector(center)) @ rot,
    )
    cutter = object_from_bmesh(obj.name + "_cutter", bm)
    mod = obj.modifiers.new("drill", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.object = cutter
    mod.solver = "EXACT"
    apply_modifiers(obj)
    bpy.data.objects.remove(cutter)


def smooth_by_angle(obj: bpy.types.Object, angle_deg: float = 35.0) -> None:
    obj.data.shade_smooth()
    obj.data.set_sharp_from_angle(angle=math.radians(angle_deg))


def join(name: str, parts) -> bpy.types.Object:
    bm = bmesh.new()
    for part in parts:
        bm.from_mesh(part.data)
    for part in parts:
        bpy.data.objects.remove(part)
    obj = object_from_bmesh(name, bm)
    return obj


def box_uv(obj: bpy.types.Object, tile: float, grain_axis: int = 1) -> None:
    """Planar UV projection per face along its dominant normal axis.

    ``tile`` is the texture's physical size (m). The texture's grain runs
    along v; ``grain_axis`` is the world axis it follows on faces whose normal
    allows it (the other faces use the remaining axis).
    """
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    uv = bm.loops.layers.uv.verify()
    for face in bm.faces:
        n = [abs(c) for c in face.normal]
        normal_axis = max(range(3), key=lambda i: n[i])
        plane = [i for i in range(3) if i != normal_axis]
        v_axis = grain_axis if grain_axis in plane else plane[1]
        u_axis = plane[0] if plane[0] != v_axis else plane[1]
        for loop in face.loops:
            co = loop.vert.co
            loop[uv].uv = (co[u_axis] / tile + 0.37, co[v_axis] / tile + 0.21)
    bm.to_mesh(obj.data)
    bm.free()


# ----------------------------------------------------------------------------
# Materials / export
# ----------------------------------------------------------------------------


def material(name: str, color=(0.8, 0.8, 0.8, 1.0), roughness: float = 0.6, texture: Path | None = None):
    mat = bpy.data.materials.new(name)
    nodes = mat.node_tree.nodes
    bsdf = nodes.get("Principled BSDF") or nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    if texture is not None:
        tex = nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(texture))
        mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def assign(obj: bpy.types.Object, mat) -> None:
    obj.data.materials.clear()
    obj.data.materials.append(mat)


def join_objects(name: str, objs) -> bpy.types.Object:
    """Join objects into the first one, keeping each part's material slots and UVs."""
    for other in bpy.context.scene.objects:
        other.select_set(False)
    for obj in objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    objs[0].name = name
    return objs[0]


def export_obj(obj: bpy.types.Object, path: Path, *, visual: bool) -> None:
    for other in bpy.context.scene.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.wm.obj_export(
        filepath=str(path),
        export_selected_objects=True,
        export_materials=visual,
        export_uv=visual,
        export_normals=visual,
        export_triangulated_mesh=True,
        forward_axis="Y",
        up_axis="Z",
        path_mode="STRIP",
    )


class Asset:
    """Collects visual / collision parts of one rigid asset and writes its URDF."""

    def __init__(self, name: str):
        self.name = name
        self.dir = OUT / name
        shutil.rmtree(self.dir, ignore_errors=True)
        self.dir.mkdir(parents=True)
        self.visuals: list[tuple[str, tuple]] = []
        self.collisions: list[str] = []

    def visual(self, obj: bpy.types.Object, part: str, mat, *, rgba=(1, 1, 1, 1), texture: Path | None = None) -> None:
        """Export ``obj`` as one visual; ``mat=None`` keeps the materials already on it."""
        if mat is not None:
            assign(obj, mat)
        fname = f"{self.name}_{part}.obj"
        export_obj(obj, self.dir / fname, visual=True)
        if texture is not None:
            # path_mode="STRIP" writes the bare file name; ship a uniquely named copy.
            tex_name = f"{self.name}_{part}_{texture.name}"
            shutil.copy(texture, self.dir / tex_name)
            mtl = self.dir / fname.replace(".obj", ".mtl")
            mtl.write_text(mtl.read_text().replace(texture.name, tex_name))
        self.visuals.append((fname, rgba if texture is None else None))
        bpy.data.objects.remove(obj)

    def collision(self, points, part: str) -> None:
        obj = object_from_bmesh(f"{self.name}_col_{part}", hull_bmesh(points))
        fname = f"{self.name}_collision_{part}.obj"
        export_obj(obj, self.dir / fname, visual=False)
        self.collisions.append(fname)
        bpy.data.objects.remove(obj)

    def write_urdf(self, mass: float, com=(0.0, 0.0, 0.0), inertia_box=(0.1, 0.1, 0.1)) -> None:
        sx, sy, sz = inertia_box
        ixx = mass * (sy * sy + sz * sz) / 12.0
        iyy = mass * (sx * sx + sz * sz) / 12.0
        izz = mass * (sx * sx + sy * sy) / 12.0
        lines = [
            '<?xml version="1.0"?>',
            f'<robot name="{self.name}">',
            '  <link name="base_link">',
            "    <inertial>",
            f'      <origin xyz="{com[0]:.4f} {com[1]:.4f} {com[2]:.4f}" rpy="0 0 0"/>',
            f'      <mass value="{mass:.4f}"/>',
            f'      <inertia ixx="{ixx:.3e}" ixy="0" ixz="0" iyy="{iyy:.3e}" iyz="0" izz="{izz:.3e}"/>',
            "    </inertial>",
        ]
        for fname, rgba in self.visuals:
            lines += [
                "    <visual>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{fname}"/></geometry>',
            ]
            if rgba is not None:
                lines.append(
                    f'      <material name="{fname[:-4]}"><color rgba="{rgba[0]:.3f} {rgba[1]:.3f} {rgba[2]:.3f} {rgba[3]:.3f}"/></material>'
                )
            lines.append("    </visual>")
        for fname in self.collisions:
            lines += [
                "    <collision>",
                '      <origin xyz="0 0 0" rpy="0 0 0"/>',
                f'      <geometry><mesh filename="{fname}"/></geometry>',
                "    </collision>",
            ]
        lines += ["  </link>", "</robot>", ""]
        (self.dir / f"{self.name}.urdf").write_text("\n".join(lines))


# ----------------------------------------------------------------------------
# Dustpan — mouth at x = 0 facing -X, handle toward +X, origin on the floor.
# ----------------------------------------------------------------------------

DUSTPAN_W = 0.24          # mouth width (y)
DUSTPAN_D = 0.20          # pan depth (x)
DUSTPAN_H = 0.075         # back wall height
DUSTPAN_FLOOR = 0.003     # floor thickness
DUSTPAN_WALL = 0.004


def make_dustpan() -> None:
    asset = Asset("dustpan")
    w, d, h, t, wall = DUSTPAN_W / 2.0, DUSTPAN_D, DUSTPAN_H, DUSTPAN_FLOOR, DUSTPAN_WALL
    pieces = {
        # Floor with a knife-edge lip so debris slides up onto it.
        "floor": [(0.0, s * w, 0.0) for s in (-1, 1)]
        + [(0.0, s * w, 0.0006) for s in (-1, 1)]
        + [(0.03, s * w, t) for s in (-1, 1)]
        + [(d, s * w, z) for s in (-1, 1) for z in (0.0, t)],
        "back": box_points((d - wall, -w - wall, 0.0), (d, w + wall, h))
        + [(d + 0.012, s * (w + wall), h) for s in (-1, 1)],
        "hood": box_points((d - 0.035, -w - wall, h - 0.004), (d + 0.012, w + wall, h)),
        "handle": [(d + 0.005, s * 0.020, z) for s in (-1, 1) for z in (h - 0.022, h - 0.008)]
        + [(d + 0.135, s * 0.015, z) for s in (-1, 1) for z in (h - 0.006, h + 0.006)],
    }
    for side, s in (("left", 1), ("right", -1)):
        inner, outer = s * w, s * (w + wall)
        pieces[f"side_{side}"] = [
            (x, y, z)
            for y in (inner, outer)
            for (x, z) in ((0.004, 0.0), (0.004, 0.006), (d, 0.0), (d, h))
        ]

    plastic = material("dustpan_plastic", (0.16, 0.36, 0.55, 1.0), roughness=0.45)
    parts = []
    for part, points in pieces.items():
        asset.collision(points, part)
        obj = object_from_bmesh(f"dustpan_vis_{part}", hull_bmesh(points))
        if part == "handle":
            drill(obj, (d + 0.115, 0.0, h), 0.0065, axis="Z")
        bevel(obj, 0.0012 if part == "floor" else 0.0018, segments=2)
        parts.append(obj)
    # Ribs on the floor (visual only) for a moulded-plastic look.
    for i in range(7):
        x = 0.05 + i * 0.02
        rib = object_from_bmesh(f"rib{i}", hull_bmesh(box_points((x, -w + 0.01, t), (x + 0.004, w - 0.01, t + 0.0012))))
        parts.append(rib)
    body = join("dustpan_body", parts)
    smooth_by_angle(body)
    asset.visual(body, "body", plastic, rgba=(0.16, 0.36, 0.55, 1.0))
    asset.write_urdf(mass=0.25, com=(0.12, 0.0, 0.02), inertia_box=(0.3, 0.24, 0.08))


# ----------------------------------------------------------------------------
# Hand brush — long axis +X (handle at +X), bristles down, origin at the
# bristle tips' centre. Sweeps sideways (±Y).
# ----------------------------------------------------------------------------

BRISTLE_LEN = 0.030
STOCK_TOP = 0.048
HANDLE_TOP_NEAR = 0.068
HANDLE_TOP_FAR = 0.078


def make_brush() -> None:
    asset = Asset("hand_brush")
    half_x, half_y = 0.080, 0.020
    stock = box_points((-0.090, -0.025, BRISTLE_LEN), (0.090, 0.025, STOCK_TOP))
    # A tall, slightly rising handle: easy to pinch from above without the
    # fingertips reaching the table.
    handle = [
        (x, s * hy, z)
        for (x, hy, z_low, z_high) in (
            (0.085, 0.0130, BRISTLE_LEN + 0.002, HANDLE_TOP_NEAR),
            (0.205, 0.0110, BRISTLE_LEN + 0.012, HANDLE_TOP_FAR),
        )
        for s in (-1, 1)
        for z in (z_low, z_high)
    ]
    bristles = box_points((-half_x, -half_y, 0.0), (half_x, half_y, BRISTLE_LEN + 0.001))
    asset.collision(bristles, "bristles")
    asset.collision(stock, "stock")
    asset.collision(handle, "handle")

    wood = material("hand_brush_wood", roughness=0.5, texture=TEXTURES / "cherry_veneer.png")
    stock_obj = object_from_bmesh("stock", hull_bmesh(stock))
    bevel(stock_obj, 0.006, segments=4)
    handle_obj = object_from_bmesh("handle", hull_bmesh(handle))
    drill(handle_obj, (0.188, 0.0, 0.060), 0.0045, axis="Y")
    bevel(handle_obj, 0.0045, segments=4)
    wood_obj = join("brush_wood", [stock_obj, handle_obj])
    smooth_by_angle(wood_obj, 50.0)
    box_uv(wood_obj, tile=1.0, grain_axis=0)
    assign(wood_obj, wood)

    # Bristle tufts: a dense grid of slightly flared, jittered cylinders.
    rng = random.Random(7)
    bm = bmesh.new()
    nx, ny = 17, 5
    for i in range(nx):
        for j in range(ny):
            x = -half_x + 0.004 + i * (2 * half_x - 0.008) / (nx - 1)
            y = -half_y + 0.003 + j * (2 * half_y - 0.006) / (ny - 1)
            length = BRISTLE_LEN + 0.002 - rng.uniform(0.0, 0.003)
            tilt = Matrix.Rotation(rng.uniform(-0.06, 0.06), 4, "X") @ Matrix.Rotation(rng.uniform(-0.06, 0.06), 4, "Y")
            mat = Matrix.Translation((x, y, BRISTLE_LEN + 0.002 - length / 2.0)) @ tilt
            bmesh.ops.create_cone(
                bm, cap_ends=True, segments=7, radius1=0.0036, radius2=0.0028, depth=length, matrix=mat
            )
    tufts = object_from_bmesh("tufts", bm)
    smooth_by_angle(tufts, 60.0)
    fibre = material("hand_brush_bristles", (0.12, 0.11, 0.10, 1.0), roughness=0.85)
    assign(tufts, fibre)
    # One visual mesh with both materials: Nyx reads URDFs itself and keeps only
    # the first <visual> of a link, which dropped a separate bristle mesh.
    body = join_objects("brush", [wood_obj, tufts])
    asset.visual(body, "body", None, texture=TEXTURES / "cherry_veneer.png")
    asset.write_urdf(mass=0.14, com=(0.04, 0.0, 0.040), inertia_box=(0.29, 0.05, 0.06))


# ----------------------------------------------------------------------------
# Shelf — a small open wooden two-tier shelf; the top board is the goal surface.
# Origin at the bottom centre; width along Y.
# ----------------------------------------------------------------------------

SHELF_W = 0.40      # along y
SHELF_D = 0.22      # along x
SHELF_H = 0.18      # top surface height
BOARD = 0.018
LOWER_BOARD_Z = 0.075  # top of the lower board


def make_shelf() -> None:
    asset = Asset("shelf")
    w, dd, h, b = SHELF_W / 2.0, SHELF_D / 2.0, SHELF_H, BOARD
    pieces = {
        "top": box_points((-dd - 0.006, -w - 0.008, h - b), (dd + 0.006, w + 0.008, h)),
        "side_left": box_points((-dd, w - b, 0.0), (dd, w, h - b)),
        "side_right": box_points((-dd, -w, 0.0), (dd, -w + b, h - b)),
        "lower": box_points((-dd + 0.004, -w + b, LOWER_BOARD_Z - b), (dd - 0.004, w - b, LOWER_BOARD_Z)),
        "rail": box_points((dd - 0.016, -w + b, 0.012), (dd - 0.004, w - b, 0.040)),
    }
    oak = material("shelf_oak", roughness=0.55, texture=TEXTURES / "oak_veneer_01.png")
    parts = []
    for part, points in pieces.items():
        asset.collision(points, part)
        obj = object_from_bmesh(f"shelf_{part}", hull_bmesh(points))
        bevel(obj, 0.0025, segments=3)
        parts.append(obj)
    body = join("shelf_body", parts)
    smooth_by_angle(body, 35.0)
    box_uv(body, tile=1.83, grain_axis=1)
    asset.visual(body, "body", oak, texture=TEXTURES / "oak_veneer_01.png")
    asset.write_urdf(mass=2.0, com=(0.0, 0.0, 0.09), inertia_box=(0.22, 0.40, 0.18))


# ----------------------------------------------------------------------------
# Crumpled paper balls
# ----------------------------------------------------------------------------

PAPER_VARIANTS = (
    ("paper_ball_0", 0.022, (0.95, 0.95, 0.93, 1.0), 11),
    ("paper_ball_1", 0.020, (0.96, 0.93, 0.80, 1.0), 23),
    ("paper_ball_2", 0.024, (0.86, 0.91, 0.97, 1.0), 37),
)


def make_paper(name: str, radius: float, rgba, seed: int) -> None:
    asset = Asset(name)
    rng = random.Random(seed)
    offset = Vector((rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-50, 50)))
    # Flattened wads: crumpled paper slides rather than rolls when swept.
    squash = Vector((rng.uniform(0.95, 1.1), rng.uniform(0.95, 1.1), rng.uniform(0.65, 0.75)))
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=3, radius=radius)
    for v in bm.verts:
        n = v.co.normalized()
        crumple = noise.fractal(n * 2.2 + offset, 0.9, 2.1, 4)
        ridge = abs(noise.noise(n * 5.0 + offset * 1.7))
        r = radius * (1.0 + 0.22 * crumple - 0.12 * ridge)
        v.co = Vector((n.x * r * squash.x, n.y * r * squash.y, n.z * r * squash.z))
    # Rest on z = 0 (origin at the bottom of the ball's hull).
    min_z = min(v.co.z for v in bm.verts)
    for v in bm.verts:
        v.co.z -= min_z
    hull_points = [tuple(v.co) for v in bm.verts]
    obj = object_from_bmesh(name, bm)
    obj.data.shade_flat()
    asset.collision(hull_points, "hull")
    paper = material(f"{name}_paper", rgba, roughness=0.9)
    asset.visual(obj, "paper", paper, rgba=rgba)
    asset.write_urdf(mass=0.008, com=(0.0, 0.0, radius * 0.9), inertia_box=(2 * radius,) * 3)


def main() -> None:
    reset_scene()
    make_dustpan()
    reset_scene()
    make_brush()
    reset_scene()
    make_shelf()
    for variant in PAPER_VARIANTS:
        reset_scene()
        make_paper(*variant)
    print("ASSETS WRITTEN TO", OUT)


main()
