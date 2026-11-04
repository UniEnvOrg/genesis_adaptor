"""Build LEAP Hand v2 Adv MJCFs (hand alone, and on the FR3) from the upstream URDFs.

Upstream: github.com/leap-hand/LEAP_Hand_V2_Adv_API, ``leap_v2/leap_v2/leap_v2_{left,right}/`` (an Onshape export:
``robot.urdf`` + STL meshes). The repository has no license file: keep the generated files local until the authors
allow redistribution.

Writes, under ``unienv_genesis_collection/assets/robots/``:

* ``leap_v2_adv/<side>/``: simplified visual meshes, convex-hull collision meshes, ``<side>_hand.xml``;
* ``franka_fr3_leap/fr3_leap_<side>.xml``: the FR3 arm of ``franka_fr3_hand/fr3_hand.xml`` with the hand on the flange.

Conversion:

* **Root.** The URDF's root link is the thumb-side lower palm; the back housing with the wrist mount face (the URDF
  origin, at its top centre) belongs to the four-finger frame, which hinges off the root (``palm_4_finger``). The
  tree is re-rooted at the four-finger frame (``leap_base``), so the mount stays put and the palm fold moves the
  thumb side (``thumb_palm``, joint ``palm_4_finger`` with its sign kept: same angle, same motion of one half of the
  palm relative to the other).
* **Names.** Bodies are named after what they are: ``leap_base``, ``thumb_palm``, ``<finger>_{knuckle,proximal,middle,
  distal}`` (the children of joints ``<finger>_{mcpf,mcps,pip,dip}``; the thumb: ``thumb_{base,knuckle,proximal,
  distal}`` of ``palm_thumb``, ``thumb_{mcpf,mcps,ip}``), ``<finger>_tip`` (the URDF's fixed ``realtip`` links), and
  a ``palm`` frame body on ``leap_base``: origin at the centre of the four finger bases, +x out of the palm (the side
  the fingers curl to), +z along the fingers (the Wuji / Allegro palm convention).
* **Tendons.** Each finger's PIP and DIP share one tendon on the hardware: ``<finger>_dip = <finger>_pip`` (an equality
  constraint); the 17 other joints have position actuators (the hardware's 17 motors).
* **Masses.** The URDF's fixed fingertip / finger-pad marker links (``realtip*``, ``realdip*``) carry 0.1 kg each
  (1 kg in all, at the fingertips); they become massless frames (the hand: 1.5 kg).
* **Meshes.** Visuals over ``VISUAL_FACES`` faces (the four-finger frame: 227k faces in 309 pieces; the thumb frame:
  60k) are simplified to that by MeshLab's quadric edge collapse with topology and normals preserved (a cruder
  reduction to a few thousand faces tears the many small pieces apart); collisions are the links' convex hulls.
* **Mount.** On the back of the hand: the mount face on the FR3 flange (+5 mm adapter), the palm facing along the
  flange axis, the fingers along the Franka Hand's x axis (the flange turned -45 deg); a flange pointing down holds
  the palm down, fingers level.

Run from the repository root:  python scripts/build_fr3_leap_mjcf.py <path to LEAP_Hand_V2_Adv_API checkout>
"""

from __future__ import annotations

import copy
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROBOTS = pathlib.Path(__file__).resolve().parents[1] / "unienv_genesis_collection" / "assets" / "robots"
VISUAL_FACES = 30000
FINGERS = ("index", "middle", "ring", "pinky")
FLANGE_Z = 0.107
ADAPTER = 0.005
FR3_HOME = "0 0 0 -1.57079 0 1.57079 -0.7853"
NEW_ROOT = "4_finger_frame"  # link-name prefix of the four-finger frame (mirrored names on one side)


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _T(el) -> np.ndarray:
    T = np.eye(4)
    if el is not None:
        T[:3, :3] = _rpy(*[float(v) for v in el.get("rpy", "0 0 0").split()])
        T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    return T


def _quat(R: np.ndarray) -> np.ndarray:
    """wxyz of a rotation matrix."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2]); y = np.copysign(y, R[0, 2] - R[2, 0]); z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z]); return q / np.linalg.norm(q)


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def _pose(el: ET.Element, T: np.ndarray) -> None:
    el.set("pos", _fmt(T[:3, 3])); el.set("quat", _fmt(_quat(T[:3, :3])))


class _Urdf:
    def __init__(self, path: pathlib.Path):
        self.dir = path.parent
        root = ET.parse(path).getroot()
        self.links = {l.get("name"): l for l in root.findall("link")}
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.children = {}
        for j in root.findall("joint"):
            self.children.setdefault(j.find("parent").get("link"), []).append(j)


def _names(u: _Urdf) -> dict:
    """``{urdf link: new body name}``."""
    out = {}
    for f in FINGERS:
        for joint, part in (("mcpf", "knuckle"), ("mcps", "proximal"), ("pip", "middle"), ("dip", "distal")):
            out[u.joints[f"{f}_{joint}"].find("child").get("link")] = f"{f}_{part}"
    for joint, part in (("palm_thumb", "base"), ("thumb_mcpf", "knuckle"), ("thumb_mcps", "proximal"), ("thumb_ip", "distal")):
        out[u.joints[joint].find("child").get("link")] = f"thumb_{part}"
    out[u.joints["palm_4_finger"].find("child").get("link")] = "leap_base"
    out[u.joints["palm_4_finger"].find("parent").get("link")] = "thumb_palm"
    # Fixed tip / dip-pad links: named after the finger body they hang off.
    for link, j in u.parent_joint.items():
        if j.get("type") == "fixed":
            parent = out.get(j.find("parent").get("link"), j.find("parent").get("link"))
            kind = "tip" if "tip" in link else "pad"
            out[link] = parent.rsplit("_", 1)[0] + f"_{kind}"
    return out


def _mesh(u: _Urdf, link: str, out_dir: pathlib.Path, name: str):
    """Write ``<name>.stl`` (simplified) and ``<name>_hull.stl``; return the visual origin transform or None."""
    import pymeshlab
    import trimesh

    v = u.links[link].find("visual")
    if v is None or v.find("geometry/mesh") is None:
        return None
    m = trimesh.load(u.dir / v.find("geometry/mesh").get("filename").replace("package:///", ""), force="mesh")
    if len(m.faces) > VISUAL_FACES:
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(m.vertices, m.faces))
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=VISUAL_FACES, preservetopology=True, preservenormal=True,
                                                    preserveboundary=True, planarquadric=True, qualitythr=0.3)
        out = ms.current_mesh()
        m = trimesh.Trimesh(out.vertex_matrix(), out.face_matrix())
    m.export(out_dir / f"{name}.stl")
    m.convex_hull.export(out_dir / f"{name}_hull.stl")
    rgba = v.find("material/color")
    return _T(v.find("origin")), (rgba.get("rgba") if rgba is not None else "0.8 0.8 0.8 1")


def build_hand(src: pathlib.Path, side: str) -> pathlib.Path:
    u = _Urdf(src / f"leap_v2_{side}" / "robot.urdf")
    names = _names(u)
    out_dir = ROBOTS / "leap_v2_adv" / side
    out_dir.mkdir(parents=True, exist_ok=True)

    mj = ET.Element("mujoco", model=f"leap_v2_adv_{side}")
    mj.append(ET.Comment(" Generated by scripts/build_fr3_leap_mjcf.py from LEAP_Hand_V2_Adv_API; do not edit by hand. "))
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true", meshdir=".")
    default = ET.SubElement(ET.SubElement(mj, "default"), "default", {"class": "leap"})
    ET.SubElement(default, "joint", damping="0.05", armature="0.002", frictionloss="0.001")
    ET.SubElement(default, "position", kp="2", kv="0.05")
    ET.SubElement(ET.SubElement(default, "default", {"class": "leap_visual"}), "geom", type="mesh", contype="0", conaffinity="0", group="2")
    ET.SubElement(ET.SubElement(default, "default", {"class": "leap_collision"}), "geom", type="mesh", group="3", friction="1.0 0.02 0.001")
    asset = ET.SubElement(mj, "asset")
    world = ET.SubElement(mj, "worldbody")

    # Every link's pose at q = 0 in the URDF root frame (for re-rooting).
    def link_T(link):
        T = np.eye(4)
        while link in u.parent_joint:
            j = u.parent_joint[link]; T = _T(j.find("origin")) @ T; link = j.find("parent").get("link")
        return T

    new_root = u.joints["palm_4_finger"].find("child").get("link")
    old_root = u.joints["palm_4_finger"].find("parent").get("link")
    T_root = link_T(new_root)  # the new root's frame in URDF-root coordinates

    def add_link(parent_el, link, T_parent_world, joint=None, joint_axis=None, joint_anchor=None):
        T_world = link_T(link)
        body = ET.SubElement(parent_el, "body", name=names.get(link, link), childclass="leap")
        _pose(body, np.linalg.inv(T_parent_world) @ T_world)
        if joint is not None:
            lim = joint.find("limit")
            j_el = ET.SubElement(body, "joint", name=joint.get("name"), axis=_fmt(joint_axis), range=f"{lim.get('lower')} {lim.get('upper')}")
            if joint_anchor is not None:
                j_el.set("pos", _fmt(joint_anchor))
        inertial = u.links[link].find("inertial")
        marker = link in u.parent_joint and u.parent_joint[link].get("type") == "fixed"
        if inertial is not None and not marker:  # the fixed tip / pad markers carry a 0.1 kg placeholder each
            Ti = _T(inertial.find("origin")); I = inertial.find("inertia")
            ET.SubElement(body, "inertial", pos=_fmt(Ti[:3, 3]), mass=inertial.find("mass").get("value"),
                          fullinertia=_fmt([float(I.get(k)) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")]))
        mesh = _mesh(u, link, out_dir, names.get(link, link))
        if mesh is not None:
            Tv, rgba = mesh
            name = names.get(link, link)
            ET.SubElement(asset, "mesh", name=f"leap_{name}", file=f"{name}.stl")
            ET.SubElement(asset, "mesh", name=f"leap_{name}_hull", file=f"{name}_hull.stl")
            vis = ET.SubElement(body, "geom", {"class": "leap_visual", "mesh": f"leap_{name}", "rgba": rgba}); _pose(vis, Tv)
            col = ET.SubElement(body, "geom", {"class": "leap_collision", "mesh": f"leap_{name}_hull"}); _pose(col, Tv)
        return body, T_world

    def add_subtree(parent_el, link, T_parent_world, skip=None):
        for j in u.children.get(link, []):
            child = j.find("child").get("link")
            if child == skip:
                continue
            if j.get("type") == "fixed":
                b, T = add_link(parent_el, child, T_parent_world)
                add_subtree(b, child, T)
                continue
            axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
            b, T = add_link(parent_el, child, T_parent_world, j, axis)
            add_subtree(b, child, T)

    # leap_base: the four-finger frame, at its own frame; its children as in the URDF.
    base, T_base = add_link(world, new_root, np.eye(4))  # posed in the URDF-root frame (the mount face is its origin)
    add_subtree(base, new_root, T_base)
    # The old root (thumb side of the palm) hangs off leap_base through palm_4_finger, inverted: in the old root's
    # frame the joint sits at the new root's origin, its axis turned into the old root's frame and negated (the old
    # root turns by -q relative to the new one; negating the axis keeps the joint's angle and range).
    j = u.joints["palm_4_finger"]
    To = _T(j.find("origin"))
    axis = np.array([float(v) for v in j.find("axis").get("xyz").split()])
    thumb_palm, T_tp = add_link(base, old_root, T_base, j, -(To[:3, :3] @ axis), To[:3, 3])
    add_subtree(thumb_palm, old_root, T_tp, skip=new_root)

    # Palm frame on leap_base: centre of the finger bases, +x out of the palm (-z of the URDF root), +z along the
    # fingers (-y of the URDF root).
    bases = np.array([link_T(u.joints[f"{f}_mcpf"].find("child").get("link"))[:3, 3] for f in FINGERS])
    T_palm = np.eye(4)
    T_palm[:3, :3] = np.stack([[0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0]], axis=1)
    T_palm[:3, 3] = bases.mean(0)
    palm = ET.SubElement(base, "body", name="palm"); _pose(palm, np.linalg.inv(T_base) @ T_palm)
    ET.SubElement(palm, "site", name="palm_site", size="0.005")

    equality = ET.SubElement(mj, "equality")
    for f in FINGERS:
        ET.SubElement(equality, "joint", joint1=f"{f}_dip", joint2=f"{f}_pip", polycoef="0 1 0 0 0", solref="0.005 1")
    actuator = ET.SubElement(mj, "actuator")
    for name in hand_joint_names():
        lim = u.joints[name].find("limit")
        ET.SubElement(actuator, "position", {"class": "leap", "name": name, "joint": name, "forcerange": f"-{lim.get('effort')} {lim.get('effort')}"})
    contact = ET.SubElement(mj, "contact")
    ET.SubElement(contact, "exclude", body1="leap_base", body2="thumb_palm")
    for f in FINGERS:
        ET.SubElement(contact, "exclude", body1="leap_base", body2=f"{f}_proximal")
    ET.SubElement(contact, "exclude", body1="thumb_palm", body2="thumb_knuckle")
    ET.SubElement(contact, "exclude", body1="leap_base", body2="thumb_base")
    path = out_dir / f"{side}_hand.xml"
    ET.indent(mj, space="  ")
    ET.ElementTree(mj).write(path, encoding="unicode")
    return path


def hand_joint_names():
    """The 17 actuated joints (each finger's DIP follows its PIP)."""
    return [f"{f}_{j}" for f in FINGERS for j in ("mcpf", "mcps", "pip")] + ["thumb_mcpf", "thumb_mcps", "thumb_ip", "palm_4_finger", "palm_thumb"]


def build_arm(side: str) -> pathlib.Path:
    hand = ET.parse(ROBOTS / "leap_v2_adv" / side / f"{side}_hand.xml").getroot()
    src = ET.parse(ROBOTS / "franka_fr3_hand" / "fr3_hand.xml").getroot()
    root = ET.Element("mujoco", model=f"fr3_leap_{side}")
    root.append(ET.Comment(" Generated by scripts/build_fr3_leap_mjcf.py; do not edit by hand. "))
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", integrator="implicitfast")
    default = copy.deepcopy(src.find("default"))
    for d in list(default):
        if d.get("class") == "hand":
            default.remove(d)
    default.append(copy.deepcopy(hand.find("default").find("default")))
    root.append(default)
    asset = ET.SubElement(root, "asset")
    for m in src.find("asset"):
        if isinstance(m.tag, str) and not ((m.tag == "mesh" and "franka_emika_panda" in m.get("file", "")) or m.get("name", "").startswith("hand_")):
            asset.append(copy.deepcopy(m))
    for m in hand.find("asset"):
        c = copy.deepcopy(m); c.set("file", f"../leap_v2_adv/{side}/{c.get('file')}"); asset.append(c)
    world = ET.SubElement(root, "worldbody")
    arm = copy.deepcopy(src.find("worldbody").find("body"))
    link7 = next(b for b in arm.iter("body") if b.get("name") == "fr3_link7")
    link7.remove(next(b for b in link7 if b.tag == "body" and b.get("name") == "hand"))
    base = copy.deepcopy(hand.find("worldbody").find("body"))
    # leap_base in the URDF-root frame is (pos, quat) as written; the URDF-root frame sits on the flange with its +z
    # back toward link7 and its -y (the fingers) along the Franka Hand's x (link7 x turned -45 deg about z).
    c = np.sqrt(0.5)
    R_mount = np.stack([[-c, -c, 0.0], [-c, c, 0.0], [0.0, 0.0, -1.0]], axis=1)  # columns: URDF-root x, y, z in link7
    T_mount = np.eye(4); T_mount[:3, :3] = R_mount; T_mount[:3, 3] = [0, 0, FLANGE_Z + ADAPTER]
    T_base = np.eye(4)
    T_base[:3, 3] = [float(v) for v in base.get("pos").split()]
    w, x, y, z = [float(v) for v in base.get("quat").split()]
    T_base[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                               [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                               [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    _pose(base, T_mount @ T_base)
    link7.append(base)
    world.append(arm)
    act = ET.SubElement(root, "actuator")
    act.extend(copy.deepcopy(a) for a in src.find("actuator") if isinstance(a.tag, str) and a.get("joint", "").startswith("fr3_"))
    act.extend(copy.deepcopy(a) for a in hand.find("actuator"))
    root.append(copy.deepcopy(hand.find("equality")))
    con = ET.SubElement(root, "contact")
    ET.SubElement(con, "exclude", body1="fr3_link0", body2="fr3_link1")
    ET.SubElement(con, "exclude", body1="fr3_link7", body2="leap_base")
    ET.SubElement(con, "exclude", body1="fr3_link6", body2="leap_base")
    con.extend(copy.deepcopy(e) for e in hand.find("contact"))
    home = " ".join(["0"] * 21)
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home", qpos=f"{FR3_HOME} {home}")
    path = ROBOTS / "franka_fr3_leap" / f"fr3_leap_{side}.xml"
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="unicode")
    return path


if __name__ == "__main__":
    src = pathlib.Path(sys.argv[1]).resolve() / "leap_v2" / "leap_v2"
    for s in ("left", "right"):
        print(build_hand(src, s))
        print(build_arm(s))
