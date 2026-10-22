"""Tests for the Blender-made task props (no asset downloads, no scene builds).

A fake prop directory (a single-link URDF with two convex collision boxes and
one visual mesh) is installed in a temporary assets dir under the real
``objects/props/<name>`` layout.

Covers:
  - Factory defaults (fixed / free, names, language annotations)
  - Offline-safe construction (nothing is downloaded or read)
  - URDF parsing: collision / visual files, geometry = union of the collision pieces
  - Argument validation
"""

from pathlib import Path

import pytest
import trimesh

from unienv_genesis_collection import asset_path
from unienv_genesis_collection.objects import UrdfPropObject, props


_URDF = """\
<robot name="{name}">
    <link name="base_link">
        <visual><geometry><mesh filename="{name}_body.obj" /></geometry></visual>
        <collision><geometry><mesh filename="{name}_collision_a.obj" /></geometry></collision>
        <collision><geometry><mesh filename="{name}_collision_b.obj" /></geometry></collision>
    </link>
</robot>
"""


def _make_fake_prop(assets_dir: Path, name: str) -> Path:
    prop_dir = assets_dir / props.PROPS_ASSETS_ROOT / name
    prop_dir.mkdir(parents=True)
    (prop_dir / f"{name}.urdf").write_text(_URDF.format(name=name))
    # Collision pieces: two boxes side by side along x, spanning x in [0, 0.3].
    for suffix, centre in (("a", 0.05), ("b", 0.25)):
        box = trimesh.creation.box(extents=(0.1, 0.2, 0.05))
        box.apply_translation((centre, 0.0, 0.025))
        box.export(prop_dir / f"{name}_collision_{suffix}.obj")
    trimesh.creation.box(extents=(0.3, 0.2, 0.05)).export(prop_dir / f"{name}_body.obj")
    return prop_dir


@pytest.fixture
def assets_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(asset_path, "ASSETS_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def no_download(monkeypatch):
    """Fail the test if a Hub download is attempted."""

    def _fail(*args, **kwargs):
        raise AssertionError(f"Unexpected download attempt: {args} {kwargs}")

    monkeypatch.setattr(asset_path, "download_assets", _fail)
    monkeypatch.setattr(asset_path, "download_asset", _fail)


def test_factories_construct_offline(assets_dir, no_download):
    """Construction neither downloads nor reads anything (the dir does not exist)."""
    pan = props.dustpan()
    assert isinstance(pan, UrdfPropObject)
    assert pan.name == "dustpan" and not pan.fixed
    assert props.dustpan(fixed=True).fixed
    assert props.dustpan(name="pan").name == "pan"
    assert not props.hand_brush().fixed
    assert props.shelf().fixed
    assert props.paper_ball(2).asset == "paper_ball_2"
    assert "dustpan" in pan.language_annotations


def test_every_prop_name_has_a_factory(assets_dir, no_download):
    built = {
        props.dustpan().asset,
        props.hand_brush().asset,
        props.shelf().asset,
        *(props.paper_ball(i).asset for i in range(props.PAPER_BALL_VARIANTS)),
    }
    assert built == set(props.PROP_NAMES)


def test_geometry_is_the_union_of_the_collision_pieces(assets_dir, no_download):
    prop_dir = _make_fake_prop(assets_dir, "dustpan")
    pan = props.dustpan()
    assert pan.urdf_file == prop_dir / "dustpan.urdf"
    assert [path.name for path in pan.collision_files] == ["dustpan_collision_a.obj", "dustpan_collision_b.obj"]
    assert [path.name for path in pan.visual_files] == ["dustpan_body.obj"]
    low, high = pan._load_trimesh().bounds
    assert low == pytest.approx((0.0, -0.1, 0.0))
    assert high == pytest.approx((0.3, 0.1, 0.05))
    assert props.surface_mesh(pan).bounds[1] == pytest.approx((0.15, 0.1, 0.025))


def test_invalid_arguments():
    with pytest.raises(KeyError):
        UrdfPropObject("broom")
    with pytest.raises(ValueError):
        props.paper_ball(props.PAPER_BALL_VARIANTS)
