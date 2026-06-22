"""Tests for EnvMapManager.

Covers:
  - Construction with explicit env_maps.
  - Manifest-backed list_assets() / from_asset().
  - reload() selection modes and metadata.
  - Via WorldEnv.reset(reload=True) the manager mutates world.env_map.
  - hdr_path=None raises ValueError.
"""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.options.light_option import GenesisWorldEnvMapOption
from unienv_genesis_collection.scenes import env_map_manager as emm_module
from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_interface.world import WorldEnv


def _make_dummy_hdrs(tmpdir, names):
    """Create empty dummy .hdr files and return their paths."""
    env_map_dir = os.path.join(tmpdir, "scenes", "env_maps")
    os.makedirs(env_map_dir, exist_ok=True)
    paths = []
    for name in names:
        path = os.path.join(env_map_dir, f"{name}.hdr")
        with open(path, "wb") as f:
            f.write(b"")
        paths.append(path)
    return paths


def _write_manifest(tmpdir, entries):
    env_map_dir = os.path.join(tmpdir, "scenes", "env_maps")
    os.makedirs(env_map_dir, exist_ok=True)
    manifest_path = os.path.join(env_map_dir, "info.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f)
    return manifest_path


@contextmanager
def _patched_asset_dir(tmpdir):
    original_dir = emm_module.asset_path.ASSETS_DIR
    try:
        emm_module.asset_path.ASSETS_DIR = Path(tmpdir)
        yield
    finally:
        emm_module.asset_path.ASSETS_DIR = original_dir


# ------------------------------------------------------------------
# Construction and manifest helpers
# ------------------------------------------------------------------

def test_construction_env_maps():
    """Pre-built GenesisWorldEnvMapOption are accepted directly."""
    world = GenesisWorld()
    opts = [
        GenesisWorldEnvMapOption(hdr_path="/tmp/x.hdr"),
        GenesisWorldEnvMapOption(hdr_path="/tmp/y.hdr"),
    ]
    mgr = EnvMapManager(world, env_maps=opts)
    assert mgr.pool_size == 2
    print("  PASS: construction with env_maps")
    world.scene.destroy()


def test_construction_empty_env_maps_raises():
    """Empty env_maps are rejected."""
    world = GenesisWorld()
    try:
        EnvMapManager(world, env_maps=[])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "at least one env map" in str(e).lower()
        print(f"  PASS: empty env_maps raises — {e}")
    world.scene.destroy()


def test_construction_none_hdr_path_raises():
    """An option with hdr_path=None raises ValueError."""
    world = GenesisWorld()
    try:
        EnvMapManager(world, env_maps=[GenesisWorldEnvMapOption(hdr_path=None)])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "hdr_path" in str(e).lower()
        print(f"  PASS: hdr_path=None raises — {e}")
    world.scene.destroy()


def test_list_assets_only_approved():
    """list_assets() returns only approved asset ids."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_manifest(
            tmp,
            {
                "approved_map": {
                    "hdr_path": "approved_map.hdr",
                    "rotation_deg": 45.0,
                    "status": "approved",
                },
                "rejected_map": {
                    "hdr_path": "rejected_map.hdr",
                    "rotation_deg": -10.0,
                    "status": "rejected",
                },
            },
        )
        with _patched_asset_dir(tmp):
            assert EnvMapManager.list_assets() == ["approved_map"]
            print("  PASS: list_assets filters to approved ids")


def test_from_asset_reads_rotation_and_path():
    """from_asset() resolves the HDR path and rotation from the manifest."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_dummy_hdrs(tmp, ["approved_map"])
        _write_manifest(
            tmp,
            {
                "approved_map": {
                    "hdr_path": "approved_map.hdr",
                    "rotation_deg": 45.0,
                    "status": "approved",
                },
            },
        )
        with _patched_asset_dir(tmp):
            opt = EnvMapManager.from_asset("approved_map")
            assert opt.hdr_path == os.path.join(tmp, "scenes", "env_maps", "approved_map.hdr")
            assert opt.hdr_rotation == 45.0
            print("  PASS: from_asset resolves approved asset")


def test_from_asset_rejected_raises():
    """from_asset() rejects non-approved manifest entries."""
    with tempfile.TemporaryDirectory() as tmp:
        _make_dummy_hdrs(tmp, ["rejected_map"])
        _write_manifest(
            tmp,
            {
                "rejected_map": {
                    "hdr_path": "rejected_map.hdr",
                    "rotation_deg": 0.0,
                    "status": "rejected",
                },
            },
        )
        with _patched_asset_dir(tmp):
            try:
                EnvMapManager.from_asset("rejected_map")
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "not approved" in str(e).lower()
                print(f"  PASS: rejected asset raises — {e}")


# ------------------------------------------------------------------
# Reload selection
# ------------------------------------------------------------------

def test_reload_sets_world_env_map():
    """reload() sets world.env_map to a pool member."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
            seed=0,
        )
        mgr.reload(priority=150, seed=0)
        assert mgr.current_index is not None
        assert world.env_map is not None
        assert world.env_map.hdr_path in paths
        assert world.env_map is mgr._pool[mgr.current_index]
        print(f"  PASS: reload sets world.env_map (idx={mgr.current_index})")
        world.scene.destroy()


def test_seed_reproducibility_random_selection():
    """Same seed -> same random selection."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c", "d", "e"])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
        )

        mgr.reload(priority=150, seed=123)
        idx_a = mgr.current_index

        mgr.reload(priority=150, seed=123)
        idx_b = mgr.current_index

        assert idx_a == idx_b, f"Same seed gave {idx_a} then {idx_b}"
        print(f"  PASS: seed reproducibility (idx={idx_a})")
        world.scene.destroy()


def test_different_seeds_differ_random_selection():
    """Different seeds produce (likely) different random selections."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, [f"hdri_{i}" for i in range(20)])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
        )

        indices = set()
        for seed in range(10):
            mgr.reload(priority=150, seed=seed)
            indices.add(mgr.current_index)
        assert len(indices) > 1, f"Only {len(indices)} distinct indices across 10 seeds"
        print(f"  PASS: different seeds -> {len(indices)} distinct selections")
        world.scene.destroy()


def test_selection_first_picks_first():
    """selection='first' always picks index 0."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
            selection="first",
        )
        for seed in range(5):
            mgr.reload(priority=150, seed=seed)
            assert mgr.current_index == 0, f"selection='first' but got idx={mgr.current_index}"
        print("  PASS: selection='first' picks index 0")
        world.scene.destroy()


def test_selection_cycle_advances_in_order():
    """selection='cycle' walks the pool in order."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
            selection="cycle",
        )
        indices = []
        for _ in range(5):
            mgr.reload(priority=150)
            indices.append(mgr.current_index)
        assert indices == [0, 1, 2, 0, 1]
        print(f"  PASS: selection='cycle' indices={indices}")
        world.scene.destroy()


# ------------------------------------------------------------------
# Via WorldEnv
# ------------------------------------------------------------------

def test_via_world_env_reload():
    """WorldEnv.reset(reload=True) triggers the manager and sets world.env_map."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c", "d"])
        world = GenesisWorld()
        plane = PlaneBuilder(world, name="floor")
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
            seed=0,
        )
        env = WorldEnv(world, [plane, mgr], render_mode=None)
        env.reset(seed=0, reload=True)

        assert mgr.current_index is not None
        assert world.env_map is not None
        assert world.env_map.hdr_path in paths

        first = mgr.current_index
        env.reset(seed=99, reload=True)
        assert mgr.current_index is not None
        assert world.env_map.hdr_path in paths

        info = mgr.get_info()
        assert info["env_map_selection"] == "random"
        assert info["env_map_index"] == mgr.current_index
        assert info["env_map_id"] == os.path.splitext(os.path.basename(world.env_map.hdr_path))[0]
        assert info["env_map_pool_size"] == 4

        print(f"  PASS: WorldEnv reload (first idx={first}, second idx={mgr.current_index})")
        env.close()


def test_via_world_env_priority_order():
    """Manager runs before camera priority (0). Verify priority is > 0."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a"])
        world = GenesisWorld()
        mgr = EnvMapManager(
            world,
            env_maps=[GenesisWorldEnvMapOption(hdr_path=p) for p in paths],
        )
        assert next(iter(mgr.reload_priorities)) > 0, (
            "EnvMapManager reload priority must be > 0 to run before cameras"
        )
        print(f"  PASS: reload priority = {next(iter(mgr.reload_priorities))} > 0")
        world.scene.destroy()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_construction_env_maps ===")
    test_construction_env_maps()
    print()
    print("=== test_construction_empty_env_maps_raises ===")
    test_construction_empty_env_maps_raises()
    print()
    print("=== test_construction_none_hdr_path_raises ===")
    test_construction_none_hdr_path_raises()
    print()
    print("=== test_list_assets_only_approved ===")
    test_list_assets_only_approved()
    print()
    print("=== test_from_asset_reads_rotation_and_path ===")
    test_from_asset_reads_rotation_and_path()
    print()
    print("=== test_from_asset_rejected_raises ===")
    test_from_asset_rejected_raises()
    print()
    print("=== test_reload_sets_world_env_map ===")
    test_reload_sets_world_env_map()
    print()
    print("=== test_seed_reproducibility_random_selection ===")
    test_seed_reproducibility_random_selection()
    print()
    print("=== test_different_seeds_differ_random_selection ===")
    test_different_seeds_differ_random_selection()
    print()
    print("=== test_selection_first_picks_first ===")
    test_selection_first_picks_first()
    print()
    print("=== test_selection_cycle_advances_in_order ===")
    test_selection_cycle_advances_in_order()
    print()
    print("=== test_via_world_env_reload ===")
    test_via_world_env_reload()
    print()
    print("=== test_via_world_env_priority_order ===")
    test_via_world_env_priority_order()
    print()
    print("All tests passed!")
