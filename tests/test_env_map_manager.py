"""Tests for EnvMapManager.

Covers:
  - Construction with hdr_paths and env_maps.
  - Auto-discovery of .hdr files in the asset folder.
  - reload() sets world.env_map to a pool member.
  - Seed reproducibility.
  - shuffle=False picks index 0.
  - Different seeds produce different selections (with enough pool size).
  - Via WorldEnv.reset(reload=True) the manager runs and mutates world.env_map.
  - hdr_path=None raises ValueError.
"""

import os
import tempfile
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis.options.light_option import GenesisWorldEnvMapOption
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes import env_map_manager as emm_module
from unienv_genesis_collection.scenes.env_map_manager import EnvMapManager
from unienv_interface.world import WorldEnv


def _make_dummy_hdrs(tmpdir, names):
    """Create empty dummy .hdr files and return their paths."""
    paths = []
    for n in names:
        p = os.path.join(tmpdir, f"{n}.hdr")
        with open(p, "wb") as f:
            f.write(b"")
        paths.append(p)
    return paths


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_hdr_paths():
    """hdr_paths are wrapped into GenesisWorldEnvMapOption."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths)
        assert mgr.pool_size == 3
        for i, p in enumerate(paths):
            assert mgr._pool[i].hdr_path == p
        print("  PASS: construction with hdr_paths")
        world.scene.destroy()


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


def test_auto_discovery():
    """When neither env_maps nor hdr_paths is given, auto-discover .hdr files."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["alpha", "beta", "gamma"])
        # Patch the module-level asset dir to point at our temp folder.
        original_dir = emm_module._ENV_MAP_ASSET_DIR
        try:
            emm_module._ENV_MAP_ASSET_DIR = __import__("pathlib").Path(tmp)
            world = GenesisWorld()
            mgr = EnvMapManager(world)  # no paths — should auto-discover
            assert mgr.pool_size == 3
            for i, p in enumerate(paths):
                assert mgr._pool[i].hdr_path == p
            print(f"  PASS: auto-discovery found {mgr.pool_size} HDRIs")
            world.scene.destroy()
        finally:
            emm_module._ENV_MAP_ASSET_DIR = original_dir


def test_auto_discovery_empty_dir_raises():
    """Auto-discovery with no .hdr files raises ValueError with download hint."""
    with tempfile.TemporaryDirectory() as tmp:
        original_dir = emm_module._ENV_MAP_ASSET_DIR
        try:
            emm_module._ENV_MAP_ASSET_DIR = __import__("pathlib").Path(tmp)
            world = GenesisWorld()
            try:
                EnvMapManager(world)
                assert False, "Should have raised ValueError"
            except ValueError as e:
                assert "no .hdr files" in str(e).lower()
                assert "prepare_env_maps" in str(e)
                print(f"  PASS: empty asset dir raises — {e}")
            world.scene.destroy()
        finally:
            emm_module._ENV_MAP_ASSET_DIR = original_dir


def test_auto_discovery_missing_dir_raises():
    """Auto-discovery with a non-existent folder raises ValueError."""
    original_dir = emm_module._ENV_MAP_ASSET_DIR
    try:
        emm_module._ENV_MAP_ASSET_DIR = __import__("pathlib").Path(
            "/nonexistent/path/that/does/not/exist"
        )
        world = GenesisWorld()
        try:
            EnvMapManager(world)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "not found" in str(e).lower()
            assert "prepare_env_maps" in str(e)
            print(f"  PASS: missing asset dir raises — {e}")
        world.scene.destroy()
    finally:
        emm_module._ENV_MAP_ASSET_DIR = original_dir


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


# ------------------------------------------------------------------
# Reload randomization
# ------------------------------------------------------------------

def test_reload_sets_world_env_map():
    """reload() sets world.env_map to a pool member."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths, seed=0)
        mgr.reload(priority=150, seed=0)
        assert mgr.current_index is not None
        assert world.env_map is not None
        assert world.env_map.hdr_path in paths
        assert world.env_map is mgr._pool[mgr.current_index]
        print(f"  PASS: reload sets world.env_map (idx={mgr.current_index})")
        world.scene.destroy()


def test_seed_reproducibility():
    """Same seed -> same selection."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c", "d", "e"])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths)

        mgr.reload(priority=150, seed=123)
        idx_a = mgr.current_index

        mgr.reload(priority=150, seed=123)
        idx_b = mgr.current_index

        assert idx_a == idx_b, f"Same seed gave {idx_a} then {idx_b}"
        print(f"  PASS: seed reproducibility (idx={idx_a})")
        world.scene.destroy()


def test_different_seeds_differ():
    """Different seeds produce (likely) different selections."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, [f"hdri_{i}" for i in range(20)])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths)

        indices = set()
        for s in range(10):
            mgr.reload(priority=150, seed=s)
            indices.add(mgr.current_index)
        assert len(indices) > 1, f"Only {len(indices)} distinct indices across 10 seeds"
        print(f"  PASS: different seeds -> {len(indices)} distinct selections")
        world.scene.destroy()


def test_no_shuffle_picks_first():
    """shuffle=False always picks index 0."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a", "b", "c"])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths, shuffle=False)
        for s in range(5):
            mgr.reload(priority=150, seed=s)
            assert mgr.current_index == 0, f"shuffle=False but got idx={mgr.current_index}"
        print("  PASS: shuffle=False picks index 0")
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
        mgr = EnvMapManager(world, hdr_paths=paths, seed=0)
        env = WorldEnv(world, [plane, mgr], render_mode=None)
        env.reset(seed=0, reload=True)

        assert mgr.current_index is not None
        assert world.env_map is not None
        assert world.env_map.hdr_path in paths

        # Second reload should (likely) pick a different one.
        first = mgr.current_index
        env.reset(seed=99, reload=True)
        assert mgr.current_index is not None
        assert world.env_map.hdr_path in paths

        info = mgr.get_info()
        assert info["env_map_index"] == mgr.current_index
        assert info["env_map_pool_size"] == 4

        print(f"  PASS: WorldEnv reload (first idx={first}, second idx={mgr.current_index})")
        env.close()


def test_via_world_env_priority_order():
    """Manager runs before camera priority (0). Verify priority is > 0."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = _make_dummy_hdrs(tmp, ["a"])
        world = GenesisWorld()
        mgr = EnvMapManager(world, hdr_paths=paths)
        assert next(iter(mgr.reload_priorities)) > 0, (
            "EnvMapManager reload priority must be > 0 to run before cameras"
        )
        print(f"  PASS: reload priority = {next(iter(mgr.reload_priorities))} > 0")
        world.scene.destroy()


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    print("=== test_construction_hdr_paths ===")
    test_construction_hdr_paths()
    print()
    print("=== test_construction_env_maps ===")
    test_construction_env_maps()
    print()
    print("=== test_auto_discovery ===")
    test_auto_discovery()
    print()
    print("=== test_auto_discovery_empty_dir_raises ===")
    test_auto_discovery_empty_dir_raises()
    print()
    print("=== test_auto_discovery_missing_dir_raises ===")
    test_auto_discovery_missing_dir_raises()
    print()
    print("=== test_construction_none_hdr_path_raises ===")
    test_construction_none_hdr_path_raises()
    print()
    print("=== test_reload_sets_world_env_map ===")
    test_reload_sets_world_env_map()
    print()
    print("=== test_seed_reproducibility ===")
    test_seed_reproducibility()
    print()
    print("=== test_different_seeds_differ ===")
    test_different_seeds_differ()
    print()
    print("=== test_no_shuffle_picks_first ===")
    test_no_shuffle_picks_first()
    print()
    print("=== test_via_world_env_reload ===")
    test_via_world_env_reload()
    print()
    print("=== test_via_world_env_priority_order ===")
    test_via_world_env_priority_order()
    print()
    print("All tests passed!")
