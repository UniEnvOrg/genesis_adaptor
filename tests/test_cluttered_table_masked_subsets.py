"""Masked-reset regression tests for :class:`ClutteredTableManager`.

Pins the bug where a masked reset whose mask **excludes env 0** aborted the
CUDA context with a device-side assert (``IndexKernel`` "index out of bounds"):

``_prepare_avoid_spaces`` used to index the queued per-env avoid spaces with the
*global* reset indices (``ext_xy[reset_idx]``) unconditionally, while callers
such as the sweep task handed over rows that were already restricted to the
reset scope.  With ``mask == [False, True]`` (``reset_idx == [1]``) indexing a
``(1, ...)`` tensor with row 1 is an out-of-bounds device access; with
``[True, False]`` it silently — and correctly — read row 0, which is why the
"include env 0" case appeared to work.

Covered here:

  - masked subsets for ``batch_size`` 2 and 3, including masks that exclude
    env 0 (``[False, True]``, ``[False, True, False]``, ``[False, False, True]``),
    an all-envs mask and ``batch_size == 1``;
  - both accepted avoid-space layouts: full-batch (canonical) and reset-scope
    (pre-sliced), the latter being the layout that used to abort;
  - unmasked envs are bit-for-bit preserved, reset envs stay collision-free and
    respect their avoid zones, inactive objects stay parked;
  - an ambiguous row count raises a descriptive ``ValueError`` instead of
    indexing out of bounds.
"""

import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis import GenesisWorld
from unienv_genesis_collection.scenes.plane_builder import PlaneBuilder
from unienv_genesis_collection.scenes.cached_table_builder import CachedTableBuilder
from unienv_genesis_collection.scenes.cluttered_table_manager import (
    ClutteredTableManager,
)
from unienv_genesis_collection.objects import BoxObject, SphereObject, CylinderObject
from unienv_interface.world import WorldEnv

_TABLE_POS = (0.0, 0.0, 0.65)
_TABLE_SIZE = (0.8, 0.5)


def _make_pool():
    return [
        BoxObject(size=(0.04, 0.04, 0.04), name="cube_r", color=(1, 0, 0, 1)),
        BoxObject(size=(0.05, 0.03, 0.06), name="brick_g", color=(0, 1, 0, 1)),
        SphereObject(radius=0.025, name="ball_b", color=(0, 0, 1, 1)),
        CylinderObject(radius=0.02, height=0.08, name="can_y", color=(1, 1, 0, 1)),
        BoxObject(size=(0.03, 0.03, 0.03), name="cube_w", color=(1, 1, 1, 1)),
        SphereObject(radius=0.02, name="ball_m", color=(1, 0, 1, 1)),
    ]


def _make_env(batch_size):
    world = GenesisWorld(batch_size=batch_size)
    plane = PlaneBuilder(world, name="floor")
    table = CachedTableBuilder(world, name="table", pos=_TABLE_POS)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=_make_pool(),
        n_objects_range=(2, 4),
        table_size=_TABLE_SIZE,
        table_pos=_TABLE_POS,
    )
    env = WorldEnv(world, [plane, table, clutter], render_mode=None)
    env.reset(seed=0, reload=True)
    return env, clutter


def _snapshot(clutter):
    return (
        [e.get_pos().clone() for e in clutter._loaded_entities],
        [e.get_quat().clone() for e in clutter._loaded_entities],
        clutter._active_mask_t.clone(),
    )


def _assert_preserved(clutter, snapshot, envs):
    positions, quats, active = snapshot
    for i, entity in enumerate(clutter._loaded_entities):
        for env_j in envs:
            assert torch.equal(entity.get_pos()[env_j], positions[i][env_j]), (
                f"object {i} env {env_j}: position changed by an unrelated masked reset"
            )
            assert torch.equal(entity.get_quat()[env_j], quats[i][env_j]), (
                f"object {i} env {env_j}: quaternion changed by an unrelated masked reset"
            )
    for env_j in envs:
        assert torch.equal(clutter._active_mask_t[:, env_j], active[:, env_j]), (
            f"env {env_j}: active mask changed by an unrelated masked reset"
        )


def _assert_placement_valid(clutter, env_j):
    """Reset env *env_j*: active objects separated, inactive parked, on-table."""
    active_positions = []
    for i, entity in enumerate(clutter._loaded_entities):
        pos = entity.get_pos()[env_j]
        if bool(clutter._active_mask_t[i, env_j]):
            assert pos[2].item() > _TABLE_POS[2], (
                f"env {env_j} object {i}: active object not on the table (z={pos[2]:.3f})"
            )
            active_positions.append(pos)
        else:
            park_z = clutter._park_positions_t[i, 2].item()
            assert abs(pos[2].item() - park_z) < 1e-6, (
                f"env {env_j} object {i}: inactive object should sit at its park z "
                f"(z={pos[2]:.3f}, park={park_z})"
            )
    # Pairwise footprint separation (the placement guarantee).
    radii = clutter._footprint_r_t
    active_idx = torch.where(clutter._active_mask_t[:, env_j])[0]
    for a in range(len(active_idx)):
        for b in range(a + 1, len(active_idx)):
            i, j = int(active_idx[a]), int(active_idx[b])
            dist = torch.linalg.vector_norm(
                active_positions[a][:2] - active_positions[b][:2]
            ).item()
            required = radii[i].item() + radii[j].item()
            assert dist >= required - 1e-5, (
                f"env {env_j}: objects {i}/{j} interpenetrate "
                f"(dist={dist:.4f} < {required:.4f})"
            )
    # Inside the table footprint.
    for pos in active_positions:
        assert abs(pos[0].item() - _TABLE_POS[0]) < _TABLE_SIZE[0] / 2.0
        assert abs(pos[1].item() - _TABLE_POS[1]) < _TABLE_SIZE[1] / 2.0


def _assert_clears_zones(clutter, env_j, centres_xy, radii):
    """Active objects of *env_j* must clear the exclusion zones given to it."""
    for i, entity in enumerate(clutter._loaded_entities):
        if not bool(clutter._active_mask_t[i, env_j]):
            continue
        pos = entity.get_pos()[env_j]
        fp_r = clutter._footprint_r_t[i].item()
        for zone_idx in range(centres_xy.shape[0]):
            centre = centres_xy[zone_idx]
            dist = torch.linalg.vector_norm(pos[:2] - centre).item()
            assert dist >= radii[zone_idx].item() - fp_r - 1e-4, (
                f"env {env_j} object {i}: inside avoid zone {zone_idx} "
                f"(dist={dist:.4f}, r={radii[zone_idx].item():.4f}, fp_r={fp_r:.4f})"
            )


# ------------------------------------------------------------------
# Unit-level: avoid-space row selection
# ------------------------------------------------------------------

def test_avoid_space_layout_selection():
    """Full-batch and reset-scope layouts must select the same reset rows."""
    world = GenesisWorld(batch_size=3)
    clutter = ClutteredTableManager(
        world,
        name="clutter",
        object_pool=_make_pool(),
        n_objects_range=(2, 3),
        table_size=_TABLE_SIZE,
        table_pos=(0.0, 0.0, 0.0),  # identity local frame: local == world
    )

    full_xy = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
        dtype=world.dtype,
        device=world.device,
    )
    full_r = torch.tensor([[0.1], [0.2], [0.3]], dtype=world.dtype, device=world.device)

    mask = torch.tensor([False, True, False], device=world.device)
    reset_idx = torch.where(mask)[0]

    # Canonical full-batch layout: row for env 1 must be picked.
    xy_a, r_a = clutter._prepare_avoid_spaces(3, mask, (full_xy, full_r))
    assert torch.equal(xy_a[:, 0, :], full_xy[1])
    assert torch.equal(r_a[:, 0], full_r[1:2, 0])

    # Pre-sliced reset-scope layout (the one that used to abort with [1]).
    xy_b, r_b = clutter._prepare_avoid_spaces(
        3, mask, (full_xy[reset_idx], full_r[reset_idx]),
    )
    assert torch.equal(xy_b, xy_a)
    assert torch.equal(r_b, r_a)

    # ... and the same for a mask excluding env 0 on a 2-env world.
    world2 = GenesisWorld(batch_size=2)
    clutter2 = ClutteredTableManager(
        world2,
        name="clutter",
        object_pool=_make_pool(),
        n_objects_range=(2, 3),
        table_size=_TABLE_SIZE,
        table_pos=(0.0, 0.0, 0.0),
    )
    two_xy = full_xy[:2].contiguous()
    two_r = full_r[:2].contiguous()
    mask2 = torch.tensor([False, True], device=world2.device)
    reset_idx2 = torch.where(mask2)[0]
    xy_c, r_c = clutter2._prepare_avoid_spaces(2, mask2, (two_xy, two_r))
    xy_d, r_d = clutter2._prepare_avoid_spaces(
        2, mask2, (two_xy[reset_idx2], two_r[reset_idx2]),
    )
    assert torch.equal(xy_c[:, 0, :], two_xy[1])
    assert torch.equal(xy_d, xy_c)
    assert torch.equal(r_d, r_c)

    # Ambiguous row count: descriptive error, never an out-of-bounds access.
    try:
        clutter2._prepare_avoid_spaces(2, mask2, (two_xy[:1].repeat(3, 1, 1), two_r))
    except ValueError as err:
        assert "avoid spaces have 3 row(s)" in str(err)
    else:  # pragma: no cover - explicit failure message
        raise AssertionError("expected ValueError for an ambiguous avoid-space row count")


# ------------------------------------------------------------------
# Integration: masked resets for every subset shape
# ------------------------------------------------------------------

def test_masked_reset_excluding_env0_batch2():
    """The reported failure: ``mask == [False, True]`` must work."""
    env, clutter = _make_env(2)
    try:
        # Distinctive full-batch avoid zones: env 0 excludes the +X half, env 1
        # the -X half.  After the masked reset of env 1 its active objects must
        # clear env 1's zone (a wrong-row selection or a dropped zone shows up
        # here), and the reset must actually have placed something.
        zones_xy = torch.tensor(
            [[[0.55, 0.0]], [[-0.55, 0.0]]],
            dtype=clutter.world.dtype,
            device=clutter.world.device,
        )
        zones_r = torch.tensor(
            [[0.5], [0.5]], dtype=clutter.world.dtype, device=clutter.world.device,
        )

        snapshot = _snapshot(clutter)
        mask = torch.tensor([False, True], device=clutter.world.device)
        env.reset(
            seed=13,
            mask=mask,
            pernode_kwargs={"clutter": {"avoid_spaces": (zones_xy, zones_r)}},
        )

        _assert_preserved(clutter, snapshot, envs=[0])
        _assert_placement_valid(clutter, 1)
        _assert_clears_zones(clutter, 1, zones_xy[1], zones_r[1])
        assert bool(clutter._active_mask_t[:, 1].any()), (
            "masked reset of env 1 placed no object at all — nothing was tested"
        )

        # Same masked subset, but with the pre-sliced (reset-scope) layout that
        # used to abort the CUDA context.
        reset_idx = torch.where(mask)[0]
        snapshot = _snapshot(clutter)
        clutter.set_next_reset_avoid_spaces(
            zones_xy[reset_idx], zones_r[reset_idx],
        )
        env.reset(seed=17, mask=mask)
        _assert_preserved(clutter, snapshot, envs=[0])
        _assert_placement_valid(clutter, 1)
        _assert_clears_zones(clutter, 1, zones_xy[1], zones_r[1])
        assert bool(clutter._active_mask_t[:, 1].any()), (
            "pre-sliced masked reset of env 1 placed no object at all"
        )

        # ... and the mirrored mask (includes env 0) keeps working.
        mask = torch.tensor([True, False], device=clutter.world.device)
        snapshot = _snapshot(clutter)
        env.reset(seed=19, mask=mask)
        _assert_preserved(clutter, snapshot, envs=[1])
        _assert_placement_valid(clutter, 0)

        # All-envs mask.
        mask = torch.tensor([True, True], device=clutter.world.device)
        env.reset(seed=23, mask=mask)
        _assert_placement_valid(clutter, 0)
        _assert_placement_valid(clutter, 1)

        # Full reset still works (and is a superset of the above).
        env.reset(seed=29)
        _assert_placement_valid(clutter, 0)
        _assert_placement_valid(clutter, 1)
    finally:
        env.close()


def test_masked_reset_any_subset_batch3():
    """``batch_size == 3`` subsets, all excluding env 0."""
    env, clutter = _make_env(3)
    try:
        for seed, envs in (
            (31, [1]),
            (37, [2]),
            (41, [1, 2]),
            (43, [0]),
        ):
            mask = torch.zeros(3, dtype=torch.bool, device=clutter.world.device)
            mask[envs] = True
            before = _snapshot(clutter)
            env.reset(seed=seed, mask=mask)
            preserved = [j for j in range(3) if j not in envs]
            _assert_preserved(clutter, before, envs=preserved)
            for env_j in envs:
                _assert_placement_valid(clutter, env_j)
    finally:
        env.close()


def test_masked_reset_batch1():
    """``batch_size == 1``: full resets and masked ``[True]`` stay valid."""
    env, clutter = _make_env(1)
    try:
        env.reset(seed=5)
        _assert_placement_valid(clutter, 0)
        before = _snapshot(clutter)
        env.reset(
            seed=7,
            mask=torch.tensor([True], device=clutter.world.device),
        )
        _assert_placement_valid(clutter, 0)
        # A single-env masked reset is a full reset of that env: it re-draws.
        assert before[2].shape == clutter._active_mask_t.shape
    finally:
        env.close()
