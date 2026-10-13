"""Regression tests for the contype/conaffinity synthesis shim.

Genesis 1.4.2's z3 ascending-bit search spins for minutes on dense USD
FilteredPairsAPI graphs (>=315 colliders); the constructive replacement must
realise the identical collision matrix in milliseconds.
"""
import random
import time

import pytest

from unienv_genesis_collection import genesis_compat


def _realised_ok(n, invalid_pairs, masks):
    for i in range(n):
        if (masks[i][0] | masks[i][1]) == 0:
            return False  # zero mask demotes the geom to visual-only
        for j in range(i + 1, n):
            collide = bool((masks[i][0] & masks[j][1]) | (masks[j][0] & masks[i][1]))
            if collide == (frozenset((i, j)) in invalid_pairs):
                return False
    return True


def test_constructive_matches_desired_matrix_random():
    rng = random.Random(0)
    for _ in range(50):
        n = rng.randint(2, 20)
        pairs = {
            frozenset((i, j))
            for i in range(n)
            for j in range(i + 1, n)
            if rng.random() < 0.35
        }
        masks = genesis_compat.solve_contype_conaffinity(n, pairs)
        assert masks is not None
        assert _realised_ok(n, pairs, masks)


def test_constructive_handles_dense_fridge_scale_fast():
    # n=340 colliders in 9 forbidden-neighbour groups with ~46k invalid pairs —
    # the real MolmoSpaces fridge profile.  The z3 path needs >200 s here; the
    # constructive path must be milliseconds.  (A *uniformly* random 340-geom
    # graph instead yields ~340 distinct groups and exercises the z3 fallback,
    # which is why this fixture mirrors the clustered real topology.)
    rng = random.Random(1)
    n, n_groups = 340, 9
    group_of = [rng.randrange(n_groups) for _ in range(n)]
    forbidden_group_pairs = {
        frozenset((g, h))
        for g in range(n_groups)
        for h in range(g + 1, n_groups)
        if rng.random() < 0.5
    }
    pairs = {
        frozenset((i, j))
        for i in range(n)
        for j in range(i + 1, n)
        if frozenset((group_of[i], group_of[j])) in forbidden_group_pairs
    }
    start = time.time()
    masks = genesis_compat.solve_contype_conaffinity(n, pairs)
    assert masks is not None
    assert _realised_ok(n, pairs, masks)
    assert time.time() - start < 5.0


def test_install_is_idempotent_and_patches_importers():
    genesis_compat.install()
    genesis_compat.install()  # second call must not raise or double-wrap
    import genesis.utils.collision as col
    import genesis.utils.usd.usd_collision as usd_col

    assert col.solve_contype_conaffinity is genesis_compat.solve_contype_conaffinity
    assert usd_col.solve_contype_conaffinity is genesis_compat.solve_contype_conaffinity


def test_memoization_reuses_result():
    n, pairs = 8, {frozenset((0, 1)), frozenset((2, 3))}
    first = genesis_compat.solve_contype_conaffinity(n, pairs)
    second = genesis_compat.solve_contype_conaffinity(n, pairs)
    assert first is second  # lru_cache returns the same object


def test_decimate_patch_clamps_target_to_post_validate_face_count():
    """Mesh.decimate must clamp ``target_count`` after ``process(validate=True)``.

    Genesis 1.4.2 guards ``len(faces) > decimate_face_num`` *before* the
    validation pass, then calls ``fast_simplification.simplify`` with the
    unclamped target on the post-validate mesh — whose duplicate/degenerate
    faces have been merged away.  With ``agg != 0`` (lossless=False)
    fast_simplification raises ``ValueError: ``target_count`` must be less
    than the number of faces N``.  The installed shim clamps the target to
    the actual face count (equality is a fast_simplification no-op).
    """
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu)

    import genesis.engine.mesh as mesh_module
    import numpy as np
    import trimesh

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    # Triplicated faces: 36 pre-validate faces collapse to 12 in
    # ``process(validate=True)``, so a target of 16 passes the pre-validate
    # guard but exceeds the post-validate face count (the crash).
    degenerate = trimesh.Trimesh(
        vertices=box.vertices.copy(), faces=np.vstack([box.faces] * 3), process=False
    )
    assert len(degenerate.faces) == 36

    genesis_compat.install()
    mesh = mesh_module.Mesh.from_trimesh(
        degenerate.copy(), surface=gs.surfaces.Default()
    )
    mesh.decimate(16, 0.5)  # agg != 0 → lossless=False (the raising path)
    assert len(mesh._mesh.faces) <= 16
    assert mesh._metadata.get("decimated") is True

    # The install is idempotent — a second call must not re-wrap the patch.
    patched = mesh_module.Mesh.decimate
    genesis_compat.install()
    assert mesh_module.Mesh.decimate is patched


def test_decimate_patch_matches_unpatched_on_benign_geometry():
    """The patch must not change the outcome when the target is reachable."""
    import genesis as gs

    if not getattr(gs, "_initialized", False):
        gs.init(backend=gs.cpu)

    import genesis.engine.mesh as mesh_module
    import trimesh

    genesis_compat.install()
    from unienv_genesis_collection import genesis_compat as compat

    def _unpatched_decimate(self, decimate_face_num, decimate_aggressiveness):
        """Genesis 1.4.2's own body (the pre-patch behaviour, verbatim)."""
        import fast_simplification

        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_face_num,
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True
        self._invalidate_geometry_cache()
        self.clear_visuals()

    # A dense, non-degenerate mesh whose target is reachable: both paths must
    # produce the identical face count.
    dense = trimesh.creation.icosphere(subdivisions=4)
    reference = mesh_module.Mesh.from_trimesh(dense.copy(), surface=gs.surfaces.Default())
    _unpatched_decimate(reference, 600, 0.5)
    patched = mesh_module.Mesh.from_trimesh(dense.copy(), surface=gs.surfaces.Default())
    patched.decimate(600, 0.5)
    assert len(patched._mesh.faces) == len(reference._mesh.faces)
    assert patched._metadata.get("decimated") is True
    # The clamp only ever lowers the target: an honest decimation is untouched.
    assert compat.decimate_target_after_validate(4, 4) == 4
    assert compat.decimate_target_after_validate(12, 16) == 12
