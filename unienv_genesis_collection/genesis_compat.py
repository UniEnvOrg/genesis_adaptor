"""Performance shim for Genesis's collision-bitmask synthesis.

Genesis 1.4.2/1.4.3's :func:`genesis.utils.collision.solve_contype_conaffinity`
translates a desired collision matrix into MuJoCo-style ``contype`` /
``conaffinity`` bitmasks with an **ascending bit-width z3 search**: for every
bit width ``1..31`` it rebuilds the full O(n²·bits) Boolean formula and calls
``solver.check()`` with **no timeout**.  For dense USD assets (many collider
geoms + ``FilteredPairsAPI``) the per-attempt cost roughly doubles per bit and
the check goes ``unknown``, so a single parse spins one thread at 100 % CPU
for minutes — and the packaging sanity gate parses the same asset three times
(closed / opened / render scenes) with no memoization, so objects with ≳315
colliders effectively hang.

This module patches that function with a **constructive, z3-free assignment**:

* group the geoms by their forbidden-neighbour set ``F(i)`` (geoms that may
  not collide with exactly the same partners); one bit per group;
* a group pair ``(g, h)`` is *fully forbidden* iff **no** geom pair across the
  groups is allowed; ``conaffinity`` then enables every group not fully
  forbidden against the geom's own group.

That realizes the exact collision matrix in O(n² + G²) time (≈11–20 ms for the
≈340-collider MolmoSpaces fridges, versus 100–220 s for z3) with a bit width
identical to what z3 finds.  When the grouping needs more than ``max_bits``
groups it falls back to the original z3 search (with a solver timeout so the
fallback is bounded too).  Results are memoized per ``(n, invalid_pairs)`` so
the gate's three parses of one asset pay the cost once.

The patch is installed by :func:`install`, which is idempotent.
"""

from __future__ import annotations

import functools
import os
from typing import List, Optional, Set, Tuple

_INSTALLED = False


def _solve_constructive(
    n: int, invalid_pairs: Set[frozenset], max_bits: int
) -> Optional[List[Tuple[int, int]]]:
    """One-bit-per-equivalence-group assignment, or ``None`` if it needs more
    than *max_bits* groups (caller falls back to the z3 search)."""
    forbidden = [set() for _ in range(n)]
    for pair in invalid_pairs:
        i, j = tuple(pair)
        forbidden[i].add(j)
        forbidden[j].add(i)

    group_id_of_signature = {}
    group_of = [0] * n
    for i in range(n):
        signature = frozenset(forbidden[i])
        if signature not in group_id_of_signature:
            group_id_of_signature[signature] = len(group_id_of_signature)
        group_of[i] = group_id_of_signature[signature]
    n_groups = len(group_id_of_signature)
    if n_groups > max_bits:
        return None

    # A group pair is *fully forbidden* iff no allowed geom pair spans it.
    allowed = [[False] * n_groups for _ in range(n_groups)]
    for i in range(n):
        gi = group_of[i]
        for j in range(i + 1, n):
            if frozenset((i, j)) not in invalid_pairs:
                allowed[gi][group_of[j]] = allowed[group_of[j]][gi] = True
    masks: List[Tuple[int, int]] = []
    for i in range(n):
        gi = group_of[i]
        contype = 1 << gi
        conaffinity = 0
        for h in range(n_groups):
            if h == gi or allowed[gi][h]:
                conaffinity |= 1 << h
        masks.append((contype, conaffinity))
    return masks


def _key(n: int, invalid_pairs: Set[frozenset], max_bits: int):
    return (n, max_bits, frozenset(invalid_pairs))


@functools.lru_cache(maxsize=None)
def _solve_memoized(key) -> Optional[List[Tuple[int, int]]]:
    n, max_bits, invalid_pairs = key
    masks = _solve_constructive(n, invalid_pairs, max_bits)
    if masks is not None:
        return masks
    # Fallback: the original ascending-bit z3 search, but with a solver
    # timeout so it stays bounded.
    return _z3_search_with_timeout(n, invalid_pairs, max_bits)


def _z3_search_with_timeout(
    n: int, invalid_pairs: Set[frozenset], max_bits: int = 31
) -> Optional[List[Tuple[int, int]]]:
    """The original ascending-bit z3 search with a per-attempt solver timeout.

    Only reached when the constructive grouping needs more than ``max_bits``
    bits — rare for real assets.  ``Z3_TIMEOUT_MS`` bounds each ``check()``.
    """
    import z3

    timeout_ms = int(os.environ.get("Z3_TIMEOUT_MS", "30000"))
    for num_bits in range(1, max_bits + 1):
        solver = z3.Solver()
        solver.set("timeout", timeout_ms)
        contype_bits = [[z3.Bool(f"contype_{i}_{b}") for b in range(num_bits)] for i in range(n)]
        conaffinity_bits = [[z3.Bool(f"conaffinity_{i}_{b}") for b in range(num_bits)] for i in range(n)]
        for i in range(n):
            solver.add(z3.Or([*contype_bits[i], *conaffinity_bits[i]]))
            for j in range(i + 1, n):
                cond1 = z3.Or([z3.And(contype_bits[i][b], conaffinity_bits[j][b]) for b in range(num_bits)])
                cond2 = z3.Or([z3.And(contype_bits[j][b], conaffinity_bits[i][b]) for b in range(num_bits)])
                if frozenset((i, j)) in invalid_pairs:
                    solver.add(z3.Not(cond1), z3.Not(cond2))
                else:
                    solver.add(z3.Or(cond1, cond2))
        if solver.check() == z3.sat:
            model = solver.model()
            return [
                tuple(
                    sum((1 << b) if z3.is_true(model[e]) else 0 for b, e in enumerate(bits))
                    for bits in (contype_bits[i], conaffinity_bits[i])
                )
                for i in range(n)
            ]
    return None


def solve_contype_conaffinity(
    n: int, invalid_pairs: Set[frozenset], max_bits: int = 31
) -> Optional[List[Tuple[int, int]]]:
    """Drop-in replacement (constructive fast path + bounded z3 fallback)."""
    return _solve_memoized(_key(n, invalid_pairs, max_bits))


def decimate_target_after_validate(face_count: int, decimate_face_num: int) -> int:
    """``decimate_face_num`` clamped to a post-validate face count.

    ``fast_simplification.simplify`` allows ``target_count == len(faces)``
    (a no-op) but rejects a larger target, which is exactly what the
    pre-validate Genesis guard can hand it.
    """
    return min(int(decimate_face_num), int(face_count))


def _patch_decimate(module) -> None:
    """Patch ``genesis.engine.mesh.Mesh.decimate``'s post-validate target.

    Genesis 1.4.2/1.4.3 guard ``len(faces) > decimate_face_num`` *before*
    ``self._mesh.process(validate=True)`` and then hands
    ``target_count=decimate_face_num`` to ``fast_simplification.simplify``
    on the *post-validate* mesh.  ``process(validate=True)`` merges
    duplicate/degenerate geometry, so the mesh can now hold **fewer** faces
    than the target — and fast_simplification raises
    ``ValueError: ``target_count`` must be less than the number of faces N``.
    After the validation pass the target is clamped to the actual face
    count (fast_simplification allows ``target_count == len(faces)``, where
    it is a no-op).  Everything else (aggressiveness, lossless mode,
    metadata flag, cache invalidation) is preserved verbatim.
    """
    import inspect

    source = inspect.getsource(module.Mesh.decimate)
    if "_decimate_target" in source or "decimate_target_after_validate" in source:
        return  # already patched (a reloaded module came pre-patched)
    import trimesh  # noqa: F401 — bound by the replacement body
    import fast_simplification  # noqa: F401 — bound by the replacement

    def decimate(self, decimate_face_num, decimate_aggressiveness):
        """
        Decimate the mesh.
        """
        if self._mesh.vertices.shape[0] > 3 and len(self._mesh.faces) > decimate_face_num:
            self._mesh.process(validate=True)
            self._mesh = trimesh.Trimesh(
                *fast_simplification.simplify(
                    self._mesh.vertices,
                    self._mesh.faces,
                    target_count=decimate_target_after_validate(
                        len(self._mesh.faces), decimate_face_num
                    ),
                    agg=decimate_aggressiveness,
                    lossless=(decimate_aggressiveness == 0),
                ),
            )
            self._metadata["decimated"] = True

        # Genesis <= 1.4.4 drops the mesh's derived geometry here; 1.4.5 has no such method and does it in the
        # ``mutates_tmesh`` decorator instead (applied below).
        invalidate = getattr(self, "_invalidate_geometry_cache", None)
        if invalidate is not None:
            invalidate()
        self.clear_visuals()

    if hasattr(module, "mutates_tmesh"):
        decimate = module.mutates_tmesh(decimate)
    module.Mesh.decimate = decimate


def install() -> None:
    """Patch Genesis's collision-bitmask synthesis + mesh decimation.

    Two independent fixes (both idempotent, each a no-op when its target is
    absent; Genesis 1.4.6 dropped the bitmask synthesis, so only the second
    applies there):

    * ``genesis.utils.collision.solve_contype_conaffinity`` — replaced with
      the constructive (z3-free) assignment.  The USD and MJCF importers
      bind the symbol by value at module load, so the already-bound
      references are re-pointed too.
    * ``genesis.engine.mesh.Mesh.decimate`` — clamps the
      ``fast_simplification`` ``target_count`` to the face count *after*
      ``process(validate=True)`` (the post-validate mesh can hold fewer
      faces than the pre-validate guard allowed for).
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_collision()
    try:
        import trimesh  # noqa: F401
        import fast_simplification  # noqa: F401
        from genesis.engine import mesh as _mesh_module
    except Exception:
        pass
    else:
        try:
            _patch_decimate(_mesh_module)
        except Exception:  # noqa: BLE001 — never make loading worse
            pass
    _INSTALLED = True


def _patch_collision() -> None:
    """Point ``solve_contype_conaffinity`` at the constructive solver (Genesis <= 1.4.5; 1.4.6 has no such module)."""
    try:
        import z3  # noqa: F401 — required by the fallback path
        from genesis.utils import collision as _collision
    except Exception:
        return
    _collision.solve_contype_conaffinity = solve_contype_conaffinity
    # The importers imported the symbol by value at module load; patch the
    # already-bound references too so the shim actually takes effect.
    for module_name in (
        "genesis.utils.usd.usd_collision",
        "genesis.utils.mjcf",
    ):
        try:
            module = __import__(module_name, fromlist=["*"])
        except Exception:
            continue
        if getattr(module, "solve_contype_conaffinity", None) is not None:
            module.solve_contype_conaffinity = solve_contype_conaffinity
