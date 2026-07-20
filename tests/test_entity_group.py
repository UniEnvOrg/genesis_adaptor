"""Device/dtype consistency tests for ``EntityGroup`` (Phase 2 remediation).

Covers the Oracle gate finding that an ``EntityGroup`` could become
internally mixed-device when separate pose setters received tensors from
different devices (e.g. a CPU-initialised group followed by
``set_pos(cuda_tensor)`` or ``set_quat(cuda_tensor)``).

Invariants under test, for every setter path (``set_pos``, ``set_quat``,
``set_pose``) and for member-pose composition (``_world_pose`` /
``_apply_all`` / ``world_pose``):

  - The anchor (``group.pos`` + ``group.quat``) always shares a single
    canonical device and dtype.
  - A floating tensor input makes its device/dtype canonical and migrates
    the other anchor component alongside; non-tensor inputs adopt the
    current canonical device/dtype.
  - Member entities receive world poses on the canonical device/dtype.

These tests are Genesis-free: member entities are fakes that record the
tensors passed to ``set_pos`` / ``set_quat``.
"""

from __future__ import annotations

import pytest
import torch
import genesis as gs
from genesis.constants import backend as gs_backend

if not gs._initialized:
    gs.init(backend=gs_backend.cpu)

from unienv_genesis.utils.entity_group import EntityGroup

_CUDA_AVAILABLE = torch.cuda.is_available()
_requires_cuda = pytest.mark.skipif(not _CUDA_AVAILABLE, reason="CUDA not available")


class _FakeMember:
    """Records tensors forwarded by EntityGroup._apply_all."""

    def __init__(self):
        self.calls = []  # list of ("pos"|"quat", tensor, envs_idx)

    def set_pos(self, pos, envs_idx=None):
        self.calls.append(("pos", pos, envs_idx))

    def set_quat(self, quat, envs_idx=None):
        self.calls.append(("quat", quat, envs_idx))

    @property
    def last_pos(self):
        return [c for c in self.calls if c[0] == "pos"][-1][1]

    @property
    def last_quat(self):
        return [c for c in self.calls if c[0] == "quat"][-1][1]


def _make_group(**kwargs):
    group = EntityGroup(pos=(0.0, 0.0, 0.8), quat=(1.0, 0.0, 0.0, 0.0), **kwargs)
    member = _FakeMember()
    group.add_member(member, local_pos=(0.1, 0.0, -0.01))
    return group, member


def _assert_canonical(group, device, dtype):
    assert group.pos.device == torch.device(device) or group.pos.device.type == device
    assert group.quat.device == torch.device(device) or group.quat.device.type == device
    assert group.pos.dtype == dtype
    assert group.quat.dtype == dtype


def _assert_member_canonical(member, device, dtype):
    assert member.last_pos.device.type == device
    assert member.last_quat.device.type == device
    assert member.last_pos.dtype == dtype
    assert member.last_quat.dtype == dtype


# ---------------------------------------------------------------------------
# CPU canonical-device/dtype semantics (always run)
# ---------------------------------------------------------------------------

def test_set_pos_tensor_makes_dtype_canonical():
    """set_pos with a float64 tensor migrates the whole anchor to float64."""
    group, member = _make_group()
    group.set_pos(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    _assert_canonical(group, "cpu", torch.float64)
    _assert_member_canonical(member, "cpu", torch.float64)
    assert torch.allclose(group.pos, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))


def test_set_quat_tensor_makes_dtype_canonical():
    """set_quat with a float64 tensor migrates the position component too."""
    group, member = _make_group()
    group.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64))
    _assert_canonical(group, "cpu", torch.float64)
    _assert_member_canonical(member, "cpu", torch.float64)


def test_set_pose_tensors_make_dtype_canonical():
    """set_pose with float64 tensors keeps both components float64."""
    group, member = _make_group()
    group.set_pose(
        torch.tensor([0.5, 0.0, 0.8], dtype=torch.float64),
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
    )
    _assert_canonical(group, "cpu", torch.float64)
    _assert_member_canonical(member, "cpu", torch.float64)


def test_sequence_input_adopts_current_canonical():
    """Plain-sequence setters never break the established canonical dtype."""
    group, member = _make_group()
    group.set_pos(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    group.set_quat((1.0, 0.0, 0.0, 0.0))  # sequence: must stay float64
    _assert_canonical(group, "cpu", torch.float64)
    _assert_member_canonical(member, "cpu", torch.float64)


def test_explicit_dtype_is_preserved():
    """An explicit dtype=... keeps tensor inputs cast to the group dtype."""
    group, member = _make_group(dtype=torch.float32)
    group.set_pos(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    _assert_canonical(group, "cpu", torch.float32)
    _assert_member_canonical(member, "cpu", torch.float32)


def test_composition_world_pose_follows_anchor():
    """world_pose composes on the anchor's canonical device/dtype."""
    group, _ = _make_group()
    group.set_pos(torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64))
    w_pos, w_quat = group.world_pose(local_pos=(0.1, 0.0, -0.01))
    assert w_pos.dtype == torch.float64
    assert w_quat.dtype == torch.float64
    assert torch.allclose(
        w_pos, torch.tensor([1.1, 0.0, -0.01], dtype=torch.float64),
    )


def test_batched_anchor_masked_apply_cpu():
    """Batched anchor + envs_idx still applies only the selected rows."""
    group, member = _make_group()
    n_envs = 4
    group.set_pose(
        torch.zeros(n_envs, 3),
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(n_envs, 4),
    )
    assert group.pos.shape == (n_envs, 3)
    assert group.quat.shape == (n_envs, 4)
    member.calls.clear()
    group.set_pos(torch.ones(n_envs, 3), envs_idx=torch.tensor([1, 3]))
    assert member.last_pos.shape == (2, 3)
    assert torch.allclose(member.last_pos[:, 0], torch.ones(2) + 0.1)
    # Anchor retains the full batch.
    assert group.pos.shape == (n_envs, 3)


# ---------------------------------------------------------------------------
# CUDA canonical-device semantics (each setter path)
# ---------------------------------------------------------------------------

@_requires_cuda
def test_cuda_set_pos_on_cpu_initialized_group():
    """CPU-initialised group + set_pos(cuda) must not become mixed-device."""
    group, member = _make_group()
    assert group.pos.device.type == "cpu" and group.quat.device.type == "cpu"

    group.set_pos(torch.tensor([1.0, 2.0, 3.0], device="cuda"))
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)

    # A later sequence setter must adopt (not fight) the canonical device.
    group.set_quat((1.0, 0.0, 0.0, 0.0))
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)


@_requires_cuda
def test_cuda_set_quat_on_cpu_initialized_group():
    """CPU-initialised group + set_quat(cuda) migrates the position anchor."""
    group, member = _make_group()
    group.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"))
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)

    # Independently, a following CPU sequence setter stays on CUDA.
    group.set_pos((0.0, 0.0, 0.8))
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)
    assert torch.allclose(
        group.pos, torch.tensor([0.0, 0.0, 0.8], device="cuda"),
    )


@_requires_cuda
def test_cuda_set_pose_on_cpu_initialized_group():
    """CPU-initialised group + set_pose(cuda, cuda) keeps one canonical pair."""
    group, member = _make_group()
    group.set_pose(
        torch.tensor([0.5, 0.0, 0.8], device="cuda"),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda"),
    )
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)


@_requires_cuda
def test_cuda_set_pose_mixed_inputs_single_canonical():
    """set_pose with a CUDA tensor pos and a CPU sequence quat stays CUDA."""
    group, member = _make_group()
    group.set_pose(
        torch.tensor([0.5, 0.0, 0.8], device="cuda"),
        (1.0, 0.0, 0.0, 0.0),
    )
    _assert_canonical(group, "cuda", torch.float32)
    _assert_member_canonical(member, "cuda", torch.float32)


@_requires_cuda
def test_cuda_composition_and_batched_masked_apply():
    """Member composition and masked batched apply run fully on CUDA."""
    group, member = _make_group()
    n_envs = 4
    group.set_pose(
        torch.zeros(n_envs, 3, device="cuda"),
        torch.tensor([1.0, 0.0, 0.0, 0.0], device="cuda").expand(n_envs, 4),
    )
    _assert_canonical(group, "cuda", torch.float32)
    assert group.pos.shape == (n_envs, 3)
    assert member.last_pos.device.type == "cuda"
    assert member.last_pos.shape == (n_envs, 3)

    # Masked apply forwards only the selected rows, on device.
    member.calls.clear()
    mask = torch.zeros(n_envs, dtype=torch.bool, device="cuda")
    mask[[0, 2]] = True
    group.set_pos(torch.ones(n_envs, 3, device="cuda"), envs_idx=mask)
    assert member.last_pos.device.type == "cuda"
    assert member.last_pos.shape == (2, 3)
    assert group.pos.shape == (n_envs, 3)

    # world_pose composes on the canonical device.
    w_pos, w_quat = group.world_pose(local_pos=(0.1, 0.0, -0.01))
    assert w_pos.device.type == "cuda" and w_quat.device.type == "cuda"


if __name__ == "__main__":
    test_set_pos_tensor_makes_dtype_canonical()
    test_set_quat_tensor_makes_dtype_canonical()
    test_set_pose_tensors_make_dtype_canonical()
    test_sequence_input_adopts_current_canonical()
    test_explicit_dtype_is_preserved()
    test_composition_world_pose_follows_anchor()
    test_batched_anchor_masked_apply_cpu()
    if _CUDA_AVAILABLE:
        test_cuda_set_pos_on_cpu_initialized_group()
        test_cuda_set_quat_on_cpu_initialized_group()
        test_cuda_set_pose_on_cpu_initialized_group()
        test_cuda_set_pose_mixed_inputs_single_canonical()
        test_cuda_composition_and_batched_masked_apply()
    print("All tests passed!")
