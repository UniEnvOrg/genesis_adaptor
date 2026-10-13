"""Small shared helpers for :mod:`unienv_genesis_refpolicy`."""

from __future__ import annotations

import logging
from typing import Any, Optional

LOGGER = logging.getLogger(__name__)

__all__ = ["prime_ik_scratch"]


def prime_ik_scratch(
    eef_controller: Any,
    *,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Allocate Genesis's IK scratch buffers *outside* inference mode.

    Genesis allocates its IK scratch lazily, on the first solve.  Reference
    policies reset inside ``torch.inference_mode()`` (e.g. the grasp policy
    evaluates grasp candidates with IK), so a naive first solve would create the
    scratch as an **inference tensor** — which the (non-inference-mode)
    environment step may then no longer update in place, failing with
    ``RuntimeError: Inplace update to inference tensor outside InferenceMode is
    not allowed``.

    Solving once for the *current* end-effector pose here pre-allocates the
    scratch as a normal tensor.  It commands no motion (the solved pose is the
    one the arm already holds), so it is safe to call before/while building a
    policy graph.

    Args:
        eef_controller: An end-effector pose controller exposing
            ``get_eef_pose_absolute()`` and ``solve_ik_absolute(position,
            quaternion)``.
        logger: Optional logger for the failure warning.  Defaults to this
            module's logger.
    """
    log = logger if logger is not None else LOGGER
    try:
        position, quaternion = eef_controller.get_eef_pose_absolute()
        eef_controller.solve_ik_absolute(position.detach(), quaternion.detach())
    except Exception as error:  # pragma: no cover - solver/environment specific
        log.warning(
            "prime_ik_scratch: could not pre-allocate the IK scratch buffers: %s",
            error,
        )
