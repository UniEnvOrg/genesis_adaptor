"""Batch the Genesis sanity gate across memory-isolated worker subprocesses.

Genesis scene teardown leaks host RAM through glibc arena retention (freed
watertighten/SDF buffers never return to the OS), so a long in-process gate
balloons RSS until the kernel OOM-killer fires — and the dying taichi probe
workers then respawn each other into a storm that exhausts memory.  Running
each batch in a worker subprocess means the child exits and the OS reclaims
ALL of its memory (and any probe workers die with it), bounding peak RSS.

Each ``prepare_*_assets.py`` keeps its own ``run_sanity_gate`` (per-object
loader + bbox + render checks); this driver merely re-invokes the script's
module inside a child for a slice of the object ids and merges the results.
Set ``GATE_BATCH_SIZE=0`` to disable batching (run in-process, old behaviour).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_WORKER = Path(__file__).with_name("_gate_worker.py")


def run_sanity_gate_batched(
    script_file: str,
    run_sanity_gate,
    output_dir: Path,
    object_ids: Sequence[str],
    *,
    render_dir: Optional[Path],
    bbox_tol_abs: float,
    bbox_tol_rel: float,
    batch_size: Optional[int] = None,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Run ``run_sanity_gate`` over *object_ids* in memory-isolated batches.

    Falls back to a single in-process call when ``GATE_BATCH_SIZE`` is ``0``
    or there is only one (small) batch.
    """
    if batch_size is None:
        batch_size = int(os.environ.get("GATE_BATCH_SIZE", "25"))
    ids = list(object_ids)
    if batch_size <= 0 or len(ids) <= batch_size:
        return run_sanity_gate(
            output_dir, ids,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
        )

    kept: List[str] = []
    passed: List[dict] = []
    dropped: List[dict] = []
    n_batches = (len(ids) + batch_size - 1) // batch_size
    for index in range(0, len(ids), batch_size):
        chunk = ids[index:index + batch_size]
        batch_no = index // batch_size + 1
        print(
            f"\n[gate batch {batch_no}/{n_batches}] {len(chunk)} object(s) "
            f"in a worker subprocess ..."
        )
        kept_b, passed_b, dropped_b = _run_one_batch(
            script_file, output_dir, chunk,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
        )
        kept.extend(kept_b)
        passed.extend(passed_b)
        dropped.extend(dropped_b)
        print(
            f"[gate batch {batch_no}/{n_batches}] done: "
            f"{len(kept_b)} OK / {len(dropped_b)} dropped"
        )
    return kept, passed, dropped


def _run_one_batch(
    script_file: str,
    output_dir: Path,
    object_ids: Sequence[str],
    *,
    render_dir: Optional[Path],
    bbox_tol_abs: float,
    bbox_tol_rel: float,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Run one batch in a worker, with a watchdog timeout.

    A gate object can wedge a worker (a single Python thread spinning in an
    unbounded loop — observed on a MolmoSpaces fridge whose joint handling
    never terminates).  The watchdog kills a worker that exceeds
    ``GATE_OBJECT_TIMEOUT`` seconds **per object** (default 600 s, generous
    for the slowest watertighten); a timed-out multi-object batch is retried
    object-by-object so only the genuinely stuck object is dropped, and a
    timed-out singleton is dropped with a ``TimeoutError`` record.
    """
    per_object_timeout = float(os.environ.get("GATE_OBJECT_TIMEOUT", "600"))
    timeout = max(per_object_timeout * len(object_ids), per_object_timeout)
    try:
        return _invoke_worker(
            script_file, output_dir, object_ids,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        if len(object_ids) == 1:
            object_id = object_ids[0]
            print(
                f"  [gate] {object_id}: DROPPED — TimeoutError: gate exceeded "
                f"{per_object_timeout:.0f}s (worker killed; likely an "
                f"unbounded loop in this object's load/articulation)"
            )
            return [], [], [dict(
                object_id=object_id, status="dropped",
                error=f"TimeoutError: gate exceeded {per_object_timeout:.0f}s",
            )]
        # Split the batch: salvage the objects around the stuck one.
        print(
            f"  [gate batch] TIMEOUT after {timeout:.0f}s — retrying the "
            f"{len(object_ids)} object(s) one-by-one to isolate the stuck one."
        )
        return _run_objects_one_by_one(
            script_file, output_dir, object_ids,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
        )
    except _WorkerCrashed as exc:
        # A non-zero worker exit (e.g. SIGABRT from a native USD/CUDA abort, or
        # the kernel OOM-killer SIGKILL) means the batch's result payload was
        # lost even though some objects may have gated OK before the crash.
        # Retry object-by-object so the run survives a single crashing object
        # instead of aborting wholesale; a singleton that keeps crashing is
        # dropped with a crash record.
        if len(object_ids) == 1:
            object_id = object_ids[0]
            print(
                f"  [gate] {object_id}: DROPPED — gate worker crashed "
                f"(exit {exc.returncode})"
            )
            return [], [], [dict(
                object_id=object_id, status="dropped",
                error=f"WorkerCrash: gate worker exited {exc.returncode}",
            )]
        print(
            f"  [gate batch] worker CRASHED (exit {exc.returncode}) on a batch "
            f"of {len(object_ids)} object(s) starting {object_ids[0]!r} — "
            f"retrying one-by-one to isolate the crashing object."
        )
        return _run_objects_one_by_one(
            script_file, output_dir, object_ids,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
        )


def _run_objects_one_by_one(
    script_file: str,
    output_dir: Path,
    object_ids: Sequence[str],
    *,
    render_dir: Optional[Path],
    bbox_tol_abs: float,
    bbox_tol_rel: float,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Gate each object in its own worker (salvage path after a batch failure)."""
    kept: List[str] = []
    passed: List[dict] = []
    dropped: List[dict] = []
    for object_id in object_ids:
        k, p, d = _run_one_batch(
            script_file, output_dir, [object_id],
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
        )
        kept.extend(k)
        passed.extend(p)
        dropped.extend(d)
    return kept, passed, dropped


class _WorkerCrashed(RuntimeError):
    """The gate worker subprocess exited non-zero (crash / OOM-kill)."""

    def __init__(self, returncode: int, message: str) -> None:
        super().__init__(message)
        self.returncode = returncode


def _invoke_worker(
    script_file: str,
    output_dir: Path,
    object_ids: Sequence[str],
    *,
    render_dir: Optional[Path],
    bbox_tol_abs: float,
    bbox_tol_rel: float,
    timeout: float,
) -> Tuple[List[str], List[dict], List[dict]]:
    cmd = [
        sys.executable, str(_WORKER), str(Path(script_file).resolve()),
        str(output_dir), str(render_dir) if render_dir is not None else "-",
        str(bbox_tol_abs), str(bbox_tol_rel), *object_ids,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
    )
    # Echo the worker's gate lines so the log keeps the familiar [gate] output.
    if result.stdout:
        for line in result.stdout.splitlines():
            if not line.startswith("GATE_RESULT_JSON="):
                print(line)
    if result.returncode != 0:
        sys.stderr.write(result.stderr[-4000:])
        raise _WorkerCrashed(
            result.returncode,
            f"Gate worker exited {result.returncode} on a batch of "
            f"{len(object_ids)} object(s) starting {object_ids[0]!r}.",
        )
    marker = "GATE_RESULT_JSON="
    payload = next(
        (line[len(marker):] for line in result.stdout.splitlines()
         if line.startswith(marker)),
        None,
    )
    if payload is None:
        sys.stderr.write(result.stderr[-4000:])
        raise RuntimeError("Gate worker produced no result payload.")
    data = json.loads(payload)
    return data["kept"], data["passed"], data["dropped"]
