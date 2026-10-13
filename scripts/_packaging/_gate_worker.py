"""Run one batch of a prepare_*.py sanity gate in a child process.

Genesis scene teardown leaks host RAM through glibc arena retention (freed
watertighten/SDF buffers never return to the OS), so a long in-process gate
balloons RSS until the kernel OOM-killer fires — and its dying taichi probe
workers respawn each other into a storm that then exhausts memory.  Running
each batch in a subprocess means the child exits and the OS reclaims ALL of
its memory (and any probe workers die with it), bounding peak RSS.

Usage:
    python scripts/_packaging/_gate_worker.py <script_module_path> <output_dir> \
        <render_dir> <batch_size> <bbox_tol_abs> <bbox_tol_rel> <object_id...>

Prints one JSON line to stdout:
    {"kept": [...], "passed": [...], "dropped": [...]}
"""
import json
import sys
from pathlib import Path


def main() -> int:
    import faulthandler

    faulthandler.enable()  # dump the Python stack if the parent SIGKILLs us
    script_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    render_dir = None if sys.argv[3] == "-" else Path(sys.argv[3])
    bbox_tol_abs = float(sys.argv[4])
    bbox_tol_rel = float(sys.argv[5])
    object_ids = sys.argv[6:]

    # The repo root (for `unienv_genesis_collection.*` and the shared
    # `scripts/_packaging` package the script imports) — the script lives in
    # the repo's scripts/ directory, so walk up from there to the root that
    # carries the collection package.
    repo_root = script_path.parent
    while not (repo_root / "unienv_genesis_collection").is_dir():
        if repo_root.parent == repo_root:
            repo_root = script_path.parent.parent.parent  # scripts/packaging fallback
            break
        repo_root = repo_root.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    scripts_dir = script_path.parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    import importlib.util

    # Patch Genesis's z3 collision-bitmask synthesis BEFORE any USD is loaded:
    # dense FilteredPairsAPI assets (>=315 colliders) otherwise spin z3 for
    # 100-220 s per parse (x3 scenes) and hit the watchdog.  The constructive
    # fast path solves them in milliseconds.  Idempotent; no-op without z3.
    try:
        from unienv_genesis_collection import genesis_compat

        genesis_compat.install()
    except Exception:
        pass

    spec = importlib.util.spec_from_file_location("gate_script", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    kept, passed, dropped = module.run_sanity_gate(
        output_dir,
        object_ids,
        render_dir=render_dir,
        bbox_tol_abs=bbox_tol_abs,
        bbox_tol_rel=bbox_tol_rel,
    )
    print("GATE_RESULT_JSON=" + json.dumps(
        {"kept": kept, "passed": passed, "dropped": dropped}
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
