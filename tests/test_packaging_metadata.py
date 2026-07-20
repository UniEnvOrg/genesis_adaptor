"""Packaging metadata checks for pyproject.toml.

Verifies that direct runtime imports are declared as dependencies, that the
NYX plugin is exposed as an optional extra, and that built wheels contain
only the three intended package roots (no build/, dist/, scripts/, tests,
examples, or duplicate stale package trees).

Python 3.10 compatible: ``tomllib`` is stdlib only from 3.11, so we fall
back to ``tomli`` (declared in the ``test`` optional extra for
``python_version < '3.11'``; install with ``pip install -e '.[test]'``).
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PYPROJECT_PATH = Path(__file__).resolve().parents[1] / "pyproject.toml"
PROJECT_ROOT = PYPROJECT_PATH.parent

INTENDED_PACKAGE_ROOTS = {
    "unienv_genesis",
    "unienv_genesis_collection",
    "unienv_genesis_refpolicy",
}
FORBIDDEN_WHEEL_ROOTS = {"build", "dist", "scripts", "tests", "examples", "training"}


def _load_project() -> dict:
    with open(PYPROJECT_PATH, "rb") as fh:
        return tomllib.load(fh)["project"]


def _normalize_name(requirement: str) -> str:
    """Extract a normalized PEP 503-ish package name from a requirement."""
    name = requirement.split(";")[0].strip()
    for sep in ("<", ">", "=", "!", "~", "[", " "):
        name = name.split(sep)[0]
    return name.strip().lower().replace("_", "-")


def test_direct_dependencies_declared():
    project = _load_project()
    names = {_normalize_name(dep) for dep in project["dependencies"]}
    # Newly declared direct runtime imports.
    for required in ("numpy", "xbarray", "trimesh"):
        assert required in names, f"missing direct dependency: {required}"
    # Pre-existing dependencies must be preserved.
    for kept in ("unienv", "genesis-world", "torch", "pyyaml", "huggingface-hub"):
        assert kept in names, f"pre-existing dependency removed: {kept}"


def test_tomli_not_a_runtime_dependency():
    """Test-only tomli must not leak into runtime dependencies."""
    project = _load_project()
    names = {_normalize_name(dep) for dep in project["dependencies"]}
    for test_only in ("tomli", "build", "pytest"):
        assert test_only not in names, (
            f"{test_only} is test-only and must live in the test extra, not "
            f"runtime dependencies"
        )


def test_test_extra_declared():
    """The test extra carries the runner, wheel builder, and 3.10 fallback."""
    project = _load_project()
    extras = project.get("optional-dependencies", {})
    assert "test" in extras, "missing [project.optional-dependencies] test extra"
    test_reqs = extras["test"]
    names = {_normalize_name(dep) for dep in test_reqs}
    assert "pytest" in names, f"test extra must include pytest: {test_reqs}"
    assert "build" in names, (
        f"test extra must include build (wheel-content regression): {test_reqs}"
    )
    tomli_reqs = [dep for dep in test_reqs if _normalize_name(dep) == "tomli"]
    assert tomli_reqs, f"test extra must include conditional tomli: {test_reqs}"
    assert all(
        "python_version" in dep and "3.11" in dep for dep in tomli_reqs
    ), f"tomli must be conditioned on python_version < 3.11: {tomli_reqs}"


def test_nyx_optional_extra_declared():
    project = _load_project()
    extras = project.get("optional-dependencies", {})
    assert "nyx" in extras, "missing [project.optional-dependencies] nyx extra"
    nyx_reqs = extras["nyx"]
    assert any(
        _normalize_name(dep) == "gs-nyx-plugin" for dep in nyx_reqs
    ), f"nyx extra does not require gs-nyx-plugin: {nyx_reqs}"
    assert any(
        ">=0.1.4" in dep.replace(" ", "") for dep in nyx_reqs
    ), f"nyx extra must pin gs-nyx-plugin>=0.1.4: {nyx_reqs}"


def test_package_discovery_restricted_to_intended_roots():
    with open(PYPROJECT_PATH, "rb") as fh:
        full = tomllib.load(fh)
    find_cfg = full["tool"]["setuptools"]["packages"]["find"]
    include = set(find_cfg.get("include", []))
    assert include != {"*"}, "package discovery must not use a broad include"
    for root in INTENDED_PACKAGE_ROOTS:
        assert root in include or f"{root}.*" in include, (
            f"intended package root {root} is not covered by include patterns"
        )
    for forbidden in FORBIDDEN_WHEEL_ROOTS:
        assert not any(
            pat == forbidden or pat.startswith(forbidden + ".") for pat in include
        ), f"include patterns must not cover {forbidden}"


def test_built_wheel_contains_only_intended_packages(tmp_path):
    """Build a wheel and assert no stale/extra trees are shipped.

    Requires the ``build`` package, which is mandatory under the ``test``
    optional extra (``pip install -e '.[test]'``).
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "-o",
            str(tmp_path),
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert names, "wheel is empty"

    top_levels = set()
    for name in names:
        top = name.split("/")[0]
        if top.endswith(".dist-info"):
            continue
        top_levels.add(top)

    shipped_forbidden = top_levels & FORBIDDEN_WHEEL_ROOTS
    assert not shipped_forbidden, (
        f"wheel must not ship {sorted(shipped_forbidden)} trees"
    )
    assert top_levels <= INTENDED_PACKAGE_ROOTS, (
        f"unexpected top-level wheel entries: "
        f"{sorted(top_levels - INTENDED_PACKAGE_ROOTS)}"
    )
    assert top_levels == INTENDED_PACKAGE_ROOTS, (
        f"missing intended package roots: "
        f"{sorted(INTENDED_PACKAGE_ROOTS - top_levels)}"
    )

    # No duplicate stale package trees nested anywhere (e.g. build/lib/...
    # copies of the real packages).
    for name in names:
        dirs = name.split("/")[:-1]
        assert not any(d in FORBIDDEN_WHEEL_ROOTS for d in dirs), (
            f"stale/duplicate tree leaked into wheel member: {name}"
        )
