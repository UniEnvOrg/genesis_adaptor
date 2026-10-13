#!/usr/bin/env python3
# Upstream asset license: CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial
# 4.0 International) per the Lightwheel-simready-asset repo README — NON-COMMERCIAL
# USE ONLY.  This packager only rehosts the USD payloads unchanged; the original
# license applies to the packaged assets.  Attribution required (lightwheel_ prefix).
"""Prepare Lightwheel SimReady (open bundle) USD assets for the collection's assets repo.

Downloads (or reuses) the `LightwheelAI/Lightwheel-simready-asset
<https://github.com/LightwheelAI/Lightwheel-simready-asset>`_ bundle's
``Manipulation/`` USD assets and repackages them into the layout expected by
``unienv_genesis_collection.objects.hub_objects.articulated.lightwheel_furniture``
(one self-contained directory per object, loaded with ``gs.morphs.USD`` via
``scene.add_stage``)::

    <output>/objects/lightwheel/LICENSE                  # CC BY-NC 4.0 (canonical legal-code text)
    <output>/objects/lightwheel/README.md                # provenance + NC caveat
    <output>/objects/lightwheel/catalog.json             # generated annotations + tags
    <output>/objects/lightwheel/packaging_log.json       # per-object gate log
    <output>/objects/lightwheel/<object_id>/<object_id>.usd
    <output>/objects/lightwheel/<object_id>/texture/...  # relative textures

**MDL → UsdPreviewSurface converter (mandatory).**  The bundle authors
shading as OmniPBR MDL, which usd-core cannot read — Genesis loses the
materials silently.  Every packaged stage is rewritten with
:func:`~unienv_genesis_collection.utils.usd_conversions.convert_mdl_to_preview_surface`
(the shared converter also used by the Warehouse pipeline): diffuse/ORM
(roughness g, metallic b)/normal maps re-wired into UsdPreviewSurface, the
MDL shader prim removed, the articulation untouched.

**No joint-body swap needed** — the bundle authors ``body0`` = parent /
``body1`` = child, already Genesis's order (verified: Microwave043 →
3 links / 2 revolute DOFs single-entity as shipped).

**License — CC BY-NC 4.0 (non-commercial).**  The bundle's README states
"Creative Commons Attribution-NonCommercial 4.0 International"
("Free for non-commercial use") and require the ``lightwheel_{asset_name}``
naming format for attribution — hence the ``lightwheel_``-prefixed object
ids.  The archive ships **no LICENSE file** (only the GitHub repo carries
one); the pack always writes one — the canonical CC BY-NC 4.0 Creative
Commons legal-code text — next to the assets; the non-commercial clause is
accepted by the collection maintainer and flagged in
``docs/articulated_rooms.md`` (mirroring the ReplicaCAD caveat).

**Locomotion subset (``--subset locomotion``/``all``).**  The same bundle's
``Locomotion/`` directory holds eight scene categories: seven are *jointless*
static stages (terrains, an apartment, a two-storey staircase) and one —
``KitchenRoom`` — is a genuinely articulated room (35 revolute + 23 prismatic
joints, 131 rigid bodies).  Routing is an explicit per-category table:

* ``KitchenRoom_RSS.usd`` → the existing **articulated** pipeline, packaged as
  ``lightwheel_KitchenRoom_RSS`` into ``objects/lightwheel/`` (the sibling
  ``KitchenRoom.usd`` is unusable — an absolute escaping payload plus missing
  sublayers);
* the other seven → the new **static environment** family in
  ``objects/lightwheel_env/lightwheel_env_<Category>/`` (``2-StoryStaircase``
  is sanitised to ``lightwheel_env_2_story_staircase``), loaded by
  ``unienv_genesis_collection.objects.hub_objects.lightwheel_env.LightwheelEnv``.

Each category's root layer comes from the measured :data:`LOCOMOTION_ROOTS`
map and is repackaged by a **reference-closure copy**
(:func:`repackage_scene`): the root plus ``_usd_required_files`` — everything
else (``E/``, ``.thumbs``, ``Table049/``, ``ClothDemo_002/``, the S3-mirror
dir) is excluded by construction, while referenced payload dirs are kept
exactly when needed.  Apartment's 20 upstream UE material records
(``Game/.../MI_leaf_32_1.MI_leaf_32_1``) are warn-and-recorded as
``unresolvable_refs`` rather than failing the pack; missing ``.usd`` layers and
textures stay fatal — except the one upstream path-authoring bug the bundle
ships: a texture path with a stale exporter prefix whose file sits in the
expected directory under the stripped name (KitchenRoom's
``./textures/3d66Model-19913701-files-024.JPG`` vs the shipped
``textures/19913701-files-024.JPG``) is repaired by copying that file under
the referenced name and printing the repair.  The static gate
(:func:`sanity_check_static_env`) asserts no joints, collision present and an
offline/loaded bbox agreement (the offline bbox mirrors the loader's geometry
selection — see ``LightwheelEnv.get_local_bounding_box``).

Usage::

    # from a local extraction of Lightwheel_OpenSource.zip (Manipulation/…):
    python scripts/prepare_lightwheel_assets.py \\
        --source <Lightwheel_OpenSource> --output-dir /tmp/lw_assets \\
        --objects Microwave043,Stovetop002,Toaster003
    # the Locomotion scenes (static environments + lightwheel_KitchenRoom_RSS):
    python scripts/prepare_lightwheel_assets.py --subset locomotion \\
        --source <Lightwheel_OpenSource> --output-dir /tmp/lw_assets
    # or download the bundle (3.41 GB Google Drive, anonymous link):
    python scripts/prepare_lightwheel_assets.py --subset all --output-dir /tmp/lw_assets

Then upload to the Hub (needs write access to the org)::

    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/lw_assets/objects/lightwheel objects/lightwheel --repo-type dataset
    hf upload UniEnvOrg/unienv_genesis_collection_assets \\
        /tmp/lw_assets/objects/lightwheel_env objects/lightwheel_env --repo-type dataset
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Shared packaging/gating library (scripts/_packaging) — the single source
# of truth for the four packagers' gates (see the architecture review §1):
# quat_to_matrix/_WorldStub/_render_thumbnail, write_catalog, the
# asset-completeness check and the sanity gate + its batching all live there.
# Importing the package also installs the genesis_compat shims (idempotent).
# The scripts run as __main__ from the repo root (and the tests load them via
# spec_from_file_location), so scripts/ is not necessarily importable — bind
# it explicitly.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
for _path in (str(_REPO_ROOT), str(_SCRIPTS_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from _packaging._gate_batch import (  # noqa: E402
    run_sanity_gate_batched as _run_sanity_gate_batched,
)
from _packaging.catalog import write_catalog as _write_catalog_impl  # noqa: E402
from _packaging.gate_common import (  # noqa: E402
    _WorldStub,
    _finite_midpoint,
    _gs_joint_type_name,
    _opened_reference_joints,
    _render_thumbnail,
    quat_to_matrix,
)
from _packaging.sanity import (  # noqa: E402
    BBOX_TOL_ABS,
    BBOX_TOL_REL,
    run_sanity_gate as _run_sanity_gate_impl,
    sanity_check_articulated,
)
from _packaging.usd_verify import (  # noqa: E402
    IMAGE_SUFFIXES as _IMAGE_SUFFIXES,
    USD_SUFFIXES as _USD_SUFFIXES,
    is_fatal_missing_ref as _is_fatal_missing_ref,
    recoverable_texture as _recoverable_texture,
    verify_object_assets as _verify_object_assets,
)

sanity_check_object = sanity_check_articulated  # the pre-refactor family name


DRIVE_FILE_ID = "1S1W-vDNvsOOQU0qViEax_QS0zr0zGLfX"
DRIVE_VIEW_URL = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view"
BUNDLE_NAME = "Lightwheel_OpenSource.zip"
# The Drive file advertises no checksum; its size is the sanity gate for a
# truncated anonymous pull (content-length 3 409 389 226 ≈ 3.41 GB).
BUNDLE_EXPECTED_BYTES = 3_409_389_226
BUNDLE_MIN_BYTES = int(BUNDLE_EXPECTED_BYTES * 0.9)
UPSTREAM_MANIPULATION_DIR = "Manipulation"
UPSTREAM_LOCOMOTION_DIR = "Locomotion"
OUTPUT_ASSETS_SUBDIR = "objects/lightwheel"
# Static (jointless) Locomotion categories live in their own family tree: the
# `objects/lightwheel/` catalog semantics are per-object furniture, its
# descriptor asserts a single articulated entity and its gate requires moving
# joints — a terrain/room violates all three.
OUTPUT_ENV_ASSETS_SUBDIR = "objects/lightwheel_env"
HF_UPLOAD_REPO = "UniEnvOrg/unienv_genesis_collection_assets"

# ---------------------------------------------------------------------------
# Locomotion subset (measured root layers — plan §3a)
# ---------------------------------------------------------------------------
# Do NOT auto-derive the root: every category's composed scene root was
# verified by reference closure.  ``KitchenRoom.usd`` is unusable (an absolute
# escaping payload ``../../../../../Downloads/Table049/Table049.usd`` plus two
# missing ``UnitsAdjust-*.metricsAssembler`` sublayers); ``KitchenRoom_RSS.usd``
# composes the full room.  ``Ground_stone.usd`` is a *payload* of
# ``SLATEGround.usd``, not an alternative root.
LOCOMOTION_ROOTS: Dict[str, str] = {
    "2-StoryStaircase": "2-StoryStaircase.usd",
    "Apartment": "scene_04.usd",
    "Grass": "Grass.usd",
    "GravelGround": "GravelGround.usd",
    "KitchenRoom": "KitchenRoom_RSS.usd",
    "MudGround": "MudGround.usd",
    "SLATEGround": "SLATEGround.usd",
    "SnowGround": "SnowGround.usd",
}

# Per-category routing (plan §4): ``category → (family, out_root, root_layer)``.
# KitchenRoom is a *genuinely articulated* scene (35 revolute + 23 prismatic
# joints, 131 rigid bodies) and goes through the existing articulated pipeline
# unchanged; the other seven are jointless static stages.  The routing is an
# explicit table — deriving it from joint counts at packaging time would be
# non-deterministic if upstream re-exports a category.
LOCOMOTION_ARTICULATED_CATEGORIES = ("KitchenRoom",)


def _sanitise_category_token(category: str) -> str:
    """``2-StoryStaircase`` → ``2_story_staircase`` (id-safe token).

    Categories whose name is already id-safe keep their upstream spelling
    (``GravelGround``, ``SLATEGround``); only names carrying non-alphanumeric
    characters are camel-split and lowercased.
    """
    import re

    if re.fullmatch(r"[0-9A-Za-z_]+", category):
        return category
    token = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", category)
    return re.sub(r"[^0-9A-Za-z]+", "_", token).strip("_").lower()


def locomotion_object_id(category: str) -> str:
    """The packaged object id of a Locomotion category."""
    if category in LOCOMOTION_ARTICULATED_CATEGORIES:
        return f"lightwheel_{LOCOMOTION_ROOTS[category].removesuffix('.usd')}"
    return f"lightwheel_env_{_sanitise_category_token(category)}"


def locomotion_family(category: str) -> str:
    """``"articulated"`` (KitchenRoom) or ``"static"`` (the other seven)."""
    return (
        "articulated"
        if category in LOCOMOTION_ARTICULATED_CATEGORIES
        else "static"
    )


def locomotion_out_root(category: str, output_dir: Path) -> Path:
    """The output tree a category is packaged into (plan §1/§4)."""
    subdir = (
        OUTPUT_ASSETS_SUBDIR
        if locomotion_family(category) == "articulated"
        else OUTPUT_ENV_ASSETS_SUBDIR
    )
    return output_dir / subdir


DEFAULT_OBJECTS: Optional[str] = None  # all Manipulation assets with joints

# Bbox cross-check tolerances (offline-FK box proxy vs real mesh geometry)
# — defined once in ``_packaging.sanity`` (BBOX_TOL_ABS/BBOX_TOL_REL) and
# imported above.

# CC BY-NC 4.0 — the Drive bundle ships **no LICENSE file**; the GitHub repo
# (LightwheelAI/Lightwheel-simready-asset) carries the CC BY-NC 4.0 LICENSE
# and its README states "Free for non-commercial use".  To keep the pack
# never without an attribution file the standard Creative Commons legal-code
# text is embedded below (character-for-character
# https://creativecommons.org/licenses/by-nc/4.0/legalcode.txt).
LICENSE_TEXT = """\
Lightwheel SimReady open-source assets (Manipulation subset).

Source: "Lightwheel OpenSource" — LightwheelAI/Lightwheel-simready-asset,
https://github.com/LightwheelAI/Lightwheel-simready-asset
(assets distributed via the linked Google Drive bundle).

License: Creative Commons Attribution-NonCommercial 4.0 International
("Attribution-NonCommercial 4.0 International"; the bundle README states
"Free for non-commercial use").  The upstream archive ships **no LICENSE
file** — the license text is stated in the upstream GitHub repo's README —
so the pack embeds the canonical Creative Commons legal-code text below.
The licence condition, in the Commons deed's words: "NonCommercial — You
may not use the material for commercial purposes."

Attribution required: use the naming format `lightwheel_{asset_name}` when
redistributing these assets (the packaged object ids carry the
`lightwheel_` prefix for this reason).

NON-COMMERCIAL USE ONLY — see docs/articulated_rooms.md for the caveat.

Repackaged by genesis_adaptor/scripts/prepare_lightwheel_assets.py.

---
Attribution-NonCommercial 4.0 International

Creative Commons Corporation ("Creative Commons") is not a law firm and
does not provide legal services or legal advice. Distribution of
Creative Commons public licenses does not create a lawyer-client or
other relationship. Creative Commons makes its licenses and related
information available on an "as-is" basis. Creative Commons gives no
warranties regarding its licenses, any material licensed under their
terms and conditions, or any related information. Creative Commons
disclaims all liability for damages resulting from their use to the
fullest extent possible.

Using Creative Commons Public Licenses

Creative Commons public licenses provide a standard set of terms and
conditions that creators and other rights holders may use to share
original works of authorship and other material subject to copyright
and certain other rights specified in the public license below. The
following considerations are for informational purposes only, are not
exhaustive, and do not form part of our licenses.

     Considerations for licensors: Our public licenses are
     intended for use by those authorized to give the public
     permission to use material in ways otherwise restricted by
     copyright and certain other rights. Our licenses are
     irrevocable. Licensors should read and understand the terms
     and conditions of the license they choose before applying it.
     Licensors should also secure all rights necessary before
     applying our licenses so that the public can reuse the
     material as expected. Licensors should clearly mark any
     material not subject to the license. This includes other CC-
     licensed material, or material used under an exception or
     limitation to copyright. More considerations for licensors:
    wiki.creativecommons.org/Considerations_for_licensors

     Considerations for the public: By using one of our public
     licenses, a licensor grants the public permission to use the
     licensed material under specified terms and conditions. If
     the licensor's permission is not necessary for any reason--for
     example, because of any applicable exception or limitation to
     copyright--then that use is not regulated by the license. Our
     licenses grant only permissions under copyright and certain
     other rights that a licensor has authority to grant. Use of
     the licensed material may still be restricted for other
     reasons, including because others have copyright or other
     rights in the material. A licensor may make special requests,
     such as asking that all changes be marked or described.
     Although not required by our licenses, you are encouraged to
     respect those requests where reasonable. More considerations
     for the public:
    wiki.creativecommons.org/Considerations_for_licensees

=======================================================================

Creative Commons Attribution-NonCommercial 4.0 International Public
License

By exercising the Licensed Rights (defined below), You accept and agree
to be bound by the terms and conditions of this Creative Commons
Attribution-NonCommercial 4.0 International Public License ("Public
License"). To the extent this Public License may be interpreted as a
contract, You are granted the Licensed Rights in consideration of Your
acceptance of these terms and conditions, and the Licensor grants You
such rights in consideration of benefits the Licensor receives from
making the Licensed Material available under these terms and
conditions.


Section 1 -- Definitions.

  a. Adapted Material means material subject to Copyright and Similar
     Rights that is derived from or based upon the Licensed Material
     and in which the Licensed Material is translated, altered,
     arranged, transformed, or otherwise modified in a manner requiring
     permission under the Copyright and Similar Rights held by the
     Licensor. For purposes of this Public License, where the Licensed
     Material is a musical work, performance, or sound recording,
     Adapted Material is always produced where the Licensed Material is
     synched in timed relation with a moving image.

  b. Adapter's License means the license You apply to Your Copyright
     and Similar Rights in Your contributions to Adapted Material in
     accordance with the terms and conditions of this Public License.

  c. Copyright and Similar Rights means copyright and/or similar rights
     closely related to copyright including, without limitation,
     performance, broadcast, sound recording, and Sui Generis Database
     Rights, without regard to how the rights are labeled or
     categorized. For purposes of this Public License, the rights
     specified in Section 2(b)(1)-(2) are not Copyright and Similar
     Rights.
  d. Effective Technological Measures means those measures that, in the
     absence of proper authority, may not be circumvented under laws
     fulfilling obligations under Article 11 of the WIPO Copyright
     Treaty adopted on December 20, 1996, and/or similar international
     agreements.

  e. Exceptions and Limitations means fair use, fair dealing, and/or
     any other exception or limitation to Copyright and Similar Rights
     that applies to Your use of the Licensed Material.

  f. Licensed Material means the artistic or literary work, database,
     or other material to which the Licensor applied this Public
     License.

  g. Licensed Rights means the rights granted to You subject to the
     terms and conditions of this Public License, which are limited to
     all Copyright and Similar Rights that apply to Your use of the
     Licensed Material and that the Licensor has authority to license.

  h. Licensor means the individual(s) or entity(ies) granting rights
     under this Public License.

  i. NonCommercial means not primarily intended for or directed towards
     commercial advantage or monetary compensation. For purposes of
     this Public License, the exchange of the Licensed Material for
     other material subject to Copyright and Similar Rights by digital
     file-sharing or similar means is NonCommercial provided there is
     no payment of monetary compensation in connection with the
     exchange.

  j. Share means to provide material to the public by any means or
     process that requires permission under the Licensed Rights, such
     as reproduction, public display, public performance, distribution,
     dissemination, communication, or importation, and to make material
     available to the public including in ways that members of the
     public may access the material from a place and at a time
     individually chosen by them.

  k. Sui Generis Database Rights means rights other than copyright
     resulting from Directive 96/9/EC of the European Parliament and of
     the Council of 11 March 1996 on the legal protection of databases,
     as amended and/or succeeded, as well as other essentially
     equivalent rights anywhere in the world.

  l. You means the individual or entity exercising the Licensed Rights
     under this Public License. Your has a corresponding meaning.


Section 2 -- Scope.

  a. License grant.

       1. Subject to the terms and conditions of this Public License,
          the Licensor hereby grants You a worldwide, royalty-free,
          non-sublicensable, non-exclusive, irrevocable license to
          exercise the Licensed Rights in the Licensed Material to:

            a. reproduce and Share the Licensed Material, in whole or
               in part, for NonCommercial purposes only; and

            b. produce, reproduce, and Share Adapted Material for
               NonCommercial purposes only.

       2. Exceptions and Limitations. For the avoidance of doubt, where
          Exceptions and Limitations apply to Your use, this Public
          License does not apply, and You do not need to comply with
          its terms and conditions.

       3. Term. The term of this Public License is specified in Section
          6(a).

       4. Media and formats; technical modifications allowed. The
          Licensor authorizes You to exercise the Licensed Rights in
          all media and formats whether now known or hereafter created,
          and to make technical modifications necessary to do so. The
          Licensor waives and/or agrees not to assert any right or
          authority to forbid You from making technical modifications
          necessary to exercise the Licensed Rights, including
          technical modifications necessary to circumvent Effective
          Technological Measures. For purposes of this Public License,
          simply making modifications authorized by this Section 2(a)
          (4) never produces Adapted Material.

       5. Downstream recipients.

            a. Offer from the Licensor -- Licensed Material. Every
               recipient of the Licensed Material automatically
               receives an offer from the Licensor to exercise the
               Licensed Rights under the terms and conditions of this
               Public License.

            b. No downstream restrictions. You may not offer or impose
               any additional or different terms or conditions on, or
               apply any Effective Technological Measures to, the
               Licensed Material if doing so restricts exercise of the
               Licensed Rights by any recipient of the Licensed
               Material.

       6. No endorsement. Nothing in this Public License constitutes or
          may be construed as permission to assert or imply that You
          are, or that Your use of the Licensed Material is, connected
          with, or sponsored, endorsed, or granted official status by,
          the Licensor or others designated to receive attribution as
          provided in Section 3(a)(1)(A)(i).

  b. Other rights.

       1. Moral rights, such as the right of integrity, are not
          licensed under this Public License, nor are publicity,
          privacy, and/or other similar personality rights; however, to
          the extent possible, the Licensor waives and/or agrees not to
          assert any such rights held by the Licensor to the limited
          extent necessary to allow You to exercise the Licensed
          Rights, but not otherwise.

       2. Patent and trademark rights are not licensed under this
          Public License.

       3. To the extent possible, the Licensor waives any right to
          collect royalties from You for the exercise of the Licensed
          Rights, whether directly or through a collecting society
          under any voluntary or waivable statutory or compulsory
          licensing scheme. In all other cases the Licensor expressly
          reserves any right to collect such royalties, including when
          the Licensed Material is used other than for NonCommercial
          purposes.


Section 3 -- License Conditions.

Your exercise of the Licensed Rights is expressly made subject to the
following conditions.

  a. Attribution.

       1. If You Share the Licensed Material (including in modified
          form), You must:

            a. retain the following if it is supplied by the Licensor
               with the Licensed Material:

                 i. identification of the creator(s) of the Licensed
                    Material and any others designated to receive
                    attribution, in any reasonable manner requested by
                    the Licensor (including by pseudonym if
                    designated);

                ii. a copyright notice;

               iii. a notice that refers to this Public License;

                iv. a notice that refers to the disclaimer of
                    warranties;

                 v. a URI or hyperlink to the Licensed Material to the
                    extent reasonably practicable;

            b. indicate if You modified the Licensed Material and
               retain an indication of any previous modifications; and

            c. indicate the Licensed Material is licensed under this
               Public License, and include the text of, or the URI or
               hyperlink to, this Public License.

       2. You may satisfy the conditions in Section 3(a)(1) in any
          reasonable manner based on the medium, means, and context in
          which You Share the Licensed Material. For example, it may be
          reasonable to satisfy the conditions by providing a URI or
          hyperlink to a resource that includes the required
          information.

       3. If requested by the Licensor, You must remove any of the
          information required by Section 3(a)(1)(A) to the extent
          reasonably practicable.

       4. If You Share Adapted Material You produce, the Adapter's
          License You apply must not prevent recipients of the Adapted
          Material from complying with this Public License.


Section 4 -- Sui Generis Database Rights.

Where the Licensed Rights include Sui Generis Database Rights that
apply to Your use of the Licensed Material:

  a. for the avoidance of doubt, Section 2(a)(1) grants You the right
     to extract, reuse, reproduce, and Share all or a substantial
     portion of the contents of the database for NonCommercial purposes
     only;

  b. if You include all or a substantial portion of the database
     contents in a database in which You have Sui Generis Database
     Rights, then the database in which You have Sui Generis Database
     Rights (but not its individual contents) is Adapted Material; and

  c. You must comply with the conditions in Section 3(a) if You Share
     all or a substantial portion of the contents of the database.

For the avoidance of doubt, this Section 4 supplements and does not
replace Your obligations under this Public License where the Licensed
Rights include other Copyright and Similar Rights.


Section 5 -- Disclaimer of Warranties and Limitation of Liability.

  a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
     EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
     AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
     ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
     IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
     WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
     PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
     ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
     KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
     ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.

  b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
     TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
     NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
     INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
     COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
     USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
     ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
     DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
     IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.

  c. The disclaimer of warranties and limitation of liability provided
     above shall be interpreted in a manner that, to the extent
     possible, most closely approximates an absolute disclaimer and
     waiver of all liability.


Section 6 -- Term and Termination.

  a. This Public License applies for the term of the Copyright and
     Similar Rights licensed here. However, if You fail to comply with
     this Public License, then Your rights under this Public License
     terminate automatically.

  b. Where Your right to use the Licensed Material has terminated under
     Section 6(a), it reinstates:

       1. automatically as of the date the violation is cured, provided
          it is cured within 30 days of Your discovery of the
          violation; or

       2. upon express reinstatement by the Licensor.

     For the avoidance of doubt, this Section 6(b) does not affect any
     right the Licensor may have to seek remedies for Your violations
     of this Public License.

  c. For the avoidance of doubt, the Licensor may also offer the
     Licensed Material under separate terms or conditions or stop
     distributing the Licensed Material at any time; however, doing so
     will not terminate this Public License.

  d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
     License.


Section 7 -- Other Terms and Conditions.

  a. The Licensor shall not be bound by any additional or different
     terms or conditions communicated by You unless expressly agreed.

  b. Any arrangements, understandings, or agreements regarding the
     Licensed Material not stated herein are separate from and
     independent of the terms and conditions of this Public License.


Section 8 -- Interpretation.

  a. For the avoidance of doubt, this Public License does not, and
     shall not be interpreted to, reduce, limit, restrict, or impose
     conditions on any use of the Licensed Material that could lawfully
     be made without permission under this Public License.

  b. To the extent possible, if any provision of this Public License is
     deemed unenforceable, it shall be automatically reformed to the
     minimum extent necessary to make it enforceable. If the provision
     cannot be reformed, it shall be severed from this Public License
     without affecting the enforceability of the remaining terms and
     conditions.

  c. No term or condition of this Public License will be waived and no
     failure to comply consented to unless expressly agreed to by the
     Licensor.

  d. Nothing in this Public License constitutes or may be interpreted
     as a limitation upon, or waiver of, any privileges and immunities
     that apply to the Licensor or You, including from the legal
     processes of any jurisdiction or authority.

=======================================================================

Creative Commons is not a party to its public
licenses. Notwithstanding, Creative Commons may elect to apply one of
its public licenses to material it publishes and in those instances
will be considered the “Licensor.” The text of the Creative Commons
public licenses is dedicated to the public domain under the CC0 Public
Domain Dedication. Except for the limited purpose of indicating that
material is shared under a Creative Commons public license or as
otherwise permitted by the Creative Commons policies published at
creativecommons.org/policies, Creative Commons does not authorize the
use of the trademark "Creative Commons" or any other trademark or logo
of Creative Commons without its prior written consent including,
without limitation, in connection with any unauthorized modifications
to any of its public licenses or any other arrangements,
understandings, or agreements concerning use of licensed material. For
the avoidance of doubt, this paragraph does not form part of the
public licenses.

Creative Commons may be contacted at creativecommons.org.
"""

PROVENANCE_README = """\
# Lightwheel SimReady (open bundle) articulated USD assets

Repackaged from the [Lightwheel open-source bundle](https://github.com/LightwheelAI/Lightwheel-simready-asset)
(259 SimReady USD assets, `Manipulation/` subset, one USD per object +
`texture/` sidecars): one self-contained directory per object under
`objects/lightwheel/<object_id>/`, containing the root USD stage
(`<object_id>.usd`, renamed from the upstream `<Asset>.usd`) plus its
`texture/` sidecar.

> **License note — non-commercial.**  The bundle is **CC BY-NC 4.0**
> ("Free for non-commercial use"); the non-commercial
> clause is accepted by the collection maintainer and flagged in
> `docs/articulated_rooms.md`.  The upstream archive ships **no LICENSE
> file** (the GitHub repo carries one); this pack always writes a `LICENSE`
> with the canonical CC BY-NC 4.0 legal-code text.  Attribution is required
> in the `lightwheel_{asset_name}` naming format — hence the `lightwheel_`
> prefix on the object ids.  See `LICENSE`.

* **Materials are rewritten from OmniPBR MDL into native UsdPreviewSurface**
  (`unienv_genesis_collection.utils.usd_conversions.convert_mdl_to_preview_surface`
  — usd-core cannot read `.mdl`; without the rewrite Genesis loses the
  materials silently).  The articulation is untouched.
* **No joint-body swap** — the bundle authors `body0` = parent / `body1` =
  child (Genesis's order), the opposite of MolmoSpaces.
* Some assets weld their base to the world upstream
  (`physics:kinematicEnabled` on the base body, e.g. Stovetop002); the
  descriptor's `fixed` flag overrides root fixity at load.
* Genesis grants USD joints by **full prim path**; the descriptor exposes
  *clean*, leaf-prim joint/link names (see the Articulated Rooms docs).
* Every shipped object passed the Genesis sanity gate: single-entity
  `add_stage` load, non-empty moving-joint set, and the offline box-proxy
  FK union bbox cross-checked against `entity.get_vAABB()` at the closed
  *and* a halfway-opened configuration.  Per-object details are recorded
  in `packaging_log.json`.

When using these models, please attribute the Lightwheel bundle
(`lightwheel_{asset_name}`, CC BY-NC 4.0) — see `LICENSE` and the license
note above.
"""


ENV_PROVENANCE_README = """\
# Lightwheel SimReady (open bundle) static environment USD stages

Repackaged from the [Lightwheel open-source bundle](https://github.com/LightwheelAI/Lightwheel-simready-asset)
(`Locomotion/` subset, the seven **jointless** scene categories): one
self-contained directory per environment under
`objects/lightwheel_env/<object_id>/`, containing the composed root USD stage
(`<object_id>.usd`) plus the payload files its reference closure needs
(relative paths preserved).

| object id | upstream category | root layer |
|---|---|---|
| `lightwheel_env_2_story_staircase` | `2-StoryStaircase` | `2-StoryStaircase.usd` |
| `lightwheel_env_Apartment` | `Apartment` | `scene_04.usd` |
| `lightwheel_env_Grass` | `Grass` | `Grass.usd` |
| `lightwheel_env_GravelGround` | `GravelGround` | `GravelGround.usd` |
| `lightwheel_env_MudGround` | `MudGround` | `MudGround.usd` |
| `lightwheel_env_SLATEGround` | `SLATEGround` | `SLATEGround.usd` |
| `lightwheel_env_SnowGround` | `SnowGround` | `SnowGround.usd` |

`KitchenRoom` is **not** here: `KitchenRoom_RSS.usd` is a genuinely articulated
scene (35 revolute + 23 prismatic joints, 131 rigid bodies) and ships through
the existing articulated tree as `objects/lightwheel/lightwheel_KitchenRoom_RSS`.

> **License note — non-commercial.**  The whole bundle is **CC BY-NC 4.0**
> ("Free for non-commercial use"); the non-commercial clause is accepted by
> the collection maintainer and flagged in `docs/articulated_rooms.md`.  The
> upstream archive ships **no LICENSE file** (the GitHub repo carries one);
> this pack always writes a `LICENSE` with the canonical CC BY-NC 4.0
> legal-code text.  Attribution is required in the
> `lightwheel_{asset_name}` naming format — hence the `lightwheel_env_`
> prefix on the object ids.  See `LICENSE`.

* **Packaging is a reference-closure copy**: only the root layer plus
  `_usd_required_files(root)` ship, so upstream junk (`E/`, `.thumbs`,
  `.DS_Store`, `.collect.mapping.json`, the S3-mirror dir) is excluded by
  construction while a referenced payload directory (e.g. the terrain mesh
  dirs) is kept exactly when the stage needs it.
* **Materials are rewritten from OmniPBR MDL into native UsdPreviewSurface**
  (`unienv_genesis_collection.utils.usd_conversions.convert_mdl_to_preview_surface`
  — usd-core cannot read `.mdl`).
* **Apartment** ships 20 upstream UE-export material records
  (`Game/.../MI_leaf_32_1.MI_leaf_32_1`) that exist nowhere in the bundle;
  they are recorded under `unresolvable_refs` in `packaging_log.json` instead
  of failing the pack (missing `.usd` layers and textures stay fatal).  The
  gate's load check is the tripwire.
* Every shipped environment passed the static sanity gate: no joints,
  collision present, offline pxr bbox consistent with the loaded entity's
  `get_vAABB()`, and sane extents.  Per-object details are recorded in
  `packaging_log.json`.
* Loaded with `gs.morphs.USD` + `scene.add_stage` by
  `unienv_genesis_collection.objects.hub_objects.lightwheel_env.LightwheelEnv`.

When using these environments, please attribute the Lightwheel bundle
(`lightwheel_{asset_name}`, CC BY-NC 4.0) — see `LICENSE` and the license
note above.
"""


# ---------------------------------------------------------------------------
# Upstream discovery
# ---------------------------------------------------------------------------

# Asset-name → (kind, rooms, detail) templates.  The bundle names assets
# ``<Kind><NNN>`` (Microwave043, Stovetop002, Toaster003, Refrigerator046…);
# unknown names fall back to the generic template.
NAME_TEMPLATES: Dict[str, Dict[str, object]] = {
    "microwave": dict(kind="appliance", rooms=("kitchen",), detail="a microwave with a hinged door"),
    "stovetop": dict(kind="appliance", rooms=("kitchen",), detail="a stovetop with knobs"),
    "oven": dict(kind="appliance", rooms=("kitchen",), detail="an oven with a hinged door"),
    "toaster": dict(kind="appliance", rooms=("kitchen",), detail="a toaster with a lever and a lid"),
    "refrigerator": dict(kind="appliance", rooms=("kitchen",), detail="a refrigerator with hinged doors"),
    "dishwasher": dict(kind="appliance", rooms=("kitchen",), detail="a dishwasher with a hinged door"),
    "rangehood": dict(kind="appliance", rooms=("kitchen",), detail="a range hood"),
    "coffee": dict(kind="appliance", rooms=("kitchen",), detail="a coffee machine"),
    "standmixer": dict(kind="appliance", rooms=("kitchen",), detail="a stand mixer"),
    "blender": dict(kind="appliance", rooms=("kitchen",), detail="a blender"),
    "cabinet": dict(kind="furniture", rooms=("kitchen",), detail="a cabinet with hinged doors and drawers"),
    "drawer": dict(kind="furniture", rooms=("kitchen",), detail="a drawer unit"),
    "door": dict(kind="furniture", rooms=("interior",), detail="a door with a hinged panel"),
    "wardrobe": dict(kind="furniture", rooms=("bedroom",), detail="a wardrobe"),
    "furniture": dict(kind="furniture", rooms=("living_room",), detail="furniture with moving parts"),
    # Locomotion's KitchenRoom_RSS is the one articulated Locomotion category;
    # without this entry its annotations would fall back to the generic
    # "the kitchen room rss asset" template.
    "kitchenroom": dict(kind="room", rooms=("kitchen",), detail="a kitchen room interior scene with appliances"),
}


def name_kind_of(asset_name: str) -> str:
    """The template key of a bundle asset name (``Microwave043`` → ``microwave``)."""
    token = "".join(
        (" " + char.lower()) if char.isupper() and index and not asset_name[index - 1].isupper()
        else char.lower()
        for index, char in enumerate(asset_name.rstrip("0123456789"))
    )
    compact = token.replace(" ", "").replace("_", "")
    best = None
    for template_name in NAME_TEMPLATES:
        if template_name.replace("_", "") in compact:
            if best is None or len(template_name) > len(best):
                best = template_name
    return best if best is not None else token.strip().replace(" ", "_")


def list_lightwheel_assets(source: Path) -> List[Tuple[str, Path]]:
    """List ``(packaged_object_id, asset_dir)`` pairs of the bundle's Manipulation assets.

    ``Manipulation/<Asset>/<Asset>.usd`` + ``texture/``; the packaged id is
    ``lightwheel_<Asset>`` (the attribution naming format the license
    requires).
    """
    manipulation = source / UPSTREAM_MANIPULATION_DIR
    if not manipulation.is_dir():
        raise SystemExit(f"No '{UPSTREAM_MANIPULATION_DIR}/' directory in --source: {source}")
    pairs = []
    for asset_dir in sorted(p for p in manipulation.iterdir() if p.is_dir()):
        pairs.append((f"lightwheel_{asset_dir.name}", asset_dir))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Repackaging (MDL → UsdPreviewSurface)
# ---------------------------------------------------------------------------

def retarget_joint_bodies(usd_file: Path, asset_name: Optional[str] = None) -> Tuple[int, int]:
    """Re-anchor every joint ``body0``/``body1`` target onto a rigid body.

    Some bundle assets author joints against a **bare ``Visuals`` mesh prim**
    instead of the ``RigidBodyAPI`` body (CoffeeMachine103, RangeHood010,
    ToasterOven058): Genesis resolves joint bodies by RigidBodyAPI prim path,
    so the mesh-targeted side never maps onto a link — the joint stops
    connecting and the stage loads as several entities (mixed entity / entity
    splits) or with dangling unreferenced bodies.

    Two repairs, both applied in-place on the (single-layer) packaged stage:

    * a target prim that **exists** but is not a rigid body is retargeted to
      the nearest *enclosing ancestor* carrying ``RigidBodyAPI`` (its authored
      transform is the identity in this bundle, so the joint frames are
      preserved verbatim);
    * a target path that **does not exist at all** (ToasterOven056's
      ``/root/ToasterOven056_window``) leaves a stale ``PhysicsFixedJoint``
      behind that Genesis still resolves as a parent link — such a stale
      **fixed** joint is *deactivated* (``SetActive(False)``).  A non-fixed
      joint with a missing target is left alone: it is a genuine stage defect,
      not a recoverable one.

    Additionally — Stove070 — a ``PhysicsFixedJoint`` with an **empty**
    ``body0`` (a weld-to-world) whose ``body1`` is a *non-root* body is
    retargeted onto the asset's root body: ``/root/Stove070_Body002/FixedJoint``
    welds Body002 to the world instead of to the machine, splitting it off
    from the rest of the articulation.  The asset root body is the unique
    RigidBodyAPI prim named after the asset (``/root/Stove070``); the root
    FixedJoint's own ``body1`` *is* that body, so it is left untouched.

    Args:
        usd_file: The packaged (single-layer) stage to repair in place.
        asset_name: The upstream asset name (``CoffeeMachine103``) used to
            identify the asset root body for the empty-body0 repair.

    Returns:
        ``(n_retargeted, n_deactivated)``.
    """
    from pxr import Sdf, Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd_file))
    rigid_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    asset_root_path: Optional[str] = None
    if asset_name:
        named_roots = [
            path for path in rigid_paths if Sdf.Path(path).name == asset_name
        ]
        if len(named_roots) == 1:
            asset_root_path = named_roots[0]

    def _nearest_rigid_ancestor(path: Sdf.Path) -> Optional[str]:
        walked = path
        while str(walked) != "/":
            if str(walked) in rigid_paths:
                return str(walked)
            walked = walked.GetParentPath()
        return None

    retargeted = 0
    deactivated = 0
    for prim in list(stage.Traverse()):  # bind prims before editing
        if not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        is_fixed = prim.IsA(UsdPhysics.FixedJoint)
        body0_rel, body1_rel = joint.GetBody0Rel(), joint.GetBody1Rel()
        targets0 = list(body0_rel.GetTargets()) if body0_rel is not None else []
        targets1 = list(body1_rel.GetTargets()) if body1_rel is not None else []

        if (
            is_fixed and not targets0 and len(targets1) == 1
            and asset_root_path is not None
            and str(targets1[0]) in rigid_paths
            and str(targets1[0]) != asset_root_path
        ):
            body0_rel.SetTargets([Sdf.Path(asset_root_path)])
            retargeted += 1
            continue

        for rel, targets in ((body0_rel, targets0), (body1_rel, targets1)):
            if rel is None or len(targets) != 1:
                continue  # empty (world weld) or multi-target: leave alone
            target = targets[0]
            if str(target) in rigid_paths:
                continue  # already a rigid body
            if not stage.GetPrimAtPath(target).IsValid():
                # A stale authored path — only fatal for a fixed joint (it
                # still resolves as a parent link in Genesis); deactivate it.
                if is_fixed:
                    prim.SetActive(False)
                    deactivated += 1
                continue
            ancestor = _nearest_rigid_ancestor(target)
            if ancestor is None:
                continue  # no rigid body above it: the loader's problem
            rel.SetTargets([Sdf.Path(ancestor)])
            retargeted += 1

    if retargeted or deactivated:
        stage.GetRootLayer().Save()
    return retargeted, deactivated


def repackage_object(asset_dir: Path, object_id: str, out_root: Path,
                     convert: bool = True) -> Tuple[Path, int]:
    """Repackage one upstream asset dir into ``out_root/<object_id>/``.

    Copies the asset tree verbatim (the ``<Asset>.usd`` stage + its
    ``texture/`` sidecar), renames the root layer to
    ``<object_id>.usd`` (relative texture references are directory-based)
    and applies the **MDL → UsdPreviewSurface** rewrite plus the joint
    body retargeting (:func:`retarget_joint_bodies`).  Verifies that every
    file the pxr layer stack + texture references name exists.

    Returns:
        ``(packaged_dir, n_converted_materials)``.
    """
    from unienv_genesis_collection.utils.usd_conversions import convert_mdl_to_preview_surface

    stage_files = sorted(p for p in asset_dir.glob("*.usd") if p.is_file())
    if len(stage_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .usd root layer in {asset_dir}, found "
            f"{[p.name for p in stage_files]}."
        )
    stage_file = stage_files[0]

    obj_out_dir = out_root / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(asset_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(asset_dir)
        (obj_out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, obj_out_dir / rel)
        copied += 1
    packaged_stage = obj_out_dir / f"{object_id}.usd"
    if stage_file.name != packaged_stage.name:
        (obj_out_dir / stage_file.name).rename(packaged_stage)

    # Joint body retargeting BEFORE the MDL rewrite: joints whose body0/body1
    # target a bare Visuals mesh prim (or a stale/nonexistent path) otherwise
    # leave Genesis with unreferenced bodies — entity splits / mixed entity /
    # dangling links.  The MDL conversion below re-exports the (repaired)
    # layer verbatim, so the retargeted opinions survive it.
    retarget_joint_bodies(packaged_stage, asset_name=stage_file.stem)

    # `drop_missing_normal_maps`: Stove070's 11 dangling `…_N001.png` normal
    # inputs (the texture is genuinely absent upstream) must drop the slot
    # instead of failing the required-file check below and dropping the asset.
    n_converted = convert_mdl_to_preview_surface(
        packaged_stage, drop_missing_normal_maps=True,
    ) if convert else 0
    _verify_object_assets(packaged_stage, obj_out_dir)
    return obj_out_dir, n_converted



def _transcode_exr_textures(obj_out_dir: Path) -> List[str]:
    """Re-encode every ``.exr`` texture of the packaged tree to a sibling ``.png``.

    Genesis reads every ``UsdUVTexture.file`` through PIL
    (``genesis/utils/usd/usd_material.py``), and PIL cannot identify an
    OpenEXR buffer (no FreeImage plugin) — SnowGround's road materials are
    authored with ``.EXR`` snow-road maps, so the stage would load with every
    material failing.  The EXRs are re-encoded 8-bit into a sibling ``.png``
    (the MDL → UsdPreviewSurface rewrite then picks the PNG for every
    converter-authored texture input, see
    :func:`unienv_genesis_collection.utils.usd_conversions._relocate_texture`)
    while the ``.exr`` files stay — they are part of the reference closure the
    descriptor verifies (the payload layers keep their own EXR opinions).

    Returns:
        The packaged-relative PNG paths (empty when the tree has no EXR).
    """
    exr_files = sorted(
        p for p in obj_out_dir.rglob("*")
        if p.is_file() and p.suffix.lower() == ".exr"
    )
    if not exr_files:
        return []

    import imageio.v2 as imageio

    converted: List[Path] = []
    for exr_file in exr_files:
        png_file = exr_file.with_suffix(".png")
        array = imageio.imread(str(exr_file))
        array = np.asarray(array, dtype=np.float64)[..., :3]
        if not np.isfinite(array).all():
            array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        imageio.imwrite(str(png_file), np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
        converted.append(png_file)
    return [p.relative_to(obj_out_dir).as_posix() for p in converted]


# ---------------------------------------------------------------------------
# Locomotion discovery (closure-copy repackaging)
# ---------------------------------------------------------------------------

def list_locomotion_assets(source: Path) -> List[Tuple[str, Path, str]]:
    """List ``(object_id, asset_dir, root_layer)`` for the Locomotion subset.

    The root layer comes from the explicit :data:`LOCOMOTION_ROOTS` map (see
    the constant's note on the measured reference closures); categories
    present upstream but absent from the map are reported and skipped, and
    categories absent from the source are skipped with a note (partial
    extracts stay usable).
    """
    locomotion = source / UPSTREAM_LOCOMOTION_DIR
    if not locomotion.is_dir():
        raise SystemExit(
            f"No '{UPSTREAM_LOCOMOTION_DIR}/' directory in --source: {source}"
        )
    pairs: List[Tuple[str, Path, str]] = []
    for asset_dir in sorted(p for p in locomotion.iterdir() if p.is_dir()):
        category = asset_dir.name
        if category not in LOCOMOTION_ROOTS:
            print(
                f"NOTE: Locomotion category {category!r} has no entry in "
                "LOCOMOTION_ROOTS (its root layer was never verified) — "
                "skipping."
            )
            continue
        pairs.append(
            (locomotion_object_id(category), asset_dir, LOCOMOTION_ROOTS[category])
        )
    missing = sorted(set(LOCOMOTION_ROOTS) - {p[1].name for p in pairs})
    if missing:
        print(
            f"NOTE: Locomotion category/ies not in the source snapshot "
            f"({source}): {missing}; skipping."
        )
    return sorted(pairs)


def repackage_scene(
    asset_dir: Path,
    object_id: str,
    root_layer: str,
    out_root: Path,
    convert: bool = True,
) -> Tuple[Path, int, List[str]]:
    """Closure-copy a whole scene stage into ``out_root/<object_id>/``.

    Unlike :func:`repackage_object` (verbatim copy of a per-object tree),
    this copies **exactly the reference closure** — the root layer plus
    ``_usd_required_files(root, asset_dir)`` — each file to the same relative
    path.  Everything else (``E/``, ``.thumbs``, ``.DS_Store``,
    ``.collect.mapping.json``, ``Table049/``, ``ClothDemo_002/``, the
    ``omniverse-content-production.s3…`` mirror) is excluded **by
    construction**, while a referenced payload dir (e.g. the 68-437 MB
    ``草坪草地3D模型`` of the ground terrains) is kept exactly when the
    stage needs it.

    The root layer is renamed to ``<object_id>.usd`` (its extra top-level
    payload layers are referenced *by* it, nothing references it, so the
    rename is safe) and the MDL → UsdPreviewSurface rewrite is applied.
    Missing references are fatal for USD layers and textures; the UE
    ``Foo.Foo`` material-record pattern is warn-and-record (Apartment ships
    20 of them) — see :func:`_verify_object_assets` — and a missing texture
    whose expected directory holds the same file under a prefix-stripped
    name (an upstream path-authoring bug) is repaired by copying it under
    the referenced name (:func:`_recoverable_texture`).

    Returns:
        ``(packaged_dir, n_converted_materials, unresolvable_refs)``.
    """
    from unienv_genesis_collection.objects.hub_objects.articulated.hub_articulated_object import (
        _usd_required_files,
    )
    from unienv_genesis_collection.utils.usd_conversions import convert_mdl_to_preview_surface

    stage_file = asset_dir / root_layer
    if not stage_file.is_file():
        raise FileNotFoundError(
            f"Locomotion root layer {root_layer!r} is not in {asset_dir}."
        )

    # The closure is computed from the *upstream* (unresolved) paths and
    # re-anchored onto asset_dir: HF-cache snapshots are symlink farms whose
    # ``.resolve()`` lands in ``blobs/``.
    closure = [stage_file, *_usd_required_files(stage_file, asset_dir)]
    obj_out_dir = out_root / object_id
    obj_out_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    unresolvable: List[str] = []
    recovered: List[str] = []
    for path in closure:
        rel = _closure_relative_path(path, asset_dir)
        if rel is None:
            raise FileNotFoundError(
                f"{stage_file.name} references {path}, which is outside the "
                f"asset directory {asset_dir}."
            )
        if not path.is_file():
            if _is_fatal_missing_ref(path):
                # Upstream path-authoring bug repair: the bundle ships crates
                # whose texture paths carry a stale exporter prefix while the
                # texture itself sits in the expected directory under the
                # stripped name (KitchenRoom's
                # ``./textures/3d66Model-19913701-files-024.JPG`` vs the
                # shipped ``textures/19913701-files-024.JPG`` — same stem and
                # tile id, the ``3d66.com`` exporter's prefix dropped when the
                # bundle was published).  A same-directory file whose
                # (case-folded) name is a *suffix* of the referenced one is
                # unambiguously that texture: copy it under the referenced
                # name, and record the repair.  Nothing else stays fatal.
                stand_in = _recoverable_texture(path, asset_dir)
                if stand_in is not None:
                    target = obj_out_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stand_in, target)
                    recovered.append(rel.as_posix())
                    copied += 1
                    print(
                        f"  [repackage] {object_id}: repaired missing texture "
                        f"{rel.as_posix()!r} from the shipped {stand_in.name!r}"
                    )
                    continue
                raise FileNotFoundError(
                    f"{stage_file.name} references the missing "
                    f"{'texture' if path.suffix.lower() in _IMAGE_SUFFIXES else 'USD layer'} "
                    f"{path} (checked {asset_dir})."
                )
            unresolvable.append(rel.as_posix())
            continue
        (obj_out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, obj_out_dir / rel)
        copied += 1

    packaged_stage = obj_out_dir / f"{object_id}.usd"
    if stage_file.name != packaged_stage.name:
        (obj_out_dir / stage_file.name).rename(packaged_stage)

    _transcode_exr_textures(obj_out_dir)
    n_converted = convert_mdl_to_preview_surface(packaged_stage) if convert else 0
    # Verify the *packaged* tree (what the loader will actually open).
    extra = _verify_object_assets(
        packaged_stage, obj_out_dir, allow_unresolvable_refs=True
    )
    # Report every tolerated reference relative to the packaged directory (the
    # closure walk and the verify pass see them in different anchors).
    extra_rel = []
    for path in extra:
        try:
            extra_rel.append(path.relative_to(obj_out_dir).as_posix())
        except ValueError:
            extra_rel.append(path.as_posix())
    unresolvable = sorted(set(unresolvable) | set(extra_rel))
    if unresolvable:
        print(
            f"  [repackage] {object_id}: {len(unresolvable)} unresolvable "
            f"non-USD/non-image reference(s) recorded (UE material records): "
            f"{', '.join(sorted(unresolvable)[:3])}"
            + (" ..." if len(unresolvable) > 3 else "")
        )
    if recovered:
        print(
            f"  [repackage] {object_id}: {len(recovered)} prefixed-texture "
            f"reference(s) repaired: {', '.join(sorted(recovered))}"
        )
    return obj_out_dir, n_converted, unresolvable


def _closure_relative_path(path: Path, asset_dir: Path) -> Optional[Path]:
    """Path of a closure file relative to *asset_dir* (symlink-tolerant)."""
    for base in (asset_dir, Path(os.path.realpath(asset_dir))):
        try:
            return path.relative_to(base)
        except ValueError:
            continue
    real = Path(os.path.realpath(path))
    for base in (asset_dir, Path(os.path.realpath(asset_dir))):
        try:
            return real.relative_to(base)
        except ValueError:
            continue
    return None


# ---------------------------------------------------------------------------
# Locomotion catalog entries (plan §3d)
# ---------------------------------------------------------------------------

LOCOMOTION_TEMPLATES: Dict[str, Dict[str, object]] = {
    "2-StoryStaircase": dict(
        annotations=("environment", "staircase", "a two-storey staircase environment with a loft"),
        tags=("staircase", "interior"),
    ),
    "Apartment": dict(
        annotations=("environment", "room", "an apartment interior scene"),
        tags=("room", "interior", "apartment"),
    ),
    "Grass": dict(
        annotations=("environment", "terrain", "a grass terrain for locomotion"),
        tags=("terrain", "grass"),
    ),
    "GravelGround": dict(
        annotations=("environment", "terrain", "a gravel ground terrain for locomotion"),
        tags=("terrain", "gravel"),
    ),
    "KitchenRoom": dict(
        annotations=("environment", "room", "a kitchen room interior scene"),
        tags=("room", "interior", "kitchen"),
    ),
    "MudGround": dict(
        annotations=("environment", "terrain", "a mud ground terrain for locomotion"),
        tags=("terrain", "mud"),
    ),
    "SLATEGround": dict(
        annotations=("environment", "terrain", "a slate stone ground terrain for locomotion"),
        tags=("terrain", "slate"),
    ),
    "SnowGround": dict(
        annotations=("environment", "terrain", "a snow ground terrain for locomotion"),
        tags=("terrain", "snow"),
    ),
}


def annotate_locomotion_object(
    category: str,
    family: str,
    n_revolute: int = 0,
    n_prismatic: int = 0,
) -> dict:
    """One catalog entry for a Locomotion category (plan §3d).

    Tags: ``environment`` + ``locomotion`` + the category's own tokens plus
    ``static`` for the jointless stages (and ``articulated`` for KitchenRoom,
    which ships through the articulated tree).  The joint counts are appended
    to the detailed annotation for the articulated one.
    """
    template = LOCOMOTION_TEMPLATES.get(category)
    if template is None:
        raise KeyError(f"No Locomotion catalog template for {category!r}.")
    annotations = list(template["annotations"])  # type: ignore[arg-type]
    tags = {"environment", "locomotion", *tuple(template["tags"])}  # type: ignore[arg-type]
    if family == "articulated":
        tags.add("articulated")
        # The articulated one ships through the furniture tree, so it also
        # carries the asset-name template's kind/room/key tags (the
        # ``kitchenroom`` entry keeps KitchenRoom_RSS out of the generic
        # fallback template).
        _key, name_entry = annotate_object(category, n_revolute, n_prismatic)
        tags |= set(name_entry["tags"])
        specifics = []
        if n_revolute:
            specifics.append(f"{n_revolute} hinged part{'' if n_revolute == 1 else 's'}")
        if n_prismatic:
            specifics.append(f"{n_prismatic} sliding part{'' if n_prismatic == 1 else 's'}")
        if specifics:
            annotations.append(f"{annotations[-1]} with {', '.join(specifics)}")
    else:
        tags.add("static")
    return {
        "language_annotations": list(dict.fromkeys(annotations)),
        "tags": sorted(tags),
    }


# ---------------------------------------------------------------------------
# Generated catalog
# ---------------------------------------------------------------------------

def write_catalog(
    out_root: Path,
    records: Sequence[dict],
    merge: bool = True,
    drop: Optional[Sequence[str]] = None,
) -> int:
    """Write ``catalog.json`` from the gate records (+ ``CATALOG_OVERRIDES``).

    Thin family wrapper over :func:`_packaging.catalog.write_catalog` (the
    single implementation): entries ``{"language_annotations": [...],
    "tags": [...]}`` (the shared ``load_catalog`` schema); incremental runs
    keep the entries of objects that are not part of this run; a re-run of
    the same object rewrites its entry.
    """
    return _write_catalog_impl(
        out_root, records, merge=merge, drop=drop, overrides=CATALOG_OVERRIDES,
    )

CATALOG_OVERRIDES: Dict[str, dict] = {}


def annotate_object(asset_name: str, n_revolute: int, n_prismatic: int) -> Tuple[str, dict]:
    """One catalog entry from the asset-name template + the gate's joint counts."""
    key = name_kind_of(asset_name)
    template = NAME_TEMPLATES.get(key, dict(kind="object", rooms=(), detail=None))
    kind = str(template["kind"])
    words = key.replace("_", " ")
    specifics = []
    if n_revolute:
        specifics.append(f"{n_revolute} hinged part{'' if n_revolute == 1 else 's'}")
    if n_prismatic:
        specifics.append(f"{n_prismatic} sliding part{'' if n_prismatic == 1 else 's'}")
    annotations = [
        kind,
        f"the {words} asset",
        (
            f"the {words} asset with {', '.join(specifics)}"
            if specifics
            else f"a plain {words} asset variant"
        ),
    ]
    entry_tags = {kind.replace(" ", "_"), key}
    entry_tags.update(tuple(template["rooms"]))  # type: ignore[arg-type]
    entry = {
        "language_annotations": list(dict.fromkeys(annotations)),
        "tags": sorted(entry_tags),
    }
    return kind, entry


# ---------------------------------------------------------------------------
# Genesis sanity gate (GPU) — mirrors the ArtVIP pipeline's gate
# ---------------------------------------------------------------------------


def sanity_check_static_env(
    item,
    *,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> dict:
    """Genesis sanity-load one packaged **static environment** (plan §4).

    The static family's contract, replacing the articulated gate's
    moving-joint + opened-configuration checks:

    1. the stage must have **no joints** — this is also what routes a
       misfiled articulated asset (e.g. ``KitchenRoom_RSS``) back to the
       articulated pipeline instead of silently shipping it as static;
    2. the stage must carry collision (``UsdPhysics.CollisionAPI`` count > 0)
       — a collision-less terrain/room is a packaging bug; the count is
       recorded as ``n_colliders``;
    3. the offline bbox (the descriptor's point-accurate walk over the
       geometry ``add_stage`` instantiates — ``LightwheelEnv.get_local_bounding_box``)
       must be finite, sane (extent > 0.1 m in X/Y, z-min ≤ 0 for
       terrain-tagged assets — they are walked on) and consistent with the
       loaded entity/entities' ``get_vAABB()`` within the existing tolerances.

    A thumbnail render is best-effort (never drops).  Returns the gate
    record; raises on failure.
    """
    import torch as _torch
    import genesis as gs
    from pxr import Usd, UsdGeom, UsdPhysics

    # ---- pxr-side contract: no joints, some collision ----------------------
    usd_stage = Usd.Stage.Open(str(item.description_file))
    joints = [
        str(prim.GetPath())
        for prim in usd_stage.Traverse()
        if prim.IsA(UsdPhysics.Joint)
    ]
    if joints:
        raise AssertionError(
            f"{len(joints)} joint(s) found on a *static* environment stage "
            f"({joints[:4]}) — articulated assets must go through the "
            "articulated pipeline (lightwheel_KitchenRoom_RSS), not the "
            "static environment family."
        )
    n_colliders = sum(
        1
        for prim in usd_stage.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
        or prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    )
    if n_colliders == 0:
        raise AssertionError(
            "no CollisionAPI prims on the stage — a terrain/room without "
            "collision is a packaging bug."
        )

    # ---- offline bbox sanity ----------------------------------------------
    local = item.get_local_bounding_box()
    low = local.low.detach().cpu().numpy().astype(float)
    high = local.high.detach().cpu().numpy().astype(float)
    if not (np.isfinite(low).all() and np.isfinite(high).all()):
        raise AssertionError(f"offline bbox is not finite: {low} .. {high}")
    extent = np.maximum(high - low, 1e-9)
    if extent[0] <= 0.1 or extent[1] <= 0.1:
        raise AssertionError(
            f"offline bbox extent {np.round(extent, 4).tolist()} is degenerate "
            "in X/Y (mpu/scale disaster?)"
        )
    if "terrain" in item.info.tags and low[2] > 1e-3:
        raise AssertionError(
            f"terrain-tagged asset has z-min {low[2]:.4f} > 0 — terrain is "
            "walked on and must sit at or below the ground plane."
        )

    # ---- load + build ------------------------------------------------------
    scene = gs.Scene(show_viewer=False, renderer=gs.options.renderers.Rasterizer())
    try:
        entities = item.load_into_scene(_WorldStub(scene), pos=(0.0, 0.0, 0.0))
        scene.build(n_envs=0)
        lows: List[np.ndarray] = []
        highs: List[np.ndarray] = []
        for entity in entities:
            vaabb = entity.get_vAABB()
            lows.append(np.asarray(
                vaabb[0].detach().cpu() if _torch.is_tensor(vaabb[0]) else vaabb[0],
                dtype=float,
            ).reshape(3))
            highs.append(np.asarray(
                vaabb[1].detach().cpu() if _torch.is_tensor(vaabb[1]) else vaabb[1],
                dtype=float,
            ).reshape(3))
        low_v = np.min(lows, axis=0)
        high_v = np.max(highs, axis=0)
    finally:
        try:
            scene.destroy()
        except Exception:  # noqa: BLE001 — teardown is best-effort
            pass

    tolerance = np.maximum(bbox_tol_abs, bbox_tol_rel * extent)
    deltas = np.maximum(np.abs(low_v - low), np.abs(high_v - high))
    bbox_record = dict(
        offline_low=low.tolist(), offline_high=high.tolist(),
        vAABB_low=low_v.tolist(), vAABB_high=high_v.tolist(),
        max_delta=float(deltas.max()), tolerance_per_axis=tolerance.tolist(),
    )
    if bool((deltas > tolerance).any()):
        raise AssertionError(
            "offline bbox deviates from the loaded entity vAABB beyond "
            f"tolerance (per-axis deltas {np.round(deltas, 4).tolist()}, "
            f"tolerance {np.round(tolerance, 4).tolist()})"
        )

    rendered: Optional[str] = None
    if render_dir is not None:
        try:
            rendered = _render_thumbnail(item, item.name, render_dir)
        except Exception as exc:  # noqa: BLE001 — keep the passing environment
            print(
                f"  [gate] {item.name}: render FAILED (asset kept) — "
                f"{type(exc).__name__}: {exc}"
            )

    return dict(
        status="passed",
        family="static",
        n_colliders=n_colliders,
        n_entities=len(entities),
        bbox=bbox_record,
        render=rendered,
    )


def gate_family_of(object_id: str) -> str:
    """The gate/loader family of a packaged Lightwheel id (plan §4).

    ``lightwheel_env_*`` are the static environment stages; everything else
    (the ``Manipulation`` tree and ``lightwheel_KitchenRoom_RSS``) is
    articulated.
    """
    return "static" if object_id.startswith("lightwheel_env_") else "articulated"


def run_sanity_gate(
    output_dir: Path,
    object_ids: Sequence[str],
    render_dir: Optional[Path] = None,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    family: Optional[str] = None,
) -> Tuple[List[str], List[dict], List[dict]]:
    """Sanity-load every packaged object; drop (warn on) the failing ones.

    Dispatches per object id (``gate_family_of``): the static environment
    family goes through :func:`sanity_check_static_env` +
    ``LightwheelEnv``, the articulated family through
    :func:`sanity_check_object` + ``LightwheelFurniture``.  *family*
    overrides the inference (``"static"`` / ``"articulated"``).  The
    dispatch lives in the descriptor factory — the only per-family part of
    the shared ``_packaging.sanity`` gate loop.
    """

    def _gate_descriptor(object_id: str):
        family_of = family or gate_family_of(object_id)
        if family_of == "static":
            from unienv_genesis_collection.objects.hub_objects.lightwheel_env import (
                LightwheelEnv,
            )

            return LightwheelEnv(object_id), sanity_check_static_env
        from unienv_genesis_collection.objects.hub_objects.articulated.lightwheel_furniture import (
            LightwheelFurniture,
        )

        return LightwheelFurniture(object_id), sanity_check_object

    return _run_sanity_gate_impl(
        output_dir, object_ids,
        descriptor_factory=_gate_descriptor,
        check=sanity_check_object,
        render_dir=render_dir,
        bbox_tol_abs=bbox_tol_abs,
        bbox_tol_rel=bbox_tol_rel,
    )

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def prepare_manipulation(
    source: Path,
    output_dir: Path,
    objects: Optional[Sequence[str]] = None,
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    """Package the selected ``Manipulation/`` assets into ``objects/lightwheel``."""
    out_root = output_dir / OUTPUT_ASSETS_SUBDIR
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = list_lightwheel_assets(source)
    if objects:
        wanted = {o.strip().lower() for o in objects if o.strip()}
        selected = [
            (object_id, asset_dir)
            for object_id, asset_dir in pairs
            if (object_id.lower() in wanted
                or asset_dir.name.lower() in wanted
                or any(w in object_id.lower() or w in asset_dir.name.lower() for w in wanted))
        ]
        if not selected:
            raise SystemExit(
                f"--objects matched none of the {len(pairs)} bundle assets."
            )
    else:
        selected = pairs
    if not selected:
        raise SystemExit(f"No Lightwheel Manipulation assets found in {source}.")

    asset_names: Dict[str, str] = {}
    packaged: List[str] = []
    converted_counts: Dict[str, int] = {}
    print(f"Repackaging {len(selected)} asset(s) ...")
    for object_id, asset_dir in selected:
        try:
            _dir, n_converted = repackage_object(asset_dir, object_id, out_root)
        except Exception as exc:  # noqa: BLE001 — repackage failure ⇒ drop
            print(f"  [repackage] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
            shutil.rmtree(out_root / object_id, ignore_errors=True)
            continue
        asset_names[object_id] = asset_dir.name
        packaged.append(object_id)
        converted_counts[object_id] = n_converted
        print(f"  -> {out_root / object_id} (converted {n_converted} MDL material(s))")

    prelim_entries = [
        dict(object_id=object_id, catalog_entry=annotate_object(
            asset_names[object_id], 0, 0)[1])
        for object_id in packaged
    ]
    write_catalog(out_root, prelim_entries)

    if gate:
        print(f"\nGenesis sanity gate on {len(packaged)} asset(s) ...")
        kept, passed_records, dropped_records = _run_sanity_gate_batched(
            __file__, run_sanity_gate, output_dir, packaged,
            render_dir=render_dir,
            bbox_tol_abs=bbox_tol_abs,
            bbox_tol_rel=bbox_tol_rel,
        )
        gate_entries: List[dict] = []
        for record in passed_records:
            record["n_files"] = len(list((out_root / record["object_id"]).rglob("*")))
            record["n_mdl_converted"] = converted_counts[record["object_id"]]
            entry = annotate_object(
                asset_names[record["object_id"]],
                sum(1 for j in record["joints"] if j["type"] == "REVOLUTE"),
                sum(1 for j in record["joints"] if j["type"] == "PRISMATIC"),
            )[1]
            entry["tags"] = sorted(set(entry["tags"]) | {"articulated"})
            gate_entries.append(
                dict(object_id=record["object_id"], catalog_entry=entry)
            )
        write_catalog(
            out_root, gate_entries,
            drop=[record["object_id"] for record in dropped_records],
        )
    else:
        kept, passed_records, dropped_records = list(packaged), [], []

    (out_root / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8", newline="\n")
    (out_root / "README.md").write_text(PROVENANCE_README)
    (out_root / "packaging_log.json").write_text(json.dumps(dict(
        source=str(source),
        packaged=packaged,
        passed=passed_records,
        dropped=dropped_records,
    ), indent=1, ensure_ascii=False) + "\n")

    print(f"\nPrepared {len(kept)} asset(s) in {out_root}")
    if dropped_records:
        print("Dropped asset(s) failing the sanity gate / repackage:")
        for record in dropped_records:
            print(f"  - {record['object_id']}: {record.get('error')}")
    return out_root, kept


def prepare_locomotion(
    source: Path,
    output_dir: Path,
    objects: Optional[Sequence[str]] = None,
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[str]]:
    """Package the ``Locomotion/`` subset (plan §3/§4).

    Per-category routing: ``KitchenRoom`` (genuinely articulated) goes through
    the *articulated* pipeline into ``objects/lightwheel`` as
    ``lightwheel_KitchenRoom_RSS``; the other seven jointless categories go
    into ``objects/lightwheel_env`` as ``lightwheel_env_<Category>``.  Both
    trees get their catalog, LICENSE, provenance README and
    ``packaging_log.json``; each category's reference closure is copied
    (:func:`repackage_scene`).

    Returns ``([out_root, ...], kept_object_ids)``.
    """
    pairs = list_locomotion_assets(source)
    if objects:
        wanted = {o.strip().lower() for o in objects if o.strip()}
        selected = [
            (object_id, asset_dir, root_layer)
            for object_id, asset_dir, root_layer in pairs
            if (
                object_id.lower() in wanted
                or asset_dir.name.lower() in wanted
                or any(
                    w in object_id.lower() or w in asset_dir.name.lower()
                    for w in wanted
                )
            )
        ]
        if not selected:
            raise SystemExit(
                f"--objects matched none of the {len(pairs)} Locomotion categories."
            )
    else:
        selected = pairs
    if not selected:
        raise SystemExit(f"No Lightwheel Locomotion categories found in {source}.")

    # Per-family bookkeeping (the two trees have separate catalogs).
    packaged: Dict[str, List[str]] = {"articulated": [], "static": []}
    converted_counts: Dict[str, int] = {}
    unresolvable: Dict[str, List[str]] = {}
    categories: Dict[str, str] = {}
    out_roots: Dict[str, Path] = {
        "articulated": output_dir / OUTPUT_ASSETS_SUBDIR,
        "static": output_dir / OUTPUT_ENV_ASSETS_SUBDIR,
    }
    for family, root in out_roots.items():
        root.mkdir(parents=True, exist_ok=True)

    print(f"Repackaging {len(selected)} Locomotion category/ies ...")
    for object_id, asset_dir, root_layer in selected:
        family = locomotion_family(asset_dir.name)
        out_root = out_roots[family]
        try:
            _dir, n_converted, missing_refs = repackage_scene(
                asset_dir, object_id, root_layer, out_root,
            )
        except Exception as exc:  # noqa: BLE001 — repackage failure ⇒ drop
            print(f"  [repackage] {object_id}: DROPPED — {type(exc).__name__}: {exc}")
            shutil.rmtree(out_root / object_id, ignore_errors=True)
            continue
        categories[object_id] = asset_dir.name
        packaged[family].append(object_id)
        converted_counts[object_id] = n_converted
        unresolvable[object_id] = missing_refs
        print(
            f"  -> {out_root / object_id} "
            f"(converted {n_converted} MDL material(s))"
        )

    # Preliminary catalogs (template annotations) so the gate's descriptors
    # resolve offline; rewritten with the gate survivors below.
    for family, root in out_roots.items():
        if not packaged[family]:
            continue
        write_catalog(root, [
            dict(object_id=object_id, catalog_entry=annotate_locomotion_object(
                categories[object_id], family,
            ))
            for object_id in packaged[family]
        ])

    passed_records: List[dict] = []
    dropped_records: List[dict] = []
    if gate:
        for family, root in out_roots.items():
            ids = packaged[family]
            if not ids:
                continue
            print(f"\nGenesis sanity gate ({family}) on {len(ids)} asset(s) ...")
            _kept, passed_f, dropped_f = _run_sanity_gate_batched(
                __file__, run_sanity_gate, output_dir, ids,
                render_dir=render_dir,
                bbox_tol_abs=bbox_tol_abs,
                bbox_tol_rel=bbox_tol_rel,
            )
            passed_records.extend(passed_f)
            dropped_records.extend(dropped_f)
            gate_entries: List[dict] = []
            for record in passed_f:
                record["n_files"] = len(list((root / record["object_id"]).rglob("*")))
                record["n_mdl_converted"] = converted_counts[record["object_id"]]
                entry = annotate_locomotion_object(
                    categories[record["object_id"]],
                    family,
                    sum(1 for j in record.get("joints", []) if j["type"] == "REVOLUTE"),
                    sum(1 for j in record.get("joints", []) if j["type"] == "PRISMATIC"),
                )
                gate_entries.append(
                    dict(object_id=record["object_id"], catalog_entry=entry)
                )
            write_catalog(
                root, gate_entries,
                drop=[record["object_id"] for record in dropped_f],
            )
    else:
        passed_records = []
        dropped_records = []

    kept: List[str] = []
    for family, root in out_roots.items():
        ids = packaged[family]
        if not ids:
            continue
        dropped_ids = {r["object_id"] for r in dropped_records}
        kept_f = [object_id for object_id in ids if object_id not in dropped_ids]
        kept.extend(kept_f)
        (root / "LICENSE").write_text(LICENSE_TEXT, encoding="utf-8", newline="\n")
        (root / "README.md").write_text(
            ENV_PROVENANCE_README if family == "static" else PROVENANCE_README
        )
        (root / "packaging_log.json").write_text(json.dumps(dict(
            source=str(source),
            subset="locomotion",
            family=family,
            packaged=ids,
            passed=[r for r in passed_records if r["object_id"] in set(ids)],
            dropped=[r for r in dropped_records if r["object_id"] in set(ids)],
            # Apartment's UE-export material records (non-USD/non-image refs)
            # that exist nowhere in the bundle — warn-and-record, not fatal.
            unresolvable_refs={
                object_id: refs
                for object_id, refs in unresolvable.items()
                if refs and object_id in set(ids)
            },
        ), indent=1, ensure_ascii=False) + "\n")
        print(f"\nPrepared {len(kept_f)} asset(s) in {root}")
    if dropped_records:
        print("Dropped asset(s) failing the sanity gate / repackage:")
        for record in dropped_records:
            print(f"  - {record['object_id']}: {record.get('error')}")
    return list(out_roots.values()), kept


def prepare(
    source: Path,
    output_dir: Path,
    objects: Optional[Sequence[str]] = None,
    gate: bool = True,
    bbox_tol_abs: float = BBOX_TOL_ABS,
    bbox_tol_rel: float = BBOX_TOL_REL,
    render_dir: Optional[Path] = None,
    subset: str = "manipulation",
) -> Tuple[Path, List[str]]:
    """Package a bundle subset (``manipulation`` | ``locomotion`` | ``all``).

    ``manipulation`` (the default, backward-compatible) is the original
    pipeline into ``objects/lightwheel``; ``locomotion`` adds the static
    environment family (see :func:`prepare_locomotion`); ``all`` runs both.
    """
    if subset not in ("manipulation", "locomotion", "all"):
        raise SystemExit(
            f"Unknown --subset {subset!r}; expected manipulation, locomotion or all."
        )
    if subset in ("manipulation", "all"):
        out_root, kept = prepare_manipulation(
            source, output_dir, objects=objects, gate=gate,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
            render_dir=render_dir,
        )
    else:
        out_root, kept = output_dir / OUTPUT_ASSETS_SUBDIR, []
    if subset in ("locomotion", "all"):
        _roots, kept_locomotion = prepare_locomotion(
            source, output_dir, objects=objects, gate=gate,
            bbox_tol_abs=bbox_tol_abs, bbox_tol_rel=bbox_tol_rel,
            render_dir=render_dir,
        )
        kept = kept + kept_locomotion
    return out_root, kept


# ---------------------------------------------------------------------------
# Download (Google Drive, anonymous 3.41 GB bundle)
# ---------------------------------------------------------------------------

def _check_bundle_download(zip_path: Path, content_length: int = 0) -> None:
    """Refuse an obviously truncated Drive pull (the link has no checksum).

    The anonymous Drive link advertises no MD5, only the size (the page
    label reads 3.2 GiB / ``content-length`` 3 409 389 226 ≈ 3.41 GB), so a
    partial grab — a plausible failure mode of a multi-GB anonymous
    download — would otherwise only surface as a cryptic
    ``zipfile.BadZipFile`` at extraction time.  A file below 90 % of the
    expected size is deleted and reported; a short-but-plausible one is
    warned about (the file may still extract).
    """
    size = zip_path.stat().st_size
    if content_length and size < content_length:
        print(
            f"  [warn] {BUNDLE_NAME} is {size:,} bytes, short of the "
            f"advertised {content_length:,} bytes."
        )
    if size < BUNDLE_MIN_BYTES:
        zip_path.unlink(missing_ok=True)
        raise SystemExit(
            f"Truncated {BUNDLE_NAME} download: {size:,} bytes (expected "
            f"~{BUNDLE_EXPECTED_BYTES:,}).  The file was deleted — re-run to "
            "retry, or download it manually from "
            f"{DRIVE_VIEW_URL} and pass --source."
        )


def fetch_bundle(output_dir: Path) -> Path:
    """Download ``Lightwheel_OpenSource.zip`` (3.41 GB) and extract it.

    The bundle lives on a Google Drive *anonymous* link — no registration —
    but files >100 MB need the "virus scan" confirm-token flow (verified
    against the live link).  The completed file is size-checked (the link
    advertises no checksum).  Requires ``requests``.
    """
    import re

    try:
        import requests
    except ImportError as exc:  # noqa: F841
        raise SystemExit("`pip install requests` to download the Drive bundle.") from exc

    zip_path = Path(output_dir) / BUNDLE_NAME
    content_length = 0
    if not zip_path.is_file():
        session = requests.Session()
        # 1. the view page sets the anonymous session + the confirm form
        view = session.get(DRIVE_VIEW_URL, timeout=60)
        confirm = re.search(r'name="confirm" value="([^"]+)"', view.text)
        uuid = re.search(r'name="uuid" value="([^"]+)"', view.text)
        action = re.search(r'action="(https://drive\.usercontent\.google\.com/download[^"]*)"', view.text)
        if not (confirm and action):
            raise SystemExit(
                "Could not resolve the Drive confirm token — download the "
                f"bundle manually from {DRIVE_VIEW_URL} and pass --source."
            )
        params = {"id": DRIVE_FILE_ID, "export": "download",
                  "confirm": confirm.group(1)}
        if uuid:
            params["uuid"] = uuid.group(1)
        response = session.get(action.group(1), params=params, stream=True, timeout=120)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        content_length = total
        with open(zip_path, "wb") as sink:
            done = 0
            for chunk in response.iter_content(chunk_size=1 << 22):
                sink.write(chunk)
                done += len(chunk)
                if total:
                    print(f"  {done / 1e9:.2f}/{total / 1e9:.2f} GB", end="\r")
        print()
    # Size sanity: the anonymous link advertises no checksum, so a truncated
    # pull is caught here rather than as a BadZipFile during extraction.
    _check_bundle_download(zip_path, content_length)

    extract_root = Path(output_dir) / BUNDLE_NAME.removesuffix(".zip")
    if not (extract_root / UPSTREAM_MANIPULATION_DIR).is_dir():
        print(f"Extracting {zip_path} (3.41 GB) ...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(extract_root)
    return extract_root


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--subset", choices=("manipulation", "locomotion", "all"),
        default="manipulation",
        help="Bundle subset to package: 'manipulation' (default; the "
             "articulated furniture into objects/lightwheel), 'locomotion' "
             "(the Locomotion scenes — the 7 static environments into "
             "objects/lightwheel_env plus lightwheel_KitchenRoom_RSS into "
             "objects/lightwheel), or 'all'.",
    )
    parser.add_argument(
        "--source", type=Path, default=None,
        help="An extracted Lightwheel_OpenSource tree (with Manipulation/ "
             "and/or Locomotion/). If omitted, the bundle is downloaded from "
             f"the anonymous Google Drive link ({DRIVE_VIEW_URL}; 3.41 GB) "
             "and extracted.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Directory to place the prepared 'objects/lightwheel' tree in.",
    )
    parser.add_argument(
        "--objects", type=str, default=DEFAULT_OBJECTS,
        help="Comma-separated subset of bundle asset names (packaged ids or "
             "upstream names, substring match) to ingest (default: all "
             "Manipulation assets that pass the gate).",
    )
    parser.add_argument(
        "--skip-gate", action="store_true",
        help="Skip the Genesis GPU sanity-load gate (loader + joints + bbox + render).",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Skip the per-object thumbnail render (the gate still checks loads + bboxes).",
    )
    parser.add_argument(
        "--render-dir", type=Path, default=Path("/tmp/lightwheel_sanity"),
        help="Directory for the per-object eyeball renders (default /tmp/lightwheel_sanity).",
    )
    parser.add_argument(
        "--bbox-tol-abs", type=float, default=BBOX_TOL_ABS,
        help=f"Absolute bbox cross-check tolerance in metres (default {BBOX_TOL_ABS}).",
    )
    parser.add_argument(
        "--bbox-tol-rel", type=float, default=BBOX_TOL_REL,
        help=f"Relative bbox cross-check tolerance, as a fraction of the "
             f"per-axis extent (default {BBOX_TOL_REL}).",
    )
    args = parser.parse_args(argv)

    if args.source is not None:
        source = args.source.expanduser().resolve()
        required_dirs = (
            [UPSTREAM_MANIPULATION_DIR] if args.subset == "manipulation"
            else [UPSTREAM_LOCOMOTION_DIR] if args.subset == "locomotion"
            else [UPSTREAM_MANIPULATION_DIR, UPSTREAM_LOCOMOTION_DIR]
        )
        missing = [d for d in required_dirs if not (source / d).is_dir()]
        if missing:
            raise SystemExit(
                f"No {missing} directory/ies in --source: {source} "
                f"(subset={args.subset!r})."
            )
    else:
        print(f"Downloading the Lightwheel bundle ({DRIVE_VIEW_URL}; 3.41 GB) ...")
        source = fetch_bundle(args.output_dir.expanduser().resolve())
        print(f"  -> {source}")

    out_root, _kept = prepare(
        source,
        args.output_dir.expanduser().resolve(),
        objects=(
            [o for o in args.objects.split(",") if o.strip()] if args.objects else None
        ),
        gate=not args.skip_gate,
        bbox_tol_abs=args.bbox_tol_abs,
        bbox_tol_rel=args.bbox_tol_rel,
        render_dir=None if args.no_render else args.render_dir,
        subset=args.subset,
    )

    output_dir = args.output_dir.expanduser().resolve()
    print("\nTo upload (requires write access to the org):")
    for subdir in (
        (OUTPUT_ASSETS_SUBDIR,) if args.subset == "manipulation"
        else (OUTPUT_ENV_ASSETS_SUBDIR,) if args.subset == "locomotion"
        else (OUTPUT_ASSETS_SUBDIR, OUTPUT_ENV_ASSETS_SUBDIR)
    ):
        if (output_dir / subdir).is_dir():
            print(
                f"  hf upload {HF_UPLOAD_REPO} \\\n"
                f"      {output_dir / subdir} {subdir} --repo-type dataset"
            )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
