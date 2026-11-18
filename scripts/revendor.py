#!/usr/bin/env python3
"""Re-vendor `neural_mesh/` from upstream, re-applying this repo's local edits.

Vendoring by hand is how the documented commit and the shipped code drift apart:
the last refresh updated `ledger.py` while `__init__.py` and `__main__.py` stayed
one release behind, and PROVENANCE.md named a commit that two of the four shipped
files did not match. This script does the copy and the edits in one pass, and
**fails loudly** if upstream moves the code an edit targets — a silent no-op here
would ship the credential-loading `.integrations` import into a plugin whose whole
claim is that it holds no credentials.

    python3 scripts/revendor.py <upstream-ref> [--clone DIR]

It asserts afterwards, with the same checks a reviewer would run:
  * the three excluded things really are absent (`.so`, `integrations/`)
  * `neural_mesh` imports and reports the expected version
  * `__all__` resolves — every name is real, so `from neural_mesh import *` works
"""

from __future__ import annotations

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

UPSTREAM = "https://github.com/D0xedDevi0/NEURAL_MESH"
HERE = pathlib.Path(__file__).resolve().parent.parent
PKG = "neural_mesh"

# ── local edits, expressed as exact text operations ──────────────────────────
# Each is (what, old, new, count). A missing or duplicated `old` is a hard error:
# upstream moved the code, so a human has to look at the new shape.

HELIXA_IMPORT = (
    "the .integrations.helixa_provenance import block",
    """from .integrations.helixa_provenance import (
    HelixaStamp, stamp_node, verify_stamp, aura_trust_weight,
    export_manifest, make_stamp,
)
""",
    """# NOTE (vendoring): upstream also exports a Helixa provenance/stamp API from
# .integrations.helixa_provenance at this point. That subpackage is deliberately NOT vendored
# — its signer loads a private key out of a credential file, which a plugin installed into a
# memory slot must never do — so the import and its names are removed rather than shipped.
# See PROVENANCE.md; `Mesh`, the store, and every recall path are unaffected.
""",
    1,
)

HELIXA_LAZY = (
    "the HelixaSigner / YantrikDBBridge lazy-import branches",
    """    if name == "HelixaSigner":
        if "HelixaSigner" not in _LAZY:
            from .integrations.helixa_signer import HelixaSigner as _HS
            _LAZY["HelixaSigner"] = _HS
        return _LAZY["HelixaSigner"]
    if name == "YantrikDBBridge":
        if "YantrikDBBridge" not in _LAZY:
            from .integrations.yantrikdb_bridge import YantrikDBBridge as _YB
            _LAZY["YantrikDBBridge"] = _YB
        return _LAZY["YantrikDBBridge"]
""",
    "",
    1,
)

X402_NAMES = (
    "the _X402_NAMES lazy-loader",
    """def __getattr__(name):""",
    '''# NOTE (vendoring): these six names sit in __all__ upstream with no loader behind them, so
# `from neural_mesh import *` raised AttributeError on each one. They import cleanly from
# .x402_recall, so the vendored copy exposes them lazily rather than leaving __all__ untrue.
_X402_NAMES = {
    "TIERS", "SERVICE_NAME", "RECEIPT_CONTRACT", "FEE_RECIPIENT", "BASE_RPC",
    "verify_receipt_onchain",
}

def __getattr__(name):''',
    1,
)

X402_BRANCH = (
    "the x402 lazy branch inside __getattr__",
    """    raise AttributeError(f"module 'neural_mesh' has no attribute {name!r}")""",
    """    if name in _X402_NAMES:
        if name not in _LAZY:
            from .x402_recall import (
                TIERS, SERVICE_NAME, RECEIPT_CONTRACT,
                FEE_RECIPIENT, BASE_RPC, verify_receipt_onchain,
            )
            _LAZY.update({
                "TIERS": TIERS, "SERVICE_NAME": SERVICE_NAME,
                "RECEIPT_CONTRACT": RECEIPT_CONTRACT, "FEE_RECIPIENT": FEE_RECIPIENT,
                "BASE_RPC": BASE_RPC, "verify_receipt_onchain": verify_receipt_onchain,
            })
        return _LAZY[name]
    raise AttributeError(f"module 'neural_mesh' has no attribute {name!r}")""",
    1,
)

ALL_HELIXA_STAMPS = (
    "the Helixa stamp names in __all__",
    """           "HelixaStamp", "stamp_node", "verify_stamp", "aura_trust_weight",
           "export_manifest", "make_stamp",
""",
    "",
    1,
)

ALL_HELIXA_SIGNER = (
    "HelixaSigner in __all__",
    """           "HelixaSigner",
""",
    "",
    1,
)

EDITS = (HELIXA_IMPORT, HELIXA_LAZY, X402_NAMES, X402_BRANCH,
         ALL_HELIXA_STAMPS, ALL_HELIXA_SIGNER)


def clone_upstream(ref: str, dest: pathlib.Path) -> pathlib.Path:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "--quiet", UPSTREAM, str(dest)], check=True)
    subprocess.run(["git", "-C", str(dest), "checkout", "--quiet", ref], check=True)
    got = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                         capture_output=True, text=True, check=True).stdout.strip()
    if not got.startswith(ref) and got[:7] != ref[:7]:
        print(f"!! checked out {got}, expected {ref}", file=sys.stderr)
        sys.exit(2)
    return dest


def copy_package(src: pathlib.Path, dst: pathlib.Path) -> list[str]:
    """Copy the package minus what must never be vendored. Returns what was skipped."""
    if dst.exists():
        shutil.rmtree(dst)
    skipped: list[str] = []
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_dir():
            if rel.parts and rel.parts[0] == "integrations":
                skipped.append(f"{rel}/ (credential store — see PROVENANCE.md)")
                continue
            continue
        if "__pycache__" in rel.parts or path.suffix in (".so", ".pyd", ".pyc"):
            skipped.append(str(rel))
            continue
        if rel.parts and rel.parts[0] == "integrations":
            skipped.append(str(rel))
            continue
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)
    return skipped


def apply_edits(pkg: pathlib.Path) -> None:
    init = pkg / "__init__.py"
    text = init.read_text()
    for what, old, new, count in EDITS:
        found = text.count(old)
        if found != count:
            raise SystemExit(
                f"!! revendor: {what}: expected {count} occurrence(s), found {found}.\n"
                f"   Upstream moved this code. Read it and update EDITS in "
                f"{pathlib.Path(__file__).name} rather than shipping the unfixed copy."
            )
        text = text.replace(old, new)
    init.write_text(text)


def verify(repo: pathlib.Path, expect_version: str) -> None:
    bad: list[str] = []
    for path in repo.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix in (".so", ".pyd", ".dylib", ".dll", ".node", ".wasm"):
            bad.append(f"compiled artifact shipped: {path}")
    if (repo / PKG / "integrations").exists():
        bad.append("neural_mesh/integrations/ is present")
    for ref in (repo / PKG).rglob("*.py"):
        for line in ref.read_text(errors="replace").splitlines():
            if re.match(r"\s*(from|import)\s+\.integrations", line) and \
                    "x402_recall" not in str(ref):
                bad.append(f"live .integrations import in {ref.relative_to(repo)}: {line.strip()}")
    if bad:
        raise SystemExit("!! revendor verification failed:\n  " + "\n  ".join(bad))

    probe = (
        "import neural_mesh as m, sys\n"
        "missing = [n for n in m.__all__ if not hasattr(m, n)]\n"
        "assert not missing, f'__all__ names that do not resolve: {missing}'\n"
        f"assert m.__version__ == '{expect_version}', m.__version__\n"
        "from neural_mesh import Mesh\n"
        "import tempfile, os\n"
        "d = tempfile.mkdtemp()\n"
        "mesh = Mesh(os.path.join(d, 'm.db'), validator=False)\n"
        "mesh.add('revendor smoke memory', agent_id='revendor')\n"
        "hits = mesh.recall('smoke')\n"
        "rep = mesh.verify_memory(deep=True)\n"
        "assert rep.ok, rep.summary()\n"
        "mesh.db.close()\n"
        "print('probe ok —', m.__version__, 'ledger', rep.summary())\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"!! revendor probe failed:\n{out.stdout}\n{out.stderr}")
    print(out.stdout.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref", help="upstream commit/tag to vendor")
    ap.add_argument("--clone", default="/tmp/nm-revendor", help="scratch clone directory")
    ap.add_argument("--expect-version", default="", help="assert the vendored __version__")
    args = ap.parse_args()

    src = clone_upstream(args.ref, pathlib.Path(args.clone))
    skipped = copy_package(src / PKG, HERE / PKG)
    apply_edits(HERE / PKG)
    print(f"vendored {PKG}/ from {args.ref}")
    for s in skipped:
        print(f"  skipped {s}")
    version = args.expect_version
    if not version:
        text = (HERE / PKG / "__init__.py").read_text()
        version = text.split('__version__ = "')[1].split('"')[0]
    verify(HERE, version)
    print(f"OK — {PKG} {version} vendored and verified. "
          f"Now update PROVENANCE.md (commit {args.ref}) and plugin.yaml version.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
