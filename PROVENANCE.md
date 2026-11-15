# Provenance of the vendored `neural_mesh/` package

This repository ships the NEURAL_MESH memory engine **as source, vendored**, so that the
plugin is dependency-free and reviewable line by line.

| | |
| --- | --- |
| Upstream | https://github.com/D0xedDevi0/NEURAL_MESH |
| Upstream commit | `b4a9d9b01932d317e948eb362a1f0833290ecdae` (2026-11-15) |
| Upstream version | `neural-mesh` 0.40.0 |
| License | MIT (see `LICENSE`) |
| Vendored | `neural_mesh/` — 47 Python files, 536K |
| Not vendored | `rust_mesh.so` (compiled acceleration), `neural_mesh/integrations/` (loads a credential file), `data/`, `bench/`, `server.py`, the HTTP service, tests and docs of the upstream project |

## Why it is vendored rather than declared as a dependency

`neural-mesh` is not published on PyPI, so there is nothing for a dependency resolver to
fetch. It is also not installed into Hermes's environment: a plugin that pip-installs itself
into the agent's interpreter can break the agent, which is exactly what the catalog's
dependency policy exists to prevent. Vendoring reviewed source is the honest option.

## Why the Rust module is missing, and why that costs nothing

`rust_mesh.so` is a compiled CPython extension (ELF). Catalog rule 17 forbids shipping a
compiled binary in a plugin — detected by file header, not extension, and a checksum
committed alongside proves nothing about where the binary came from. The only allowance is a
binary built by the repository's own public GitHub Actions with a build attestation tying it
to the commit. Shipping it is not an option, and working around the rule is not a
consideration.

Retrieval does not depend on it. `neural_mesh/bm25.py` documents the pure-Python
implementation as the definition and the Rust functions (`bulk_bm25`, `bm25_score`,
`bm25_idf`, `Bm25Index`) as "the accelerated (abi3, no-deps) twins". The import is guarded,
and the mesh reports its own backend:

```
>>> mesh.stats()
{'total': 2, 'by_type': {'semantic': 2}, 'hot': 2, 'cold': 0,
 'quarantined': 0, 'resonance_backend': 'python'}
```

`resonance_backend: python` is that fallback working. Any user who wants the accelerated path
can install `neural-mesh` from the upstream repository themselves — the plugin does not need
it and does not fetch it.

## Why `integrations/` is missing

Upstream ships `neural_mesh/integrations/` holding a Helixa signer, a Helixa provenance API, an
x402 helper and a YantrikDB bridge. The signer loads a private key out of a credential file on
disk. A catalog plugin must not reach for credential stores, and this plugin's whole claim is
that it holds no credentials at all — so the subpackage is not vendored, and the core's three
references to it were removed as well (a top-level import in `neural_mesh/__init__.py`, its two
lazy `__getattr__` branches, and its `__all__` entries). `scripts/trust_check.py` fails the
build if the file or a reference to it ever returns.

Nothing on the store/recall path touches it. `Mesh`, `add`, `recall`, `recall_asof`, the
temporal helpers, resonance and BM25 all import and run without it, which the test suite
exercises on every run.

Two references to the bridge survive inside `x402_recall.py`, both wrapped in
`try/except Exception: pass` — they degrade to no-op, so they are left as upstream wrote them.
Separately, upstream's `__all__` lists six `x402_recall` names that had no loader behind them,
so `from neural_mesh import *` raised `AttributeError` on each; the vendored copy gives them
lazy branches, as noted in the code.

## The provenance ledger is part of this surface

`neural_mesh/ledger.py` (v0.40.0) is vendored with the rest, and it is the reason
`mesh_verify` / `mesh_provenance` exist. It is pure stdlib, writes only to the same sqlite
file the store already uses, and opens no network connection — its Base (8453) anchor support
*produces* calldata, it never broadcasts. The ledger reads no credential file, so nothing here
contradicts the `integrations/` exclusion above.

One consequence worth knowing before an upgrade: an existing store has no ledger, so the first
`mesh_verify` reports unwitnessed memories (invariant I3, `coverage`) rather than OK. That is
correct — the log genuinely has no record of them — and adoption is explicit (`adopt: true`),
never silent. Adoption attests those memories **as of the moment it runs**; it does not and
cannot prove they were never altered before it.

## Keeping it current

Upstream moves. To refresh the vendored copy:

```bash
git clone https://github.com/D0xedDevi0/NEURAL_MESH /tmp/nm
git -C /tmp/nm checkout <commit>
tar cf - --exclude='*.so' --exclude='__pycache__' --exclude='*.pyc' \
    -C /tmp/nm neural_mesh | tar xf - -C .
```

Then update the commit hash in this file, bump `version` in `plugin.yaml`, re-run the tests,
and re-pin the catalog entry — the version on the entry's card must match the code behind it.
