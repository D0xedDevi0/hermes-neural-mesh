# hermes-neural-mesh — packaging scope

**NEURAL_MESH as a catalog plugin.** Proposed second entry. Feasibility proven, not assumed.

Source: `github.com/D0xedDevi0/NEURAL_MESH` · `neural-mesh` v0.39.0 · MIT · author owns the repo
(catalog rule 5 satisfied: owner or major contributor).

## Why this is the stronger artifact than basekit

Not a vibe — the measured difference:

| | basekit | hermes-neural-mesh |
| --- | --- | --- |
| what it is | read-only Base chain data | a memory architecture agents *live in* |
| novelty | useful, conventional | "self-organizing, self-forgetting mesh — resonance retrieval beats flat vector search" |
| receipts in-repo | 1 powered A/B, **p=0.17, effect unproven** | LOCOMO + LongMemEval harnesses and result reports, 50 modules, 40+ tests |
| onboarding | no keys | **no keys, no deps, no binary** (proven below) |
| catalog shelf | a read-only Base plugin — **no competitor among 995 entries** | **21 memory entries** — the loudest shelf |
| ceiling | high floor, low ceiling | **high ceiling, higher variance** |

The announcement names this category outright ("where it stores its memory"), so judges will look
at memory plugins — and most of the 21 are vector-store wrappers. The differentiator has to be the
architecture and a measured comparison, not the category.

## The rule that shapes the package — rule 17

`rust_mesh.so` is a compiled Python extension (ELF). Rule 17 forbids shipping a compiled binary,
detected **by file header, not extension**, and a committed checksum proves nothing about origin —
the only allowance is a binary built by the repo's own public GitHub Actions with a build
attestation. So the `.so` cannot ship, and must not be worked around.

That costs nothing, because the package is built for it: `bm25.py` documents the Rust twins as the
*accelerated* path, pure Python being the definition. Verified by vendoring source only:

```
vendored: 50 py files, 540K, binaries: 0
rust_mesh visible: False
IMPORT OK without the binary, v 0.39.0
RECALL OK -> ['the deploy key rotates every 90 days', 'Kumo is a husky']
NO KEY, NO DEPS, NO BINARY: works
```

## Other gates, already cleared

- **Rule 10 (deps):** `pyproject.toml` declares no required dependencies — *"Core is pure stdlib."*
  fastembed / eth-account / web3 are optional extras and every optional import is guarded; the core
  imports clean with all three absent. A zero-dependency plugin has no dependency list to argue about.
- **Rule 11 (credentials):** nothing is needed. Storage and retrieval are local; the optional reader
  LLM takes `api_key=` as a parameter, so if we ever expose it, it goes through
  `agent.secret_scope.get_secret()` — never `os.environ`, never a file read. Preferred route for the
  answer step is `credentialed_provider_request` so a user's existing sign-in is used and the plugin
  never holds a token.
- **Rule 9 (no core override):** the package's single `setattr` is on its own SQLite connection
  object, not a Hermes core object. No `sys.modules` patching, no private core methods.
- **Rule 13 (disclosure):** to write — local SQLite store, no network by default, optional LLM call
  when the reader is enabled and disclosed as such.

## The design that matters: it should be a memory *provider*, not a toolbox

Hermes memory providers are a first-class plugin type — a developer guide exists for exactly this
("How to build a memory provider plugin for Hermes Agent"), and the slot is a single-select list of
named competitors: **honcho, openviking, mem0, hindsight, holographic, retaindb, byterover,
supermemory**. Only one external provider can be active at a time, and built-in `MEMORY.md`/`USER.md`
stays active regardless. `yantrikdb` — what this agent runs on right now (`memory.provider:
yantrikdb`) — is the local reference implementation: a `MemoryProvider` subclass that declares its
`hooks` (`on_session_end`, `on_pre_compress`, `on_memory_write`, `on_delegation`) in `plugin.yaml`.

So the entry is not "six tools and a hook". It is a `MemoryProvider` subclass wrapping the vendored
core: the agent's long-term memory, in-process, owned by the user, zero keys. Tools remain as the
manual surface (`mesh_remember` · `mesh_recall` · `mesh_relate` · `mesh_forget` · `mesh_stats` ·
`mesh_asof`), and note that `recall_asof` / `valid_at` — "what was true on this date" — is a
capability flat vector memory does not have at all.

Rule 18's tier restriction is specific to `register_login_backend` and does **not** cover memory
providers, so a community-tier provider listing is a supported path, not a loophole.

The hook is also measurable in a way basekit's was not: memory injection can be scored on answer
*accuracy* (a memory-dependent question answered right with injection, wrong without) — a capability
gain, not a call-count argument that dies at p=0.17.

## Why it isn't wired in today — the honest answer

Nothing mysterious, and it is worth stating in the entry rather than pretending the integration
exists:

- **No provider adapter.** The mesh has a `Mesh` class and an HTTP service, not a `MemoryProvider`
  subclass, so Hermes has no route to select it as `memory.provider`.
- **A boundary mismatch.** Hermes providers are in-process Python; the live mesh answers over HTTP on
  `127.0.0.1:4021` with a Rust resonance backend. Vendoring the pure-Python core removes the boundary
  — each installer gets their own local mesh instead of pointing at our VPS.
- **The bridge exists one level down, not at the provider level.** `integrations/yantrikdb_bridge.py`
  (`YantrikDBBridge`) is already imported by the package, and `x402_recall.py` prices recall modes
  that include it (`deep $0.05`, `ultra $0.10`). So mesh↔yantrikdb interop is real code; what is
  missing is mesh↔Hermes.

Two things this also tells us, which the entry should not overstate: the mesh is **not** this agent's
memory today (yantrikdb is), and `api.d0xeddev.com/mesh/health` returns **404** — the mesh is live on
localhost, not published at that path.

## Lineage worth naming — it has already been judged

The mesh descends from a real judged entry: `/opt/data/sibyl-hackathon/SUBMISSION.md` records **THE
SPINE** (team NEURAL_MESH, repo `dejavu-sibyl-memory`, MIT) with reproduced measurements on disk —
`demo/ablation_results.json`: 200 seeded frames, 150 stressed, **7.072 pp mean simulated loss
averted**, 75% of decisions changed, and a fresh-process policy of **0.024 equity with memory vs
0.55 after the store is wiped**. That is the shape of evidence the hackathon's reliability axis
rewards, and it already exists.

## The 90-second demo writes itself

Same question, same corpus (LOCOMO ships in the repo), two retrieval paths side by side: flat vector
similarity vs the mesh. Then the thing nothing else on the shelf does — ask what was true *last
month* and get the value from then, not the current one (`recall_asof`). Show, don't claim.

## Risks

- **Legibility.** 10k lines can lose to a 200-line plugin in a 90s video. The demo must be two
  queries and one comparison, not an architecture tour.
- **The shelf is loud.** 21 entries, including one already installed here (`yantrikdb`, listed by its
  own authors). Winning means the benchmark page is better than theirs.
- **Time.** Six days to the deadline, and a vendored 540 KB tree needs review-grade hygiene
  (provenance note, upstream SHA recorded, licence kept).

## Decision

Park `pmkit` (it was the hedge for a low ceiling; this is a better use of the same days). Build
`hermes-neural-mesh` as the second entry — a separate repo, separate PR, separate demo. Whether it is
a second *entry* or only a second *listing* depends on the multi-entry answer from the organizers;
**nothing about building it depends on that.**
