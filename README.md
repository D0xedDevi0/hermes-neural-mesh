# hermes-neural-mesh

**Long-term memory for Hermes that remembers *when* things were true.**

A Hermes `MemoryProvider` backed by [NEURAL_MESH](https://github.com/D0xedDevi0/NEURAL_MESH) —
a self-organizing memory engine with bi-temporal recall, supersession instead of overwrite,
and resonance + lexical retrieval over an auto-linking typed graph.

No API keys. No network. No compiled binary. The core is vendored as pure source, so the
plugin works offline on a fresh install and every line of it is reviewable.

```python
mesh_asof("how often does the deploy key rotate?", as_of="2026-03-01")
→ "the deploy key rotates every 90 days"     # what we believed then

mesh_recall("how often does the deploy key rotate?")
→ "the deploy key rotates every 30 days"     # what is true now, the 90-day value superseded
```

That distinction is the whole point. A flat vector store overwrites the old value and then
cannot answer the first question at all.

---

## What it gives an agent

- **Bi-temporal recall.** Every memory carries `valid_from` / `valid_to`, and `mesh_asof`
  ranks only what was true at that moment. "What did we believe in March" is answerable.
- **Supersession, not deletion.** A corrected fact is stored with `supersedes=<id>`, so the
  history survives and the correction is visible as a correction.
- **Hybrid retrieval.** Dense + lexical fusion by default, so exact-keyword matches that
  dense vectors miss are not lost. The upstream core reports hybrid beating dense,
  resonance-only and lexical-only on MRR (0.260 vs 0.238 / 0.238 / 0.225) on its own
  20-case LongMemEval retrieval experiment — that is the core's measurement, not ours.
- **Poisoning defence.** Content is scanned before a node becomes retrievable; what the
  validator flags is quarantined rather than recalled.
- **A precision floor, measured.** See [Measured behaviour](#measured-behaviour).

## Requirements

- Hermes Agent `>= 0.22.0` (developed and verified against `0.22.7`)
- Python 3.10+ (stdlib only)
- Nothing else. No API key, no network access, no build step.

## Install

The plugin is a directory under your Hermes home. Clone it there and select it:

```bash
git clone https://github.com/D0xedDevi0/hermes-neural-mesh ~/.hermes/plugins/neural-mesh
hermes config set memory.provider neural-mesh
hermes memory status     # expect: Provider: neural-mesh / Plugin: installed ✓ / Status: available ✓
```

Long-lived sessions pick the provider up at startup, so restart the agent (or start a new
session) for the change to take effect. Your memory store is created on first use.

To go back:

```bash
hermes config set memory.provider yantrikdb   # or whichever provider you used before
```

## Tools

Four tools are registered with the agent:

| Tool | What it does |
| --- | --- |
| `mesh_remember` | Store a fact, with optional `kind` (semantic / episodic / procedural / working), `valid_from`, and `supersedes` |
| `mesh_recall` | Search memory for what is relevant to a query |
| `mesh_asof` | Point-in-time recall: what was true on a given date |
| `mesh_stats` | Size, lanes, retrieval backend, and **the path of the store it is using** |

## Configuration

Settings live in `<hermes_home>/neural-mesh/config.json` and can be set through the host's
normal config surface:

- `recall_top_k` (default `5`) — how many memories are offered as context per turn.
- `store_turns` (default `true`) — store substantive turns as episodic memory. Trivial
  prompts (`hi`, `thanks`) and short turns are always skipped.
- `inject_min_score` (default `0.25`) — the precision floor for automatic injection.

The runtime passes `recall_top_k` through `initialize()` if you prefer to set it there; host
kwargs win over the saved file.

## Where your memory lives

```bash
hermes -z "call mesh_stats and tell me the store path"
```

The store is a single SQLite file at `<hermes_home>/neural-mesh/mesh.db`, owned by you and
readable without the plugin. `mesh_stats` and `backup_paths()` both report it. Uninstalling
the plugin does not delete it; removing the file does.

**This is a memory provider, so treat the store as sensitive.** Everything the agent
decides to remember is written there in plain text, and anything in it gets recalled into
session context. It is not encrypted. If a secret should not persist, do not let it reach
memory — the scanning described above reduces accidental poisoning, it does not make the
store a secret store.

## Measured behaviour

### The precision floor

Automatic injection needs to be *strict*: a memory that is merely nearby pollutes every
turn. Measured with three memories stored and nine candidate floors
(`scripts/calibrate_floor.py` reproduces the table):

| query | no floor | ≥ 0.05 | 0.25 | 0.3 | 0.4 |
| --- | --- | --- | --- | --- | --- |
| relevant — deploy key rotation | 3 | 1 | 1 | 1 | **0** |
| relevant — the dog's name | 3 | 1 | 1 | 1 | 1 |
| relevant — hackathon deadline | 3 | 1 | 1 | 1 | 1 |
| unrelated — `zzzz-nonexistent-topic` | **3** | 0 | 0 | 0 | 0 |
| unrelated — quantum chromodynamics | **3** | 0 | 0 | 0 | 0 |

Unrelated queries returned **every stored memory** with no floor. A relative floor
(fraction of the top score) does not help here — it cannot trim a single candidate.
Absolute `0.25` sits inside the measured band where every relevant probe still hits and
unrelated ones return nothing, so automatic injection uses `0.25` and an explicit
`mesh_recall` uses `0.05`.

### Threading

Hermes calls a provider from several threads: `initialize()` on the startup thread, tool
calls on the executor thread, and `sync_turn` on a background **daemon** worker. A sqlite3
connection may only be used by the thread that created it, and the core owns its own
connection — so a single shared handle raises *"SQLite objects created in a thread can only
be used in that same thread"*. In `sync_turn` that exception is swallowed by design, which
makes the failure look exactly like *"memory silently didn't save"*. Each thread therefore
gets its own handle on the same file (`_m()`); the core commits per call, so writes are
immediately visible to the others. Regression tests cover it.

## What this does **not** claim

- **No token or cost savings are claimed.** Injecting memories adds input tokens. What it
  buys is recall, not thrift.
- **No benchmark leadership is claimed.** The MRR figure above is the upstream core's own
  experiment, quoted and attributed. This plugin has not been benchmarked against other
  memory providers, and no comparison is asserted.
- **The Rust acceleration module is not shipped.** `rust_mesh.so` is a compiled extension and
  catalog rule 17 forbids shipping binaries outside a CI build attestation. Retrieval runs on
  the pure-Python path, which the core documents as the definition and the Rust functions as
  the accelerated twins. The mesh reports its own backend: `resonance_backend: python`.

## Disclosure

How it works, and what it touches:

- Runs **in-process** with the agent in pure Python. It reads no `.env` file, no
  `auth.json`, no browser profile, no keychain, and no other tool's credential store — it
  has no credentials at all.
- Makes **no network calls**. There is no telemetry, no remote store, no sync.
- Spawns **no child processes** and no Hermes children.
- Writes only inside its own store directory under the Hermes home, plus the config file
  named above.
- On a completed turn it stores a redacted turn snippet, and it *mirrors* built-in
  `MEMORY.md` / `USER.md` writes into the mesh. Destructive native writes (`remove`) are
  deliberately not mirrored: without exact identity in the callback, mirroring a wrong
  delete is worse than not mirroring one.

## Development

```bash
python3 -m unittest discover -s tests -v     # 22 tests, offline, no credentials
python3 scripts/calibrate_floor.py           # reproduce the precision-floor table
```

The tests pass both against the real `MemoryProvider` ABC (`PYTHONPATH=/opt/hermes`) and on
a bare interpreter with no Hermes installed, because the plugin falls back to a local stub
of the ABC when the real one is not importable.

## Provenance and licence

MIT. The vendored `neural_mesh/` package is MIT, by the same author, pinned to upstream
commit `fd7b35c7b5b9a61a8c7f4fbcebfdda25070a1d6e` — see [PROVENANCE.md](PROVENANCE.md) for the
exact vendoring command, what was excluded, and why.
