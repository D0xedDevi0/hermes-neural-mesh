# Why NEURAL_MESH

This is the written answer to the hackathon brief. The short version: the brief asks for
plugins that do more than a prompt or a skill could, and it names memory as an eligible
surface. NEURAL_MESH changes **where Hermes stores memory and what recall is allowed to
answer** — it is a `MemoryProvider`, not a tool wrapper.

> "A plugin can change anything from the model Hermes runs on, **where it stores its
> memory**, where it runs commands, which messaging apps it works in and what Hermes
> Desktop shows." — the hackathon announcement

## What it is

A first-class Hermes memory provider. It registers through the provider ABC, so Hermes
calls it on its own lifecycle — startup, executor thread, `sync_turn` on the daemon
worker — rather than only when a model decides to call a tool.

The part that matters: **every fact is bi-temporal.** A memory carries `valid_from`,
`valid_to` and `supersedes`, so retrieval can answer two different questions:

- what is true *now*
- what was true **on a given date**

A conventional vector store can only answer the first. When a fact changes, the old one is
not overwritten — its validity window is closed at the successor's start, and
`recall_asof(date)` still returns it.

## Why a prompt or a skill cannot do this

A skill can ask a model to *remember* something. It cannot change where Hermes stores
memory, cannot intercept the memory lifecycle, and cannot change how recall ranks. This
plugin supplies the storage and the retrieval path itself.

## Measured, not asserted

Everything below comes from scripts in this repository, run through the shipped tool
surface, with the raw output committed alongside.

**Temporal ablation** — same engine, same 15 fact families, both arms through the
provider's tools (`scripts/evidence_temporal.py`, `docs/evidence/temporal-accuracy.json`):

| question | temporal (this plugin) | flat append-only store |
|---|---|---|
| what is true now | **15/15** | 1/15 |
| what was true on a date | **15/15** | 1/15 |

The flat arm scores 1/15 because a store that only appends returns the *newest* matching
text for a dated question — the wrong fact, confidently.

**Retrieval calibration** — `scripts/calibrate_floor.py`:

| score floor | unrelated queries | relevant query |
|---|---|---|
| ≥ 0.05 | 0 hits | retained |
| 0.30 | 0 hits | retained |

Automatic context injection uses a **0.25** floor, explicit recall **0.05**. The point of
the higher automatic floor is that injected context must not add unrelated memories.

**Behavioural coverage** — 23 tests, including per-thread storage safety (Hermes invokes
the provider from startup, executor and daemon-worker threads), supersession, point-in-time
recall, and an unmatched `supersedes` ID, which warns rather than silently appending a
contradiction.

## Disclosures

- **Storage is local SQLite and unencrypted.** Do not put credentials or secrets in it.
- **No network calls at runtime, no API keys, no credential-store access.** It does not
  read `.env`, `auth.json`, browser profiles or keychains.
- **No compiled binaries shipped.** The upstream Rust extension is excluded under catalog
  rule 17; the vendored core is pure Python, with exclusions documented in `PROVENANCE.md`.
- **One external provider is active at a time.** Installing NEURAL_MESH means choosing it
  over whatever else supplies memory in that Hermes install.

## Honest limits

- The catalog PR is **open, not merged**, and had 0 CI check runs at the last inspection.
- The pure-Python retrieval path is slower than the Rust extension it deliberately excludes.
- The two injection floors are calibrated on this corpus, not a broad benchmark; they are a
  measured default, not a universal one.

## Try it

```bash
git clone https://github.com/D0xedDevi0/hermes-neural-mesh
cd hermes-neural-mesh && python3 -m unittest discover -s tests    # 23 tests
python3 scripts/evidence_temporal.py                              # the ablation above
```

Demo: [`docs/demo.mp4`](docs/demo.mp4) — 85 seconds, recorded from real command output.
