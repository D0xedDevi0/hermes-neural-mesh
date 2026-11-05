"""LoRA-ready distillation helpers for NEURAL_MESH.

`Mesh.distill()` produces (instruction, response, weight) pairs from the
consolidated mesh. This module persists them in formats a LoRA trainer expects:

  * `write_jsonl(mesh, path)`        — one JSON object per line, our schema
  * `write_hf_jsonl(mesh, path)`     — minimal Alpaca-style
                                       {instruction, input, output} rows
  * `write_weights(mesh, path)`      — "weight <TAB> instruction <TAB> response"
                                       for trainers that support per-example
                                       sample weights (e.g. some PEFT setups)

The weighting lets a downstream trainer emphasize high-trust / corroborated
memory and de-emphasize low-signal nodes — so the LoRA adapter learns the
agent's *curated* knowledge instead of its raw noise.

Pure stdlib. No torch/peft import required at write time.
"""
from __future__ import annotations

import json


def _distill(mesh, **kw) -> dict:
    return mesh.distill(**kw)


def write_jsonl(mesh, path: str, **kw) -> dict:
    d = _distill(mesh, **kw)
    with open(path, "w", encoding="utf-8") as f:
        f.write(d["jsonl"] + ("\n" if d["jsonl"] else ""))
    return {"path": path, "examples": d["count"]}


def write_hf_jsonl(mesh, path: str, **kw) -> dict:
    """Alpaca-style {instruction, input, output} — drop the weight/meta so it
    loads directly into `datasets.load_dataset('json', ...)` for PEFT/LoRA."""
    d = _distill(mesh, **kw)
    rows = [{"instruction": p["instruction"], "input": "", "output": p["response"]}
            for p in d["pairs"]]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": path, "examples": len(rows)}


def write_weights(mesh, path: str, **kw) -> dict:
    """Per-example weight file: `weight\\tinstruction\\tresponse`."""
    d = _distill(mesh, **kw)
    with open(path, "w", encoding="utf-8") as f:
        for p in d["pairs"]:
            f.write(f"{p['weight']}\t{p['instruction']}\t{p['response']}\n")
    return {"path": path, "examples": d["count"]}


def summarize(mesh, **kw) -> dict:
    d = _distill(mesh, **kw)
    by_type = {}
    for p in d["pairs"]:
        t = p["meta"]["type"]
        by_type[t] = by_type.get(t, 0) + 1
    return {
        "examples": d["count"],
        "by_type": by_type,
        "weight_range": (
            min(p["weight"] for p in d["pairs"]),
            max(p["weight"] for p in d["pairs"]),
        ) if d["pairs"] else (0, 0),
    }
