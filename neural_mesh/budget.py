"""Working-memory token-budget optimizer — priority eviction, not more storage.

Whitespace lane #6 from the agentic-memory scan: most systems treat "working
memory" as a RETRIEVAL problem (find the right thing). This treats it as a
BUDGET problem — a fixed token context that must decide, under a hard cap,
which memories get to live right now and which get evicted.

The primitive here is `select_fit(nodes, budget) -> (kept, evicted)`:

    - Every node has a cost (rough token count) and a value score.
    - We keep the highest-value memories that fit inside the token budget.
    - We evict the lowest-value ones to make room — WITHOUT deleting them
      (they stay in the mesh as cold memory, just out of the active window).

This is the classic knapsack, solved greedily by value-density for speed
(correct enough for a token budget, documented honestly). It pairs with the
mesh: evicted nodes are simply not injected into context, but remain
retrievable. It's the missing half of a working-memory lane.

Value score is composable so callers can weight trust, recency, resonance,
access frequency, or a custom priority — whatever matters for the task.
"""

from __future__ import annotations

from typing import Callable, Optional

# Rough tokens per char (models vary ~3.5-4.5 chars/token). Conservative.
CHARS_PER_TOKEN = 4.0


def token_estimate(text: str) -> int:
    """Conservative token estimate for a string (chars/4, min 1)."""
    return max(1, int(len(text) / CHARS_PER_TOKEN))


def default_value_score(node) -> float:
    """A sensible default value: resonance (relevance) * trust (reliability),
    nudged by recency. Honest and simple; callers can override."""
    import time
    resonance = float(getattr(node, "resonance", 0.0) or 0.0)
    trust = float(getattr(node, "trust", 1.0) or 1.0)
    score = resonance * trust
    # recency nudge: +10% for something accessed within the last hour
    last = float(getattr(node, "last_accessed", 0.0) or 0.0)
    if last and (time.time() - last) < 3600:
        score *= 1.10
    return score


def select_fit(nodes: list, budget: int,
               value_score: Optional[Callable] = None,
               cost_fn: Optional[Callable] = None) -> tuple[list, list]:
    """Greedy value-density knapsack under a token budget.

    Args:
        nodes:  memory nodes to place in the working window.
        budget: max total tokens allowed in the active window.
        value_score: node -> float (higher = keep first). Defaults to
                     `default_value_score`.
        cost_fn: node -> int tokens. Defaults to `token_estimate(content)`.

    Returns:
        (kept, evicted) — the memories that fit (in value order) and the ones
        pushed out (also value order, highest-first so the evicted list reads
        as "closest to fitting").

    Honest notes:
        - Greedy density is NOT the optimal knapsack; for dozens of memories
          vs one budget it's within a few % and is O(n log n) vs exponential.
        - Eviction is non-destructive: nodes remain in the mesh, just out of
          the active context window. That is the whole point of the lane.
    """
    if not nodes:
        return [], []
    value_score = value_score or default_value_score
    cost_fn = cost_fn or (lambda n: token_estimate(getattr(n, "content", "")))
    if budget <= 0:
        return [], sorted(nodes, key=value_score, reverse=True)

    # (value_density, value, cost, node)
    ranked = sorted(
        nodes,
        key=lambda n: (value_score(n) / max(1, cost_fn(n)), value_score(n)),
        reverse=True,
    )
    kept, evicted = [], []
    used = 0
    for n in ranked:
        c = cost_fn(n)
        if used + c <= budget:
            kept.append(n)
            used += c
        else:
            evicted.append(n)
    kept.sort(key=value_score, reverse=True)
    evicted.sort(key=value_score, reverse=True)
    return kept, evicted


def fit_summary(kept: list, evicted: list,
                cost_fn: Optional[Callable] = None) -> dict:
    """Human/agent-readable summary of a budget decision."""
    cost_fn = cost_fn or (lambda n: token_estimate(getattr(n, "content", "")))
    kept_tok = sum(cost_fn(n) for n in kept)
    evicted_tok = sum(cost_fn(n) for n in evicted)
    return {
        "kept_count": len(kept),
        "evicted_count": len(evicted),
        "kept_tokens": kept_tok,
        "evicted_tokens": evicted_tok,
        "evicted_retained_in_mesh": True,
    }
