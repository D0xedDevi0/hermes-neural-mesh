"""NEURAL_MESH security layer — memory poisoning defense (OWASP ASI06).

WHY
---
Persistent agent memory is an attack surface: an attacker plants a payload in
a node that is ingested as "trusted context" days later. The mesh must not
trust ingested content by default. This module provides:

  1. ContentValidator — static scan of *content* for injection markers
     (prompt-injection idioms, tool-call chains, shell commands, base64
     confusion) BEFORE the content enters the mesh.
  2. The QUARANTINE lane — content that fails validation lands in
     lane="quarantine": zero resonance, excluded from every default retrieval
     path, visible ONLY to explicit audit queries.
  3. Trust decay — unverified (non-corroborated) nodes decay trust *= 0.85
     per sleep cycle so unconfirmed claims fade instead of compounding.
  4. Cross-source corroboration — 2+ independent agents/provenances confirming
     the same fact get a trust bumper (1-(1-t_a)(1-t_b)) and a
     meta["corroborated"] flag.

Pure stdlib. No network code, no signing, no key material — metadata + policy
only, matching the Helixa safety contract.
"""
from __future__ import annotations

import base64
import hashlib
import re
from dataclasses import dataclass, field

QUARANTINE_LANE = "quarantine"

# severities — CRITICAL alone quarantines; HIGH counts toward the score
CRITICAL, HIGH, MEDIUM = "critical", "high", "medium"

_SEV_WEIGHT = {CRITICAL: 3.0, HIGH: 2.0, MEDIUM: 1.0}


@dataclass
class Pattern:
    name: str
    regex: str
    severity: str = HIGH
    note: str = ""


@dataclass
class Verdict:
    level: str                      # "safe" | "suspicious" | "malicious"
    score: float                    # 0..1 (weighted, capped)
    patterns: list[dict] = field(default_factory=list)  # [{name, severity, note}]

    @property
    def is_safe(self) -> bool:
        return self.level == "safe"

    @property
    def is_suspicious(self) -> bool:
        return self.level == "suspicious"

    @property
    def is_malicious(self) -> bool:
        return self.level == "malicious"


# --------------------------------------------------------------------------
# Static pattern catalog — keep auditable in ONE place. Add new vectors here.
# --------------------------------------------------------------------------
PATTERNS: list[Pattern] = [
    # --- prompt-injection idioms (the ASI06 core) ---
    Pattern("ignore-prior-instructions",
            r"(?i)\bignore\s+(all\s+)?(previous|prior|above|earlier)\s+"
            r"(instructions?|prompts?|messages?|context|directions?)\b",
            CRITICAL, "classic prompt injection: override prior instructions"),
    Pattern("disregard-prior",
            r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+"
            r"(instructions?|prompts?|messages?|context)\b",
            CRITICAL),
    Pattern("you-are-now",
            r"(?i)\byou\s+are\s+now\b(?!\s+(?:a|the)?\s*"
            r"(?:mesh|memory|agent\s+assistant))",
            CRITICAL, "identity override / roleplay takeover"),
    Pattern("forget-instructions",
            r"(?i)\b(forget|forget\s+all|erase)\s+(your\s+)?"
            r"(instructions?|rules?|guidelines?|system\s+prompt)\b",
            CRITICAL),
    Pattern("override-system-prompt",
            r"(?i)\boverride\s+(your\s+)?(system\s+)?prompt\b",
            CRITICAL),
    Pattern("jailbreak",
            r"(?i)\bjailbreak\b|\bDAN\s+mode\b|\bdo\s+anything\s+now\b",
            HIGH),
    Pattern("act-as",
            r"(?i)\bact\s+as\b",
            HIGH, "roleplay impersonation — frequently injection-laced"),
    Pattern("repeat-back",
            r"(?i)\b(repeat|echo|print)\s+(back\s+)?(this\s+)?(exactly|verbatim)\b",
            HIGH, "exfiltration idiom"),
    Pattern("hidden-instruction",
            r"(?i)\b(hidden|embedded|secret)\s+instruction[s]?\b",
            HIGH),
    Pattern("ignore-above",
            r"(?i)\bignore\s+(everything\s+)?above\b",
            CRITICAL),
    Pattern("start-fresh",
            r"(?i)\b(start\s+fresh|start\s+over|reset\s+(your|all))\s+"
            r"(instructions?|context|memory)\b",
            HIGH),
    # --- tool-call chain idioms ---
    Pattern("call-tool",
            r"(?i)\b(call|invoke|use|execute)\s+(the\s+)?(tool|function)\s+"
            r"(named\s+)?[A-Za-z_][A-Za-z0-9_]*\b",
            HIGH, "tool-call chain: weaponized memory driving tool abuse"),
    Pattern("tool-result-format",
            r"(?i)\b(tool[_\- ]?result|function[_\- ]?call|"
            r"tool[_\- ]?call)\b\s*[:=]",
            HIGH, "forged tool-response framing"),
    Pattern("read-your-files",
            r"(?i)\b(read|list|open|exfiltrate|send)\s+(my\s+)?"
            r"(files|credentials|keys|secrets|env|\.env)\b",
            HIGH, "credential exfiltration idiom"),
    # --- shell / code execution markers ---
    Pattern("shell-rm",
            r"(?i)\brm\s+-rf\b|\brm\s+-fr\b",
            CRITICAL, "destructive shell command"),
    Pattern("shell-curl-pipe",
            r"(?i)\bcurl\s+[^\n|;]*\s*\|\s*(sh|bash|zsh)\b",
            CRITICAL, "curl|sh remote-code-execution idiom"),
    Pattern("shell-download-exec",
            r"(?i)\b(wget|curl)\s+\S+\s+-o\s+\S+\s*(\&\&|;)\s*"
            r"(chmod|\./|bash|python)",
            CRITICAL),
    Pattern("os-system",
            r"(?i)\bos\.system\s*\(|\bsubprocess\b|\bPopen\s*\(",
            HIGH),
    Pattern("eval-exec",
            r"(?i)\beval\s*\(|\bexec\s*\(|\bexecfile\b|\b__import__\b",
            HIGH),
    Pattern("base64-decode",
            r"(?i)\bbase64\s*-d\b|\bfrom\s+base64\b",
            HIGH),
    Pattern("chmod-x",
            r"(?i)\bchmod\s+\+?x\b",
            MEDIUM),
    Pattern("shell-pipe-sh",
            r"(?i)\b(\|\s*(sh|bash|zsh|python)\b|;\s*(sh|bash|zsh|python)\b)",
            HIGH),
]

# base64 confusion: a long run of base64 alphabet with padding — often an
# encoded payload ("base64 confusion" attack on tokenizers/interpreters).
_B64_RUN = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{32,}={0,2})(?![A-Za-z0-9+/])")


class ContentValidator:
    """Static, deterministic content scanner. Pure pattern match — no LLM,
    no network, no false-negative hiding behind "semantics". Configurable via
    `severity_weights` and `extra_patterns`."""

    def __init__(self, patterns: "list[Pattern] | None" = None,
                 severity_weights: "dict | None" = None,
                 max_b64_run: int = 32):
        self.patterns = list(patterns) if patterns is not None else list(PATTERNS)
        self.weights = dict(severity_weights or _SEV_WEIGHT)
        self.max_b64_run = max_b64_run
        self._compiled = [(p, re.compile(p.regex)) for p in self.patterns]

    # -- public API --------------------------------------------------------
    def scan(self, content: str) -> Verdict:
        """Scan content → Verdict. Safe content passes; suspicious/malicious
        content is flagged for quarantine. Deterministic, side-effect free."""
        if not content or not content.strip():
            return Verdict("safe", 0.0)
        hits = []
        for pat, rx in self._compiled:
            if rx.search(content):
                hits.append({"name": pat.name, "severity": pat.severity,
                             "note": pat.note or pat.name})
        b64 = self._scan_base64(content)
        if b64:
            hits.append(b64)
        return self._verdict(hits)

    def scan_many(self, contents: "list[str]") -> "list[Verdict]":
        return [self.scan(c) for c in contents]

    # -- internals ---------------------------------------------------------
    def _scan_base64(self, content: str) -> "dict | None":
        for m in _B64_RUN.finditer(content):
            chunk = m.group(1)
            if len(chunk) < self.max_b64_run:
                continue
            # avoid flagging ordinary hashes/uuids: require padding OR
            # successful decode to printable text of meaningful length
            try:
                dec = base64.b64decode(chunk, validate=True)
            except Exception:
                continue
            if dec and len(dec) >= 16:
                printable = sum(1 for b in dec if 32 <= b < 127)
                if printable / len(dec) > 0.8:
                    return {"name": "base64-blob", "severity": HIGH,
                            "note": f"encoded payload ({len(chunk)} chars)"}
        return None

    def _verdict(self, hits: "list[dict]") -> Verdict:
        if not hits:
            return Verdict("safe", 0.0, [])
        critical = any(h["severity"] == CRITICAL for h in hits)
        high = sum(1 for h in hits if h["severity"] == HIGH)
        score = min(1.0, sum(self.weights.get(h["severity"], 0.0) for h in hits) / 4.0)
        if critical or high >= 2:
            return Verdict("malicious", round(score, 3), hits)
        if high >= 1 or any(h["severity"] == MEDIUM for h in hits):
            return Verdict("suspicious", round(score, 3), hits)
        return Verdict("safe", 0.0, hits)


# --------------------------------------------------------------------------
# corroboration helpers
# --------------------------------------------------------------------------
def content_fingerprint(content: str) -> str:
    """Stable content identity for cross-source corroboration — the SAME
    normalization sharing.py uses for merge fusion."""
    return hashlib.sha1(content.strip().lower().encode()).hexdigest()[:16]


def is_corroborated(node) -> bool:
    """A node is corroborated if: (a) it carries the explicit flag set by the
    cross-source bumper, or (b) it carries a verified Helixa stamp, or
    (c) its agent_id encodes a fusion (a+b)."""
    if node.meta.get("corroborated"):
        return True
    stamp = node.meta.get("helixa_stamp") or {}
    if isinstance(stamp, dict) and stamp.get("verified"):
        return True
    if node.agent_id and "+" in node.agent_id:
        return True
    return False


def corroboration_bump(trust_a: float, trust_b: float) -> float:
    """Combined trust for two independent confirmations of the same fact —
    same math as peer fusion: 1-(1-t_a)(1-t_b)."""
    return round(min(1.0, 1.0 - (1.0 - trust_a) * (1.0 - trust_b)), 4)
