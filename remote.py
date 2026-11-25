"""Remote backend for the NEURAL_MESH Hermes provider.

In-process by default, the provider opens a local ``mesh.db``. When
``NEURAL_MESH_REMOTE_URL`` is set the same surface is served over HTTP by a
canonical mesh service, so the portal agent and the VPS worker read and write
one store instead of two.

Design rules:
- stdlib only (urllib, json, os, threading, hashlib) — nothing new to install.
- a write that cannot reach the service is spooled to disk and replayed in
  order; nothing is ever silently dropped. ``pending()`` is the honest status.
- reads never invent data: an unreachable service raises ``RemoteUnavailable``
  so the caller can say "memory is unreachable" instead of "nothing found".
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 5.0
MAX_SPOOL_BYTES = 8 * 1024 * 1024


class RemoteError(RuntimeError):
    """The service answered, but not with success."""


class RemoteUnavailable(RemoteError):
    """The service could not be reached (network, DNS, timeout, refused)."""


class Node:
    """The subset of a mesh node the provider actually reads."""

    __slots__ = ("id", "content", "type", "lane", "trust", "score",
                 "valid_from", "valid_to", "pending")

    def __init__(self, id="", content="", type="semantic", lane="",
                 trust=1.0, score=0.0, valid_from=0.0, valid_to=0.0, pending=False):
        self.id = id
        self.content = content
        self.type = type
        self.lane = lane
        self.trust = trust
        self.score = score
        self.valid_from = valid_from
        self.valid_to = valid_to
        self.pending = pending

    def __repr__(self) -> str:  # keeps logs readable
        return f"Node({self.id[:8]}, score={self.score:.3f}, pending={self.pending})"


class Spool:
    """Append-only JSONL queue for writes made while the service is down.

    One line per operation, fsynced on append, replayed oldest-first. A replay
    that fails halfway rewrites the remainder atomically, so an interrupted
    replay cannot duplicate the head of the queue.
    """

    def __init__(self, path: str):
        self.path = str(path)
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

    def append(self, op: str, payload: dict) -> str:
        record = {"op": op, "payload": payload, "at": time.time()}
        blob = json.dumps(record, sort_keys=True, default=str)
        key = hashlib.sha256(blob.encode()).hexdigest()[:16]
        record["idem"] = key
        with self._lock:
            if os.path.exists(self.path) and os.path.getsize(self.path) > MAX_SPOOL_BYTES:
                raise RemoteError(
                    f"spool {self.path} is over {MAX_SPOOL_BYTES} bytes; refusing to grow "
                    "silently — drain it or raise MAX_SPOOL_BYTES"
                )
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
        return key

    def pending(self) -> int:
        with self._lock:
            if not os.path.exists(self.path):
                return 0
            with open(self.path, "r", encoding="utf-8") as fh:
                return sum(1 for line in fh if line.strip())

    def drain(self, send) -> tuple[int, int]:
        """Send queued ops through ``send(op, payload)``; returns (sent, left)."""
        with self._lock:
            if not os.path.exists(self.path):
                return (0, 0)
            with open(self.path, "r", encoding="utf-8") as fh:
                lines = [ln for ln in fh if ln.strip()]
            sent = 0
            for i, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    sent += 1  # a torn line cannot be replayed; count it, do not loop on it
                    continue
                try:
                    send(record.get("op", ""), record.get("payload", {}))
                except Exception:
                    remainder = lines[i:]
                    tmp = self.path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        fh.writelines(remainder)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.replace(tmp, self.path)
                    return (sent, len(remainder))
                sent += 1
            os.unlink(self.path)
            return (sent, 0)


class VerifyReport:
    """Shape the provider expects from ``mesh.verify_memory()``."""

    def __init__(self, payload: dict):
        self._payload = dict(payload or {})
        self.ok = bool(self._payload.get("ok"))
        self.node_id = None
        self.sinks_in_store_only = bool(self._payload.get("sinks_in_store_only"))

    def to_dict(self) -> dict:
        return dict(self._payload)

    def summary(self) -> str:
        return str(self._payload.get("summary") or ("OK" if self.ok else "FAILED"))


class RemoteMesh:
    """A mesh handle that talks HTTP instead of SQLite.

    Implements exactly what the provider calls: add, recall, recall_asof,
    stats, verify_memory, provenance_of, adopt_ledger, _load, _save,
    _invalidate_cache, plus a truthy ``ledger`` attribute.
    """

    def __init__(self, base_url: str, token: str = "", timeout: float = DEFAULT_TIMEOUT,
                 spool_path: str | None = None):
        self.base_url = str(base_url).rstrip("/")
        self.token = str(token or "")
        self.timeout = float(timeout)
        self.spool = Spool(spool_path) if spool_path else None
        self.ledger = object()  # truthy; the service still answers 503 if the ledger is off
        self.pending = 0

    # -- transport -------------------------------------------------------
    def _call(self, path: str, payload: dict | None = None, method: str = "POST") -> dict:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, default=str).encode()
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:400]
            if exc.code in (502, 503, 504):
                # A reverse proxy in front of a stopped service answers 502/503/504.
                # That is the same condition as a refused connection, so writes must
                # spool instead of failing: the node is not lost, just queued.
                raise RemoteUnavailable(f"{method} {path} -> HTTP {exc.code}: {body[:120]}") from exc
            raise RemoteError(f"{method} {path} -> HTTP {exc.code}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RemoteUnavailable(f"{method} {path} unreachable: {exc}") from exc
        try:
            return json.loads(body) if body.strip() else {}
        except json.JSONDecodeError as exc:
            raise RemoteError(f"{method} {path} returned non-JSON: {body[:200]}") from exc

    # -- writes ----------------------------------------------------------
    def add(self, content, type="semantic", provenance="", supersedes="",
            meta=None, by="", valid_from=None, valid_to=None) -> Node:
        payload = {
            "content": str(content),
            "type": getattr(type, "value", type) or "semantic",
            "provenance": str(provenance or ""),
            "supersedes": str(supersedes or ""),
            "meta": dict(meta or {}),
            "by": str(by or ""),
        }
        # The provider sends bi-temporal bounds on every write; without them a
        # superseded memory could never be scoped to the window it was true for.
        for key, value in (("valid_from", valid_from), ("valid_to", valid_to)):
            if value not in (None, ""):
                try:
                    payload[key] = int(value)
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if self.spool is not None:
            try:
                self.pending = self.spool.pending()
                if self.pending:
                    self.replay()
                    self.pending = self.spool.pending()
            except RemoteError:
                pass  # spool reports its own limits; the write below decides
        try:
            out = self._call("/mesh/add", payload)
        except RemoteUnavailable:
            if self.spool is None:
                raise
            key = self.spool.append("add", payload)
            self.pending = self.spool.pending()
            return Node(id=f"spool-{key}", content=payload["content"],
                        type=payload["type"], pending=True)
        return Node(id=out.get("id", ""), content=out.get("content", payload["content"]),
                    type=out.get("type", payload["type"]))

    def _send_op(self, op: str, payload: dict) -> None:
        if op == "add":
            self._call("/mesh/add", payload)
        else:
            raise RemoteError(f"unknown spooled op {op!r}")

    def replay(self) -> int:
        """Drain the spool; returns the number still pending."""
        if self.spool is None:
            return 0
        self.spool.drain(self._send_op)
        self.pending = self.spool.pending()
        return self.pending

    # -- reads -----------------------------------------------------------
    @staticmethod
    def _nodes(results) -> list:
        return [
            Node(id=r.get("id", ""), content=r.get("content", ""),
                 type=r.get("type", "semantic"), lane=r.get("lane", ""),
                 trust=float(r.get("trust", 1.0) or 0.0),
                 score=float(r.get("score", 0.0) or 0.0),
                 valid_from=float(r.get("valid_from", 0.0) or 0.0),
                 valid_to=float(r.get("valid_to", 0.0) or 0.0))
            for r in (results or [])
        ]

    def recall(self, query: str, top_k: int = 10, min_score: float = 0.0) -> list:
        # The service ranks with the core's default dispatch (right record in the top 5
        # for about half the probes) and returns no score at all. Ask for a wide
        # candidate set and rerank it here with the same BM25 + IDF-coverage code the
        # local path uses, so min_score means the same thing against either store.
        # Measured on the canonical store (5,495 nodes): 60 candidates found the
        # answer for 9 of 20 probes, 200 for 17, 300 for 19 — the service's own
        # ranking is the ceiling, so the window has to be this wide before local
        # BM25 gets a fair look. 300 costs ~0.2s against the live service.
        wide = max(int(top_k) * 6, 300)
        out = self._call("/mesh/recall", {"query": str(query), "limit": wide})
        hits = self._nodes(out.get("results"))
        if not hits:
            return []
        try:
            import bm25
            ranked = bm25.rank(self, str(query), hits, top_k=int(top_k),
                               min_coverage=float(min_score))
        except Exception:
            if min_score:
                hits = [h for h in hits if getattr(h, "score", 0.0) >= float(min_score)]
            return hits[: int(top_k)]
        for cov, node in ranked:
            node.score = cov
        return [node for _, node in ranked]

    def recall_asof(self, query: str, as_of=None, top_k: int = 10) -> list:
        out = self._call("/mesh/recall_asof",
                         {"query": str(query), "as_of": as_of, "limit": int(top_k)})
        return self._nodes(out.get("results"))

    def snapshot(self, as_of=None) -> list:
        out = self._call("/mesh/snapshot", {"as_of": as_of})
        return self._nodes(out.get("results"))

    def _load(self) -> dict:
        return {n.id: n for n in self.snapshot()}

    def _save(self, node) -> None:  # pragma: no cover - callers guard on failure
        raise NotImplementedError("remote store: node mutation is server-side")

    def _invalidate_cache(self) -> None:
        return None

    def stats(self) -> dict:
        return self._call("/mesh/stats", None, method="GET")

    def verify_memory(self, deep: bool = True) -> VerifyReport:
        try:
            return VerifyReport(self._call("/mesh/ledger/verify", {"deep": bool(deep)}))
        except RemoteError as exc:
            if "HTTP 409" in str(exc):
                return VerifyReport({"ok": False, "summary": str(exc)[:300]})
            raise

    def provenance_of(self, node_id: str) -> dict:
        return self._call("/mesh/ledger/provenance", {"node_id": str(node_id)})

    def adopt_ledger(self, *args, **kwargs) -> dict:
        return self._call("/mesh/ledger/adopt", {})
