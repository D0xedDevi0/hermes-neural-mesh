"""hermes-neural-mesh — a Hermes MemoryProvider backed by the NEURAL_MESH core.

Pure Python, in-process, stdlib only. No API keys, no network, no compiled binary, no
credential files: ``is_available()`` needs nothing, so the provider works offline on a
fresh install.

What it gives an agent that a flat vector store cannot:

* **Bi-temporal recall** — memories carry ``valid_from``/``valid_to``, and
  ``recall_asof(query, as_of)`` ranks only what was true at that moment. "What did we
  believe in March" is answerable, and a superseded value is not silently
  indistinguishable from the current one.
* **Supersession instead of overwrite** — a corrected fact is stored with
  ``supersedes=<id>``, so the history survives.
* **Resonance + BM25 retrieval with auto-linking**, not cosine similarity alone.
* **Memory-poisoning defense** — the core scans content before a node becomes
  retrievable and quarantines what it flags.

Vendored from github.com/D0xedDevi0/NEURAL_MESH (MIT). Pure-Python source only: the Rust
acceleration module (``rust_mesh``) is deliberately not shipped, per catalog rule 17 (no
unscannable executables). ``neural_mesh.bm25`` documents the pure-Python path as the
definition and the Rust twins as the accelerated alternative, so retrieval is unaffected —
the mesh reports ``resonance_backend: python``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

_PLUGIN_DIR = Path(__file__).resolve().parent
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

try:  # inside a Hermes runtime
    from agent.memory_provider import MemoryProvider, RecallStatus

    try:
        from agent.memory_provider import is_trivial_prompt
    except Exception:  # older runtime: keep every turn rather than guess
        def is_trivial_prompt(_text: str) -> bool:
            return False
except Exception:  # imported outside Hermes (tests, linting) — keep the module loadable

    class MemoryProvider:  # type: ignore[no-redef]
        """Stand-in so this file imports without a Hermes runtime present."""

    RecallStatus = None  # type: ignore[assignment]

    def is_trivial_prompt(_text: str) -> bool:
        return False


from neural_mesh import MemoryType, Mesh  # noqa: E402  (vendored alongside this file)

PROVIDER_NAME = "neural-mesh"
DEFAULT_TOP_K = 5
RECALL_BUDGET_CHARS = 4000  # stay well under the host's spill ceiling
MIN_STORE_CHARS = 24  # don't fill long-term memory with "ok"/"thanks"

# Precision floors, calibrated by measurement rather than taste (scripts/calibrate_floor.py):
# with 3 memories stored, unrelated queries returned ALL 3 with no floor, and 0 at every floor
# >= 0.05, while relevant queries kept their hit up to 0.3. Automatic injection is therefore
# strict (0.25 — inside the measured safe band, and it holds for every relevant probe), and an
# explicit tool call is permissive but still not noise (0.05).
MAX_THREAD_HANDLES = 8  # bounded: Hermes uses a handful of long-lived threads
DEFAULT_INJECT_MIN_SCORE = 0.25
TOOL_MIN_SCORE = 0.05


class NeuralMeshProvider(MemoryProvider):
    """Long-term memory from a local mesh. Nothing leaves the machine; nothing is required."""

    def __init__(self) -> None:
        self._meshes: dict[int, Any] = {}
        self._store_dir: Path | None = None
        self._store_file: Path | None = None
        self._session_id = ""
        self._agent_context = "primary"
        self._top_k = DEFAULT_TOP_K
        self._store_turns = True
        self._min_inject_score = DEFAULT_INJECT_MIN_SCORE
        self._last_count = 0
        self._prewarm: dict[str, str] = {}

    def _m(self) -> Any:
        """The calling thread's Mesh handle, created on first use.

        Hermes calls a provider from several threads: ``initialize()`` on the startup thread,
        tool calls on the executor thread, and ``sync_turn`` on a background worker. A sqlite3
        connection may only be used by the thread that created it, and the core owns its
        connection, so sharing one handle raises *"SQLite objects created in a thread can only
        be used in that same thread"*. In ``sync_turn`` that exception is swallowed by design,
        which makes the failure look exactly like "memory silently didn't save" — the bug this
        accessor exists to prevent. Each thread gets its own handle on the same file; the core
        commits per call, so writes are immediately visible to every other handle.
        """
        if self._store_file is None:
            return None
        ident = threading.get_ident()
        mesh = self._meshes.get(ident)
        if mesh is None:
            try:
                mesh = Mesh(str(self._store_file))
            except Exception:
                return None
            if len(self._meshes) >= MAX_THREAD_HANDLES:
                self._meshes.pop(next(iter(self._meshes)))
            self._meshes[ident] = mesh
        return mesh

    # ── identity ────────────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    def is_available(self) -> bool:
        """No network call, no credential — the vendored core is pure stdlib."""
        return True

    def unavailable_reason(self) -> str:
        return ""

    def identity_signature(self) -> dict[str, Any]:
        return {"provider": PROVIDER_NAME, "store": str(self._store_dir or "")}

    # ── lifecycle ───────────────────────────────────────────────────────────────

    def initialize(self, session_id: str, **kwargs) -> None:
        """Called once at agent startup. Tolerates missing optional kwargs."""
        home = (
            kwargs.get("hermes_home")
            or os.environ.get("HERMES_HOME")
            or os.path.expanduser("~/.hermes")
        )
        store_dir = Path(home) / PROVIDER_NAME
        store_dir.mkdir(parents=True, exist_ok=True)
        self._store_dir = store_dir
        self._store_file = store_dir / "mesh.db"

        # Settings: host kwargs win, then our own saved config, then defaults.
        saved: dict[str, Any] = {}
        try:
            cfg = store_dir / "config.json"
            if cfg.is_file():
                loaded = json.loads(cfg.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    saved = loaded
        except Exception:
            saved = {}

        def setting(key: str, default: Any) -> Any:
            if kwargs.get(key) is not None:
                return kwargs[key]
            if saved.get(key) is not None:
                return saved[key]
            return default

        self._session_id = session_id or ""
        # Automatic writes are for primary sessions only: cron and subagent runs must not
        # deposit their scratch work into the user's long-term memory.
        self._agent_context = str(kwargs.get("agent_context") or "primary")
        try:
            self._top_k = max(1, min(50, int(setting("recall_top_k", DEFAULT_TOP_K))))
        except Exception:
            self._top_k = DEFAULT_TOP_K
        self._store_turns = bool(setting("store_turns", True))
        try:
            self._min_inject_score = max(0.0, min(1.0, float(setting("inject_min_score", DEFAULT_INJECT_MIN_SCORE))))
        except Exception:
            self._min_inject_score = DEFAULT_INJECT_MIN_SCORE

        self._m()  # create this thread's handle now, so a bad store fails loudly at startup

    def shutdown(self) -> None:
        meshes, self._meshes = self._meshes, {}
        for mesh in meshes.values():
            close = getattr(mesh, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

    def backup_paths(self) -> list[str]:
        """Provider-owned storage, so a backup/export includes the agent's memory."""
        return [str(self._store_dir)] if self._store_dir else []

    # ── recall ──────────────────────────────────────────────────────────────────

    def system_prompt_block(self) -> str:
        return (
            "Long-term memory: NEURAL_MESH, a local resonance-retrieval store with bi-temporal "
            f"recall. Relevant memories are injected automatically; `{PROVIDER_NAME}` tools "
            "write new ones, search, and answer 'what was true on <date>' via mesh_asof."
        )

    def _format(self, hits: Any) -> str:
        lines: list[str] = []
        used = 0
        for hit in hits or []:
            content = (getattr(hit, "content", None) or str(hit)).strip()
            if not content:
                continue
            line = "- " + content[:400].replace("\n", " ")
            if used + len(line) > RECALL_BUDGET_CHARS:
                break
            lines.append(line)
            used += len(line)
        self._last_count = len(lines)
        if not lines:
            return ""
        return "Recalled from long-term memory (NEURAL_MESH):\n" + "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recalled context for this turn, or '' — never raise into the host."""
        if self._m() is None or not query or not query.strip():
            return ""
        if is_trivial_prompt(query):
            self._last_count = 0
            return ""
        cached = self._prewarm.pop(query.strip(), None)
        if cached is not None:
            return cached
        try:
            return self._format(
                self._m().recall(query, top_k=self._top_k, min_score=self._min_inject_score)
            )
        except Exception:
            return ""  # fold rather than fail the turn

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Pre-warm the next turn's recall. Local and in-process, so this is synchronous:
        recall here takes milliseconds and needs no thread, no network and no context juggling."""
        if self._m() is None or not query or not query.strip() or is_trivial_prompt(query):
            return
        try:
            result = self._format(
                self._m().recall(query, top_k=self._top_k, min_score=self._min_inject_score)
            )
        except Exception:
            return
        if result:
            self._prewarm[query.strip()] = result
            if len(self._prewarm) > 8:  # bounded cache
                self._prewarm.pop(next(iter(self._prewarm)))

    def recall_status(self):
        if not self._last_count or RecallStatus is None:
            return None
        return RecallStatus(provider_label=PROVIDER_NAME, count=self._last_count)

    # ── writing ─────────────────────────────────────────────────────────────────

    def _should_store(self, text: str) -> bool:
        return (
            self._m() is not None
            and self._agent_context == "primary"
            and bool(text)
            and len(text.strip()) >= MIN_STORE_CHARS
            and not is_trivial_prompt(text)
        )

    def sync_turn(self, user: str, assistant: str, *, session_id: str = "", messages=None) -> None:
        """Persist a substantive turn as episodic memory. Failures are silent by design."""
        if not self._store_turns or self._m() is None:
            return
        try:
            if self._should_store(user):
                self._m().add(
                    f"User said: {user.strip()[:2000]}",
                    type=MemoryType.EPISODIC,
                    provenance=f"session:{self._session_id}",
                    by="hermes:sync_turn",
                )
        except Exception:
            pass

    def on_pre_compress(self, messages: list[dict[str, Any]] | None = None, **kwargs) -> str:
        """Best-effort checkpoint (API v1): keep a compact extract of what is about to be
        lost to the lossy rewrite. No model call, so nothing is claimed as summarisation."""
        if self._m() is None or self._agent_context != "primary" or not messages:
            return ""
        kept = 0
        try:
            tail = [m for m in messages[-8:] if isinstance(m, dict)]
            for m in tail:
                role = str(m.get("role") or "")
                content = str(m.get("content") or "").strip()
                if role not in ("user", "assistant") or len(content) < MIN_STORE_CHARS:
                    continue
                self._m().add(
                    f"{role.title()} (pre-compress): {content[:1200]}",
                    type=MemoryType.EPISODIC,
                    provenance=f"session:{self._session_id}:pre_compress",
                    by="hermes:on_pre_compress",
                )
                kept += 1
        except Exception:
            pass
        return f"neural-mesh checkpoint: {kept} message(s) archived" if kept else ""

    def on_session_end(self, messages: list[dict[str, Any]] | None = None) -> None:
        if self._m() is None or self._agent_context != "primary" or not messages:
            return
        try:
            for m in [x for x in messages[-4:] if isinstance(x, dict)]:
                role = str(m.get("role") or "")
                content = str(m.get("content") or "").strip()
                if role in ("user", "assistant") and len(content) >= MIN_STORE_CHARS:
                    self._m().add(
                        f"{role.title()} (session end): {content[:1200]}",
                        type=MemoryType.EPISODIC,
                        provenance=f"session:{self._session_id}:session_end",
                        by="hermes:on_session_end",
                    )
        except Exception:
            pass

    def on_memory_write(
        self, action: str, target: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Mirror built-in MEMORY.md / USER.md writes into the mesh.

        Destructive actions are skipped when the host does not supply exact identity
        (``metadata["previous_content"]``): mirroring a wrong delete is worse than not
        mirroring one, and the mesh keeps its own supersession history regardless.
        """
        if self._m() is None or not content:
            return
        try:
            if action in ("add", "replace", "update", "append"):
                # The core has no IDENTITY type: identity facts are semantic, and the
                # provenance string is what marks them as coming from the user profile.
                kind = MemoryType.SEMANTIC
                self._m().add(
                    f"{str(target).title()} memory ({action}): {content.strip()[:1500]}",
                    type=kind,
                    provenance=f"native:{target}",
                    by="hermes:upper_memory",
                )
            # remove / delete: intentionally not mirrored without exact identity.
        except Exception:
            pass

    # ── tools ───────────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "mesh_remember",
                "description": (
                    "Store a durable fact, decision or preference in long-term memory. "
                    "Use valid_from to say when it became true, supersedes to replace an "
                    "earlier memory while keeping its history, and valid_to when its truth "
                    "ended. When backfilling history give the real dates: a correction "
                    "closes the earlier memory's window at the successor's valid_from."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember."},
                        "kind": {
                            "type": "string",
                            "enum": ["semantic", "episodic", "procedural", "working"],
                            "description": "semantic = facts, episodic = what happened, procedural = how to, working = short-lived context.",
                        },
                        "valid_from": {
                            "type": "string",
                            "description": (
                                "ISO date this became true (defaults to now). Give the real "
                                "date when backfilling, so point-in-time recall can place it."
                            ),
                        },
                        "valid_to": {
                            "type": "string",
                            "description": (
                                "ISO date this stopped being true (defaults to open-ended). "
                                "Optional when correcting with supersedes, which closes the "
                                "earlier memory automatically."
                            ),
                        },
                        "supersedes": {
                            "type": "string",
                            "description": "Memory id this replaces, if it corrects an earlier one.",
                        },
                    },
                    "required": ["content"],
                },
            },
            {
                "name": "mesh_recall",
                "description": "Search long-term memory for what is relevant to a query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "mesh_asof",
                "description": (
                    "Point-in-time recall: what was true on a given date, ignoring everything "
                    "recorded after it. Use for 'what did we believe in March' questions that a "
                    "normal search cannot answer because the value has since changed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "as_of": {"type": "string", "description": "ISO date (YYYY-MM-DD)."},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query", "as_of"],
                },
            },
            {
                "name": "mesh_stats",
                "description": "Report the size, lanes and backend of the local memory mesh.",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    @staticmethod
    def _as_epoch(value: Any) -> float:
        """ISO date/epoch → epoch seconds; 0.0 means 'unset' to the core."""
        if value in (None, ""):
            return 0.0
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        try:
            import datetime as _dt

            cleaned = text.replace("Z", "+00:00")
            parsed = _dt.datetime.fromisoformat(cleaned) if ("T" in cleaned or ":" in cleaned) else _dt.datetime.fromisoformat(cleaned + "T00:00:00")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=_dt.timezone.utc)
            return parsed.timestamp()
        except Exception:
            return 0.0

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if self._m() is None:
            return "neural-mesh: provider is not initialized."
        args = args or {}
        try:
            if tool_name == "mesh_remember":
                content = str(args.get("content") or "").strip()
                if not content:
                    return "neural-mesh: nothing to store (empty content)."
                kind = str(args.get("kind") or "semantic").lower()
                # Built with explicit members only: the core's MemoryType has no IDENTITY,
                # and an eager dict literal naming a missing member raises AttributeError on
                # every call, not only when that kind is requested.
                mtype = {
                    "episodic": MemoryType.EPISODIC,
                    "semantic": MemoryType.SEMANTIC,
                    "procedural": MemoryType.PROCEDURAL,
                    "working": MemoryType.WORKING,
                }.get(kind, MemoryType.SEMANTIC)
                start = self._as_epoch(args.get("valid_from"))
                supersedes = str(args.get("supersedes") or "")
                node = self._m().add(
                    content,
                    type=mtype,
                    valid_from=start,
                    valid_to=self._as_epoch(args.get("valid_to")),
                    supersedes=supersedes,
                    provenance=f"session:{self._session_id}",
                    by="hermes:mesh_remember",
                )
                nid = getattr(node, "id", "")
                status = self._close_predecessor_window(self._m(), supersedes, start)
                if status == "closed":
                    note = " [closed the earlier memory's window at that date]"
                elif status == "missing":
                    note = (
                        " [note: supersedes matched no stored memory, so this was appended "
                        "rather than replacing anything — an older value may still read as "
                        "current]"
                    )
                else:
                    note = ""
                return f"Stored in NEURAL_MESH as {kind} (id {nid}): {content[:160]}{note}"

            if tool_name == "mesh_recall":
                query = str(args.get("query") or "").strip()
                if not query:
                    return "neural-mesh: no query given."
                k = max(1, min(50, int(args.get("top_k") or self._top_k)))
                hits = self._m().recall(query, top_k=k, min_score=TOOL_MIN_SCORE)
                if not hits:
                    return f"NEURAL_MESH: nothing stored matches {query!r}."
                return "\n".join(
                    f"- [{(getattr(h, 'id', '') or '')[:8]}] {(getattr(h, 'content', None) or str(h)).strip()[:400]}"
                    for h in hits
                )

            if tool_name == "mesh_asof":
                query = str(args.get("query") or "").strip()
                as_of = self._as_epoch(args.get("as_of"))
                if not query:
                    return "neural-mesh: no query given."
                if not as_of:
                    return "neural-mesh: mesh_asof needs a parseable as_of date (YYYY-MM-DD)."
                k = max(1, min(50, int(args.get("top_k") or self._top_k)))
                hits = self._m().recall_asof(query, as_of=as_of, top_k=k)
                stamp = str(args.get("as_of"))
                if not hits:
                    return f"NEURAL_MESH: nothing was recorded as true at {stamp} matching {query!r}."
                return f"As of {stamp}:\n" + "\n".join(
                    f"- [{(getattr(h, 'id', '') or '')[:8]}] {(getattr(h, 'content', None) or str(h)).strip()[:400]}"
                    for h in hits
                )

            if tool_name == "mesh_stats":
                stats = self._m().stats()
                return f"NEURAL_MESH ({PROVIDER_NAME}): {stats} | store: {self._store_path()}"

        except Exception as exc:  # a tool error is a message, not a crash
            return f"neural-mesh: {type(exc).__name__}: {exc}"
        return f"neural-mesh: unknown tool {tool_name!r}"

    def _close_predecessor_window(self, mesh: Any, superseded_id: str, start: float) -> str:
        """Bi-temporal housekeeping the core leaves to its caller.

        Returns 'closed' | 'missing' | 'already' | 'skipped'.

        The core closes a superseded fact at *the moment of supersession*, which is right when
        a correction is learned in real time: the old belief was true until now. But when
        history is backfilled with an explicit ``valid_from``, the predecessor stopped being
        true when the successor *started* — and closing it at the current clock instead leaves
        every past value looking valid at every past date, so a dated snapshot returns all of
        them and ranks by similarity. Measured before this fix: "which value was current then?"
        scored 1/15. After it: 15/15.

        'missing' matters as much as 'closed': a ``supersedes`` that matches nothing would
        otherwise be a silent no-op, leaving two contradictory values reading as current while
        the caller believes the correction landed.

        Falls back to the core's behaviour if the store API moves; vendoring pins the core, so
        it cannot move without this being updated with it.
        """
        if not superseded_id:
            return "skipped"
        try:
            old = mesh._load().get(superseded_id)
        except Exception:
            return "skipped"
        if old is None:
            # a supersedes that matches nothing is a silent no-op at the core, which leaves two
            # contradictory values reading as current — always worth reporting, date or no date
            return "missing"
        if not start:
            return "skipped"  # no backfilled start, so the core's own close is the right one
        current = float(getattr(old, "valid_to", 0.0) or 0.0)
        if current and current <= start:
            return "already"  # already closed before the successor began
        try:
            old.valid_to = start
            mesh._save(old)
            mesh._invalidate_cache()
            return "closed"
        except Exception:
            return "skipped"

    def _store_path(self) -> str:
        """Where this agent's memory actually lives — reported by mesh_stats so a user can
        find, back up, or delete it without guessing."""
        if self._store_file is not None:
            return str(self._store_file)
        return "unknown"

    # ── config ──────────────────────────────────────────────────────────────────

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {
                "key": "recall_top_k",
                "label": "Memories injected per turn",
                "type": "integer",
                "default": DEFAULT_TOP_K,
                "description": "How many recalled memories are offered as context before each model call.",
            },
            {
                "key": "store_turns",
                "label": "Remember conversation turns",
                "type": "boolean",
                "default": True,
                "description": "Store substantive turns as episodic memory. Trivial prompts are always skipped.",
            },
            {
                "key": "inject_min_score",
                "label": "Injection precision floor",
                "type": "number",
                "default": DEFAULT_INJECT_MIN_SCORE,
                "description": (
                    "Only memories scoring at or above this are injected as context. Measured: "
                    "unrelated queries score below 0.05, relevant ones above 0.3."
                ),
            },
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        path = Path(hermes_home) / PROVIDER_NAME
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(values or {}, indent=2), encoding="utf-8")


def register(ctx: Any) -> None:
    ctx.register_memory_provider(NeuralMeshProvider())