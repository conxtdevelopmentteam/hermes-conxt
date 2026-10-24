"""Conxt memory provider for Hermes Agent.

Conxt (https://conxt.dev) is governed memory: what an agent learns is kept per
person, shared with a team only after a person on that team approves it, and
every read is on record. This provider connects a Hermes profile to it.

What it does, and when:

- Before each turn, recall. The user's message goes to Conxt's /context and
  the memories that match come back as context for the turn: the person's own,
  plus the team's approved memories when a team is configured.
- After the conversation, capture. Turns are buffered on the instance and sent
  to /capture every 5 exchanges, at the session's end, before compression,
  when the session switches and at shutdown, in batches Conxt accepts. Each
  stored batch is then handed to /extract, which turns it into memories.
- On request, three tools: conxt_search, conxt_remember (a personal memory) and
  conxt_propose (a team memory, which waits for a person's approval in Conxt;
  an API key cannot approve its own proposal).
- When Hermes's built-in memory tool adds an entry, the same text is saved to
  Conxt as a personal memory.

Nothing is written automatically from cron, subagent or flush runs. No call
waits for a person, so unattended runs never hang: a slow or failed request is
logged and skipped.

Config: CONXT_API_KEY (required, a scoped secret); CONXT_TEAM_ID and
CONXT_BASE_URL (optional; env first, then $HERMES_HOME/conxt.json).
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread
from agent.secret_scope import get_secret

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://mcp.conxt.dev"
CONFIG_FILE = "conxt.json"
SOURCE = "hermes"

# Hermes abandons a prefetch after 8s. 5s was too tight: the first recall of a
# live session on 24 Oct took longer and the turn went without memory.
RECALL_TIMEOUT_S = 7.0
WRITE_TIMEOUT_S = 20.0
# Extraction runs an LLM and takes 9-45s. Conxt finishes a request after the
# client hangs up, so the provider asks and does not wait for the answer.
EXTRACT_WAIT_S = 3.0
RECALL_LIMIT = 8
QUERY_CHARS = 500        # Conxt embeds only the first 500 characters.

# Conxt's /capture limits (conxt-engine routers/capture.py). A batch over them
# is refused whole, so batches are cut to fit.
CAPTURE_MAX_TURNS = 50
CAPTURE_MAX_TURN_CHARS = 10_000
CAPTURE_MAX_TOTAL_CHARS = 100_000
# Every 5 exchanges. At 40, a short conversation lived only in this process
# until the session ended, and closing the terminal window lost it.
FLUSH_EVERY_MESSAGES = 10
BUFFER_CAP_MESSAGES = 400  # while Conxt is unreachable; oldest dropped first

MEMORY_TYPES = ("decision", "preference", "entity", "open_question", "coding_rule", "tool_choice", "workflow")
_UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def _error(message: str) -> str:
    return json.dumps({"error": message})


def _read_config(home: Path) -> dict[str, Any]:
    try:
        data = json.loads((home / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


class ConxtError(RuntimeError):
    """A request Conxt refused or could not answer, with its reason."""

    def __init__(self, message: str, status: int = 0, sent: bool = False):
        super().__init__(message)
        self.status = status
        self.sent = sent  # the request reached Conxt; only the answer is missing


class _Client:
    """The Conxt routes this provider uses, over HTTPS with the API key."""

    def __init__(self, api_key: str, base_url: str):
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, body: dict[str, Any], timeout: float) -> Any:
        import requests

        try:
            resp = requests.post(
                f"{self.base_url}{path}", json=body, timeout=timeout,
                headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": "hermes-conxt"},
            )
        except requests.ReadTimeout as exc:
            raise ConxtError(f"Conxt did not answer {path} in time", sent=True) from exc
        except requests.RequestException as exc:
            raise ConxtError(f"Conxt could not be reached: {exc.__class__.__name__}") from exc
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        if not resp.ok:
            detail = payload.get("detail") if isinstance(payload, dict) else None
            raise ConxtError(f"Conxt refused {path} ({resp.status_code}): {detail or resp.reason}", resp.status_code)
        return payload

    def recall(self, query: str, team_id: Optional[str], limit: int = RECALL_LIMIT,
               timeout: float = RECALL_TIMEOUT_S) -> dict[str, Any]:
        body: dict[str, Any] = {"session_opener": query[:QUERY_CHARS], "limit": limit}
        if team_id:
            body["team_id"] = team_id
        return self._post("/context/", body, timeout) or {}

    def capture(self, turns: list[dict[str, Any]], timeout: float = WRITE_TIMEOUT_S) -> dict[str, Any]:
        return self._post("/capture/", {"source": SOURCE, "raw_turns": turns}, timeout) or {}

    def extract(self, session_id: str, timeout: float = EXTRACT_WAIT_S) -> None:
        """Ask Conxt to turn a stored capture into memories. /capture only
        stores; without this call a conversation never becomes memory. An
        answer that does not arrive in time is fine: the request was sent,
        and Conxt also retries sessions it holds unextracted."""
        try:
            self._post("/extract/", {"session_id": session_id, "source": SOURCE}, timeout)
        except ConxtError as exc:
            if not exc.sent:
                raise

    def remember(self, content: str, memory_type: str) -> dict[str, Any]:
        # /import is Conxt's personal write: it redacts secrets and reinforces
        # a memory it already holds instead of duplicating it. dry_run defaults
        # to true there, so it is set explicitly.
        body = {"payload": {"format": "conxt-memory", "memories": [{"type": memory_type, "content": content}]},
                "dry_run": False}
        return self._post("/import/", body, WRITE_TIMEOUT_S) or {}

    def propose(self, team_id: str, content: str, memory_type: str, title: str) -> dict[str, Any]:
        # A write to a team with an API key is stored as pending; a person on
        # the team approves it before it is recalled for anyone.
        body = {"type": memory_type, "title": title, "content": content, "source": SOURCE, "tags": []}
        return self._post(f"/teams/{team_id}/memories/create", body, WRITE_TIMEOUT_S) or {}


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


_TYPE_PROP = {"type": "string", "enum": list(MEMORY_TYPES),
              "description": "What kind of memory this is (default: decision)."}

TOOL_SCHEMAS = [
    _schema("conxt_search",
            "Search Conxt memory for what is known about a topic: the user's own memories, plus the "
            "team's approved memories when a team is configured.",
            {"query": {"type": "string", "description": "What to look for."}}, ["query"]),
    _schema("conxt_remember",
            "Save one fact, decision or preference to the user's personal Conxt memory. Secrets are "
            "redacted before storage. Use for things worth knowing in future sessions.",
            {"content": {"type": "string", "description": "The memory, as one self-contained statement."},
             "memory_type": _TYPE_PROP}, ["content"]),
    _schema("conxt_propose",
            "Propose a memory to the configured Conxt team. It is not shared until a person on the team "
            "approves it in Conxt. Use for decisions or rules the whole team should follow.",
            {"content": {"type": "string", "description": "The memory, as one self-contained statement."},
             "memory_type": _TYPE_PROP,
             "title": {"type": "string", "description": "A short title (optional)."}}, ["content"]),
]


class ConxtMemoryProvider(MemoryProvider):
    """Governed memory from Conxt: recall before a turn, capture after, team writes as proposals."""

    def __init__(self) -> None:
        self._client: Optional[_Client] = None
        self._team_id: Optional[str] = None
        self._writes = False
        self._session_id = ""
        self._lock = threading.Lock()
        # One flush at a time. Shutdown flushed while the session-end flush was
        # still sending, and the same batch went to Conxt twice.
        self._flush_lock = threading.Lock()
        self._buffer: list[dict[str, Any]] = []
        self._flusher: Optional[threading.Thread] = None
        self._last_count: Optional[int] = None

    @property
    def name(self) -> str:
        return "conxt"

    # -- setup ---------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(get_secret("CONXT_API_KEY"))

    def unavailable_reason(self) -> str:
        return "Set CONXT_API_KEY (create one at https://conxt.dev) or run `hermes memory setup`."

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [
            {"key": "api_key", "description": "Conxt API key", "secret": True, "required": True,
             "env_var": "CONXT_API_KEY", "url": "https://conxt.dev"},
            {"key": "team_id", "description": "Conxt team id, for team recall and proposals (optional)",
             "default": ""},
        ]

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        keep = {k: str(v).strip() for k, v in values.items() if k in ("team_id", "base_url") and v is not None}
        path = Path(hermes_home) / CONFIG_FILE
        try:
            from utils import atomic_json_write
            atomic_json_write(path, keep)
        except ImportError:
            path.write_text(json.dumps(keep, indent=2), encoding="utf-8")

    def initialize(self, session_id: str, **kwargs) -> None:
        home = Path(kwargs.get("hermes_home") or ".")
        cfg = _read_config(home)
        base_url = (get_secret("CONXT_BASE_URL", "") or str(cfg.get("base_url") or "") or DEFAULT_BASE_URL).strip()
        if not base_url.startswith("https://"):
            logger.warning("Conxt: base URL must be https; using %s", DEFAULT_BASE_URL)
            base_url = DEFAULT_BASE_URL
        team = (get_secret("CONXT_TEAM_ID", "") or str(cfg.get("team_id") or "")).strip()
        if team and not _UUID.match(team):
            logger.warning("Conxt: team id %r is not a Conxt team id; team recall and proposals are off", team)
            team = ""
        self._client = _Client(get_secret("CONXT_API_KEY", "") or "", base_url)
        self._team_id = team or None
        self._session_id = session_id
        self._writes = (kwargs.get("agent_context") or "primary") == "primary"

    def system_prompt_block(self) -> str:
        team = ("Team memory is on: conxt_propose sends a memory to the team, where a person approves it "
                "before anyone recalls it.") if self._team_id else "No team is configured, so conxt_propose is off."
        return ("# Conxt memory\nRelevant memories are recalled before each turn. Use conxt_search to look "
                "further and conxt_remember to keep something for future sessions. " + team)

    # -- recall --------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_count = None
        if not self._client or is_trivial_prompt(query):
            return ""
        try:
            result = self._client.recall(query, self._team_id)
        except ConxtError as exc:
            logger.info("Conxt recall skipped: %s", exc)
            return ""
        block = str(result.get("context_block") or "").strip()
        self._last_count = int(result.get("memory_count") or 0) + int(result.get("pinned_count") or 0)
        return block

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_count:
            return None
        return RecallStatus(provider_label="Conxt", count=self._last_count)

    # -- capture -------------------------------------------------------------

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "",
                  messages: Optional[list[dict[str, Any]]] = None, **kwargs) -> None:
        if not self._writes or not self._client:
            return
        now = int(time.time() * 1000)
        turns = [{"role": "user", "content": user_content or "", "ts": now},
                 {"role": "assistant", "content": assistant_content or "", "ts": now}]
        with self._lock:
            self._buffer.extend(t for t in turns if t["content"].strip())
            if len(self._buffer) > BUFFER_CAP_MESSAGES:
                del self._buffer[: len(self._buffer) - BUFFER_CAP_MESSAGES]
            due = len(self._buffer) >= FLUSH_EVERY_MESSAGES
        if due:
            self._flush_in_background()

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        self._flush_in_background()

    def on_pre_compress(self, messages: list[dict[str, Any]]) -> str:
        self._flush_in_background()
        return ""

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        self._flush_in_background()
        self._session_id = new_session_id

    def shutdown(self) -> None:
        if self._flusher and self._flusher.is_alive():
            self._flusher.join(timeout=2.0)
        # Hermes gives shutdown about 30s in all, shared with everything else,
        # so the last batch gets short timeouts.
        self.flush(timeout=3.0, wait=1.0)

    def _flush_in_background(self) -> None:
        if not self._writes or not self._buffer:
            return
        if self._flusher and self._flusher.is_alive():
            return  # the running flush takes whatever is buffered when it starts its next batch
        self._flusher = spawn_context_thread(self.flush, name="conxt-capture")
        self._flusher.start()

    def flush(self, timeout: float = WRITE_TIMEOUT_S, wait: Optional[float] = None) -> None:
        """Send buffered turns to Conxt in batches it accepts, and ask for each
        stored batch to be extracted. A batch is removed from the buffer once
        Conxt has it, or when Conxt refuses it as invalid; if Conxt cannot be
        reached, it stays for the next flush. If another flush is running and
        does not finish within `wait` seconds, this one sends nothing."""
        if not self._client:
            return
        if not self._flush_lock.acquire(timeout=timeout if wait is None else wait):
            return
        try:
            self._flush_batches(timeout)
        finally:
            self._flush_lock.release()

    def _flush_batches(self, timeout: float) -> None:
        assert self._client is not None
        while True:
            with self._lock:
                batch = _next_batch(self._buffer)
            if not batch:
                return
            stored = None
            try:
                stored = self._client.capture(batch, timeout=timeout)
            except ConxtError as exc:
                if 400 <= exc.status < 500 and exc.status not in (401, 403, 408, 429):
                    logger.warning("Conxt refused a capture; it was dropped: %s", exc)
                else:
                    logger.info("Conxt capture postponed: %s", exc)
                    return
            with self._lock:
                del self._buffer[: len(batch)]
            if stored and stored.get("status") == "captured" and stored.get("session_id"):
                logger.info("Conxt: sent %d messages", len(batch))
                try:
                    self._client.extract(str(stored["session_id"]), timeout=min(timeout, EXTRACT_WAIT_S))
                except ConxtError as exc:
                    logger.info("Conxt extraction not requested; a later one retries it: %s", exc)

    # -- built-in memory mirror ----------------------------------------------

    def on_memory_write(self, action: str, target: str, content: str,
                        metadata: Optional[dict[str, Any]] = None) -> None:
        if action != "add" or not content or not content.strip() or not self._writes or not self._client:
            return
        memory_type = "preference" if target == "user" else "decision"
        client = self._client

        def _mirror() -> None:
            try:
                client.remember(content.strip()[:CAPTURE_MAX_TURN_CHARS], memory_type)
            except ConxtError as exc:
                logger.info("Conxt mirror of a memory write skipped: %s", exc)

        spawn_context_thread(_mirror, name="conxt-mirror").start()

    # -- tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [s for s in TOOL_SCHEMAS if s["name"] != "conxt_propose" or self._team_id]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._client:
            return _error("Conxt is not initialized.")
        try:
            return json.dumps(self._dispatch(tool_name, args or {}))
        except ConxtError as exc:
            return _error(str(exc))

    def _dispatch(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert self._client is not None
        if tool_name == "conxt_search":
            query = str(args.get("query") or "").strip()
            if not query:
                return {"error": "query is required"}
            result = self._client.recall(query, self._team_id, limit=12, timeout=WRITE_TIMEOUT_S)
            return {"context": result.get("context_block") or "",
                    "memories": int(result.get("memory_count") or 0),
                    "pinned": int(result.get("pinned_count") or 0)}

        content = str(args.get("content") or "").strip()
        if not content:
            return {"error": "content is required"}
        if len(content) > CAPTURE_MAX_TURN_CHARS:
            return {"error": f"content is longer than {CAPTURE_MAX_TURN_CHARS} characters"}
        memory_type = str(args.get("memory_type") or "decision")
        if memory_type not in MEMORY_TYPES:
            return {"error": f"memory_type must be one of: {', '.join(MEMORY_TYPES)}"}

        if tool_name == "conxt_remember":
            report = self._client.remember(content, memory_type)
            return {"saved": bool(report.get("imported") or report.get("reinforced")),
                    "already_known": bool(report.get("reinforced")),
                    "redacted": bool(report.get("redacted")),
                    "message": report.get("message") or ""}

        if tool_name == "conxt_propose":
            if not self._team_id:
                return {"error": "No Conxt team is configured, so there is nowhere to propose to."}
            title = str(args.get("title") or "").strip() or content.splitlines()[0]
            row = self._client.propose(self._team_id, content, memory_type, title[:120])
            status = row.get("status") or "pending"
            return {"proposed": True, "status": status, "memory_id": row.get("id"),
                    "message": ("Waiting for a person on the team to approve it in Conxt; it is not shared "
                                "until then.") if status == "pending" else f"Stored as {status}."}

        return {"error": f"Unknown tool: {tool_name}"}


def _next_batch(buffer: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The longest prefix of the buffer within Conxt's capture limits, each
    message cut to the per-turn limit. Called under the provider's lock."""
    batch: list[dict[str, Any]] = []
    total = 0
    for turn in buffer[:CAPTURE_MAX_TURNS]:
        text = turn["content"][:CAPTURE_MAX_TURN_CHARS]
        if batch and total + len(text) > CAPTURE_MAX_TOTAL_CHARS:
            break
        batch.append({**turn, "content": text})
        total += len(text)
    return batch


def register(ctx) -> None:
    ctx.register_memory_provider(ConxtMemoryProvider())
