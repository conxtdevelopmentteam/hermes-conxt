"""The Conxt provider against a stand-in Conxt server.

Each test names a behaviour a user relies on. The stand-in records every
request, so a test can check what left the machine, not only what came back.

Run with Hermes on the path (the provider imports agent.memory_provider):
    PYTHONPATH=/path/to/hermes-agent python -m pytest tests -q
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEAM = "6f1c2a4e-0b7d-4c55-9a8e-3d2f1b0c9e77"


def _load():
    spec = importlib.util.spec_from_file_location("conxt_plugin", ROOT / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["conxt_plugin"] = module
    spec.loader.exec_module(module)
    return module


conxt = _load()


class StandIn:
    """A tiny Conxt: answers the routes the provider uses and records each request."""

    def __init__(self):
        self.requests: list[dict] = []
        self.replies: dict[str, tuple[int, dict]] = {
            "/context/": (200, {"context_block": "- Deploys freeze on Fridays.", "memory_count": 2, "pinned_count": 1}),
            "/capture/": (200, {"status": "captured", "session_id": "s1", "turn_count": 2}),
            "/extract/": (200, {"status": "extracted", "session_id": "s1"}),
            "/import/": (200, {"imported": 1, "reinforced": 0, "redacted": 0, "message": "Imported 1 memory."}),
            f"/teams/{TEAM}/memories/create": (200, {"id": "m1", "status": "pending"}),
        }
        self.delays: dict[str, float] = {}  # seconds a route takes to answer
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                stand_in.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                if self.path in stand_in.delays:
                    time.sleep(stand_in.delays[self.path])
                status, reply, *kind = stand_in.replies.get(self.path, (404, {"detail": "Not Found"}))
                # (status, body) is JSON; (status, text, content type) is sent as given.
                data = reply.encode() if kind else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", kind[0] if kind else "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def paths(self):
        return [r["path"] for r in self.requests]

    def sent(self, path):
        return [r for r in self.requests if r["path"] == path]


@pytest.fixture
def server():
    s = StandIn()
    yield s
    s.server.shutdown()


@pytest.fixture
def secrets(monkeypatch):
    values = {"CONXT_API_KEY": "cx_test_key"}
    monkeypatch.setattr(conxt, "get_secret", lambda name, default=None: values.get(name, default))
    return values


def provider(server, tmp_path, *, team=None, context="primary", secrets=None):
    """A provider initialized as Hermes would, then pointed at the stand-in.
    The stand-in is plain http, which the provider itself refuses, so the
    client is swapped in after initialize."""
    if team:
        secrets["CONXT_TEAM_ID"] = team
    p = conxt.ConxtMemoryProvider()
    p.initialize("sess-1", hermes_home=str(tmp_path), platform="cli", agent_context=context)
    p._client = conxt._Client("cx_test_key", server.url)
    return p


def wait_for_flush(p):
    if p._flusher:
        p._flusher.join(timeout=5)


# -- setup ---------------------------------------------------------------------

def test_it_is_available_only_with_an_api_key(monkeypatch):
    p = conxt.ConxtMemoryProvider()
    monkeypatch.setattr(conxt, "get_secret", lambda name, default=None: default)
    assert not p.is_available()
    monkeypatch.setattr(conxt, "get_secret", lambda name, default=None: "k" if name == "CONXT_API_KEY" else default)
    assert p.is_available()


def test_each_registration_is_a_new_instance_so_profiles_never_share_one():
    seen = []

    class Ctx:
        def register_memory_provider(self, prov):
            seen.append(prov)

    conxt.register(Ctx())
    conxt.register(Ctx())
    assert len(seen) == 2 and seen[0] is not seen[1]


def test_a_base_url_that_is_not_https_is_refused(secrets, tmp_path):
    secrets["CONXT_BASE_URL"] = "http://conxt.example"
    p = conxt.ConxtMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    assert p._client.base_url == conxt.DEFAULT_BASE_URL


def test_a_team_id_that_is_not_a_conxt_id_turns_team_memory_off(secrets, tmp_path):
    secrets["CONXT_TEAM_ID"] = "engineering"
    p = conxt.ConxtMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    assert p._team_id is None
    assert "conxt_propose" not in [s["name"] for s in p.get_tool_schemas()]


def test_setup_keeps_the_team_in_the_profile_and_never_the_key(secrets, tmp_path):
    p = conxt.ConxtMemoryProvider()
    p.save_config({"team_id": TEAM, "api_key": "cx_should_not_be_here"}, str(tmp_path))
    saved = json.loads((tmp_path / "conxt.json").read_text())
    assert saved == {"team_id": TEAM}
    p.initialize("s", hermes_home=str(tmp_path))
    assert p._team_id == TEAM


# -- recall --------------------------------------------------------------------

def test_recall_sends_the_message_with_the_key_and_returns_the_memories(server, secrets, tmp_path):
    p = provider(server, tmp_path, team=TEAM, secrets=secrets)
    block = p.prefetch("When can we deploy the billing service?")
    assert block == "- Deploys freeze on Fridays."
    req = server.requests[0]
    assert req["path"] == "/context/"
    assert req["auth"] == "Bearer cx_test_key"
    assert req["body"] == {"session_opener": "When can we deploy the billing service?", "limit": 8, "team_id": TEAM}
    assert p.recall_status().count == 3


def test_recall_without_a_team_reads_personal_memory_only(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    p.prefetch("What did we decide about retries?")
    assert "team_id" not in server.requests[0]["body"]


def test_a_greeting_is_not_sent_to_conxt(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    assert p.prefetch("thanks!") == ""
    assert server.requests == []


def test_when_conxt_fails_the_turn_goes_ahead_without_recall(server, secrets, tmp_path):
    server.replies["/context/"] = (500, {"detail": "boom"})
    p = provider(server, tmp_path, secrets=secrets)
    assert p.prefetch("What is our on-call rota?") == ""
    assert p.recall_status() is None


def test_when_conxt_is_unreachable_the_turn_goes_ahead_without_recall(secrets, tmp_path):
    p = conxt.ConxtMemoryProvider()
    p.initialize("s", hermes_home=str(tmp_path))
    p._client = conxt._Client("k", "http://127.0.0.1:9")
    assert p.prefetch("What is our on-call rota?") == ""


# -- capture -------------------------------------------------------------------

def test_a_session_is_captured_as_hermes_when_it_ends(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    assert server.requests == []  # buffered, not sent per turn
    p.on_session_end([])
    wait_for_flush(p)
    (req,) = server.sent("/capture/")
    assert req["body"]["source"] == "hermes"
    assert [t["role"] for t in req["body"]["raw_turns"]] == ["user", "assistant"]
    assert p._buffer == []


def test_a_stored_conversation_is_handed_on_to_become_memories(server, secrets, tmp_path):
    # /capture only stores. Without /extract a conversation never became a
    # memory: on 24 Oct one sat unprocessed in production.
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    p.flush()
    assert server.paths() == ["/capture/", "/extract/"]
    (req,) = server.sent("/extract/")
    assert req["body"] == {"session_id": "s1", "source": "hermes"}  # the API key names the user


def test_a_capture_conxt_already_had_is_not_extracted_again(server, secrets, tmp_path):
    server.replies["/capture/"] = (200, {"status": "already_processed", "session_id": "s1"})
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    p.flush()
    assert server.paths() == ["/capture/"]


def test_a_slow_extraction_does_not_hold_up_the_flush(server, secrets, tmp_path, monkeypatch):
    # Extraction takes 9-45s in production; the provider asks and moves on.
    monkeypatch.setattr(conxt, "EXTRACT_WAIT_S", 0.2)
    server.delays["/extract/"] = 2.0
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    started = time.monotonic()
    p.flush()
    assert time.monotonic() - started < 1.5
    assert p._buffer == [] and server.paths() == ["/capture/", "/extract/"]


def test_a_refused_extraction_still_counts_the_conversation_as_sent(server, secrets, tmp_path):
    server.replies["/extract/"] = (422, {"detail": "user_id required"})
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    p.flush()
    p.flush()
    assert p._buffer == [] and len(server.sent("/capture/")) == 1


def test_shutdown_never_sends_a_batch_the_session_end_flush_is_sending(server, secrets, tmp_path):
    # On 24 Oct the same conversation reached Conxt twice, 0.7s apart.
    server.delays["/capture/"] = 3.0
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("We ship on Tuesdays now.", "Noted: Tuesdays.")
    p.on_session_end([])
    time.sleep(0.2)  # the session-end flush is now waiting on Conxt
    p.shutdown()
    wait_for_flush(p)
    assert len(server.sent("/capture/")) == 1


def test_a_long_session_is_sent_in_batches_conxt_accepts(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    p._writes = True
    with p._lock:  # 60 messages, one far over the per-message limit
        p._buffer = [{"role": "user", "content": "x" * 25_000 if i == 0 else f"m{i}", "ts": i} for i in range(60)]
    p.flush()
    batches = [r["body"]["raw_turns"] for r in server.sent("/capture/")]
    assert [len(b) for b in batches] == [50, 10]
    assert len(batches[0][0]["content"]) == conxt.CAPTURE_MAX_TURN_CHARS
    for b in batches:
        assert sum(len(t["content"]) for t in b) <= conxt.CAPTURE_MAX_TOTAL_CHARS


def test_capture_starts_on_its_own_every_five_exchanges(server, secrets, tmp_path):
    # At 40 messages a short conversation lived only in memory until the
    # session ended, and closing the terminal window lost it.
    p = provider(server, tmp_path, secrets=secrets)
    for i in range(4):
        p.sync_turn(f"question {i}", f"answer {i}")
    assert server.requests == []
    p.sync_turn("question 4", "answer 4")
    wait_for_flush(p)
    assert server.paths() == ["/capture/", "/extract/"]


def test_nothing_is_captured_from_a_cron_run(server, secrets, tmp_path):
    p = provider(server, tmp_path, context="cron", secrets=secrets)
    p.sync_turn("Rotate the keys.", "Done.")
    p.on_session_end([])
    p.on_memory_write("add", "memory", "Keys rotate monthly.")
    p.shutdown()
    assert server.requests == []


def test_when_conxt_is_down_the_turns_wait_for_the_next_flush(server, secrets, tmp_path):
    server.replies["/capture/"] = (503, {"detail": "down"})
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("Use Postgres 16.", "Okay.")
    p.flush()
    assert len(p._buffer) == 2
    server.replies["/capture/"] = (200, {"status": "captured"})
    p.flush()
    assert p._buffer == []


def test_a_capture_conxt_rejects_as_invalid_is_dropped_not_retried_forever(server, secrets, tmp_path):
    server.replies["/capture/"] = (400, {"detail": "Session has no turns"})
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("hello there friend", "hi")
    p.flush()
    assert p._buffer == []


def test_whatever_is_left_is_sent_at_shutdown(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    p.sync_turn("Our staging URL changed.", "Got it.")
    p.shutdown()
    assert server.paths() == ["/capture/", "/extract/"]


# -- writes --------------------------------------------------------------------

def test_remember_saves_a_personal_memory_for_real_not_as_a_dry_run(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_remember", {"content": "Prefer small PRs.", "memory_type": "preference"}))
    assert out["saved"] is True
    (req,) = server.requests
    assert req["path"] == "/import/"
    assert req["body"]["dry_run"] is False
    assert req["body"]["payload"]["memories"] == [{"type": "preference", "content": "Prefer small PRs."}]


def test_a_team_memory_is_proposed_and_waits_for_a_person(server, secrets, tmp_path):
    p = provider(server, tmp_path, team=TEAM, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_propose", {"content": "All services log in JSON."}))
    assert out["status"] == "pending" and "approve" in out["message"]
    (req,) = server.requests
    assert req["path"] == f"/teams/{TEAM}/memories/create"
    assert req["body"]["source"] == "hermes" and req["body"]["title"] == "All services log in JSON."


def test_without_a_team_there_is_no_propose_tool_and_a_call_is_refused(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    assert "conxt_propose" not in [s["name"] for s in p.get_tool_schemas()]
    out = json.loads(p.handle_tool_call("conxt_propose", {"content": "x"}))
    assert "error" in out and server.requests == []


def test_a_refusal_from_conxt_reaches_the_agent_with_its_reason(server, secrets, tmp_path):
    server.replies[f"/teams/{TEAM}/memories/create"] = (403, {"detail": "Not a member of this team"})
    p = provider(server, tmp_path, team=TEAM, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_propose", {"content": "x"}))
    assert "Not a member of this team" in out["error"]


def test_an_unknown_memory_type_is_refused_before_anything_is_sent(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_remember", {"content": "x", "memory_type": "fact"}))
    assert "memory_type" in out["error"] and server.requests == []


def test_search_returns_what_conxt_recalls(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_search", {"query": "deploys"}))
    assert out == {"context": "- Deploys freeze on Fridays.", "memories": 2, "pinned": 1}


def test_a_built_in_memory_add_is_mirrored_and_a_removal_is_not(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    p.on_memory_write("remove", "memory", "old note")
    p.on_memory_write("add", "user", "Prefers British spelling.")
    for t in threading.enumerate():
        if t.name == "conxt-mirror":
            t.join(timeout=5)
    (req,) = server.requests
    assert req["path"] == "/import/"
    assert req["body"]["payload"]["memories"][0] == {"type": "preference", "content": "Prefers British spelling."}


# -- agent folders (0.2.0) -----------------------------------------------------

def _mcp_reply(text, is_error=False):
    return (200, {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": text}],
                                                         "isError": is_error}})


def _tool_calls(server):
    return [r["body"]["params"] for r in server.sent("/mcp/") if r["body"].get("method") == "tools/call"]


def test_the_agent_folder_tools_are_offered_without_any_setup_beyond_the_key(server, secrets, tmp_path):
    names = {s["name"] for s in provider(server, tmp_path, secrets=secrets).get_tool_schemas()}
    assert {"conxt_define_agent", "conxt_list_agents", "conxt_start_run", "conxt_finish_run"} <= names


def test_a_run_starts_on_conxts_mcp_server_with_the_key(server, secrets, tmp_path):
    server.replies["/mcp/"] = _mcp_reply("RUN r1\nAgent: grokbot\nINSTRUCTIONS ...")
    p = provider(server, tmp_path, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_start_run", {"folder": "grokbot_project"}))
    assert out["brief"].startswith("RUN r1")
    assert _tool_calls(server) == [{"name": "start_run", "arguments": {"project": "grokbot_project"}}]
    assert all(r["auth"] == "Bearer cx_test_key" for r in server.sent("/mcp/"))
    # initialize, then initialized, then the call: what any MCP server expects
    assert [r["body"].get("method") for r in server.sent("/mcp/")] == [
        "initialize", "notifications/initialized", "tools/call"]


def test_a_run_is_finished_with_its_report_lessons_and_grades(server, secrets, tmp_path):
    server.replies["/mcp/"] = _mcp_reply("Run kept. 2 of 2 lessons saved.")
    p = provider(server, tmp_path, secrets=secrets)
    out = json.loads(p.handle_tool_call("conxt_finish_run", {
        "run_id": "r1", "report": "# Report", "lessons": ["a", "b"], "graded": 10, "hits": "6",
        "carry": {"picks": [{"symbol": "X", "last": 1}]}}))
    assert out["result"].startswith("Run kept")
    assert _tool_calls(server)[0] == {"name": "finish_run", "arguments": {
        "run_id": "r1", "report": "# Report", "lessons": ["a", "b"], "graded": 10, "hits": 6,
        "carry": {"picks": [{"symbol": "X", "last": 1}]}}}


def test_an_agent_md_is_saved_to_its_folder(server, secrets, tmp_path):
    server.replies["/mcp/"] = _mcp_reply("Agent 'grokbot' lives in the folder 'grokbot_project'.")
    p = provider(server, tmp_path, secrets=secrets)
    p.handle_tool_call("conxt_define_agent", {"folder": "grokbot_project", "agent_md": "---\nname: grokbot\n---\nDo it."})
    assert _tool_calls(server)[0]["arguments"] == {"project": "grokbot_project",
                                                    "agent_md": "---\nname: grokbot\n---\nDo it."}


def test_a_refusal_from_the_mcp_server_reaches_the_agent(server, secrets, tmp_path):
    server.replies["/mcp/"] = _mcp_reply("The project 'x' is not an agent folder yet.", is_error=True)
    out = json.loads(provider(server, tmp_path, secrets=secrets).handle_tool_call("conxt_start_run", {"folder": "x"}))
    assert "not an agent folder yet" in out["error"]


def test_a_rejected_key_says_so(server, secrets, tmp_path):
    server.replies["/mcp/"] = (401, {"detail": "bad key"})
    out = json.loads(provider(server, tmp_path, secrets=secrets).handle_tool_call("conxt_list_agents", {}))
    assert "API key" in out["error"]


def test_missing_arguments_are_refused_before_anything_is_sent(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    for tool, args in (("conxt_start_run", {}), ("conxt_define_agent", {"folder": "f"}),
                       ("conxt_finish_run", {"report": "r"}), ("conxt_finish_run", {"run_id": "r", "graded": "many"})):
        assert "error" in json.loads(p.handle_tool_call(tool, args)), tool
    assert server.sent("/mcp/") == []


def _sse(*events):
    return "".join("".join(f"data: {line}\n" for line in e.split("\n")) + "\n" for e in events)


def test_a_streamed_answer_is_read_too(server, secrets, tmp_path):
    body = _sse('{"jsonrpc":"2.0","method":"notifications/progress","params":{}}',
                '{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"RUN r9"}]}}')
    server.replies["/mcp/"] = (200, body, "text/event-stream")
    p = provider(server, tmp_path, secrets=secrets)
    assert json.loads(p.handle_tool_call("conxt_start_run", {"folder": "f"}))["brief"] == "RUN r9"


def test_an_event_split_over_several_data_lines_is_joined():
    lines = ['data: {"jsonrpc":"2.0","id":2,', 'data: "result":{"content":[]}}', ""]
    events = list(conxt._sse_events(iter(lines), time.monotonic() + 5))
    assert events == [{"jsonrpc": "2.0", "id": 2, "result": {"content": []}}]


def test_a_batched_answer_is_searched_for_the_calls_response():
    batch = [{"jsonrpc": "2.0", "method": "notifications/progress"},
             {"jsonrpc": "2.0", "id": 2, "result": {"content": [{"type": "text", "text": "ok"}]}}]
    assert conxt._response_for(batch, 2)["result"]["content"][0]["text"] == "ok"
    assert conxt._response_for(batch, 3) is None


def test_a_stream_that_ends_before_the_answer_is_an_error_not_a_success(server, secrets, tmp_path):
    server.replies["/mcp/"] = (200, _sse('{"jsonrpc":"2.0","method":"notifications/progress","params":{}}'),
                               "text/event-stream")
    out = json.loads(provider(server, tmp_path, secrets=secrets).handle_tool_call(
        "conxt_finish_run", {"run_id": "r1", "report": "x"}))
    assert "may or may not have run" in out["error"]
    # sent once: a finish whose outcome is unknown is never repeated
    assert len([c for c in _tool_calls(server)]) == 1


def test_an_empty_or_unreadable_answer_is_an_error(server, secrets, tmp_path):
    for body in ({}, {"jsonrpc": "2.0", "id": 2}):
        server.replies["/mcp/"] = (200, body)
        out = json.loads(provider(server, tmp_path, secrets=secrets).handle_tool_call("conxt_list_agents", {}))
        assert "error" in out, body


def test_a_stream_kept_open_with_heartbeats_stops_at_the_deadline():
    def forever():
        while True:
            yield ": heartbeat"
            yield ""
    started = time.monotonic()
    assert list(conxt._sse_events(forever(), time.monotonic() + 0.2)) == []


def test_progress_events_that_never_end_in_the_answer_stop_at_the_deadline():
    def progress():
        while True:
            yield 'data: {"jsonrpc":"2.0","method":"notifications/progress"}'
            yield ""
    started = time.monotonic()
    for _ in conxt._sse_events(progress(), time.monotonic() + 0.2):
        assert time.monotonic() - started < 2


def test_bytes_that_never_form_a_line_stop_at_the_deadline_and_the_cap():
    class Raw:
        def read1(self, n):
            time.sleep(0.01)
            return b"x" * 4096
    class Resp:
        raw = Raw()
    started = time.monotonic()
    assert list(conxt._stream_lines(Resp(), time.monotonic() + 0.3)) == []
    assert time.monotonic() - started < 2
    assert time.monotonic() - started < 2


def test_fractional_or_negative_grades_are_refused_not_rounded(server, secrets, tmp_path):
    p = provider(server, tmp_path, secrets=secrets)
    for bad in (3.9, -1, True, "2.5", "many", "²", "٣"):
        out = json.loads(p.handle_tool_call("conxt_finish_run", {"run_id": "r", "report": "x", "graded": bad}))
        assert "whole number" in out["error"], bad
    assert server.sent("/mcp/") == []
    assert conxt._whole(4.0) == 4 and conxt._whole("6") == 6 and conxt._whole(None) == 0


def test_a_short_answer_on_a_stream_left_open_is_read_at_once():
    """Codex on #5: a short event on a stream the server keeps open sat in
    requests' 512-byte read buffer until the deadline, then was thrown away."""
    class Open(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if body.get("id") != 2:
                data = b'{"jsonrpc":"2.0","id":1,"result":{}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            # A plain stream: no Content-Length, no chunking, left open.
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(b'data: {"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"fast"}]}}\n\n')
            self.wfile.flush()
            for _ in range(20):   # heartbeats, well past the client's deadline
                time.sleep(0.3)
                try:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
                except OSError:
                    return

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Open)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = conxt._Client("k", f"http://127.0.0.1:{srv.server_address[1]}")
        started = time.monotonic()
        assert client.mcp_tool("start_run", {"project": "f"}, timeout=3.0) == "fast"
        assert time.monotonic() - started < 2.0
    finally:
        srv.shutdown()


def test_a_compressed_streamed_answer_is_decompressed(server, secrets, tmp_path):
    """Codex on #5: raw reads skip decompression, and requests asks for gzip."""
    import gzip
    body = gzip.compress(_sse('{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"unzipped"}]}}').encode())

    class Gz(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Gz)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = conxt._Client("k", f"http://127.0.0.1:{srv.server_address[1]}")
        assert client.mcp_tool("start_run", {"project": "f"}, timeout=3.0) == "unzipped"
    finally:
        srv.shutdown()


def test_a_stream_that_stalls_mid_answer_is_reported_not_raised():
    """Codex on #5: a stalled raw read raised urllib3's ReadTimeoutError,
    which escaped handle_tool_call instead of becoming the tool's error."""
    class Stall(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if body.get("id") != 2:
                data = b'{"jsonrpc":"2.0","id":1,"result":{}}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b": ping\n\n")
            self.wfile.flush()
            time.sleep(3)   # longer than the client's read timeout

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stall)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = conxt.ConxtMemoryProvider()
        p._client = conxt._Client("k", f"http://127.0.0.1:{srv.server_address[1]}")
        original = p._client.mcp_tool
        p._client.mcp_tool = lambda tool, args, timeout=1.0: original(tool, args, timeout=1.0)
        out = json.loads(p.handle_tool_call("conxt_finish_run", {"run_id": "r", "report": "x"}))
        assert "may or may not have run" in out["error"]
    finally:
        srv.shutdown()


def test_a_tool_call_that_times_out_says_its_outcome_is_unknown(server, secrets, tmp_path):
    server.delays["/mcp/"] = 2.0
    server.replies["/mcp/"] = _mcp_reply("late")
    p = provider(server, tmp_path, secrets=secrets)
    original = p._client.mcp_tool
    p._client.mcp_tool = lambda tool, args, timeout=0.5: original(tool, args, timeout=0.5)
    out = json.loads(p.handle_tool_call("conxt_list_agents", {}))
    # the very first request (initialize) times out: nothing has run, and it says so plainly
    assert "in time" in out["error"] and "may or may not" not in out["error"]


def test_a_finish_that_times_out_says_it_may_have_run():
    """CodeRabbit on #5: a finish_run whose answer timed out said only "did
    not answer in time", though the run may already be finished in Conxt."""
    class SlowCall(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
            if body.get("method") == "tools/call":
                time.sleep(2)
            data = b'{"jsonrpc":"2.0","id":1,"result":{}}'
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:
                pass

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), SlowCall)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = conxt._Client("k", f"http://127.0.0.1:{srv.server_address[1]}")
        with pytest.raises(conxt.ConxtError) as caught:
            client.mcp_tool("finish_run", {"run_id": "r", "report": "x"}, timeout=0.5)
        assert "may or may not have run" in str(caught.value)
    finally:
        srv.shutdown()
