"""Tests for `chad serve` — the MLX engine behind llama.cpp's /completion protocol.

No model, no weights, no MLX: the engine is a fake that records what it was asked and
replays a canned generation, and `stream_completion` writes through an injected sink
rather than a socket. The end-to-end tests bind a real loopback HTTP server (fast — no
model to load) and drive it with the REAL client, `CompletionEngine`, which is the only
way to prove the two halves actually agree on the wire.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from chad import serve
from chad.base_engine import GenStats
from chad.completion_engine import CompletionEngine


class FakeEngine:
    """Stands in for `engine.Engine`: same surface the server touches, plus a log of
    what happened so tests can assert on cache bookkeeping."""

    def __init__(self, out="hello world", gen_ids=(101, 102), effective_ctx=8192):
        self.out = out
        self.cache_dir = "/tmp/kv"          # engine.Engine has these; spill_status reads them
        self.kv_bytes_per_token = 1234.0
        self.kv_cache_max_bytes = 8 * 1024**3
        self.gen_ids = list(gen_ids)
        self.effective_ctx = effective_ctx
        self.temp = 0.0
        self.min_p = 0.0
        self.top_p = 0.0
        self.tok = None
        self._cached_ids = []
        self.calls = []
        self.threads = set()
        self.sampler_seen = None
        self.stop_after = None      # emit N tokens then pretend the client is gone

    def _note_thread(self):
        # MLX streams are thread-local, so EVERY engine call must land on the same
        # thread or a prefill dies with "no Stream(gpu, N) in current thread".
        self.threads.add(threading.get_ident())

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, should_stop=None,
                 **kw):
        self._note_thread()
        self.calls.append(("generate", list(prompt_ids), max_tokens))
        self.sampler_seen = (self.temp, self.min_p, self.top_p)
        pieces = self.out.split(" ")
        emitted = 0
        for i, p in enumerate(pieces):
            if should_stop and should_stop():
                break
            on_token and on_token(p if i == 0 else " " + p)
            emitted += 1
        self._cached_ids = list(prompt_ids) + self.gen_ids
        return self.out, GenStats(prompt_tokens=7, cached_tokens=3, prefill_s=0.5,
                                  generated_tokens=emitted, gen_s=0.25)

    def reset(self):
        self._note_thread()
        self.calls.append(("reset",))
        self._cached_ids = []

    def push_cache(self):
        self._note_thread()
        self.calls.append(("push",))

    def pop_cache(self):
        self._note_thread()
        self.calls.append(("pop",))

    def warm_prefix(self, prefix_ids, should_stop=None):
        self._note_thread()
        self.calls.append(("warm", list(prefix_ids)))
        self._cached_ids = list(prefix_ids)
        return ("hit", len(prefix_ids))


def sink():
    """A `write` that collects lines, and a decoder for the payloads it captured."""
    lines = []

    def write(chunk):
        lines.append(chunk)

    def payloads():
        out = []
        for raw in lines:
            body = raw.strip()
            if not body.startswith("data:"):
                continue
            body = body[len("data:"):].strip()
            if body == "[DONE]":
                continue
            out.append(json.loads(body))
        return out

    return write, payloads


# --- pure helpers ---------------------------------------------------------

def test_props_advertises_ctx_and_capabilities():
    p = serve.props_payload(32768, "ornith-35b")
    # the field the client actually reads to size its window
    assert p["default_generation_settings"]["n_ctx"] == 32768
    caps = p["chad"]["capabilities"]
    assert serve.CAP_CACHE_QUARANTINE in caps and serve.CAP_WARM_PREFIX in caps


def test_sse_line_shape_matches_the_client_parser():
    from chad.completion_engine import parse_sse_chunk
    line = serve.sse({"content": "hi", "stop": False})
    assert line.endswith("\n\n")
    assert parse_sse_chunk(line.strip()) == {"content": "hi", "stop": False}


def test_parse_completion_request_takes_token_ids_verbatim():
    p = serve.parse_completion_request({"prompt": [1, 2, 3], "n_predict": 64,
                                        "temperature": 0.7, "min_p": 0.05})
    assert p["prompt_ids"] == [1, 2, 3]
    assert p["n_predict"] == 64
    assert p["temperature"] == 0.7
    assert p["min_p"] == 0.05
    assert p["top_p"] == 0.0        # unset knob stays OFF, chad's convention
    assert p["cache_prompt"] is True


def test_parse_completion_request_rejects_junk():
    for bad in ({}, {"prompt": []}, {"prompt": [1, "x"]}, {"prompt": 5},
                {"prompt": [1], "temperature": "hot"}):
        with pytest.raises(ValueError):
            serve.parse_completion_request(bad)


def test_parse_completion_request_encodes_a_string_prompt_for_curl():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
    p = serve.parse_completion_request({"prompt": "hi"}, tok=Tok())
    assert p["prompt_ids"] == [104, 105]
    # …but without a tokenizer it is an error, not a silent empty prompt
    with pytest.raises(ValueError):
        serve.parse_completion_request({"prompt": "hi"})


def test_nonpositive_n_predict_falls_back_rather_than_generating_nothing():
    # llama.cpp spells "until the context ends" as -1; we substitute a sane budget
    assert serve.parse_completion_request({"prompt": [1], "n_predict": -1})["n_predict"] > 0


def test_timings_map_genstats_onto_the_llama_cpp_fields():
    t = serve.timings_payload(GenStats(prompt_tokens=12, cached_tokens=900,
                                       prefill_s=1.5, generated_tokens=30, gen_s=3.0))
    assert t["prompt_n"] == 12          # tokens actually prefilled, not prompt length
    assert t["prompt_ms"] == 1500.0
    assert t["predicted_n"] == 30
    assert t["predicted_ms"] == 3000.0
    assert t["cache_n"] == 900


# --- streaming ------------------------------------------------------------

def test_stream_emits_one_chunk_per_token_then_ids_and_timings():
    eng = FakeEngine(out="a b c", gen_ids=[7, 8, 9])
    write, payloads = sink()
    serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1, 2]}),
                            write)
    got = payloads()
    assert [p["content"] for p in got[:-1]] == ["a", " b", " c"]
    final = got[-1]
    assert final["stop"] is True
    assert final["tokens"] == [7, 8, 9]          # what the client mirrors its cache on
    assert final["timings"]["predicted_n"] == 3


def test_stream_applies_then_restores_per_request_sampler_knobs():
    eng = FakeEngine()
    eng.temp, eng.min_p, eng.top_p = 0.0, 0.0, 0.0
    write, _ = sink()
    serve.stream_completion(eng, serve.parse_completion_request(
        {"prompt": [1], "temperature": 0.7, "min_p": 0.05, "top_p": 0.9}), write)
    assert eng.sampler_seen == (0.7, 0.05, 0.9)     # the request's knobs were live…
    assert (eng.temp, eng.min_p, eng.top_p) == (0.0, 0.0, 0.0)   # …and didn't leak


def test_cache_prompt_false_drops_the_prefix_cache_first():
    eng = FakeEngine()
    write, _ = sink()
    serve.stream_completion(eng, serve.parse_completion_request(
        {"prompt": [1, 2], "cache_prompt": False}), write)
    assert ("reset",) in eng.calls
    eng2 = FakeEngine()
    serve.stream_completion(eng2, serve.parse_completion_request({"prompt": [1, 2]}),
                            sink()[0])
    assert ("reset",) not in eng2.calls     # the default keeps the cache, obviously


def test_a_dead_client_stops_generation_and_sends_no_final_chunk():
    """The cancel path: a write that raises IS the signal, and it has to reach
    `should_stop` — otherwise the box keeps decoding a turn nobody is reading."""
    eng = FakeEngine(out="a b c d e")
    n = {"w": 0}

    def write(chunk):
        n["w"] += 1
        if n["w"] >= 2:
            raise BrokenPipeError("client hung up")

    info = serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1]}),
                                   write)
    assert info["cancelled"] is True
    # generation stopped early rather than running out the full 5 tokens
    assert n["w"] == 2


# --- end to end over a real socket, driven by the real client -------------

@pytest.fixture
def live_server():
    """Bind on an ephemeral loopback port and serve until the test is done."""
    eng = FakeEngine()
    state = serve.ServerState(eng, model_id="ornith-test", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", eng, state
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def _client(url, **kw):
    """A CompletionEngine wired to the live server, with the tokenizer load skipped
    (nothing here needs to render a prompt — only the wire behavior is under test)."""
    c = CompletionEngine(model_id="ornith", base_url=url, **kw)
    c._absorb_props(c._fetch_props() or {})   # exactly what load() does, minus the tokenizer
    return c


def test_client_reads_context_window_and_capabilities_from_props(live_server):
    url, _, _ = live_server
    c = _client(url)
    assert c.effective_ctx == 8192      # the server's wall, not the 32768 fallback
    assert c._caps == set(serve.CAPABILITIES)


def test_round_trip_gives_the_client_exact_stats_not_estimates(live_server):
    url, eng, _ = live_server
    c = _client(url)
    text, stats = c.generate([1, 2, 3, 4, 5], max_tokens=32)
    assert text == "hello world"
    # the whole reason for this protocol: server-side timings, so nothing is guessed
    assert stats.approximate is False
    assert stats.prompt_tokens == 7 and stats.generated_tokens == 2
    # and the prompt crossed the wire as ids, never as text
    assert eng.calls[0][:1] == ("generate",) and eng.calls[0][1] == [1, 2, 3, 4, 5]


def test_round_trip_mirrors_the_server_cache_state(live_server):
    url, eng, _ = live_server
    c = _client(url)
    c.generate([1, 2, 3], max_tokens=8)
    # client mirror == prompt + generated ids == what the server's own cache holds
    assert c._cached_ids == [1, 2, 3] + eng.gen_ids == eng._cached_ids


def test_quarantine_round_trip_pushes_pops_and_restores_the_mirror(live_server):
    url, eng, _ = live_server
    c = _client(url)
    c.generate([1, 2, 3], max_tokens=8)
    main_prefix = list(c._cached_ids)
    c.push_cache()
    assert ("push",) in eng.calls
    assert c._cached_ids == []                  # sub-agent starts cold, as on the server
    c.generate([9, 9], max_tokens=8)            # the sub-agent's excursion
    c.pop_cache()
    assert ("pop",) in eng.calls
    assert c._cached_ids == main_prefix         # main transcript's prefix is back


def test_warm_prefix_round_trip(live_server):
    url, eng, _ = live_server
    c = _client(url)
    assert c.warm_prefix([1, 2, 3, 4]) == ("hit", 4)
    assert ("warm", [1, 2, 3, 4]) in eng.calls
    assert c._cached_ids == [1, 2, 3, 4]


def test_cache_calls_stay_no_ops_against_a_server_without_the_capabilities(live_server):
    """The stock-llama.cpp path: no `chad` block in /props → the client must not call
    the extension endpoints at all, and must not fail when it doesn't."""
    url, eng, _ = live_server
    c = _client(url)
    c._caps = set()                             # pretend a plain llama-server
    c.push_cache()
    c.pop_cache()
    assert c.warm_prefix([1, 2, 3]) == ("skip", 0)
    assert not [x for x in eng.calls if x[0] in ("push", "pop", "warm")]


def test_capabilities_are_handshaken_lazily_when_the_window_is_pinned(live_server):
    """A pinned window skips the /props probe at load (an operator who pinned it all
    shouldn't wait on a socket), so the extensions have to discover it themselves —
    once — the first time one is used."""
    url, eng, _ = live_server
    c = CompletionEngine(model_id="ornith", base_url=url, effective_ctx=4096)
    assert c._caps == set()                 # nothing probed yet
    c.push_cache()
    assert c._caps == set(serve.CAPABILITIES) and ("push",) in eng.calls
    assert c.effective_ctx == 4096          # the pinned window still wins
    probes = {"n": 0}
    real = c._fetch_props
    c._fetch_props = lambda: (probes.update(n=probes["n"] + 1), real())[1]
    c.pop_cache()
    assert probes["n"] == 0                 # handshake is once per process, not per call


def test_every_engine_call_lands_on_one_dedicated_thread(live_server):
    """MLX streams are thread-local. An HTTP server hands each connection to a fresh
    thread, so running the engine on the handler's thread dies mid-prefill with
    `RuntimeError: There is no Stream(gpu, N) in current thread` — which is exactly what
    happened on a real 9B eval run. A lock would NOT catch this: it serializes access
    but still runs the work wherever the request arrived. Pin the thread, not just the
    order."""
    url, eng, _ = live_server
    c = _client(url)
    for _ in range(3):                      # each is a separate connection → own handler thread
        c.generate([1, 2, 3], max_tokens=4)
    c.push_cache()
    c.pop_cache()
    c.warm_prefix([1, 2, 3])
    assert len(eng.threads) == 1, f"engine touched from {len(eng.threads)} threads"
    # …and it is emphatically not the main thread doing the HTTP work either
    assert eng.threads != {threading.get_ident()}


def test_warm_start_gate_opens_only_when_the_server_can_warm(live_server):
    """agent.py gates its disk warm-start on `engine.cache_dir` being truthy — the one
    and only consumer of that attribute. The client used to hardcode it to None, so the
    warm-start block was skipped and `warm_prefix` was never called at all: the endpoint
    existed and nothing ever reached it. This pins the gate to what the server can
    actually do."""
    url, _, _ = live_server
    c = _client(url)
    assert c.cache_dir                       # server advertises warm_prefix -> gate open
    c._caps = set()                          # a stock llama-server cannot warm for us
    assert c.cache_dir is None               # …so the gate closes again, as it must


def test_bad_request_is_a_400_with_a_message_not_a_broken_stream(live_server):
    url, _, _ = live_server
    req = urllib.request.Request(url + "/completion", data=b'{"prompt": []}',
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert "prompt" in e.value.read().decode()


def test_unknown_endpoint_is_404(live_server):
    url, _, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/v1/chat/completions", timeout=5)
    assert e.value.code == 404


# --- auth -----------------------------------------------------------------

@pytest.fixture
def guarded_server():
    eng = FakeEngine()
    state = serve.ServerState(eng, model_id="ornith-test", api_key="s3cret", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", eng
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def test_api_key_gates_a_lan_bound_server(guarded_server):
    url, _ = guarded_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/props", timeout=5)
    assert e.value.code == 401
    # the client passes it the same way it does for a remote llama.cpp server
    c = CompletionEngine(model_id="m", base_url=url, api_key="s3cret")
    text, _ = c.generate([1, 2], max_tokens=8)
    assert text == "hello world"


def test_health_answers_while_the_engine_thread_is_busy(live_server):
    """`/health` and `/props` must answer during a generation — a poll that queued
    behind the engine would make a long turn look like a hung server."""
    url, _, state = live_server
    started, release = threading.Event(), threading.Event()

    def hog():
        started.set()
        release.wait(10)

    blocked = threading.Thread(target=lambda: state.call(hog), daemon=True)
    blocked.start()
    started.wait(5)
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            body = json.loads(r.read())
        assert body["status"] == "ok" and body["n_ctx"] == 8192
    finally:
        release.set()
        blocked.join(5)


def test_health_reports_whether_the_disk_spill_path_is_armed(live_server):
    """A quarantined cache is reclaimed to disk under memory pressure — but only if the
    engine has BOTH somewhere to write and a measured per-token cost. Either missing and
    the spill silently never happens, which is invisible from the client, so /health
    reports it."""
    url, eng, _ = live_server
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        spill = json.loads(r.read())["spill"]
    assert spill["armed"] is True
    assert spill["cache_dir"] == "/tmp/kv" and spill["kv_bytes_per_token"] == 1234.0
    # lose either half and it must report disarmed rather than quietly doing nothing
    eng.kv_bytes_per_token = 0.0
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        assert json.loads(r.read())["spill"]["armed"] is False
