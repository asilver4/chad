"""chad serve — the in-process MLX engine behind llama.cpp's `/completion` wire API.

WHAT THIS IS
------------
The mirror image of `completion_engine.py`. That module is the CLIENT: chad's agent
loop driving a llama.cpp server over raw `/completion` when chad runs somewhere MLX
cannot (a Linux container). This module is the SERVER for that same protocol, backed
by the local `engine.Engine` — so a chad running in a container on this machine (or
anywhere on the LAN) drives THIS Mac's MLX model with the exact bytes it would
otherwise have sent to llama.cpp. Nothing on either side has to learn a new dialect.

Why it exists: a container measuring chad against a remote GGUF measures a different
quantization, a different kernel and a different sampler than the model people
actually run locally. Pointed at `chad serve`, that same container measures *the*
model, through the real persistent prefix cache, with only the harness boundary left
as a difference.

THE PROTOCOL (exactly what `completion_engine.CompletionEngine` sends)
----------------------------------------------------------------------
- ``GET  /props``       → ``{"default_generation_settings": {"n_ctx": N}, ...}``
  Also carries a ``chad`` block advertising the extensions below. A stock llama.cpp
  server has no such block, which is precisely how the client feature-detects.
- ``POST /completion``  → SSE. Body is ``{"prompt": [token ids], "n_predict": …,
  "temperature": …, "stream": true, "cache_prompt": true, "return_tokens": true}``.
  Each generated token is one ``data: {"content": "…"}`` line; the final line carries
  the generated ids and real ``timings`` so the client's stats stay exact rather than
  estimated. Dropping the connection cancels generation, as llama.cpp does.
- ``GET  /health``      → liveness, plus whether a generation is in flight.

THE EXTENSIONS (only chad speaks these; a stock server simply won't advertise them)
-----------------------------------------------------------------------------------
Because both ends are ours, the two capabilities the remote boundary otherwise
forfeits come back:

- ``POST /cache/push`` + ``POST /cache/pop`` — cache quarantine. On a single-slot
  llama.cpp server a sub-agent's prompt evicts the main transcript's prefix and the
  return trip re-prefills it. Here the sub-agent's excursion is bracketed by a real
  ``Engine.push_cache``/``pop_cache``, so the main transcript's KV survives it.
- ``POST /warm`` — the on-disk KV warm-start of the stable system+tools prefix, which
  a remote client cannot do because the checkpoint lives on the *server's* disk.

Both are latency, never correctness: a client that doesn't speak them, or a call that
fails, degrades to the plain remote behavior.

ONE ENGINE THREAD (NOT JUST A LOCK)
-----------------------------------
Every call that touches the engine — load, generate, push/pop, warm — is submitted to a
single-worker executor and awaited, so all MLX work happens on ONE thread for the life of
the process. Two independent reasons, and the second is not optional:

1. `Engine` holds a single persistent prefix cache. A second concurrent generation would
   not merely queue, it would thrash the prefix the first one is built on.
2. **MLX streams are thread-local.** An HTTP server hands each connection to a fresh
   thread, and the engine's own GPU stream does not exist there — evaluating a cache on
   the wrong thread dies with `RuntimeError: There is no Stream(gpu, N) in current
   thread` partway through a prefill. A lock alone does not fix this: it serializes
   access but still runs the work on whichever thread the request arrived on.

Read-only endpoints (`/props`, `/health`) touch no MLX and never enter the executor, so a
monitor can poll while a long turn decodes.

WHAT IS STILL NOT THE LOCAL PRODUCT
-----------------------------------
Generated token ids ride in the final SSE line rather than per-chunk (the engine's
token callback yields decoded text, not ids). A *cancelled* request therefore lands no
ids in the client's cache mirror, which makes its next pre-generation prefill estimate
conservative — the server's own diff, and the stats it reports, stay exact.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

# Advertised in /props under "chad". The client turns each name into a real call only
# when it appears here, so an older/newer server on either side degrades instead of
# erroring — and a stock llama.cpp server, which sends no "chad" block at all, is
# indistinguishable from a chad server with every extension switched off.
CAP_CACHE_QUARANTINE = "cache_quarantine"
CAP_WARM_PREFIX = "warm_prefix"
CAPABILITIES = [CAP_CACHE_QUARANTINE, CAP_WARM_PREFIX]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8081
MAX_BODY_BYTES = 64 * 1024 * 1024   # a 128k-token id array is ~1 MB; this is slack


# --- pure helpers (no engine, no socket — unit-tested offline) --------------

def props_payload(n_ctx: int, model_id: str = "",
                  capabilities: Optional[list] = None) -> dict:
    """The `/props` body. `default_generation_settings.n_ctx` is the field the client
    reads to size its window to the wall this server actually enforces; the `chad`
    block is the extension handshake (absent on stock llama.cpp)."""
    return {
        "default_generation_settings": {"n_ctx": int(n_ctx)},
        "model_path": model_id,
        "chad": {
            "server": "chad-serve",
            "backend": "mlx",
            "capabilities": list(CAPABILITIES if capabilities is None else capabilities),
        },
    }


def sse(payload: dict) -> str:
    """Encode one payload as a Server-Sent-Events line, in the shape
    `completion_engine.parse_sse_chunk` consumes (`data: {…}` + a blank line)."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def parse_completion_request(body: dict, tok: Any = None) -> dict:
    """Validate + normalize a `/completion` body into the args `Engine.generate` wants.

    `prompt` is normally the token-id array chad sends (the point of this protocol:
    no detokenize, no second chat template). A plain string is accepted too — purely
    so the endpoint is reachable from `curl` for a smoke test — and is encoded with
    the server's own tokenizer. Raises ValueError on anything malformed so the handler
    can answer 400 with a message instead of failing mid-stream."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        if tok is None:
            raise ValueError("string prompts need a tokenizer; send token ids")
        prompt_ids = list(tok.encode(prompt, add_special_tokens=False))
    elif isinstance(prompt, list):
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            raise ValueError("prompt array must contain only integer token ids")
        prompt_ids = list(prompt)
    else:
        raise ValueError("prompt is required (array of token ids, or a string)")
    if not prompt_ids:
        raise ValueError("prompt is empty")

    def _num(name: str, default: float) -> float:
        v = body.get(name, default)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(v)

    n_predict = int(_num("n_predict", 512))
    if n_predict <= 0:                      # llama.cpp's -1 means "until context end"
        n_predict = 512
    return {
        "prompt_ids": prompt_ids,
        "n_predict": n_predict,
        "temperature": _num("temperature", 0.0),
        # 0 = OFF, chad's convention across every sampler knob: an unset knob leaves
        # whatever the engine was configured with rather than forcing a neutral value.
        "min_p": _num("min_p", 0.0),
        "top_p": _num("top_p", 0.0),
        "cache_prompt": bool(body.get("cache_prompt", True)),
    }


def timings_payload(stats: Any) -> dict:
    """Map a `GenStats` onto llama.cpp's final-chunk `timings`. These are what let the
    client report EXACT numbers (`approximate` stays False): `prompt_n` is the count
    actually prefilled after the prefix diff, not the whole prompt."""
    return {
        "prompt_n": int(getattr(stats, "prompt_tokens", 0)),
        "prompt_ms": float(getattr(stats, "prefill_s", 0.0)) * 1000.0,
        "predicted_n": int(getattr(stats, "generated_tokens", 0)),
        "predicted_ms": float(getattr(stats, "gen_s", 0.0)) * 1000.0,
        # Not a llama.cpp field; harmless to a client that ignores unknown keys and
        # useful in a raw curl session to see the prefix cache doing its job.
        "cache_n": int(getattr(stats, "cached_tokens", 0)),
    }


# --- generation (engine-driven, socket-agnostic: `write` is injected) -------

def stream_completion(eng: Any, params: dict, write: Callable[[str], None]) -> dict:
    """Run ONE `/completion` against `eng`, pushing SSE lines through `write`.

    `write` is the only I/O: tests pass a list-appender, the handler passes a socket
    writer. A write that raises (the client hung up) is the cancel signal — the flag it
    sets is what `should_stop` returns, so `Engine.generate` stops decoding on the next
    token instead of finishing a turn nobody is listening to.

    Returns a small dict summarizing the request (for the server's own log line)."""
    prompt_ids = params["prompt_ids"]
    gone = threading.Event()

    def emit(payload: dict) -> None:
        if gone.is_set():
            return
        try:
            write(sse(payload))
        except (BrokenPipeError, ConnectionResetError, OSError):
            gone.set()

    def on_token(seg: str) -> None:
        # One SSE line per decoded token: the client counts emitted tokens by
        # non-empty content chunks, so batching text here would undercount its
        # think-ceiling and stop-condition bookkeeping.
        emit({"content": seg, "stop": False})

    if not params["cache_prompt"]:
        # Explicit opt-out: forfeit the prefix cache for this request. The measurement
        # arm that wants every prompt to prefill from scratch.
        eng.reset()

    # Per-request sampler knobs, restored afterwards so one request can't silently
    # re-tune the next. Held under the caller's lock (see _Handler), so no interleave.
    saved = (getattr(eng, "temp", 0.0), getattr(eng, "min_p", 0.0),
             getattr(eng, "top_p", 0.0))
    eng.temp, eng.min_p, eng.top_p = (params["temperature"], params["min_p"],
                                      params["top_p"])
    try:
        text, stats = eng.generate(
            prompt_ids,
            max_tokens=params["n_predict"],
            on_token=on_token,
            should_stop=gone.is_set,
        )
    finally:
        eng.temp, eng.min_p, eng.top_p = saved

    if gone.is_set():
        # Client hung up: no final chunk to deliver it to. The engine's own cache is
        # intact and correct; only the client's mirror of it goes stale.
        return {"cancelled": True, "generated": 0}

    # The generated ids, recovered from the engine's own cache bookkeeping
    # (`_cached_ids` is prompt + generation after a turn) — this is what lets the
    # client mirror the server's cache state and keep its prefill estimate honest.
    cached_ids = list(getattr(eng, "_cached_ids", []) or [])
    gen_ids = cached_ids[len(prompt_ids):] if len(cached_ids) >= len(prompt_ids) else []
    emit({
        "content": "",
        "stop": True,
        "tokens": gen_ids,
        "timings": timings_payload(stats),
    })
    emit_done(write, gone)
    return {"cancelled": False, "generated": int(getattr(stats, "generated_tokens", 0)),
            "prefilled": int(getattr(stats, "prompt_tokens", 0)),
            "cached": int(getattr(stats, "cached_tokens", 0)), "text_len": len(text)}


def emit_done(write: Callable[[str], None], gone: threading.Event) -> None:
    """Terminal `data: [DONE]` sentinel. The client treats it as end-of-stream (its
    parser returns None for it), so this is politeness for other SSE consumers."""
    if gone.is_set():
        return
    try:
        write("data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError, OSError):
        gone.set()


# --- server state -----------------------------------------------------------

class ServerState:
    """Everything the handlers touch: the engine, the single thread all engine work is
    funnelled onto, and the optional bearer token."""

    def __init__(self, eng: Any, model_id: str = "", api_key: str = "",
                 quiet: bool = False):
        self.eng = eng
        self.model_id = model_id
        self.api_key = api_key
        self.quiet = quiet
        self.busy = False
        # One worker: serializes access to the single KV cache AND keeps every MLX call
        # on one thread, which the thread-local GPU streams require (see module docstring).
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chad-engine")

    def call(self, fn: Callable, *args: Any, **kw: Any) -> Any:
        """Run `fn` on the engine thread and wait for it. Exceptions propagate to the
        caller unchanged, so a handler still answers with a real error."""
        return self._pool.submit(fn, *args, **kw).result()

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def n_ctx(self) -> int:
        # Reads a plain int off the engine object — no MLX, so no executor hop, so
        # /props and /health answer during a long generation.
        return int(getattr(self.eng, "effective_ctx", 0) or 0)

    def spill_status(self) -> dict:
        """Whether the engine can reclaim a quarantined cache to disk under memory
        pressure. Observability, because this is invisible from the client side and
        silently disarms if either half is missing: the engine needs somewhere to write
        (`cache_dir`) AND a measured per-token cost to decide with. Both are plain
        attribute reads — no MLX, so no executor hop."""
        cache_dir = getattr(self.eng, "cache_dir", None)
        per_token = float(getattr(self.eng, "kv_bytes_per_token", 0.0) or 0.0)
        return {"cache_dir": cache_dir, "kv_bytes_per_token": per_token,
                "budget_bytes": int(getattr(self.eng, "kv_cache_max_bytes", 0) or 0),
                "armed": bool(cache_dir and per_token)}


def _make_handler(state: ServerState) -> type:
    """Build a handler class bound to `state` (avoids module-level mutable globals,
    which would make two servers in one process — as the tests run — share an engine)."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "chad-serve"

        # -- plumbing --------------------------------------------------------

        def log_message(self, fmt: str, *a: Any) -> None:
            if not state.quiet:
                sys.stderr.write("[serve] %s\n" % (fmt % a))

        def _authorized(self) -> bool:
            if not state.api_key:
                return True
            hdr = self.headers.get("Authorization", "")
            return hdr.strip() == f"Bearer {state.api_key}"

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))

        def _send_json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str) -> None:
            self._send_json(code, {"error": {"code": code, "message": msg}})

        # -- routes ----------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            if not self._authorized():
                return self._error(401, "missing or bad Authorization bearer token")
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/props":
                # Deliberately lock-free: a monitor polling /props must not queue
                # behind a multi-minute generation.
                return self._send_json(200, props_payload(state.n_ctx(), state.model_id))
            if path in ("/health", "/"):
                return self._send_json(200, {"status": "ok", "busy": state.busy,
                                             "n_ctx": state.n_ctx(),
                                             "model": state.model_id,
                                             "spill": state.spill_status()})
            return self._error(404, f"no such endpoint: {path}")

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._error(401, "missing or bad Authorization bearer token")
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                return self._error(400, f"bad request body: {e}")

            if path == "/completion":
                return self._completion(body)
            if path == "/cache/push":
                return self._cache_op("push")
            if path == "/cache/pop":
                return self._cache_op("pop")
            if path == "/warm":
                return self._warm(body)
            return self._error(404, f"no such endpoint: {path}")

        # -- handlers --------------------------------------------------------

        def _completion(self, body: dict) -> None:
            try:
                params = parse_completion_request(body, tok=getattr(state.eng, "tok", None))
            except ValueError as e:
                return self._error(400, str(e))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # No Content-Length is knowable up front, so the stream IS the message and
            # the connection ends it. HTTP/1.1 keep-alive is off for this response only.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def write(chunk: str) -> None:
                self.wfile.write(chunk.encode("utf-8"))
                self.wfile.flush()

            # Onto the engine thread: one KV cache → one generation (a queued request
            # waits here rather than racing the prefix the running one is decoding
            # against), and MLX's thread-local streams only exist on that thread. The
            # SSE `write` closure runs there too — only one thread ever writes this
            # socket, which is what makes the hung-up-client cancel safe.
            state.busy = True
            try:
                info = state.call(stream_completion, state.eng, params, write)
            finally:
                state.busy = False
            if not state.quiet:
                sys.stderr.write(f"[serve] completion {info}\n")

        def _cache_op(self, which: str) -> None:
            """Cache quarantine: bracket a sub-agent's excursion so its prompt doesn't
            evict the main transcript's prefix. A failure here is reported but is never
            fatal — the caller degrades to the plain remote behavior (re-prefill)."""
            try:
                state.call(state.eng.push_cache if which == "push"
                           else state.eng.pop_cache)
            except Exception as e:  # noqa: BLE001 — latency feature, not correctness
                return self._send_json(200, {"ok": False, "error": str(e)})
            return self._send_json(200, {"ok": True, "op": which})

        def _warm(self, body: dict) -> None:
            """On-disk KV warm-start of a stable prefix — impossible for a remote client
            to do itself, since the checkpoint lives on this machine's disk."""
            prefix = body.get("prefix")
            if not isinstance(prefix, list) or not all(
                    isinstance(t, int) and not isinstance(t, bool) for t in prefix):
                return self._error(400, "prefix must be an array of token ids")
            state.busy = True
            try:
                status, fed = state.call(state.eng.warm_prefix, list(prefix))
            except Exception as e:  # noqa: BLE001 — same: degrade, don't fail a run
                return self._send_json(200, {"status": "error", "fed": 0,
                                             "error": str(e)})
            finally:
                state.busy = False
            return self._send_json(200, {"status": status, "fed": int(fed)})

    return _Handler


def build_server(state: ServerState, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind the HTTP server. Threaded so `/props` and `/health` answer while a
    generation holds the engine lock; the lock, not the thread count, is what keeps
    the single KV cache safe."""
    srv = ThreadingHTTPServer((host, port), _make_handler(state))
    srv.daemon_threads = True
    return srv


# --- entry point (dispatched from cli.main on the literal task `serve`) -----

def run(args: Any) -> int:
    """Load the local model and serve it. Returns the process exit code."""
    from . import cli, config

    if getattr(args, "backend", "mlx") != "mlx":
        sys.stderr.write(
            f"chad serve exposes the LOCAL MLX engine; --backend {args.backend} has "
            "nothing to serve (it is itself a client of a remote server).\n")
        return 2
    try:
        cli._preflight("mlx")
    except SystemExit:
        return 2

    host = args.host or config.env_str("CHAD_SERVE_HOST", DEFAULT_HOST)
    port = int(args.port or config.env_int("CHAD_SERVE_PORT", DEFAULT_PORT))
    # Unauthenticated by default, which is why the default bind is loopback. A LAN
    # bind (what a container on another host needs) should set a token.
    api_key = config.env_str("CHAD_SERVE_API_KEY", "") or ""
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        sys.stderr.write(
            f"[serve] warning: binding {host} with no CHAD_SERVE_API_KEY — anyone who "
            "can reach this port can spend your GPU.\n")

    from .engine import Engine, sweep_orphan_spills
    model_id, why = cli._pick_model()
    cli._ensure_model(model_id)
    cache_dir = os.path.expanduser("~/.cache/chad/kv")
    sweep_orphan_spills(cache_dir, max_age_s=6 * 3600)
    kv_cache_max_gb = cli._env_int("CHAD_KV_CACHE_MAX_GB")
    eng = Engine(
        model_id=model_id,
        draft_id=None,
        kv_bits=cli._env_int("CHAD_KV_BITS"),
        max_context=cli._env_int("CHAD_MAX_CONTEXT"),
        cache_dir=cache_dir,
        kv_cache_max_bytes=(kv_cache_max_gb if kv_cache_max_gb is not None else 8) * 1024**3,
    )
    sys.stderr.write(f"loading {os.path.basename(model_id.rstrip('/'))} [{why}] ...\n")
    # The weights load ON the engine thread, not this one: MLX's streams are
    # thread-local, so whatever thread loads the model must be the thread that later
    # prefills against its cache.
    state = ServerState(eng, model_id=model_id, api_key=api_key)
    try:
        load_s = state.call(eng.load)
    except Exception as e:  # noqa: BLE001 — same guidance the CLI gives on a bad load
        state.close()
        cli._fail_model_load(model_id, e)
        return 2
    try:
        srv = build_server(state, host, port)
    except OSError as e:
        state.close()
        sys.stderr.write(f"chad serve: cannot bind {host}:{port} — {e}\n")
        return 2

    sys.stderr.write(
        f"ready in {load_s:.1f}s | context {state.n_ctx()} tokens\n"
        f"serving {model_id} on http://{host}:{port} "
        f"(llama.cpp /completion protocol{', auth on' if api_key else ''})\n"
        f"point a client at it with:  chad \"…\" --backend llama --base-url "
        f"http://{host}:{port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[serve] shutting down\n")
    finally:
        srv.server_close()
        state.close()
    return 0
