"""otpu-serve: an OpenAI-style HTTP API for one model on openTPU (docs/serve.md).

    otpu-serve --model qwen3 --backend board            # http://127.0.0.1:8000/v1
    otpu-serve --model lfm2 --kv-ram-gb 8 --kv-disk-gb 32 --kv-dir /data/kv

Endpoints: POST /v1/chat/completions and /v1/completions (stream or not, n choices, stop
strings and token ids, the model's sampling defaults, usage with the prompt tokens the KV
cache already held), SGLang's POST /generate (text or input_ids, sampling_params, meta_info),
GET /v1/models, /health (503 "warming" while a streamed MoE's expert pool is read into the page
cache), /v1/stats (the queue, throughput, the slots, the KV store), POST
/flush_cache (forget the saved contexts), vLLM's POST /tokenize and /detokenize. The monitor (opentpu/serve/metrics.py, docs/dashboard.md):
GET /metrics (Prometheus, vLLM's names), the dashboard at / (live from GET /api/events, server-sent
events; GET /api/stats the snapshot), the card's telemetry with --telemetry SEC. Many clients at once: a model whose image holds
several sequences decodes their requests together in KV slots (opentpu/serve/batching.py),
the others run them in turns (opentpu/serve/scheduler.py); contexts move in and out of the
device through the KV store (opentpu/serve/store.py: host RAM, then disk), and a prompt that
continues a saved context (a chat's next turn, a shared system prompt) feeds only the rest.

The chat template renders the messages as otpu-chat does (thinking off unless --think or the
request's chat_template_kwargs {"enable_thinking": true}); with --reasoning-parser qwen3 a
reply's <think> ... </think> goes to reasoning_content. With --preserve-thinking (or a request's
chat_template_kwargs {"preserve_thinking": true}) a template that has the switch (Qwen3.6's)
renders every assistant turn with its think block, as the model wrote it: a chat's context only
grows (an assistant message's reasoning_content goes to the template). Tool calls (`tools`,
tool_choice "none" or auto) on the chat template's format (opentpu/serve/tools.py). A model with a
vision side (MiniCPM-V, docs/minicpmv.md) takes image_url content parts (data: URLs; http(s)
with --image-fetch) and a request's max_slices; a context's key holds its images' content keys
(minicpmv.keyed), so a saved context is reused past an image only for the same image. The
standard library's HTTP server: no dependency beyond the chat's.
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import re
import socket
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import progstate as PS
from . import scheduler as SC
from . import store as ST
from . import tools as TL

IMAGE_BYTES = 32 << 20      # an image_url's file at most
FETCH_S = 20.0              # an http(s) image's fetch timeout (--image-fetch)
MAX_SLICES = 36             # a request's max_slices at most


WARM_CAP = 300.0         # /health's 503 "warming" at most this long (s; the card host's HDD
                         # reads an 18 GB pool in ~80 s, its SSD in ~37: docs/minicpmv.md 6)


class APIError(Exception):
    def __init__(self, status: int, message: str, kind: str = "invalid_request_error",
                 param: str | None = None):
        super().__init__(message)
        self.status, self.message, self.kind, self.param = status, message, kind, param

    def body(self) -> dict:
        return {"error": {"message": self.message, "type": self.kind, "param": self.param,
                          "code": self.status}}


def _content(c) -> str:
    """A message's content: a string, or a list of parts whose text parts are joined."""
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for p in c:
            if isinstance(p, str):
                out.append(p)
            elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
                out.append(str(p.get("text", "")))
            else:
                raise APIError(400, f"content part of type {p.get('type')!r} is not supported "
                                    f"(text only)", param="messages")
        return "".join(out)
    raise APIError(400, "a message's content is a string or a list of parts", param="messages")


class ThinkSplit:
    """A reply's text split into reasoning (between <think> and </think>) and content, as it
    streams: feed(delta) -> (reasoning delta, content delta); final=True with the last one.
    inside: the prompt opened the block (Qwen3.5 thinking: the generation prompt ends in
    <think>). The newlines after </think> are dropped."""

    def __init__(self, start: str = "<think>", end: str = "</think>", inside: bool = False):
        self.start, self.end = start, end
        self.state = "inside" if inside else "before"
        self.buf = ""

    def feed(self, delta: str, final: bool = False) -> tuple[str, str]:
        self.buf += delta
        rs = cs = ""
        while True:
            if self.state == "before":          # the reply may open a block
                t = self.buf.lstrip()
                if t.startswith(self.start):
                    self.buf, self.state = t[len(self.start):], "inside"
                    continue
                if not final and self.start.startswith(t):
                    break                       # (whitespace or part of the tag so far)
                self.state = "after"
                continue
            if self.state == "inside":
                i = self.buf.find(self.end)
                if i >= 0:
                    rs += self.buf[:i]
                    self.buf, self.state = self.buf[i + len(self.end):], "strip"
                    continue
                keep = 0 if final else _partial(self.buf, self.end)
                rs += self.buf[:len(self.buf) - keep]
                self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.state == "strip":
                self.buf = self.buf.lstrip("\n")
                if not self.buf and not final:
                    break
                self.state = "after"
                continue
            cs += self.buf
            self.buf = ""
            break
        return rs, cs


def _partial(s: str, tag: str) -> int:
    """How many of s's last characters start `tag`."""
    for j in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:j]):
            return j
    return 0


def _parts(c, urls: list):
    """A message's content for a model that takes images: its text and image parts in order
    ({"type": "image"} for each image_url / input_image part, whose url goes to `urls`); as
    _content's string when it has no image."""
    if not isinstance(c, list) or not any(
            isinstance(p, dict) and p.get("type") in ("image_url", "input_image") for p in c):
        return _content(c)
    out = []
    for p in c:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
        elif isinstance(p, dict) and p.get("type") in ("text", "input_text"):
            out.append({"type": "text", "text": str(p.get("text", ""))})
        elif isinstance(p, dict) and p.get("type") in ("image_url", "input_image"):
            u = p.get("image_url")
            u = u.get("url") if isinstance(u, dict) else u
            if not isinstance(u, str) or not u:
                raise APIError(400, "an image_url part without a url", param="messages")
            urls.append(u)
            out.append({"type": "image"})
        else:
            raise APIError(400, f"content part of type {p.get('type')!r} is not supported "
                                f"(text and image_url)", param="messages")
    return out


class Server:
    """The API over a scheduler.Worker: requests parsed, rendered, submitted; responses built."""

    def __init__(self, worker: SC.Worker, tok, spec, name: str, defaults: dict,
                 think: bool = False, reasoning: str | None = None, aliases=(),
                 api_key: str | None = None, max_queue: int = 256, info: dict | None = None,
                 max_slices: int | None = None, fetch: bool = False,
                 warm_cap: float = WARM_CAP, monitor=None, preserve_thinking: bool = False):
        self.w, self.tok, self.spec, self.name = worker, tok, spec, name
        self.defaults, self.think, self.reasoning = dict(defaults), think, reasoning
        self.preserve = preserve_thinking       # (the default: a request's chat_template_kwargs)
        self.names = {name, *aliases}
        self.api_key, self.max_queue = api_key, max_queue
        self.created = int(time.time())
        self.info = info or {}
        self.lock = worker.tok_lock
        self.t0 = time.time()
        self.vs = getattr(spec, "vision", None)     # (MiniCPM-V's: images)
        self.max_slices, self.fetch = max_slices, fetch
        # a streamed MoE's expert pool file: its read into the page cache gates /health
        self.pool = getattr(getattr(worker.eng, "server", None), "pool_file", None)
        self.warm_cap = warm_cap
        self.monitor = monitor          # (metrics.Monitor: /metrics, /api/*, the dashboard)
        self.tool_dialect = TL.dialect(getattr(tok, "chat_template", None))

    # ---- rendering
    def encode(self, text: str) -> list[int]:
        with self.lock:
            return list(self.tok(text)["input_ids"])

    def render(self, messages, think: bool, add_prompt: bool = True,
               images=(), tools=None, keep: bool = False) -> list[int]:
        """images: the messages' images (minicpmv.ImageInput, in order): each one's
        placeholder in place of its <|image_pad|> (as otpu-chat's). tools: the request's
        (tools.request_tools), to the template. keep: the template's preserve_thinking
        (_keeps)."""
        kw = {"tools": tools} if tools else {}
        if keep:
            kw["preserve_thinking"] = True
        with self.lock:
            if getattr(self.tok, "chat_template", None) is None:
                return list(self.tok("".join(m["content"] for m in messages))["input_ids"])
            if images:
                from ..llm import minicpmv as MV
                text = self.tok.apply_chat_template(messages, add_generation_prompt=add_prompt,
                                                    enable_thinking=think, tokenize=False, **kw)
                return list(self.tok(MV.expand(text, images))["input_ids"])
            ids = self.tok.apply_chat_template(messages, add_generation_prompt=add_prompt,
                                               enable_thinking=think, tokenize=True, **kw)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def image_bytes(self, url: str) -> bytes:
        """An image_url's file: a data: URL's (base64), or with --image-fetch an http(s) URL's;
        never a file of the server's."""
        import base64
        import binascii
        if url.startswith("data:"):
            head, _, data = url.partition(",")
            if not head.endswith(";base64"):
                raise APIError(400, "image_url: a data: URL's image in base64", param="messages")
            try:
                raw = base64.b64decode(data)
            except (binascii.Error, ValueError):
                raise APIError(400, "image_url: the base64 does not decode",
                               param="messages") from None
        elif url.startswith(("http://", "https://")):
            if not self.fetch:
                raise APIError(400, "image_url: data: URLs (this server fetches none: "
                                    "otpu-serve --image-fetch)", param="messages")
            import urllib.request
            try:
                with urllib.request.urlopen(url, timeout=FETCH_S) as f:
                    raw = f.read(IMAGE_BYTES + 1)
            except Exception as e:
                raise APIError(400, f"image_url: {e}", param="messages") from None
        else:
            raise APIError(400, "image_url: a data: or http(s) URL", param="messages")
        if len(raw) > IMAGE_BYTES:
            raise APIError(400, f"image_url: an image over {IMAGE_BYTES >> 20} MB",
                           param="messages")
        return raw

    def images(self, urls: list, body: dict) -> list:
        """The request's images (minicpmv.prepare, numbered in order) at its max_slices (the
        server's default, else the processor's)."""
        from ..llm import minicpmv as MV
        ms = body.get("max_slices", self.max_slices)
        if ms is not None:
            if isinstance(ms, bool) or not isinstance(ms, int) or not 1 <= ms <= MAX_SLICES:
                raise APIError(400, f"max_slices: 1 .. {MAX_SLICES}", param="max_slices")
        out = []
        for i, u in enumerate(urls):
            raw = self.image_bytes(u)
            try:
                out.append(MV.prepare(raw, self.vs, ms, image_id=i))
            except Exception as e:
                raise APIError(400, f"image {i}: {type(e).__name__}: {e}",
                               param="messages") from None
        rows = sum(im.tokens for im in out)
        room = getattr(self.spec, "vision_rows", rows)
        if rows > room:
            raise APIError(400, f"the images' {rows} tokens: the model's image holds {room} "
                                f"(fewer images or a lower max_slices)", param="messages")
        return out

    def reply_start(self, messages, ids, think: bool, images=(), tools=None,
                    keep: bool = False) -> int | None:
        """Where the next turn's rendering parts from this one (chat.Chat.reply_start): a
        model with state keeps a saved context there. keep (the earlier turns' thinking kept):
        nowhere, the next turn continues the reply; the prompt's last position then, where a
        next turn whose reply text tokenizes otherwise (or is trimmed) parts from the reply."""
        if not self.w.stateful:
            return None
        if keep:
            return len(ids) - 1
        stand_in = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
        try:
            nxt = self.render(messages + stand_in, think, add_prompt=False, images=images,
                              tools=tools, keep=keep)
        except Exception:
            return None
        return SC.Worker._common(ids, nxt)

    def prefix_marks(self, messages, ids, think: bool, images=(), tools=None,
                     keep: bool = False) -> tuple:
        """Where this prompt's prefix that other conversations share ends, for a model with
        state to keep it there as the prompt is fed (store.PrefixCache.match takes a saved
        context's end only): the last message boundary before the final user turn, at the
        <|im_start|> (the tokenizer's special token) that opens it; an agent's first request:
        its tools and system prompt. () without one."""
        j = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=0)
        special = self._specials()
        if not self.w.stateful or j == 0 or not special:
            return ()
        try:
            alt = self.render(messages[:j] + [{"role": "user", "content": "\x00"}], think,
                              add_prompt=False, images=images, tools=tools, keep=keep)
        except Exception:
            return ()
        c = SC.Worker._common(ids, alt)          # (inside the final user turn's header)
        b = next((i for i in range(c - 1, 0, -1) if ids[i] in special), 0)
        return (b,) if b else ()

    def _keeps(self, kw: dict) -> bool:
        """Whether the template keeps every assistant turn's thinking (Qwen3.6's
        preserve_thinking: the tokens the model saw as it replied, so a chat's context only
        grows): --preserve-thinking, a request's chat_template_kwargs over it; only for a
        template that has the switch."""
        t = getattr(self.tok, "chat_template", None)
        return isinstance(t, str) and "preserve_thinking" in t and \
            bool(kw.get("preserve_thinking", self.preserve))

    def _specials(self) -> set:
        """The tokenizer's special tokens' ids (its added tokens marked special: Qwen's
        <|im_start|> is not in all_special_ids)."""
        if getattr(self, "_special", None) is None:
            dec = getattr(self.tok, "added_tokens_decoder", None) or {}
            self._special = {int(i) for i, t in dec.items() if getattr(t, "special", False)} | \
                {int(i) for i in getattr(self.tok, "all_special_ids", None) or ()}
        return self._special

    # ---- requests
    def _sampling(self, body: dict, i: int) -> SC.Sampling:
        d = self.defaults
        temp = body.get("temperature")
        temp = d["temperature"] if temp is None else float(temp)
        top_p = body.get("top_p")
        top_p = d["top_p"] if top_p is None else float(top_p)
        top_k = body.get("top_k")
        top_k = d["top_k"] if top_k is None or int(top_k) < 0 else int(top_k)
        rp = body.get("repetition_penalty")
        rp = d["repetition_penalty"] if rp is None else float(rp)
        if not 0 <= top_p <= 1:
            raise APIError(400, f"top_p {top_p}: 0 .. 1", param="top_p")
        if temp < 0 or rp <= 0:
            raise APIError(400, "temperature >= 0 and repetition_penalty > 0")
        seed = body.get("seed")
        seed = None if seed is None else int(seed) + i
        return SC.Sampling(temp, top_k, top_p, rp, seed)

    def _stops(self, body: dict) -> tuple[frozenset, tuple]:
        stop = body.get("stop")
        strs = () if stop is None else (stop,) if isinstance(stop, str) else tuple(stop)
        if len(strs) > 16 or not all(isinstance(s, str) and s for s in strs):
            raise APIError(400, "stop: a string or up to 16 non-empty strings", param="stop")
        ids = set() if body.get("ignore_eos") else {int(x) for x in self.spec.eos}
        ids |= {int(x) for x in body.get("stop_token_ids") or ()}
        if not ids:
            ids = {-1}                              # (a stop id no token has)
        return frozenset(ids), strs

    def _max_new(self, body: dict, prompt: int) -> int:
        m = body.get("max_completion_tokens", body.get("max_tokens"))
        room = self.w.eng.cap - prompt
        if m is None:
            return room
        m = int(m)
        if m < 1:
            raise APIError(400, "max_tokens must be at least 1", param="max_tokens")
        return min(m, room)

    def _check(self, body: dict, ids) -> None:
        if not ids:
            raise APIError(400, "an empty prompt")
        if len(ids) >= self.w.eng.cap:
            raise APIError(400, f"This model's maximum context length is {self.w.eng.cap} "
                                f"tokens; the prompt has {len(ids)} (and needs one to "
                                f"generate)", param="messages", kind="invalid_request_error")
        n = int(body.get("n", 1) or 1)
        if not 1 <= n <= 16:
            raise APIError(400, "n: 1 .. 16", param="n")
        if body.get("logprobs") or body.get("top_logprobs"):
            raise APIError(400, "logprobs are not supported", param="logprobs")
        if self.w.busy() >= self.max_queue:
            raise APIError(503, "the server's queue is full; retry later", kind="server_error")

    def submit(self, body: dict, ids, at=None, images=(), marks=(),
               keep_reply: str | None = None) -> list[SC.Request]:
        """ids: the prompt's key (with images: minicpmv.keyed); at, marks, keep_reply:
        Request's."""
        self._check(body, ids)
        stop_ids, stop_strs = self._stops(body)
        n = int(body.get("n", 1) or 1)
        out = []
        for i in range(n):
            r = SC.Request(list(ids), max_new=self._max_new(body, len(ids)),
                           sampling=self._sampling(body, i), stop_ids=stop_ids,
                           stop_strs=stop_strs, at=at, marks=tuple(marks),
                           keep_reply=keep_reply,
                           skip_special=bool(body.get("skip_special_tokens", True)),
                           images=list(images))
            try:
                out.append(self.w.submit(r))
            except ValueError as e:
                for x in out:
                    x.cancel()
                raise APIError(400, str(e)) from None
        return out

    def _rendered(self, body: dict, add_prompt: bool = True):
        """The request's messages, their images and thinking switch, their rendering, the
        tools rendered with them (tools.request_tools: None for none) and whether the earlier
        turns' thinking is kept (_keeps)."""
        msgs = body.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise APIError(400, "messages: a non-empty list", param="messages")
        messages, urls = [], []
        try:
            tools = TL.request_tools(body)
            for m in msgs:
                if not isinstance(m, dict) or "role" not in m:
                    raise APIError(400, "each message has a role and content", param="messages")
                c = m.get("content")
                m = dict(m, role="system") if m["role"] == "developer" else m
                messages.append(TL.message(m, _content(c) if self.vs is None else
                                           _parts(c, urls)))
        except TL.ToolError as e:
            raise APIError(400, str(e), param="messages") from None
        kw = body.get("chat_template_kwargs") or {}
        think = bool(kw.get("enable_thinking", self.think))
        keep = self._keeps(kw)
        images = self.images(urls, body) if urls else []
        try:
            ids = self.render(messages, think, add_prompt, images=images, tools=tools,
                              keep=keep)
        except Exception as e:
            raise APIError(400, f"the chat template refused the messages: {e}") from None
        return messages, images, think, ids, tools, keep

    def chat(self, body: dict):
        """(requests, prompt tokens, reasoning split, its block opened by the prompt, the
        tools to parse the replies with or None)."""
        messages, images, think, ids, tools, keep = self._rendered(body)
        at = self.reply_start(messages, ids, think, images, tools, keep)
        marks = self.prefix_marks(messages, ids, think, images, tools, keep)
        key = ids
        if images:
            from ..llm import minicpmv as MV
            try:
                key = MV.keyed(ids, images, self.vs.image_token)
            except ValueError as e:
                raise APIError(400, f"the images: {e}", param="messages") from None
        # the replies the next request continues: every one with the thinking kept, else a
        # tool call (the agent's next request)
        reply = "" if keep else self.tool_dialect.start if tools else None
        reqs = self.submit(body, key, at, images, marks, keep_reply=reply)
        split = self.reasoning and body.get("separate_reasoning", True)
        opened = split and self._opens_think(ids)
        return reqs, len(ids), split, opened, tools

    def tokenize(self, body: dict) -> dict:
        """vLLM's /tokenize: `prompt` (encoded as /v1/completions does) or `messages` (rendered
        as /v1/chat/completions does; add_generation_prompt, default true, and
        chat_template_kwargs). Nothing runs on the device."""
        if "messages" in body:
            ids = self._rendered(body, bool(body.get("add_generation_prompt", True)))[3]
        elif isinstance(body.get("prompt"), str):
            ids = self.encode(body["prompt"])
        else:
            raise APIError(400, "one of prompt (a string) and messages", param="prompt")
        return {"count": len(ids), "max_model_len": self.w.eng.cap, "tokens": ids,
                "token_strs": None}

    def detokenize(self, body: dict) -> dict:
        """vLLM's /detokenize: `tokens` (token ids) to their text."""
        t = body.get("tokens")
        if not isinstance(t, list) or not all(isinstance(x, int) for x in t):
            raise APIError(400, "tokens: a list of token ids", param="tokens")
        with self.lock:
            return {"prompt": self.tok.decode(t)}

    def _opens_think(self, ids) -> bool:
        """The generation prompt ends inside a thinking block (the template opened it)."""
        with self.lock:
            tail = self.tok.decode(ids[-4:], skip_special_tokens=False)
        t = tail.rstrip()
        return t.endswith("<think>")

    def completion(self, body: dict):
        p = body.get("prompt")
        if isinstance(p, list) and p and isinstance(p[0], list):
            if len(p) != 1:
                raise APIError(400, "one prompt per request", param="prompt")
            p = p[0]
        if isinstance(p, list) and len(p) == 1 and isinstance(p[0], str):
            p = p[0]
        if isinstance(p, str):
            ids = self.encode(p)
        elif isinstance(p, list) and all(isinstance(x, int) for x in p):
            ids = [int(x) for x in p]
        else:
            raise APIError(400, "prompt: a string or a list of token ids", param="prompt")
        if body.get("suffix"):
            raise APIError(400, "suffix is not supported", param="suffix")
        return self.submit(body, ids), len(ids), p if body.get("echo") else None

    def generate(self, body: dict):
        """SGLang's native /generate: `text` or `input_ids` (one prompt, or a list of them) and
        `sampling_params` (max_new_tokens (default 128), temperature, top_p, top_k (-1: all),
        repetition_penalty, stop, stop_token_ids, ignore_eos, skip_special_tokens, n). Returns
        [(requests, prompt tokens)] per prompt, and whether the input was a list."""
        sp = body.get("sampling_params") or {}
        if not isinstance(sp, dict):
            raise APIError(400, "sampling_params: an object", param="sampling_params")
        text, ids = body.get("text"), body.get("input_ids")
        if (text is None) == (ids is None):
            raise APIError(400, "one of text and input_ids", param="text")
        if text is not None:
            batch = isinstance(text, list)
            texts = text if batch else [text]
            if not texts or not all(isinstance(t, str) for t in texts):
                raise APIError(400, "text: a string or a list of strings", param="text")
            prompts = [self.encode(t) for t in texts]
        else:
            batch = isinstance(ids, list) and bool(ids) and isinstance(ids[0], list)
            prompts = ids if batch else [ids]
            if not all(isinstance(p, list) and all(isinstance(x, int) for x in p)
                       for p in prompts):
                raise APIError(400, "input_ids: token ids or a list of them", param="input_ids")
        top_k = sp.get("top_k")
        ob = {"max_tokens": sp.get("max_new_tokens", 128), "temperature": sp.get("temperature"),
              "top_p": sp.get("top_p"), "top_k": 0 if top_k == -1 else top_k,
              "repetition_penalty": sp.get("repetition_penalty"), "stop": sp.get("stop"),
              "stop_token_ids": sp.get("stop_token_ids"), "ignore_eos": sp.get("ignore_eos"),
              "skip_special_tokens": sp.get("skip_special_tokens", True),
              "n": sp.get("n", 1), "seed": sp.get("sampling_seed", body.get("seed"))}
        out = []
        try:
            for p in prompts:
                out.append((self.submit(ob, [int(x) for x in p]), len(p)))
        except BaseException:
            for reqs, _ in out:
                for r in reqs:
                    r.cancel()
            raise
        return out, batch

    def gen_item(self, r: SC.Request, prompt: int, text: str, t0: float, done: bool) -> dict:
        """One /generate result (or a streamed state of it: the text so far)."""
        fin = None
        if done:
            fin = ({"type": "abort", "message": r.error} if r.finish == "error" else
                   {"type": "length", "length": len(r.out)} if r.finish == "length" else
                   {"type": "abort"} if r.finish == "abort" else {"type": "stop"})
        return {"text": text, "output_ids": list(r.out),
                "meta_info": {"id": r.rid, "finish_reason": fin, "prompt_tokens": prompt,
                              "completion_tokens": len(r.out), "cached_tokens": r.cached,
                              "e2e_latency": round(time.perf_counter() - t0, 4)}}

    # ---- responses
    def usage(self, reqs, prompt: int) -> dict:
        out = sum(len(r.out) for r in reqs)
        return {"prompt_tokens": prompt, "completion_tokens": out,
                "total_tokens": prompt + out,
                "prompt_tokens_details": {"cached_tokens": max((r.cached for r in reqs),
                                                               default=0)}}

    def finish(self, r: SC.Request, calls=()) -> str:
        """The choice's finish_reason ("tool_calls" with calls in it, as vLLM's)."""
        if r.finish == "error":
            raise APIError(500, f"generation failed: {r.error}", kind="server_error")
        return "tool_calls" if calls else {"stop": "stop", "length": "length"}.get(r.finish,
                                                                                   "stop")

    def models(self) -> dict:
        return {"object": "list", "data": [
            {"id": self.name, "object": "model", "created": self.created, "owned_by": "opentpu",
             "root": self.name, "max_model_len": self.w.eng.cap}]}

    def stats(self) -> dict:
        u = self.w.usage()
        up = time.time() - self.t0
        dec = u["decode_s"]
        warm = {}
        if self.pool is not None:
            from ..host.offload import warm_report
            warm = {"pool_warm": warm_report(self.pool.warm_t)}
        return {"model": self.name, "uptime_s": round(up, 1), **self.info,
                "throughput": {"generated_tok_s": round(u["generated"] / up, 2) if up else 0,
                               "decode_tok_s": round(u["generated"] / dec, 2) if dec else None},
                "prog_cache": PS.saves(), **warm, **u}

    def health(self) -> tuple[int, dict]:
        """/health's status and body. While the expert pool's read into the page cache runs
        (its warm thread: moe.rewarm's, after the engine's open), 503 "warming" with the read's
        progress, for at most warm_cap seconds from its start; then ok, with "warm":
        "incomplete" and a note where it did not read all it was to (still running past the
        cap, an error, stopped). Advisory: requests are served while it runs, their misses
        from the disk (docs/minicpmv.md 6, pooldisk)."""
        t = getattr(self.pool, "warm_t", None)
        if t is None:
            return 200, {"status": "ok"}
        prog = round(t.bytes / t.total, 3) if t.total else 1.0
        if t.is_alive():
            if time.time() - t.t0 < self.warm_cap:
                return 503, {"status": "warming", "progress": prog}
            return 200, {"status": "ok", "warm": "incomplete", "progress": prog,
                         "note": f"the pool's warm runs past the {self.warm_cap:g} s cap"}
        if t.error or t.t1 is None or t.bytes < t.total:
            return 200, {"status": "ok", "warm": "incomplete", "progress": prog,
                         "note": t.error or ("the pool's warm thread died" if t.t1 is None
                                             else "the pool's warm stopped")}
        return 200, {"status": "ok"}


def _sse(obj) -> bytes:
    return b"data: " + json.dumps(obj, separators=(",", ":")).encode() + b"\n\n"


DASHBOARD = Path(__file__).with_name("dashboard.html")
CLIENT_POLL_S = 0.25            # how often a handler with nothing to write looks at its client
KEEPALIVE = {"TCP_KEEPIDLE": 30, "TCP_KEEPINTVL": 10, "TCP_KEEPCNT": 3}     # (Linux's: _keepalive)
_KEY_RE = re.compile(r"([?&]key=)[^&\s]*")


def _keepalive(sock) -> None:
    """A client's machine gone without a FIN fails its socket in about a minute (the kernel's
    default: two hours). Through ssh -L the socket is sshd's on this host: its ClientAlive
    settings are the ones that see the far machine."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for name, v in KEEPALIVE.items():
        if hasattr(socket, name):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, name), v)


def make_handler(srv: Server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "otpu-serve"

        def log_message(self, fmt, *args):     # one line per request on stderr (no ?key=)
            if os.environ.get("OTPU_SERVE_QUIET"):
                return
            line = _KEY_RE.sub(r"\1-", fmt % args)
            sys.stderr.write(f"{self.address_string()} - {line}\n")

        # ---- plumbing
        def _json(self, status: int, obj) -> None:
            b = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _error(self, e: APIError) -> None:
            self._json(e.status, e.body())

        def _auth(self, query: bool = False) -> None:
            """The API key: 'Authorization: Bearer <key>'; query: or ?key=<key> (the
            dashboard's requests: a browser's EventSource sets no headers)."""
            if not srv.api_key:
                return
            got = self.headers.get("Authorization", "")
            if query and not got.startswith("Bearer "):
                q = self.path.partition("?")[2]
                got = "Bearer " + next((v for k, _, v in (x.partition("=") for x in q.split("&"))
                                        if k == "key"), "")
            if not hmac.compare_digest(got.encode(), f"Bearer {srv.api_key}".encode()):
                raise APIError(401, "invalid API key", kind="authentication_error")

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                b = json.loads(self.rfile.read(n) or b"{}")
            except ValueError:
                raise APIError(400, "the body is not JSON") from None
            if not isinstance(b, dict):
                raise APIError(400, "the body is a JSON object")
            return b

        def setup(self) -> None:
            super().setup()
            _keepalive(self.connection)

        def _client_gone(self, stream: bool) -> bool:
            """The client has closed. A FIN alone does not say so: a client that half-closed after
            sending reads on, and through ssh -L a client's close arrives as sshd's half-close.
            So each look from the FIN on writes something a client ignores (an SSE comment; before
            a plain reply's headers an interim 100): a closed client answers with a reset and a
            later write fails (the second, through ssh the fourth: docs/serve.md), a half-closed
            one takes them and its reply. A client that has sent no FIN sees none of this."""
            try:
                if self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                    return False                # (a next request's bytes)
                if stream:
                    self._chunk(b": keepalive\n\n")
                else:
                    self.wfile.write(b"HTTP/1.1 100 Continue\r\n\r\n")
            except BlockingIOError:
                return False                    # (nothing to read: no FIN)
            except OSError:
                return True
            return False

        def _deltas(self, r: SC.Request, stream: bool = True):
            """r.stream(), with the client looked at while none comes: a prompt writes nothing
            until its first token (a long one for minutes). A client gone: BrokenPipeError, as a
            failed write, and every caller cancels its requests on that."""
            while True:
                try:
                    yield from r.stream(timeout=CLIENT_POLL_S)
                    return
                except queue.Empty:
                    if self._client_gone(stream):
                        raise BrokenPipeError("the client closed") from None

        def _stream_start(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

        def _chunk(self, data: bytes) -> None:
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _stream_end(self) -> None:
            self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        # ---- the monitor's routes
        def _monitor(self):
            if srv.monitor is None:
                raise APIError(404, "no monitor on this server")
            return srv.monitor

        def _fan_gains(self, body: dict) -> None:
            """POST /api/fan_gains {"fan": 1, "kp": .., "ki": .., "kd": ..}: for fan_pid's next
            step (fans.py). JSON's content type only: another site's page cannot send that
            unasked."""
            from . import fans as FN
            if self._monitor().fans is None:
                raise APIError(404, "no fans on this server")
            if self.headers.get_content_type() != "application/json":
                raise APIError(415, "the body's Content-Type is application/json")
            try:
                gains = FN.set_gains(body.get("fan"), body)
            except ValueError as e:
                raise APIError(400, str(e)) from None
            except OSError as e:
                raise APIError(500, f"the gains were not written: {e}") from None
            self._json(200, {"fan": body["fan"], "gains": gains})

        def _dashboard(self) -> None:
            try:
                page = DASHBOARD.read_bytes()
            except OSError:
                raise APIError(404, f"the dashboard's page is not installed ({DASHBOARD})") \
                    from None
            self._send(200, page, "text/html; charset=utf-8")

        def _events(self) -> None:
            """GET /api/events: server-sent events, `hello` (the snapshot with its series and
            the recent requests) then a `tick` per monitor tick (`recent` in it when a request
            ended since the last), a comment every 15 s without one."""
            from .metrics import sse
            mon = self._monitor()
            snap = mon.snapshot(recent=True, history=True)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            seq, rv = snap["seq"], snap["rec_version"]
            try:
                self._chunk(b"retry: 3000\n" + sse("hello", snap))
                while not mon.stopped():
                    s = mon.wait(seq, timeout=15.0)
                    if s is None:
                        self._chunk(b": keepalive\n\n")
                        continue
                    seq = s["seq"]
                    if s["rec_version"] != rv:
                        s = dict(s, recent=mon.recent())
                        rv = s["rec_version"]
                    self._chunk(sse("tick", s))
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass

        # ---- routes
        def do_GET(self):
            try:
                path = self.path.split("?")[0].rstrip("/")
                if path in ("", "/dashboard", "/dashboard.html"):
                    return self._dashboard()        # (the page holds no data: its API calls
                #                                     carry the key)
                self._auth(query=path.startswith("/api/") or path == "/metrics")
                if path == "/metrics":
                    return self._send(200, self._monitor().prometheus().encode(),
                                      "text/plain; version=0.0.4; charset=utf-8")
                if path == "/api/stats":
                    return self._json(200, self._monitor().snapshot(recent=True,
                                                                    history="history" in
                                                                    self.path))
                if path == "/api/events":
                    return self._events()
                if path in ("/health", "/v1/health", "/health_generate"):
                    return self._json(*srv.health())
                if path == "/v1/models":
                    return self._json(200, srv.models())
                if path.startswith("/v1/models/"):
                    m = srv.models()["data"][0]
                    if path[len("/v1/models/"):] not in srv.names:
                        raise APIError(404, f"model {path[11:]!r} not found", param="model")
                    return self._json(200, m)
                if path in ("/v1/stats", "/stats", "/get_server_info"):
                    return self._json(200, srv.stats())
                if path == "/get_model_info":
                    return self._json(200, {"model_path": srv.info.get("path", srv.name),
                                            "is_generation": True,
                                            "max_context_length": srv.w.eng.cap})
                raise APIError(404, f"no route {self.path}")
            except APIError as e:
                self._error(e)

        def do_POST(self):
            try:
                self._auth()
                path = self.path.split("?")[0].rstrip("/")
                if path == "/flush_cache":
                    srv.w.flush()
                    return self._json(200, {"flushed": True})
                body = self._body()
                if path == "/api/fan_gains":
                    return self._fan_gains(body)
                if path == "/v1/chat/completions":
                    return self._chat(body)
                if path == "/v1/completions":
                    return self._completion(body)
                if path == "/generate":
                    return self._generate(body)
                if path == "/tokenize":
                    return self._json(200, srv.tokenize(body))
                if path == "/detokenize":
                    return self._json(200, srv.detokenize(body))
                raise APIError(404, f"no route {self.path}")
            except APIError as e:
                self._error(e)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _wait(self, reqs) -> None:
            for r in reqs:
                for _ in self._deltas(r, stream=False):
                    pass

        def _chat(self, body: dict) -> None:
            reqs, plen, split, opened, tools = srv.chat(body)
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            base = {"id": cid, "object": "chat.completion.chunk", "created": created,
                    "model": srv.name}
            if body.get("stream"):
                return self._chat_stream(body, reqs, plen, split, opened, base, tools)
            try:
                self._wait(reqs)
            except BaseException:
                for r in reqs:
                    r.cancel()
                raise
            choices = []
            for i, r in enumerate(reqs):
                msg = {"role": "assistant", "content": r.text}
                if split:
                    sp = ThinkSplit(inside=opened)
                    rs, cs = sp.feed(r.text, final=True)
                    msg = {"role": "assistant", "content": cs,
                           "reasoning_content": rs.strip() or None}
                calls = ()
                if tools:
                    msg["content"], calls = TL.parse(msg["content"], tools, srv.tool_dialect)
                    if calls:
                        msg["tool_calls"] = [TL.openai(c) for c in calls]
                choices.append({"index": i, "message": msg, "logprobs": None,
                                "finish_reason": srv.finish(r, calls)})
            self._json(200, {"id": cid, "object": "chat.completion", "created": created,
                             "model": srv.name, "choices": choices,
                             "usage": srv.usage(reqs, plen)})

        def _chat_stream(self, body, reqs, plen, split, opened, base, tools=None) -> None:
            self._stream_start()
            try:
                for i, r in enumerate(reqs):
                    self._chunk(_sse({**base, "choices": [{"index": i, "delta": {
                        "role": "assistant", "content": ""}, "logprobs": None,
                        "finish_reason": None}]}))
                    sp = ThinkSplit(inside=opened) if split else None
                    ts = TL.ToolStream(tools, srv.tool_dialect) if tools else None
                    for delta in self._deltas(r):
                        self._chat_delta(base, i, delta, sp, ts)
                    if sp is not None or ts is not None:
                        self._chat_delta(base, i, "", sp, ts, final=True)
                    calls = ts.calls if ts is not None else ()
                    fin = srv.finish(r, calls) if r.finish != "error" else "error"
                    self._chunk(_sse({**base, "choices": [{"index": i, "delta": {},
                                                           "logprobs": None,
                                                           "finish_reason": fin}]}))
                if (body.get("stream_options") or {}).get("include_usage"):
                    self._chunk(_sse({**base, "choices": [], "usage": srv.usage(reqs, plen)}))
                self._stream_end()
            except (BrokenPipeError, ConnectionResetError, OSError):
                for r in reqs:
                    r.cancel()

        def _chat_delta(self, base, i, delta, sp, ts=None, final=False) -> None:
            """A delta of choice i: its reasoning (sp: a ThinkSplit), its content and, with
            ts (a tools.ToolStream), a tool_calls delta for each call it closes (the call
            whole: index, id, type, name, arguments)."""
            if sp is None and ts is None:
                if delta:
                    self._chunk(_sse({**base, "choices": [{"index": i, "delta": {
                        "content": delta}, "logprobs": None, "finish_reason": None}]}))
                return
            rs, cs = sp.feed(delta, final=final) if sp is not None else ("", delta)
            calls = []
            if ts is not None:
                k = len(ts.calls)
                cs, new = ts.feed(cs, final=final)
                calls = [TL.openai(c, k + j) for j, c in enumerate(new)]
            d = {}
            if rs:
                d["reasoning_content"] = rs
            if cs:
                d["content"] = cs
            if d:
                self._chunk(_sse({**base, "choices": [{"index": i, "delta": d, "logprobs": None,
                                                       "finish_reason": None}]}))
            for c in calls:
                self._chunk(_sse({**base, "choices": [{"index": i, "delta": {"tool_calls": [c]},
                                                       "logprobs": None,
                                                       "finish_reason": None}]}))

        def _generate(self, body: dict) -> None:
            t0 = time.perf_counter()
            groups, batch = srv.generate(body)
            items = [(r, plen) for reqs, plen in groups for r in reqs]
            if body.get("stream"):              # SGLang's: each event the whole text so far
                self._stream_start()
                try:
                    for r, plen in items:
                        text = ""
                        for delta in self._deltas(r):
                            text += delta
                            self._chunk(_sse(srv.gen_item(r, plen, text, t0, False)))
                        self._chunk(_sse(srv.gen_item(r, plen, text, t0, True)))
                    self._stream_end()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    for r, _ in items:
                        r.cancel()
                return
            try:
                self._wait([r for r, _ in items])
            except BaseException:
                for r, _ in items:
                    r.cancel()
                raise
            res = [srv.gen_item(r, plen, r.text, t0, True) for r, plen in items]
            self._json(200, res if batch or len(res) > 1 else res[0])

        def _completion(self, body: dict) -> None:
            reqs, plen, echo = srv.completion(body)
            cid = "cmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            pre = (echo if isinstance(echo, str) else
                   srv.tok.decode(echo) if echo is not None else "")
            if body.get("stream"):
                base = {"id": cid, "object": "text_completion", "created": created,
                        "model": srv.name}
                self._stream_start()
                try:
                    for i, r in enumerate(reqs):
                        if pre:
                            self._chunk(_sse({**base, "choices": [{"index": i, "text": pre,
                                                                   "logprobs": None,
                                                                   "finish_reason": None}]}))
                        for delta in self._deltas(r):
                            self._chunk(_sse({**base, "choices": [{"index": i, "text": delta,
                                                                   "logprobs": None,
                                                                   "finish_reason": None}]}))
                        fin = srv.finish(r) if r.finish != "error" else "error"
                        self._chunk(_sse({**base, "choices": [{"index": i, "text": "",
                                                               "logprobs": None,
                                                               "finish_reason": fin}]}))
                    if (body.get("stream_options") or {}).get("include_usage"):
                        self._chunk(_sse({**base, "choices": [],
                                          "usage": srv.usage(reqs, plen)}))
                    self._stream_end()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    for r in reqs:
                        r.cancel()
                return
            try:
                self._wait(reqs)
            except BaseException:
                for r in reqs:
                    r.cancel()
                raise
            self._json(200, {"id": cid, "object": "text_completion", "created": created,
                             "model": srv.name,
                             "choices": [{"index": i, "text": pre + r.text, "logprobs": None,
                                          "finish_reason": srv.finish(r)}
                                         for i, r in enumerate(reqs)],
                             "usage": srv.usage(reqs, plen)})

    return Handler


class HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True
    request_queue_size = 1024           # the listen backlog (socketserver's 5: 16 clients
                                        # connecting at once were reset on the card host)

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):
            return                              # a client gone: nothing to report
        super().handle_error(request, client_address)


def serve(srv: Server, host: str = "127.0.0.1", port: int = 8000) -> HTTPServer:
    """The HTTP server (not started: serve_forever(), or start it on a thread)."""
    return HTTPServer((host, port), make_handler(srv))


# ------------------------------------------------------------------------------ the command
def load(a):
    """(engine, tokenizer, spec, path, sampling defaults, weights label) for the arguments, as
    otpu-chat opens a model (the image cache, the card's configuration)."""
    from ..host import chat as CH
    from ..host import streamed as STR
    from ..host.board import ConfigMismatch
    from ..llm import formats as FM
    from ..llm import load_spec, model_dir
    from ..llm.qwen3 import Engine, load_weights
    path = model_dir(a.model)
    tokenizer = CH.load_tokenizer(path)
    spec = load_spec(path)
    wformat = FM.auto(spec, a.wformat)
    streams = STR.streams(spec)
    formats = None
    if streams and a.wformat == FM.AUTO:
        wformat, head, formats = STR.defaults(spec)
        a.head_format = a.head_format or head
    print(f"loading {path.name} onto openTPU ({a.backend}) ...", flush=True)
    try:
        backend, cfg = CH.make_backend(a.backend, spec, a.cap, a.dev, path.name, lookup=True,
                                       wformat=wformat, head_format=a.head_format)
    except ConfigMismatch as e:
        raise SystemExit(f"otpu-serve: {e}") from None
    if cfg is None and not streams and not CH._maps_on_design(spec):
        from ..host.board import sim_config
        from ..isasim import board_config
        cfg = sim_config(spec, a.cap, board_config(PAIR=True, DSTEP=True, STREAM=True),
                         lookup=True, wformat=wformat, head_format=a.head_format)
    t0 = time.perf_counter()
    if streams:
        from ..isasim import board_config
        mtp = {"mtp": True, "mtp_from": a.mtp_from} if getattr(a, "mtp", False) else {}
        eng = STR.engine(path, cfg or board_config(), backend, a.cap, a.pool or STR.DEFAULT,
                         experts=a.experts, wformat=wformat, head_format=a.head_format,
                         formats=formats, prog_cache=not a.no_prog_cache, snapshots=0, **mtp)
    elif getattr(a, "mtp", False):          # (a dense Qwen3.5 with its own mtp.* layer)
        from ..llm.mtp import mtp_engine
        eng = mtp_engine(spec, load_weights(path, mtp=True), cap=a.cap, cfg=cfg, backend=backend,
                         wformat=wformat, head_format=a.head_format,
                         prog_cache=not a.no_prog_cache, prompt_runs=not a.no_prompt_runs,
                         snapshots=0)
    else:
        B = a.slots if batchable(spec) else 1
        if B > 1 and cfg is None:           # the simulator: room for the slots and their area, one
            from dataclasses import replace  # slice as the card (sampled rows need it)

            from ..llm.qwen3 import PREFILL_ROWS, device_config
            c = device_config(spec, a.cap, batch=B, rows=PREFILL_ROWS, lookup=True,
                              wformat=wformat, head_format=a.head_format, S=1)
            cfg = replace(c, DRAM_BYTES=2 * c.DRAM_BYTES)
        eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, backend=backend,
                     wformat=wformat, head_format=a.head_format, resident=True,
                     prog_cache=not a.no_prog_cache, prompt_runs=not a.no_prompt_runs,
                     batch=B, batch_area=B > 1)
    weights = CH.weights_label(eng.image, wformat)
    print(f"weights: {weights}; loaded in {time.perf_counter() - t0:.1f} s", flush=True)
    tok = tokenizer()
    ns = argparse.Namespace(temperature=None, top_k=None, top_p=None, repetition_penalty=None)
    defaults = CH.sampling(spec, ns, path)
    return eng, tok, spec, path, defaults, weights


def log_warm(t, out=None) -> None:
    """The expert pool's read into the page cache (a PoolFile.warm thread) on `out` (stdout):
    a line now, and one when it ends (a watcher thread): its start and end, GB, MB/s, the
    pool's share in the page cache at its end (mincore), and what ended it early."""
    import threading

    from ..host.offload import warm_report
    if t is None:
        return
    hm = lambda x: time.strftime("%H:%M:%S", time.localtime(x))     # noqa: E731
    pr = lambda s: print(f"pool warm: {s}", file=out or sys.stdout, flush=True)   # noqa: E731
    r = warm_report(t)
    pr(f"started {hm(r['start'])}, {r['gb']:g} of {r['of_gb']:g} GB read"
       + ("" if r["end"] is None else ", ended"))

    def watch():
        t.join()
        r = warm_report(t)
        how = "done" if not r["error"] and not r["stopped"] and t.bytes >= t.total else \
            f"incomplete ({r['error'] or 'stopped'})"
        res = "?" if r["resident"] is None else f"{100 * r['resident']:.1f}%"
        pr(f"{how}: {hm(r['start'])} - {hm(r['end'])}, {r['gb']:g} GB, {r['mb_s']} MB/s; "
           f"{res} of the pool in the page cache")
    threading.Thread(target=watch, daemon=True, name="otpu-warm-log").start()


def batchable(spec) -> bool:
    """The model's image holds several sequences' KV caches (Qwen3 and the Llama-likes)."""
    from ..llm import qwen3
    return type(spec) is qwen3.Spec


def parser() -> argparse.ArgumentParser:
    """otpu-serve's arguments (otpu-serve35's server takes its defaults: serve35.py)."""
    from ..llm import MODELS
    from ..llm import formats as FM
    ap = argparse.ArgumentParser(prog="otpu-serve", description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--backend", default="isa", choices=["isa", "bend", "board", "board-sim"])
    ap.add_argument("--dev", help="XDMA device prefix (--backend board; default OTPU_DEV, else "
                    "the one openTPU card present, else /dev/xdma0)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=2048, help="the context's tokens at most")
    ap.add_argument("--served-model-name", help="the model's name in the API (default: the "
                                                "checkpoint directory's)")
    ap.add_argument("--wformat", default=FM.AUTO,
                    choices=[FM.AUTO, "int8", "fp4", "int4", "nvfp4", FM.MIX])
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4", "nvfp4"])
    ap.add_argument("--slots", type=int, default=8,
                    help="KV slots on the card: requests decoded together, batched (Qwen3 and "
                         "the Llama-likes; other models run one sequence, requests in turns)")
    ap.add_argument("--max-running", type=int, default=8,
                    help="without slots: requests that run in turns at once (the others "
                         "queue)")
    ap.add_argument("--chunked-prefill-size", type=int, default=64,
                    help="batched slots: prompt tokens fed per iteration, between decode steps "
                         "(more: shorter time to first token, longer pauses of the decoding "
                         "requests)")
    ap.add_argument("--mixed", action="store_true",
                    help="batched slots: feed a prompt in the spare rows of the decoding "
                         "requests' steps too (shorter worst pauses; on the card longer "
                         "typical ones and time to first token: docs/serve.md)")
    ap.add_argument("--quantum", type=int, default=SC.QUANTUM,
                    help="tokens a request decodes per turn while others wait")
    ap.add_argument("--max-queue", type=int, default=256)
    ap.add_argument("--kv-ram-gb", type=float, default=4.0,
                    help="host RAM for saved contexts (the KV store)")
    ap.add_argument("--kv-disk-gb", type=float, default=0.0,
                    help="disk for saved contexts past the RAM (0: none)")
    ap.add_argument("--kv-dir", default=str(Path.home() / "otpu-build" / "kvcache"),
                    help="the disk tier's directory (kept across restarts)")
    ap.add_argument("--prefix-block", type=int, default=SC.PREFIX_BLOCK, metavar="N",
                    help="a model with state: a prompt's prefix kept in the KV store every N "
                         "positions and at its last message boundary as it is fed, for other "
                         "conversations that share it (0: none)")
    ap.add_argument("--think", action="store_true",
                    help="thinking mode by default (a request's chat_template_kwargs "
                         "enable_thinking overrides it)")
    ap.add_argument("--reasoning-parser", choices=["qwen3"],
                    help="a reply's <think> ... </think> to reasoning_content")
    ap.add_argument("--preserve-thinking", action=argparse.BooleanOptionalAction, default=False,
                    help="every assistant turn rendered with its thinking, as the model wrote "
                         "it (a template with the switch: Qwen3.6's): a chat's context only "
                         "grows (a request's chat_template_kwargs preserve_thinking overrides)")
    ap.add_argument("--api-key", default=os.environ.get("OTPU_API_KEY"),
                    help="require 'Authorization: Bearer <key>' (default $OTPU_API_KEY)")
    ap.add_argument("--no-prog-cache", action="store_true")
    ap.add_argument("--no-prompt-runs", action="store_true")
    ap.add_argument("--pool", help="a streamed MoE's expert pool file (with --mtp the -mtp "
                                   "one: tools/offload/pack_pool.py extend)")
    ap.add_argument("--mtp", action="store_true",
                    help="Qwen3.5 (a streamed MoE with the -mtp pool, or dense): every request "
                         "decodes with the MTP drafter, the loop on the device "
                         "(opentpu/serve/scheduler.py; sampling the device's loop takes only)")
    ap.add_argument("--mtp-from", metavar="DIR",
                    help="with --mtp: the MTP drafter's checkpoint when the model has none "
                         "(MiniCPM-V 4.7: its base Qwen3.5-35B-A3B's)")
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--max-slices", type=int,
                    help="images (MiniCPM-V): an image's slices at most besides its thumbnail, "
                         "unless a request's max_slices says (default the processor's 9): the "
                         "latency knob, each crop about 3.7 s of vision at MCOLS 4 and 64 "
                         "prompt tokens")
    ap.add_argument("--image-fetch", action="store_true",
                    help="images: fetch http(s) image_urls (default: data: URLs only)")
    ap.add_argument("--warm-cap", type=float, default=WARM_CAP,
                    help="a streamed MoE: /health says 503 'warming' while its expert pool is "
                         "read into the page cache, at most this many seconds (0: never)")
    ap.add_argument("--telemetry", type=float, default=0.0, metavar="SEC",
                    help="--backend board: the card's telemetry for /metrics and the dashboard "
                         "(otpu-smi's reads: utilization, the governor's die, the BMC's power) "
                         "every SEC seconds on a thread of its own (0: off, the default)")
    ap.add_argument("--telemetry-i2c", action="store_true",
                    help="with --telemetry: read the card's I2C buses too (otpu-smi --i2c)")
    ap.add_argument("--demo", action="store_true",
                    help="no model, no card: the dashboard and /metrics over simulated "
                         "requests and a simulated card, each marked demo (docs/dashboard.md)")
    return ap


def main(argv=None):
    from ..host.runstate import busy_exits
    a = parser().parse_args(argv)
    if a.backend == "board" and not a.dev:
        from ..host.board import default_dev
        a.dev = default_dev()
    if a.demo:
        from .demo import main as demo
        return demo(a)
    return busy_exits(_main)(a)


def _main(a):
    eng, tok, spec, path, defaults, weights = load(a)
    from ..host import gen_warm as GW       # the decode loop's bucket programs ahead, in the
    eng.gen_warm = None if a.mtp else GW.start(eng, defaults)   # background (gen_warm.py; the
    #                                         MTP loop's: compiled at their first use)
    return run(a, eng, tok, spec, path, defaults, weights)


def card_monitor(a, worker, name: str, info: dict):
    """The worker's metrics.Monitor, with the card's telemetry (telemetry.CardSampler) when
    --telemetry SEC > 0 on the card, the cards' (telemetry.CardsSampler) for cards in tensor
    parallel (--dev "/dev/xdmaA+/dev/xdmaB"; a.card_names, comma-separated, their names); else
    why there is none (the dashboard's n/a)."""
    from . import metrics as M
    from . import telemetry as TM
    card, note = None, None
    if a.backend == "board" and a.telemetry > 0 and a.dev and "+" in a.dev:
        names = getattr(a, "card_names", None)
        card = TM.CardsSampler(a.dev.split("+"), a.telemetry, i2c=a.telemetry_i2c,
                               names=names.split(",") if names else None)
    elif a.backend == "board" and a.telemetry > 0:
        card = TM.CardSampler(a.dev, a.telemetry, i2c=a.telemetry_i2c)
    elif a.backend != "board":
        note = f"no card: the {a.backend} backend"
    else:
        note = "the card's telemetry is off (--telemetry 2 reads it)"
    mon = M.Monitor(worker, name, card=card, card_note=note, info=info)
    return mon


def store_key(a, eng, path) -> str:
    """The KV store's model key (its directory under --kv-dir): what decides a context's bytes;
    with slots under the channel hash, pages in the first sequence's lanes (kvio.Layout.relane:
    a store written before them is not read)."""
    return ST.model_key(str(Path(path).resolve()), type(eng.image).__module__, eng.image.wformat,
                        eng.image.head_format, getattr(eng.image, "formats", None), eng.cap,
                        eng.block, eng.cfg.S, eng.cfg.D, eng.cfg.MCOLS, eng.cfg.PAIR,
                        *(("mtp", str(a.mtp_from)) if a.mtp else ()),
                        *(("lanes",) if eng.batch > 1 and eng.cfg.CHASH else ()))


def run(a, eng, tok, spec, path, defaults, weights, on_idle=None, label: str = "otpu-serve"):
    """Serve an open engine until interrupted: the KV store, the worker (on_idle: called on
    its thread once its queue empties), the monitor, the HTTP server (otpu-serve35's too)."""
    mk = store_key(a, eng, path)
    store = ST.BlobStore(int(a.kv_ram_gb * 2**30), int(a.kv_disk_gb * 2**30),
                         a.kv_dir + f"/{mk}" if a.kv_disk_gb > 0 else None)
    from .batching import BatchWorker, batches
    from . import batching35 as BW35
    if BW35.batches(eng):               # (a Qwen3.5 image's sequence slots: their state too)
        w = BW35.BatchWorker35(eng, tok, store=store, model_key=mk, quantum=a.quantum,
                               prefix_block=a.prefix_block)
        t0 = time.perf_counter()        # (its steps' programs for greedy and the defaults')
        n = w.warm_steps([SC.Sampling(defaults["temperature"], defaults["top_k"],
                                      defaults["top_p"], defaults["repetition_penalty"])])
        if n:
            print(f"{label}: {n} step programs live in {time.perf_counter() - t0:.1f} s",
                  flush=True)
        w.start()
    elif batches(eng):
        w = BatchWorker(eng, tok, store=store, model_key=mk, quantum=a.quantum,
                        prefill_budget=a.chunked_prefill_size, mixed=a.mixed).start()
    else:
        w = SC.Worker(eng, tok, store=store, model_key=mk, max_running=a.max_running,
                      quantum=a.quantum, mtp=a.mtp, prefix_block=a.prefix_block)
        w.on_idle = on_idle
        w.start()
    name = a.served_model_name or Path(path).name
    info = {"path": str(path), "backend": a.backend, "weights": weights,
            "cap": a.cap, "kv_page_bytes": w.io.layout.page_bytes(),
            "staged": w.io.area is not None, "slots": eng.batch,
            "images": getattr(getattr(eng, "image", None), "vision", None) is not None,
            "warm": PS.startup(eng, label)}
    mon = card_monitor(a, w, name, dict(info, kv_ram_bytes=store.ram_bytes,
                                        kv_disk_bytes=store.disk_bytes, label=label,
                                        dram_bytes=getattr(eng.cfg, "DRAM_BYTES", None),
                                        image_bytes=getattr(eng.image, "nbytes", None)))
    srv = Server(w, tok, spec, name, defaults, think=a.think, reasoning=a.reasoning_parser,
                 aliases=(a.model,), api_key=a.api_key, max_queue=a.max_queue, info=info,
                 max_slices=a.max_slices, fetch=a.image_fetch, warm_cap=a.warm_cap,
                 monitor=mon, preserve_thinking=a.preserve_thinking)
    if srv.pool is not None:
        log_warm(srv.pool.warm_t)
    httpd = serve(srv, a.host, a.port)
    mon.start()
    if mon.card is not None:
        mon.card.start()
    print(f"{label}: {name} on http://{a.host}:{a.port}/v1 (context {a.cap}, KV store "
          f"{a.kv_ram_gb:g} GB RAM + {a.kv_disk_gb:g} GB disk, sampling {defaults}); the "
          f"dashboard on http://{a.host}:{a.port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        mon.stop()
        w.stop()


if __name__ == "__main__":
    sys.exit(main())
