"""Prefixes kept for other conversations (a model with state: store.PrefixCache.match takes a saved
context's end only): the scheduler keeps a prompt's prefix in the store as it is fed, at its
block positions (Worker.prefix_block) and at the mark the API gives (the last message boundary
before the final user turn: an agent's tools and system prompt); another conversation sharing
it restores it and feeds only the rest, its tokens a fresh run's; one whose system prompt's tail
differs (today's date) restores the last block before; the API's mark at the final user turn's
<|im_start|> (Qwen3.6's template). Tiny Qwen3.5 (DeltaNet states) on the ISA simulator."""
from types import SimpleNamespace

import pytest

from opentpu.llm.qwen3 import Engine
from opentpu.serve import api as API
from opentpu.serve import scheduler as SC

from test_serve_kvio import _model
from test_serve_scheduler import CAP, _Tok, _prompts, _ref
from test_serve_tools import CT, TEMPLATE, TOOLS, _qwen_tok, _Script

torch = pytest.importorskip("torch")


def test_a_new_conversation_restores_the_shared_prefix():
    """Conversation A (a 150-token system prompt, then its question): its prefix kept at the
    blocks 64 and 128 and at its mark 150 as its prompt is fed (no feed restarted: the prompt
    in parts). B (the same system prompt, another question) restores 150 and feeds only its
    question; D (the system prompt's last tokens changed past 140) restores the block at 128.
    Every reply is its fresh run's; the store counts the prefix hits and their tokens."""
    W, spec, cfg = _model("qwen35")
    stop = frozenset()
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True, snapshots=1)
    w = SC.Worker(eng, _Tok(), quantum=64, prefix_block=64, prefix_skip=16).start()
    try:
        system = _prompts(30, [150])[0]
        qa, qb, qd = _prompts(31, [12, 9, 10])
        a = system + qa
        ra = w.submit(SC.Request(a, max_new=5, stop_ids=stop, at=len(a) - 2,
                                 marks=(150,))).wait(600)
        assert ra.error is None and ra.out == _ref(spec, W, cfg, a, 5, stop)
        kept = sorted(e.n for e in w.cache.entries.values() if e.prefix)
        assert kept == [64, 128, 150] and w.stats["prefix_saves"] == 3
        assert ra.phases.get("prefix_save", 0) > 0 and ra.cached == 0
        assert w.usage()["last_io"]["prefix_save"]["link_bytes"] > 0
        b = system + qb
        rb = w.submit(SC.Request(b, max_new=5, stop_ids=stop, at=len(b) - 2,
                                 marks=(150,))).wait(600)
        assert rb.cached == 150 and rb.out == _ref(spec, W, cfg, b, 5, stop)
        pc = w.cache.usage()
        assert pc["prefix_hits"] == 1 and pc["prefix_hit_tokens"] == 150
        assert w.stats["prefix_saves"] == 3          # (B fed from 150: nothing new to keep)
        d = system[:140] + _prompts(32, [10])[0] + qd
        rd = w.submit(SC.Request(d, max_new=5, stop_ids=stop, at=len(d) - 2,
                                 marks=(150,))).wait(600)
        assert rd.cached == 128 and rd.out == _ref(spec, W, cfg, d, 5, stop)
        pc = w.cache.usage()
        assert pc["prefix_hits"] == 2 and pc["prefix_hit_tokens"] == 278
        assert w.stats["prefix_saves"] == 4          # (D's own system prompt's end, 150)
        # off: prefix_block 0 keeps none
        w2 = SC.Worker(Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True,
                              snapshots=1), _Tok(), prefix_block=0).start()
        try:
            r = w2.submit(SC.Request(a, max_new=2, stop_ids=stop, marks=(150,))).wait(600)
            assert r.out == ra.out[:2] and w2.stats["prefix_saves"] == 0
        finally:
            w2.stop()
    finally:
        w.stop()


def test_prefix_cuts_skip_the_feeds_start():
    """The cuts: the mark and the blocks past where the feed starts by more than prefix_skip,
    short of the prompt's last fed position; none for a model without state."""
    w = SimpleNamespace(stateful=True, prefix_block=64, prefix_skip=16)
    cuts = lambda r, n, last: SC.Worker._prefix_cuts(w, r, n, last)
    r = SimpleNamespace(marks=(150,))
    assert cuts(r, 0, 170) == {64, 128, 150}
    assert cuts(r, 50, 170) == {128, 150}            # (64 within 16 of 50)
    assert cuts(r, 140, 170) == set()                # (150 within 16 of 140)
    assert cuts(r, 0, 150) == {64, 128}              # (the last part: past 150)
    w.prefix_block = 0
    assert cuts(r, 0, 170) == set()
    w.stateful, w.prefix_block = False, 64
    assert cuts(r, 0, 170) == set()


class _STok:
    """Characters, but <|im_start|> and <|im_end|> one special token each (ids 1, 2)."""
    chat_template = TEMPLATE
    all_special_ids = [1, 2]
    SP = {"<|im_start|>": 1, "<|im_end|>": 2}

    def __call__(self, text):
        out, i = [], 0
        while i < len(text):
            t = next((s for s in self.SP if text.startswith(s, i)), None)
            out.append(self.SP[t] if t else ord(text[i]) + 16)
            i += len(t) if t else 1
        return {"input_ids": out}

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=False,
                            enable_thinking=True, tokenize=True):
        text = CT.render_jinja_template(conversations=[msgs], tools=tools,
                                        chat_template=TEMPLATE,
                                        add_generation_prompt=add_generation_prompt,
                                        enable_thinking=enable_thinking)[0][0]
        return self(text)["input_ids"] if tokenize else text

    def decode(self, ids, skip_special_tokens=True):
        inv = {v: k for k, v in self.SP.items()}
        return "".join(inv.get(i) or chr(i - 16) for i in ids)


def _server(tok, stateful=True):
    w = _Script(lambda p: "ok", stateful=stateful)
    srv = API.Server(w, tok, SimpleNamespace(eos=[10 ** 7]), "qwen36-35b",
                     dict(temperature=0.0, top_k=20, top_p=1.0, repetition_penalty=1.0))
    return srv, w


AGENT = [{"role": "system", "content": "You are a coding agent. Today's date: 2026-11-03."},
         {"role": "user", "content": "What does hello.py print?"}]
LATER = AGENT + [
    {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                          "function": {"name": "read",
                                                                       "arguments": "{\"filePath\": \"/w/hello.py\"}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "1: print('hello')"}]


def test_the_api_marks_the_final_user_turns_start():
    """The mark: the <|im_start|> of the final user turn (the tools and the system prompt
    before it), the same for the agent's later requests in that conversation (tool results
    after the user turn); none without a message before the user's, for a model without
    state, or with a tokenizer without special tokens. With tools a reply that calls one
    stays on the device (keep_reply)."""
    tok = _STok()
    srv, w = _server(tok)
    for msgs in (AGENT, LATER):
        srv.chat({"messages": msgs, "tools": TOOLS})
        r = w.got[-1]
        text = tok.decode(r.prompt)
        want = len(tok(text[:text.index("<|im_start|>user")])["input_ids"])
        assert r.marks == (want,) and r.prompt[want] == 1
        assert r.keep_reply == "<tool_call>"
    srv.chat({"messages": AGENT})                     # (no tools: the system prompt's end)
    r = w.got[-1]
    assert r.marks and tok.decode(r.prompt[:r.marks[0]]).endswith(
        "Today's date: 2026-11-03.<|im_end|>\n")
    assert r.keep_reply is None
    srv.chat({"messages": AGENT[1:]})
    assert w.got[-1].marks == ()
    srv, w = _server(tok, stateful=False)
    srv.chat({"messages": AGENT, "tools": TOOLS})
    assert w.got[-1].marks == ()


def test_the_api_marks_in_qwen_tokens():
    """With Qwen3.6's tokenizer: the mark's token is <|im_start|>, "user" after it."""
    tok = _qwen_tok()
    if tok is None:
        pytest.skip("no Qwen3.6 tokenizer ($OTPU_QWEN36_TOKENIZER or models/Qwen3.5-0.8B)")
    srv, w = _server(tok)
    srv.chat({"messages": AGENT, "tools": TOOLS})
    r = w.got[-1]
    (b,) = r.marks
    assert tok.convert_ids_to_tokens(r.prompt[b]) == "<|im_start|>"
    assert tok.decode(r.prompt[b:b + 3]) == "<|im_start|>user\n"
