"""otpu-serve's tool calling (opentpu/serve/tools.py, api.py) with Qwen3.6's chat template
(tests/data/qwen36_chat_template.jinja: Qwen3.6-35B-A3B's, Apache-2.0) over HTTP, a scripted
worker's replies in place of the model's: the tools rendered (tool_choice "none": not), the
qwen3_coder calls parsed (one, several, typed values), the stream's chunks in order, an
assistant's null content with tool_calls and tool messages accepted, and the next request's
prompt continuing the one before it through the call the model wrote (the store's and the
device's common prefix): characters here, Qwen3.6's tokens when its tokenizer is at
$OTPU_QWEN36_TOKENIZER or models/Qwen3.5-0.8B (the same tokenizer.json)."""
import http.client
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from opentpu.serve import api as API
from opentpu.serve import scheduler as SC
from opentpu.serve import tools as TL

CT = pytest.importorskip("transformers.utils.chat_template_utils")

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = (ROOT / "tests" / "data" / "qwen36_chat_template.jinja").read_text()

TOOLS = [
    {"type": "function", "function": {
        "name": "read", "description": "Read a file.",
        "parameters": {"type": "object", "properties": {
            "filePath": {"type": "string", "description": "The file's absolute path"},
            "offset": {"type": "integer"}, "limit": {"type": ["integer", "null"]}},
            "required": ["filePath"]}, "strict": False}},
    {"type": "function", "function": {
        "name": "shell", "description": "Run a command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"}, "timeout": {"type": "number"},
            "background": {"type": "boolean"}, "env": {"type": "object"},
            "args": {"type": "array", "items": {"type": "string"}}}}}},
]

READ = ("<tool_call>\n<function=read>\n<parameter=filePath>\n/work/hello.py\n</parameter>\n"
        "<parameter=limit>\n50\n</parameter>\n</function>\n</tool_call>")
SHELL = ("<tool_call>\n<function=shell>\n<parameter=command>\nfor f in *.py; do\n  wc -l \"$f\"\n"
         "done\n\n</parameter>\n<parameter=timeout>\n2.5\n</parameter>\n<parameter=background>\n"
         "false\n</parameter>\n<parameter=env>\n{\"LANG\": \"C\", \"N\": 2}\n</parameter>\n"
         "<parameter=args>\n[\"-l\", \"x y\"]\n</parameter>\n</function>\n</tool_call>")
TWO = "I'll look at both.\n\n" + READ + "\n" + SHELL


class _Tok:
    """A tokenizer of code points (an id a character) over the real template (transformers'
    renderer, as apply_chat_template's)."""
    chat_template = TEMPLATE

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(self, msgs, tools=None, add_generation_prompt=False,
                            enable_thinking=True, tokenize=True, **kw):
        text = CT.render_jinja_template(conversations=[msgs], tools=tools,
                                        chat_template=self.chat_template,
                                        add_generation_prompt=add_generation_prompt,
                                        enable_thinking=enable_thinking, **kw)[0][0]
        return self(text)["input_ids"] if tokenize else text

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


class _Script:
    """scheduler.Worker's part the API uses: each request's reply reply(prompt text), streamed
    `step` characters at a time."""

    def __init__(self, reply, stateful=False, step=3):
        self.reply, self.stateful, self.step = reply, stateful, step
        self.tok_lock = threading.Lock()
        self.eng = SimpleNamespace(cap=1 << 20)
        self.got = []

    def busy(self):
        return 0

    def submit(self, r):
        self.got.append(r)
        text = self.reply("".join(map(chr, r.prompt)))
        r.out, r.text, r.finish = [ord(c) for c in text], text, "stop"
        for k in range(0, len(text), self.step):
            r.events.put(("delta", text[k:k + self.step]))
        r.events.put(("done", "stop"))
        return r


@pytest.fixture
def server():
    made = []

    def make(reply, tok=None, stateful=False, step=3):
        w = _Script(reply if callable(reply) else (lambda p: reply), stateful, step)
        srv = API.Server(w, tok or _Tok(), SimpleNamespace(eos=[10 ** 7]), "qwen36-35b",
                         dict(temperature=0.0, top_k=20, top_p=1.0, repetition_penalty=1.0))
        httpd = API.serve(srv, "127.0.0.1", 0)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        made.append(httpd)
        return srv, httpd.server_address[1], w
    yield make
    for h in made:
        h.shutdown()
        h.server_close()


def _call(port, body, path="/v1/chat/completions"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse()
    data = json.loads(r.read())
    c.close()
    return r.status, data


def _stream(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200
    out = [x[6:] for x in r.read().decode().split("\n") if x.startswith("data: ")]
    c.close()
    return ["DONE" if x == "[DONE]" else json.loads(x) for x in out]


def _ask(text="What does hello.py print?"):
    return [{"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": text}]


def _text(srv, body) -> str:
    return "".join(map(chr, srv.tokenize(body)["tokens"]))


def test_the_tools_are_rendered_and_tool_choice_none_renders_none(server):
    """The request's {type, function} dicts each one line of the template's <tools> (tojson:
    the dict as given), the system text after the format's notes; tool_choice none: the
    template's plain system message (as without tools)."""
    srv, port, w = server("ok")
    t = _text(srv, {"messages": _ask(), "tools": TOOLS})
    assert t.startswith("<|im_start|>system\n# Tools\n\nYou have access to the following "
                        "functions:\n\n<tools>\n" + json.dumps(TOOLS[0], ensure_ascii=False) +
                        "\n" + json.dumps(TOOLS[1], ensure_ascii=False) + "\n</tools>")
    assert "</IMPORTANT>\n\nYou are a coding agent.<|im_end|>\n<|im_start|>user\n" in t
    assert t.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    plain = _text(srv, {"messages": _ask()})
    assert _text(srv, {"messages": _ask(), "tools": TOOLS, "tool_choice": "none"}) == plain
    assert plain.startswith("<|im_start|>system\nYou are a coding agent.<|im_end|>")
    for choice in ("auto", "required", {"type": "function", "function": {"name": "read"}}):
        assert _text(srv, {"messages": _ask(), "tools": TOOLS, "tool_choice": choice}) == t
    for bad in ({"tools": {"read": 1}}, {"tools": [{"type": "function"}]},
                {"tools": [{"type": "function", "function": {"name": ""}}]}):
        assert _call(port, {"messages": _ask(), **bad})[0] == 400


def test_calls_parsed_with_their_types():
    """One call and several: each value by its parameter's type (a string as is, newlines
    inside kept, the format's one either side dropped; integer, number, boolean, object, array
    as JSON; a value that is not its type's JSON, or a parameter the schema does not name, the
    string); the text before the first call the content."""
    content, calls = TL.parse(READ, TOOLS)
    assert content is None and calls == [{"name": "read", "arguments": {
        "filePath": "/work/hello.py", "limit": 50}}]
    content, calls = TL.parse(TWO, TOOLS)
    assert content == "I'll look at both." and [c["name"] for c in calls] == ["read", "shell"]
    assert calls[1]["arguments"] == {"command": "for f in *.py; do\n  wc -l \"$f\"\ndone\n",
                                     "timeout": 2.5, "background": False,
                                     "env": {"LANG": "C", "N": 2}, "args": ["-l", "x y"]}
    assert list(calls[1]["arguments"]) == ["command", "timeout", "background", "env", "args"]
    odd = READ.replace("50", "fifty").replace("</function>",
                                              "<parameter=extra>\n7\n</parameter>\n</function>")
    assert TL.parse(odd, TOOLS)[1][0]["arguments"] == {"filePath": "/work/hello.py",
                                                       "limit": "fifty", "extra": "7"}
    # a reply cut short inside its call: what it has; text without a call: all content
    assert TL.parse(READ[:READ.index("<parameter=limit>")], TOOLS)[1] == [
        {"name": "read", "arguments": {"filePath": "/work/hello.py"}}]
    assert TL.parse("No call: <tool_call> is the tag.", TOOLS) == (
        "No call: <tool_call> is the tag.", [])
    assert TL.parse("Plain answer.\n", TOOLS) == ("Plain answer.\n", [])
    one = TL.openai(calls[1])
    assert one["id"].startswith("call_") and one["type"] == "function"
    assert json.loads(one["function"]["arguments"]) == calls[1]["arguments"]


def test_a_reply_with_calls_over_http(server):
    """message.tool_calls (arguments JSON strings), its content the text before them (null
    without), finish_reason tool_calls; tool_choice none: the reply's text as content."""
    srv, port, w = server(TWO)
    st, r = _call(port, {"messages": _ask(), "tools": TOOLS})
    assert st == 200, r
    ch = r["choices"][0]
    assert ch["finish_reason"] == "tool_calls" and ch["message"]["content"] == "I'll look at both."
    tc = ch["message"]["tool_calls"]
    assert [c["function"]["name"] for c in tc] == ["read", "shell"]
    assert json.loads(tc[0]["function"]["arguments"]) == {"filePath": "/work/hello.py",
                                                          "limit": 50}
    assert len({c["id"] for c in tc}) == 2 and all(c["type"] == "function" for c in tc)
    srv, port, w = server(READ)
    ch = _call(port, {"messages": _ask(), "tools": TOOLS})[1]["choices"][0]
    assert ch["message"]["content"] is None and len(ch["message"]["tool_calls"]) == 1
    ch = _call(port, {"messages": _ask(), "tools": TOOLS, "tool_choice": "none"})[1]["choices"][0]
    assert ch["finish_reason"] == "stop" and ch["message"]["content"] == READ
    assert "tool_calls" not in ch["message"]
    ch = _call(port, {"messages": _ask()})[1]["choices"][0]           # (no tools: as before)
    assert ch["message"] == {"role": "assistant", "content": READ}


@pytest.mark.parametrize("step", [1, 2, 5, 64])
def test_the_stream_holds_the_call_back_until_it_closes(server, step):
    """The role chunk, the content's deltas (no part of a <tool_call> tag in them, nor the
    whitespace before a call), a tool_calls delta per call when its </tool_call> arrives
    (index, id, type, name, the whole arguments), the finish chunk (tool_calls), the usage,
    [DONE]; the plain reply's content and calls."""
    srv, port, w = server(TWO, step=step)
    ev = _stream(port, {"messages": _ask(), "tools": TOOLS,
                        "stream_options": {"include_usage": True}})
    assert ev[-1] == "DONE" and ev[-2]["choices"] == [] and ev[-2]["usage"]["completion_tokens"]
    deltas = [e["choices"][0] for e in ev[:-2]]
    assert deltas[0]["delta"] == {"role": "assistant", "content": ""}
    assert deltas[-1]["finish_reason"] == "tool_calls" and deltas[-1]["delta"] == {}
    body = [d["delta"] for d in deltas[1:-1]]
    kinds = ["c" if "content" in d else "t" for d in body]
    assert kinds == sorted(kinds) and kinds.count("t") == 2       # (content, then the calls)
    content = "".join(d.get("content", "") for d in body)
    assert content == "I'll look at both." and not any("<" in d.get("content", "") for d in body)
    calls = [d["tool_calls"][0] for d in body if "tool_calls" in d]
    assert [c["index"] for c in calls] == [0, 1]
    assert [c["function"]["name"] for c in calls] == ["read", "shell"]
    want = TL.parse(TWO, TOOLS)[1]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [c["arguments"] for c in want]
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in calls)
    # without a call: the text whole (its last whitespace too), finish stop
    srv, port, w = server("Just text, no call.\n\n", step=step)
    ev = _stream(port, {"messages": _ask(), "tools": TOOLS})
    assert "".join(e["choices"][0]["delta"].get("content", "") for e in ev[:-1]) == \
        "Just text, no call.\n\n"
    assert ev[-2]["choices"][0]["finish_reason"] == "stop"


def _turn2(calls, content=None, results=("1: print('hello')",)):
    """The agent's next request: the call(s) as the client sends them back (content null,
    arguments JSON strings), a tool message per call (one with content parts)."""
    tc = [TL.openai(c) for c in calls]
    msgs = _ask() + [{"role": "assistant", "content": content, "tool_calls": tc}]
    for c, res in zip(tc, results):
        msgs.append({"role": "tool", "tool_call_id": c["id"],
                     "content": [{"type": "text", "text": res}]})
    return msgs


def test_null_content_and_tool_messages_are_accepted(server):
    """An assistant message with tool_calls and content null or "", then tool messages (a
    string or text parts): rendered as the template writes them, the tool results in one user
    turn; a call whose arguments are not a JSON object is refused."""
    srv, port, w = server("It prints hello.")
    calls = TL.parse(TWO, TOOLS)[1]
    for content in (None, ""):
        msgs = _turn2(calls, content, results=("1: print('hello')", "3 hello.py"))
        msgs[-1]["content"] = "3 hello.py"
        st, r = _call(port, {"messages": msgs, "tools": TOOLS})
        assert st == 200 and r["choices"][0]["message"]["content"] == "It prints hello."
        t = _text(srv, {"messages": msgs, "tools": TOOLS})
        assert ("<|im_start|>assistant\n<think>\n\n</think>\n\n" + READ + "\n" + SHELL +
                "<|im_end|>\n<|im_start|>user\n<tool_response>\n1: print('hello')\n"
                "</tool_response>\n<tool_response>\n3 hello.py\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n") in t
    for bad in ("not json", "[1, 2]"):
        msgs = _turn2(calls[:1])
        msgs[2]["tool_calls"][0]["function"]["arguments"] = bad
        st, r = _call(port, {"messages": msgs, "tools": TOOLS})
        assert st == 400 and "arguments" in r["error"]["message"]


@pytest.mark.parametrize("reply", [READ, "I'll read it first.\n\n" + READ, TWO, SHELL],
                         ids=["read", "text-read", "two", "shell"])
def test_the_next_request_continues_the_context(server, reply):
    """The agent's next request re-renders the call from its parsed arguments as the model wrote
    it: its prompt starts with the last prompt + the reply + <|im_end|>, so the context the
    device or the store holds is its common prefix (only the tool results are fed)."""
    srv, port, w = server(reply)
    p1 = _text(srv, {"messages": _ask(), "tools": TOOLS})
    content, calls = TL.parse(reply, TOOLS)
    p2 = _text(srv, {"messages": _turn2(calls, content, ("ok",) * len(calls)), "tools": TOOLS})
    assert p2.startswith(p1 + reply + "<|im_end|>\n")


def test_a_saved_context_keeps_the_tools(server):
    """A stateful model's reply start (where the next user turn's rendering parts from this
    one) renders the stand-in turn with the request's tools: past the tools block, at the
    generation prompt's <think> (not at 0)."""
    srv, port, w = server("ok", stateful=True)
    st, r = _call(port, {"messages": _ask(), "tools": TOOLS})
    p = _text(srv, {"messages": _ask(), "tools": TOOLS})
    assert st == 200 and w.got[-1].at == len(p) - len("<think>\n\n</think>\n\n")


def _qwen_tok():
    p = Path(os.environ.get("OTPU_QWEN36_TOKENIZER") or ROOT / "models" / "Qwen3.5-0.8B")
    if not (p / "tokenizer.json").exists():
        return None
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(p))
    tok.chat_template = TEMPLATE
    return tok


@pytest.mark.parametrize("reply", [READ, "I'll read it first.\n\n" + READ, TWO],
                         ids=["read", "text-read", "two"])
def test_the_next_request_continues_the_context_in_qwen_tokens(server, reply):
    """As above in Qwen3.6's tokens: the last prompt's ids + the reply's (its encoding) +
    <|im_end|> are the next prompt's first ids."""
    tok = _qwen_tok()
    if tok is None:
        pytest.skip("no Qwen3.6 tokenizer ($OTPU_QWEN36_TOKENIZER or models/Qwen3.5-0.8B)")
    srv, port, w = server(reply, tok=tok)
    p1 = srv.tokenize({"messages": _ask(), "tools": TOOLS})["tokens"]
    content, calls = TL.parse(reply, TOOLS)
    p2 = srv.tokenize({"messages": _turn2(calls, content, ("ok",) * len(calls)),
                       "tools": TOOLS})["tokens"]
    gen = tok(reply, add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids(
        "<|im_end|>")]
    assert p2[:len(p1) + len(gen)] == p1 + gen


ANSWER = "It prints hello."
NEXT = "And with python2?"


def _session(think=False):
    """An opencode-shaped session's three requests: the first (tools, system prompt, question);
    after its call, the call and its result; after the answer, the user's next message. With
    thinking, each reply's reasoning sent back as its reasoning_content."""
    content, calls = TL.parse(READ, TOOLS)
    m2 = _turn2(calls, content)
    if think:
        m2[2]["reasoning_content"] = "Read it."
    m3 = m2 + [{"role": "assistant", "content": ANSWER,
                **({"reasoning_content": "It prints."} if think else {})},
               {"role": "user", "content": NEXT}]
    return _ask(), m2, m3


@pytest.mark.parametrize("think", [False, True], ids=["plain", "think"])
def test_with_preserve_thinking_a_session_feeds_only_whats_new(server, think):
    """--preserve-thinking (otpu-serve35's default): every assistant turn rendered with its
    think block as the model wrote it, so each request's prompt is the last one's, its reply
    and <|im_end|>, then only the new part (the call's result; the user's next message): the
    device's context continues, nothing fed again; a model with state keeps every reply
    (keep_reply ""), its reply start at the prompt's last position (where a reply tokenized
    otherwise parts). Without it the user's next message parts at the first
    turn's reply (the template drops the earlier turns' empty think blocks), and a request's
    chat_template_kwargs preserve_thinking sets it either way."""
    srv, port, w = server("ok", stateful=True)
    srv.preserve = True
    kw = {"chat_template_kwargs": {"enable_thinking": think}}
    p1, p2, p3 = (_text(srv, {"messages": m, "tools": TOOLS, **kw}) for m in _session(think))
    r1 = ("Read it.\n</think>\n\n" if think else "") + READ
    r2 = ("It prints.\n</think>\n\n" if think else "") + ANSWER
    head = "<|im_start|>assistant\n<think>\n" + ("" if think else "\n</think>\n\n")
    assert p2 == (p1 + r1 + "<|im_end|>\n<|im_start|>user\n<tool_response>\n1: print('hello')"
                  "\n</tool_response><|im_end|>\n" + head)
    assert p3 == p2 + r2 + "<|im_end|>\n<|im_start|>user\n" + NEXT + "<|im_end|>\n" + head
    st, r = _call(port, {"messages": _session(think)[2], "tools": TOOLS, **kw})
    assert st == 200 and w.got[-1].at == len(p3) - 1 and w.got[-1].keep_reply == ""
    off = {"chat_template_kwargs": {"enable_thinking": think, "preserve_thinking": False}}
    q3 = _text(srv, {"messages": _session(think)[2], "tools": TOOLS, **off})
    assert not q3.startswith(p2 + r2)
    first = p1.rindex("<|im_start|>assistant\n") + len("<|im_start|>assistant\n")
    assert q3[:first] == p3[:first] and p3[first:].startswith("<think>\n")
    assert q3[first:].startswith("<tool_call>")              # (its think block dropped)
    srv.preserve = False                                     # (otpu-serve's default)
    assert not API.parser().parse_args([]).preserve_thinking
    assert _text(srv, {"messages": _session(think)[2], "tools": TOOLS, **kw}) == q3
    on = {"chat_template_kwargs": {"enable_thinking": think, "preserve_thinking": True}}
    assert _text(srv, {"messages": _session(think)[2], "tools": TOOLS, **on}) == p3


def test_with_preserve_thinking_a_session_feeds_only_whats_new_in_qwen_tokens(server):
    """As above in Qwen3.6's tokens (thinking off): each prompt's ids are the last prompt's,
    its reply's encoding and <|im_end|>, then the new part; the reply start the prompt's last
    position."""
    tok = _qwen_tok()
    if tok is None:
        pytest.skip("no Qwen3.6 tokenizer ($OTPU_QWEN36_TOKENIZER or models/Qwen3.5-0.8B)")
    srv, port, w = server("ok", tok=tok, stateful=True)
    srv.preserve = True
    p1, p2, p3 = (srv.tokenize({"messages": m, "tools": TOOLS})["tokens"] for m in _session())
    end = [tok.convert_tokens_to_ids("<|im_end|>")]
    for p, reply, q, new in ((p1, READ, p2, "<|im_start|>user\n<tool_response>"),
                             (p2, ANSWER, p3, "<|im_start|>user\n" + NEXT)):
        k = len(p) + len(tok(reply, add_special_tokens=False)["input_ids"]) + 1
        assert q[:k] == p + tok(reply, add_special_tokens=False)["input_ids"] + end
        assert tok.decode(q[k:]).lstrip("\n").startswith(new)
    st, r = _call(port, {"messages": _session()[2], "tools": TOOLS})
    assert st == 200 and w.got[-1].at == len(p3) - 1 and w.got[-1].keep_reply == ""


# ---- LFM2's format (tests/data/lfm2_chat_template.jinja: LFM2-2.6B's, LFM Open License v1.0)

LFM_READ = "<|tool_call_start|>[read(path='hello.py', limit=20)]<|tool_call_end|>"
LFM_TWO = "<|tool_call_start|>[read(path='a b.py'), shell(command='echo \\'hi\\'\\ncat x')]<|tool_call_end|>"


class _LfmTok(_Tok):
    chat_template = (ROOT / "tests" / "data" / "lfm2_chat_template.jinja").read_text()

    def apply_chat_template(self, msgs, **kw):
        return super().apply_chat_template(msgs, bos_token="<|startoftext|>", **kw)


def test_the_template_names_its_format():
    assert TL.dialect(_LfmTok.chat_template) is TL.LFM2
    assert TL.dialect(TEMPLATE) is TL.QWEN3_CODER and TL.dialect(None) is TL.QWEN3_CODER


def test_lfm_tools_are_rendered_and_its_calls_parsed(server):
    """The tools in the system message's tool list, a reply's calls in one block with their
    values typed, the finish reason tool_calls; plain text stays content."""
    srv, port, w = server(LFM_READ, tok=_LfmTok())
    t = _text(srv, {"messages": _ask(), "tools": TOOLS})
    assert t.startswith("<|startoftext|><|im_start|>system\nYou are a coding agent.\nList of tools: "
                        "<|tool_list_start|>[" + json.dumps(TOOLS[0]))
    assert t.endswith("<|tool_list_end|><|im_end|>\n<|im_start|>user\nWhat does hello.py print?"
                      "<|im_end|>\n<|im_start|>assistant\n")
    st, r = _call(port, {"messages": _ask(), "tools": TOOLS})
    c = r["choices"][0]
    assert st == 200 and c["finish_reason"] == "tool_calls" and c["message"]["content"] is None
    f = c["message"]["tool_calls"][0]["function"]
    assert f["name"] == "read" and json.loads(f["arguments"]) == {"path": "hello.py", "limit": 20}
    assert TL.parse("no calls here", TOOLS, TL.LFM2) == ("no calls here", [])


@pytest.mark.parametrize("block, calls", [
    ("[f()]", [("f", {})]),
    ("f(a='x')", [("f", {"a": "x"})]),
    ("[f(a='x', b=1), g(c=[1, 2], d={'e': None})]", [("f", {"a": "x", "b": 1}), ("g", {"c": [1, 2], "d": {"e": None}})]),
    ("[f(a=true, b=false, c=null)]", [("f", {"a": True, "b": False, "c": None})]),
    ("[f(d=[true, null])]", [("f", {"d": [True, None]})]),
    ('[f(a="double \'q\'")]', [("f", {"a": "double 'q'"})]),
    ("[f(a=-1.5)]", [("f", {"a": -1.5})]),
    ("[functions.f(a=1)]", [("functions.f", {"a": 1})]),
    ("[f(a='one\ntwo')]", [("f", {"a": "one\ntwo"})]),       # (a raw newline in the string)
    ('{"name": "f", "arguments": {"a": 1}}', [("f", {"a": 1})]),
    ('[{"name": "f", "arguments": "{\\"a\\": 1}"}, {"name": "g"}]', [("f", {"a": 1}), ("g", {})]),
], ids=["empty", "no-list", "several", "json-names", "json-nested", "double-quotes", "negative", "dotted", "raw-newline",
        "json", "json-list"])
def test_lfm_blocks_read_as_calls(block, calls):
    assert [(c["name"], c["arguments"]) for c in TL.LFM2.calls(block, {})] == calls


@pytest.mark.parametrize("block", ["", "[]", "hello there", "[f(a=", "[f(a=os.system('x'))]", "[f(*a)]", "[f(**a)]",
                                   "[1, 2]", "[f(a=b)]", "{not json", '{"arguments": {}}', "__import__('os')()"])
def test_lfm_blocks_that_are_not_calls_are_content(block):
    """Nothing is evaluated; a block that is no call list (or an empty one) stays the reply's text."""
    t = "<|tool_call_start|>" + block + "<|tool_call_end|>"
    assert TL.parse(t, TOOLS, TL.LFM2) == (t, [])


@pytest.mark.parametrize("block, calls", [
    ("[shell(command=ls -la /tmp), read(filePath=/work/a.txt)]",     # (LFM2-2.6B's own, fp32)
     [("shell", {"command": "ls -la /tmp"}), ("read", {"filePath": "/work/a.txt"})]),
    ("[read(filePath=/work/a b.txt, limit=3)]", [("read", {"filePath": "/work/a b.txt", "limit": 3})]),
    ("[shell(command=echo $(date), a, timeout=2.5)]", [("shell", {"command": "echo $(date), a", "timeout": 2.5})]),
    ("read(filePath=x)", [("read", {"filePath": "x"})]),
    ("[shell(command=it's, background=true)]", [("shell", {"command": "it's", "background": True})]),
    ("[read(filePath='a.py', limit=x)]", [("read", {"filePath": "a.py", "limit": "x"})]),
    ('[shell(command="say "hi" now")]', [("shell", {"command": 'say "hi" now'})]),
], ids=["two", "spaces", "brackets", "no-list", "apostrophe", "mixed", "inner-quotes"])
def test_lfm_unquoted_values_read_as_calls(block, calls):
    """What Python does not read, for the request's tools: each value up to the next argument or
    the call's end, by its parameter's type."""
    ty = TL.types(TOOLS)
    assert [(c["name"], c["arguments"]) for c in TL.LFM2.calls(block, ty)] == calls


@pytest.mark.parametrize("block", ["[nope(a=b c)]", "[shell(command=ls", "[shell(command=ls -la), plain words]",
                                   "[shell(ls -la)]"])
def test_lfm_unquoted_values_only_for_the_tools(block):
    assert TL.LFM2.calls(block, TL.types(TOOLS)) == []


def test_lfm_values_take_their_parameters_types():
    """Keyword and positional values by the schema: a string kept whatever it spells, an integer
    parameter's string its number, a string parameter's number its text; positional past the
    schema's parameters is no call."""
    ty = TL.types([{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "s": {"type": "string"}, "n": {"type": "integer"}, "o": {"type": "object"}}}}}])
    assert TL.LFM2.calls("[f(s='12', n='12', o='{\"a\": 1}')]", ty) == [{"name": "f", "arguments": {
        "s": "12", "n": 12, "o": {"a": 1}}}]
    assert TL.LFM2.calls("[f(s=12)]", ty) == [{"name": "f", "arguments": {"s": "12"}}]
    assert TL.LFM2.calls("[f('x', 3)]", ty) == [{"name": "f", "arguments": {"s": "x", "n": 3}}]
    assert TL.LFM2.calls("[f('x', 3, {}, 4)]", ty) == []


@pytest.mark.parametrize("step", [1, 2, 5, 64])
def test_lfm_stream(server, step):
    """The stream's chunks at any split: text first, each call whole with its index, the tags
    never in a content delta, and the same calls as the whole reply's."""
    reply = "Reading.\n\n" + LFM_TWO
    srv, port, w = server(reply, tok=_LfmTok(), step=step)
    chunks = _stream(port, {"messages": _ask(), "tools": TOOLS})
    assert chunks[-1] == "DONE"
    ds = [c["choices"][0]["delta"] for c in chunks[:-1] if c.get("choices")]
    text = "".join(d.get("content") or "" for d in ds)
    tcs = [t for d in ds for t in d.get("tool_calls") or []]
    assert text.strip() == "Reading." and "tool_call" not in text
    assert [t["index"] for t in tcs] == [0, 1]
    assert [t["function"]["name"] for t in tcs] == ["read", "shell"]
    assert json.loads(tcs[1]["function"]["arguments"]) == {"command": "echo 'hi'\ncat x"}
    assert [c for c in chunks[:-1] if c.get("choices")][-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_lfm_a_cut_off_block_is_content():
    t = "<|tool_call_start|>[read(path='hel"
    assert TL.parse(t, TOOLS, TL.LFM2) == (t, [])
    ts = TL.ToolStream(TOOLS, TL.LFM2)
    assert ts.feed(t[:10], False) == ("", []) and ts.feed(t[10:], True) == (t, [])


def test_lfm_tool_messages_are_accepted(server):
    """An assistant message with tool_calls and content null, then a tool message (a string or text
    parts): the call as the template writes it, the result in its tags."""
    srv, port, w = server("It prints hello.", tok=_LfmTok())
    calls = TL.parse(LFM_READ, TOOLS, TL.LFM2)[1]
    msgs = _turn2(calls, None)
    st, r = _call(port, {"messages": msgs, "tools": TOOLS})
    assert st == 200 and r["choices"][0]["message"]["content"] == "It prints hello."
    t = _text(srv, {"messages": msgs, "tools": TOOLS})
    assert ("<|im_start|>assistant\n" + LFM_READ + "<|im_end|>\n<|im_start|>tool\n<|tool_response_start|>"
            "1: print('hello')<|tool_response_end|><|im_end|>\n<|im_start|>assistant\n") in t


@pytest.mark.parametrize("reply", [LFM_READ, "I'll read it first.\n\n" + LFM_READ, LFM_TWO],
                         ids=["read", "text-read", "two"])
def test_lfm_the_next_request_continues_the_context(server, reply):
    """LFM's next request re-renders a call as the model wrote it (single quotes, its escapes): the
    prompt holds the last prompt + the reply + <|im_end|>."""
    srv, port, w = server(reply, tok=_LfmTok())
    p1 = _text(srv, {"messages": _ask(), "tools": TOOLS})
    content, calls = TL.parse(reply, TOOLS, TL.LFM2)
    p2 = _text(srv, {"messages": _turn2(calls, content, ("ok",) * len(calls)), "tools": TOOLS})
    assert p2.startswith(p1 + reply + "<|im_end|>\n")
