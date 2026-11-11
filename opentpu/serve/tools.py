"""OpenAI tool calling over the formats the chat templates write: qwen3_coder, as Qwen3.5's and
Qwen3.6's do (docs/serve.md "Tool calls"; vLLM's qwen3_coder parser reads the same):

    <tool_call>
    <function=NAME>
    <parameter=P>
    VALUE
    </parameter>
    ...
    </function>
    </tool_call>

and LFM2's Python-like calls (dialect: LFM2, for a template that writes them), or the JSON form
the model takes when its system prompt asks for it:

    <|tool_call_start|>[NAME(P='VALUE', Q=2), OTHER()]<|tool_call_end|>

The request's side: request_tools(body) the tools to render (the request's OpenAI {type,
function} dicts as given: the template's `tool | tojson`), None without tools or with tool_choice
"none" (then nothing is parsed either); "auto", "required" and a named function are all auto
(the model chooses). message(m) an assistant message's tool_calls with each function.arguments
JSON string a dict (the template's `|items`) and its reasoning_content, a tool message's
tool_call_id kept.

The reply's side: ToolStream(tools).feed(delta, final) -> (content delta, calls closed by it):
the text before a call is content (without the whitespace before the call), a possible
"<tool_call>" prefix held back, a call buffered until its </tool_call>; a value by its tool's
JSON-schema type (string as is; integer, number, boolean, object, array json.loads, else the
string). parse(text, tools) the whole reply's (content or None, calls). openai(call, i) a
call's OpenAI form, its arguments a JSON string. Both take the template's dialect (default
qwen3_coder), its opening tag dialect.start: <tool_call> above.

The next request re-renders a call from its arguments (the template: each parameter in the
arguments' order, "<parameter=P>\\n" + VALUE + "\\n</parameter>\\n", a string as is, the rest
tojson): the text the model wrote, so the prompt continues the context the device or the store
holds (a value's one newline either side is the format's; a non-string value the model wrote
other than json.dumps would is the exception: the prompt parts from the context there). LFM2's
template renders them as it writes them: a string in single quotes (backslash, the quote,
newline and return escaped), an object or a list tojson, else its Python text; a call the model
wrote otherwise (double quotes) parts from the context there too.
"""
from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Callable, NamedTuple

START, END = "<tool_call>", "</tool_call>"
FN, FN_END = "<function=", "</function>"
PAR, PAR_END = "<parameter=", "</parameter>"
STRING = ("string", "str", "text", "varchar", "char", "enum")


class ToolError(ValueError):
    """A request's tools or tool messages malformed (api: a 400)."""


def request_tools(body: dict) -> list | None:
    """The request's tools to render, or None (none given, or tool_choice "none")."""
    tools = body.get("tools")
    if tools is None or body.get("tool_choice") == "none":
        return None
    if not isinstance(tools, list):
        raise ToolError("tools: a list of {type: function, function: {name, ...}}")
    for t in tools:
        f = t.get("function") if isinstance(t, dict) else None
        if not isinstance(f, dict) or t.get("type", "function") != "function" or \
                not isinstance(f.get("name"), str) or not f["name"]:
            raise ToolError("tools: each {type: function, function: {name, ...}}")
    return tools or None


def message(m: dict, content) -> dict:
    """A chat message for the template: role and content, an assistant's reasoning_content (a
    string: the template's think block) and tool_calls (their arguments dicts), a tool
    message's tool_call_id and name."""
    out = {"role": m["role"], "content": content}
    if m["role"] == "assistant" and isinstance(m.get("reasoning_content"), str):
        out["reasoning_content"] = m["reasoning_content"]
    if m["role"] == "assistant" and m.get("tool_calls"):
        calls = m["tool_calls"]
        if not isinstance(calls, list):
            raise ToolError("an assistant's tool_calls: a list")
        out["tool_calls"] = [_call_in(c) for c in calls]
    elif m["role"] == "tool":
        for k in ("tool_call_id", "name"):
            if m.get(k) is not None:
                out[k] = m[k]
    return out


def _call_in(c) -> dict:
    f = c.get("function") if isinstance(c, dict) else None
    if not isinstance(f, dict) or not isinstance(f.get("name"), str):
        raise ToolError("a tool call: {id, type: function, function: {name, arguments}}")
    a = f.get("arguments")
    if a is None or (isinstance(a, str) and not a.strip()):
        a = {}
    elif isinstance(a, str):
        try:
            a = json.loads(a)
        except ValueError:
            raise ToolError(f"tool call {f['name']!r}: its arguments are not JSON") from None
    if not isinstance(a, dict):
        raise ToolError(f"tool call {f['name']!r}: its arguments are not a JSON object")
    return {"id": c.get("id"), "type": "function", "function": {"name": f["name"],
                                                                "arguments": a}}


def types(tools) -> dict:
    """{function name: {parameter: its JSON-schema type, lower case, or None}}."""
    out = {}
    for t in tools or ():
        f = t["function"]
        props = (f.get("parameters") or {}).get("properties") or {}
        out[f["name"]] = {p: _type(s) for p, s in props.items()}
    return out


def _type(s) -> str | None:
    t = s.get("type") if isinstance(s, dict) else None
    if isinstance(t, list):                     # (["string", "null"])
        t = next((x for x in t if x != "null"), None)
    return t.lower() if isinstance(t, str) else None


def value(v: str, t: str | None):
    """A parameter's text as its type's value: a string (or no type known) as is; any other
    type json.loads, else the text."""
    if t is None or t in STRING:
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


def parse_call(block: str, ty: dict) -> dict | None:
    """A <tool_call> block's {name, arguments} (the parameters in order), or None without a
    <function=NAME>. Closers may be missing (a reply cut short): a parameter ends at its
    </parameter>, the next <parameter=, the </function> or the block's end."""
    i = block.find(FN)
    j = block.find(">", i + len(FN)) if i >= 0 else -1
    if j < 0 or not block[i + len(FN):j].strip():
        return None
    name = block[i + len(FN):j].strip()
    body = block[j + 1:]
    e = body.find(FN_END)
    body = body if e < 0 else body[:e]
    pt, args, k = ty.get(name, {}), {}, body.find(PAR)
    while k >= 0:
        g = body.find(">", k + len(PAR))
        if g < 0:
            break
        p = body[k + len(PAR):g].strip()
        rest = body[g + 1:]
        ends = [x for x in (rest.find(PAR_END), rest.find(PAR)) if x >= 0]
        n = min(ends) if ends else len(rest)
        v = rest[:n]
        v = v[1:] if v.startswith("\n") else v
        v = v[:-1] if v.endswith("\n") else v
        args[p] = value(v, pt.get(p))
        nxt = rest.find(PAR, n)
        k = -1 if nxt < 0 else g + 1 + nxt
    return {"name": name, "arguments": args}


def _qwen3_calls(block: str, ty: dict) -> list:
    c = parse_call(block, ty)
    return [] if c is None else [c]


class _JSONNames(ast.NodeTransformer):
    """The JSON spellings of Python's constants, which LFM2 sometimes writes in its calls."""
    NAMES = {"true": True, "false": False, "null": None}

    def visit_Name(self, n):
        return ast.Constant(self.NAMES[n.id]) if n.id in self.NAMES else n


def _typed(v, t: str | None):
    """A value the model wrote as a literal, as its parameter's type: a string by value(), a
    string parameter's other value its JSON text, any other as written."""
    if isinstance(v, str):
        return value(v, t)
    return json.dumps(v, ensure_ascii=False) if t in STRING else v


def _raw_newlines(s: str) -> str:
    """s with a newline or return inside a quoted string escaped (a file's text written across
    lines in one, which the template's own rendering does not do)."""
    out, q, i = [], None, 0
    while i < len(s):
        c = s[i]
        if q is not None and c == "\\":
            out.append(s[i:i + 2])
            i += 2
            continue
        if q is None and c in "'\"":
            q = c
        elif c == q:
            q = None
        out.append({"\n": "\\n", "\r": "\\r"}.get(c, c) if q is not None else c)
        i += 1
    return "".join(out)


def _name(f) -> str | None:
    """A call's function name, dots kept (functions.read), or None."""
    if isinstance(f, ast.Name):
        return f.id
    base = _name(f.value) if isinstance(f, ast.Attribute) else None
    return None if base is None else f"{base}.{f.attr}"


def _json_calls(s: str, ty: dict) -> list:
    """LFM2's JSON form: {name, arguments} or a list of them; [] for anything else."""
    try:
        v = json.loads(s)
    except ValueError:
        return []
    out = []
    for c in v if isinstance(v, list) else [v]:
        if not isinstance(c, dict) or not isinstance(c.get("name"), str):
            return []
        a = c.get("arguments", c.get("parameters", {}))
        try:
            a = json.loads(a) if isinstance(a, str) else a
        except ValueError:
            return []
        if not isinstance(a, dict):
            return []
        pt = ty.get(c["name"], {})
        out.append({"name": c["name"], "arguments": {k: _typed(x, pt.get(k)) for k, x in a.items()}})
    return out


_CALL = re.compile(r"\s*([A-Za-z_][\w.]*)\(")
_KEY = re.compile(r"\s*([A-Za-z_]\w*)\s*=\s*")
_NEXT = re.compile(r"\s*,\s*[A-Za-z_][\w.]*\s*[=(]|\s*\Z")


def _bare(s: str, i: int) -> int:
    """Where the unquoted value at s[i:] ends: at depth 0, the comma before the next argument or
    call, or the bracket that closes the call (before the next call or the block's end; else
    any at depth 0); -1 when the block ends first."""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c in "([{":
            depth += 1
        elif depth and c in ")]}":
            depth -= 1
        elif not depth and c == ")" or not depth and c == "," and _NEXT.match(s, j):
            return j
    return -1


def _loose_value(s: str, i: int, t: str | None) -> tuple:
    """The value at s[i:] and where it ends: quoted, a literal as _typed takes it, else bare (the
    text up to _bare's end, without quotes around it: their own inside) by value(); its end -1
    when the block ends first."""
    if s[i:i + 1] in ("'", '"'):
        j = i + 1
        while j < len(s) and s[j] != s[i]:
            j += 2 if s[j] == "\\" else 1
        m = _NEXT.match(s, j + 1) if s[j + 1:j + 2].strip() != ")" else True
        if j < len(s) and m:
            try:
                v = ast.literal_eval(_raw_newlines(s[i:j + 1]))
            except (SyntaxError, ValueError):
                v = None
            if isinstance(v, str):
                return _typed(v, t), j + 1
    e = _bare(s, i)
    if e < 0:
        return None, -1
    v = s[i:e].strip()
    return value(v[1:-1] if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"" else v, t), e


def _loose_calls(s: str, ty: dict) -> list:
    """The calls of a block Python's syntax does not read, as LFM2 writes them beside its quoted
    ones: "[bash(command=ls -la /tmp), read(filePath=/work/a.txt)]", each value up to the next
    argument or the call's end. Only the request's tools; anything else or a block cut short
    is no call: []."""
    if s[:1] == "[" and s[-1:] == "]":
        s = s[1:-1]
    out, i = [], 0
    while True:
        m = _CALL.match(s, i)
        if m is None or m.group(1) not in ty:
            return []
        name, pt, args, i = m.group(1), ty[m.group(1)], {}, m.end()
        while True:
            while s[i:i + 1].isspace():
                i += 1
            if s[i:i + 1] == ")":
                i += 1
                break
            m = _KEY.match(s, i)
            if m is None:
                return []
            args[m.group(1)], i = _loose_value(s, m.end(), pt.get(m.group(1)))
            if i < 0:
                return []
            while s[i:i + 1].isspace():
                i += 1
            if s[i:i + 1] == ",":
                i += 1
        out.append({"name": name, "arguments": args})
        while s[i:i + 1].isspace():
            i += 1
        if i >= len(s):
            return out
        if s[i] != ",":
            return []
        i += 1


def _lfm2_calls(block: str, ty: dict) -> list:
    """A <|tool_call_start|> block's calls, in order, or [] when it is not one (its text is
    content): "[f(a='x', b=1), g()]" (no list: one call), read as Python's syntax and literals
    alone, nothing evaluated; positional values by the schema's parameter order; else
    _loose_calls'. A reply cut short inside a call does not parse: its text is content."""
    s = block.strip()
    if s[:1] == "{" or s[:2] == "[{":
        return _json_calls(s, ty)
    for text in (s, _raw_newlines(s)):
        try:
            tree = ast.parse(text, mode="eval").body
            break
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
    if tree is None:
        return _loose_calls(s, ty)
    return _python_calls(tree, ty) or _loose_calls(s, ty)


def _python_calls(tree, ty: dict) -> list:
    """The calls of a parsed block; [] for anything but calls with literal values."""
    out = []
    try:
        for e in tree.elts if isinstance(tree, ast.List) else [tree]:
            name = _name(e.func) if isinstance(e, ast.Call) else None
            if name is None:
                return []
            pt, args = ty.get(name, {}), {}
            for p, a in zip(pt, e.args):
                args[p] = _typed(ast.literal_eval(_JSONNames().visit(a)), pt[p])
            if len(e.args) > len(pt) or any(k.arg is None for k in e.keywords):
                return []
            for k in e.keywords:
                args[k.arg] = _typed(ast.literal_eval(_JSONNames().visit(k.value)), pt.get(k.arg))
            out.append({"name": name, "arguments": args})
    except (ValueError, TypeError, SyntaxError, RecursionError):
        return []
    return out


class Dialect(NamedTuple):
    """A chat template's tool call format: its block's tags, calls(block, types) the block's
    calls, [{name, arguments}], [] when it is not one, and strip: the whitespace before a block
    not content, where the template writes its own there (LFM2's writes none: content keeps it,
    so the next request's rendering continues through it)."""
    start: str
    end: str
    calls: Callable
    strip: bool = True


QWEN3_CODER = Dialect(START, END, _qwen3_calls)
LFM2 = Dialect("<|tool_call_start|>", "<|tool_call_end|>", _lfm2_calls, False)


def dialect(template) -> Dialect:
    """The format of a chat template (its text, or None): LFM2's where it writes that block."""
    return LFM2 if isinstance(template, str) and LFM2.start in template else QWEN3_CODER


def _partial(s: str, tag: str) -> int:
    """How many of s's last characters start `tag`."""
    for j in range(min(len(tag) - 1, len(s)), 0, -1):
        if s.endswith(tag[:j]):
            return j
    return 0


class ToolStream:
    """A reply's text as content and tool calls, as it streams (see the module's notes)."""

    def __init__(self, tools, dialect: Dialect = QWEN3_CODER):
        self.ty = types(tools)
        self.d = dialect
        self.buf = ""
        self.inside = False
        self.lead = False               # right after a call: the whitespace before text dropped
        self.ws = ""                    # the whitespace before the block being read
        self.calls: list = []           # every call so far, {name, arguments}

    def feed(self, delta: str, final: bool = False) -> tuple[str, list]:
        self.buf += delta
        out, new = [], []
        while True:
            if not self.inside:
                i = self.buf.find(self.d.start)
                if i >= 0:
                    s = self._lead(self.buf[:i])
                    t = s.rstrip() if self.d.strip else s
                    out.append(t)
                    self.ws = s[len(t):]
                    self.buf, self.inside = self.buf[i + len(self.d.start):], True
                    continue
                if final:
                    s = self._lead(self.buf)
                    out.append(s.rstrip() if self.calls else s)
                    self.buf = ""
                    break
                head = self.buf[:len(self.buf) - _partial(self.buf, self.d.start)]
                s = self._lead(head)
                t = s.rstrip()                          # (whitespace waits: a call may follow)
                out.append(t)
                self.buf = self.buf[len(head) - len(s) + len(t):]
                break
            i = self.buf.find(self.d.end)
            if i < 0 and not final:
                break
            block = self.buf if i < 0 else self.buf[:i]
            self.buf = "" if i < 0 else self.buf[i + len(self.d.end):]
            self.inside = False
            cs = self.d.calls(block, self.ty)
            if not cs:                                  # not a call: its text is content
                out.append(self.ws + self.d.start + block + ("" if i < 0 else self.d.end))
            else:
                self.calls.extend(cs)
                new.extend(cs)
                self.lead = True
            if i < 0:
                break
        return "".join(out), new

    def _lead(self, s: str) -> str:
        """s without the whitespace after a call (until text follows it)."""
        if not self.lead:
            return s
        s = s.lstrip()
        self.lead = not s
        return s


def parse(text: str, tools, dialect: Dialect = QWEN3_CODER) -> tuple[str | None, list]:
    """A whole reply's content (None when empty beside calls) and calls."""
    ts = ToolStream(tools, dialect)
    content, calls = ts.feed(text, final=True)
    return (content or None) if calls else content, calls


def openai(call: dict, index: int | None = None) -> dict:
    """A call's OpenAI form: {id, type, function: {name, arguments (a JSON string)}}, with its
    index in a stream's delta."""
    d = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
         "function": {"name": call["name"],
                      "arguments": json.dumps(call["arguments"], ensure_ascii=False)}}
    return d if index is None else {"index": index, **d}
