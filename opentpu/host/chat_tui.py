"""otpu-chat's full-screen interface (Textual), in the style of Claude Code: one column of
conversation on the terminal's own background, a spinner line while a reply runs, a rounded
input box, and under it one status line with the numbers that matter (TTFT, prefill and decode
tok/s on the wall clock and the device, Mcycles/token, the KV context). /stats prints the rest
(DRAM, session totals, sampling) inline; Ctrl-S toggles it as a side panel.

Generation runs in a worker thread. It hands each token to the interface without waiting for
it (the tokens that arrive while the interface draws are merged into one update), so neither
side blocks the other; Esc stops a reply.
"""
from __future__ import annotations

import asyncio
import threading
import time

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Input, Markdown, OptionList, Static
from textual.widgets.option_list import Option

from .chat import NOTES, Chat, Turn

ACCENT = "#D97757"
GREY = "ansi_bright_black"             # in CSS
DIM = "bright_black"                   # the same grey in Rich text
AMBER = "#E0A030"
RED = "#E05050"
SPIN = "·✢✳✶✻✽✻✶✳✢"
COMMANDS = {"/help": "the commands and keys",
            "/continue": "continue a reply cut at max_new",
            "/reset": "forget the conversation and the KV cache",
            "/stats": "DRAM, session totals, sampling",
            "/think": "on|off  thinking mode (Qwen3, Qwen3.5)",
            "/quit": "leave"}
ARGS = {"/think"}                       # commands that take an argument
KEYS = ["enter send · esc interrupt · ^s panel · ^c quit", "esc · ^s panel · ^c quit",
        "^c quit", ""]                  # the longest that fits beside the numbers

# The terminal's own colours (ANSI default foreground and background, so a light or dark
# terminal stays light or dark), one accent, greys for the rest; plain bold headings.
THEME = Theme(
    name="otpu", ansi=True, dark=True, primary=ACCENT, secondary=GREY, accent=ACCENT,
    warning=AMBER, error=RED, success="ansi_green", foreground="ansi_default",
    background="ansi_default", surface="ansi_default", panel="ansi_default",
    boost="ansi_default",
    variables={
        "border": ACCENT, "border-blurred": GREY, "ansi-background": "ansi_black",
        "ansi-foreground": "ansi_white",
        "block-cursor-foreground": "ansi_black", "block-cursor-background": ACCENT,
        "input-cursor-background": "ansi_default", "input-cursor-foreground": "ansi_default",
        "input-cursor-text-style": "reverse",
        "input-selection-background": "ansi_bright_black",
        "input-selection-foreground": "ansi_default",
        "screen-selection-background": "ansi_bright_black",
        "screen-selection-foreground": "ansi_default",
        **{f"markdown-h{i}-{k}": v for i in range(1, 7) for k, v in (
            ("color", ACCENT if i == 1 else "ansi_default"), ("background", "transparent"),
            ("text-style", "bold" if i < 4 else "italic"))},
    })


def _rate(x, unit="tok/s") -> str:
    return "-" if x is None else f"{x:.2f} {unit}"


def _mib(n: int) -> str:
    m = n / 2**20
    return f"{m:,.0f} MiB" if m >= 10 else f"{m:.2f} MiB"


def _meter(ctx: int, cap: int, cells: int = 8) -> tuple[str, str]:
    """(bar, colour): dim up to 75 %, amber over 75 %, red over 90 %."""
    frac = ctx / cap if cap else 0.0
    fill = min(cells, round(cells * frac))
    colour = RED if frac > 0.9 else AMBER if frac > 0.75 else DIM
    return "▰" * fill + "▱" * (cells - fill), colour


def status_line(meta: dict, chat: Chat, turn: Turn | None, width: int = 0) -> Text:
    """The one line under the input: model and clock, TTFT, prefill and decode tok/s (wall,
    and device when there is a device clock), Mcycles/token, the KV context meter. In a
    narrow terminal (width) it drops, in turn, the model, the prefill device rate and the
    Mcycles/token, then the prefill; the context meter stays."""
    ctx, cap = chat.eng.pos, chat.eng.cap
    bar, colour = _meter(ctx, cap)
    tail = [(f"ctx {ctx}/{cap} ", DIM), (bar, colour),
            (f" {100 * ctx / cap:.0f}%" if cap else "", colour)]
    head = f"{meta['model']} · {meta.get('short', meta['backend'])}"

    def rate(label, wall, dev):
        s = f"{label} {'-' if wall is None else f'{wall:.1f}'} tok/s"
        return s + (f" (dev {dev:.1f})" if dev is not None else "")
    for level in range(5):
        segs = [] if level else [head]
        if turn is not None:
            t = turn
            segs.append(f"TTFT {'-' if t.ttft_s is None else f'{t.ttft_s:.2f}s'}")
            if level < 4:
                segs.append(rate("prefill", t.prefill_tok_s,
                                 t.prefill_dev_tok_s if level < 2 else None))
            dec = rate("decode", t.decode_tok_s, t.decode_dev_tok_s)
            if t.mcycles_per_token is not None and level < 3:
                dec += f" · {t.mcycles_per_token:.2f} Mcyc/tok"
            segs.append(dec)
        parts = []
        for x in segs:
            parts += [(x, DIM), (" │ ", DIM)]
        line = Text.assemble(*parts, *tail, no_wrap=True, overflow="ellipsis")
        if not width or line.cell_len <= width:
            break
    return line


def stats_markup(meta: dict, chat: Chat, turn: Turn | None) -> str:
    """The detail: model and device, the last turn, KV context, DRAM, session, sampling.
    meta: model, backend, device, bitstream (text lines), sampling (dict), dram (a callable
    returning runstate's DRAM layout, or None)."""
    eng = chat.eng
    ctx, cap = eng.pos, eng.cap
    dev = bool(chat.clock_mhz)
    L = [f"[b]{meta['model']}[/b]", f"backend  {meta['backend']}", f"device   {meta['device']}"]
    L += [f"         {x}" for x in meta.get("bitstream", [])]
    L += ["", "[b]last turn[/b]"]
    t = turn
    if t is None:
        L.append("  (none yet)")
    else:
        L.append(f"TTFT     {'-' if t.ttft_s is None else f'{t.ttft_s:.2f} s'}")
        L.append(f"prefill  {t.prefill_tokens} tok, {_rate(t.prefill_tok_s)}")
        if dev:
            L.append(f"         device {_rate(t.prefill_dev_tok_s)}")
        L.append(f"decode   {t.gen_tokens} tok, {_rate(t.decode_tok_s)}")
        if dev:
            L.append(f"         device {_rate(t.decode_dev_tok_s)}")
            mc = t.mcycles_per_token
            L.append(f"         {'-' if mc is None else f'{mc:.2f}'} Mcycles/token")
        if t.restarted:
            L.append("         (history re-fed)")
    bar, colour = _meter(ctx, cap, 24)
    L += ["", "[b]KV context[/b]", f"[{colour}]{bar}[/]",
          f"{ctx} / {cap} tokens ({100 * ctx / cap:.0f}%)" if cap else ""]
    dr = meta.get("dram") and meta["dram"]()
    if dr:
        L += ["", "[b]DRAM[/b]", f"image    {_mib(dr['image'])} / {_mib(dr['total'])}"]
        if dr.get("kv_capacity"):
            L.append(f"KV       {_mib(dr['kv_used'])} / {_mib(dr['kv_capacity'])}")
    s = chat.session
    L += ["", "[b]session[/b]", f"turns    {s.turns}", f"tokens   {s.tokens_in} in, "
          f"{s.tokens_out} out", f"decode   {_rate(s.decode_tok_s)} avg"]
    sp = meta.get("sampling") or {}
    short = {"temperature": "temp", "repetition_penalty": "rep pen"}
    L += ["", "[b]sampling[/b]"] + [f"{short.get(k, k):<8} {v}" for k, v in sp.items()]
    L.append(f"think    {'on' if chat.think else 'off'}")
    return "\n".join(L)


def welcome(meta: dict) -> Text:
    bits = [meta["model"], meta["backend"], meta["device"]] + list(meta.get("bitstream", []))
    return Text.assemble(("✻ ", ACCENT), ("openTPU chat", "bold"), "\n\n",
                         " · ".join(bits), "\n", ("/help for commands · esc to interrupt", DIM))


class Reply(Horizontal):
    """An assistant turn: the accent bullet, the Markdown indented beside it."""

    def __init__(self):
        super().__init__(classes="reply")
        self.md = Markdown("")

    def compose(self) -> ComposeResult:
        yield Static("⏺", classes="bullet")
        yield self.md


class ChatApp(App):
    CSS = f"""
    Screen {{ background: ansi_default; }}
    #main {{ width: 1fr; }}
    #log {{ height: 1fr; padding: 1 4 0 4; scrollbar-size-vertical: 1;
            scrollbar-color: {GREY}; scrollbar-background: ansi_default; }}
    #welcome {{ width: auto; max-width: 100%; border: round {ACCENT}; padding: 0 2;
                margin: 0 0 1 0; }}
    .user {{ margin: 0 0 1 0; }}
    .reply {{ height: auto; margin: 0 0 1 0; }}
    .bullet {{ width: 2; color: {ACCENT}; }}
    .reply Markdown {{ width: 1fr; margin: 0; padding: 0; background: ansi_default; }}
    MarkdownH1 {{ content-align: left top; }}
    MarkdownHeader {{ margin: 1 0 1 0; }}
    .reply Markdown > *:first-child {{ margin-top: 0; }}
    .reply Markdown > *:last-child {{ margin-bottom: 0; }}
    MarkdownFence {{ border-left: outer {GREY}; padding: 0 1; margin: 0 0 1 0; }}
    MarkdownFence > Label {{ padding: 0; }}
    .note {{ color: {GREY}; margin: 0 0 1 2; }}
    .block {{ color: {GREY}; border-left: outer {GREY}; padding: 0 1; margin: 0 0 1 2; }}
    #spinner {{ height: 1; padding: 0 4; display: none; }}
    #cmds {{ margin: 0 2; max-height: 9; border: round {GREY}; background: ansi_default;
             display: none; }}
    #cmds > .option-list--option-highlighted {{ color: {ACCENT}; background: ansi_default;
                                               text-style: bold; }}
    #input {{ margin: 0 2; border: round {GREY}; background: ansi_default; padding: 0 1; }}
    #input:focus {{ border: round {ACCENT}; }}
    #bar {{ height: 1; padding: 0 4; }}
    #status {{ width: 1fr; }}
    #keys {{ width: auto; color: {GREY}; padding: 0 0 0 2; }}
    #panel {{ width: 44; padding: 1 2; border-left: solid {GREY}; display: none; }}
    """
    BINDINGS = [Binding("escape", "escape", show=False),
                Binding("ctrl+c", "quit", show=False, priority=True),
                Binding("ctrl+d", "quit", show=False, priority=True),
                Binding("ctrl+s", "panel", show=False, priority=True),
                Binding("up", "cmd_move(-1)", show=False, priority=True),
                Binding("down", "cmd_move(1)", show=False, priority=True),
                Binding("tab", "cmd_complete", show=False, priority=True)]
    ENABLE_COMMAND_PALETTE = False
    TITLE = "otpu-chat"

    def __init__(self, chat: Chat, meta: dict):
        super().__init__()
        self.chat, self.meta = chat, meta
        self._stop = False
        self._busy = ""                  # "", "prefill", "decode", "stopping"
        self._t_submit = 0.0
        self._turn: Turn | None = None   # the turn running (or the last one)
        self._reply: Reply | None = None
        self._text = ""
        self._marker: Static | None = None
        self._lock = threading.Lock()
        self._pending: tuple[list[str], Turn] | None = None   # tokens not yet shown
        self._frame = 0

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="main"):
                with VerticalScroll(id="log"):
                    yield Static(welcome(self.meta), id="welcome")
                yield Static(id="spinner")
                yield OptionList(id="cmds")
                yield Input(placeholder="Ask anything  (/ for commands)", id="input")
                with Horizontal(id="bar"):
                    yield Static(id="status")
                    yield Static(id="keys")
            yield Static(id="panel")

    def on_mount(self) -> None:
        self._ui_loop = asyncio.get_running_loop()
        self.register_theme(THEME)
        self.theme = "otpu"
        self._refresh()
        self.call_after_refresh(self._refresh)          # the key hints need the laid-out width
        self.set_interval(0.1, self._tick)
        self.query_one(Input).focus()

    def on_resize(self) -> None:
        self.call_after_refresh(self._refresh)

    # ---- helpers (UI thread)
    def _refresh(self) -> None:
        turn = self._turn or self.chat.last
        width = self.query_one("#bar").size.width              # the content area
        line = status_line(self.meta, self.chat, turn, width)
        self.query_one("#status", Static).update(line)
        room = width - line.cell_len - 2
        self.query_one("#keys", Static).update(next(k for k in KEYS if len(k) <= max(room, 0)))
        panel = self.query_one("#panel", Static)
        if panel.display:
            panel.update(stats_markup(self.meta, self.chat, turn))

    def _add(self, w) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(w)
        log.scroll_end(animate=False)

    def _note(self, text: str, cls: str = "note") -> Static:
        w = Static(text, classes=cls, markup=cls == "block")
        self._add(w)
        return w

    def _tick(self) -> None:
        sp = self.query_one("#spinner", Static)
        sp.display = bool(self._busy)
        if not self._busy:
            return
        self._frame += 1
        t = self._turn
        if self._busy == "stopping":
            what = "Stopping…"
        elif t is not None and t.gen_tokens:
            what = f"Decoding… {t.gen_tokens} tok · {_rate(t.decode_tok_s)}"
        elif t is not None and t.prefill_total:
            what = (f"Prefilling… {t.prefill_tokens}/{t.prefill_total} tok · "
                    f"{time.perf_counter() - self._t_submit:.1f}s")
        else:
            what = f"Working… {time.perf_counter() - self._t_submit:.1f}s"
        sp.update(Text.assemble((SPIN[self._frame % len(SPIN)] + " ", ACCENT), (what, ACCENT),
                                (" · esc to interrupt", DIM)))

    # ---- input and the command popup
    def _matches(self, value: str) -> list[str]:
        if not value.startswith("/") or " " in value:
            return []
        return [c for c in COMMANDS if c.startswith(value)]

    def on_input_changed(self, ev: Input.Changed) -> None:
        cmds = self.query_one("#cmds", OptionList)
        m = self._matches(ev.value)
        cmds.display = bool(m)
        if m:
            cmds.clear_options()
            cmds.add_options([Option(Text.assemble((f"{c:<11}", "bold"), (COMMANDS[c], DIM)),
                                     id=c) for c in m])
            cmds.highlighted = 0

    def _picked(self) -> str | None:
        cmds = self.query_one("#cmds", OptionList)
        if not cmds.display or cmds.highlighted is None:
            return None
        return cmds.get_option_at_index(cmds.highlighted).id

    def check_action(self, action: str, parameters) -> bool | None:
        if action in ("cmd_move", "cmd_complete"):
            return self.query_one("#cmds", OptionList).display
        return True

    def action_cmd_move(self, d: int) -> None:
        cmds = self.query_one("#cmds", OptionList)
        cmds.highlighted = ((cmds.highlighted or 0) + d) % cmds.option_count

    def action_cmd_complete(self) -> None:
        c = self._picked()
        if c:
            inp = self.query_one(Input)
            inp.value = c + (" " if c in ARGS else "")
            inp.cursor_position = len(inp.value)

    def on_input_submitted(self, ev: Input.Submitted) -> None:
        text = ev.value.strip()
        picked = self._picked()
        if picked and text not in COMMANDS:          # Enter takes the highlighted command
            if picked in ARGS:
                self.action_cmd_complete()
                return
            text = picked
        ev.input.value = ""
        self.query_one("#cmds", OptionList).display = False
        if not text or self._busy:
            return
        if text.startswith("/"):
            self._command(text)
            return
        self._add(Static(Text.assemble(("> ", DIM), text), classes="user"))
        self._start(text)

    def _start(self, text: str | None) -> None:
        """Run ask(text), or resume() for None, in the worker."""
        self._stop, self._busy = False, "prefill"
        self._t_submit = time.perf_counter()
        self._turn = None
        if text is not None:
            self._reply, self._text = None, ""
        self._tick()
        self._generate(text)

    def _command(self, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        if cmd == "/reset":
            self.chat.reset()
            self._turn = self.chat.last = None
            self._reply = self._marker = None
            for w in list(self.query_one("#log", VerticalScroll).children)[1:]:
                w.remove()
            self._note("⎿ conversation and KV cache cleared")
        elif cmd == "/stats":
            self._note(stats_markup(self.meta, self.chat, self._turn or self.chat.last),
                       "block")
        elif cmd == "/think" and arg in ("on", "off"):
            self.chat.think = arg == "on"
            self._note(f"⎿ thinking {arg} (the history is re-fed on the next turn)")
        elif cmd == "/continue":
            if self.chat.can_resume and self._reply is not None:
                if self._marker is not None:
                    self._marker.remove()
                    self._marker = None
                self._start(None)
            else:
                self._note("⎿ nothing to continue")
        elif cmd == "/quit":
            self.exit()
        else:
            self._note("\n".join(f"{c:<11}{d}" for c, d in COMMANDS.items()) + f"\n\n{KEYS[0]}")
        self._refresh()

    def action_escape(self) -> None:
        cmds = self.query_one("#cmds", OptionList)
        if cmds.display:
            cmds.display = False
        elif self._busy:
            self._stop, self._busy = True, "stopping"
            self._tick()

    def action_panel(self) -> None:
        panel = self.query_one("#panel", Static)
        panel.display = not panel.display
        self._refresh()

    # ---- generation (worker thread)
    @work(thread=True, exclusive=True)
    def _generate(self, text: str | None) -> None:
        turn = None
        try:
            if text is None:
                _, turn = self.chat.resume(self._post, stop=lambda: self._stop)
            else:
                _, turn = self.chat.ask(text, self._post, stop=lambda: self._stop)
        except Exception as e:                                      # noqa: BLE001
            self.call_from_thread(self._note, f"⎿ error: {type(e).__name__}: {e}")
        self.call_from_thread(self._done, turn)

    def _post(self, delta: str, turn: Turn) -> None:
        """From the worker: queue the token and schedule one flush if none is pending.
        (call_from_thread would wait for the interface to draw, on every token.)"""
        with self._lock:
            if self._pending is not None:
                self._pending[0].append(delta)
                self._pending = (self._pending[0], turn)
                return
            self._pending = ([delta], turn)
        self._ui_loop.call_soon_threadsafe(self.call_next, self._drain)

    def _drain(self) -> None:
        with self._lock:
            p, self._pending = self._pending, None
        if p is not None:
            self._update("".join(p[0]), p[1])

    def _update(self, delta: str, turn: Turn) -> None:
        self._turn = turn
        if turn.gen_tokens and self._busy == "prefill":
            self._busy = "decode"
        if delta:
            if self._reply is None:
                self._reply = Reply()
                self._add(self._reply)
            self._text += delta
            self._reply.md.update(self._text)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        self._refresh()

    def _done(self, turn: Turn | None) -> None:
        self._drain()
        self._busy = ""
        self._tick()
        if turn is not None:
            if turn.end == "cap" and not turn.prefill_tokens and not turn.gen_tokens:
                self._note(f"⎿ context full: this message does not fit in the {turn.cap}-token "
                           "KV cache · /reset to start a new conversation")
            elif turn.end in NOTES:
                self._marker = self._note("⎿ " + NOTES[turn.end].strip("()").format(
                    max_new=self.chat.max_new, cap=turn.cap, cmd="/continue"))
            if turn.prefill_tokens or turn.gen_tokens:
                self._turn = turn
        self._refresh()
