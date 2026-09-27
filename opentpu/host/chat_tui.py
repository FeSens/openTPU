"""otpu-chat's full-screen interface (Textual): the conversation on the left, the live numbers
on the right (model and device, the last turn's TTFT / prefill / decode rates, the KV context,
DRAM, session totals, sampling).

Generation runs in a worker thread. It hands each token to the interface without waiting for
it (the tokens that arrive while the interface draws are merged into one update), so neither
side blocks the other; Esc stops a reply. /reset, /stats, /think on|off, /help.
"""
from __future__ import annotations

import asyncio
import threading

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Input, Markdown, Static

from .chat import Chat, Turn

BAR = 26


def _rate(x, unit="tok/s") -> str:
    return "-" if x is None else f"{x:.2f} {unit}"


def _mib(n: int) -> str:
    m = n / 2**20
    return f"{m:,.0f} MiB" if m >= 10 else f"{m:.2f} MiB"


def stats_markup(meta: dict, chat: Chat, turn: Turn | None, busy: str = "") -> str:
    """The side panel. meta: model, backend, device, bitstream (text lines), sampling (dict),
    dram (a callable returning runstate's DRAM layout, or None)."""
    eng = chat.eng
    ctx, cap = eng.pos, eng.cap
    fill = round(BAR * ctx / cap) if cap else 0
    dev = bool(chat.clock_mhz)
    L = [f"[b]{meta['model']}[/b]", f"backend  {meta['backend']}", f"device   {meta['device']}"]
    L += [f"         {x}" for x in meta.get("bitstream", [])]
    L += ["", "[b]last turn[/b]" + (f"  [i]{busy}[/i]" if busy else "")]
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
    L += ["", "[b]KV context[/b]", f"[green]{'█' * fill}[/green][dim]{'░' * (BAR - fill)}[/dim]",
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


HELP = ("/reset clears the conversation and the KV cache; /stats shows the session; "
        "/think on|off switches thinking mode; Esc stops a reply; Ctrl-C or Ctrl-D quits.")


class ChatApp(App):
    CSS = """
    #chat { width: 1fr; }
    #log { height: 1fr; padding: 0 1; }
    #stats { width: 40; padding: 0 1; border-left: solid $primary; }
    .user { margin: 1 0 0 0; color: $text; text-style: bold; }
    .assistant { margin: 0; }
    .note { color: $text-muted; margin: 1 0 0 0; }
    Markdown { margin: 0; padding: 0; }
    Input { dock: bottom; }
    """
    BINDINGS = [Binding("escape", "stop", "stop reply"),
                Binding("ctrl+c", "quit", "quit", priority=True),
                Binding("ctrl+d", "quit", "quit", show=False, priority=True)]
    TITLE = "otpu-chat"

    def __init__(self, chat: Chat, meta: dict):
        super().__init__()
        self.chat, self.meta = chat, meta
        self._stop = False
        self._busy = ""
        self._reply = ""
        self._md: Markdown | None = None
        self._lock = threading.Lock()
        self._pending: tuple[list[str], Turn] | None = None   # tokens not yet shown

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="chat"):
                yield VerticalScroll(id="log")
                yield Input(placeholder="message (Enter sends; /help)", id="input")
            yield Static(id="stats")
        yield Footer()

    def on_mount(self) -> None:
        self._loop = asyncio.get_running_loop()
        self.sub_title = f"{self.meta['model']} on {self.meta['backend']}"
        self._note(HELP)
        self._refresh()
        self.query_one(Input).focus()

    # ---- helpers (UI thread)
    def _refresh(self) -> None:
        self.query_one("#stats", Static).update(
            stats_markup(self.meta, self.chat, self.chat.last, self._busy))

    def _add(self, w) -> None:
        log = self.query_one("#log", VerticalScroll)
        log.mount(w)
        log.scroll_end(animate=False)

    def _note(self, text: str) -> None:
        self._add(Static(text, classes="note", markup=False))

    # ---- input
    def on_input_submitted(self, ev: Input.Submitted) -> None:
        text = ev.value.strip()
        ev.input.value = ""
        if not text or self._busy:
            return
        if text.startswith("/"):
            self._command(text)
            return
        self._add(Static(f"> {text}", classes="user", markup=False))
        self._reply = ""
        self._md = Markdown("", classes="assistant")
        self._add(self._md)
        self._stop, self._busy = False, "prefill ..."
        self.chat.last = None
        self._refresh()
        self._generate(text)

    def _command(self, text: str) -> None:
        cmd, _, arg = text.partition(" ")
        if cmd == "/reset":
            self.chat.reset()
            self.query_one("#log", VerticalScroll).remove_children()
            self._note("conversation and KV cache cleared")
        elif cmd == "/stats":
            s = self.chat.session
            last = self.chat.last.line() if self.chat.last else "no turn yet"
            self._note(f"session: {s.turns} turns, {s.tokens_in} tokens in, {s.tokens_out} "
                       f"out, decode {_rate(s.decode_tok_s)} on average; context "
                       f"{self.chat.eng.pos}/{self.chat.eng.cap}\nlast turn: {last}")
        elif cmd == "/think" and arg in ("on", "off"):
            self.chat.think = arg == "on"
            self._note(f"thinking {arg} (the history is re-fed on the next turn)")
        else:
            self._note(HELP)
        self._refresh()

    def action_stop(self) -> None:
        if self._busy:
            self._stop = True
            self._busy = "stopping ..."
            self._refresh()

    # ---- generation (worker thread)
    @work(thread=True, exclusive=True)
    def _generate(self, text: str) -> None:
        try:
            _, turn = self.chat.ask(text, self._post, stop=lambda: self._stop)
        except Exception as e:                                      # noqa: BLE001
            self.call_from_thread(self._note, f"error: {type(e).__name__}: {e}")
            turn = None
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
        self._loop.call_soon_threadsafe(self.call_next, self._flush)

    def _flush(self) -> None:
        with self._lock:
            p, self._pending = self._pending, None
        if p is not None:
            self._update("".join(p[0]), p[1])

    def _update(self, delta: str, turn: Turn) -> None:
        self.chat.last = turn
        if not self._stop:
            self._busy = "decoding ..." if turn.gen_tokens else "first token ..."
        if delta:
            self._reply += delta
            self._md.update(self._reply)
            self.query_one("#log", VerticalScroll).scroll_end(animate=False)
        self._refresh()

    def _done(self, turn: Turn | None) -> None:
        self._flush()
        self._busy = ""
        if turn is not None and turn.stopped:
            self._note("(stopped)")
        self._refresh()
