"""otpu-chat: chat with Qwen3 (or LFM2, or Qwen3.5) running on openTPU.

    otpu-chat                                 # ISA simulator (~3 s/token on a laptop)
    otpu-chat --plain                         # a plain REPL instead of the full-screen one
    otpu-chat --model lfm2                    # LFM2.5-230M instead of Qwen3-0.6B
    otpu-chat --model qwen35                  # Qwen3.5-0.8B (text only)
    otpu-chat --backend board                 # the FPGA over PCIe (opentpu/host/board.py)
    otpu-chat --backend board-sim             # the Verilator board model (very slow)
    otpu-chat --prompt "Why is the sky blue?" # one-shot
    otpu-chat --think                         # Qwen3 thinking mode
    otpu-chat --model qwen35-2b --mtp         # Qwen3.5 with its MTP drafter (docs/mtp.md)
    otpu-chat --model phi4-mini --wformat fp4 # the weights' format (default auto: the model's
                                              # recommended mix, else int8; docs/formats.md)

The interactive mode is a full-screen interface (opentpu/host/chat_tui.py): the conversation,
and under the input a status line with TTFT, prefill and decode tokens/s (wall and device) and
the KV context, updated while the reply streams; /stats adds DRAM, session totals and sampling.
--plain and --prompt print the same numbers as one line per reply.

The model runs on the device, the prompt several tokens per run (Engine.prefill_chunks; a
dense model's from programs at run-time positions, compiled once per bucket: docs/prefill.md)
and the reply token by token; the host only tokenizes, looks up the embedding rows, applies the
chat template and samples from the logits. The KV cache stays in device DRAM
across turns; only the new turn's tokens are fed. On the card the tool holds the device lock
and publishes its status (model, DRAM, tokens/s) for otpu-smi; the next token's program is
compiled while the card runs the current one (Engine pipelining). --mtp (Qwen3.5) decodes with
the model's MTP drafter, the loop on the device (opentpu/llm/mtp.py: a verify of two rows and a
draft per iteration): the same replies, greedy or sampled, in fewer runs of the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

import numpy as np

from opentpu.host.runstate import busy_exits
from opentpu.llm import MODELS, load_spec, model_dir
from opentpu.llm import formats as FM
from opentpu.llm import generate as G
from opentpu.llm.qwen3 import Engine, load_weights


# Sampling defaults per model family (Spec module); command-line flags override them. LFM2's
# are its generation_config.json; Qwen3.5's its model card's non-thinking settings (without the
# presence penalty).
SAMPLING = {"qwen3": dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0),
            "lfm2": dict(temperature=0.1, top_k=50, top_p=1.0, repetition_penalty=1.05),
            "qwen35": dict(temperature=0.7, top_k=20, top_p=0.8, repetition_penalty=1.0),
            "gemma4": dict(temperature=1.0, top_k=64, top_p=0.95, repetition_penalty=1.0)}


def sampler(temperature: float, top_k: int, top_p: float, seed: int | None,
            repetition_penalty: float = 1.0):
    """pick(logits, context) -> token id. The repetition penalty (as Hugging Face's) divides
    the positive logits and multiplies the negative ones of every token in `context`; it
    applies to greedy decoding (temperature 0) too.

    Top-k runs on the float32 logits: when the k largest are distinct and larger than the
    next one, the candidates and their order are unique, so this gives the picks of the
    float64 path (_top_k_f64) that it replaces in the common case; any tie falls back to it.

    pick.stream(context) takes the logits in pieces instead, as they arrive from the card
    (Engine.step(sink=...)): begin(n, dtype), feed(lo, values) for [lo, lo + len(values)) in
    any order, then result() -> the token id of pick(logits, context). The penalty and the
    block maxima are applied per piece, so after the last piece only the selection is left.
    `context` is read at begin()."""
    rng = np.random.default_rng(seed)
    seen = _Seen()
    block = 64

    class Stream:
        def __init__(self, context=()):
            self.context = context

        def begin(self, n: int, dtype=np.float32) -> None:
            self.buf = np.empty(n, dtype)
            self.bm = np.empty(-(-n // block), dtype)
            self.bm_ok = np.zeros(len(self.bm), bool)
            self.got = 0
            ctx = self.context
            self.ix = seen(ctx) if repetition_penalty != 1.0 and len(ctx) else None

        def feed(self, lo: int, v) -> None:
            hi = lo + len(v)
            b = self.buf[lo:hi]
            b[:] = v
            if self.ix is not None:
                sel = self.ix[np.searchsorted(self.ix, lo):np.searchsorted(self.ix, hi)]
                w = self.buf[sel]
                self.buf[sel] = np.where(w > 0, w / repetition_penalty, w * repetition_penalty)
            self.got += hi - lo
            if temperature > 0 and lo % block == 0 and (hi % block == 0 or hi == len(self.buf)):
                m = (hi - lo) // block * block      # the piece's block maxima
                if m:
                    self.bm[lo // block:(lo + m) // block] = b[:m].reshape(-1, block).max(axis=1)
                if m < hi - lo:
                    self.bm[-1] = b[m:].max()
                self.bm_ok[lo // block:-(-hi // block)] = True

        def result(self) -> int:
            assert self.got == len(self.buf), "logits missing"
            logits = self.buf
            if temperature <= 0:
                return int(np.argmax(logits))
            top = None
            if 0 < top_k < len(logits):
                if not self.bm_ok.all():            # pieces off the block grid
                    self._block_max()
                top = _top_k_f32(logits, top_k, temperature, block, self.bm)
            idx, z = top if top is not None else _top_k_f64(logits, top_k, temperature)
            p = np.exp(z - z[0])
            p /= p.sum()
            keep = min(len(p), np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            return int(idx[rng.choice(keep, p=p)])

        def _block_max(self) -> None:
            n, lg = len(self.buf), self.buf
            m = n // block * block
            self.bm[:m // block] = lg[:m].reshape(-1, block).max(axis=1)
            if m < n:
                self.bm[-1] = lg[m:].max()

    def pick(logits, context=()):
        s = Stream(context)
        s.begin(len(logits), logits.dtype)
        s.feed(0, logits)
        return s.result()

    def warm() -> None:
        """Run numpy's code paths of a pick once (the first pick of a process took 2-10 ms:
        np.union1d, the top-k selection and the generator's choice, first called) on a
        throwaway sampler with its own generator: this one's draws do not change."""
        w = sampler(temperature, top_k, top_p, 0, repetition_penalty)
        w(np.linspace(-1.0, 1.0, 4096, dtype=np.float32), [1, 2, 3])

    pick.stream = Stream
    pick.warm = warm
    pick.greedy = temperature <= 0 and repetition_penalty == 1.0   # argmax of the raw logits
    pick.params = dict(temperature=temperature, top_k=top_k, top_p=top_p,
                       repetition_penalty=repetition_penalty)
    pick.rng = rng
    return pick


def _top_k_f64(logits, top_k: int, temperature: float):
    """(indices, logits / temperature in float64), the top_k largest in descending order
    (all of them for top_k 0)."""
    z = logits.astype(np.float64) / temperature
    idx = np.argpartition(-z, top_k)[:top_k] if top_k else np.arange(len(z))
    z = z[idx]
    order = np.argsort(-z)
    return idx[order], z[order]


def _top_k_f32(logits, k: int, temperature: float, block: int = 64, bm=None):
    """_top_k_f64 without converting or selecting over the whole vocabulary, or None when a
    tie makes the choice among equal values depend on the selection algorithm (then
    _top_k_f64 decides, as before).

    The k-th largest of the per-block maxima (blocks of `block` logits; `bm` if the caller has
    them) is a lower bound t of the k-th largest logit (k blocks each hold a logit >= t), so
    the top k are among the logits >= t, which lie in the blocks whose maximum is >= t:
    usually a few times k of them; the selection runs on those. Dividing by the temperature
    in float64 keeps the order of distinct float32 values, so the unique top k and their
    order are the ones of _top_k_f64."""
    if bm is None:
        m = len(logits) // block * block
        bm = logits[:m].reshape(-1, block).max(axis=1)
        if m < len(logits):
            bm = np.append(bm, logits[m:].max())
    if len(bm) <= k or np.isnan(bm).any():
        return None
    t = np.partition(bm, len(bm) - k)[len(bm) - k]
    blocks = np.flatnonzero(bm >= t)
    span = (blocks[:, None] * block + np.arange(block)).reshape(-1)
    span = span[span < len(logits)]
    cand = span[logits[span] >= t]              # every logit outside is < t <= the top k
    lv = logits[cand]
    if len(cand) > k:
        part = np.argpartition(-lv, k)
        nxt = lv[part[k]]
        cand, lv = cand[part[:k]], lv[part[:k]]
    else:
        nxt = None
    order = np.argsort(-lv)
    lv = lv[order]
    if (nxt is not None and not lv[-1] > nxt) or np.any(lv[1:] == lv[:-1]):
        return None
    return cand[order], lv.astype(np.float64) / temperature


class _Seen:
    """The distinct token ids of a context list that only grows (Chat.fed), kept up to date
    with the tokens appended since the last call instead of converting the whole list."""

    def __init__(self):
        self.ctx, self.n, self.ids = None, 0, np.zeros(0, np.int64)

    def __call__(self, context) -> np.ndarray:
        if context is not self.ctx or len(context) < self.n:
            self.ctx, self.n, self.ids = context, 0, np.zeros(0, np.int64)
        if len(context) > self.n:
            new = np.asarray(context[self.n:], np.int64)
            self.ids = np.union1d(self.ids, new)
            self.n = len(context)
        return self.ids


def sampling(spec, args) -> dict:
    """The model family's SAMPLING defaults, overridden by the flags given on the command
    line (None when not given)."""
    d = dict(SAMPLING[type(spec).__module__.rsplit(".", 1)[-1]])
    d.update({k: getattr(args, k) for k in d if getattr(args, k, None) is not None})
    return d


class _Picked(int):
    """A first token MTP's prefill picked (Chat._prefill_mtp): shown, not fed yet."""


@dataclass
class Turn:
    """The numbers of one reply. Wall times are seconds from the submit; device numbers come
    from the engine's per-step cycles at `clock_mhz` (0: no device clock, wall only).
    Prefill: the prompt tokens fed this turn (the KV cache keeps the earlier turns). TTFT:
    submit -> first generated token. Decode: the tokens after the first, over the time since
    the first. `end`: why the reply ended ("eos", "max_new", "cap": the KV cache is full,
    "stopped": by the user; "" while it runs)."""
    clock_mhz: float = 0.0
    cap: int = 0
    prefill_total: int = 0            # the prompt tokens this turn has to feed
    prefill_tokens: int = 0           # ... fed so far
    prefill_s: float = 0.0
    prefill_cycles: int = 0
    ttft_s: float | None = None
    gen_tokens: int = 0
    decode_s: float = 0.0             # first -> latest generated token
    decode_steps: int = 0             # device steps after the prefill
    decode_cycles: int = 0
    context: int = 0                  # KV positions filled
    restarted: bool = False           # the template changed the history: KV rebuilt
    end: str = ""
    mtp_iters: int = 0                # MTP's verify iterations (--mtp)
    mtp_accepted: int = 0             # ... whose draft was accepted

    def _dev(self, n: int, cycles: int) -> float | None:
        return n * self.clock_mhz * 1e6 / cycles if self.clock_mhz and cycles else None

    @property
    def prefill_tok_s(self) -> float | None:
        return self.prefill_tokens / self.prefill_s if self.prefill_s else None

    @property
    def prefill_dev_tok_s(self) -> float | None:
        return self._dev(self.prefill_tokens, self.prefill_cycles)

    @property
    def decode_tok_s(self) -> float | None:
        return (self.gen_tokens - 1) / self.decode_s if self.gen_tokens > 1 and self.decode_s \
            else None

    @property
    def decode_dev_tok_s(self) -> float | None:
        return self._dev(self.decode_steps, self.decode_cycles)

    @property
    def mcycles_per_token(self) -> float | None:
        return self.decode_cycles / self.decode_steps / 1e6 \
            if self.clock_mhz and self.decode_cycles and self.decode_steps else None

    def line(self) -> str:
        """The plain-mode summary: TTFT, prefill tok/s, decode tok/s, context."""
        def r(x, dev):
            return "n/a" if x is None else f"{x:.2f}" + ("" if dev is None else
                                                         f" (device {dev:.1f})")
        ttft = "n/a" if self.ttft_s is None else f"{self.ttft_s:.2f}s"
        mc = "" if self.mcycles_per_token is None else \
            f", {self.mcycles_per_token:.2f} Mcycles/token at {self.clock_mhz:.0f} MHz"
        end = {"max_new": ", stopped at max_new", "cap": ", context full",
               "stopped": ", stopped"}.get(self.end, "")
        mtp = f", MTP acceptance {self.mtp_accepted / self.mtp_iters:.2f}" \
            if self.mtp_iters else ""
        return (f"[TTFT {ttft}; prefill {self.prefill_tokens} tokens, "
                f"{r(self.prefill_tok_s, self.prefill_dev_tok_s)} tok/s; decode "
                f"{self.gen_tokens} tokens, {r(self.decode_tok_s, self.decode_dev_tok_s)} tok/s"
                f"{mc}{mtp}; context {self.context}/{self.cap}{end}]")


class Detok:
    """Incremental detokenization: add(token) -> the text it adds to the reply. Each call
    decodes only the tokens since the last emitted text, from one token earlier (the prefix):
    new text = decode(prefix..) minus decode(prefix..read), which keeps the spaces and merges
    that depend on the token before. Text that ends in an incomplete UTF-8 sequence (U+FFFD)
    is held back until the rest of the character arrives."""

    def __init__(self, tok, ids=()):
        """ids: the reply so far (resume), already shown."""
        self.tok, self.ids = tok, list(ids)
        self.prefix, self.read = max(0, len(self.ids) - 1), len(self.ids)

    def _dec(self, ids) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)

    def add(self, t: int) -> str:
        self.ids.append(t)
        before = self._dec(self.ids[self.prefix:self.read])
        now = self._dec(self.ids[self.prefix:])
        if len(now) > len(before) and not now.endswith("\ufffd"):
            self.prefix, self.read = self.read, len(self.ids)
            return now[len(before):]
        return ""


@dataclass
class Session:
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    decode_s: float = 0.0
    decode_tokens: int = 0            # tokens counted in decode_s (the first of a turn is not)

    def add(self, t: Turn) -> None:
        self.turns += 1
        self.tokens_in += t.prefill_tokens
        self.tokens_out += t.gen_tokens
        if t.gen_tokens > 1:
            self.decode_s += t.decode_s
            self.decode_tokens += t.gen_tokens - 1

    @property
    def decode_tok_s(self) -> float | None:
        return self.decode_tokens / self.decode_s if self.decode_s else None


class Chat:
    """A conversation on an Engine: the KV cache keeps every fed token across turns, so a turn
    feeds only what the chat template added since (when the template rewrites the history, it
    starts over). The engine has no sliding window: a message that does not fit in the cache
    is refused (end "cap", nothing fed) and a reply that fills it stops; /reset starts over."""

    def __init__(self, engine: Engine, tok, think: bool, pick, max_new: int,
                 clock_mhz: float = 0.0):
        self.eng, self.tok, self.think, self.pick, self.max_new = engine, tok, think, pick, max_new
        self.clock_mhz = clock_mhz
        self.history: list[dict] = []
        self.fed: list[int] = []            # tokens whose K/V are in the device cache
        self.session = Session()
        self.last: Turn | None = None
        self._next = None                   # logits after a reply cut at max_new (resume())
        self._reply: list[int] = []         # the last reply's tokens
        self.mtp = None                     # an MTP engine's decoder (opentpu/llm/mtp.py)
        if getattr(engine.spec, "mtp", False):
            from opentpu.llm.mtp import MTPDecoder
            self.mtp = MTPDecoder(engine)
        warm = getattr(pick, "warm", None)
        if warm is not None:                # the first pick then costs what the others do
            warm()

    def _template(self, add_prompt=True) -> list[int]:
        if getattr(self.tok, "chat_template", "") is None:     # a base model: the text as is
            return list(self.tok("".join(m["content"] for m in self.history))["input_ids"])
        ids = self.tok.apply_chat_template(self.history, add_generation_prompt=add_prompt,
                                           enable_thinking=self.think, tokenize=True)
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    def reset(self) -> None:
        """Forget the conversation and the KV cache."""
        self.history, self.fed, self._next = [], [], None
        self.eng.reset()

    @property
    def can_resume(self) -> bool:
        return self._next is not None

    def ask(self, text: str, on_update=None, stop=lambda: False) -> tuple[str, Turn]:
        """One turn. on_update(delta_text, turn) after every prefill run (delta "") and for every
        generated token (Detok: text that ends in an incomplete character comes with a later
        token, or in one last call after the reply). A generated token's call comes once the
        card runs the next step (Engine.step's on_start), so the interface draws during the
        run, not while the host starts it; at the end of the reply it comes at once. stop()
        is polled between runs (the reply so far is kept)."""
        on_update = on_update or (lambda delta, turn: None)
        t0 = time.perf_counter()
        turn = Turn(clock_mhz=self.clock_mhz, cap=self.eng.cap, context=self.eng.pos)
        self.history.append({"role": "user", "content": text})
        ids = self._template()
        if len(ids) >= self.eng.cap:         # no room for the prompt and a reply token
            self.history.pop()
            turn.end = "cap"
            return "", turn
        self._next = None
        n = len(self.fed)
        if ids[:n] != self.fed:              # template rewrote history: start over
            self.eng.reset()
            self.fed, n, turn.restarted = [], 0, True
        turn.prefill_total = len(ids) - n
        logits = None
        if self.use_mtp:
            logits = self._prefill_mtp(ids, n, turn, t0, on_update, stop)
        else:
            for part, logits in self.eng.prefill_chunks(ids[n:]):   # up to Engine.rows a run
                self.fed += part
                turn.prefill_tokens += len(part)
                turn.prefill_s = time.perf_counter() - t0
                turn.prefill_cycles += (self.eng.stats[-1] or {}).get("cycles", 0)
                turn.context = self.eng.pos
                on_update("", turn)
                if stop() and len(self.fed) < len(ids):
                    turn.end = "stopped"
                    break
        reply = self._decode(logits, [], turn, t0, on_update, stop)
        self.history.append({"role": "assistant", "content": reply})
        self.session.add(turn)
        self.last = turn
        return reply, turn

    def resume(self, on_update=None, stop=lambda: False) -> tuple[str, Turn]:
        """Continue the last reply where max_new cut it (can_resume): the same reply grows
        by up to max_new tokens more."""
        assert self.can_resume, "no reply to continue"
        on_update = on_update or (lambda delta, turn: None)
        turn = Turn(clock_mhz=self.clock_mhz, cap=self.eng.cap, context=self.eng.pos)
        logits, self._next = self._next, None
        reply = self._decode(logits, self._reply, turn, time.perf_counter(), on_update, stop)
        turn.ttft_s = None                   # its first token was already computed: no TTFT
        self.history[-1]["content"] = reply
        self.session.add(turn)
        self.last = turn
        return reply, turn

    @property
    def on_card(self) -> bool:
        """The decode loop runs on the device (Engine.generate_card) on an engine that runs it,
        for greedy picks and for the sampler's settings the device takes (generate.Sampling:
        top_k 1 .. 64, repetition_penalty >= 1). The host then only shows the tokens."""
        if not getattr(self.eng, "can_generate", False):
            return False
        sp = getattr(self.pick, "params", None)
        return bool(getattr(self.pick, "greedy", False)) or (sp is not None
                                                             and G.Sampling.fits(**sp))

    @property
    def use_mtp(self) -> bool:
        """An MTP engine decodes with its drafter when the loop runs on the device."""
        return self.mtp is not None and self.on_card

    def _prefill_mtp(self, ids, n: int, turn: Turn, t0: float, on_update, stop):
        """ids[n:] through the MTP decoder's prefill (each run's rows' hidden, then the MTP
        layer over them): the first token is picked from the last row's logits inside it (the
        draft after it needs it), so this returns it, _Picked; None when stopped."""
        from opentpu.llm.mtp import MTPStats
        st, k = MTPStats(), [0]

        def on_run(part) -> bool:
            self.fed += part
            turn.prefill_tokens += len(part)
            turn.prefill_s = time.perf_counter() - t0
            turn.prefill_cycles += sum((r or {}).get("cycles", 0) for _, _, r in st.runs[k[0]:])
            k[0] = len(st.runs)
            turn.context = len(self.fed)
            on_update("", turn)
            if stop() and len(self.fed) < len(ids):
                turn.end = "stopped"
                return True
            return False
        a0, _ = self.mtp.prefill(ids[n:], st, pick=lambda lg: self.pick(lg, ids), on_run=on_run)
        return None if a0 is None else _Picked(a0)

    def _mtp_run(self, t: int, left: int, take, stop, samp, turn: Turn) -> tuple[list, int]:
        """One device run of the MTP loop from t (fed, shown already) with up to `left` tokens
        after it; (the tokens, the run's cycles)."""
        first = [True]

        def shown(x: int) -> None:
            if not first[0]:
                take(x)
            first[0] = False
        st = self.mtp.loop_card(t, self.mtp.draft, left + 1, stop=self.eng.spec.eos,
                                on_token=shown, samp=samp, context=list(self.fed),
                                rng=getattr(self.pick, "rng", None), halt=stop)
        turn.mtp_iters += st.iterations
        turn.mtp_accepted += sum(st.accepted)
        return st.tokens[1:], sum((r or {}).get("cycles", 0) for _, _, r in st.runs)

    def _decode(self, logits, out: list[int], turn: Turn, t0: float, on_update, stop) -> str:
        """Generate after `out` (the reply so far) from `logits` (or, after a reply cut on the
        device, from its last token, an int: picked, not fed yet); returns the whole reply."""
        if self.on_card:
            return self._decode_card(logits, out, turn, t0, on_update, stop)
        if isinstance(logits, int):          # the reply's last token, picked on the device
            self.fed.append(logits)
            logits = self.eng.step(logits)
        k1 = len(self.eng.stats)
        out, n0, t_first = list(out), len(out), None
        detok = Detok(self.tok, out)
        shown = self.tok.decode(out, skip_special_tokens=True) if out else ""
        sink = None                          # the next pick, fed while the device runs
        while logits is not None and not turn.end:
            if len(out) - n0 >= self.max_new:
                turn.end = "max_new"
                break
            t = sink.result() if sink is not None else self.pick(logits, self.fed)
            if t in self.eng.spec.eos:
                turn.end = "eos"
                break
            out.append(t)
            now = time.perf_counter()
            if t_first is None:
                t_first, turn.ttft_s = now, now - t0
            turn.gen_tokens, turn.decode_s = len(out) - n0, now - t_first
            if self.eng.pos >= self.eng.cap:
                turn.end = "cap"
            elif stop():
                turn.end = "stopped"
            if turn.end:
                delta = detok.add(t)
                shown += delta
                on_update(delta, turn)
                break

            def started(t=t):
                # the token's text and the interface's drawing once the card runs the next
                # one: they overlap the run instead of delaying its start
                nonlocal shown
                delta = detok.add(t)
                shown += delta
                on_update(delta, turn)
            # the sampler takes the logits as they come (the next pick's context includes t)
            self.fed.append(t)
            stream = getattr(self.pick, "stream", None)
            sink = stream(self.fed) if stream is not None else None
            try:
                logits = self.eng.step(t, on_start=started, sink=sink)
            except BaseException:
                self.fed.pop()
                raise
            turn.decode_steps = len(self.eng.stats) - k1
            turn.decode_cycles += (self.eng.stats[-1] or {}).get("cycles", 0)
            turn.context = self.eng.pos
        self._next = logits if turn.end == "max_new" else None
        return self._finish(out, shown, turn, on_update)

    def _finish(self, out: list[int], shown: str, turn: Turn, on_update) -> str:
        self._reply = out
        reply = self.tok.decode(out, skip_special_tokens=True)
        if len(reply) > len(shown) and reply.startswith(shown):
            on_update(reply[len(shown):], turn)       # a held-back incomplete character
        return reply

    def _decode_card(self, logits, out: list[int], turn: Turn, t0: float, on_update,
                     stop) -> str:
        """_decode with the decode loop on the device: the host picks the first token from the
        prefill's logits (resume: the reply's last token, picked on the device), then each
        Engine.generate_card run feeds it and generates the rest there (the sampler's settings
        and generator), up to max_new or an EOS id; the tokens are shown as they land. The
        last token of a run is not fed (a cut reply resumes from it)."""
        eng = self.eng
        samp = None if getattr(self.pick, "greedy", False) else G.Sampling(**self.pick.params)
        out, n0 = list(out), len(out)
        detok = Detok(self.tok, out)
        shown = self.tok.decode(out, skip_special_tokens=True) if out else ""
        t_first = None

        def take(t: int) -> None:
            nonlocal shown, t_first
            if t in eng.spec.eos:
                return
            out.append(t)
            now = time.perf_counter()
            if t_first is None:
                t_first, turn.ttft_s = now, now - t0
            turn.gen_tokens, turn.decode_s = len(out) - n0, now - t_first
            delta = detok.add(t)
            shown += delta
            on_update(delta, turn)

        if isinstance(logits, _Picked):     # by MTP's prefill
            t = int(logits)
            if t in eng.spec.eos:
                turn.end = "eos"
            take(t)
        elif isinstance(logits, int):
            t = logits
        elif logits is not None and not turn.end:
            t = self.pick(logits, self.fed)
            if t in eng.spec.eos:
                turn.end = "eos"
            take(t)
        else:
            t = None
        room = eng.cap - (2 if self.use_mtp else 0)     # (a verify's two rows)
        while t is not None and not turn.end:
            left = self.max_new - (len(out) - n0)
            if left <= 0:
                turn.end = "max_new"
            elif eng.pos >= room:
                turn.end = "cap"
            elif stop():
                turn.end = "stopped"
            if turn.end:
                break
            k1 = len(eng.stats)
            self.fed.append(t)
            try:
                if self.use_mtp:
                    got, cycles = self._mtp_run(t, left, take, stop, samp, turn)
                else:
                    got = eng.generate_card(t, left, stop_ids=eng.spec.eos, on_token=take,
                                            stop=stop, sampling=samp, context=self.fed,
                                            rng=getattr(self.pick, "rng", None))
                    cycles = sum((st or {}).get("cycles", 0) for st in eng.stats[k1:])
            except BaseException:
                self.fed.pop()
                raise
            self.fed += got[:-1]                  # the last one is fed by the next run
            turn.decode_steps += len(got)
            turn.decode_cycles += cycles
            turn.context = eng.pos
            if not got:
                break
            t = got[-1]
            if t in eng.spec.eos:
                turn.end = "eos"
        self._next = t if turn.end == "max_new" else None
        return self._finish(out, shown, turn, on_update)

    def ask_plain(self, text: str | None, stream=sys.stdout) -> str:
        """ask() (resume() for None) printing the reply as it streams, then the turn's
        numbers."""
        def show(delta, turn):
            stream.write(delta)
            stream.flush()
        reply, turn = self.ask(text, show) if text is not None else self.resume(show)
        stream.write("\n" + turn.line() + "\n")
        if turn.end in NOTES:
            stream.write(NOTES[turn.end].format(max_new=self.max_new, cap=turn.cap,
                                                cmd="/continue") + "\n")
        return reply


NOTES = {"max_new": "(stopped at max_new={max_new} tokens · {cmd} or raise --max-new)",
         "cap": "(context full at {cap} tokens · /reset to start a new conversation)",
         "stopped": "(interrupted)"}


def make_backend(name: str, spec, cap: int, dev: str, model: str | None = None,
                 lookup: bool = False, wformat: str = "int8", head_format: str | None = None):
    """(backend, configuration) for Engine (lookup: room for the resident decode's tables;
    the weight formats size the board model's DRAM where an int8 image would not fit)."""
    if name == "isa":
        return "isa", None
    if name == "board":
        from opentpu.host.board import (Board, BoardBackend, XdmaTransport,   # the PCIe driver
                                        device_config)
        tr = XdmaTransport(dev)
        cfg = device_config(Board(tr, lock=False).info())     # MCOLS / LANES of the bitstream
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model)), cfg
    if name == "board-sim":
        from opentpu.host.board import BoardBackend, SimTransport, sim_config
        cfg = sim_config(spec, cap, lookup=lookup, wformat=wformat, head_format=head_format)
        tr = SimTransport(ch_bytes=cfg.DRAM_BYTES // 2)
        return (lambda c, imgs: BoardBackend(c, imgs, transport=tr, model=model)), cfg
    if name == "rtl":
        from opentpu.llm.rtl_backend import RtlBackend
        return RtlBackend, None
    raise SystemExit(f"unknown backend {name}")


def weights_label(image, wformat: str) -> str:
    """An image's weight formats in words, as otpu-chat and otpu-smi show them: "mix: int8 +
    gateup@9-35=fp4, head int8" (wformat the choice before formats.named: "mix" names it)."""
    rest = ",".join(i for i in (image.formats or "").replace(" ", "").split(",")
                    if i and not i.startswith("head="))
    return (f"{FM.MIX}: " if wformat == FM.MIX else "") + image.wformat + \
        (f" + {rest}" if rest else "") + f", head {image.head_format}"


def panel_meta(eng, backend: str, dev: str, model: str, sp: dict, max_new: int,
               clock_mhz: float = 0.0, weights: str | None = None) -> dict:
    """What the interface shows about the model, its weights, the device and the sampling."""
    info = getattr(eng.backend, "info", None)
    meta = {"model": model, "backend": backend, "sampling": {**sp, "max_new": max_new},
            "weights": weights,
            "device": {"board": dev, "board-sim": "Verilator board model"}.get(
                backend, "ISA simulator (host)"), "dram": None,
            "short": f"{backend} {clock_mhz:g} MHz" if clock_mhz else backend}
    c = eng.cfg
    if info:
        meta["bitstream"] = [f"D={info['D']} MCOLS={info['MCOLS']} LANES={info['LANES']}",
                             "build " + ("n/a" if info["build_id"] is None
                                         else f"{info['build_id']:08x}")
                             + (f", {info['core_khz'] / 1e3:g} MHz" if info["core_khz"]
                                else "")]
    else:
        meta["bitstream"] = [f"D={c.D} MCOLS={c.MCOLS} (simulated)"]
    if hasattr(eng.backend, "image_bytes"):
        from opentpu.host.board import dram_layout
        be = eng.backend
        meta["dram"] = lambda: dram_layout(c, be.image_bytes, be.prog_at, eng.image,
                                           list(eng.poss))
    return meta


@busy_exits
def main(argv=None):
    ap = argparse.ArgumentParser(prog="otpu-chat", description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3",
                    help=f"{' or '.join(MODELS)} (models/<name>), or a checkpoint directory")
    ap.add_argument("--backend", default="isa", choices=["isa", "board", "board-sim", "rtl"])
    ap.add_argument("--dev", default="/dev/xdma0", help="XDMA device prefix (--backend board)")
    ap.add_argument("--clock-mhz", type=float,
                    help="core clock, to turn device cycles into tokens/s (default: the "
                         "bitstream's CORE_KHZ, or 100 on a register map 1 bitstream)")
    ap.add_argument("--cap", type=int, default=2048, help="KV cache capacity (tokens)")
    ap.add_argument("--per-position", action="store_true",
                    help="compile a decode program per position (the fallback; by default "
                         "one resident program per 256-token bucket takes the position as a "
                         "run argument when the bitstream has them, CAPS bit25)")
    ap.add_argument("--prompt", help="ask one question and exit (plain output)")
    ap.add_argument("--plain", action="store_true",
                    help="a line-by-line REPL instead of the full-screen interface")
    ap.add_argument("--think", action="store_true", help="enable Qwen3 / Qwen3.5 thinking mode")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float,
                    help="sampling flags default per model: " + "; ".join(
                        f"{m} " + " ".join(f"{k}={v}" for k, v in d.items())
                        for m, d in SAMPLING.items()))
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--top-p", type=float)
    ap.add_argument("--repetition-penalty", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-new", type=int, default=1024, help="tokens per reply at most")
    ap.add_argument("--wformat", default=FM.AUTO, choices=[FM.AUTO, "int8", "fp4", "int4", FM.MIX],
                    help="weight format of the layers (docs/quant.md; fp4 needs a bitstream "
                         "with 4-bit MM support): auto (default) is the model's recommended "
                         "mix of int8 and fp4 where it has one (mix, docs/formats.md), else "
                         "int8")
    ap.add_argument("--head-format", default=None, choices=["int8", "fp4", "int4"],
                    help="weight format of the LM head (default: --wformat)")
    ap.add_argument("--no-prog-cache", action="store_true",
                    help="compile the decode loop's bucket programs in every process (by "
                         "default they are kept on disk: opentpu/progcache.py)")
    ap.add_argument("--no-prompt-runs", action="store_true",
                    help="compile each prompt's prefill runs at its positions (by default a "
                         "dense model's prompt runs programs at run-time positions, compiled "
                         "once per bucket: docs/prefill.md)")
    ap.add_argument("--mtp", action="store_true",
                    help="Qwen3.5: decode with the model's MTP drafter, the loop on the device "
                         "(docs/mtp.md): the same replies, greedy or sampled, 1.3-1.6x the "
                         "decode tok/s")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer
    path = model_dir(a.model)
    tok = AutoTokenizer.from_pretrained(path)
    spec = load_spec(path)
    wformat = FM.auto(spec, a.wformat)
    print(f"loading {path.name} onto openTPU ({a.backend}) ...", flush=True)
    from opentpu.host.board import ConfigMismatch
    try:
        backend, cfg = make_backend(a.backend, spec, a.cap, a.dev, path.name,
                                    lookup=not a.per_position, wformat=wformat,
                                    head_format=a.head_format)
    except ConfigMismatch as e:
        raise SystemExit(f"otpu-chat: {e}") from None
    if a.mtp and (not hasattr(spec, "mtp") or a.per_position):
        raise SystemExit("otpu-chat: --mtp takes a Qwen3.5 model and resident decode")
    if a.mtp and cfg is None:   # the simulators' default is the design's (two slices): the
        from dataclasses import replace     # card's one-slice configuration instead

        from opentpu.isasim import board_config
        from opentpu.llm.qwen3 import PREFILL_ROWS
        flags = dict(DSTEP=True, STREAM=True, PAIR=True)
        need = replace(spec, mtp=True).image(board_config(DRAM_BYTES=1 << 40, **flags), a.cap,
                                             1, PREFILL_ROWS, wformat, a.head_format,
                                             lookup=True).nbytes
        cfg = board_config(DRAM_BYTES=1 << max(20, (need - 1).bit_length()), **flags)
    try:
        if a.mtp:
            from opentpu.llm.mtp import mtp_engine
            eng = mtp_engine(spec, load_weights(path, mtp=True), cap=a.cap, cfg=cfg,
                             backend=backend, wformat=wformat, head_format=a.head_format,
                             prog_cache=not a.no_prog_cache, prompt_runs=not a.no_prompt_runs)
        else:
            eng = Engine(spec, load_weights(path), cap=a.cap, cfg=cfg, backend=backend,
                         wformat=wformat, head_format=a.head_format,
                         resident=not a.per_position, prog_cache=not a.no_prog_cache,
                         prompt_runs=not a.no_prompt_runs)
    except ConfigMismatch as e:
        raise SystemExit(f"otpu-chat: {e}") from None
    weights = weights_label(eng.image, wformat)
    print(f"weights: {weights}", flush=True)
    if getattr(eng.backend, "status", None) is not None:
        eng.backend.status.update(weights=weights)      # for otpu-smi
    sp = sampling(spec, a)
    pick = sampler(0 if a.greedy else sp["temperature"], sp["top_k"], sp["top_p"], a.seed,
                   sp["repetition_penalty"])
    clock = 0.0
    if a.backend.startswith("board"):
        khz = eng.backend.info.get("core_khz")
        clock = a.clock_mhz or (khz / 1e3 if khz else 100.0)
    try:
        chat = Chat(eng, tok, a.think, pick, a.max_new, clock_mhz=clock)
    except ValueError as e:                 # MTPDecoder's checks (one slice; PAIR, MCOLS)
        raise SystemExit(f"otpu-chat: --mtp: {e}") from None
    if a.mtp and not chat.use_mtp:
        print("otpu-chat: --mtp: these sampling settings stay on the host; plain decode",
              flush=True)
    if a.prompt:
        chat.ask_plain(a.prompt)
        return
    if not a.plain:
        from opentpu.host.chat_tui import ChatApp
        ChatApp(chat, panel_meta(eng, a.backend, a.dev, path.name,
                                 dict(sp, greedy=a.greedy), a.max_new, clock, weights)).run()
        return
    print("type a message (/continue, /reset; empty line or Ctrl-D to quit)")
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            break
        if not text:
            break
        if text == "/reset":
            chat.reset()
        elif text == "/continue":
            chat.ask_plain(None) if chat.can_resume else print("nothing to continue")
        else:
            chat.ask_plain(text)


if __name__ == "__main__":
    main()
