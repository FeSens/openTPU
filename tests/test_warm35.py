"""warm35.py (boards/bcu1525/chat35): a new context's first runs warmed ahead of the chats (start_waits, in bucket 1's
process): the wait entries of the steps before the first run-time position (per-position programs, each with its token
on an image with lookup tables) and of the snapshot copies, in the program cache's disk. After it the board's
computation of their entries (BoardBackend.prepare; start's for a program not prepared) finds every one there."""
import importlib.util
from pathlib import Path

from opentpu import isa as I
from opentpu import waits as WC
from opentpu.host import board as BD
from opentpu.llm.qwen3 import Engine

from test_mtp import CFG
from test_qwen35 import _tiny_model

D = Path(__file__).resolve().parents[1] / "boards/bcu1525/chat35"


def _warm35(monkeypatch):
    monkeypatch.syspath_prepend(str(D))
    spec_ = importlib.util.spec_from_file_location("warm35", D / "warm35.py")
    W5 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(W5)
    return W5


class _Board:
    """BoardBackend's prepare on a WAITC bitstream (the wait entries: _entries), no card."""
    waitc, I, W = True, I, WC
    prepare, _assemble, _entries = BD.BoardBackend.prepare, BD.BoardBackend._assemble, BD.BoardBackend._entries

    def __init__(self, cfg):
        self.cfg, self._prep = cfg, {}


def test_a_contexts_first_runs_are_warmed(monkeypatch, tmp_path):
    """run() for bucket 1 (its bucket's own programs left out here) with the chat template's prefixes: a new context's
    three steps (positions 0 .. conv_lo - 1, the second prefix's tokens), its snapshot and the rollback's restore then
    compute no wait entries, every one read from the disk; in an empty cache each is computed. Without the model's
    tokenizer prefix_tokens gives none."""
    from opentpu import progcache as PC
    from opentpu.host import chat as C
    from opentpu.host import gen_warm as GW
    from opentpu.llm import qwen3 as Q
    W5 = _warm35(monkeypatch)
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "qprog"))
    assert W5.prefix_tokens(3, tmp_path / "none") == []
    _, W, spec = _tiny_model(8, init=0.2)
    eng = Engine(spec, W, cap=512, cfg=CFG, resident=True, snapshots=C.SNAPSHOTS)
    assert eng.device_inputs and eng._conv_lo == 3 and eng._snaps.base is not None
    args = (eng.spec, eng.cfg, eng.cap, eng.batch, eng.rows, eng._image_kw, eng.image.stream_fill, eng.layout,
            eng.block, eng._conv_lo, False, eng.gen_debug, eng.gen_split, True, GW.samplings(None))
    got, start_waits = [], W5.start_waits
    with monkeypatch.context() as mp:          # (bucket 1's own programs: test_progcache's)
        mp.setattr(W5.os, "nice", lambda n: 0)
        mp.setattr(W5, "prefix_tokens", lambda conv_lo: [(5, 6, 7), (5, 8, 7)])
        mp.setattr(W5, "start_waits", lambda *a: got.append(start_waits(*a)) or got[-1])
        mp.setattr(GW, "warm", lambda *a: 0)
        mp.setattr(PC, "has", lambda layout, what: True)
        mp.setattr(W5, "decode_waits", lambda *a: None)
        mp.setattr(Q, "_WORKER", None)
        W5.run(0, 1, args, 1, 1)

    runs, compile, run = [], eng._compile, eng.backend.run

    def compiled(pos, tok=None):
        runs.append(compile(pos, tok))
        return runs[-1]

    def ran(progs, **kw):
        runs.append(progs)
        return run(progs, **kw)
    monkeypatch.setattr(eng, "_compile", compiled)
    for t in (5, 8, 7):
        eng.step(t)
    assert len(runs) == 3
    eng.backend.run = ran
    assert eng.snapshot()
    del eng.backend.run
    for t in (9, 10):
        eng.step(t)
    eng.backend.run = ran
    assert eng.rollback(3) == 3
    del eng.backend.run
    assert len(runs) == 5                      # (the save and the restore: a program each)

    WC._mem.clear()
    c0, d0 = WC.stats["computed"], WC.stats["disk"]
    for progs in runs:
        _Board(eng.cfg).prepare(progs)
    assert WC.stats["computed"] == c0 and WC.stats["disk"] - d0 == len(runs)
    assert got == [6]                          # (4 steps: position 1 twice)
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "empty"))
    WC._mem.clear()
    for progs in runs:
        _Board(eng.cfg).prepare(progs)
    assert WC.stats["computed"] - c0 == len(runs)
