"""otpu-serve35's path (boards/bcu1525/chat35/serve35.py) on the ISA simulator: a tiny
Qwen3.5-MoE on the 35B's image (bcu1525-wg:bhw16: Image.high, the bank-half window, ST16
states), resident, its prompts layer-major (Engine(layer_major=2)), served by the scheduler's
worker in turns: each context (DeltaNet states, convolution windows, attention KV) saved to the
KV store and written back through a staging area below the window's guard (kvio.staging_area:
qwen3.area_limit), every request's greedy tokens its own fresh run's, a chat's next turn fed
from where the device or the store held it; the monitor follows it (a prompt's progress, the
requests' records). chat35's setup as a function (install, open_engine's arguments) without a
card."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu import target as T
from opentpu.llm.qwen3 import Engine, IsaBackend, low_top
from opentpu.serve import kvio as KV
from opentpu.serve import metrics as M
from opentpu.serve import scheduler as SC

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

CAP = 512
CHAT35 = Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"


class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(0x100 + int(i)) for i in ids)


class _Staged(IsaBackend):
    """The ISA simulator with a BoardBackend's program area and high bound (prog_at past the
    image's low area and the Engine's areas, `high` its weights' start): kvio stages its copies
    there as on the card, under the simulator's guard (every write below low_top)."""

    def __init__(self, cfg, images):
        super().__init__(cfg, images, adopt=True)
        self.prog_at = -(-images.end // 4096) * 4096
        self.high = images.high


@pytest.fixture(scope="module")
def tiny():
    from test_qwen35_moe import _tiny
    _, W, spec = _tiny()
    return W, spec


def _cfg():
    return T.config("bcu1525-wg:bhw16", DRAM_BYTES=1 << 34)


def _engine(W, spec, backend="isa"):
    return Engine(spec, W, cap=CAP, cfg=_cfg(), resident=True, prompt_runs=False,
                  layer_major=2, snapshots=1, backend=backend)


def _ref(W, spec, prompt, n):
    e = _engine(W, spec)
    lg, out = e.prefill(prompt), []
    while len(out) < n:
        t = int(np.argmax(lg))
        out.append(t)
        if len(out) < n:
            lg = e.step(t)
    return out


def test_the_staging_area_ends_below_a_high_images_weights():
    """kvio.staging_area on the 35B's image: past the program area, ending at low_top (where
    the DeltaNet mixers' weights start in the bank-half window), not STAGING_MAX past it."""
    cfg = _cfg()
    top = low_top(cfg)
    be = SimpleNamespace(prog_at=top - (64 << 20), high=1 << 33)
    for high, want in ((True, top), (False, None)):
        eng = SimpleNamespace(backend=be, cfg=cfg, image=SimpleNamespace(high=high))
        base, n = KV.staging_area(eng)
        assert base >= be.prog_at and base % 4096 == 0
        if want is not None:
            assert base + n == want and n < 64 << 20
        else:
            assert n == KV.STAGING_MAX


def test_the_35b_image_serves_requests_in_turns_with_its_contexts(tiny):
    """Two conversations at once on the bhw16 image (quantum 4: each in several turns, the
    other's context saved and written back through the staging area); their tokens are their
    fresh runs'. The first conversation's next turn feeds only its new tokens. The monitor saw
    the requests end and a prompt's progress; on_idle ran once the queue emptied."""
    W, spec = tiny
    eng = _engine(W, spec, backend=_Staged)
    assert eng.image.high and eng.layer_major == 2 and eng.stateful
    w = SC.Worker(eng, _Tok(), quantum=4, max_running=4)
    assert w.io.area is not None and w.io.area[0] + w.io.area[1] <= low_top(eng.cfg)
    idle = []
    w.on_idle = lambda: idle.append(eng.pos)
    mon = M.Monitor(w, "tiny35", interval=0.05)
    seen = []
    prefill = SC.Worker._prefill

    def watched(self, r):           # what the monitor sees while the prompt is fed
        seen.append(self.feeding[2])
        return prefill(self, r)
    w._prefill = watched.__get__(w)
    w.start()
    mon.start()
    try:
        r = np.random.default_rng(3)
        pa = [int(x) for x in r.integers(0, 1000, 70)]
        pb = [int(x) for x in r.integers(0, 1000, 40)]
        ra = w.submit(SC.Request(pa, max_new=9, at=len(pa)))
        rb = w.submit(SC.Request(pb, max_new=9, at=len(pb)))
        for q in (ra, rb):
            q.wait(timeout=900)
            assert q.error is None and q.finish == "length", q.error
        assert ra.out == _ref(W, spec, pa, 9) and rb.out == _ref(W, spec, pb, 9)
        assert ra.swaps + rb.swaps >= 2 and w.stats["restores"] >= 1
        nxt = pa + ra.out + [int(x) for x in r.integers(0, 1000, 5)]
        rc = w.submit(SC.Request(nxt, max_new=6, at=len(nxt))).wait(timeout=900)
        assert rc.error is None and rc.out == _ref(W, spec, nxt, 6)
        assert rc.cached >= len(pa) and rc.prefill_tokens == len(nxt) - rc.cached
        snap = mon.tick()
        assert snap["requests"]["finish_length"] == 3 and len(mon.recent()) == 3
        assert all(x["prefill_s"] > 0 and x["ttft_s"] > 0 for x in mon.recent())
        assert snap["kv"]["positions"] == eng.pos and snap["kv"]["cap"] == CAP
        assert seen[:2] == [70, 40] and idle
        u = w.usage()
        assert u["kv_store"]["ram_bytes"] > 0 and u["kvio"]
    finally:
        mon.stop()
        w.stop()


def _until(f, timeout=300):
    import time
    t = time.monotonic()
    while not f():
        assert time.monotonic() - t < timeout, "timed out"
        time.sleep(0.02)


def test_the_35b_chats_reply_start_is_a_snapshot_then_the_store(tiny):
    """A chat on the bhw16 image as otpu-serve35 runs it (the engine's one snapshot, kvio's
    staging): A1's reply start a snapshot on the card (no store save on its path), A2 waiting
    meanwhile rolled back to it (no save, no restore); once idle, A2's reply start staged into
    the store (the device back to it); another conversation, then A3 restored from the store.
    The snapshots' area and the staging area apart, below the weights; every request's tokens
    its fresh run's."""
    W, spec = tiny
    eng = _engine(W, spec, backend=_Staged)
    w = SC.Worker(eng, _Tok(), quantum=64, max_running=1)
    sb, sn = eng._snaps.base, eng.snapshot_bytes
    assert sb is not None and sn > 0 and w.io.area is not None
    assert sb + sn <= w.io.area[0] and w.io.area[0] + w.io.area[1] <= low_top(eng.cfg)
    w.start()
    try:
        r = np.random.default_rng(11)
        a1 = [int(x) for x in r.integers(0, 1000, 30)]
        a2 = a1[:26] + [int(x) for x in r.integers(0, 1000, 10)]
        r1 = w.submit(SC.Request(a1, max_new=5, at=26))
        r2 = w.submit(SC.Request(a2, max_new=5, at=len(a2) - 3))
        for q in (r1, r2):
            q.wait(timeout=900)
            assert q.error is None and q.finish == "length", q.error
        assert r1.out == _ref(W, spec, a1, 5) and r2.out == _ref(W, spec, a2, 5)
        assert r1.phases.get("snapshot", 0) > 0 and "save" not in r1.phases
        assert r2.cached == 26 and r2.phases.get("rollback", 0) > 0
        assert "save" not in r2.phases and "restore" not in r2.phases
        _until(lambda: w.stats["idle_saves"] == 1)
        assert w.stats["saves"] == 1 and w.dev == a2[:len(a2) - 3] and not w.dirty
        b = [int(x) for x in r.integers(0, 1000, 20)]
        rb = w.submit(SC.Request(b, max_new=3, at=17)).wait(timeout=900)
        assert rb.error is None and "save" not in rb.phases and rb.out == _ref(W, spec, b, 3)
        a3 = a2[:len(a2) - 3] + [int(x) for x in r.integers(0, 1000, 7)]
        r3 = w.submit(SC.Request(a3, max_new=4, at=len(a3) - 2)).wait(timeout=900)
        assert r3.error is None and r3.cached == len(a2) - 3
        assert r3.phases.get("restore", 0) > 0 and r3.out == _ref(W, spec, a3, 4)
    finally:
        w.stop()


def _chat35(tmp_path):
    """chat35.py from a directory with its configuration's pickle (the card host's cardimg/)."""
    import pickle
    import shutil
    shutil.copy(CHAT35 / "chat35.py", tmp_path / "chat35.py")
    (tmp_path / "bcu1525-wg-bhw16.pkl").write_bytes(pickle.dumps(_cfg()))
    spec = importlib.util.spec_from_file_location("chat35_under_test", tmp_path / "chat35.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_chat35_main_and_open_engine_share_the_setup(tmp_path, monkeypatch):
    """chat35.main() runs otpu-chat with FIXED and the command line's arguments, as before the
    split; open_engine() runs it with FIXED and --prompt - and gives back what the chat would
    have had (the engine, the tokenizer, the sampler, the clock) and warm_layers; otpu-chat's
    Chat is its own again after it."""
    import opentpu.host.chat as C
    from opentpu import imgcache as IC
    from opentpu.host import gen_warm as GW
    from opentpu.llm import qwen3 as Q
    for obj, names in ((C, ("main", "make_backend", "load_spec", "load_weights", "Engine",
                            "Chat")), (C.ST, ("streams",)), (C.Chat, ("ask", "resume")),
                       (IC, ("build",)), (GW, ("start",)), (Q.Engine, ("prefill_chunks",))):
        for n in names:                 # (install's patches undone after the test)
            monkeypatch.setattr(obj, n, getattr(obj, n))
    for k in ("CHAT35_BUILD", "CHAT35_KEY", "CHAT35_PROF", "CHAT35_EXPAND", "CHAT35_TIMES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CHAT35_LM", "2")
    m = _chat35(tmp_path)
    got = []
    chat, ask = C.Chat, C.Chat.ask
    monkeypatch.setattr(C, "main", lambda args: got.append(list(args)) or 0)
    monkeypatch.setattr(sys, "argv", ["chat35.py", "--max-new", "64"])
    assert m.main() == 0
    assert got[-1] == m.FIXED + ["--max-new", "64"]
    assert C.ST.streams(None) is False and C.Chat.ask is not ask      # (install's)

    def fake_main(args):                # what otpu-chat's main does with the Chat it opens
        got.append(list(args))
        c = C.Chat("ENG", "TOK", False, SimpleNamespace(params={"t": 1}), 1024, clock_mhz=150.0)
        assert not c.use_mtp
        c.max_slices = None
        c.ask_plain("-")
    monkeypatch.setattr(C, "main", fake_main)
    o = m.open_engine(["--max-new", "32"])
    assert got[-1] == m.FIXED + ["--max-new", "32", "--prompt", "-"]
    assert (o.eng, o.tok, o.pick.params, o.clock_mhz) == ("ENG", "TOK", {"t": 1}, 150.0)
    assert callable(o.warm_layers) and C.Chat is chat and C.Chat is not o.stub



@pytest.mark.parametrize("loop", [False, True])
def test_chat35_loop_env_reaches_the_engines(tmp_path, monkeypatch, loop):
    """CHAT35_LOOP=1 with CHAT35_LM (otpu-chat35 / otpu-serve35 pass it through; chat35x2's too,
    for otpu-serve35x2): the engine with layer_loop, chat35's warm_layers the loops' keys of the
    context's bucket and the next two, the runs of the buckets behind dropped by their bucket
    (a loop's third: ("L", layer, bucket, rows)); without it the engine and keys as before."""
    import opentpu.host.chat as C
    from opentpu import imgcache as IC
    from opentpu.host import gen_warm as GW
    from opentpu.llm import qwen3 as Q
    for obj, names in ((C, ("main", "make_backend", "load_spec", "load_weights", "Engine",
                            "Chat")), (C.ST, ("streams",)), (C.Chat, ("ask", "resume")),
                       (IC, ("build",)), (GW, ("start",)), (Q.Engine, ("prefill_chunks",))):
        for n in names:                 # (install's patches undone after the test)
            monkeypatch.setattr(obj, n, getattr(obj, n))
    for k in ("CHAT35_BUILD", "CHAT35_KEY", "CHAT35_PROF", "CHAT35_EXPAND", "CHAT35_TIMES",
              "CHAT35_AHEAD", "CHAT35_LOOP"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("CHAT35_LM", "2")
    if loop:
        monkeypatch.setenv("CHAT35_LOOP", "1")
    made, sent = [], []
    L = [("L", 0, 1, 2), ("L", 0, 2, 2), (0, 1, 2, True), (1, 2, 2, True)]

    def engine(*a, **kw):
        made.append(kw)
        return SimpleNamespace(pos=300, block=256, cap=1024, spec=SimpleNamespace(layers=2),
                               layer_major=kw["layer_major"], _layer_next={},
                               _layer_runs=dict.fromkeys(L), _precompile_layers=sent.append)
    monkeypatch.setattr(C, "Engine", engine)
    m = _chat35(tmp_path)
    m.install()
    eng = C.Engine("spec")
    assert made == [{"layer_major": 2}] and eng.layer_loop is loop
    if loop:
        want = ["head"] + [k for b in (2, 3) for k in [("L", -1, b, 1)] + [
            ("L", li, b, n) for li in range(2) for n in (2, 1)]]
    else:
        want = ["head"] + [k for b in (2, 3) for k in [(-1, b, 1, True)] + [
            (li, b, n, True) for li in range(2) for n in (2, 1)]]
    assert sent == [want]
    assert list(eng._layer_runs) == [("L", 0, 2, 2), (1, 2, 2, True)]
    sys.path.insert(0, str(CHAT35))             # chat35x2's engine
    spec_ = importlib.util.spec_from_file_location("chat35x2_under_test", CHAT35 / "chat35x2.py")
    X2 = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(X2)
    for obj, names in ((C, ("Engine", "load_spec", "load_weights", "make_backend", "main")),
                       (C.Chat, ("ask",)), (C.ST, ("streams",)), (IC, ("build",)),
                       (GW, ("start",)), (Q.Engine, ("prefill_chunks",))):
        for n in names:
            monkeypatch.setattr(obj, n, getattr(obj, n))
    from opentpu.host import runwords
    installed = []                              # (chat35x2's engine_lm: runwords on its engine)
    monkeypatch.setattr(runwords, "install", lambda e: installed.append(e) or False)
    monkeypatch.setattr(C, "Engine", lambda *a, **kw: made.append(kw))
    monkeypatch.setattr(C, "main", lambda args: C.Engine("spec") or 0)
    monkeypatch.setenv("OTPU_TARGET", X2.TARGET)
    monkeypatch.setenv("CHAT35_DEV", "/dev/xdma0+/dev/xdma1")
    monkeypatch.setattr(sys, "argv", ["chat35x2.py"])
    assert X2.main() == 0
    assert made[-1] == {"layer_major": 2, **({"layer_loop": True} if loop else {})}
    assert len(installed) == 1

def test_serve35_slots_open_the_engine_with_a_slots_context():
    """otpu-serve35 --slots N opens otpu-chat's engine with --slots N and a slot's share of the
    whole context (--cap 262,144 / N unless --cap says otherwise: two 262,144-token KV caches
    do not fit below 4 GiB); without --slots chat35's FIXED alone."""
    import signal
    sys.path.insert(0, str(CHAT35))
    import serve35
    got = []

    def opened(argv=()):
        got.append(list(argv))
        return None                             # (chat35.open_engine's after CHAT35_EXPAND)
    term = signal.getsignal(signal.SIGTERM)
    try:
        for argv in ([], ["--slots", "2"], ["--slots", "4", "--cap", "32768"]):
            assert serve35.main(argv + ["--port", "0", "--kv-disk-gb", "0"],
                                open_engine=opened) == 0
    finally:
        signal.signal(signal.SIGTERM, term)
    assert got == [[], ["--slots", "2", "--cap", "131072"], ["--slots", "4", "--cap", "32768"]]


def test_serve35s_kv_disk_within_the_free_space(tmp_path, monkeypatch):
    """otpu-serve35 keeps its KV store on disk by default (64 GB under kv35 on the data disk, a
    prompt's prefixes every 2048 positions), cut to what leaves 32 GB free on that filesystem
    (the store's files there counted as room), none under 1 GB; the earlier turns' thinking
    kept by default (--no-preserve-thinking: not)."""
    import shutil
    sys.path.insert(0, str(CHAT35))
    import serve35
    a = serve35.parse([])
    assert (a.kv_disk_gb, a.kv_dir, a.prefix_block) == (64.0, serve35.KV_DIR, 2048)
    assert serve35.KV_DIR.endswith("/kv35")
    assert a.preserve_thinking and not serve35.parse(["--no-preserve-thinking"]).preserve_thinking
    d = tmp_path / "kv35"
    d.mkdir()
    (d / "x.bin").write_bytes(bytes(1 << 20))
    free = {}
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=free["gb"] * 2**30))
    for gb, want in ((200.0, 64.0), (80.0, 48.0), (32.5, 0.0)):
        free["gb"] = gb
        a = serve35.parse(["--kv-dir", str(d)])
        serve35.kv_disk(a)
        assert a.kv_disk_gb == want, gb
