"""otpu-serve35x2 (boards/bcu1525/chat35/serve35x2.py, otpu-serve35x2): otpu-serve35's server on both BCU1525s as one
machine, without a card. The wrapper's order on a stub card host (test_chat35_wrappers' harness, two cards): each
card's bring-up in a session of its own, the server's options and both devices passed through, Ctrl-C and a hangup
stopping the server first and then both bring-ups (each card's SAFE line from its wg.json), either card's hold ending
stopping the server and the other card. The cards' telemetry as one machine (telemetry.CardsSampler over two in-memory
BCU1525s: the hottest die, the governors' RUN together, the power summed, each card by name; a card that fails n/a,
never the other's alone) and the monitor's choice of it for "/dev/a+/dev/b". serve35x2's engine from chat35x2 (its
open_engine, or its main over a stand-in for otpu-chat's Chat). With opentpu.host.cards (skipped without
it): a context's pages and state through the staging area on the S = 4 machine of two cards (IsaCards: the ISA
simulator behind the cards' registers), written back after another context bit for bit against the ISA simulator's
machine fed the same tokens."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from opentpu.host.fake import FakeBcu
from opentpu.serve import telemetry as TM

from test_chat35_wrappers import BASH, CHAT35, GOV, LOG, SERVER, _host

TWO = {"0000:0a:00.0": "xdma1", "0000:0b:00.0": "xdma0"}      # card 0 (CRD0), card 1 (CRD1): test_chat35_wrappers' CARDS
BRINGUP2 = LOG + '''import json, signal, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
name = sys.argv[sys.argv.index("--serial") + 1][-4:]
dev = sys.argv[sys.argv.index("--dev") + 1]
def interrupted(sig, frame):                    # (wg_bringup.main's)
    raise KeyboardInterrupt(f"signal {sig}")
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
out.mkdir(parents=True, exist_ok=True)
log(f"bringup {name} {dev} up, own session {os.getsid(0) == os.getpid()}, tree {os.path.samefile(sys.argv[sys.argv.index('--tree') + 1], '.')}")
(out / "HOLDING").write_text("held")
try:
    while True:
        time.sleep(0.05)
        if os.environ.get("STUB_END") == name and "server up" in open(os.environ["STUB_EVENTS"]).read():
            time.sleep(0.3)
            log(f"bringup {name} hold ended")
            print("hold: 1 min reached", flush=True)
            break
except KeyboardInterrupt as e:
    log(f"bringup {name} stopped ({e})")
finally:
    (out / "HOLDING").unlink(missing_ok=True)
    time.sleep(0.3)                             # (g.safe(), the guard's stop)
    (out / "wg.json").write_text(json.dumps({"status_safe": {"run": False, "acc_on": False}}))
    log(f"bringup {name} safe")
'''
SERVER2 = SERVER.replace('log("server up")', 'log("server up " + " ".join(sys.argv[1:]) + " dev " + '
                                              'os.environ.get("CHAT35_DEV", ""))')
SERVER2 = SERVER2.replace('signal.signal(', '''if "OTPU_LOCK_HELD" in os.environ:              # (an enclosing otpu-lock's)
    log(f"server lock {os.environ['OTPU_LOCK_HELD']} {os.environ.get('OTPU_LOCK_PID')} {'OTPU_STRAY' in os.environ}")
signal.signal(''', 1)


def _host2(tmp_path: Path, cards=TWO) -> tuple[Path, dict]:
    """test_chat35_wrappers' stub host for otpu-serve35x2: the x2 tree's bring-up (tools/x2_session/bringup.py),
    serve35x2.py the server, `ls` showing each card's XDMA device (cards: PCI function -> device), tmux's fan
    session there."""
    ev, env = _host(tmp_path, "otpu-serve35x2")
    t = next((tmp_path / "otpu-build" / "cardimg" / "x2").glob("tree-*"))
    (t / "tools" / "x2_session").mkdir(parents=True, exist_ok=True)      # (the chat's harness makes it)
    (t / "tools" / "x2_session" / "bringup.py").write_text(BRINGUP2)
    (t / "opentpu" / "host" / "gov.py").write_text(GOV)
    (t.parent / "serve35x2.py").write_text(SERVER2)
    bins = tmp_path / "bin"
    case = "".join(f"  /sys/bus/pci/devices/{f}/xdma/) echo {d}_user ;;\n" for f, d in cards.items())
    (bins / "ls").write_text(f'#!/bin/sh\ncase "$1" in\n{case}  /sys/*) exit 2 ;;\n  *) exec /bin/ls "$@" ;;\nesac\n')
    (bins / "tmux").write_text("#!/bin/sh\nexit 0\n")
    (bins / "tmux").chmod(0o755)
    return ev, env


def _run2(tmp_path, sig, ready, args=("--port", "8001"), end=None, cards=TWO, extra=None):
    """otpu-serve35x2 as a foreground job (extra: the caller's environment besides); `sig` to its group once `ready`
    is in the events (None: wait for its exit). Its exit code, output and events."""
    ev, env = _host2(tmp_path, cards)
    env.update(extra or {})
    if end:
        env["STUB_END"] = end
    out = tmp_path / "out.txt"
    with open(out, "w") as f:
        p = subprocess.Popen([BASH, str(CHAT35 / "otpu-serve35x2"), *args], stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        t = time.monotonic()
        while sig is not None and not (ev.exists() and ready in ev.read_text()):
            assert p.poll() is None, out.read_text()
            assert time.monotonic() - t < 60, out.read_text()
            time.sleep(0.05)
        if sig is not None:
            time.sleep(0.3)
            os.killpg(p.pid, sig)
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    return rc, out.read_text(), ev.read_text().splitlines() if ev.exists() else []


NEEDS = pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
UP = {"bringup CRD0 /dev/xdma1 up, own session True, tree True", "bringup CRD1 /dev/xdma0 up, own session True, tree True"}


def _after_server(ev, i):
    """Each card's events after the server's: stopped, then safe."""
    rest = ev[i:]
    for n in ("CRD0", "CRD1"):
        mine = [e for e in rest if e.startswith(f"bringup {n} ")]
        assert len(mine) == 2 and mine[1] == f"bringup {n} safe", ev
    return rest


@NEEDS
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP])
def test_serve35x2_stops_the_server_then_both_bringups(tmp_path, sig):
    rc, out, ev = _run2(tmp_path, sig, "server up")
    assert set(ev[:2]) == UP, (ev, out)
    assert ev[2:5] == ["server up --port 8001 dev /dev/xdma1+/dev/xdma0", "server term", "server exit"], (ev, out)
    rest = _after_server(ev, 5)
    assert set(rest) == {f"bringup {n} {e}" for n in ("CRD0", "CRD1") for e in
                         (f"stopped (signal {int(signal.SIGTERM)})", "safe")}, ev
    assert rc == 130, out
    assert "otpu-serve35x2: CRD0 back in SAFE" in out and "otpu-serve35x2: CRD1 back in SAFE" in out, out


@NEEDS
def test_serve35x2_keeps_an_enclosing_otpu_lock(tmp_path):
    """otpu-serve35x2 inside a card window's `otpu-lock --dev /dev/xdma1 -- ...`: the server sees the hold's
    OTPU_LOCK_HELD and OTPU_LOCK_PID (otpu-chat35x2's: the bring-ups borrow that card's lock), no other OTPU_* of the
    caller's (the image key's environment stays the entry's)."""
    rc, out, ev = _run2(tmp_path, signal.SIGINT, "server up", extra={"OTPU_LOCK_HELD": "/dev/xdma1",
                                                                    "OTPU_LOCK_PID": "4242", "OTPU_STRAY": "1"})
    assert "server lock /dev/xdma1 4242 False" in ev, (ev, out)
    assert rc == 130, out


@NEEDS
def test_either_cards_hold_ending_stops_the_server_and_the_other_card(tmp_path):
    rc, out, ev = _run2(tmp_path, None, None, end="CRD0")
    assert set(ev[:2]) == UP and ev[2].startswith("server up"), (ev, out)
    i = ev.index("bringup CRD0 hold ended")
    assert ev.index("server term") > i and ev.index("server exit") > ev.index("server term"), ev
    assert ev.index("bringup CRD1 stopped (signal 15)") > ev.index("server exit"), ev
    assert {"bringup CRD0 safe", "bringup CRD1 safe"} <= set(ev), ev
    assert rc == 1 and "CRD0: the bring-up's hold ended (hold: 1 min reached)" in out, out
    assert "CRD0 back in SAFE" in out and "CRD1 back in SAFE" in out, out


@NEEDS
def test_a_card_without_its_device_is_refused_before_any_bringup(tmp_path):
    rc, out, ev = _run2(tmp_path, None, None, cards={"0000:0b:00.0": "xdma0"})
    assert rc == 1 and ev == [] and "no XDMA device on CRD0 ('')" in out, out


# ---------------------------------------------------------------------------- telemetry
def _cards(*fakes, names=("CRD0", "CRD1")):
    cs = TM.CardsSampler([f.dev for f in fakes], 2.0, open_transport=None, names=list(names))
    for c, f in zip(cs.cards, fakes):
        c.open_transport = lambda dev, f=f: f
    return cs


def test_two_cards_sample_as_one_machine():
    """Two in-memory BCU1525s in RUN, the CRD1 hotter: the machine's die the CRD1's (its source named), the governor
    RUN, the power the cards' sum (labelled so), the DRAM traffic summed, each card by name with its own die; one
    card's register transport each."""
    a = FakeBcu(run=True, die_c=55.0, kick_ms=300, run_ms=754_000, devname="xdma1")
    b = FakeBcu(run=True, die_c=63.0, kick_ms=300, run_ms=754_000, devname="xdma0")
    cs = _cards(a, b)
    one = [TM.CardSampler(f.dev, 2.0, open_transport=lambda dev, f=f: f).sample() for f in (a, b)]
    m = cs.sample()
    assert cs.dev == "/dev/xdma1+/dev/xdma0" and m["device"] == cs.dev and m["ok"]
    assert m["die_c"] == one[1]["die_c"] and m["die_src"] == f"{one[1]['die_src']}, CRD1"
    assert m["gov"]["state"] == "RUN" and m["gov"]["acc_on"] and m["gov"]["trips_thermal"] == 0
    assert m["power"]["w"] == pytest.approx(one[0]["power"]["w"] + one[1]["power"]["w"], abs=0.02)
    assert m["power"]["label"].endswith("2 cards summed") and m["power"]["vin"] is None
    assert [c["name"] for c in m["cards"]] == ["CRD0", "CRD1"]
    assert [c["die_c"] for c in m["cards"]] == [one[0]["die_c"], one[1]["die_c"]]
    assert m["bitstream"]["build"] == one[0]["bitstream"]["build"] and m["ddr_mts"] == one[0]["ddr_mts"]
    assert cs.samples == 1 and cs.failures == 0 and cs.error is None and cs.took_s is not None
    # one card in SAFE: the machine is not in RUN
    b.run = b.acc_on = False
    assert cs.sample()["gov"]["state"] == "SAFE"


def test_a_card_that_fails_is_na_not_the_others_alone():
    """The CRD1's transport gone: the machine's die, power, governor and utilization n/a (None), the CRD0's own
    sample still under its name, the error by name; both gone: the sample raises, as CardSampler's does."""
    a = FakeBcu(run=True, die_c=55.0, kick_ms=300, run_ms=754_000, devname="xdma1")
    cs = TM.CardsSampler(["/dev/xdma1", "/dev/xdma0"], 2.0, names=["CRD0", "CRD1"])

    def gone(dev):
        raise OSError(19, "No such device", dev)
    cs.cards[0].open_transport = lambda dev: a
    cs.cards[1].open_transport = gone
    m = cs.sample()
    assert not m["ok"] and m["die_c"] is None and m["power"] is None and m["gov"] is None and m["util"] is None
    assert m["cards"][0]["name"] == "CRD0" and m["cards"][0]["die_c"] is not None
    assert m["cards"][1] == {"name": "CRD1", "device": "/dev/xdma0", "ok": False}
    assert cs.error.startswith("CRD1: OSError") and "No such device" in cs.error and cs.failures == 1
    cs.cards[0].open_transport = gone
    cs.cards[0]._t = None
    with pytest.raises(RuntimeError, match="CRD0: OSError"):
        cs.sample()
    assert cs.failures == 2 and cs.samples == 1


def test_the_monitor_samples_both_cards_for_a_two_card_device():
    """api.card_monitor: --dev "/dev/a+/dev/b" -> CardsSampler with the cards' names; one device -> CardSampler."""
    from opentpu.serve import api as API
    w = SimpleNamespace(eng=SimpleNamespace(pos=0, cap=512, batch=1), stats={}, observers=[])
    a = SimpleNamespace(backend="board", telemetry=2.0, telemetry_i2c=False, dev="/dev/xdma1+/dev/xdma0",
                        card_names="CRD0,CRD1")
    mon = API.card_monitor(a, w, "m", {})
    assert isinstance(mon.card, TM.CardsSampler) and mon.card.names == ["CRD0", "CRD1"]
    assert [c.dev for c in mon.card.cards] == ["/dev/xdma1", "/dev/xdma0"]
    mon.stop()
    a.dev = "/dev/xdma0"
    mon = API.card_monitor(a, w, "m", {})
    assert type(mon.card) is TM.CardSampler and mon.card.dev == "/dev/xdma0"
    mon.stop()


# ---------------------------------------------------------------------------- serve35x2.py
def _serve35x2(monkeypatch, x2: ModuleType):
    sys.path.insert(0, str(CHAT35))
    monkeypatch.setitem(sys.modules, "chat35x2", x2)
    import serve35x2
    return serve35x2


def test_the_engine_from_chat35x2s_main(monkeypatch):
    """chat35x2 without open_engine: its main run over the stand-in (its Chat.ask wrap applied to the stand-in), the
    options passed with --prompt -; what main opened returned, warm_layers None; otpu-chat's Chat and sys.argv
    restored. With open_engine: that one."""
    import opentpu.host.chat as C
    seen = {}

    def main():
        seen["argv"] = list(sys.argv[1:])
        C.Chat.ask = (lambda ask: lambda self, *a, **k: ask(self, *a, **k))(C.Chat.ask)
        chat = C.Chat("eng", "tok", False, "pick", 6, 150.0)
        chat.ask_plain("-")
        return 0
    chat, ask, argv = C.Chat, C.Chat.ask, list(sys.argv)
    x2 = ModuleType("chat35x2")
    x2.main = main
    S = _serve35x2(monkeypatch, x2)
    o = S.open_engine(["--think"])
    assert (o.eng, o.tok, o.pick, o.clock_mhz, o.warm_layers) == ("eng", "tok", "pick", 150.0, None)
    assert seen["argv"] == ["--think", "--prompt", "-"]
    assert C.Chat is chat and C.Chat.ask is ask and sys.argv == argv
    x2.open_engine = lambda argv=(): SimpleNamespace(eng="own", argv=argv)
    assert S.open_engine(["-x"]).argv == ["-x"]


def test_serve35x2_main_is_serve35s_on_both_cards(monkeypatch):
    """CHAT35_DEV must name both cards; serve35.main gets them, serve35x2's engine, its label and the cards' names."""
    x2 = ModuleType("chat35x2")
    S = _serve35x2(monkeypatch, x2)
    import serve35
    got = {}
    monkeypatch.setattr(serve35, "main", lambda argv=None, **kw: got.update(kw, argv=argv) or 0)
    monkeypatch.setenv("CHAT35_DEV", "/dev/xdma0")
    with pytest.raises(SystemExit, match="must name both cards"):
        S.main([])
    monkeypatch.setenv("CHAT35_DEV", "/dev/xdma1+/dev/xdma0")
    assert S.main(["--port", "8001"]) == 0
    assert got["argv"] == ["--port", "8001"] and got["dev"] == "/dev/xdma1+/dev/xdma0"
    assert got["label"] == "otpu-serve35x2" and got["card_names"] == S.NAMES == "card0,card1"
    assert got["open_engine"] is S.open_engine


def test_serve35_parses_as_otpu_serve35x2():
    sys.path.insert(0, str(CHAT35))
    import serve35
    a = serve35.parse(["--port", "8001"], "otpu-serve35x2", "x2")
    assert a.port == 8001 and a.served_model_name == "qwen36-35b"
    with pytest.raises(SystemExit):
        serve35.parse(["--help"], "otpu-serve35x2", "x2")


# ---------------------------------------------------------------------------- the cards (their sim view)
def test_a_context_saved_and_written_back_on_two_cards(monkeypatch):
    """The S = 4 machine of two cards (bcu1525-x2:st16, CardsTransport over IsaCards): a context of 13 tokens (its
    attention layer's page and the DeltaNet layer's state, each slice's) read through the staging area and directly
    (the same bytes); each slice's page one of the two KV heads' (each on two slices: the store keeps it twice);
    another context in between; written back through the area: the next tokens' logits and greedy steps those of the
    ISA simulator's S = 4 machine fed the context and them, bit for bit."""
    cards = pytest.importorskip("opentpu.host.cards")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from opentpu import target as T
    from opentpu.host.board import BoardBackend
    from opentpu.llm.qwen3 import Engine
    from opentpu.serve import kvio as KV
    from test_moe_s2 import _tiny
    monkeypatch.setenv("OTPU_HIGH", "0")
    monkeypatch.setenv("OTPU_POOL", "1")
    W, spec = _tiny(("linear", "attn"), nk=8, nv=16, H=512, F=512)
    cfg = T.config(T.load("bcu1525-x2:st16"), S=4, DRAM_BYTES=1 << 26, BHW=0)
    kw = dict(wformat="fp4", rows=1, resident=True)
    sh = cards.IsaCards(cfg)
    eng = Engine(spec, W, cap=256, cfg=cfg, **kw,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=cards.CardsTransport(sh.cards),
                                                      status=False))
    area = KV.staging_area(eng)
    assert cfg.S == 4 and area is not None
    io, direct = KV.ContextIO(eng, area=area), KV.ContextIO(eng, area=None)
    assert io.layout.stateful
    r = np.random.default_rng(7)
    A = [int(x) for x in r.integers(0, 1000, 13)]
    B = [int(x) for x in r.integers(0, 1000, 9)]
    nxt = [int(x) for x in r.integers(0, 1000, 2)]
    eng.prefill(A)
    pages = io.layout.split(len(A))
    saved, state = io.read_pages(pages), io.read_state()
    assert all(np.array_equal(x, y) for x, y in zip(saved, direct.read_pages(pages)))
    assert np.array_equal(state, direct.read_state()) and io.stats["copies"] >= 1
    per = saved[0].reshape(cfg.S, -1)
    assert len({per[s].tobytes() for s in range(cfg.S)}) == 2      # (2 KV heads on 4 slices)
    eng.reset()
    eng.prefill(B)
    io.write_pages([(i, m, b) for (i, m), b in zip(pages, saved)])
    io.write_state(state)
    eng.adopt(len(A))
    got = eng.prefill(nxt)
    ref = Engine(spec, W, cap=256, cfg=cfg, **kw)
    want = ref.prefill(A + nxt)
    assert np.array_equal(np.asarray(got, np.float32).view(np.uint32), np.asarray(want, np.float32).view(np.uint32))
    t = int(np.argmax(got))
    for _ in range(2):
        x, y = eng.step(t), ref.step(t)
        assert np.array_equal(np.asarray(x, np.float32).view(np.uint32), np.asarray(y, np.float32).view(np.uint32))
        t = int(np.argmax(x))
