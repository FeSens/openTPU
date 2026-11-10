"""The BCU wrappers' signal order (otpu-serve35, otpu-chat35, otpu-chat, otpu-chat35x2: its two cards'
bring-ups), without a card: each run as
a terminal's foreground job (a process group of its own, HOME a stub card host's), its bring-up a stub that installs
SIGINT and SIGTERM handlers as wg_bringup.main does, its server or chat a stub. Ctrl-C (SIGINT to the group) and a hangup (SIGHUP) stop the server
first, then the bring-up (the wrapper's TERM: its SAFE), then the wrapper says the card is back in SAFE (wg.json's
status_safe); in the chat, Ctrl-C reaches the chat alone (the card still held after it). The governor's check and the
device's are stubs too (an opentpu package in the tree, `ls` of the device's sysfs directory)."""
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

CHAT35 = Path(__file__).resolve().parents[1] / "boards" / "bcu1525" / "chat35"
BASH = os.environ.get("OTPU_TEST_BASH") or shutil.which("bash")

LOG = '''import os
def log(m):
    with open(os.environ["STUB_EVENTS"], "a") as f:
        f.write(m + "\\n")
'''
BRINGUP = LOG + '''import json, signal, sys, time
from pathlib import Path
out = Path(sys.argv[sys.argv.index("--out") + 1])
def interrupted(sig, frame):                    # (wg_bringup.main's)
    raise KeyboardInterrupt(f"signal {sig}")
signal.signal(signal.SIGTERM, interrupted)
signal.signal(signal.SIGINT, interrupted)
if os.environ.get("STUB_CARDS"):                # (which card: its JTAG serial and XDMA device)
    log(f"bringup card {sys.argv[sys.argv.index('--serial') + 1]} {sys.argv[sys.argv.index('--dev') + 1]}")
if os.environ.get("STUB_RESUME"):               # (the RESUME it was given)
    log(f"bringup resume {sys.argv[sys.argv.index('--resume') + 1] if '--resume' in sys.argv else None}")
if os.environ.get("STUB_WAIT"):                 # (go_cool's line while GO is refused, then the GO)
    print("12:00:00 waiting for GO: die 70.1 C, RESUME 65.0 C; 30 of 1800 s", flush=True)
    time.sleep(float(os.environ["STUB_WAIT"]))
log(f"bringup up, own session {os.getsid(0) == os.getpid()}")
(out / "HOLDING").write_text("held")
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt as e:
    log(f"bringup stopped ({e})")
finally:
    (out / "HOLDING").unlink(missing_ok=True)
    time.sleep(0.3)                             # (g.safe(), the guard's stop)
    (out / "wg.json").write_text(json.dumps({"status_safe": {"run": False, "acc_on": False}}))
    log("bringup safe")
'''
SERVER = LOG + '''import signal, sys, time
def term(sig, frame):                           # (serve35.py's: a graceful stop)
    log("server term")
    time.sleep(0.3)
    log("server exit")
    sys.exit(0)
signal.signal(signal.SIGTERM, term)
if os.environ.get("STUB_CARDS"):
    log(f"server dev {os.environ.get('OTPU_DEV')}")
log("server up")
while True:
    time.sleep(0.05)
'''
# the stub host's two cards (BCU_ENV's variables, here in the environment): JTAG serial, PCI function, name
CARDS = {"BCU_CARD0_SERIAL": "SERIALCRD0", "BCU_CARD0_BDF": "0000:0a:00.0", "BCU_CARD0_NAME": "CRD0",
         "BCU_CARD1_SERIAL": "SERIALCRD1", "BCU_CARD1_BDF": "0000:0b:00.0", "BCU_CARD1_NAME": "CRD1"}
CHAT = LOG + '''import glob, signal, time
got = []
if "OTPU_LOCK_HELD" in os.environ:              # (an enclosing otpu-lock's, kept; any other OTPU_* not)
    log(f"chat lock {os.environ['OTPU_LOCK_HELD']} {os.environ.get('OTPU_LOCK_PID')} {'OTPU_STRAY' in os.environ}")
signal.signal(signal.SIGINT, lambda s, f: got.append(s))    # (the chat's: Ctrl-C stops a reply, the chat goes on)
if os.environ.get("STUB_CARDS"):                # (the card the chat opens: --dev, else the 35B wrappers' variables)
    import sys
    dev = sys.argv[sys.argv.index("--dev") + 1] if "--dev" in sys.argv else \
        os.environ.get("CHAT35_DEV") or os.environ.get("OTPU_DEV")
    log(f"chat dev {dev} target {os.environ.get('OTPU_TARGET')}")
log("chat up")
t = time.monotonic()
while not got and time.monotonic() - t < 60:
    time.sleep(0.05)
log("chat interrupted")
time.sleep(1.0)
held = glob.glob(os.path.join(os.environ["HOME"], "otpu-build/**/runs/**/HOLDING"), recursive=True)
log(f"chat held {bool(held)}")
log("chat exit")
'''
GOV = '''import os


class Gov:
    def __init__(self, t):
        pass

    def present(self):
        return not os.environ.get("STUB_NO_GOV")

    def status(self):
        return {"temp_c": float(os.environ.get("STUB_DIE_C", 50.0))}
'''


def _host(tmp_path: Path, wrapper: str) -> tuple[Path, dict]:
    """A stub card host under tmp_path: the wrapper's tree (T) and cardimg (B) under HOME, the venv's python, and a
    PATH whose ls shows the card's sysfs directory and whose pgrep finds no card job."""
    text = (CHAT35 / wrapper).read_text()

    def var(k, b=""):
        m = re.search(rf"^{k}=(\S+)$", text, re.M)
        return m and Path(m.group(1).replace("$BB", str(tmp_path / "otpu-build")).replace("$B", str(b)))
    b = var("B") or tmp_path / "otpu-build" / "cardimg"
    t = var("T", b)
    (t / "boards" / "bcu1525" / "scripts").mkdir(parents=True)
    (t / "boards" / "bcu1525" / "scripts" / "wg_bringup.py").write_text(BRINGUP)
    (t / "tools" / "x2_session").mkdir(parents=True)
    (t / "tools" / "x2_session" / "bringup.py").write_text(BRINGUP)      # (otpu-chat35x2's, a card each)
    (t / "opentpu" / "host").mkdir(parents=True)
    (t / "opentpu" / "__init__.py").write_text("")
    (t / "opentpu" / "host" / "__init__.py").write_text("")
    (t / "opentpu" / "host" / "board.py").write_text("class XdmaTransport:\n    def __init__(self, *a, **k):\n"
                                                     "        pass\n")
    (t / "opentpu" / "host" / "gov.py").write_text(GOV)
    (t / "opentpu" / "host" / "chat.py").write_text(CHAT)              # (otpu-chat's: python -m opentpu.host.chat)
    b.mkdir(parents=True, exist_ok=True)
    for c in ("serve35.py", "serve35x2.py"):
        (b / c).write_text(SERVER)
    for c in ("chat35.py", "chat35x2.py"):
        (b / c).write_text(CHAT)
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    bins = tmp_path / "bin"
    bins.mkdir()
    for p, text in ((venv / "python3", f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'),
                    # (card 0's function is xdma0, card 1's xdma1)
                    (bins / "ls", '#!/bin/sh\ncase "$1" in /sys/bus/pci/devices/0000:0b:00.0/xdma/) echo xdma1_user ;;\n'
                                  '/sys/bus/pci/devices/*/xdma/) echo xdma0_user ;;\n'
                                  '*) exec /bin/ls "$@" ;; esac\n'),
                    (bins / "pgrep", "#!/bin/sh\nexit 1\n"),
                    (bins / "tmux", "#!/bin/sh\nexit 0\n")):          # (the fans' loop: tmux has-session -t fan)
        p.write_text(text)
        p.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OTPU_", "CHAT35_", "SERVE35_"))}
    env.update(HOME=str(tmp_path), PATH=f"{bins}:{env['PATH']}", STUB_EVENTS=str(tmp_path / "events"),
               BCU_ENV=str(tmp_path / "none.env"), BCU_PY=str(venv / "python3"), **CARDS)
    return tmp_path / "events", env


def _run(tmp_path, wrapper, sig, ready, extra=None, args=(), also=()):
    """The wrapper as a foreground job (`extra` added to its environment, `args` its arguments); `sig` to its
    process group once `ready` is in the events. Its exit code, output and the events. also: the wrappers it
    hands over to, installed in the stub host's bin/ with their own trees."""
    ev, env = _host(tmp_path, wrapper)
    for w in also:
        _host2 = _host(tmp_path / f"h-{w}", w)      # (its tree and cardimg, under the same HOME layout)
        shutil.copytree(tmp_path / f"h-{w}" / "otpu-build", tmp_path / "otpu-build", dirs_exist_ok=True)
        (tmp_path / "otpu-build" / "bin").mkdir(exist_ok=True)
        shutil.copy(CHAT35 / w, tmp_path / "otpu-build" / "bin" / w)
    env.update(extra or {})
    out = tmp_path / "out.txt"
    args = list(args)
    with open(out, "w") as f:
        p = subprocess.Popen([BASH, str(CHAT35 / wrapper), *args], stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, env=env, start_new_session=True)
    try:
        t = time.monotonic()
        while not (ev.exists() and ready in ev.read_text()):
            assert p.poll() is None, out.read_text()
            assert time.monotonic() - t < 60, out.read_text()
            time.sleep(0.05)
        time.sleep(0.3)
        os.killpg(p.pid, sig)
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
    return rc, out.read_text(), ev.read_text().splitlines()


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP])
def test_serve35_stops_the_server_then_the_bringup(tmp_path, sig):
    rc, out, ev = _run(tmp_path, "otpu-serve35", sig, "server up")
    assert ev == ["bringup up, own session True", "server up", "server term", "server exit",
                  f"bringup stopped (signal {int(signal.SIGTERM)})", "bringup safe"], (ev, out)
    assert rc == 130 and "otpu-serve35: card back in SAFE" in out, out


CHATS = ["otpu-chat35", "otpu-chat"]


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("wrapper", CHATS)
def test_ctrl_c_reaches_the_chat_alone(tmp_path, wrapper):
    rc, out, ev = _run(tmp_path, wrapper, signal.SIGINT, "chat up")
    assert ev == ["bringup up, own session True", "chat up", "chat interrupted", "chat held True", "chat exit",
                  f"bringup stopped (signal {int(signal.SIGTERM)})", "bringup safe"], (ev, out)
    assert rc == 0 and ": card back in SAFE" in out, out


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("wrapper", CHATS)
def test_a_hangup_ends_the_chat_then_the_bringup(tmp_path, wrapper):
    rc, out, ev = _run(tmp_path, wrapper, signal.SIGHUP, "chat up")
    assert ev == ["bringup up, own session True", "chat up", f"bringup stopped (signal {int(signal.SIGTERM)})",
                  "bringup safe"], (ev, out)
    assert rc == 129 and ": card back in SAFE" in out, out


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGHUP])
def test_otpu_chat35x2_holds_both_cards_through_the_chat(tmp_path, sig):
    """otpu-chat35x2: both cards' bring-ups in sessions of their own, the chat once both hold; Ctrl-C reaches the chat
    alone (both still held after it), a hangup ends the chat; then both bring-ups stop on the wrapper's TERM and the
    wrapper says each card is back in SAFE."""
    rc, out, ev = _run(tmp_path, "otpu-chat35x2", sig, "chat up")
    up = int(signal.SIGTERM)
    assert ev[:3] == ["bringup up, own session True"] * 2 + ["chat up"], (ev, out)
    chat = ["chat interrupted", "chat held True", "chat exit"] if sig == signal.SIGINT else []
    assert ev[3:3 + len(chat)] == chat, (ev, out)
    tail = ev[3 + len(chat):]
    assert sorted(tail) == sorted([f"bringup stopped (signal {up})"] * 2 + ["bringup safe"] * 2), (ev, out)
    assert rc == (0 if sig == signal.SIGINT else 129), out
    assert "CRD0 back in SAFE" in out and "CRD1 back in SAFE" in out, out


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
def test_otpu_chat35x2_keeps_an_enclosing_otpu_lock(tmp_path):
    """otpu-chat35x2 inside a card window's `otpu-lock --dev /dev/xdma1 -- ...`: the chat sees the hold's
    OTPU_LOCK_HELD and OTPU_LOCK_PID (runtime variables: the chat borrows that card's lock), no other OTPU_* of the
    caller's (the image key's environment stays the entry's)."""
    rc, out, ev = _run(tmp_path, "otpu-chat35x2", signal.SIGINT, "chat up",
                       {"OTPU_LOCK_HELD": "/dev/xdma1", "OTPU_LOCK_PID": "4242", "OTPU_STRAY": "1"})
    assert "chat lock /dev/xdma1 4242 False" in ev, (ev, out)
    assert rc == 0, out


# ---------------------------------------------------------------------------- which card
# the wrapper, its arguments and environment -> the cards brought up (serial, device) and the device the chat or
# the server opens, with its target
WHICH = [
    ("otpu-chat", [], {}, [("SERIALCRD0", "/dev/xdma0")], "chat dev /dev/xdma0 target bcu1525-wg", ()),
    ("otpu-chat", ["--card", "1"], {}, [("SERIALCRD1", "/dev/xdma1")], "chat dev /dev/xdma1 target bcu1525-wg", ()),
    ("otpu-chat35", [], {}, [("SERIALCRD0", "/dev/xdma0")], "chat dev /dev/xdma0 target bcu1525-wg:bhw16", ()),
    ("otpu-chat35", [], {"BCU_CARD": "1"}, [("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma1 target bcu1525-wg:bhw16", ()),
    ("otpu-serve35", [], {"BCU_CARD": "1"}, [("SERIALCRD1", "/dev/xdma1")], "server dev /dev/xdma1", ()),
    ("otpu-chat35x2", [], {}, [("SERIALCRD0", "/dev/xdma0"), ("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma0+/dev/xdma1 target bcu1525-x2:st16", ()),
    # otpu-chat on the two-card image (BCU_IMAGE, or the marker beside the two-card wrappers): both cards
    ("otpu-chat", [], {"BCU_IMAGE": "x2"}, [("SERIALCRD0", "/dev/xdma0"), ("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma0+/dev/xdma1 target bcu1525-x2:st16", ()),
    ("otpu-chat", ["--prompt", "Hi", "--card", "x2"], {"BCU_IMAGE": "x2", "BCU_X2_TARGET": "bcu1525-x2"},
     [("SERIALCRD0", "/dev/xdma0"), ("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma0+/dev/xdma1 target bcu1525-x2", ()),
    # otpu-chat's hand-over of the 35B: one card (--card N), or both
    ("otpu-chat", ["--model", "qwen36-35b", "--card", "1"], {}, [("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma1 target bcu1525-wg:bhw16", ("otpu-chat35",)),
    ("otpu-chat", ["--model", "qwen36-35b", "--card", "x2"], {},
     [("SERIALCRD0", "/dev/xdma0"), ("SERIALCRD1", "/dev/xdma1")],
     "chat dev /dev/xdma0+/dev/xdma1 target bcu1525-x2:st16", ("otpu-chat35x2",)),
]


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("wrapper,args,extra,cards,opened,also", WHICH,
                         ids=[f"{w}{'-' + '-'.join(a) if a else ''}{'-' + '-'.join(e.values()) if e else ''}"
                              for w, a, e, *_ in WHICH])
def test_each_front_end_opens_the_card_it_was_asked_for(tmp_path, wrapper, args, extra, cards, opened, also):
    """Card 0, card 1 or both: the bring-up gets that card's JTAG serial and the XDMA device at its PCI function,
    and the chat (or the server) opens the same device, under that image's target."""
    ready = "server up" if "serve" in wrapper else "chat up"
    rc, out, ev = _run(tmp_path, wrapper, signal.SIGINT, ready, {"STUB_CARDS": "1", **extra}, args, also)
    got = sorted(tuple(e.split()[2:4]) for e in ev if e.startswith("bringup card "))
    assert got == sorted(cards), (ev, out)
    assert opened in ev, (ev, out)


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_bcu_dev_must_be_the_cards_own_device(tmp_path):
    """BCU_DEV naming another card's device is refused before anything is brought up."""
    ev, env = _host(tmp_path, "otpu-chat35")
    env.update(BCU_CARD="1", BCU_DEV="/dev/xdma0")
    p = subprocess.run([BASH, str(CHAT35 / "otpu-chat35")], stdin=subprocess.DEVNULL, capture_output=True,
                       text=True, env=env, timeout=60)
    assert p.returncode == 1 and "CRD1 is /dev/xdma1, not BCU_DEV=/dev/xdma0" in p.stdout, p.stdout + p.stderr
    assert not ev.exists()


# ---------------------------------------------------------------------------- the image, RESUME, BCU_ENV
@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("how", ["BCU_IMAGE", "marker", "registers"])
@pytest.mark.parametrize("card", ["0", "1"])
def test_otpu_chat_refuses_one_card_of_the_two_card_image_at_once(tmp_path, how, card):
    """--card 0 or 1 while the cards hold the two-card image (BCU_IMAGE=x2, the marker beside the two-card wrappers,
    or the card's own registers when they answer): exit 2 with what to do, nothing brought up."""
    ev, env = _host(tmp_path, "otpu-chat")
    tree = tmp_path / "otpu-build" / "bcu-wg" / "tree"
    if how == "BCU_IMAGE":
        env["BCU_IMAGE"] = "x2"
    elif how == "marker":
        (tmp_path / "otpu-build" / "cardimg" / "x2").mkdir(parents=True)
        (tmp_path / "otpu-build" / "cardimg" / "x2" / "IMAGE").write_text("x2\n")
    else:                                       # (the card's own answer, opentpu.host.cardimage's: over BCU_IMAGE)
        env["BCU_IMAGE"] = "one"
        (tree / "opentpu" / "host" / "cardimage.py").write_text(
            'print("x2  bcu1525-x2:st16  S=2 of 2 cards  (the governor)  build 1a2b3c4d  CFG_HASH e331f79c")\n')
    p = subprocess.run([BASH, str(CHAT35 / "otpu-chat"), "--prompt", "Hi", "--card", card],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 2, p.stdout + p.stderr
    assert f"card {card} (CRD{card}) holds the two-card image" in p.stdout and "without --card" in p.stdout, p.stdout
    assert not ev.exists()
    said = f"BCU_IMAGE says one, but CRD{card} reports another image: the card's answer is used" in p.stdout
    assert said == (how == "registers"), p.stdout
    assert (f"CRD{card} holds bcu1525-x2:st16  S=2 of 2 cards  (the governor)" in p.stdout) == (how == "registers")


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_otpu_chat_card_x2_needs_the_two_card_image(tmp_path):
    ev, env = _host(tmp_path, "otpu-chat")
    p = subprocess.run([BASH, str(CHAT35 / "otpu-chat"), "--card", "x2"], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, env=env, timeout=60)
    assert p.returncode == 2 and "--card x2 needs the two-card image" in p.stdout, p.stdout + p.stderr
    assert not ev.exists()


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
def test_otpu_chat_on_the_two_card_image_holds_both_cards(tmp_path):
    """otpu-chat on the two-card image: both bring-ups, the chat on both devices, Ctrl-C to the chat alone, then both
    cards back in SAFE."""
    rc, out, ev = _run(tmp_path, "otpu-chat", signal.SIGINT, "chat up", {"BCU_IMAGE": "x2"})
    assert ev[:3] == ["bringup up, own session True"] * 2 + ["chat up"], (ev, out)
    assert ev[3:6] == ["chat interrupted", "chat held True", "chat exit"], (ev, out)
    assert sorted(ev[6:]) == sorted([f"bringup stopped (signal {int(signal.SIGTERM)})"] * 2
                                    + ["bringup safe"] * 2), (ev, out)
    assert rc == 0 and "CRD0 back in SAFE" in out and "CRD1 back in SAFE" in out, out


ALL = [("otpu-chat", {}, 1), ("otpu-chat", {"BCU_IMAGE": "x2"}, 2), ("otpu-chat35", {}, 1), ("otpu-serve35", {}, 1),
       ("otpu-chat35x2", {}, 2), ("otpu-serve35x2", {}, 2)]


@pytest.mark.skipif(BASH is None or not hasattr(os, "setsid"), reason="needs bash and setsid")
@pytest.mark.parametrize("wrapper,extra,n", ALL, ids=[w + "".join(e.values()) for w, e, _ in ALL])
@pytest.mark.parametrize("resume", [None, "69.9"])
def test_bcu_resume_goes_to_every_bringup(tmp_path, wrapper, extra, n, resume):
    """BCU_RESUME is each bring-up's --resume (none without it: wg_bringup's own); a die above the RESUME the
    bring-up will set is said before the bring-up, and its "waiting for GO" line shown while the wrapper waits."""
    ready = "server up" if "serve" in wrapper else "chat up"
    env = {"STUB_RESUME": "1", "STUB_DIE_C": "68.0", "STUB_WAIT": "2.5", "BCU_WAIT_SAY": "1", **extra}
    if resume:
        env["BCU_RESUME"] = resume
    rc, out, ev = _run(tmp_path, wrapper, signal.SIGINT, ready, env)
    assert [e for e in ev if e.startswith("bringup resume")] == [f"bringup resume {resume}"] * n, (ev, out)
    assert ("is above RESUME (65 C)" in out) == (resume is None), out
    assert "BCU_RESUME (at most 69.9)" in out or resume, out
    assert "waiting for GO: die 70.1 C, RESUME 65.0 C" in out, out


@pytest.mark.skipif(BASH is None, reason="needs bash")
@pytest.mark.parametrize("wrapper", ["otpu-chat", "otpu-chat35", "otpu-chat35x2", "otpu-serve35", "otpu-serve35x2"])
def test_the_environment_wins_over_bcu_env(tmp_path, wrapper):
    """A BCU_* variable in the environment keeps its value over the file's (BCU_BUILD for one run); what the
    environment does not set comes from the file. Here the file's BCU_BUILD has no tree: the wrapper finds the
    environment's (as far as its governor check, the card named by the file), and without the environment's it stops at the file's."""
    ev, env = _host(tmp_path, wrapper)
    f = tmp_path / "bcu.env"
    f.write_text(f"BCU_BUILD={tmp_path}/nowhere\nBCU_CARD0_NAME=FILE0\n")
    env.update(BCU_ENV=str(f), BCU_BUILD=str(tmp_path / "otpu-build"))
    env.pop("BCU_CARD0_NAME")
    p = subprocess.run([BASH, str(CHAT35 / wrapper)], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       env={**env, "STUB_NO_GOV": "1"}, timeout=60)
    assert "no governor on FILE0" in p.stdout + p.stderr, p.stdout + p.stderr
    env.pop("BCU_BUILD")
    p = subprocess.run([BASH, str(CHAT35 / wrapper)], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                       env={**env, "STUB_NO_GOV": "1"}, timeout=60)
    assert p.returncode == 2 and "no governor" not in p.stdout + p.stderr, p.stdout + p.stderr
