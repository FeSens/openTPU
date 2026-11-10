"""The image a BCU1525 holds: its target, slices, CFG_HASH and BUILD_ID, in RUN and in SAFE.

A card in RUN answers for itself: the accelerator's configuration registers (CFG_*, BUILD_ID,
CORE_KHZ). In SAFE they are fenced (otpu_axil_off: every read 0xDEAD0FF0), and only the governor
answers. held(t, dev) asks, in order, and says where its answer came from ("source"):

  "registers"  the accelerator's own registers (a card in RUN)
  "governor"   the governor's identity words (gov.Gov.image: a build with otpu_bcu_gov's IMAGE
               registers, which answer in SAFE too)
  "record"     what a host read from the accelerator's registers at this card's last bring-up
               (note(): a JSON file a card under state_dir()), for the builds before the
               governor's words
  None         nothing: a card no host has brought up since the boot, of a build without them

The record is kept only while it can still be this card's image. A JTAG-loaded image stays
through warm reboots and goes with the power (boards/bcu1525/README.md), and a new image is
loaded over JTAG, after which the host needs a warm reboot for its PCIe link; so a record counts
while the kernel's boot id (/proc/sys/kernel/random/boot_id) is the one it was written under and
the card still has a governor. One case passes both and is wrong: the FPGA reconfigured from its
flash without a reboot (the guard's JPROGRAM way out) and its PCIe function found again by a
rescan; the next bring-up's read replaces the record.

    python3 -m opentpu.host.cardimage --dev /dev/xdma0 [--json]

prints the answer's one line (line()), first the word a wrapper asks for: `x2` (a card of a
machine over several cards), `one`, or `unknown`.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

BOOT_ID = "/proc/sys/kernel/random/boot_id"
FIELDS = ("target", "platform", "slices", "cards", "cfg_hash", "build_id", "core_khz")


def state_dir() -> Path:
    """Where the records are: $BCU_BUILD/state (BCU_BUILD default ~/otpu-build)."""
    return Path(os.environ.get("BCU_BUILD") or Path.home() / "otpu-build") / "state"


def card_key(dev: str) -> str:
    """A card's name in the records: its PCI function (the XDMA device's sysfs parent; a device
    node's number follows the probe order), else the device's own name."""
    name = Path(dev).name
    p = Path(f"/sys/class/xdma/{name}_user/device")
    try:
        return p.resolve().name if p.exists() else name
    except OSError:
        return name


def boot_id() -> str | None:
    try:
        return Path(BOOT_ID).read_text().strip() or None
    except OSError:
        return None


def target_name(platform: int, slices: int, cfg_hash: int) -> tuple[str | None, int]:
    """(the name of the target whose build reports this platform and CFG_HASH, its cards): the
    platform's target or the variant of it whose build hashes so (a variant shares the platform
    id); the bare target's name with "?" when none does (a build with other knobs, or of another
    tree), None for a platform this tree has not. Cards 1 then."""
    from dataclasses import replace
    from opentpu.target import arch_hash, build_config, by_platform, cards, full_name, load, variants
    base = by_platform(platform)
    if base is None:
        return None, 1
    n = cards(base)
    for name in (base.name, *(f"{base.name}:{v}" for v in variants(base))):
        try:
            t = load(name)
            # one card of a machine over several is built at its own slices (its CFG_HASH is that
            # build's: bcu1525-x2's cards hash as two slices, not the machine's four)
            own = t if t.arch["S"] == slices else replace(t, arch=dict(t.arch, S=slices))
            if cfg_hash in (arch_hash(build_config(t)), arch_hash(build_config(own))):
                return full_name(t), cards(t)
        except Exception:               # (a variant this tree cannot build: not this image's)
            continue
    return f"{base.name}:?", n


def from_info(info: dict) -> dict | None:
    """The identity in Board.info() (a card in RUN), or None before register map 4."""
    c = info.get("cfg")
    if not c:
        return None
    name, n = target_name(c["platform"], c["S"], c["hash"])
    return {"target": name, "platform": c["platform"], "slices": c["S"], "cards": n,
            "cfg_hash": c["hash"], "build_id": info.get("build_id"),
            "core_khz": info.get("core_khz")}


def from_gov(g: dict) -> dict | None:
    """The identity in gov.Gov.image()'s words, or None (no words, or a top without an
    accelerator: cards 0)."""
    if not g or not g["cards"]:
        return None
    name, _ = target_name(g["platform"], g["slices"], g["cfg_hash"])
    return {"target": name, "platform": g["platform"], "slices": g["slices"],
            "cards": g["cards"], "cfg_hash": g["cfg_hash"], "build_id": g["build_id"],
            "core_khz": g["core_khz"]}


def live(t) -> dict | None:
    """The accelerator's own answer, or None while it is fenced or is not openTPU's."""
    from . import regs as R
    from .board import Board
    if t.reg_read(R.R_ID) != R.ID_OTPU:
        return None
    return from_info(Board(t, check=False, lock=False).info())


def _file(dev: str, d: Path | None = None) -> Path:
    return (d or state_dir()) / f"image-{card_key(dev)}.json"


def note(t, dev: str, d: Path | None = None) -> dict | None:
    """Record the running card's identity (live) for the times it is in SAFE; returns it, or
    None when the card does not answer or the record cannot be written (no error: a record is a
    convenience)."""
    img = live(t)
    if img is None:
        return None
    rec = {**img, "when": time.time(), "boot_id": boot_id(), "dev": dev}
    f = _file(dev, d)
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f".{f.name}.{os.getpid()}")
        tmp.write_text(json.dumps(rec, indent=1))
        os.replace(tmp, f)
    except OSError:
        return None
    return rec


def forget(dev: str, d: Path | None = None) -> None:
    """Drop a card's record (its image is about to change)."""
    try:
        _file(dev, d).unlink()
    except OSError:
        pass


def record(dev: str, d: Path | None = None) -> dict | None:
    """A card's record while it still counts: written under this boot (a host that cannot tell
    its boot keeps none)."""
    try:
        rec = json.loads(_file(dev, d).read_text())
    except (OSError, ValueError):
        return None
    b = boot_id()
    if not b or rec.get("boot_id") != b or any(k not in rec for k in FIELDS):
        return None
    return rec


def held(t, dev: str, d: Path | None = None) -> dict:
    """The image the card at `dev` holds (the module's docstring): FIELDS and "source"
    ("registers", "governor", "record" with the record's "when"), or {"source": None}. `t`: a
    transport's register reads (no DMA, nothing written)."""
    from . import gov as G
    img = live(t)
    if img is not None:
        return {**img, "source": "registers"}
    g = G.Gov(t)
    if not g.present():
        return {"source": None}
    img = from_gov(g.image())
    if img is not None:
        return {**img, "source": "governor"}
    rec = record(dev, d)
    if rec is not None:
        return {**{k: rec[k] for k in FIELDS}, "when": rec.get("when"), "source": "record"}
    return {"source": None}


def kind(img: dict) -> str:
    """`x2` (a card of a machine over several cards), `one`, or `unknown`."""
    if img.get("source") is None:
        return "unknown"
    return "x2" if (img.get("cards") or 1) > 1 else "one"


def source_text(img: dict) -> str:
    s = img.get("source")
    if s == "registers":
        return "the card's registers"
    if s == "governor":
        return "the governor"
    if s == "record":
        w = img.get("when")
        return "as read at the last bring-up" + (time.strftime(" %H:%M", time.localtime(w))
                                                 if w else "")
    return "unknown until a bring-up"


def texts(img: dict) -> tuple[str, str]:
    """An answer in two short lines (otpu-smi's Image rows): 'bcu1525-x2:st16  S=2 of 2 cards
    (the governor)' and 'build 1a2b3c4d  CFG_HASH e331f79c  150 MHz'; the second empty for no
    answer."""
    if img.get("source") is None:
        return source_text(img), ""
    b, k = img.get("build_id"), img.get("core_khz")
    first = [img.get("target") or f"platform {img['platform']}",
             f"S={img['slices']}" + (f" of {img['cards']} cards" if img["cards"] > 1 else ""),
             f"({source_text(img)})"]
    second = ["build " + ("n/a" if b is None else f"{b:08x}"), f"CFG_HASH {img['cfg_hash']:08x}"]
    if k:
        second.append(f"{k / 1e3:g} MHz")
    return "  ".join(first), "  ".join(second)


def text(img: dict) -> str:
    """texts() as one line."""
    return "  ".join(x for x in texts(img) if x)


def main(argv=None) -> int:
    import argparse
    from .board import XdmaTransport
    ap = argparse.ArgumentParser(prog="python3 -m opentpu.host.cardimage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", default=os.environ.get("OTPU_DEV", "/dev/xdma0"))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    t = XdmaTransport(a.dev, dma=False)
    try:
        img = held(t, a.dev)
    finally:
        close = getattr(t, "close", None)
        if close is not None:
            close()
    if a.json:
        print(json.dumps({**img, "kind": kind(img)}))
    else:
        print(f"{kind(img)}  {text(img)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
