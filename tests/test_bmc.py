"""The BCU1525's BMC bridge: the vendor protocol's framing, checksum and PEC (opentpu/host/bmc.py),
its read-only allowlist, sessions against the model BMC (opentpu/host/fake_bmc.py: replies in
pieces, a wrong PEC, a lost reply, a parity error, an unknown firmware, the lock), otpu-smi's
VCCINT power and otpu-bmc; and otpu_bmc_uart's own testbench (sim/verilator/tb_bmc_uart.sv).
"""
import fcntl
import json

import pytest

from opentpu.host import bmc as B
from opentpu.host import i2c as I
from opentpu.host import regs as R
from opentpu.host.fake import FakeTransport
from opentpu.host.fake_bmc import FakeBmc, FakeLtc3884


@pytest.fixture(autouse=True)
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("OTPU_RUN_DIR", str(tmp_path / "otpu"))
    B._known.clear()                    # (the per-process firmware and VOUT_MODE cache)
    return tmp_path / "otpu"


def card(devname="fakebmc", **kw):
    return FakeTransport(devname=devname, regmap=R.CAPS2_SINCE, bmc=FakeBmc(**kw), ddr_mts=1066)


def frames_sent(fb: FakeBmc) -> list[bytes]:
    return B.FrameDecoder().feed(bytes(fb.sent))


# ------------------------------------------------------------------------------ the protocol
def test_framing_checksum_and_pec():
    assert B.crc8(b"123456789") == 0xF4                     # CRC-8/SMBUS's check value
    inner = B.encode_inner(0x0A, bytes((0x5B, 0x84, 3, 6, 2, 0, 0x5C)))
    assert inner[:3] == bytes((0x0A, 0, 7)) and B.decode_inner(inner, 0x0A)[-1] == 0x5C
    f = B.wrap_frame(inner)
    assert f.startswith(b"\x5c\x02") and f.endswith(b"\x5c\x03") and b"\x5c\x5c" in f
    d = B.FrameDecoder()                                    # in pieces, after line garbage
    stream = b"\x00\x13" + f + B.wrap_frame(B.encode_inner(0x8C, b"\x04\x00\x01\x00"))
    got = []
    for i in range(0, len(stream), 3):
        got += d.feed(stream[i:i + 3])
    assert got == [inner, B.encode_inner(0x8C, b"\x04\x00\x01\x00")]
    assert d.feed(B.SYNC) == []                             # the sync ends no frame of its own
    with pytest.raises(B.ProtocolError, match="escape"):
        B.FrameDecoder().feed(b"\x5c\x02\x01\x5c\x07")
    with pytest.raises(B.ProtocolError, match="checksum"):
        B.decode_inner(inner[:-1] + bytes((inner[-1] ^ 1,)), 0x0A)
    with pytest.raises(B.ProtocolError, match="expected 0x8a"):
        B.decode_inner(inner, 0x8A)
    with pytest.raises(B.ProtocolError, match="length"):
        B.decode_inner(inner[:-3] + inner[-2:], 0x0A)


def test_reads_only():
    """The allowlist names reads alone: no PMBus send-byte (CLEAR_FAULTS, STORE_*), no PAGE,
    WRITE_PROTECT or LTC388x action (MFR_RESET, MFR_CLEAR_PEAKS, EEPROM, fault log); a page
    read only through PAGE_PLUS_READ of a listed code, pages 0 and 1."""
    actions = I.SEND_BYTE | {0x00, 0x10, 0xBD, 0xBE, 0xBF, 0xE3, 0xEA, 0xEC, 0xF0, 0xFD}
    assert not B.READS & actions
    assert B.read_only(b"\x88") and B.read_only(bytes((6, 2, 1, 0x8B)))
    for w in (b"\x03", b"\x00", bytes((0x21, 0x00, 0x04)), bytes((6, 2, 2, 0x8B)),
              bytes((6, 3, 0, 0x8B)), bytes((6, 2, 0, 0x03)), b"", bytes((0x88, 0x88))):
        assert not B.read_only(w), w.hex()
    t = card()
    with B.Bmc(t) as b:
        with pytest.raises(ValueError, match="not a read"):
            b.exchange(bytes((0x21, 0x00, 0x04)), 1)            # VOUT_COMMAND's write
        with pytest.raises(ValueError, match="not a read"):
            b.exchange(b"\x88", 0)
    assert not [f for f in frames_sent(t.bmc) if f[0] == B.CMD_I2C]


# ------------------------------------------------------------------------------ sessions
def test_measured_on_the_model():
    t = card()
    m = B.measured(t)
    v = m["vccint"]
    assert m["w"] == pytest.approx(24.8, abs=0.02) and m["how"] == "PIN"
    assert m["label"] == "VCCINT in" and m["firmware"] == 0x10004 and m["board_temp_c"] is None
    assert v["vin"] == pytest.approx(12.1, abs=0.01) and v["die_c"] == pytest.approx(52)
    assert [p["vout"] for p in v["pages"]] == [pytest.approx(0.85, abs=1e-3)] * 2
    assert sum(p["iout"] for p in v["pages"]) == pytest.approx(27.0, abs=0.02)
    assert [p["temp_c"] for p in v["pages"]] == [45, 47]
    fb = t.bmc
    # the session: the sync first, the firmware's version before any I2C request, every request
    # a read of the allowlist, nothing refused
    assert bytes(fb.sent[:3]) == B.SYNC
    cmds = [c for c, _ in fb.requests]
    assert cmds[0] == B.CMD_VERSION and set(cmds[1:]) == {B.CMD_I2C}
    assert not fb.refused and all(B.read_only(w) for w in fb.ltc.reads)
    assert len(fb.ltc.reads) == len(B.TELEMETRY_GLOBAL) + 2 * (len(B.TELEMETRY_PAGED) + 1)
    # a second session in the process: firmware and VOUT_MODE known, the readings alone
    n = len(fb.requests)
    B.measured(t)
    assert len(fb.requests) - n == len(B.TELEMETRY_GLOBAL) + 2 * len(B.TELEMETRY_PAGED)


def test_retries_and_failures():
    t = card()
    fb = t.bmc
    with B.Bmc(t, timeout=0.05) as b:
        fb.corrupt = 1                                      # a wrong PEC: read again
        assert b.value("vin") == pytest.approx(12.1, abs=0.01)
        fb.silent = 1                                       # a lost reply: timeout, again
        assert b.value("iin") == pytest.approx(2.05, abs=0.01)
        fb.noise = True                                     # a parity error on the line
        assert b.value("pin") == pytest.approx(24.8, abs=0.02)
        fb.corrupt = 10
        with pytest.raises(B.ProtocolError, match="PEC"):
            b.value("vin")
        fb.corrupt = 0
        fb.silent = 10
        with pytest.raises(TimeoutError):
            b.value("vin")
    # a firmware the protocol is not known for: no I2C request at all
    t = card(devname="fakebmc2", firmware=0x20001)
    with pytest.raises(B.BmcError, match="0x00020001"):
        B.measured(t)
    assert [c for c, _ in t.bmc.requests] == [B.CMD_VERSION]
    # a controller other than an LTC3884: info stops at the identification
    ltc = FakeLtc3884()
    t = card(devname="fakebmc3", ltc=ltc)
    ltc._direct = lambda cmd, page, f=ltc._direct: (b"\x00\x99" if cmd == B.MFR_SPECIAL_ID
                                                    else f(cmd, page))
    with B.Bmc(t) as b:
        d = B.info(b)
    assert not d["ltc3884"] and "global" not in d


def test_watch_lends_the_line():
    """A bridge with the watch ("BMC2"): a session holds the line from its sync to its end, a
    byte written without it never reaches the BMC, the watch taking it back mid-request (HOLD_MS)
    costs a retry; a bridge without ("BMC1") needs no hold."""
    t = card()
    fb = t.bmc
    B.Uart(t).wr(B.TX, 0x41)
    assert fb.sent == bytearray() and fb.unheld == bytearray(b"A")
    with B.Bmc(t) as b:
        assert b.watch and fb.asked and fb.holds == 1
        assert fb.unheld == bytearray()                     # the sync's TX reset
        fb.revoke_after = fb.held_bytes + 5                 # the line taken back mid-request
        assert b.value("vin") == pytest.approx(12.1, abs=0.01)
        assert fb.holds == 2 and fb.asked
    assert not fb.asked                                     # given back at the session's end
    assert not fb.refused and all(B.read_only(w) for w in fb.ltc.reads)
    t1 = card(devname="fakebmc1", watch=False)
    m = B.measured(t1)
    assert m["w"] == pytest.approx(24.8, abs=0.02) and t1.bmc.holds == 0


def test_watch_readings():
    """watch(): the fabric's readings from its registers alone (no byte on the line), decoded;
    none before the first reading or with the firmware unknown; None without the watch."""
    t = card()
    w = B.watch(t)
    assert (w["p0_c"], w["p1_c"], w["ctl_c"]) == (45.0, 47.0, 52.0)
    assert w["pin"] == pytest.approx(24.8, abs=0.02) and w["vin"] == pytest.approx(12.1, abs=0.01)
    assert w["hot_c"] == 52.0 and w["hot_age_ms"] == 3 and set(w["ages_ms"].values()) == {3}
    assert w["firmware"] == 0x10004 and not w["stopped"] and w["ok"] == 12
    assert t.bmc.sent == bytearray() and not t.bmc.requests and not t.bmc.asked
    t.bmc.ltc.pages[1]["temp_c"] = -3.25                    # a page below zero, the die hottest
    assert B.watch(t)["p1_c"] == -3.25 and B.watch(t)["hot_c"] == 52.0
    t.bmc.watch_age = B.AGE_NONE
    w = B.watch(t)
    assert w["p0_c"] is None and w["hot_c"] is None and w["hot_age_ms"] == B.AGE_NONE
    t2 = card(devname="fakebmc7", firmware=0x20001)
    w = B.watch(t2)
    assert w["stopped"] and w["ctl_c"] is None and w["firmware"] == 0x20001
    assert B.watch(card(devname="fakebmc8", watch=False)) is None


def test_lock(run_dir):
    t = card()
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "fakebmc.bmc.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        assert B.measured(t, lock_timeout=0.02) is None
    assert t.bmc.sent == bytearray()                        # nothing sent while it was held
    assert B.measured(t)["w"] is not None


def test_info_reads_every_table_entry():
    t = card()
    with B.Bmc(t) as b:
        d = B.info(b)
    assert d["ltc3884"] and (d["mfr_id"], d["mfr_model"]) == ("LTC", "LTC3884")
    assert set(d["global"]) == set(B.GLOBAL) and d["pin_accuracy_pct"] == 5.0
    assert [set(p) - {"page"} for p in d["pages"]] == [set(B.PAGED)] * 2
    assert d["errors"] == {} and not t.bmc.refused
    assert d["pages"][0]["ot_fault_c"] == 120 and d["pages"][1]["vout_command"] == \
        pytest.approx(0.85, abs=1e-3)


# ------------------------------------------------------------------------------ the tools
def test_smi_vccint(tmp_path, capsys):
    """A bridge without the watch ("BMC1"): otpu-smi's VCCINT through a session. With the watch
    ("BMC2") its registers alone: no byte on the line (test_smi_bcu has the board's rows)."""
    from opentpu.host import smi
    t2 = card(devname="fakebmc2")
    none = str(tmp_path / "none.json")
    assert smi.main(["--dev", "/dev/fakebmc2", "--power-json", none, "-i", "0"],
                    open_transport=lambda dev: t2) == 0
    tab = capsys.readouterr().out
    assert "Power 24.8W VCCINT regulator input" in tab and "W out" not in tab
    assert t2.bmc.sent == bytearray() and not t2.bmc.holds
    t = card(watch=False)
    args = ["--dev", "/dev/fakebmc", "--power-json", none, "-i", "0"]
    assert smi.main(args, open_transport=lambda dev: t) == 0
    tab = capsys.readouterr().out
    assert "Power 24.8W VCCINT in" in tab and "bmc" in tab
    assert "0.850 V  27.0 A  23.0 W out" in tab and "VR 47 °C  ctrl 52 °C" in tab
    assert all(len(r) == smi.W for r in tab.splitlines()[1:] if r.startswith("│"))
    smi.main(["--json"] + args, open_transport=lambda dev: t)
    (d,) = json.loads(capsys.readouterr().out)
    assert d["caps2"]["bmc"] and d["measured"]["vccint"]["pages"][1]["temp_c"] == 47
    csvp = tmp_path / "p.csv"
    smi.main(args + ["--csv", str(csvp)], open_transport=lambda dev: t)
    rows = list(__import__("csv").DictReader(csvp.read_text().splitlines()))
    assert float(rows[0]["power_w"]) == pytest.approx(24.8, abs=0.02)
    assert rows[0]["power_how"] == "VCCINT in"
    capsys.readouterr()
    n = len(t.bmc.sent)
    smi.main(args + ["--no-bmc"], open_transport=lambda dev: t)
    tab = capsys.readouterr().out
    assert "Power n/a" in tab and "VCCINT " not in tab and len(t.bmc.sent) == n
    # a bitstream without the bridge (register map 4: no CAPS2) is never sent a byte
    t4 = FakeTransport(devname="fakebmc4", regmap=4, bmc=FakeBmc())
    smi.main(["--dev", "/dev/fakebmc4", "--power-json", none, "-i", "0"],
             open_transport=lambda dev: t4)
    assert "Power n/a" in capsys.readouterr().out and t4.bmc.sent == bytearray()
    # a BMC that does not answer: the estimate (n/a here), no error
    tq = card(devname="fakebmc5", watch=False)
    tq.bmc.silent = 100
    smi.main(["--dev", "/dev/fakebmc5", "--power-json", none, "-i", "0"],
             open_transport=lambda dev: tq)
    assert "Power n/a" in capsys.readouterr().out


def test_otpu_bmc(capsys):
    t = card()
    assert B.main(["--dev", "/dev/fakebmc"], open_transport=lambda dev: t) == 0
    out = capsys.readouterr().out
    assert "firmware 0x00010004" in out and "page 1" in out and "24.8 W" in out
    assert B.main(["--dev", "/dev/fakebmc", "--all", "--json"], open_transport=lambda dev: t) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["mfr_model"] == "LTC3884" and d["vccint"]["pin"] == pytest.approx(24.8, abs=0.02)
    assert B.main(["--fake", "--all"]) == 0 and "pin_accuracy" in capsys.readouterr().out
    t5 = FakeTransport(devname="fakebmc6", regmap=5)
    assert B.main(["--dev", "/dev/fakebmc6"], open_transport=lambda dev: t5) == 1
    assert "no BMC bridge" in capsys.readouterr().err


# ------------------------------------------------------------------------------ the RTL
def _bmc_rtl(tb: str):
    from opentpu import rtlsim
    rtl = rtlsim.RTL / "boards/bcu1525"
    return rtlsim.build(tb, [rtl / "otpu_bmc_watch.sv", rtl / "otpu_bmc_uart.sv",
                             rtlsim.TB / f"{tb}.sv"])


def test_bmc_uart_rtl(have_verilator):
    """otpu_bmc_uart's testbench (WATCH 0): registers, both FIFOs through a loopback, the frames
    on the line (8E1), the error flags, the FIFOs' limits and resets."""
    import subprocess
    r = subprocess.run([str(_bmc_rtl("tb_bmc_uart"))], capture_output=True, text=True, timeout=120)
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout[-2000:]


def _i2c_request(w: bytes, n: int) -> bytes:
    return B.wrap_frame(B.encode_inner(B.CMD_I2C, bytes((B.ADDR, len(w) | 0x80, n)) + w))


def _read_reply(w: bytes, word: int, pec_xor: int = 0, sum_xor: int = 0) -> bytes:
    """The BMC's reply to a read of `w`: status 1, the block's count for PAGE_PLUS_READ, the
    word, the PEC (fake_bmc's)."""
    data = (b"\x02" if w[0] == B.PAGE_PLUS_READ else b"") + word.to_bytes(2, "little")
    pec = B.crc8(bytes((B.ADDR << 1,)) + w + bytes((B.ADDR << 1 | 1,)) + data) ^ pec_xor
    inner = B.encode_inner(0x8A, b"\x01" + data + bytes((pec,)))
    return B.wrap_frame(inner[:-1] + bytes((inner[-1] ^ sum_xor,)))


def watch_script():
    """tb_bmc_watch's BMC (its header has the words): the requests the watch must send, in
    bmc.py's bytes, and the replies, and the expected words (expv)."""
    from opentpu.host.i2c import to_linear11
    reads = [(bytes((B.PAGE_PLUS_READ, 2, p, 0x8D)), 4) for p in B.PAGES] + \
        [(bytes((c,)), 3) for c in (0x8E, 0x97, 0x88)]               # watch's readings 0..4
    version = B.wrap_frame(B.encode_inner(B.CMD_VERSION, b""))
    words = [to_linear11(x) for x in (45.0, 47.0, 52.0, 24.8, 12.1)]
    vin2, hot = to_linear11(12.0), to_linear11(75.5)               # 75.5 C: 0xEA5C, a 5C to double
    assert b"\x5c\x5c" in _read_reply(reads[1][0], hot)
    ver_reply = B.wrap_frame(B.encode_inner(0x8C, (0x10004).to_bytes(4, "little")))
    out = []

    def expect(b: bytes):
        out.extend([0x1000 | len(b), *b])

    def reply(b: bytes, bad: int | None = None):
        out.extend([0x2000 | len(b), *(x | (0x100 if i == bad else 0) for i, x in enumerate(b))])

    good = [_read_reply(w, x) for (w, _), x in zip(reads, words)]
    # rounds 1 and 2: the version's reply lost, then the version and the five reads
    expect(B.SYNC); expect(version)
    expect(B.SYNC); expect(version); reply(ver_reply)
    for (w, n), r in zip(reads, good):
        expect(_i2c_request(w, n)); reply(r)
    # round 3: a wrong PEC, a wrong sum, a NACK, a parity error (each: the sync again), VIN 12.0
    expect(B.SYNC)
    for (w, n), r, bad in zip(reads, [_read_reply(reads[0][0], words[0], pec_xor=0x5A),
                                      _read_reply(reads[1][0], words[1], sum_xor=1),
                                      B.wrap_frame(B.encode_inner(0x8A, b"\x00")),
                                      good[3], _read_reply(reads[4][0], vin2)],
                              (None, None, None, 5, None)):
        expect(_i2c_request(w, n)); reply(r, bad)
        if (w, n) != reads[4]:
            expect(B.SYNC)
    # round 4: page 0 at 160 C (refused), page 1 at 75.5 C (a 5C doubled), the rest good
    expect(B.SYNC)
    expect(_i2c_request(*reads[0])); reply(_read_reply(reads[0][0], to_linear11(160.0)))
    expect(B.SYNC)
    expect(_i2c_request(*reads[1])); reply(_read_reply(reads[1][0], hot))
    for (w, n), r in list(zip(reads, good))[2:]:
        expect(_i2c_request(w, n)); reply(r)
    # round 5: the host's hold, its sync and version request, its reply; then a round
    expect(B.SYNC + version); reply(ver_reply)
    for _ in range(2):                                              # round 5's, round 6's
        expect(B.SYNC)
        for (w, n), r in zip(reads, good):
            expect(_i2c_request(w, n)); reply(r)
    # round 6's hold: the host's request unanswered, HOLD_MS, a round
    expect(version)
    expect(B.SYNC)
    for (w, n), r in zip(reads, good):
        expect(_i2c_request(w, n)); reply(r)
    # a reset: the firmware 0x20001, then nothing for three rounds
    expect(B.SYNC); expect(version)
    reply(B.wrap_frame(B.encode_inner(0x8C, (0x20001).to_bytes(4, "little"))))
    out.extend([0x3000 | 120, 0])
    expv = words + [vin2, hot, 302, len(ver_reply), *ver_reply]
    return out, expv


def test_bmc_watch_rtl(have_verilator, tmp_path):
    """otpu_bmc_uart's watch against a scripted BMC (tb_bmc_watch: the rounds, the failures, the
    readings, the host's holds, the revocation, an unknown firmware)."""
    import subprocess
    script, expv = watch_script()
    (tmp_path / "script.hex").write_text("".join(f"{w:04x}\n" for w in script))
    (tmp_path / "expect.hex").write_text("".join(f"{w:08x}\n" for w in expv))
    r = subprocess.run([str(_bmc_rtl("tb_bmc_watch")), f"+script={tmp_path / 'script.hex'}",
                        f"+expect={tmp_path / 'expect.hex'}"],
                       capture_output=True, text=True, timeout=300)
    assert "PASS" in r.stdout and "FAIL" not in r.stdout, r.stdout[-3000:]
