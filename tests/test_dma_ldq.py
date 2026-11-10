"""otpu_dma's LD, ST and WAITW on their own (sim/verilator/tb_dma.sv), at LDQ 1 and 2 (LDQ 2: an
LD's DRAM reads start under the command before it, an ST's TMEM reads under an ST's write wait, its
writes held until that ST is done; docs/dma_ldq.md).

Random command sequences against a sequential reference, with commands forced onto the words of
the one before: an LD of the words an ST just stored (DRAM read after write), of the words beside
them in the same chunk, an LD into the TMEM words an LD just wrote (write after write), an ST of
them (TMEM read after write), WAITWs and zero-length commands. Port B's grants and latencies are
random and its writes posted (they land when acknowledged), on one channel or several (nch: each
channel's in order, the channels independently, as the memory's). The tb checks each done is the
oldest command's, an ST's with none of its writes pending, and that no read passes a pending write
it would see stale; here each done's destination words must be the reference's after that command,
and the final DRAM and TMEM the reference's."""
import numpy as np
import pytest

from opentpu import rtlsim

LD, ST, WAITW = 0x10, 0x11, 0x07
NDW, NTW = 1 << 15, 1 << 14              # the tb's DRAM and TMEM words
CW = 32                                  # words a chunk (D 128, the x2 targets')
BASE = 0x2_8000_0000                     # the DRAM window's first word (AW 36: w1h is used)
DWIN, TWIN = 4096, 8192                  # the windows commands fall in (words)
LENS = [1, 2, 3, 7, 8, 9, 15, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 700, 1000, 2049]

# port B and the start: grant %, read latency, write acknowledgement latency, start %
CFGS = {
    "fast": dict(gnt=100, lat=(1, 3), wlat=(1, 3), start=100, depth=512),
    "slow": dict(gnt=60, lat=(40, 120), wlat=(10, 80), start=100, depth=512),
    "card": dict(gnt=85, lat=(60, 300), wlat=(30, 200), start=100, depth=512),
    "small": dict(gnt=50, lat=(1, 300), wlat=(1, 200), start=70, depth=32),   # (SRUN: 32 at least)
}


class Ref:
    """The commands in order on DRAM and TMEM words (indices mod NDW / NTW, as the tb's)."""

    def __init__(self, dram, tmem):
        self.dram, self.tmem = dram.copy(), tmem.copy()

    def run(self, c):
        op, a1, w2, n = c["op"], c["a1"], c["w2"], c["w3"]
        da = (a1 >> 2) % NDW
        if op == LD:
            self.tmem[w2:w2 + n] = self.dram[da:da + n]
            return self.tmem[w2:w2 + n].copy()
        if op == ST:
            self.dram[da:da + n] = self.tmem[w2:w2 + n]
            return self.dram[da:da + n].copy()
        self.tmem[w2] = self.dram[da]                  # WAITW: its condition holds on the word
        return self.tmem[w2:w2 + 1].copy()


def _cmd(op, d, t, n, **kw):
    return dict(op=op, fl=0, a1=(BASE + d) * 4, w2=t, w3=n, w4=0, w5=0, w6=0, **kw)


def _waitw(ref, d, t, rng):
    """A WAITW whose condition holds on the word the reference has there: EQ, NE or GE."""
    v = int(ref.dram[(BASE + d) % NDW])
    k = int(rng.integers(0, 3))
    r = [v, v ^ 1, v][k]
    return dict(op=WAITW, fl=k, a1=(BASE + d) * 4, w2=t, w3=r, w4=0xFFFFFFFF, w5=2, w6=0)


def _program(rng, ref, n_cmds):
    """Commands, most of them on or beside the words of the one before."""
    cmds, prev = [], None
    for _ in range(n_cmds):
        kind = rng.choice(["LD"] * 10 + ["ST"] * 6 + ["WAITW", "LD0", "ST0"])
        n = int(rng.choice(LENS))
        d = int(rng.integers(0, DWIN - n + 1))
        t = int(rng.integers(0, TWIN - n + 1))
        if prev is not None and rng.random() < 0.6:
            pd, pt, pn = prev["d"], prev["t"], max(prev["n"], 1)
            how = rng.integers(0, 4)
            if how == 0:                                   # on the same words
                d, t = pd, pt
                n = pn
            elif how == 1:                                 # overlapping them
                d = pd + int(rng.integers(-n + 1, pn))
                t = pt + int(rng.integers(-n + 1, pn))
            elif how == 2:                                 # just after them (the same chunk)
                d, t = pd + pn, pt + pn
            else:                                          # just before them
                d, t = pd - n, pt - n
            d = min(max(d, 0), DWIN - n)
            t = min(max(t, 0), TWIN - n)
        if kind == "WAITW":
            c = _waitw(ref, d, t, rng)
            n = 1
        else:
            c = _cmd(ST if kind.startswith("ST") else LD, d, t, 0 if kind.endswith("0") else n)
        c["want"] = ref.run(c)
        cmds.append(c)
        prev = dict(d=d, t=t, n=n)
    return cmds


def run_dma(cmds, dram, tmem, tmp_path, ldq=1, gnt=100, lat=(1, 3), wlat=(1, 3), start=100,
            depth=512, seed=1, lda=False, act_busy=0, nch=1):
    """Runs the commands on tb_dma; returns (the dones as (k, cycle, words), cycles, early reads,
    final DRAM, final TMEM, early reads under an ST, early reads with an LDA in the head or the
    next, cycles an ST read TMEM behind an ST, writes acknowledged before an earlier one: the
    channels' reordering). lda: the DMA and tb with LDA
    (tests/test_dma_lda.py), act_busy the % of cycles the quantizer holds ACT RAM's write, nch the
    write channels (acknowledged in order on each, not across them)."""
    words = sum(1 if c["op"] == WAITW else max(c["w3"], 1) if c["op"] != 0x14 else c["n"]
                for c in cmds)                     # (a WAITW's w3: its reference; an LDA's: its words)
    max_cyc = 20000 + 8 * words // CW * (100 // gnt + 1) + len(cmds) * (2 * lat[1] + 2 * wlat[1] + 64)
    lines = [" ".join(f"{v:x}" for v in (len(cmds), gnt, *lat, *wlat, start, seed, max_cyc))]
    for c in cmds:
        lines.append(" ".join(f"{c[k]:x}" for k in ("op", "fl", "a1", "w2", "w3", "w4", "w5", "w6")))
    fin, fout = tmp_path / "in.hex", tmp_path / "out.txt"
    fin.write_text("\n".join(lines) + "\n")
    (tmp_path / "dram.hex").write_text("\n".join(f"{int(w):08x}" for w in dram) + "\n")
    (tmp_path / "tmem.hex").write_text("\n".join(f"{int(w):08x}" for w in tmem) + "\n")
    R = rtlsim.RTL
    exe = rtlsim.build("tb_dma", [R / "vpu/otpu_fp.sv", R / "top/otpu_pkg.sv",
                                  R / "tech/generic/otpu_tech_ram.sv", R / "dma/otpu_dma.sv",
                                  rtlsim.TB / "tb_dma.sv"],
                       {"DEPTH": depth, "LDQ": ldq} | ({"LDA": 1} if lda else {}))
    r = rtlsim.run_sim([exe, f"+in={fin}", f"+out={fout}", f"+dram={tmp_path / 'dram.hex'}",
                        f"+tmem={tmp_path / 'tmem.hex'}", f"+dram_out={tmp_path / 'dram_out.hex'}",
                        f"+tmem_out={tmp_path / 'tmem_out.hex'}", f"+act_busy={act_busy}",
                        f"+nch={nch}"],
                       timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    dones, cyc, early, early_st, early_la, early_ss, wro = [], None, None, None, None, None, None
    for ln in fout.read_text().split("\n"):
        t = ln.split()
        if not t:
            continue
        if t[0] == "E":
            cyc, early, early_st, early_la, early_ss, wro = (int(v) for v in t[1:7])
            continue
        dones.append((int(t[0]), int(t[1]), np.array([int(w, 16) for w in t[2:]], np.uint32)))
    assert cyc is not None, r.stdout[-3000:]
    return (dones, cyc, early, _memh(tmp_path / "dram_out.hex"), _memh(tmp_path / "tmem_out.hex"),
            early_st, early_la, early_ss, wro)


def _memh(p):
    return np.array([int(w, 16) for w in p.read_text().split() if not w.startswith(("@", "//"))],
                    np.uint32)


def check(cmds, ref, out):
    dones, _, _, dram, tmem = out[:5]
    assert [k for k, _, _ in dones] == list(range(len(cmds))), "dones out of order"
    bad = [k for k, _, v in dones if not np.array_equal(v, cmds[k]["want"])]
    assert not bad, f"commands {bad[:10]} done with other words than the reference's"
    assert np.array_equal(dram, ref.dram), f"DRAM differs at {np.nonzero(dram != ref.dram)[0][:10]}"
    assert np.array_equal(tmem, ref.tmem), f"TMEM differs at {np.nonzero(tmem != ref.tmem)[0][:10]}"


def _mems(rng):
    return (rng.integers(0, 1 << 32, NDW, dtype=np.uint64).astype(np.uint32),
            rng.integers(0, 1 << 32, NTW, dtype=np.uint64).astype(np.uint32))


@pytest.mark.parametrize("ldq", [1, 2])
@pytest.mark.parametrize("cfg", list(CFGS))
@pytest.mark.parametrize("seed", [1, 2])
def test_dma_ldq_random(have_verilator, ldq, cfg, seed, tmp_path):
    """Random sequences (most commands on or beside the words of the one before)."""
    rng = np.random.default_rng(1000 * seed + 17 * ldq + len(cfg))
    dram, tmem = _mems(rng)
    ref = Ref(dram, tmem)
    cmds = _program(rng, ref, 160 if CFGS[cfg]["depth"] > 32 else 80)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, seed=seed, nch=1 if seed == 1 else 4,
                  **CFGS[cfg])
    check(cmds, ref, out)
    if ldq == 1:
        assert out[2] == out[7] == 0
    else:                       # reads under an LD and under an ST's write wait
        assert out[2] > out[5] > 0


@pytest.mark.parametrize("ldq", [1, 2])
def test_dma_ldq_directed(have_verilator, ldq, tmp_path):
    """Pairs back to back at the card's latencies: an LD of the words an ST just stored, of the
    words beside them in its chunks, of a part of them, starting at the ST's last word (de - 1:
    it must wait for the ST's writes) and at the word after it (de: it may start under them); an
    LD into the TMEM words an LD just wrote, then an ST of them; an LD behind a zero-length LD and
    behind a WAITW."""
    rng = np.random.default_rng(7)
    dram, tmem = _mems(rng)
    ref = Ref(dram, tmem)
    seq = [_cmd(ST, 100, 0, 300), _cmd(LD, 100, 1000, 300),        # LD after ST, same words
           _cmd(ST, 517, 40, 20), _cmd(LD, 537, 2000, 11),         # ... beside them, same chunk
           _cmd(ST, 600, 80, 64), _cmd(LD, 590, 2100, 30),         # ... a part of them
           _cmd(ST, 800, 160, 64), _cmd(LD, 863, 2200, 20),        # ... from its de - 1
           _cmd(ST, 900, 240, 50), _cmd(LD, 950, 2300, 20),        # ... from its de
           _cmd(LD, 1200, 3000, 256), _cmd(LD, 1700, 3000, 256),   # TMEM write after write
           _cmd(ST, 2500, 3000, 256), _cmd(LD, 2500, 3500, 256),   # ST of them, LD back
           _cmd(LD, 50, 4000, 0), _cmd(LD, 60, 4100, 33),          # behind a zero-length LD
           _cmd(LD, 3000, 4200, 1)]
    cmds = []
    for c in seq:
        c["want"] = ref.run(c)
        cmds.append(c)
    w = _waitw(ref, 100, 4300, rng)
    w["want"] = ref.run(w)
    cmds.append(w)
    c = _cmd(LD, 101, 4301, 7)
    c["want"] = ref.run(c)
    cmds.append(c)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, **CFGS["card"])
    check(cmds, ref, out)


@pytest.mark.parametrize("ldq", [1, 2])
def test_dma_ldq_buffer_full(have_verilator, ldq, tmp_path):
    """The chunk buffer full of the command before's chunks when an LD may start (DEPTH 32, LDs of
    40 chunks at latency 300: the last requests go out with 32 chunks outstanding or unread): the
    next LD's requests wait for room (otpu_dma checks occ <= DEPTH; an overrun would show in the
    words too)."""
    rng = np.random.default_rng(11)
    dram, tmem = _mems(rng)
    ref = Ref(dram, tmem)
    cmds = []
    for i, (d, t, n) in enumerate([(0, 0, 1280), (1280, 1280, 512), (2048, 2048, 1280),
                                   (16, 4000, 1270), (3500, 6000, 37), (100, 7000, 1000)]):
        c = _cmd(LD, d, t, n)
        c["want"] = ref.run(c)
        cmds.append(c)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, gnt=100, lat=(300, 300), wlat=(1, 1),
                  start=100, depth=32)
    check(cmds, ref, out)
    if ldq == 2:
        assert out[2] > 0


def test_dma_ldq_overlap(have_verilator, tmp_path):
    """LDQ 2's gain: back-to-back LDs of 8 chunks at latency 100 overlap each one's DRAM latency
    with the delivery before it (LDQ 1: latency + delivery each)."""
    rng = np.random.default_rng(3)
    dram, tmem = _mems(rng)
    cyc = {}
    for ldq in (1, 2):
        (tmp_path / str(ldq)).mkdir()
        ref = Ref(dram, tmem)
        cmds = []
        for i in range(40):
            c = _cmd(LD, 256 * (i % 16), 256 * (i % 32), 256)
            c["want"] = ref.run(c)
            cmds.append(c)
        out = run_dma(cmds, dram, tmem, tmp_path / str(ldq), ldq=ldq, lat=(100, 100), wlat=(1, 1),
                      gnt=100, start=100, depth=512)
        check(cmds, ref, out)
        cyc[ldq] = out[1]
        if ldq == 2:
            assert out[2] > 0
    assert cyc[2] < 0.6 * cyc[1], cyc


def _run_seq(seq, ref):
    cmds = []
    for c in seq:
        c["want"] = ref.run(c)
        cmds.append(c)
    return cmds


@pytest.mark.parametrize("ldq", [1, 2])
@pytest.mark.parametrize("nch", [1, 4])
def test_dma_ss_directed(have_verilator, ldq, nch, tmp_path):
    """An ST behind an ST (LDQ 2: its TMEM reads under the first's write wait, its writes held
    until the first is done) at the card's latencies, the writes on one channel or on four
    (acknowledged out of order across them): the second on the same words (write after write: its
    words must win), on other words of the same chunks, overlapping a part; a short second (2
    chunks behind 64: fewer than the wait's latency / 4); an LD of the second's words behind the
    pair (read after write) and of other words (under the second's wait); four STs in a row (each
    behind the one before: a third waits, rdy low, while the first's done is pending); a WAITW on
    the second's word; a zero-length ST behind an ST."""
    rng = np.random.default_rng(13)
    dram, tmem = _mems(rng)
    ref = Ref(dram, tmem)
    seq = [_cmd(ST, 100, 0, 300), _cmd(ST, 100, 500, 300),            # the same words
           _cmd(ST, 517, 40, 20), _cmd(ST, 537, 900, 11),             # beside them, same chunk
           _cmd(ST, 600, 80, 64), _cmd(ST, 590, 1000, 30),            # a part of them
           _cmd(ST, 1000, 2000, 2048), _cmd(ST, 3100, 1100, 40),      # a short second
           _cmd(ST, 1200, 1200, 256), _cmd(ST, 1300, 4000, 64),       # ... then an LD of its words
           _cmd(LD, 1300, 4500, 64),
           _cmd(ST, 1400, 4600, 256), _cmd(ST, 1700, 4900, 64),       # ... then an LD of others
           _cmd(LD, 2000, 5000, 64),
           _cmd(ST, 2100, 5100, 100), _cmd(ST, 2150, 5200, 100),      # four in a row
           _cmd(ST, 2120, 5300, 100), _cmd(ST, 2100, 5400, 33),
           _cmd(ST, 2300, 5500, 64), _cmd(ST, 2400, 5600, 64)]        # ... then a WAITW
    cmds = _run_seq(seq, ref)
    w = _waitw(ref, 2410, 5700, rng)
    w["want"] = ref.run(w)
    cmds.append(w)
    cmds += _run_seq([_cmd(ST, 2500, 5800, 70), _cmd(ST, 2600, 5900, 0),   # zero-length behind
                      _cmd(ST, 2500, 6000, 70)], ref)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, nch=nch, **CFGS["card"])
    check(cmds, ref, out)
    assert (out[7] > 0) == (ldq == 2)
    assert (out[8] > 0) == (nch > 1)            # (the channels did reorder the acknowledgements)


@pytest.mark.parametrize("nch", [1, 4])
def test_dma_ss_short_front(have_verilator, nch, tmp_path):
    """Short STs (2 chunks: 8 cycles of TMEM reads) behind long ones (64 chunks) at a write wait of
    200: the short one's reads are in the buffer long before the long one's done, its writes go out
    after it, at a chunk a cycle; the next long one's reads under the short one's wait."""
    rng = np.random.default_rng(17)
    dram, tmem = _mems(rng)
    ref = Ref(dram, tmem)
    seq = []
    for i in range(6):
        seq += [_cmd(ST, 2048 * (i % 2), 2048 * (i % 3), 2048), _cmd(ST, 64 * i, 7000 + 64 * i, 64)]
    cmds = _run_seq(seq, ref)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=2, gnt=100, lat=(100, 100), wlat=(200, 200),
                  start=100, depth=512, nch=nch)
    check(cmds, ref, out)
    assert out[7] > 0


def test_dma_ss_overlap(have_verilator, tmp_path):
    """LDQ 2's ST behind ST gain: back-to-back STs of 8 chunks (32 cycles of TMEM reads) at a
    write wait of 100 read their TMEM under the wait before them (LDQ 1: reads + writes + wait
    each)."""
    rng = np.random.default_rng(19)
    dram, tmem = _mems(rng)
    cyc = {}
    for ldq in (1, 2):
        (tmp_path / str(ldq)).mkdir()
        ref = Ref(dram, tmem)
        cmds = _run_seq([_cmd(ST, 256 * (i % 16), 256 * (i % 32), 256) for i in range(40)], ref)
        out = run_dma(cmds, dram, tmem, tmp_path / str(ldq), ldq=ldq, lat=(100, 100),
                      wlat=(100, 100), gnt=100, start=100, depth=512)
        check(cmds, ref, out)
        cyc[ldq] = out[1]
        assert (out[7] > 0) == (ldq == 2)
    assert cyc[2] < 0.85 * cyc[1], cyc
