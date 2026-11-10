"""otpu_dma's LDA (a QACT ST's rows back into ACT RAM; docs/isa.md "Quantize once") on its own
(sim/verilator/tb_dma.sv with LDA), among LDs and STs, at LDQ 1 and 2 (LDQ 2: an LD or an LDA
launches its DRAM reads under an LD or LDA before it, an LD also under an ST's write wait; an LDA
never under an ST), the quantizer holding ACT RAM's write at random (act_busy). Each done's
destination (an LDA's: its rows' blocks and records in ACT RAM) is the sequential reference's
after that command, and the final DRAM and TMEM the reference's; the tb checks every ACT write is
the head LDA's and no read passes a pending write it would see stale."""
import numpy as np
import pytest

from test_dma_ldq import BASE, CFGS, DWIN, LD, LENS, NDW, ST, TWIN, Ref, _cmd, _mems, check, run_dma

LDA = 0x14
D = 128                                  # (tb_dma's D; LANES 8, ALANES 16)
AROWS, ABLK = 16, 128                    # the tb's ACT RAM: rows, blocks of D bytes


class RefA(Ref):
    """Ref with ACT RAM: rows of ABLK blocks' bytes and their records (two words: S, the offsets)."""

    def __init__(self, dram, tmem):
        super().__init__(dram, tmem)
        self.act = np.zeros((AROWS, ABLK * D), np.uint8)
        self.rec = np.zeros((AROWS, ABLK, 2), np.uint32)

    def run(self, c):
        if c["op"] != LDA:
            return super().run(c)
        rows, ab, kb = c["w2"] & 0xFF, c["w2"] >> 8 & 0xFF, c["w2"] >> 16
        da, rs, ar = (c["a1"] >> 2) % NDW, c["w3"] // 4, c["w4"]
        rw = kb * (D // 4 + 2)
        for r in range(rows):
            w = self.dram[(da + r * rs + np.arange(rw)) % NDW]
            self.act[ar + r, ab * D:(ab + kb) * D] = w[:kb * D // 4].view(np.uint8)
            self.rec[ar + r, ab:ab + kb] = w[kb * D // 4:].reshape(kb, 2)
        out = []
        for r in range(rows):
            out += [self.act[ar + r, ab * D:(ab + kb) * D].view(np.uint32),
                    self.rec[ar + r, ab:ab + kb].reshape(-1)]
        return np.concatenate(out) if out else np.zeros(0, np.uint32)


def _lda(d, rows, ab, kb, stride, arow):
    """An LDA of `rows` rows of KB blocks from DRAM word d (64-byte aligned), `stride` bytes apart,
    into ACT rows arow .. at blocks ab .. ab + kb - 1."""
    c = dict(op=LDA, fl=0, a1=(BASE + d) * 4, w2=rows | ab << 8 | kb << 16, w3=stride, w4=arow,
             w5=0, w6=0)
    c["n"] = rows * kb * (D // 4 + 2)
    return c


def _rand_lda(rng, prev):
    kb, rows = int(rng.integers(1, 5)), int(rng.integers(1, 9))
    rb = kb * (D + 8)
    stride = -(-rb // 64) * 64 + 64 * int(rng.integers(0, 3))
    span = ((rows - 1) * stride + rb) // 4
    hi = (DWIN - span) // 16
    d = 16 * int(rng.integers(0, hi + 1))
    ab, arow = int(rng.integers(0, ABLK - kb + 1)), int(rng.integers(0, AROWS - rows + 1))
    if prev is not None and rng.random() < 0.5:                # on the words of the one before
        d = min(prev["d"] // 16 * 16, 16 * hi)
        if prev["op"] == LDA and rng.random() < 0.5:           # ... and its ACT rows and blocks
            ab, arow = min(prev["ab"], ABLK - kb), min(prev["arow"], AROWS - rows)
    return _lda(d, rows, ab, kb, stride, arow), dict(op=LDA, d=d, ab=ab, arow=arow,
                                                     n=span, t=0)


def _program_a(rng, ref, n_cmds, lda_pct=50):
    """LDAs among LDs and STs, many on the DRAM words (an LDA's: its ACT rows) of the one before."""
    cmds, prev = [], None
    for _ in range(n_cmds):
        kind = "LDA" if rng.random() * 100 < lda_pct else rng.choice(["LD"] * 6 + ["ST"] * 3 + ["LD0"])
        if kind == "LDA":
            c, prev = _rand_lda(rng, prev)
        else:
            n = int(rng.choice(LENS))
            d = int(rng.integers(0, DWIN - n + 1))
            t = int(rng.integers(0, TWIN - n + 1))
            if prev is not None and rng.random() < 0.5:
                d = min(max(prev["d"] + int(rng.integers(-n + 1, max(prev["n"], 1))), 0), DWIN - n)
            c = _cmd(ST if kind == "ST" else LD, d, t, 0 if kind == "LD0" else n)
            prev = dict(op=c["op"], d=d, n=n, t=t)
        c["want"] = ref.run(c)
        cmds.append(c)
    return cmds


@pytest.mark.parametrize("ldq", [1, 2])
@pytest.mark.parametrize("cfg", list(CFGS))
@pytest.mark.parametrize("seed", [1, 2])
def test_dma_lda_random(have_verilator, ldq, cfg, seed, tmp_path):
    """Random LD / LDA / ST sequences, the quantizer holding ACT RAM's write 30 % of the cycles:
    at LDQ 2 reads go out under (or for) an LDA."""
    rng = np.random.default_rng(3000 * seed + 31 * ldq + len(cfg))
    dram, tmem = _mems(rng)
    ref = RefA(dram, tmem)
    cmds = _program_a(rng, ref, 120 if CFGS[cfg]["depth"] > 32 else 60)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, seed=seed, lda=True, act_busy=30,
                  **CFGS[cfg])
    check(cmds, ref, out)
    if ldq == 1:
        assert out[2] == 0
    else:
        assert out[6] > 0


def test_dma_lda_overlap(have_verilator, tmp_path):
    """LDQ 2 with LDA: LDA behind LDA, LD behind LDA and LDA behind LD back to back at latency 100,
    their reads under the delivery before them (LDQ 1: latency + delivery each)."""
    rng = np.random.default_rng(5)
    dram, tmem = _mems(rng)
    cyc = {}
    for ldq in (1, 2):
        (tmp_path / str(ldq)).mkdir()
        ref = RefA(dram, tmem)
        cmds = []
        for i in range(24):
            if i % 3 == 2:
                c = _cmd(LD, 256 * (i % 16), 256 * (i % 32), 256)
            else:
                c = _lda(64 * (i % 32), 4, 8 * (i % 16), 2, 320, 4 * (i % 4))
            c["want"] = ref.run(c)
            cmds.append(c)
        out = run_dma(cmds, dram, tmem, tmp_path / str(ldq), ldq=ldq, lat=(100, 100),
                      wlat=(1, 1), gnt=100, start=100, depth=512, lda=True, act_busy=10)
        check(cmds, ref, out)
        cyc[ldq] = out[1]
        if ldq == 2:
            assert out[6] > 0
    assert cyc[2] < 0.75 * cyc[1], cyc


@pytest.mark.parametrize("ldq", [1, 2])
@pytest.mark.parametrize("nch", [1, 4])
def test_dma_lda_behind_ss(have_verilator, ldq, nch, tmp_path):
    """An LDA behind an ST behind an ST (LDQ 2: the second ST's TMEM reads under the first's write
    wait, its writes held until it is done; the LDA, never behind an ST, starts once both are
    done): rows the second ST just wrote, rows of other words, and an LD and an LDA after them, at
    the card's latencies, the writes on one channel or four."""
    rng = np.random.default_rng(23)
    dram, tmem = _mems(rng)
    ref = RefA(dram, tmem)
    cmds = []
    for i, d in enumerate((0, 1024, 2048)):
        rows = 4
        st_n = rows * 320 // 4                       # 4 rows of KB 2 blocks, 320 bytes apart
        seq = [_cmd(ST, d + 1000, 100 * i, 300), _cmd(ST, d, 2000 + 512 * i, st_n),
               _lda(d if i != 1 else d + 600, rows, 8 * i, 2, 320, 4 * i),
               _cmd(LD, d + 16, 6000 + 64 * i, 40), _lda(d, 2, 40, 2, 320, 12)]
        for c in seq:
            c["want"] = ref.run(c)
            cmds.append(c)
    out = run_dma(cmds, dram, tmem, tmp_path, ldq=ldq, lda=True, act_busy=20, nch=nch,
                  **CFGS["card"])
    check(cmds, ref, out)
    assert (out[7] > 0) == (ldq == 2)
