"""Bring-up checks shared by otpu-selftest and tests/test_board.py."""
from __future__ import annotations

import dataclasses
import time

import numpy as np

from opentpu import isa as I
from opentpu.isasim import Machine

DATA, W8, SC, OUT = 0, 0x10000, 0x20000, 0x30000
PROG_AT = 0x3C0000
ZERO_AT = 0x380000              # 256 KiB of zeros below PROG_AT: run_demo clears TMEM from it first
SPAN = 0x40000                  # bytes the demo program's results can touch (below PROG_AT)


def demo_image(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros(PROG_AT, np.uint8)
    img[DATA:DATA + 4 * 4096] = rng.uniform(-2, 2, 4096).astype(np.float32).view(np.uint8)
    img[W8:W8 + 16 * 256] = rng.integers(-127, 128, 16 * 256).astype(np.int8).view(np.uint8)
    img[SC:SC + 4 * 64] = rng.uniform(0.01, 0.02, 64).astype(np.float32).view(np.uint8)
    return img


def demo_program() -> list:
    """Every unit: DMA (aligned and unaligned), VPU simple and composite lanes, quantizer,
    MXU, QST byte writes."""
    return [
        I.ld(DATA, 0, 1024),
        I.ld(DATA + 4 * 1024 + 4, 1024, 777),                         # unaligned DRAM source
        I.vop(I.V_MUL, 2048, 0, 0, 4, 256, 256, 256, 0, I.B_SCALAR, 0.5),
        I.vop(I.V_EXP2, 3072, 0, 0, 2, 200, 200, 256, 0, I.B_SCALAR, 0.0),   # composite lanes
        I.vop(I.V_ADD, 3600, 1024, 0, 1, 300, 300, 300, 256, I.B_FULL),
        I.qact(0, 2, 0, 2, 256),
        I.mm(W8, SC, 4096, 16, 2, 256, 16, 2, 0, 8),
        I.qst(2048, OUT + 0x8000, OUT + 0xC000, 2, 2, 256, 256, 1),
        I.st(OUT, 2048, 1024),
        I.st(OUT + 4 * 1024 + 8, 3072, 400),                           # unaligned DRAM target
        I.st(OUT + 0x2000, 3600, 300),
        I.st(OUT + 0x3000, 4096, 32 + 2),
        I.halt(),
    ]


def masked_program() -> list:
    """Many partial DRAM writes from the accelerator: QST byte writes (dense and strided) and
    short word-masked stores at every alignment. The board's DDR3 has no data-mask pins, so
    each of these is a read-modify-write inside the memory controller."""
    prog = [I.ld(DATA, 0, 4096)]
    for k in range(24):
        n = 1 + (k * 7) % 23                                   # 1..23 words
        prog.append(I.st(OUT + 0x10000 + 4 * (37 * k + k % 5), 64 * k, n))
    prog.append(I.qst(0, OUT + 0x14000 + 3, OUT + 0x16000 + 4, 2, 2, 256, 256, 1))   # dense
    prog.append(I.qst(512, OUT + 0x18000 + 1, OUT + 0x1A000, 3, 1, 128, 1024, 3))    # strided
    prog.append(I.halt())
    return prog


def vops_program() -> list:
    """The VPU functions added for linear-recurrence models (RDOT, OUTER, LOG2; Qwen3.5's
    DeltaNet layers). A bitstream built before them runs this program without an error but
    computes other values: the model check of Qwen3.5 would fail late and obscurely."""
    return [
        I.ld(DATA, 0, 4096),
        I.vop(I.V_ABS, 4096, 0, 0, 2, 256, 256, 256, 0, I.B_SCALAR, 0.0),
        I.vop(I.V_LOG2, 4608, 4096, 0, 2, 256, 256, 256, 0, I.B_SCALAR, 0.0),
        I.vop(I.V_RDOT, 5120, 0, 1024, 4, 256, 1, 256, 256),
        I.vop(I.V_COPY, 6144, 0, 0, 4, 64, 64, 64, 0, I.B_SCALAR, 0.0),
        I.outer(6144, 5120, 2048, 3072, 4, 64, 64, 1, "scalar"),
        I.st(OUT + 0x20000, 4096, 1028),
        I.st(OUT + 0x22000, 6144, 256),
        I.halt(),
    ]


STATES = 0x100000              # stream_program's DRAM states (below PROG_AT)


def stream_program() -> list:
    """The stream engine's subset (docs/stream.md 4.4) on states in DRAM: every dmode (DELTA,
    DELTA1, SCALE, DOT) with each gate (a constant, a column, one), A on slot 1 or 2, Q on or
    off, the ZERO flag, 64..256 columns, and a DSTEP; descriptors FILLed at the start as the
    compiler does. The states (copies of DATA) are stepped in place and the row outputs
    stored."""
    combos = [(I.D_DELTA, 1, "const", True, False), (I.D_DELTA, 2, "col", False, False),
              (I.D_DELTA1, 1, "one", True, False), (I.D_DELTA1, 2, "col", True, False),
              (I.D_SCALE, 1, "col", True, False), (I.D_SCALE, 1, "const", False, True),
              (I.D_DOT, 2, "col", True, False), (I.D_DOT, 1, "one", False, False),
              (I.D_SCALE, 1, "one", True, False)]
    shapes = [(16, 128), (7, 64), (9, 256), (32, 192)]
    DESC = 60000
    steps, descs = [], {}
    for i, (dp, a, g, q, zero) in enumerate(combos):
        rows, cols = shapes[i % len(shapes)]
        d = I.state_desc(rows, cols, dp, a, g, q)
        descs.setdefault(tuple(d.words()), DESC + 8 * len(descs))
        steps.append((d, rows, cols, zero))
    prog = [I.vop(I.V_FILL, at + j, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR,
                  imm=np.uint32(w).view(np.float32))
            for words, at in descs.items() for j, w in enumerate(words)]
    prog += [I.ld(DATA, 0, 4096),                                   # vectors, rows, states
             I.vop(I.V_ABS, 4096, 0, 0, 1, 1024, 0, 0, 0),           # decay columns in (0, 0.5]
             I.vop(I.V_MUL, 4096, 4096, 0, 1, 1024, 0, 0, 0, I.B_SCALAR, 0.25),
             I.vop(I.V_FILL, 5200, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.875),   # K0
             I.vop(I.V_FILL, 5202, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, 0.5)]     # K1 (ks 2)
    o = 6000
    for i, (d, rows, cols, zero) in enumerate(steps):
        st = STATES + 0x10000 * i
        vec = 8192 + 1024 * (i % 4)
        prog += [I.st(st, 64 * i, rows * cols),                      # the state: DATA words
                 I.vop(I.V_COPY, vec, 1024 * (i % 3), 0, 1, 3 * cols, 0, 0, 0),
                 I.vop(I.V_COPY, vec + 3 * cols, 4096, 0, 1, cols, 0, 0, 0),
                 I.stream(descs[tuple(d.words())], st, st, vec, 512 + 16 * i, 5200, o, ks=2,
                          zero=zero)]
        if d.q_en:
            prog.append(I.st(OUT + 0x24000 + 1024 * i, o, rows))
        o += 300
    prog += [I.dstep(STATES + 0x10000 * len(steps), 8192, 512, 16, 128, 5200, 2, o),
             I.st(OUT + 0x24000 + 1024 * len(steps), o, 16), I.halt()]
    return prog


def stream_check(board, cfg) -> tuple[bool, str]:
    """otpu-selftest's stream stage: stream_program() against the ISA simulator (a bitstream
    with the stream engine, CAPS bit26); others pass with a note."""
    if not cfg.STREAM:
        return True, "note: no stream engine (CAPS bit26 clear)"
    ok, msg, _ = run_demo(board, cfg, stream_program())
    return ok, f"STREAM (every mode) and DSTEP: {msg}"


def vops_check(board, cfg, need: bool = False) -> tuple[bool, str]:
    """otpu-selftest's vops stage: vops_program() against the ISA simulator. Wrong results fail
    on a bitstream that has the functions (register map >= VOPS_SINCE), and on an older one
    only when `need` (a Qwen3.5 model check follows); an older one passes with a note."""
    from .regs import VOPS_SINCE
    ok, msg, _ = run_demo(board, cfg, vops_program())
    if ok:
        # then otpu-diag's reduction and RDOT / OUTER / LOG2 programs back to back: a build
        # passed the vops program but wrote RDOT's last row sum stale (a block RAM the
        # synthesis made of an asynchronous-read buffer), one RDOT late in otpu-diag
        from .opchecks import diag_image, op_checks
        progs = [(n, p) for g, n, p in op_checks(cfg) if g in ("vpu-reduce", "vpu-new")]
        bad = []
        for name, prog in progs:
            ok2, msg2, _ = run_demo(board, cfg, prog, diag_image())
            if not ok2:
                bad.append(f"{name}: {msg2.split(',')[0]}")
        if bad:
            return False, (f"RDOT / OUTER / LOG2 program ok, but {len(bad)} of {len(progs)} "
                           "reduction / RDOT / OUTER / LOG2 op checks differ: "
                           + "; ".join(bad[:4]))
        return True, f"RDOT / OUTER / LOG2 ok ({msg}); {len(progs)} op checks ok"
    rm = board.info()["regmap"]
    msg = f"RDOT / OUTER / LOG2 differ from the ISA simulator ({msg.split(',')[0]})"
    if rm >= VOPS_SINCE:
        return False, f"{msg} on a bitstream that has them (register map {rm})"
    msg += f", a bitstream built before them (register map {rm})"
    return (False, msg) if need else (True, f"note: {msg}: Qwen3 and LFM2 only")


def run_demo(board, cfg, prog: list | None = None,
             img: np.ndarray | None = None) -> tuple[bool, str, dict]:
    """Run a program (default: the demo) on the board and on the ISA simulator, from the same
    DRAM image (default: demo_image()); compare DRAM below PROG_AT. A mismatch reports the
    number of bytes, the first addresses and got / want of the first differing words."""
    img = demo_image() if img is None else img
    # TMEM keeps its contents from one program to the next on the card, and the ISA simulator
    # starts from zeros: clear it first, so a program that stores a word it never wrote
    # (the demo's 32 + 2) compares the same after any earlier program
    prog = [I.ld(ZERO_AT, 0, cfg.TMEM_WORDS)] + (prog or demo_program())
    ref = np.zeros(min(cfg.DRAM_BYTES, 1 << 23), np.uint8)
    ref[:len(img)] = img
    sl = Machine(dataclasses.replace(cfg, DRAM_BYTES=len(ref)), [prog], [ref]).run().slices[0]
    board.write(ZERO_AT, np.zeros(4 * cfg.TMEM_WORDS, np.uint8))
    board.write(0, img)
    board.load_program(PROG_AT, np.asarray(I.assemble(prog), np.uint32))
    st = board.run(timeout=10.0)
    got = board.read(0, PROG_AT)
    want = sl.dram[:PROG_AT]
    if st["instructions"][0] != sl.icount:
        return False, f"retired {st['instructions'][0]} of {sl.icount} instructions", st
    bad = np.nonzero(got != want)[0]
    if len(bad):
        words = sorted({int(b) // 4 * 4 for b in bad[:64]})[:3]
        gw, ww = got.view("<u4"), want.view("<u4")
        diff = "; ".join(f"{a:#x}: got {int(gw[a // 4]):#010x} want {int(ww[a // 4]):#010x}"
                         for a in words)
        return False, f"{len(bad)} DRAM bytes differ from the ISA simulator, first at " \
                      f"{[hex(int(b)) for b in bad[:6]]} ({diff})", st
    return True, f"{st['cycles']} cycles", st


def pattern_test(board, regions: list[tuple[int, int]], seed: int = 1) -> tuple[bool, str]:
    """Write random bytes to every region (logical addresses), read them back."""
    rng = np.random.default_rng(seed)
    data = [rng.integers(0, 256, n).astype(np.uint8) for _, n in regions]
    for (a, _), d in zip(regions, data):
        board.write(a, d)
    for (a, n), d in zip(regions, data):
        got = board.read(a, n)
        bad = np.nonzero(got != d)[0]
        if len(bad):
            b = (a + int(bad[0])) // 64
            ch = b % 2 ^ (board.chash and bin(b // 2).count("1") & 1)
            return False, (f"region {a:#x}+{n:#x}: {len(bad)} bytes wrong, first at "
                           f"{a + int(bad[0]):#x} (logical beat {b}, channel {ch})")
    return True, f"{len(regions)} regions, {sum(n for _, n in regions)} bytes"


def channel_patterns(transport, ch: int, ch_bytes: int, seed: int = 2) -> tuple[bool, str]:
    """Random data straight to one channel (raw channel addresses) at the bottom, middle and
    top of the channel."""
    rng = np.random.default_rng(seed + ch)
    n = min(65536, ch_bytes // 8)
    for off in (0, 4096, ch_bytes // 2, ch_bytes - n):
        d = rng.integers(0, 256, n).astype(np.uint8)
        transport.mem_write(ch, off, d)
        if not np.array_equal(transport.mem_read(ch, off, n), d):
            return False, f"channel {ch} offset {off:#x}: read-back mismatch"
    return True, "4 regions"


def partial_writes(transport, ch: int, base: int = 1 << 20, seed: int = 3) -> tuple[bool, str]:
    """Sub-beat host updates (1..63 bytes at odd offsets) into a filled region of one channel.
    The card transport merges each into whole 64-byte beats on the host (XdmaTransport.mem_write):
    sub-beat DMA writes, which the ECC controller would turn into read-modify-writes, can wedge
    the card's write path. The controller's byte-strobe path is exercised by the accelerator's
    masked writes (the kernel stage)."""
    rng = np.random.default_rng(seed + ch)
    ref = rng.integers(0, 256, 4096).astype(np.uint8)
    transport.mem_write(ch, base, ref)
    for _ in range(200):
        n = int(rng.integers(1, 64))
        o = int(rng.integers(0, 4096 - n))
        d = rng.integers(0, 256, n).astype(np.uint8)
        transport.mem_write(ch, base + o, d)
        ref[o:o + n] = d
    got = transport.mem_read(ch, base, 4096)
    bad = np.nonzero(got != ref)[0]
    if len(bad):
        return False, (f"channel {ch}: {len(bad)} bytes wrong after partial writes, first at "
                       f"{base + int(bad[0]):#x}")
    return True, "200 partial writes"


def address_lines(transport, ch: int, ch_bytes: int) -> tuple[bool, str]:
    """Walking address bits on one channel (raw channel addresses): a unique 64-byte tag at
    offset 0 and at every power of two; an aliased or stuck address line shows up as a tag
    overwritten by another."""
    offs = [0] + [1 << k for k in range(6, (ch_bytes - 1).bit_length())]
    tags = {o: np.frombuffer(np.uint64(0xA5A5_0000_0000_0000 | (ch << 40) | o).tobytes() * 8,
                             np.uint8) for o in offs}
    for o in offs:
        transport.mem_write(ch, o, tags[o])
    for o in offs:
        got = transport.mem_read(ch, o, 64)
        if not np.array_equal(got, tags[o]):
            v = int(np.frombuffer(got[:8].tobytes(), np.uint64)[0])
            return False, (f"channel {ch} offset {o:#x}: read tag {v:#x} -- address bit "
                           f"{o.bit_length() - 1} aliases or is stuck")
    return True, f"{len(offs)} address bits"


def bandwidth(transport, nbytes: int) -> tuple[float, float]:
    """H2C and C2H GB/s over both channels."""
    buf = np.random.default_rng(0).integers(0, 256, nbytes // 2).astype(np.uint8)
    t = time.time()
    for c in (0, 1):
        transport.mem_write(c, 0, buf)
    w = nbytes / (time.time() - t) / 1e9
    t = time.time()
    for c in (0, 1):
        transport.mem_read(c, 0, nbytes // 2)
    r = nbytes / (time.time() - t) / 1e9
    return w, r


def model_check(t, cfg, model: str, tokens: int, sim: bool, wformat: str = "int8",
                head_format: str | None = None) -> tuple[bool, str]:
    """Greedy decoding of "What is the capital of France?" on the card (transport t, its
    configuration cfg) against the ISA simulator, token for token; the prompt runs in chunks
    (Engine.prefill_chunks) on both. sim: t is a small board
    model; the model gets its own, sized to the model's DRAM. wformat / head_format: the
    weight formats of the layers and of the LM head (Engine)."""
    from opentpu.llm import load_spec, model_dir
    from opentpu.llm.qwen3 import Engine, load_weights
    from transformers import AutoTokenizer

    from .board import BoardBackend, SimTransport, sim_config
    path = model_dir(model)
    spec = load_spec(path)
    W = load_weights(path)
    tok = AutoTokenizer.from_pretrained(path)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    cap = 256
    rcfg = sim_config(spec, cap, cfg)                     # same layout, DRAM sized to the model
    tq = SimTransport(ch_bytes=rcfg.DRAM_BYTES // 2) if sim else t
    fmt = {"wformat": wformat, "head_format": head_format}
    dev = Engine(spec, W, cap=cap, cfg=rcfg if sim else cfg,
                 backend=lambda c, imgs: BoardBackend(c, imgs, transport=tq, model=path.name),
                 **fmt)
    ref = Engine(spec, W, cap=cap, cfg=rcfg, **fmt)
    t0 = time.time()
    got = dev.generate(ids, max_new=tokens)
    dt = time.time() - t0
    want = ref.generate(ids, max_new=tokens)
    text = tok.decode(got, skip_special_tokens=True)
    pre = [s for s in dev.stats if "rows" in s]         # the prompt's multi-token runs
    pre_cyc = sum(s["cycles"] for s in pre) / max(1, sum(s["rows"] for s in pre))
    one = [s["cycles"] for s in dev.stats if "rows" not in s]
    ok = got == want
    return ok, (f"{text!r}; prompt of {len(ids)} tokens in {len(pre)} runs of up to "
                f"{dev.rows}, {pre_cyc / 1e6:.2f} Mcycles/token; {len(one)} one-token runs, "
                f"{np.mean(one) / 1e6:.2f} Mcycles/token; {dt:.1f} s wall" +
                ("" if ok else f"; ISA simulator says {tok.decode(want)!r}"))
