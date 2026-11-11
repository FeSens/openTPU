"""The KV sequence split (Config.KVSPLIT, qwen3.kv_split): with fewer KV heads than slices each
KV head is copied on S / n_kv slices (KVDesc.kv_heads); at S / n_kv = 2 a decode bucket of
KVSPLIT blocks or more attends the head's blocks between its two copies (copy p: blocks p,
p + 2, ..) for the pair's queries, and each slice combines its own queries' partial softmaxes
with its partner's over the hub (qwen3._attend_split). Both copies append every token, so the
cache stays whole on each: an unsplit program, a prefill, a snapshot or the server's KV store
read it as before. A compile choice (never in the CFG_HASH), bcu1525-x2:wmx's; OTPU_KVSPLIT
overrides it; off, every program is the one before it."""
import dataclasses
import sys
from pathlib import Path

import numpy as np
import pytest

from opentpu import isa as I
from opentpu import language as ol
from opentpu import target as T
from opentpu.compiler import Affine, CompileError, RunVar, current
from opentpu.isasim import Config, design_config

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _gathers(prog) -> int:
    return sum(1 for x in prog if x.op == I.GATHER)


def _same(a, b) -> bool:
    return len(a) == len(b) and all(np.array_equal(I.assemble(x), I.assemble(y))
                                    for x, y in zip(a, b))


def test_kvsplit_is_a_compile_choice(monkeypatch):
    """bcu1525-x2:wmx's Config has KVSPLIT (b2's too: B2's machine is wmx's), every other
    target's 0 (st16's too: its programs stay); it is no part of the arch hash (the same machine) nor of the RTL's parameters;
    OTPU_KVSPLIT overrides it."""
    from opentpu.llm.qwen3 import kv_split_min
    monkeypatch.delenv("OTPU_KVSPLIT", raising=False)
    x2 = T.config("bcu1525-x2:wmx")
    assert x2.KVSPLIT > 1 and kv_split_min(x2) == x2.KVSPLIT == T.config("bcu1525-x2:b2").KVSPLIT
    for t in T.names():
        for v in ("",) + T.variants(t):
            k = f"{t}:{v}" if v else t
            if k not in ("bcu1525-x2:wmx", "bcu1525-x2:b2"):
                assert T.config(k).KVSPLIT == 0, k
    assert T.arch_hash(dataclasses.replace(x2, KVSPLIT=0)) == T.arch_hash(x2)
    assert "KVSPLIT" not in T.rtl_params("bcu1525-x2:wmx")
    monkeypatch.setenv("OTPU_KVSPLIT", "0")
    assert kv_split_min(x2) == 0
    monkeypatch.setenv("OTPU_KVSPLIT", "3")
    assert kv_split_min(Config()) == 3


def test_diverge_region_leaves_the_slices_alike():
    """Builder.diverge: the allocator is put back as the region ends; a tile allocated in it
    that outlives it, or a run-time argument first taken in it, is refused (the slices' later
    GATHER destinations and run_args must agree); args_ahead takes an address's argument before
    it."""
    v = RunVar("t", bound=256)

    @ol.jit
    def k():
        b = current()
        keep = ol.zeros([2, 32])
        before = b.tmem_next
        with b.diverge():
            t = ol.zeros([4, 32])
            del t
        assert b.tmem_next == before
        with pytest.raises(CompileError, match="TMEM regions"):
            with b.diverge():
                leak = ol.zeros([2, 32])
        del leak
        with pytest.raises(CompileError, match="run-time arguments"):
            with b.diverge():
                b.arg_reg(v, 4)
        u = RunVar("u", bound=256)
        b.args_ahead(Affine(64) + u * -4)
        n = len(b.run_args)
        with b.diverge():
            b._host_arg(u, -4)
        assert len(b.run_args) == n
        with pytest.raises(CompileError, match="one run-time value"):
            b.args_ahead(Affine(64))
        del keep

    k.trace(Config(S=1, D=32), 0, {})


@pytest.fixture(scope="module")
def tiny():
    pytest.importorskip("transformers")
    from test_qwen35 import _tiny_model
    _, W, spec = _tiny_model(8, hidden=512, ffn=1024)  # n_q 8, n_kv 2: at S = 4 two copies a head
    return W, spec


def test_split_decode_follows_the_whole_heads(tiny, monkeypatch):
    """A tiny Qwen3.5 (n_kv 2) at S = 4, its decode across buckets 2 (copy 1 holds the masked
    block) and 3 (copy 0 does), at the masked block's last and first variants: split, its logits
    follow the unsplit decode's (and the S = 4 emulation within its tolerance); a per-position
    program gives the resident bucket's bits; both copies of each head hold the same KV cache
    after the split steps; after a rollback (snapshot.py) the unsplit decode continues from the
    same state; the generate loop gives the split steps' greedy tokens. Bucket 1 (below KVSPLIT
    2) compiles to the program it was."""
    from test_qwen35 import _cos
    from opentpu.llm.qwen3 import Engine
    from opentpu.llm.qwen35 import emulated_logits
    from opentpu.serve.kvio import ContextIO
    W, spec = tiny
    cfg = design_config(S=4, DRAM_BYTES=1 << 26)
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 514)]
    eng = Engine(spec, W, cap=1024, cfg=cfg, resident=True, snapshots=1)
    eng.prefill(toks[:510], chunk=64)
    assert eng.snapshot()

    def run(split, resident=True):
        assert eng.rollback(510) == 510
        monkeypatch.setenv("OTPU_KVSPLIT", split)
        monkeypatch.setattr(eng, "resident", resident)
        eng._decodes.clear()
        return np.array([eng.step(t) for t in toks[510:]])

    split = run("2")
    io = ContextIO(eng, area=None)
    for page in io.read_pages(io.layout.split(514)):
        s = page.reshape(4, -1)                 # slice s's bytes; head h on slices 2h, 2h + 1
        assert np.array_equal(s[0], s[1]) and np.array_equal(s[2], s[3])
    assert np.array_equal(run("2", resident=False).view(np.uint32), split.view(np.uint32))
    # the generate loop (its programs at RunPos: split too) picks the split steps' greedy tokens
    monkeypatch.setattr(eng, "resident", True)
    assert eng.rollback(510) == 510 and eng.can_generate
    want, t = [], toks[510]
    for _ in range(4):
        t = int(np.argmax(eng.step(t)))
        want.append(t)
    assert eng.rollback(510) == 510
    assert eng.generate_card(toks[510], 4, stop_ids=[]) == want
    whole = run("0")
    assert _cos(split, whole).min() > 0.9999
    emu = emulated_logits(spec, W, toks, S=4)[-4:]
    assert _cos(whole, emu).min() > 0.999 and _cos(split, emu).min() > 0.999
    # the programs: three more GATHERs in the attention (one, in the layer loop's body) from
    # bucket 2 on, bucket 1 the same
    img = eng.image
    monkeypatch.setenv("OTPU_KVSPLIT", "0")
    p0, one = img.compile_decode(2, 256)[0], img.compile_decode(1, 3)[0]
    monkeypatch.setenv("OTPU_KVSPLIT", "2")
    p2 = img.compile_decode(2, 256)[0]
    assert [_gathers(p) for p in p2] == [_gathers(p) + 3 for p in p0]
    assert _same(img.compile_decode(1, 3)[0], one)


def test_no_split_where_it_does_not_apply(tiny, monkeypatch):
    """Where every slice holds its own KV head (S = 2: n_kv = S) the programs are the same with
    OTPU_KVSPLIT on."""
    from opentpu.llm.qwen3 import Engine
    W, spec = tiny
    eng = Engine(spec, W, cap=1024, cfg=design_config(S=2, DRAM_BYTES=1 << 26), resident=True)
    monkeypatch.setenv("OTPU_KVSPLIT", "0")
    a = eng.image.compile_decode(3, 512)[0]
    monkeypatch.setenv("OTPU_KVSPLIT", "2")
    b = eng.image.compile_decode(3, 512)[0]
    assert _same(a, b)
