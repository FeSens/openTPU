"""Gemma 4's prompt runs (docs/prefill.md, gemma4.gemma4_prompt_run) on the ISA simulator, on
test_gemma4's tiny model (sliding and global layers, the KV-shared ones, per-layer inputs): the
logits and the layers' DRAM (weights and KV caches) equal those of compile-time runs of the same
split bit for bit, with the PLE table on the card (int8, fp4) and on the host (the host writes
each run's records into the slot first), from position 0 and from a later one (a chat's next
turn) across the first bucket's end; and across the sliding ring's wrap at 768."""
import numpy as np
import pytest

from opentpu.llm import prefill as PF
from opentpu.llm.qwen3 import Engine

from test_gemma4 import _cfg, tiny  # noqa: F401 (tiny: the fixture)
from test_prefill import _static


@pytest.mark.parametrize("wf, host, P1, P2", [
    ("int8", "0", 250, 20),     # the second prompt across the bucket's end at 256
    ("fp4", "0", 37, 30),       # 4-bit layers, the int8 head and PLE records
    ("fp4", "1", 37, 30),       # the PLE table on the host (OTPU_PLE_HOST=1)
    ("int8", "0", 760, 20),     # the sliding ring's wrap (768 slots)
], ids=["int8", "fp4", "ple-host", "ring-wrap"])
def test_prompt_runs_are_compile_time_runs(tiny, wf, host, P1, P2, monkeypatch):
    """Two prompts, the second from where the first left: each one's last logits, the layers'
    DRAM (KV caches) and the next step's logits equal those of compile-time runs (compile_rows,
    their tokens compiled in) of the same split; with int8 weights also today's prefill's."""
    _, W, spec = tiny
    monkeypatch.setenv("OTPU_PLE_HOST", host)
    r = np.random.default_rng(P1)
    p1, p2 = ([int(t) for t in r.integers(0, 1000, n)] for n in (P1, P2))
    kw = dict(wformat=wf, head_format="int8" if wf == "fp4" else None)
    a = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True, prompt_runs=True, **kw)
    assert PF.supported(a) and a.image.ple_host == (host == "1")
    got = [a.prefill(p1), a.prefill(p2)]
    R_max = lambda blocks: PF.r_max(a, blocks)
    assert R_max(1) == a.image.rows             # two passes (MCOLS 4): fit_chunk's 8 rows
    b = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True, **kw)
    want = [_static(b, p1, R_max), _static(b, p2, R_max)]
    assert all(np.array_equal(x, y) for x, y in zip(got, want))
    assert [s["rows"] for s in a.stats] == [s["rows"] for s in b.stats]
    img = a.image

    def layers(e):
        return e.backend.machine.slices[0].dram[img.layer0:img.head[0]]
    assert np.array_equal(layers(a), layers(b))
    assert {k[3] for k in a._prompt_progs} == {p // 256 + 1 for p in range(P1 + P2)}
    assert np.array_equal(a.step(5), b.step(5))
    if wf == "int8":
        c = Engine(spec, W, cap=1024, cfg=_cfg(), resident=True, **kw)
        assert all(np.array_equal(x, c.prefill(t)) for x, t in zip(got, (p1, p2)))
