"""The resident decode's embedding row gathered on the device (Engine(embed="gather"),
Image(lookup="gather"), kernels/gather.py) instead of read from an fp32 table: from the tied LM
head on one slice, from an int8 table of the embedding on two."""
import dataclasses

import numpy as np
import pytest

from opentpu import quant as Q
from opentpu.kernels import gather as GA
from opentpu.llm.qwen3 import PREFILL_ROWS, Engine, device_config

from test_autodecode import _tiny

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.fixture(scope="module", params=["qwen3", "lfm2", "qwen35"])
def tiny(request):
    return (request.param,) + _tiny(request.param)


@pytest.mark.parametrize("S,head_format", [(1, "int8"), (1, "fp4"), (2, "int8")])
def test_gathered_embedding(tiny, S, head_format):
    """Resident decode with the gathered row gives the per-position programs' logits bit for
    bit (the host writes the gather's values), across the attention bucket boundary; the
    generate loop gives the host loop's tokens; the image holds no fp32 table (one slice: no
    table at all, the head is read)."""
    name, W, spec = tiny
    kw = dict(rows=PREFILL_ROWS, S=S, head_format=head_format)
    cfg = device_config(spec, 512, lookup="gather", **kw)
    a = Engine(spec, W, cap=512, cfg=cfg, resident=True, embed="gather",
               head_format=head_format)
    b = Engine(spec, W, cap=512, cfg=cfg, embed="gather", head_format=head_format)
    lk = a.image.lookup
    assert a.resident and "embed" not in lk and (lk["egather"]["table"] is None) == (S == 1)
    fp32 = spec.image(dataclasses.replace(cfg, DRAM_BYTES=1 << 30), 512, 1, PREFILL_ROWS,
                      "int8", head_format, lookup=True)
    V, H, D, M, g = spec.vocab, spec.hidden, cfg.D, cfg.MCOLS, lk["egather"]
    added = 4 * M * D * GA.onehot_blocks(D, M, g["fmt"]) + (
        0 if S == 1 else V * Q.row_bytes(H, "int8", D) + 4 * V * (H // D))
    assert abs((fp32.nbytes - a.image.nbytes) - (4 * V * H - added)) < 4096
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 252)]
    assert np.array_equal(a.prefill(toks).view(np.uint32), b.prefill(toks).view(np.uint32))
    ref, t = [], int(np.argmax(b.step(toks[-1])))
    a.step(toks[-1])
    for _ in range(8):                              # past 256: the next bucket
        ref.append(t)
        la, lb = a.step(t), b.step(t)
        assert np.array_equal(la.view(np.uint32), lb.view(np.uint32))
        t = int(np.argmax(lb))
    if a.can_generate:
        c = Engine(spec, W, cap=512, cfg=cfg, resident=True, embed="gather",
                   head_format=head_format)
        t0 = int(np.argmax(c.prefill(toks + [toks[-1]])))
        assert t0 == ref[0]
        assert c.generate_card(t0, 7, stop_ids=[]) == ref[1:]
