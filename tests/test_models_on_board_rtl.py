"""The three real checkpoints in the board's 4-bit configuration (4-bit layers, int8 LM head,
column reuse) on the board's MCOLS=2 and on MCOLS=4: a token on the Verilator RTL, through the
board's memory path (AXI adapter, boot loader), equals the ISA simulator bit for bit. Slow
(several minutes per case): run with --runslow."""
from pathlib import Path

import numpy as np
import pytest

from opentpu import rtlsim
from opentpu.isasim import board_config
from opentpu.llm import load_spec
from opentpu.llm.qwen3 import Engine, load_weights

MODELS = Path(__file__).resolve().parent.parent / "models"
CASES = [("Qwen3-0.6B", [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30], False, False),
         ("LFM2.5-230M", [1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481], False, False),
         ("LFM2.5-230M", [1, 6, 6423, 708, 3493, 856, 779, 5706, 803, 4481], False, True),
         ("Qwen3.5-0.8B", [760, 6511, 314, 9338, 369], False, False),
         ("Qwen3.5-0.8B", [760, 6511, 314, 9338, 369], True, False)]


@pytest.mark.slow
@pytest.mark.parametrize("mcols", [2, 4])
@pytest.mark.parametrize("model,prompt,dstep,resident", CASES,
                         ids=["qwen3", "lfm2", "lfm2-resident", "qwen35", "qwen35-dstep"])
def test_token_on_board_rtl_is_bit_exact(have_verilator, model, prompt, dstep, resident, mcols):
    """Feed part of a prompt on the ISA simulator, then run the next token on the RTL and on the
    ISA simulator from the same DRAM state: logits and the whole DRAM image (weights, caches,
    recurrent state) must agree bit for bit (the Qwen3.5 DeltaNet with and without DSTEP; LFM2
    also with the resident decode program)."""
    from opentpu.llm.rtl_backend import RtlBackend
    path = MODELS / model
    if not path.exists():
        pytest.skip(f"models/{model} not downloaded")
    spec = load_spec(path)
    cfg = board_config(DRAM_BYTES=1 << 30, MCOLS=mcols, PAIR=True, DSTEP=dstep)
    eng = Engine(spec, load_weights(path), cap=256, cfg=cfg, wformat="fp4", head_format="int8",
                 resident=resident)
    assert eng.resident == resident
    for t in prompt[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices],
                     uarch=rtlsim.BOARD_UARCH, axi=True, boot=True)
    isa = eng.backend
    want = eng.step(prompt[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(prompt[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])
