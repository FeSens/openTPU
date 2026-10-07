"""Qwen3 on openTPU: the full decoder (layer loop, QK-norm, GQA, RoPE, tied LM head) against
Hugging Face transformers. A tiny random model always runs; the real Qwen3-0.6B runs when its
checkpoint is in models/Qwen3-0.6B."""
from pathlib import Path

import numpy as np
import pytest

from opentpu.llm.qwen3 import (Engine, Spec, device_config, emulated_logits, load_weights,
                                reference_logits, rope_tables)

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

REAL = Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B"


def _cos(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


@pytest.fixture(scope="module")
def tiny():
    torch.manual_seed(0)
    hc = transformers.Qwen3Config(hidden_size=256, num_hidden_layers=2, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=128, intermediate_size=512,
                                  vocab_size=1000, rms_norm_eps=1e-6, rope_theta=1e6,
                                  tie_word_embeddings=True, max_position_embeddings=4096)
    m = transformers.Qwen3ForCausalLM(hc).float().eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "norm" in n:
                p.copy_(1 + 0.1 * torch.randn_like(p))
    W = {k: v.float().numpy() for k, v in m.state_dict().items()}
    return m, W, Spec(256, 2, 4, 2, 128, 512, 1000)


def test_the_embedding_table_goes_to_the_host_when_it_does_not_fit(tiny):
    """A dense model's int8 embedding table of its own stays on the card (embed_host off),
    unless the image does not fit with it: then the Engine keeps it on the host, the card
    holding a slot of the prefill's rows, with the same logits (8-row prefill, steps); asked
    to keep it on the card, the image does not fit."""
    from dataclasses import replace
    _, W, spec = tiny
    spec = replace(spec, tied=False, embed="int8")
    W = dict(W, **{"lm_head.weight": W["model.embed_tokens.weight"] * np.float32(1.25)})
    big = device_config(spec, 256, rows=8, lookup=True, embed_host=False)
    a = Engine(spec, W, cap=256, cfg=big, rows=8)
    assert a.device_inputs and not a.image.embed_host and a.row_server is None
    host = spec.image(big, 256, 1, 8, lookup=True, embed_host=True).nbytes
    small = replace(big, DRAM_BYTES=-(-host // 4096) * 4096)
    assert small.DRAM_BYTES < a.image.nbytes
    b = Engine(spec, W, cap=256, cfg=small, rows=8)
    assert b.image.embed_host and b.image.nbytes == host
    with pytest.raises(MemoryError):
        Engine(spec, W, cap=256, cfg=small, rows=8, embed_host=False)
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 13)]
    assert np.array_equal(a.prefill(toks).view(np.uint32), b.prefill(toks).view(np.uint32))
    for t in toks[:3]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))


def test_tiny_matches_hf(tiny):
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 140)]  # > one KV block
    with torch.no_grad():
        hf = m(torch.tensor([toks])).logits[0].numpy()
    assert np.abs(reference_logits(spec, W, toks) - hf).max() < 1e-4
    eng = Engine(spec, W, cap=256)
    dev = np.array([eng.step(t) for t in toks])
    assert _cos(dev, hf).min() > 0.998
    # the device follows the quantized math (it differs only in fp32 rounding)
    emu = emulated_logits(spec, W, toks[:12])
    assert _cos(dev[:12], emu).min() > 0.9995


@pytest.mark.parametrize("wformat", ["int4", "fp4"])
def test_tiny_4bit_follows_emulation(tiny, wformat):
    """4-bit weights: the device follows the float64 emulation of the same 4-bit weights."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    eng = Engine(spec, W, cap=256, wformat=wformat)
    dev = np.array([eng.step(t) for t in toks])
    emu = emulated_logits(spec, W, toks, wformat=wformat)
    assert _cos(dev, emu).min() > 0.9995


@pytest.mark.parametrize("wformat", ["int4", "fp4"])
def test_tiny_4bit_column_reuse(tiny, wformat):
    """Column reuse (MCOLS=2, PAIR): every 4-bit decode MM runs PAIR on a QACT DUP operand;
    the logits follow the emulation and differ from the half-rate MXU in fp32 rounding only."""
    from opentpu import isa as I
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(0).integers(0, 1000, 12)]
    runs = {}
    for pair in (False, True):
        cfg = device_config(spec, 256, wformat=wformat, MCOLS=2, PAIR=pair)
        eng = Engine(spec, W, cap=256, cfg=cfg, wformat=wformat)
        runs[pair] = np.array([eng.step(t) for t in toks])
        prog = eng.backend.machine.slices[0].prog
        mm4 = [p for p in prog if p.op == I.MM and (p.flags >> I.WF_SHIFT) & 3]
        assert mm4 and all(bool(p.flags & I.F_PAIR) == pair for p in mm4)
        # the attention's query groups (two rows) fill both columns: no DUP
        assert all(bool(p.flags & I.F_DUP) == (pair and p.w[1] & 0xFF == 1)
                   for p in prog if p.op == I.QACT)
    emu = emulated_logits(spec, W, toks, wformat=wformat)
    assert _cos(runs[True], emu).min() > 0.9995
    assert _cos(runs[True], runs[False]).min() > 0.99999


def test_tiny_resident_decode_is_bit_exact(tiny):
    """Resident decode (compile_decode: one program per attention bucket, the token and
    position as run arguments) gives the per-position programs' logits bit for bit, from
    position 0 and across the bucket boundary 256, after a chunked prefill."""
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 262)]
    a = Engine(spec, W, cap=512, resident=True)
    b = Engine(spec, W, cap=512)
    assert a.resident
    for t in toks[:3]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert np.array_equal(a.prefill(toks[3:252]), b.prefill(toks[3:252]))
    for t in toks[252:]:
        pa = a.pos
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32)), pa
    assert sorted(a._decodes) == [1, 2]


def test_tiny_device_inputs(tiny, monkeypatch):
    """With the image's tables (resident decode's lookup tables), prefill runs and
    per-position steps read their embedding and RoPE rows from the image too (the token ids
    compiled into their programs): the host writes no inputs, only the run's arguments, and the
    logits are the host-input engine's bit for bit. Without the resident program (a backend
    without run arguments) every step is such a per-position program."""
    from opentpu.llm.qwen3 import IsaBackend
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(2).integers(0, 1000, 20)]
    monkeypatch.setattr(IsaBackend, "args", False)      # no resident program: per-position
    a = Engine(spec, W, cap=256, resident=True)
    b = Engine(spec, W, cap=256)
    assert a.device_inputs and not a.resident and not b.device_inputs
    writes = []
    wr = a.backend.write
    a.backend.write = lambda s, addr, v: (writes.append(addr), wr(s, addr, v))
    for t in toks[:2]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert np.array_equal(a.prefill(toks[2:17], chunk=6), b.prefill(toks[2:17], chunk=6))
    for t in toks[17:]:
        assert np.array_equal(a.step(t).view(np.uint32), b.step(t).view(np.uint32))
    assert writes == []
    assert [s.get("rows", 1) for s in a.stats] == [1, 1, 6, 6, 3, 1, 1, 1]


def test_tiny_prefill_runs_take_whole_mxu_passes(tiny):
    """A default prefill run of more than MCOLS rows takes whole passes of MCOLS rows (each
    pass streams every weight: 7 rows would cost what 8 do); an explicit chunk is kept."""
    from opentpu.isasim import board_config
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 15)]
    cfg = board_config(DRAM_BYTES=1 << 24, MCOLS=4)
    a, b = Engine(spec, W, cap=256, cfg=cfg), Engine(spec, W, cap=256, cfg=cfg)
    assert np.array_equal(a.prefill(toks), b.prefill(toks, chunk=7))
    assert [s.get("rows", 1) for s in a.stats] == [8, 4, 3]
    assert [s.get("rows", 1) for s in b.stats] == [7, 7, 1]


def test_tiny_reset_reuses_cache(tiny):
    _, W, spec = tiny
    eng = Engine(spec, W, cap=128)
    a = [eng.step(t) for t in (5, 6, 7)]
    eng.reset()
    b = [eng.step(t) for t in (5, 6, 7)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
def test_qwen3_0_6b_greedy_matches_hf():
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    with torch.no_grad():
        want = hf.generate(torch.tensor([ids]), max_new_tokens=8, do_sample=False)[0, len(ids):]
    eng = Engine(Spec.from_hf(REAL), load_weights(REAL), cap=256)
    got = eng.generate(ids, max_new=8)
    assert got == want.tolist()[:len(got)] and len(got) >= 7
    assert tok.decode(got).startswith("The capital of France is Paris.")


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
@pytest.mark.parametrize("pair", [False, True])
def test_qwen3_0_6b_fp4_greedy(pair):
    """Qwen3-0.6B with 4-bit (E2M1, two-level scales) layer weights and an int8 LM head on the
    ISA simulator: the greedy answer is still right, and every generated token is the argmax of
    the float64 emulation of the same 4-bit weights (the device follows the quantized math).
    pair: the board's MCOLS=2 with column reuse (full-rate 4-bit MMs)."""
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    msgs = [{"role": "user", "content": "What is the capital of France? Answer in one sentence."}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False,
                                  tokenize=True)
    ids = list(ids["input_ids"] if hasattr(ids, "keys") else ids)
    spec, W = Spec.from_hf(REAL), load_weights(REAL)
    cfg = device_config(spec, 256, wformat="fp4", head_format="int8", MCOLS=2, PAIR=True) \
        if pair else None
    eng = Engine(spec, W, cap=256, cfg=cfg, wformat="fp4", head_format="int8")
    got = eng.generate(ids, max_new=8)
    assert tok.decode(got).startswith("The capital of France is Paris.")
    emu = emulated_logits(spec, W, ids + got[:-1], wformat="fp4", head_format="int8")
    assert emu[len(ids) - 1:].argmax(-1).tolist() == got


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
@pytest.mark.parametrize("wformat,head_format,pair", [("int8", None, False),
                                                      ("fp4", "int8", False),
                                                      ("fp4", None, True)])
def test_qwen3_0_6b_token_on_rtl_is_bit_exact(wformat, head_format, pair):
    """Feed part of a prompt on the ISA simulator, then run the next token on the Verilator RTL
    and on the ISA simulator from the same DRAM state: weights, KV cache and logits must agree
    bit for bit (int8 weights; 4-bit layers with an int8 LM head; everything 4-bit on the
    board's MCOLS=2 with column reuse)."""
    from opentpu.llm.rtl_backend import RtlBackend
    spec = Spec.from_hf(REAL)
    cfg = device_config(spec, 256, wformat=wformat, head_format=head_format, MCOLS=2,
                        PAIR=True) if pair else None
    eng = Engine(spec, load_weights(REAL), cap=256, cfg=cfg, wformat=wformat,
                 head_format=head_format)
    prompt = [151644, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30]
    for t in prompt[:-1]:
        eng.step(t)
    n = eng.image.nbytes
    rtl = RtlBackend(eng.cfg, [s.dram[:n] for s in eng.backend.machine.slices])
    isa = eng.backend
    want = eng.step(prompt[-1])
    eng.backend, eng.pos = rtl, eng.pos - 1
    got = eng.step(prompt[-1])
    assert np.array_equal(want.view(np.uint32), got.view(np.uint32))
    for s in range(eng.cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n])


def test_tiny_chunked_prefill_is_bit_exact(tiny):
    """Prefill in chunks (P prompt rows per run, causal over the cache and the chunk) gives the
    same logits and KV cache as token-by-token decode; decoding continues identically."""
    m, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(1).integers(0, 1000, 23)]
    ref = Engine(spec, W, cap=256)
    want = [ref.step(t) for t in toks]
    eng = Engine(spec, W, cap=256, rows=4)
    got = eng.prefill(toks[:21], chunk=4)
    assert np.array_equal(got, want[20]) and eng.pos == 21
    assert all(np.array_equal(eng.step(t), w) for t, w in zip(toks[21:], want[21:]))
    with torch.no_grad():
        hf = m(torch.tensor([toks[:21]])).logits[0, -1].numpy()
    assert _cos(got, hf) > 0.998


def test_tiny_batched_decode_matches_separate_runs(tiny):
    """b sequences decoded together (each weight stream shared by b rows, one KV cache and
    position per sequence) equal b separate runs bit for bit."""
    _, W, spec = tiny
    rng = np.random.default_rng(2)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (5, 12, 1)]
    eng = Engine(spec, W, cap=128, batch=3)
    got = eng.generate_batch(prompts, max_new=4, chunk=3)
    for s, p in enumerate(prompts):
        assert got[s] == Engine(spec, W, cap=128).generate(p, max_new=4)
    lg = eng.step_batch([7, 8])                 # a subset; the last generated token was not fed
    for s, t in enumerate((7, 8)):
        ref = Engine(spec, W, cap=128)
        ref.prefill(prompts[s] + got[s][:-1])
        assert np.array_equal(lg[s], ref.step(t))


def test_tiny_batch_goes_on_when_a_sequence_fills_the_cache(tiny):
    """A sequence that reaches the cap ends (as Engine.generate) and leaves the batch: the
    others decode on, each equal to its separate run; a finished sequence is fed no more (its
    position stays after its last token fed). Before, the next step fed every row and the full
    sequence's raised "KV cache full"."""
    _, W, spec = tiny
    rng = np.random.default_rng(5)
    prompts = [[int(t) for t in rng.integers(0, 1000, n)] for n in (126, 3)]
    eng = Engine(spec, W, cap=128, batch=2, rows=8)
    got = eng.generate_batch(prompts, max_new=6, chunk=8)
    for s, p in enumerate(prompts):
        assert got[s] == Engine(spec, W, cap=128, rows=8).generate(p, max_new=6)
    assert [len(g) for g in got] == [3, 6] and eng.poss == [128, 8]


@pytest.mark.skipif(not REAL.exists(), reason="models/Qwen3-0.6B not downloaded")
def test_qwen3_0_6b_chunked_prefill_and_batch_match_hf():
    """Real weights: chunked prefill then decode gives HF's greedy tokens, and two prompts
    decoded as a batch give the same tokens as separate runs."""
    tok = transformers.AutoTokenizer.from_pretrained(REAL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(REAL, dtype=torch.float32).eval()
    prompts = []
    for q in ("What is the capital of France? Answer in one sentence.", "Name a prime number."):
        ids = tok.apply_chat_template([{"role": "user", "content": q}],
                                      add_generation_prompt=True, enable_thinking=False,
                                      tokenize=True)
        prompts.append(list(ids["input_ids"] if hasattr(ids, "keys") else ids))
    W, spec = load_weights(REAL), Spec.from_hf(REAL)
    eng = Engine(spec, W, cap=256, batch=2, rows=8)
    got = eng.generate_batch(prompts, max_new=6, chunk=8)
    for s, ids in enumerate(prompts):
        with torch.no_grad():
            want = hf.generate(torch.tensor([ids]), max_new_tokens=6,
                               do_sample=False)[0, len(ids):].tolist()
        assert got[s] == want[:len(got[s])] and len(got[s]) >= 5
    assert got[1] == Engine(spec, W, cap=256).generate(prompts[1], max_new=6)


def test_tiny_vt_tiles_bit_exact(tiny, monkeypatch):
    """A 512-token cache holds V^T in tiles of 256 tokens (compiler.KVDesc): a prefill chunk
    across the tile edge, then decode past it, give the same logits bit for bit as the plain
    [d, cap] layout."""
    import opentpu.compiler as C
    _, W, spec = tiny
    toks = [int(t) for t in np.random.default_rng(3).integers(0, 1000, 264)]

    def run():
        eng = Engine(spec, W, cap=512, rows=8)
        eng.step(toks[0])
        out = [eng.prefill(toks[1:257], chunk=8)]          # rows 249..256 cross the edge
        out += [eng.step(t) for t in toks[257:]]
        return out
    assert C.vt_tile(512) == 256
    tiled = run()
    monkeypatch.setattr(C, "VT_TILE", 1 << 30)
    plain = run()
    assert all(np.array_equal(a.view(np.uint32), b.view(np.uint32)) for a, b in zip(tiled, plain))


@pytest.mark.parametrize("mcols,wformat", [(2, "int8"), (4, "int8"), (2, "fp4")])
def test_tiny_prefill_chunk_with_mm_replay_on_rtl(tiny, have_verilator, mcols, wformat):
    """An 8-row prefill run at the board configuration with 8 ACT RAM rows (each projection is
    one MM that replays every weight chunk for 8 / MCOLS groups) on the RTL's board memory
    path: DRAM and TMEM equal the ISA simulator's, and the logits equal decode's. fp4: 4-bit
    layers (two blocks a replayed chunk) with an int8 head, on a PAIR board (decode rows pair,
    the 8-row MMs replay)."""
    from opentpu import rtlsim
    from opentpu.isasim import Machine, board_config
    _, W, spec = tiny
    wkw = dict(wformat=wformat, head_format="int8") if wformat != "int8" else {}
    toks = [int(t) for t in np.random.default_rng(4).integers(0, 1000, 11)]
    ref = Engine(spec, W, cap=256, **wkw)
    want = ref.prefill(toks, chunk=1)
    cfg = board_config(MCOLS=mcols, ACT_ROWS=8, DRAM_BYTES=1 << 24, PAIR=wformat != "int8")
    eng = Engine(spec, W, cap=256, cfg=cfg, **wkw)
    eng.prefill(toks[:3])                         # a cache to attend over
    rows = [(0, 3 + j) for j in range(8)]
    progs = eng.image.compile_rows(rows, [7])
    assert any(i.op == 0x20 and (i.w[5] >> 16) & 0xFF == 8 for i in progs[0])
    x = eng.embed[toks[3:]]
    io = eng.image.io
    cs = [np.stack(a).astype(np.float32) for a in zip(*[rope_tables(spec, p) for _, p in rows])]
    dram = eng.backend.machine.slices[0].dram.copy()
    for k, v in (("x", x), ("cos", cs[0]), ("sin", cs[1])):
        b = np.ascontiguousarray(v, np.float32).view(np.uint8).reshape(-1)
        dram[io[k]:io[k] + b.size] = b
    m = Machine(cfg, progs, [dram.copy()]).run()
    drams, tmems, _ = rtlsim.run(cfg, progs, [dram.copy()], uarch=rtlsim.BOARD_UARCH, axi=True,
                                 boot=True)
    assert np.array_equal(drams[0][:eng.image.nbytes], m.slices[0].dram[:eng.image.nbytes])
    v = spec.vocab
    lg = drams[0][io["logits"] + 4 * 7 * v:io["logits"] + 4 * 8 * v].view(np.float32)
    assert np.array_equal(lg, want)
