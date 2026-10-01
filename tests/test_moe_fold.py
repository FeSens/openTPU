"""opentpu/llm/moe.py's Gemma 4 folds (docs/offload.md section 11.2): the router's scales into
its weights, so that it reads the residual's unit RMSNorm (moe_ffn's), and the per-expert scale
into W_down; the experts read that norm times pre_feedforward_layernorm_2's gain (moe_ffn's
g_exp), as against transformers' Gemma4TextRouter and Gemma4TextExperts."""
import numpy as np
import pytest

from opentpu.llm import moe as MO

torch = pytest.importorskip("torch")
g4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
transformers = pytest.importorskip("transformers")


def _gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


def test_gemma_folds_match_hf():
    """Gemma 4's MoE block from the folded weights (moe.gemma_router, moe.gemma_expert; a
    width of 176, not a whole number of chunks), the experts on the norm times g2, and the
    softmax rule picks the same experts and gives transformers' output in fp32."""
    torch.manual_seed(0)
    H, E, k, F = 256, 8, 2, 176
    cfg = transformers.Gemma4TextConfig(hidden_size=H, num_experts=E, top_k_experts=k,
                                        moe_intermediate_size=F, enable_moe_block=True,
                                        hidden_activation="gelu_pytorch_tanh",
                                        rms_norm_eps=1e-6)
    router, experts = g4.Gemma4TextRouter(cfg), g4.Gemma4TextExperts(cfg)
    norm2 = g4.Gemma4RMSNorm(H, eps=1e-6)
    with torch.no_grad():
        for p in (*router.parameters(), *experts.parameters()):
            p.copy_(0.1 * torch.randn_like(p))
        router.scale.copy_(1 + 0.3 * torch.randn(H))
        router.per_expert_scale.copy_(1 + 0.3 * torch.randn(E))
        norm2.weight.copy_(1 + 0.3 * torch.randn(H))
        r = torch.randn(12, H)
        _, w, idx = router(r)
        hf = experts(norm2(r), idx, w).numpy()
    p = "model.layers.3."
    W = {p + "router.proj.weight": router.proj.weight, p + "router.scale": router.scale,
         p + "router.per_expert_scale": router.per_expert_scale,
         p + "experts.gate_up_proj": experts.gate_up_proj,
         p + "experts.down_proj": experts.down_proj,
         p + "pre_feedforward_layernorm_2.weight": norm2.weight}
    W = {n: v.detach().numpy() for n, v in W.items()}
    mo = MO.MoESpec(E=E, k=k, ffn=F, rule="softmax", act="gelu_tanh")
    x = r.numpy()
    xu = x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    xe = xu * W[p + "pre_feedforward_layernorm_2.weight"]
    lg = xu @ MO.gemma_router(W, p).T
    out = np.zeros_like(x)
    for t in range(len(x)):
        ids, wt = MO.route(lg[t], None, mo)
        assert list(ids) == idx[t].tolist()
        for e, we in zip(ids, wt):
            wg, wu, wd = MO.gemma_expert(W, p, e)
            out[t] += np.float32(we) * ((_gelu_tanh(xe[t] @ wg.T) * (xe[t] @ wu.T)) @ wd.T)
    assert np.allclose(out, hf, rtol=1e-4, atol=1e-5), np.abs(out - hf).max()
    fmt = MO.ExpertFormat(H, F, 128, "fp4")
    assert (fmt.F0, fmt.F) == (176, 256) and fmt.pack(*MO.gemma_expert(W, p, 0)).size == \
        fmt.nbytes
