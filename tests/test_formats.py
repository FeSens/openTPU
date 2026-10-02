"""opentpu/llm/formats.py: the weight-format rules every model's image reads."""
import pytest

from opentpu.llm import formats as FM

KINDS = ("attn", "mlp", "head")


def test_rules_and_pick(monkeypatch):
    r = FM.rules("mlp=fp4, attn@2-3=int4,mlp@5=int8,attn=fp4,head=int8,mlp=int4", KINDS)
    assert r[1] == ("attn", 2, 3, True, "int4") and r[2] == ("mlp", 5, 5, True, "int8")
    assert [FM.pick(r, "attn", i, "int8") for i in range(5)] == ["fp4", "fp4", "int4", "int4", "fp4"]
    assert [FM.pick(r, "mlp", i, "int8") for i in (4, 5, 6)] == ["fp4", "int8", "fp4"]    # the first plain mlp
    assert FM.plain(r) == {"mlp": "fp4", "attn": "fp4", "head": "int8"}
    assert FM.pick([], "attn", 0, "int8") == "int8"
    monkeypatch.setenv("OTPU_FORMATS", "attn=fp4")
    assert FM.rules(None, KINDS, "mlp=fp4") == [("attn", 0, 1 << 30, False, "fp4")]
    assert FM.rules("", KINDS, "mlp=fp4") == []
    monkeypatch.delenv("OTPU_FORMATS")
    assert FM.plain(FM.rules(None, KINDS, "mlp=fp4")) == {"mlp": "fp4"}


@pytest.mark.parametrize("bad", ["ple=fp4", "mlp=fp8", "mlp@a-2=fp4", "head@0-1=fp4", "mlp"])
def test_rules_reject(bad):
    with pytest.raises(ValueError, match="weight format"):
        FM.rules(bad, KINDS)


def test_named_mix(monkeypatch):
    """wformat "mix": int8 with the model's mix (Spec.mix), unless a formats string is given;
    other wformats are themselves; a model without a mix refuses it."""
    from types import SimpleNamespace
    monkeypatch.delenv("OTPU_FORMATS", raising=False)
    spec = SimpleNamespace(mix="attn=fp4,mlp@8-23=fp4")
    assert FM.named(spec, "mix", None) == ("int8", "attn=fp4,mlp@8-23=fp4")
    assert FM.named(spec, "mix", "attn=int4") == ("int8", "attn=int4")
    assert FM.named(spec, "fp4", None) == ("fp4", None)
    monkeypatch.setenv("OTPU_FORMATS", "down=fp4")
    assert FM.named(spec, "mix", None) == ("int8", "down=fp4")
    monkeypatch.delenv("OTPU_FORMATS")
    with pytest.raises(ValueError, match="no recommended mix"):
        FM.named(SimpleNamespace(mix=""), "mix", None)
    assert FM.mix_for({"model_type": "none"}) == ""
    assert FM.auto(spec, "auto") == "mix" and FM.auto(SimpleNamespace(mix=""), "auto") == "int8"
    assert FM.auto(SimpleNamespace(), "auto") == "int8" and FM.auto(spec, "fp4") == "fp4"


def test_mixes():
    """The recommended mixes (docs/formats.md): found by their config, the head int8, at most
    two layer layouts (Qwen3.5: one, its second run does not fit the IMEM)."""
    from opentpu.llm import lfm2, qwen3, qwen35
    fam = {"qwen3_5_text": (qwen35, 1), "lfm2": (lfm2, 2)}
    assert FM.MIXES
    for (mt, layers, hidden, vocab), mix in FM.MIXES.items():
        assert FM.mix_for({"model_type": mt, "num_hidden_layers": layers, "hidden_size": hidden,
                           "vocab_size": vocab}) == mix
        M, most = fam.get(mt, (qwen3, 2))
        f = FM.resolver(mix, M.KINDS, "", "int8")
        assert f("head") == "int8"
        kinds = [k for k in M.KINDS if k not in ("mlp", "head")]
        assert len({tuple(f(k, i) for k in kinds) for i in range(layers)}) <= most
