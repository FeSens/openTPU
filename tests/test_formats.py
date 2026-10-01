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
