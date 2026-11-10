"""The expert slots of the MoE models other than GPT-OSS keep their layout (ExpertFormat without
tight: moe.ExpertFormat's offsets, parts, descriptors, pool key and packed bytes) as bcu-wg
5d76fd38 had it, for Qwen3.5 / 3.6-35B-A3B and MiniCPM-V 4.7 (H 2048, F 512), Gemma 4 26B-A4B
(2816, 704: the slot's 768) and LFM2-8B-A1B (2048, 1792), in every weight format at WG 1 and 8.
GOLDEN holds that code's digests (digest() run on it)."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from opentpu.llm import moe as MO

SHAPES = {"qwen35-35b": (2048, 512), "gemma4-26b": (2816, 704), "lfm2-8b": (2048, 1792)}
FORMATS = ("int8", "fp4", "nvfp4", "int4")


def digest(H: int, F: int, wformat: str, grp: int, pack: bool = True) -> str:
    """A hash of ExpertFormat(H, F, 128, wformat, grp)'s layout: its widths, chunk, offsets,
    data runs, QACT ranges, the descriptors of a slot at 1 MiB, the pool key's layout and
    format, and (pack) the bytes of a seeded random expert."""
    try:
        f = MO.ExpertFormat(H, F, 128, wformat, grp)
    except ValueError as e:
        return f"refused: {e}"
    ns = f.descs(1 << 20)

    def q(t):
        return [t.data.const, None if t.scale is None else t.scale.const, list(t.shape), t.rs,
                t.srs, t.pw, t.k0, t.grp, [q(p) for p in t.parts or ()]]
    img = SimpleNamespace(fmt=f, offload=SimpleNamespace(layers=2, E=4, slot_bytes=f.nbytes))
    key = MO.pool_key(img)
    rec = [f.H, f.F, f.F0, f.C, f.D, f.wformat, f.grp, list(f.wg), list(f.wu),
           [list(x) for x in f.wd], f.nbytes, [list(x) for x in f.data], f.ranges(),
           q(ns.wg), q(ns.wu), q(ns.wd), key["layout"], key["format"], key["D"], key["key"]]
    h = hashlib.blake2b(json.dumps(rec).encode(), digest_size=12)
    if pack:
        r = np.random.default_rng(H + F + grp)

        def w(shape):       # exact on every host (a normal's tail goes through the C library's log)
            return (r.integers(-1 << 15, 1 << 15, shape, np.int16).astype(np.float32)
                    * np.float32(2.0 ** -18))
        wg, wu, wd = w((F, H)), w((F, H)), w((H, F))
        h.update(f.pack(wg, wu, wd).tobytes())
    return h.hexdigest()


GOLDEN = {                   # digest() at bcu-wg 5d76fd38
    ("gemma4-26b", 1, "int8"): "a3d7ac4d27a2fae54b325da3",
    ("gemma4-26b", 1, "fp4"): "3371d4528bde679f363efbbf",
    ("gemma4-26b", 1, "nvfp4"): "6fc610ee76fb72e83b03c2c0",
    ("gemma4-26b", 1, "int4"): "a9b0d1f60470aa6917981f07",
    ("gemma4-26b", 8, "int8"): "b5f8c5c9bb39693c5d776af7",
    ("gemma4-26b", 8, "fp4"): "7b23855867222381bbee2f7b",
    ("gemma4-26b", 8, "nvfp4"): "f49eb6084caa6403befc5d98",
    ("gemma4-26b", 8, "int4"): "dba584892a952255bdd72bcb",
    ("lfm2-8b", 1, "int8"): "ac68171c372391839baab5fc",
    ("lfm2-8b", 1, "fp4"): "2c93491e53c809bf4781d0cf",
    ("lfm2-8b", 1, "nvfp4"): "ca4a135653463cf1ebd319c1",
    ("lfm2-8b", 1, "int4"): "41e3dc14a0bf6ce7875eb47d",
    ("lfm2-8b", 8, "int8"): "c245b6afd1eb63cc46cc33d3",
    ("lfm2-8b", 8, "fp4"): "1190a68c2993badc42d5c023",
    ("lfm2-8b", 8, "nvfp4"): "b131725b24524deabf94d232",
    ("lfm2-8b", 8, "int4"): "ab5f6f3fbde9c6224a8669d2",
    ("qwen35-35b", 1, "int8"): "fe81a8b0857db40c3bee45a1",
    ("qwen35-35b", 1, "fp4"): "c6ff85ac838ea1c274a31b7c",
    ("qwen35-35b", 1, "nvfp4"): "8a2b492c186c96d0d42a9a79",
    ("qwen35-35b", 1, "int4"): "b54338dcc40295dd85388037",
    ("qwen35-35b", 8, "int8"): "24fc2705216cdd8cdef3b497",
    ("qwen35-35b", 8, "fp4"): "5d9180395bd29c3388f829b6",
    ("qwen35-35b", 8, "nvfp4"): "570cc49f664b639da65a3a1c",
    ("qwen35-35b", 8, "int4"): "d72675916ba54c6bbdc18685",
}


@pytest.mark.parametrize("model", sorted(SHAPES))
@pytest.mark.parametrize("grp", [1, 8])
def test_untight_layouts_are_bcu_wgs(model, grp):
    H, F = SHAPES[model]
    got = {w: digest(H, F, w, grp) for w in FORMATS}
    assert got == {w: GOLDEN[model, grp, w] for w in FORMATS}
