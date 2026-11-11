"""A layer's runs of a prefill chunk in one device run (the layer loop,
docs/pf500.md): the layer-major prefill's run of n rows (qwen35_layer_run, or the
expert-major tiled run tiled.qwen35_layer_tile; either unchanged) in a hardware loop over the
chunk's runs in one attention bucket, so the host starts a layer's runs in a bucket once instead
of once a run.

    FILL the loop's state words from the run arguments tpos0 and row0 (2^23 + it, then - 2^23)
    LOOP runs times (the argument):
        each run-time argument c * var of the run: RLD MUL of var's state word into its
            argument register (generate's run_words: the registers the host wrote before a run)
        the run: n rows from chunk row `row` at the bucket's position tpos
        tpos += n, row += n
    HALT

Bit for bit the runs one after another (the same body, the same argument values in order). The
host's arguments: tpos0 (the first run's tpos), row0 (its chunk row), runs (how many runs). A
body whose run-time arguments are other than tpos and row (a token id, a ring row) does not loop.
The embed runs (a token's row into the chunk, qwen35_embed_run: its token an argument) loop as
qwen35_embed_out, the token from the generate area's out[] (prefill.write_tokens: the host
writes the loop's tokens there before it, as the prompt runs' docs/prefill.md 3).
"""
from __future__ import annotations

from .. import isa as I
from .. import language as ol
from ..compiler import CompileError, RunVar, current, mul_bound
from .moe import _TWO23
from .qwen3 import RunPos, RunRows, _embed_word
from .qwen35 import ATTN_BLOCK, qwen35_layer_run

STATE = ("tpos", "row")         # the loop's state words, in this order
LOOP_ARGS = ("tpos0", "row0", "runs")


@ol.jit
def layer_loop(body, kw: dict, pos, row, n: int):
    """`runs` runs of body.fn(pos=pos, row=row, **kw), n rows each: run k at tpos0 + k n and
    chunk row row0 + k n, every row in the bucket."""
    b = current()
    t0 = RunVar("tpos0", pos.tpos.bound)
    r0 = RunVar("row0", row.bound)
    count = RunVar("runs", row.bound // n + 1)
    st = ol.empty((len(STATE),))
    for w, v in enumerate((t0, r0)):
        b.emit(I.vop(I.V_FILL, st.base + w, 0, 0, 1, 1, 0, 0, 0, I.B_SCALAR, _TWO23,
                     rd=b.arg_reg(v, 1), comment=f"{v.name} (2^23 + it)"))
    st.set(st - _TWO23)
    for v in (t0, r0):                  # (their registers serve addresses from here)
        ol.release(v)
    loop = b.begin_loop(0, rcount=b.arg_reg(count, 1))
    body_ins = b.stack[-1]
    mark = len(body_ins)                # the argument loads go here (known after the run)
    b.run_words = {name: st.base + w for w, name in enumerate(STATE)}
    body.fn(pos=pos, row=row, **kw)
    st.set(st + float(n))
    other = sorted({v.name for v, _ in b.run_args} - set(STATE) - set(LOOP_ARGS))
    if other:
        raise CompileError(f"a layer loop's run takes run-time arguments {other}: not looped")
    bound = {"tpos": pos.tpos.bound, "row": row.bound}
    body_ins[mark:mark] = [x for v, c in list(b.run_args) if v.name in b.run_words
                           for x in I.rld_mul(b.arg_reg(v, c), b.run_words[v.name], int(c),
                                              bound=mul_bound(c, bound[v.name], b.aw),
                                              comment=f"argument {c}*{v.name}", aw=b.aw)]
    b.run_words = None
    b.end_loop(loop)


@ol.jit
def qwen35_embed_out(m, pos, row, em: bool = False):
    """qwen35_embed_run's row with its token from out[] instead of an argument: the embedding
    row of out[t0 + tpos] -> m.xbuf[row] (em: its record's X); the same gather (_embed_word)."""
    from .moe import em_x
    X = em_x(m.moe_dev, row, 1, m.spec.hidden) if em else m.xbuf[row:row + 1, :]
    tk = ol.load(m.gen.out[pos.pos:pos.pos + 1])
    ol.store(X, _embed_word(m, tk))
    del tk


def compile_layer_loop(image, li: int, blocks: int, block: int = ATTN_BLOCK, n: int = 1,
                       tile: bool = False, em: bool = False, slot: int = 0):
    """(programs, run_args): layer_loop over layer li's runs of n rows in bucket `blocks` (from
    conv_k - 1): the image's compile_layer_run(li, blocks, R=n, embedded=True, em=, slot=)
    body (a program per slice), or with tile tiled.compile_layer_tile(li, blocks, n, slot=)'s
    (a program per slice; its first form that fits, tiled._first_fit). li < 0: the embed runs'
    (qwen35_embed_out, a row a run, the tokens in out[])."""
    if not image.lookup or not image.prefill_rows:
        raise ValueError("a layer loop needs lookup tables and the image's prefill rows")
    cfg, lk, row = image.cfg, image.lookup, RunVar("row", image.prefill_rows)
    lo = max((blocks - 1) * block, getattr(image.spec, "conv_k", 1) - 1 if li >= 0 else 0)

    def m(s):
        return image.descriptors(s, slot) if slot else image.descriptors(s)

    if tile:
        from . import tiled as TL
        fam = TL.layer_family(image)
        if fam is None or cfg.S != 1 and image.offload is not None:
            raise ValueError("a tiled layer loop is an image's with tiled runs (S > 1: a "
                             "resident pool's)")
        pos = RunRows(blocks, block, lo, lk.get("zmask", 0), image.cap, n, 0,
                      (lk["amask"], 4 * lk["amask_M"] * (block + 1)))

        def trace(kw, fit):
            return [layer_loop.trace(cfg, s, {
                "body": TL.LAYER_KERNELS[fam], "pos": pos, "row": row, "n": n,
                "kw": {"m": m(s), "li": li, "T": n, "block": block, **kw}}, fit=fit)
                for s in range(cfg.S)]

        return TL._first_fit(trace, TL.layer_variants(image, fam, n), cfg)
    pos = RunPos(blocks, block, lo, lk["zmask"], image.cap)
    pos.tpos.bound -= n - 1             # the run's last row in the block: tpos <= block - n
    if li < 0:
        if n != 1 or getattr(image, "embed_host", False) or "gen" not in lk:
            raise ValueError("an embed loop takes a row a run, its tokens from out[] (the "
                             "image's generate area, the table on the card)")
        body, kw = qwen35_embed_out, {"em": em}
    else:
        body, kw = qwen35_layer_run, {"li": li, "block": block, "R": n, "embedded": True,
                                      "em": em}
    bs = [layer_loop.trace(cfg, s, {"body": body, "pos": pos, "row": row, "n": n,
                                    "kw": {"m": m(s), **kw}})
          for s in range(cfg.S)]
    return [b.finish() for b in bs], list(bs[0].run_args)


def loop_key(key, embed: bool = False):
    """A layer run's key that loops (prefill_layers': (li, blocks, n, True), or tiled (li,
    blocks, n, True, "tile"); embed: the embed runs' (-1, blocks, 1, True) too) -> its loop's
    key ("L", li, blocks, n[, "tile"]), else None."""
    if not (isinstance(key, tuple) and key and isinstance(key[0], int)
            and (key[0] >= 0 or embed and key[0] == -1 and key[2] == 1)
            and len(key) in (4, 5) and key[1] is not None and key[3] is True):
        return None
    if len(key) == 5 and (key[4] != "tile" or key[0] < 0):
        return None
    return ("L",) + key[:3] + key[4:]


def loop_runs(runs, embed: bool = False) -> list:
    """Engine.prefill_layers' runs of a chunk with each layer's consecutive runs of n rows in
    one bucket (loop_key; their rows the chunk's, positions and rows n apart) as one loop run:
    arguments tpos0, row0, runs (tpos and row: the loop's, 0). embed: the embed runs too (the
    host's rows their tokens, a token a run: the loop run's rows the tokens, for out[]). A lone
    run that loops is a loop of one (a bucket's programs are then its loops alone: warm35's)."""
    out = []
    for key, vals, rows in runs:
        lk = loop_key(key, embed) if rows is None or key[0] == -1 else None
        last = out[-1] if out else None
        if lk is not None and last is not None and last[0] == lk and \
                last[1]["row0"] + key[2] * last[1]["runs"] == vals["row"] and \
                last[1]["tpos0"] + key[2] * last[1]["runs"] == vals["tpos"]:
            last[1]["runs"] += 1
            last[3].append((key, vals, rows))
        elif lk is not None:
            out.append([lk, {"tpos0": vals["tpos"], "row0": vals["row"], "runs": 1, "tpos": 0,
                             "row": 0}, None, [(key, vals, rows)]])
        else:
            out.append([key, vals, rows, None])
    return [tuple(x[:3]) if x[3] is None else
            (x[0], x[1], [t for _, _, r in x[3] for t in r] if x[0][1] < 0 else None)
            for x in out]
