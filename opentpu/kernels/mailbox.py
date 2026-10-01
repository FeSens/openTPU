"""The card's side of a mailbox the host serves (opentpu/host/offload.py: seq, a request row and
served, fp32 words on 64-byte lines of their own; docs/offload.md 5.2): the fence and the post,
one slice. A MoE layer posts its experts' ids (llm/moe.py, the same pattern inline), Gemma 4
E4B's generate loop each token's id (its PLE record from the host, docs/gemma4_e4b.md)."""
from __future__ import annotations

from .. import isa as I
from .. import language as ol
from ..compiler import Affine, Tensor, current
from ..host.offload import LINE


def wait_served(mbox: int, comment: str = "fence: served >= seq"):
    """WAITW until the host has served every request posted to the mailbox at mbox
    (served >= seq: fp32 bits compare as the non-negative floats they are). Returns the seq
    tile [1]. One scratch register while it waits."""
    b = current()
    seq = ol.load(Tensor(Affine(mbox), (1,), (1,)))
    word = ol.empty((1,))
    r = b.scratch()
    b.rld(r, seq, raw=True, comment="seq (bits)")
    b.waitw(word, mbox + 2 * LINE, 0, I.C_GE, rc=r, comment=comment)
    b.unscratch(r)
    del word
    return seq


def post(mbox: int, ids) -> None:
    """A request: after the fence (the host's row is free), the ids (fp32, a [k] tile) to the
    row, then seq + 1."""
    seq = wait_served(mbox)
    ol.store(Tensor(Affine(mbox + LINE), (ids.rows * ids.cols,), (1,)), ids)
    seq.set(seq + 1.0)
    ol.store(Tensor(Affine(mbox), (1,), (1,)), seq)
