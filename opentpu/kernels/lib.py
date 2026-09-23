"""Building blocks written in the openTPU language (they trace inline into the caller)."""
from .. import language as ol


def rmsnorm(x, gamma, eps: float):
    """x: [M, H] tile, gamma: [H] tile."""
    ss = ol.sum(x * x, axis=1)
    r = ol.rsqrt(ss * (1.0 / x.cols) + eps)
    return (x * r[:, None]) * gamma[None, :]


def silu(x):
    """x * sigmoid(x) with sigmoid(x) = 1 / (1 + 2^(-x*log2(e)))."""
    return x * ol.recip(ol.exp2(x * -ol.LOG2E) + 1.0)


def rope(x, cos, sin):
    """Rotate-half RoPE on the rows of x [M, d]; cos, sin: [d/2] tiles for one position."""
    half = x.cols // 2
    x1, x2 = x[:, :half], x[:, half:]
    out = ol.empty(x.shape)
    out[:, :half].set(x1 * cos[None, :] - x2 * sin[None, :])
    out[:, half:].set(x2 * cos[None, :] + x1 * sin[None, :])
    return out


def rope_rows(x, cos, sin, out=None):
    """Rotate-half RoPE with one position per row: x [M, d]; cos, sin: [M, d/2] tiles.
    Writes into `out` (e.g. a strided view) when given."""
    half = x.cols // 2
    x1, x2 = x[:, :half], x[:, half:]
    out = ol.empty(x.shape) if out is None else out
    out[:, :half].set(x1 * cos - x2 * sin)
    out[:, half:].set(x2 * cos + x1 * sin)
    return out
