"""Building blocks written in the openTPU language (they trace inline into the caller)."""
import math

from .. import language as ol


def rmsnorm(x, gamma, eps: float):
    """x: [M, H] tile, gamma: [H] tile (None: no weight, Gemma 4's v_norm)."""
    ss = ol.sum(x * x, axis=1)
    r = ol.rsqrt(ss * (1.0 / x.cols) + eps)
    if gamma is None:
        return x * r[:, None]
    return (x * r[:, None]) * gamma[None, :]


def sigmoid(x):
    """1 / (1 + 2^(-x*log2(e)))."""
    return ol.recip(ol.exp2(x * -ol.LOG2E) + 1.0)


def silu(x):
    """x * sigmoid(x)."""
    return x * sigmoid(x)


# gelu_tanh's exponent x (A + B x^2): -2 sqrt(2 / pi) log2(e) (x + 0.044715 x^3)
GELU_A = -2.0 * math.sqrt(2.0 / math.pi) / math.log(2.0)
GELU_B = GELU_A * 0.044715


def gelu_tanh(x):
    """GELU, tanh approximation (gelu_pytorch_tanh): 0.5 x (1 + tanh(z)) = x sigmoid(2 z),
    z = sqrt(2 / pi) (x + 0.044715 x^3), as x / (1 + 2^(x (A + B x^2))): 8 VOPs."""
    return x * ol.recip(ol.exp2(x * (x * x * GELU_B + GELU_A)) + 1.0)


def softcap(x, c: float):
    """The logit soft cap c tanh(x / c) (Gemma's final_logit_softcapping), as 2c sigmoid(2x / c)
    - c = 2c / (1 + 2^(-2 log2(e) x / c)) - c: 5 VOPs."""
    return ol.recip(ol.exp2(x * (-2.0 * ol.LOG2E / c)) + 1.0) * (2.0 * c) - c


def softplus(x):
    """log(1 + e^x) = max(x, 0) + ln2 * log2(1 + 2^(-|x| log2(e))), exact to ~1e-6 absolute
    (1 + e^x rounds to 1 for x < -17: the result is then 0 instead of e^x)."""
    return ol.maximum(x, 0.0) + ol.log2(ol.exp2(ol.abs(x) * -ol.LOG2E) + 1.0) * ol.LN2


def rope(x, cos, sin, out=None):
    """Rotate-half RoPE on the rows of x [M, d]; cos, sin: [d/2] tiles for one position.
    Writes into `out` (e.g. a view of a zero-padded tile) when given."""
    half = x.cols // 2
    x1, x2 = x[:, :half], x[:, half:]
    out = ol.empty(x.shape) if out is None else out
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
