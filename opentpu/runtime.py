"""Host runtime: argument placement in each slice's DRAM, compilation, launch, readback.

Host arguments:
  Input(a)            fp32 array replicated into every slice's DRAM   -> Tensor
  Weight(w, shard=0)  fp32 [N, K] quantized to int8 + per-D-block scales, optionally sharded
                      over slices by rows (shard=0) or columns (shard=1) -> QTensor
  Output(shape)       fp32 result; the kernel `store`s into it          -> Tensor
  KVCache(k, v, cap)  [Hkv, T, d] K/V, quantized, heads round-robin over slices -> KVDesc
Anything else (ints, floats) is passed to the kernel unchanged (compile-time constants).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import numpy as np

from . import fp32 as F
from .compiler import Affine, Compiled, KVDesc, Kernel, QTensor, Tensor
from .isasim import Config, Machine

ALIGN = 128                  # DRAM allocations: whole MXU chunks (D <= 128)


class Input:
    def __init__(self, array):
        self.array = F.ftz(np.asarray(array, dtype=np.float32))


class Output:
    def __init__(self, shape):
        self.shape = tuple(shape)


class Weight:
    def __init__(self, array, shard: int | None = None):
        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("Weight must be 2-D [N, K]")
        if shard not in (None, 0, 1):
            raise ValueError("Weight shard must be None, 0 (rows) or 1 (columns)")
        self.array, self.shard = a, shard


class KVCache:
    def __init__(self, k, v, capacity: int):
        k = np.asarray(k, dtype=np.float32)
        v = np.asarray(v, dtype=np.float32)
        if k.shape != v.shape or k.ndim != 3:
            raise ValueError("k and v must both be [Hkv, T, d]")
        self.k, self.v, self.cap = k, v, int(capacity)
        self.H, self.T, self.d = k.shape
        if self.T > self.cap:
            raise ValueError("more tokens than capacity")


def quantize_rows(a: np.ndarray, D: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-(row, D-block) int8 quantization, bit-identical to the hardware quantizer."""
    N, K = a.shape
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    q, s = F.quantize(a.reshape(N, K // D, D), axis=2)
    return q.reshape(N, K), s.astype(np.float32)


class _Dram:
    def __init__(self, S: int, size: int):
        self.size = size
        self.next = 0
        self.images = [np.zeros(size, dtype=np.uint8) for _ in range(S)]

    def alloc(self, nbytes: int) -> int:
        a = self.next
        self.next = (a + nbytes + ALIGN - 1) // ALIGN * ALIGN
        if self.next > self.size:
            raise MemoryError("DRAM image does not fit")
        return a

    def put(self, s: int, addr: int, data: np.ndarray) -> None:
        b = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.images[s][addr:addr + len(b)] = b


@dataclass
class Layout:
    params: dict = field(default_factory=dict)       # name -> (host arg, placement info)


def place(cfg: Config, args: dict) -> tuple[_Dram, Layout, list]:
    """Lay out host args in DRAM. Returns images, layout and per-slice bound kernel arguments."""
    S, D = cfg.S, cfg.D
    dram = _Dram(S, cfg.DRAM_BYTES)
    layout = Layout()
    bound = [dict() for _ in range(S)]
    for name, a in args.items():
        if isinstance(a, Input):
            addr = dram.alloc(a.array.nbytes)
            for s in range(S):
                dram.put(s, addr, a.array)
            layout.params[name] = (a, addr)
            desc = _tensor(addr, a.array.shape, a)
            for s in range(S):
                bound[s][name] = desc
        elif isinstance(a, Output):
            n = int(np.prod(a.shape))
            addr = dram.alloc(4 * n)
            layout.params[name] = (a, addr)
            for s in range(S):
                bound[s][name] = _tensor(addr, a.shape, a)
        elif isinstance(a, Weight):
            N, K = a.array.shape
            if a.shard == 0:
                if N % S:
                    raise ValueError(f"{name}: {N} rows not divisible by {S} slices")
                n_loc = N // S
                parts = [a.array[s * n_loc:(s + 1) * n_loc] for s in range(S)]
            elif a.shard == 1:
                if K % (S * D):
                    raise ValueError(f"{name}: {K} columns not divisible by {S} slices x D")
                n_loc, K = N, K // S
                parts = [a.array[:, s * K:(s + 1) * K] for s in range(S)]
            else:
                n_loc, parts = N, [a.array] * S
            KB = K // D
            daddr = dram.alloc(n_loc * K)
            saddr = dram.alloc(4 * n_loc * KB)
            for s, p in enumerate(parts):
                q, sc = quantize_rows(p, D)
                dram.put(s, daddr, q)
                dram.put(s, saddr, sc)
            layout.params[name] = (a, (daddr, saddr))
            desc = QTensor(Affine(daddr), Affine(saddr), (n_loc, K), K, 4 * KB, D)
            for s in range(S):
                bound[s][name] = desc
        elif isinstance(a, KVCache):
            if a.d % D:
                raise ValueError("head dim must be a multiple of D")
            cap, d = a.cap, a.d
            if cap % 4:
                raise ValueError("KV capacity must be a multiple of 4")
            per_slice = [[h for h in range(a.H) if h % S == s] for s in range(S)]
            nloc = max(len(p) for p in per_slice)
            regions = []
            for _ in range(nloc):
                regions.append({"k": dram.alloc(cap * d), "ks": dram.alloc(4 * cap * (d // D)),
                                "vt": dram.alloc(d * cap), "vs": dram.alloc(4 * cap)})
            heads = [dict() for _ in range(S)]
            for s in range(S):
                for j, h in enumerate(per_slice[s]):
                    r = regions[j]
                    heads[s][h] = r
                    kq, ks = quantize_rows(a.k[h], D)
                    vq, vs = F.quantize(a.v[h], axis=1)
                    kfull = np.zeros((cap, d), np.int8)
                    kfull[:a.T] = kq
                    ksf = np.zeros((cap, d // D), np.float32)
                    ksf[:a.T] = ks
                    vt = np.zeros((d, cap), np.int8)
                    vt[:, :a.T] = vq.T
                    vsf = np.zeros(cap, np.float32)
                    vsf[:a.T] = vs
                    dram.put(s, r["k"], kfull)
                    dram.put(s, r["ks"], ksf)
                    dram.put(s, r["vt"], vt)
                    dram.put(s, r["vs"], vsf)
            layout.params[name] = (a, heads)
            for s in range(S):
                bound[s][name] = KVDesc(heads[s], cap, d, D, S, s)
        else:
            for s in range(S):
                bound[s][name] = a
    return dram, layout, bound


def _tensor(addr: int, shape, host) -> Tensor:
    shape = tuple(shape)
    strides, st = [], 1
    for n in reversed(shape):
        strides.append(st)
        st *= n
    return Tensor(Affine(addr), shape, tuple(reversed(strides)), host, Affine(0))


def compile_kernel(kernel: Kernel, cfg: Config, **args):
    sig = inspect.signature(kernel.fn)
    missing = [p for p, v in sig.parameters.items()
               if p not in args and v.default is inspect.Parameter.empty]
    if missing:
        raise TypeError(f"{kernel.__name__}: missing arguments {missing}")
    dram, layout, bound = place(cfg, args)
    programs, stores = [], []
    for s in range(cfg.S):
        b = kernel.trace(cfg, s, bound[s])
        programs.append(b.finish())
        stores.append(b.stores)
    return Compiled(cfg, programs, stores, layout), dram.images


@dataclass
class Result:
    outputs: dict
    drams: list
    tmems: list
    compiled: Compiled
    stats: dict

    def kv(self, name: str):
        """Dequantized K and V ([Hkv, cap, d]) as stored in the slices' DRAM."""
        a, heads = self.compiled.layout.params[name]
        D = self.compiled.cfg.D
        K = np.zeros((a.H, a.cap, a.d), np.float32)
        V = np.zeros((a.H, a.cap, a.d), np.float32)
        for s, hs in enumerate(heads):
            img = self.drams[s]
            for h, r in hs.items():
                kq = img[r["k"]:r["k"] + a.cap * a.d].view(np.int8).reshape(a.cap, a.d)
                ks = img[r["ks"]:r["ks"] + 4 * a.cap * (a.d // D)].view(np.float32)
                ks = ks.reshape(a.cap, a.d // D)
                K[h] = kq.astype(np.float32) * np.repeat(ks, D, axis=1)
                vt = img[r["vt"]:r["vt"] + a.d * a.cap].view(np.int8).reshape(a.d, a.cap)
                vs = img[r["vs"]:r["vs"] + 4 * a.cap].view(np.float32)
                V[h] = vt.T.astype(np.float32) * vs[:, None]
        return K, V


def collect(compiled: Compiled, drams: list, tmems: list, stats: dict) -> Result:
    outputs = {}
    for name, (a, addr) in compiled.layout.params.items():
        if not isinstance(a, Output):
            continue
        flat = np.full(int(np.prod(a.shape)), np.nan, dtype=np.float32)
        written = np.zeros(flat.shape, dtype=bool)
        for s, recs in enumerate(compiled.stores):
            m32 = drams[s].view(np.uint32)
            for host, off, n in recs:
                if host is not a:
                    continue
                words = m32[addr // 4 + off: addr // 4 + off + n]
                flat[off:off + n] = words.view(np.float32)
                written[off:off + n] = True
        if not written.all():
            raise RuntimeError(f"output {name}: {int((~written).sum())} elements never stored")
        outputs[name] = flat.reshape(a.shape)
    return Result(outputs, drams, tmems, compiled, stats)


def launch(kernel: Kernel, cfg: Config, backend: str = "isa", **args) -> Result:
    compiled, images = compile_kernel(kernel, cfg, **args)
    return run(compiled, images, backend)


def run(compiled: Compiled, images: list, backend: str = "isa") -> Result:
    if backend == "isa":
        m = Machine(compiled.cfg, compiled.programs, images).run()
        stats = {"instructions": [s.icount for s in m.slices]}
        return collect(compiled, [s.dram for s in m.slices], [s.tmem for s in m.slices], stats)
    if backend == "rtl":
        from . import rtlsim
        drams, tmems, stats = rtlsim.run(compiled.cfg, compiled.programs, images)
        return collect(compiled, drams, tmems, stats)
    raise ValueError(f"unknown backend {backend}")
