"""Host runtime: argument placement in each slice's DRAM, compilation, launch, readback.

Host arguments:
  Input(a)            fp32 array replicated into every slice's DRAM   -> Tensor
  Weight(w, shard=0, fmt="int8")  fp32 [N, K] quantized to int8 + per-D-block scales (or 4-bit:
                      fmt="int4" | "fp4"), optionally sharded
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
from . import quant as Q
from .compiler import (Affine, Compiled, KVDesc, Kernel, QTensor, Tensor, k_scale_bytes,
                       vt_tiled, vt_untiled)
from .isasim import Config, Machine

ALIGN = 128                  # DRAM allocations: whole MXU chunks (D <= 128)


class Input:
    def __init__(self, array):
        self.array = F.ftz(np.asarray(array, dtype=np.float32))


class Output:
    def __init__(self, shape):
        self.shape = tuple(shape)


class Weight:
    """`fmt`: an MXU format, "int8", "int4", "fp4" or "nvfp4" (a per-row fp32 scale and
    power-of-two block offsets under it; opentpu/quant.py, docs/isa.md "Weight formats"). A
    slice's part (sharded by rows or columns) is quantized on its own."""

    def __init__(self, array, shard: int | None = None, fmt: str = "int8"):
        a = np.asarray(array, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError("Weight must be 2-D [N, K]")
        if shard not in (None, 0, 1):
            raise ValueError("Weight shard must be None, 0 (rows) or 1 (columns)")
        if fmt not in Q.MXU_FORMATS:
            raise ValueError(f"Weight fmt must be one of {Q.MXU_FORMATS}")
        self.array, self.shard, self.fmt = a, shard, fmt


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


def quantize_rows(a: np.ndarray, D: int, row: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Rows as QST quantizes and stores them, bit for bit: int8 values [N, K] and each row's
    scale row (uint8 [N, 4] with `row`, QST ROW's fp32 scale; else the int8 scale row, the S and
    an offset byte per D-block: quant.scale_bytes(K, "int8"))."""
    N, K = a.shape
    if K % D:
        raise ValueError(f"inner dimension {K} is not a multiple of D={D}")
    q, S, d = F.quantize_mx(np.asarray(a, np.float32), D, row)
    if row:
        return q, S.astype(np.float32).view(np.uint8).reshape(N, 4)
    return q, Q.pack_scales("int8", S, dw=d)


REGION = 1 << 32             # a DRAM region above DAW 32 (isa.py "DRAM address width")


class _Dram:
    """The host arguments' DRAM: bump allocation, 128-byte aligned. Above DAW 32 (`wide`) a
    bump per 4 GiB region: an allocation goes to the region asked for (place's `regions`), else
    after the one before, and one that would cross a region boundary starts at the next region
    (no instruction's DRAM range crosses one). The images are np.zeros (lazy: a 64 GiB DRAM costs
    what is written)."""

    def __init__(self, S: int, size: int, wide: bool = False):
        self.size, self.wide = size, wide
        self.next = 0
        self.cur = 0                            # (wide: the region allocations go to)
        self.tops: dict = {}                    # (wide: region -> its next free address)
        self.images = [np.zeros(size, dtype=np.uint8) for _ in range(S)]

    def alloc(self, nbytes: int, region: int | None = None) -> int:
        return self.alloc_group([nbytes], region)[0]

    def alloc_group(self, sizes, region: int | None = None) -> list[int]:
        """Allocations one after the other (each aligned) in one region: an operand and its
        scale rows (one instruction reads both: they share its region)."""
        span, offs = 0, []
        for n in sizes:
            offs.append(span)
            span = (span + n + ALIGN - 1) // ALIGN * ALIGN
        if not self.wide:
            if region:
                raise ValueError(f"DRAM region {region}: a configuration with DAW > 32")
            a = self.next
            self.next = a + span
            if self.next > self.size:
                raise MemoryError("DRAM image does not fit")
            return [a + o for o in offs]
        r = self.cur if region is None else region
        a = self.tops.get(r, r * REGION)
        if a + span > (r + 1) * REGION:         # (it would cross the region's end)
            if region is not None:
                raise MemoryError(f"DRAM region {region} is full")
            r += 1
            a = self.tops.get(r, r * REGION)
        if a + span > self.size:
            raise MemoryError("DRAM image does not fit")
        self.tops[r] = a + span
        if region is None:
            self.cur = r
        self.next = max(self.next, a + span)
        return [a + o for o in offs]

    def put(self, s: int, addr: int, data: np.ndarray) -> None:
        b = np.ascontiguousarray(data).view(np.uint8).reshape(-1)
        self.images[s][addr:addr + len(b)] = b


@dataclass
class Layout:
    params: dict = field(default_factory=dict)       # name -> (host arg, placement info)


def place(cfg: Config, args: dict, regions=None) -> tuple[_Dram, Layout, list]:
    """Lay out host args in DRAM. Returns images, layout and per-slice bound kernel arguments.
    regions (DAW > 32): the 4 GiB region of every argument (an int), or of those a dict names
    ({name: region}), the others after the one before; a Weight's rows and scale rows, and a KV
    head's K, K scales, V^T and V scales, lie in one region (isa.py: an instruction's DRAM
    operands share its region)."""
    S, D = cfg.S, cfg.D
    wide = getattr(cfg, "DAW", 32) > 32
    if regions and not wide:
        raise ValueError("place(regions=...): a configuration with DAW > 32")
    dram = _Dram(S, cfg.DRAM_BYTES, wide)
    layout = Layout()
    bound = [dict() for _ in range(S)]
    for name, a in args.items():
        reg = regions.get(name) if isinstance(regions, dict) else regions
        alloc = (lambda n, reg=reg: dram.alloc(n, reg))                 # noqa: E731
        if isinstance(a, Input):
            addr = alloc(a.array.nbytes)
            for s in range(S):
                dram.put(s, addr, a.array)
            layout.params[name] = (a, addr)
            desc = _tensor(addr, a.array.shape, a)
            for s in range(S):
                bound[s][name] = desc
        elif isinstance(a, Output):
            n = int(np.prod(a.shape))
            addr = alloc(4 * n)
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
            rs, srs = Q.row_bytes(K, a.fmt, D), Q.scale_bytes(K, a.fmt, D)
            daddr, saddr = dram.alloc_group([n_loc * rs, n_loc * srs], reg)
            for s, p in enumerate(parts):
                q, sc = Q.quantize_mxu(p, a.fmt, D)
                dram.put(s, daddr, q)
                dram.put(s, saddr, sc)
            layout.params[name] = (a, (daddr, saddr))
            desc = QTensor.weight(daddr, saddr, n_loc, K, a.fmt, D)
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
            ksb = k_scale_bytes(d, D)
            hregs = []
            for _ in range(nloc):
                k, ks, vt, vs = dram.alloc_group([cap * d, cap * ksb, d * cap, 4 * cap], reg)
                hregs.append({"k": k, "ks": ks, "vt": vt, "vs": vs})
            heads = [dict() for _ in range(S)]
            for s in range(S):
                for j, h in enumerate(per_slice[s]):
                    r = hregs[j]
                    heads[s][h] = r
                    kq, ks = quantize_rows(a.k[h], D, row=d == D)
                    vq, vs = F.quantize(a.v[h], axis=1)
                    kfull = np.zeros((cap, d), np.int8)
                    kfull[:a.T] = kq
                    ksf = np.zeros((cap, ksb), np.uint8)
                    ksf[:a.T] = ks
                    vt = np.zeros((d, cap), np.int8)
                    vt[:, :a.T] = vq.T
                    vsf = np.zeros(cap, np.float32)
                    vsf[:a.T] = vs
                    dram.put(s, r["k"], kfull)
                    dram.put(s, r["ks"], ksf)
                    dram.put(s, r["vt"], vt_tiled(vt))
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


def compile_kernel(kernel: Kernel, cfg: Config, *, regions=None, **args):
    """The kernel's programs for `args` placed in each slice's DRAM (place; regions: the
    arguments' 4 GiB regions, DAW > 32), and the images. No kernel argument is called
    `regions`."""
    sig = inspect.signature(kernel.fn)
    missing = [p for p, v in sig.parameters.items()
               if p not in args and v.default is inspect.Parameter.empty]
    if missing:
        raise TypeError(f"{kernel.__name__}: missing arguments {missing}")
    dram, layout, bound = place(cfg, args, regions)
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
        ksb = k_scale_bytes(a.d, D)
        for s, hs in enumerate(heads):
            img = self.drams[s]
            for h, r in hs.items():
                kq = img[r["k"]:r["k"] + a.cap * a.d].reshape(a.cap, a.d)
                ks = img[r["ks"]:r["ks"] + a.cap * ksb].reshape(a.cap, ksb)
                K[h] = Q.dequantize_mxu(kq, ks, "int8", a.d, D, rsc=a.d == D)
                vt = vt_untiled(img[r["vt"]:r["vt"] + a.d * a.cap].view(np.int8), a.d, a.cap)
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


def launch(kernel: Kernel, cfg: Config, backend: str = "isa", *, regions=None,
           **args) -> Result:
    compiled, images = compile_kernel(kernel, cfg, regions=regions, **args)
    return run(compiled, images, backend)


def run(compiled: Compiled, images: list, backend: str = "isa",
        adopt: bool | None = None) -> Result:
    """adopt: the ISA simulator takes the image arrays as its DRAM (no copy; they hold the
    run's results afterwards: the caller gives them up). Default: where the DRAM is over 4 GiB
    (DAW > 32), whose copy would touch every page; else the images are copied."""
    if backend == "isa":
        if adopt is None:
            adopt = compiled.cfg.DRAM_BYTES > REGION
        m = Machine(compiled.cfg, compiled.programs, images, adopt=adopt).run()
        stats = {"instructions": [s.icount for s in m.slices]}
        return collect(compiled, [s.dram for s in m.slices], [s.tmem for s in m.slices], stats)
    if backend == "rtl":
        from . import rtlsim
        drams, tmems, stats = rtlsim.run(compiled.cfg, compiled.programs, images)
        return collect(compiled, drams, tmems, stats)
    raise ValueError(f"unknown backend {backend}")
