"""MiniCPM-V 4.7 (openbmb/MiniCPM-V-4.7-35B-A3B) on openTPU: docs/minicpmv.md.

The language model is Qwen3.5-MoE (qwen35.Spec, streamed experts). This module holds what is
MiniCPM-V's own:
  VisionSpec        the vision tower's and projector's dimensions and the image token ids
  host side         what the host does to an image, as a tokenizer does to text (no model
                    math): decode, the slice grid, bicubic resizes, normalization, 14 x 14
                    patches; the NaViT position ids; the placeholder text; the canvas M-RoPE
                    positions (integer index arithmetic). Ports of transformers v5.18.0's
                    MiniCPMV4_6ImageProcessor, MiniCPMV4_6Processor and
                    MiniCPMV4_7Model.get_rope_index (Apache-2.0)
  vision_reference  the SigLIP2 tower, its window merger and the projector, fp32 numpy (HF's
                    math): the reference the card's runs are checked against
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import quant as Q

TOWER, MERGER = "model.vision_tower.", "model.merger."


# =============================================================================== the spec
@dataclass(frozen=True)
class VisionSpec:
    hidden: int = 1152              # SigLIP2 so400m
    heads: int = 16                 # of 72
    ffn: int = 4304
    layers: int = 27
    patch: int = 14
    side: int = 70                  # the position table's side: image_size 980 / patch 14
    eps: float = 1e-6               # every LayerNorm's (the projector's too)
    insert: int = 6                 # the window merger after this layer (insert_layer_id)
    window: int = 2                 # its window and the projector's merge: 2 x 2
    out: int = 2048                 # the language model's hidden size
    scale: int = 448                # the processor's scale_resolution
    max_slices: int = 9             # its max_slice_nums (preprocessor_config.json)
    image_token: int = 248056       # <|image_pad|>
    image_start: int = 248078       # <image>
    image_end: int = 248079
    slice_start: int = 248088       # <slice>
    slice_end: int = 248089
    newline: int = 198              # "\n"

    @property
    def head_dim(self) -> int:
        return self.hidden // self.heads

    @property
    def merged(self) -> int:
        """The patches per language-model token: two 2 x 2 merges (16x mode)."""
        return (self.window * self.window) ** 2

    @staticmethod
    def from_hf(model_dir) -> "VisionSpec":
        top = json.loads((Path(model_dir) / "config.json").read_text())
        if top.get("downsample_mode", "16x") != "16x" or top.get("merger_times", 1) != 1:
            raise ValueError(f"{model_dir}: only downsample_mode 16x, merger_times 1")
        v = top["vision_config"]
        if v.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh":
            raise ValueError(f"{model_dir}: vision activation {v['hidden_act']}")
        pre = Path(model_dir) / "preprocessor_config.json"
        p = json.loads(pre.read_text()) if pre.exists() else {}
        if any(x != 0.5 for x in p.get("image_mean", [0.5]) + p.get("image_std", [0.5])):
            raise ValueError(f"{model_dir}: image mean / std other than 0.5")
        return VisionSpec(hidden=v["hidden_size"], heads=v["num_attention_heads"],
                          ffn=v["intermediate_size"], layers=v["num_hidden_layers"],
                          patch=v["patch_size"], side=v["image_size"] // v["patch_size"],
                          eps=v.get("layer_norm_eps", 1e-6),
                          insert=top.get("insert_layer_id", 6),
                          out=top["text_config"]["hidden_size"],
                          scale=p.get("scale_resolution", 448),
                          max_slices=p.get("max_slice_nums", 9),
                          image_token=top["image_token_id"], image_start=top["image_start_id"],
                          image_end=top["image_end_id"], slice_start=top["slice_start_id"],
                          slice_end=top["slice_end_id"], newline=top["newline_id"])


# =============================================================================== host side
def ensure_divide(length, divisor: int) -> int:
    return max(round(length / divisor) * divisor, divisor)


def best_resize(h, w, scale: int, patch: int, upscale: bool = False) -> tuple:
    """find_best_resize: (h, w) of about scale^2 pixels at the aspect ratio when the image is
    larger (or upscale), each a multiple of 4 patches (the two 2 x 2 merges)."""
    if h * w > scale * scale or upscale:
        ar = w / h
        h = int(scale / math.sqrt(ar))
        w = int(h * ar)
    return ensure_divide(h, patch * 4), ensure_divide(w, patch * 4)


def slice_grid(h: int, w: int, max_slices: int, scale: int):
    """get_sliced_grid: (rows, cols) of the slices, or None (the thumbnail alone)."""
    log_ratio = math.log(w / h)
    multiple = min(math.ceil(w * h / (scale * scale)), max_slices)
    if multiple <= 1:
        return None
    best, err = [1, 1], float("inf")
    for n in (multiple - 1, multiple, multiple + 1):
        if n == 1 or n > max_slices:
            continue
        for rows in range(1, n + 1):
            if n % rows == 0:
                e = abs(log_ratio - math.log((n // rows) / rows))
                if e < err:
                    best, err = [rows, n // rows], e
                elif e == err and rows > best[0]:
                    best = [rows, n // rows]
    return tuple(best)


def refine_size(h: int, w: int, grid, scale: int, patch: int) -> tuple:
    """get_refine_size: the size the image is resized to before it is cut into the grid."""
    rows, cols = grid
    bh, bw = best_resize(ensure_divide(h, rows) / rows, ensure_divide(w, cols) / cols, scale,
                         patch, upscale=True)
    return bh * rows, bw * cols


def decode(src):
    """An image as HF's processor reads it: a path, a data: URL or base64 string, or the file's
    bytes (torchvision's decode_image, RGB; its EXIF orientation is not applied, as HF's for a
    path), or a PIL image (EXIF-transposed, RGB) -> uint8 [3, H, W] (a torch tensor)."""
    import torch
    try:
        from torchvision.io import ImageReadMode, decode_image
    except ImportError:                     # (no torchvision: PIL's decoder, RGB)
        decode_image = None
    if isinstance(src, (bytes, bytearray)):
        src, raw = None, bytes(src)
    elif not isinstance(src, (str, Path)):
        from PIL import ImageOps
        im = ImageOps.exif_transpose(src).convert("RGB")
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()
    s = None if src is None else str(src)
    if s is None:
        pass
    elif not s.startswith("data:") and len(s) < 4096 and Path(s).is_file():
        raw = Path(s).read_bytes()
    else:
        import base64
        raw = base64.decodebytes((s.split(",", 1)[1] if s.startswith("data:image/") else s)
                                 .encode())
    if decode_image is not None:
        return decode_image(torch.frombuffer(bytearray(raw), dtype=torch.uint8),
                            mode=ImageReadMode.RGB)
    import io
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()


def _resize(img, h: int, w: int):
    """torchvision's bicubic resize with antialiasing of a uint8 [3, H, W] tensor (uint8), as
    HF's TorchvisionBackend; without torchvision, torch's own (the same kernel in float, rounded
    and clamped: not checked bit for bit)."""
    try:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.v2 import functional as tvF
        return tvF.resize(img, [h, w], interpolation=InterpolationMode.BICUBIC, antialias=True)
    except ImportError:
        import torch
        y = torch.nn.functional.interpolate(img[None].float(), size=(h, w), mode="bicubic",
                                            antialias=True, align_corners=False)[0]
        return y.round().clamp(0, 255).to(torch.uint8)


def crops(img, vs: VisionSpec, max_slices: int | None = None):
    """An image (uint8 [3, H, W]) as the processor cuts it: ([the thumbnail, slices...] as
    uint8 tensors, the grid (rows, cols) or None)."""
    max_slices = vs.max_slices if max_slices is None else max_slices
    H, W = img.shape[-2:]
    grid = slice_grid(H, W, max_slices, vs.scale) if max_slices >= 1 else None
    th, tw = best_resize(H, W, vs.scale, vs.patch, upscale=grid is None)
    out = [_resize(img, th, tw)]
    if grid is not None:
        rh, rw = refine_size(H, W, grid, vs.scale, vs.patch)
        ref = _resize(img, rh, rw)
        ph, pw = rh // grid[0], rw // grid[1]
        out += [ref[:, i:i + ph, j:j + pw] for i in range(0, rh, ph) for j in range(0, rw, pw)]
    return out, grid


def patches(crop, patch: int = 14) -> np.ndarray:
    """A crop's patch rows [h w, 3 patch^2] (fp32): its patch x patch blocks row-major, each
    (channel, row, column), normalized (x - 127.5) / 127.5 as HF's fused rescale and
    normalize (mean = std = 0.5): the patch embedding is then a matrix product."""
    x = np.asarray(crop, np.float32)
    x = (x - np.float32(127.5)) / np.float32(127.5)
    C, H, W = x.shape
    h, w = H // patch, W // patch
    return np.ascontiguousarray(x.reshape(C, h, patch, w, patch).transpose(1, 3, 0, 2, 4)
                                .reshape(h * w, C * patch * patch))


def position_ids(h: int, w: int, side: int) -> np.ndarray:
    """The NaViT position ids of an h x w patch grid in the side x side table
    (get_vision_nearest_position_ids: float32 bucketize, as torch computes it)."""
    import torch
    b = torch.arange(1 / side, 1.0, 1 / side)
    bh = torch.bucketize(torch.arange(h) / h, b, right=True)
    bw = torch.bucketize(torch.arange(w) / w, b, right=True)
    return (bh[:, None] * side + bw).flatten().numpy()


@dataclass
class ImageInput:
    """One image as the host prepares it: its crops' patch rows and (h, w) in patches (the
    thumbnail first), the slice grid, the placeholder text that replaces its <|image_pad|> in
    the prompt, and its content's key (sha256 of the crops' sizes and patch rows: token_keys)."""
    patches: list
    sizes: list
    grid: tuple | None
    text: str
    key: str = ""

    @property
    def tokens(self) -> int:
        """Its language-model rows (the <|image_pad|> tokens of its placeholder)."""
        return sum(h * w for h, w in self.sizes) // 16


def prepare(src, vs: VisionSpec, max_slices: int | None = None, image_id: int | None = 0,
            pad: str = "<|image_pad|>") -> ImageInput:
    """An image (path, data URL, base64 or PIL) -> ImageInput: replace_image_token's text with
    use_image_id's <image_id>N</image_id> prefix (image_id None: none)."""
    cs, grid = crops(decode(src), vs, max_slices)
    ps = [patches(c, vs.patch) for c in cs]
    sizes = [(c.shape[-2] // vs.patch, c.shape[-1] // vs.patch) for c in cs]
    n = [h * w // vs.merged for h, w in sizes]
    text = "<image>" + pad * n[0] + "</image>"
    if grid is not None:
        row = ("<slice>" + pad * n[1] + "</slice>") * grid[1]
        text += "\n".join([row] * grid[0])
    if image_id is not None:
        text = f"<image_id>{image_id}</image_id>" + text
    h = hashlib.sha256(repr(sizes).encode())
    for p in ps:
        h.update(np.ascontiguousarray(p).tobytes())
    return ImageInput(ps, sizes, grid, text, h.hexdigest())


def expand(text: str, images: list, pad: str = "<|image_pad|>") -> str:
    """The chat template's text with its i-th `pad` replaced by images[i].text (one per image,
    in order: get_text_with_replacements)."""
    parts = text.split(pad)
    if len(parts) - 1 != len(images):
        raise ValueError(f"{len(parts) - 1} image placeholders for {len(images)} images")
    return parts[0] + "".join(im.text + p for im, p in zip(images, parts[1:]))


def token_keys(im: ImageInput) -> list[int]:
    """Stand-ins for an image's <|image_pad|> tokens in a context's key (otpu-serve's prefix
    matching and KV store compare token ids, and every image's pads are the one id): negative
    int32s, the k-th a hash of the image's content key and k, so the keys of two contexts part
    at their first differing image's first token."""
    h = bytes.fromhex(im.key)
    return [-1 - int.from_bytes(hashlib.blake2b(h + k.to_bytes(4, "little"),
                                                digest_size=4).digest(), "little") % (2**31 - 1)
            for k in range(im.tokens)]


def keyed(ids, images: list, image_token: int) -> list[int]:
    """A prompt's key: its token ids with each image's pads its token_keys, in order."""
    keys = [k for im in images for k in token_keys(im)]
    out, i = [], 0
    for t in ids:
        if t == image_token:
            if i == len(keys):
                raise ValueError(f"more image tokens than the {len(images)} images' {len(keys)}")
            out.append(keys[i])
            i += 1
        else:
            out.append(int(t))
    if i != len(keys):
        raise ValueError(f"{i} image tokens for the {len(images)} images' {len(keys)}")
    return out


def unkeyed(keys, image_token: int) -> list[int]:
    """The token ids of a key (keyed's inverse)."""
    return [image_token if t < 0 else t for t in keys]


def _frame_end(last: int, ids: list, markers: tuple, limit: int) -> int:
    e = last
    while e < limit and ids[e] in markers:
        e += 1
    return e


def _groups(ids: list, mm: list, vs: VisionSpec):
    """_group_visual_frames: [(group start, group end, [(thumb start, thumb end, frame end,
    slices [(start, end)])])] of one sequence; mm 1 on the image tokens."""
    markers = (vs.slice_start, vs.slice_end, vs.image_start, vs.image_end, vs.newline)
    groups, open_slices, prev_end = [], None, 0
    for modality, run in itertools.groupby(enumerate(mm), lambda t: t[1]):
        if modality == 0:
            continue
        run = list(run)
        crop = (run[0][0], run[-1][0] + 1)
        if open_slices is not None and ids[crop[0] - 1] == vs.slice_start:
            open_slices.append(crop)
        else:
            open_slices = []
            frame = (crop[0], crop[1], open_slices)
            gap = ids[prev_end:crop[0] - 1]
            if groups and all(t in markers for t in gap):
                groups[-1][1].append(frame)
            else:
                groups.append((crop[0] - 1, [frame]))
        prev_end = crop[1]
    out = []
    for g0, frames in groups:
        fr = [(a, b, _frame_end(sl[-1][1] if sl else b, ids, markers, len(ids)), sl)
              for a, b, sl in frames]
        out.append((g0, fr[-1][2], fr))
    return out


def _grid_positions(o: int, grid, canvas_h: int = 0, canvas_w: int = 0, h_off: int = 0,
                    w_off: int = 0, merge: int = 4):
    """get_vision_position_ids: [3, n] (t, h, w) of a crop placed on the canvas."""
    import torch
    gh, gw = int(grid[0]) // merge, int(grid[1]) // merge
    ch, cw = canvas_h or gh, canvas_w or gw
    hc = torch.linspace(0, ch - 1, gh).round().long() + h_off
    wc = torch.linspace(0, cw - 1, gw).round().long() + w_off
    H, W = torch.meshgrid(hc, wc, indexing="ij")
    return (torch.stack([torch.zeros_like(H), H, W]).reshape(3, -1) + o).numpy()


def rope_index(ids, sizes, vs: VisionSpec, merge: int = 4) -> tuple:
    """Canvas M-RoPE (get_rope_index for one unpadded sequence): (positions [3, T] of (t, h,
    w), delta), delta = max position + 1 - T (decode: slot s is at position s + delta). sizes:
    every crop's (h, w) in patches, in order (thumbnail, slices; image after image). A text-only
    sequence is arange(T) in each row, delta 0."""
    ids = [int(t) for t in ids]
    mm = [1 if t == vs.image_token else 0 for t in ids]
    T = len(ids)
    pos = np.tile(np.arange(T, dtype=np.int64), (3, 1))
    it = iter([tuple(int(v) for v in s) for s in sizes])
    cur, idx = 0, 0
    for g0, g1, frames in _groups(ids, mm, vs):
        if g0 > idx:
            pos[:, idx:g0] = np.arange(g0 - idx) + cur
            cur += g0 - idx
        cursor = g0
        for t0, t1, fend, slices in frames:
            thumb = next(it)
            fstart = t0 - 1
            if fstart > cursor:
                n = fstart - cursor
                pos[:, cursor:fstart] = np.arange(n) + cur
                cur += n + 1
            fend = min(fend, g1)
            o = cur
            halo = max(o - 1, 0)
            sh = sw = rows = cols = 0
            ch, cw = thumb[0] // merge, thumb[1] // merge
            if slices:
                cols = len(slices)
                for k in range(len(slices) - 1):
                    if slices[k + 1][0] - slices[k][1] > 2:
                        cols = k + 1
                        break
                rows = len(slices) // cols if cols > 0 else 1
                if rows * cols != len(slices):
                    rows, cols = 1, len(slices)
                first = next(it)
                sh, sw = first[0] // merge, first[1] // merge
                ch, cw = rows * sh, cols * sw
            pos[:, fstart:fend] = o
            pos[1:, fstart] = halo
            if t1 < fend:
                pos[1, t1], pos[2, t1] = o + ch, o + cw
            pos[:, t0:t1] = _grid_positions(o, thumb, ch, cw, merge=merge)
            for k, (s0, s1) in enumerate(slices):
                ho, wo = (k // cols) * sh, (k % cols) * sw
                sz = first if k == 0 else next(it)
                gh, gw = sz[0] // merge, sz[1] // merge
                if s0 - 1 >= fstart:
                    pos[1, s0 - 1], pos[2, s0 - 1] = o + ho, o + wo
                if s1 < fend:
                    pos[1, s1], pos[2, s1] = o + ho + gh - 1, o + wo + gw - 1
                pos[:, s0:s1] = _grid_positions(o, sz, h_off=ho, w_off=wo, merge=merge)
            for k in range(len(slices) - 1):
                a, b = slices[k][1], slices[k + 1][0]
                if b - a <= 2:
                    continue
                for nl in range(a + 1, b - 1):
                    pos[1, nl] = o + (k // cols + 1) * sh - 1
                    pos[2, nl] = o + cols * sw
            cur = o + max(ch, cw) + 1
            cursor = fend
        if cursor < g1:
            pos[:, cursor:g1] = np.arange(g1 - cursor) + cur
            cur += g1 - cursor
        idx = g1
    if idx < T:
        pos[:, idx:T] = np.arange(T - idx) + cur
    return pos, int(pos.max()) + 1 - T if T else 0


# =============================================================================== reference
def load_vision(model_dir) -> dict:
    """The vision tower's and projector's tensors (fp32 numpy), by their checkpoint names."""
    import torch
    from safetensors import safe_open
    W = {}
    for f in sorted(Path(model_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                if k.startswith((TOWER, MERGER)):
                    W[k] = h.get_tensor(k).to(torch.float32).numpy()
    return W


class VisionWeights:
    """load_vision's tensors read from the checkpoint as they are asked for, none kept (the
    staged pieces' packing and the vectors of a staged image's build)."""

    def __init__(self, model_dir):
        from safetensors import safe_open
        self.model_dir = Path(model_dir)
        self._at = {}
        for f in sorted(self.model_dir.glob("*.safetensors")):
            with safe_open(str(f), "pt") as h:
                for k in h.keys():
                    if k.startswith((TOWER, MERGER)):
                        self._at[k] = str(f)

    def __contains__(self, k) -> bool:
        return k in self._at

    def keys(self):
        return self._at.keys()

    def __getitem__(self, k) -> np.ndarray:
        import torch
        from safetensors import safe_open
        with safe_open(self._at[k], "pt") as h:
            return h.get_tensor(k).to(torch.float32).numpy()


def _ln(x, w, b, eps):
    m = x.mean(-1, keepdims=True)
    xc = x - m
    return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps) * w + b


def _gelu_tanh(x):
    return 0.5 * x * (1 + np.tanh(np.float32(math.sqrt(2 / math.pi)) * (x + 0.044715 * x ** 3)))


def gelu_erf(x):
    """nn.GELU(): x Phi(x), the exact erf form (the projector's)."""
    import torch
    t = torch.from_numpy(np.ascontiguousarray(x, np.float32))
    return torch.nn.functional.gelu(t).numpy()


def _lin(x, W, n):
    return x @ W[n + ".weight"].T + W[n + ".bias"]


def _attention(x, W, p, vs: VisionSpec, groups):
    """Bidirectional attention of x's rows within each group of row indices."""
    q, k, v = (_lin(x, W, p + n) for n in ("q_proj", "k_proj", "v_proj"))
    d = vs.head_dim
    o = np.zeros_like(q)
    for g in groups:
        for h in range(vs.heads):
            c = slice(h * d, (h + 1) * d)
            s = q[g, c] @ k[g, c].T * np.float32(d ** -0.5)
            s = np.exp(s - s.max(axis=1, keepdims=True))
            o[g, c] = (s / s.sum(axis=1, keepdims=True)) @ v[g, c]
    return _lin(o, W, p + "out_proj")


def windows(h: int, w: int, k: int = 2) -> np.ndarray:
    """[h w / k^2, k^2]: each k x k window's row indices of the h x w grid (row-major), windows
    row-major: the merges' order (the window merger's attention groups and concatenation, the
    projector's 2 x 2)."""
    i = np.arange(h * w).reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3)
    return i.reshape(-1, k * k)


def card_order(h: int, w: int) -> np.ndarray:
    """[h w]: the raster indices of an h x w crop's patches in the card's window order (h, w
    multiples of 4): each 4 x 4 block row-major, within it its 2 x 2 windows row-major, within
    those their 4 patches (opentpu/kernels/vit.py: both merges' windows are then consecutive
    rows; the projector's output rows come out row-major over the blocks, HF's order)."""
    i = np.arange(h * w).reshape(h // 4, 2, 2, w // 4, 2, 2)
    return i.transpose(0, 3, 1, 4, 2, 5).reshape(-1)


def _layer(x, W, i, vs, groups):
    p = f"{TOWER}encoder.layers.{i}."
    x = x + _attention(_ln(x, W[p + "layer_norm1.weight"], W[p + "layer_norm1.bias"], vs.eps),
                       W, p + "self_attn.", vs, groups)
    y = _ln(x, W[p + "layer_norm2.weight"], W[p + "layer_norm2.bias"], vs.eps)
    return x + _lin(_gelu_tanh(_lin(y, W, p + "mlp.fc1")), W, p + "mlp.fc2")


def vision_reference(vs: VisionSpec, W: dict, P: np.ndarray, size, capture=None,
                     gelu=gelu_erf) -> np.ndarray:
    """One crop's language-model rows [h w / 16, out] (fp32): P its patch rows [h w, 3 p^2],
    size its (h, w) in patches. capture: a dict that gets the stages' rows ("embed",
    "layer6" .. the input of the window merger, "merged" its output, "post" after
    post_layernorm, "out"). gelu: the projector's (nn.GELU's erf; lib's tanh for the
    measurement of docs/minicpmv.md 4.2)."""
    h, w = size
    cap = capture if capture is not None else {}
    f32 = np.float32
    Wp = W[TOWER + "embeddings.patch_embedding.weight"].reshape(vs.hidden, -1)
    x = (np.asarray(P, f32) @ Wp.T + W[TOWER + "embeddings.patch_embedding.bias"]
         + W[TOWER + "embeddings.position_embedding.weight"][position_ids(h, w, vs.side)])
    cap["embed"] = x
    every = [np.arange(h * w)]
    for i in range(vs.insert + 1):
        x = _layer(x, W, i, vs, every)
    cap[f"layer{vs.insert}"] = x
    m = TOWER + "vit_merger."
    win = windows(h, w, vs.window)
    x = x + _attention(_ln(x, W[m + "layer_norm1.weight"], W[m + "layer_norm1.bias"], vs.eps),
                       W, m + "self_attn.", vs, list(win))
    z = x[win].reshape(len(win), -1)                        # [h w / 4, 4 C]
    y = _ln(z, W[m + "pre_norm.weight"], W[m + "pre_norm.bias"], vs.eps)
    x = _lin(_gelu_tanh(_lin(y, W, m + "linear_1")), W, m + "linear_2") + x[win].mean(axis=1)
    cap["merged"] = x
    h2, w2 = h // 2, w // 2
    every = [np.arange(h2 * w2)]
    for i in range(vs.insert + 1, vs.layers):
        x = _layer(x, W, i, vs, every)
    x = _ln(x, W[TOWER + "post_layernorm.weight"], W[TOWER + "post_layernorm.bias"], vs.eps)
    cap["post"] = x
    z = x[windows(h2, w2, vs.window)].reshape(h2 * w2 // 4, -1)
    p = MERGER + "mlp.0."
    y = _ln(z, W[p + "pre_norm.weight"], W[p + "pre_norm.bias"], vs.eps)
    out = _lin(gelu(_lin(y, W, p + "linear_1")), W, p + "linear_2")
    cap["out"] = out
    return out


# =============================================================================== on the card
def _up(n: int, q: int) -> int:
    return -(-n // q) * q


class VisionImage:
    """The vision tower's and projector's part of a MiniCPM-V image in one slice's DRAM
    (docs/minicpmv.md 4.2): the weights in MXU formats (the checkpoint's shapes, but the MLPs'
    widths padded to their chunks with zero rows and columns and the patch rows' 588 to 640;
    each LayerNorm's bias folded into the next linear's; `fmt`: their formats, formats()), in
    pieces of at most `max_piece` bytes (kernels/pieces.py: an expert slot's, so the host can
    stage them into slots); the vectors (gains, biases) in a block a layer (block `layers` is
    the window merger's attention), `lstride` bytes apart; the work area of a crop of up to
    `nmax` patches (the host's patch rows P and position-table rows E, the residual rows X and
    X2, the Q and O scratch, the KV scratch) and the projector's output rows `vis`
    (`out_rows`: the prompt's image rows). `b`: the image's allocator (qwen3._Bump), from its
    next byte.

    The pieces' addresses: `table` (the default) has the kernels read them from a table in
    DRAM (`pieces()` order: each layer block's PL, then the window merger's, the projector's
    and the patch embedding's), so the same programs run with the pieces resident (`resident`:
    laid out after each layer's vectors, the table holding their addresses, written at build)
    or staged into expert slots by the host (resident False: the pieces' bytes are the pool's
    entries, the table written per image). table False: static addresses, the resident
    pieces' (the reference the table runs are checked against, bit for bit).

    Under GALL (compiler.gall_wg: every MM in groups) the pieces are grouped (kernels/pieces.py
    wg; with the channel hash the resident ones hashed at their addresses), a head's q / k / v
    rows padded to whole groups (dq rows a head, zero rows past the head's), and the KV
    scratch in groups (kernels.vit.VitKV wg)."""

    CHUNK = 256         # the MLPs' F chunk (a 4-bit W2's column slices: whole 2D)
    WCHUNK = 512        # the window merger's (17216 -> 17408)
    WUNIT = 256         # its W1 rows a piece holds at least (a chunk then spans two pieces)
    PARTS = (768, 512, 256, 128)    # vit_merge's pieces: the widest that splits 4C (1152: 768,
    #                                 6 blocks: an even first block for a 4-bit W1)
    MAX_PIECE = 1712128             # a piece's bytes by default: qwen35-35b's fp4 expert slot
    LMATS = ("wq", "wk", "wv", "wo", "w1", "w2")

    def __init__(self, vs: VisionSpec, cfg, b, nmax: int = 2048, out_rows: int = 2048,
                 fmt: str = "int8", max_piece: int | None = None, resident: bool = True,
                 table: bool = True):
        from ..compiler import gall_wg
        from ..kernels import pieces as PC
        D = cfg.D
        self.vs, self.cfg = vs, cfg
        self.wg = wg = gall_wg(cfg)
        self.tables = {}                # the host's tables (build: "pos", the position table)
        self.fmts = self.formats(fmt)
        F_ = self.fmts
        C, H = vs.hidden, vs.heads
        self.C, self.heads, self.dv = C, H, vs.head_dim
        self.dp = _up(vs.head_dim, D)
        self.F = _up(vs.ffn, self.CHUNK)
        self.FW = _up(4 * vs.ffn, self.WCHUNK)
        self.KP = _up(3 * vs.patch ** 2, D)
        self.part = next(p for p in self.PARTS if (4 * C) % p == 0)
        self.nmax, self.out_rows = _up(nmax, D), out_rows     # V^T rows: whole D stretches
        if not (resident or table):
            raise ValueError("staged pieces are read from the table")
        self.resident, self.use_table = resident, table
        self.max_piece = mp = max_piece or self.MAX_PIECE
        HP = H * self.dp

        def mat(n, k, f, unit, parts=False):
            return PC.plan(n, k, f, D, unit, mp, parts, wg)
        dh = vs.head_dim
        # a head's q / k / v rows (in groups: padded to whole groups)
        self.dq = dq = _up(dh, Q.grp_r(F_["attn"], wg))
        # the matrices in pieces of whole units: a head's rows (q, k, v), D rows (out, the
        # embedding: a piece's output columns an MM), MLP chunks of rows (W1), column parts (W2)
        self.lmats = dict(wq=mat(H * dq, C, F_["attn"], dq), wk=mat(H * dq, C, F_["attn"], dq),
                          wv=mat(H * dq, C, F_["attn"], dq), wo=mat(C, C, F_["attn"], D),
                          w1=mat(self.F, C, F_["mlp"], self.CHUNK),
                          w2=mat(C, self.F, F_["mlp"], self.CHUNK, parts=True))
        self.mmats = {"wm": dict(w1=mat(self.FW, 4 * C, F_["wm"], self.WUNIT),
                                 w2=mat(C, self.FW, F_["wm"], self.WCHUNK, parts=True)),
                      "proj": dict(w1=mat(4 * C, 4 * C, F_["proj"], self.CHUNK),
                                   w2=mat(vs.out, 4 * C, F_["proj"], self.CHUNK, parts=True))}
        self.emat = mat(C, self.KP, F_["embed"], D)
        # the table's order: per layer block its matrices' pieces (PL words), then the window
        # merger's, the projector's, the embedding's
        self.lfirst, a = {}, 0
        for n in self.LMATS:
            self.lfirst[n], a = a, a + len(self.lmats[n].groups)
        self.PL = a
        self.mfirst, a = {}, (vs.layers + 1) * self.PL
        for m in ("wm", "proj"):
            for n in ("w1", "w2"):
                self.mfirst[m, n], a = a, a + len(self.mmats[m][n].groups)
        self.efirst, self.npieces = a, a + len(self.emat.groups)
        # a layer block: its vectors, then (resident) its pieces
        from .qwen3 import _Bump
        lb = _Bump()
        self.lofs = dict(ln1w=lb.alloc(4 * C), ln2w=lb.alloc(4 * C), bq=lb.alloc(4 * C),
                         bk=lb.alloc(4 * C), bv=lb.alloc(4 * C), bo=lb.alloc(4 * C),
                         b1=lb.alloc(4 * self.F), b2=lb.alloc(4 * C))
        self.lpiece = {}                # (matrix, piece) -> offset in the block (resident)
        if resident:
            lb.next = _up(lb.next, 4096)
            for n in self.LMATS:
                for i in range(len(self.lmats[n].groups)):
                    lay = self.lmats[n]
                    self.lpiece[n, i] = lb.alloc(_up(lay.piece_bytes(i), lay.unit))
        self.lstride = _up(lb.next, 4096)
        self.layer0 = _up(b.next, 4096)
        b.next = self.layer0 + (vs.layers + 1) * self.lstride
        self.start = self.layer0
        self.wm = dict(pre_w=b.alloc(16 * C), pre_b=b.alloc(16 * C), b1=b.alloc(4 * self.FW),
                       b2=b.alloc(4 * C))
        self.post = (b.alloc(4 * C), b.alloc(4 * C))
        self.proj = dict(pre_w=b.alloc(16 * C), pre_b=b.alloc(16 * C), b1=b.alloc(16 * C),
                         b2=b.alloc(4 * vs.out))
        self.emb = dict(bp=b.alloc(4 * C))
        self.table = b.alloc(4 * self.npieces)
        self.mpiece = {}                # (merge, matrix, piece) / ("emb", piece) -> address
        if resident:
            b.next = _up(b.next, 4096)
            for m in ("wm", "proj"):
                for n in ("w1", "w2"):
                    lay = self.mmats[m][n]
                    for i in range(len(lay.groups)):
                        b.next = _up(b.next, lay.unit)
                        self.mpiece[m, n, i] = b.alloc(_up(lay.piece_bytes(i), lay.unit))
            for i in range(len(self.emat.groups)):
                b.next = _up(b.next, self.emat.unit)
                self.mpiece["emb", i] = b.alloc(_up(self.emat.piece_bytes(i), self.emat.unit))
        self.weights_end = b.next
        # the RLD bound of a piece's address: the resident pieces' span (set_bound: the slots')
        self.bound = (self.layer0, max(self.weights_end - 1, self.layer0))
        n = self.nmax
        self.work = dict(P=b.alloc(4 * n * self.KP), E=b.alloc(4 * n * C),
                         X=b.alloc(4 * n * C), X2=b.alloc(4 * n // 4 * C),
                         Q=b.alloc(4 * n * HP), O=b.alloc(4 * n * C),
                         vis=b.alloc(4 * out_rows * vs.out))
        self.kvh = {h: {"k": b.alloc(n * self.dp), "ks": b.alloc(4 * n),
                        "vt": b.alloc(self.dp * n), "vs": b.alloc(4 * n)} for h in range(H)}
        self.end = b.next

    KINDS = ("attn", "mlp", "wm", "proj", "embed")

    @classmethod
    def formats(cls, fmt: str) -> dict:
        """The weights' MXU formats by kind: "int8" (every kind), or "kind=fmt,..." over the
        others' int8 (attn: the encoder layers' and the window merger's attention; mlp: the
        encoder layers' MLPs; wm: the window merger's MLP; proj: the projector; embed: the
        patch embedding)."""
        if "=" not in fmt:
            return dict.fromkeys(cls.KINDS, fmt)
        out = dict.fromkeys(cls.KINDS, "int8")
        for part in fmt.split(","):
            k, f = part.split("=")
            if k not in out:
                raise ValueError(f"vision format kind {k!r}: one of {cls.KINDS}")
            out[k] = f
        return out

    def set_bound(self, lo: int, hi: int) -> None:
        """Staged pieces: the addresses a piece may be at (the slot region's), RLD's bound."""
        self.bound = (int(lo), int(hi))

    # ---- the pieces
    def pieces(self) -> list:
        """Every piece in the table's order: (key, Layout, piece index), key ("l", layer,
        matrix), ("m", "wm" / "proj", matrix) or ("emb",)."""
        out = []
        for li in range(self.vs.layers + 1):
            for n in self.LMATS:
                lay = self.lmats[n]
                out += [(("l", li, n), lay, i) for i in range(len(lay.groups))]
        for m in ("wm", "proj"):
            for n in ("w1", "w2"):
                lay = self.mmats[m][n]
                out += [(("m", m, n), lay, i) for i in range(len(lay.groups))]
        out += [(("emb",), self.emat, i) for i in range(len(self.emat.groups))]
        return out

    def resident_addresses(self) -> np.ndarray:
        """The table of the resident pieces' addresses (uint32, pieces() order)."""
        if not self.resident:
            raise ValueError("the pieces are not resident")
        out = []
        for li in range(self.vs.layers + 1):
            for n in self.LMATS:
                out += [self.layer0 + li * self.lstride + self.lpiece[n, i]
                        for i in range(len(self.lmats[n].groups))]
        for m in ("wm", "proj"):
            for n in ("w1", "w2"):
                out += [self.mpiece[m, n, i] for i in range(len(self.mmats[m][n].groups))]
        out += [self.mpiece["emb", i] for i in range(len(self.emat.groups))]
        return np.array(out, np.uint32)

    # ---- descriptors
    @staticmethod
    def _v(base, a, n):
        from ..compiler import Affine, Tensor
        return Tensor(Affine.of(base) + a, (n,), (1,))

    def _table(self, first: int, n: int, li=None):
        from ..compiler import Affine, Tensor
        from ..kernels import pieces as PC
        at = Affine(self.table + 4 * first)
        if li is not None:
            at = at + Affine.of(li) * (4 * self.PL)
        return PC.Table(Tensor(at, (n,), (1,)), n, self.bound)

    def layer(self, li):
        """Encoder layer li's weights (li an int or a run-time value; vs.layers: the window
        merger's attention, whose block has the layers' layout)."""
        from types import SimpleNamespace
        from ..compiler import Affine
        from ..kernels import pieces as PC
        base = Affine(self.layer0) + Affine.of(li) * self.lstride
        o, C = self.lofs, self.C
        if self.use_table:
            t = self._table(0, self.PL, li)
            ws = {n: PC.PQ(self.lmats[n], table=t, first=self.lfirst[n]) for n in self.LMATS}
        else:
            ws = {n: PC.PQ(self.lmats[n], bases=[base + self.lpiece[n, i] for i in
                                                 range(len(self.lmats[n].groups))])
                  for n in self.LMATS}
        return SimpleNamespace(
            ln1w=self._v(base, o["ln1w"], C), ln2w=self._v(base, o["ln2w"], C),
            bq=self._v(base, o["bq"], C), bk=self._v(base, o["bk"], C),
            bv=self._v(base, o["bv"], C), bo=self._v(base, o["bo"], C),
            b1=self._v(base, o["b1"], self.F), b2=self._v(base, o["b2"], C), **ws)

    def _mat(self, m, n, lay):
        from ..kernels import pieces as PC
        if self.use_table:
            first = self.efirst if m == "emb" else self.mfirst[m, n]
            return PC.PQ(lay, table=self._table(first, len(lay.groups)), first=0)
        return PC.PQ(lay, bases=[self.mpiece[(m, i) if m == "emb" else (m, n, i)]
                                 for i in range(len(lay.groups))])

    def merge_weights(self, which: str):
        """The window merger's MLP ("wm") or the projector's ("proj") for vit_merge."""
        from types import SimpleNamespace
        m, C, mm = getattr(self, which), self.C, self.mmats[which]
        return SimpleNamespace(pre_w=self._v(0, m["pre_w"], 4 * C),
                               pre_b=self._v(0, m["pre_b"], 4 * C),
                               w1=self._mat(which, "w1", mm["w1"]),
                               b1=self._v(0, m["b1"], mm["w1"].n),
                               w2=self._mat(which, "w2", mm["w2"]),
                               b2=self._v(0, m["b2"], mm["w2"].n))

    def descriptors(self):
        """The phases' `v`: the work area, the embedding's weights, the post norm."""
        from types import SimpleNamespace
        from ..kernels.vit import VitKV
        from .qwen3 import _tdesc as t
        w, n, C = self.work, self.nmax, self.C
        return SimpleNamespace(
            C=C, heads=self.heads, dp=self.dp, dv=self.dv, dq=self.dq,
            P=t(w["P"], (n, self.KP)), E=t(w["E"], (n, C)), X=t(w["X"], (n, C)),
            X2=t(w["X2"], (n // 4, C)),
            Q=[t(w["Q"] + 4 * h * n * self.dp, (n, self.dp)) for h in range(self.heads)],
            O=t(w["O"], (n, C)), vis=t(w["vis"], (self.out_rows, self.vs.out)),
            kv=VitKV(self.kvh, n, self.dp, self.cfg.D, 1, 0, dv=self.dv, wg=self.wg),
            emb=SimpleNamespace(wp=self._mat("emb", None, self.emat),
                                bp=self._v(0, self.emb["bp"], C)),
            post=(self._v(0, self.post[0], C), self._v(0, self.post[1], C)))

    # ---- contents
    def _matrix(self, W, key) -> np.ndarray:
        """The float matrix of piece key `key` (pieces()), padded as laid out, from the
        checkpoint's tensors W (load_vision's, or a mapping that reads each when asked)."""
        vs, C = self.vs, self.C
        f32 = np.float32

        def pad(a, shape):
            a = np.asarray(a, f32)
            out = np.zeros(shape, f32)
            out[tuple(slice(0, n) for n in a.shape)] = a
            return out
        if key[0] == "l":
            _, li, n = key
            p = f"{TOWER}vit_merger." if li == vs.layers else f"{TOWER}encoder.layers.{li}."
            if n in ("wq", "wk", "wv"):
                w = np.asarray(W[p + f"self_attn.{n[1]}_proj.weight"], f32)
                if self.dq == vs.head_dim:
                    return w
                out = np.zeros((self.heads, self.dq, C), f32)   # (a head's rows: whole groups)
                out[:, :vs.head_dim] = w.reshape(self.heads, vs.head_dim, C)
                return out.reshape(-1, C)
            if n == "wo":
                return W[p + "self_attn.out_proj.weight"]
            shape = (self.F, C) if n == "w1" else (C, self.F)
            if li == vs.layers:         # (the window merger's block has no MLP: zeros)
                return np.zeros(shape, f32)
            return pad(W[p + ("mlp.fc1.weight" if n == "w1" else "mlp.fc2.weight")], shape)
        if key[0] == "m":
            _, m, n = key
            p, F = (TOWER + "vit_merger.", self.FW) if m == "wm" else (MERGER + "mlp.0.", 4 * C)
            if n == "w1":
                return pad(W[p + "linear_1.weight"], (F, 4 * C))
            return pad(W[p + "linear_2.weight"], (self.mmats[m]["w2"].n, F))
        wp = np.asarray(W[TOWER + "embeddings.patch_embedding.weight"], f32).reshape(C, -1)
        return pad(wp, (C, self.KP))

    def _vectors(self, W: dict) -> list:
        """The vectors (address, values): the checkpoint's tensors (load_vision's), each
        LayerNorm's bias folded into the next linear's bias (float64)."""
        vs = self.vs

        def pad(a, n):
            out = np.zeros(n, np.float32)
            out[:len(a)] = a
            return out

        def folded(p, ln):      # the linear's bias with the LayerNorm's before it folded in
            return W[p + ".bias"] + W[p + ".weight"].astype(np.float64) @ W[ln + ".bias"]
        vecs = []
        o = self.lofs
        for li in range(vs.layers + 1):
            base = self.layer0 + li * self.lstride
            merger = li == vs.layers
            p = f"{TOWER}vit_merger." if merger else f"{TOWER}encoder.layers.{li}."
            for n in "qkv":
                vecs.append((base + o["b" + n],
                             folded(p + f"self_attn.{n}_proj", p + "layer_norm1")))
            vecs += [(base + o["bo"], W[p + "self_attn.out_proj.bias"]),
                     (base + o["ln1w"], W[p + "layer_norm1.weight"])]
            if merger:
                continue
            vecs += [(base + o["ln2w"], W[p + "layer_norm2.weight"]),
                     (base + o["b1"], pad(folded(p + "mlp.fc1", p + "layer_norm2"), self.F)),
                     (base + o["b2"], W[p + "mlp.fc2.bias"])]
        for m, p, F in (("wm", TOWER + "vit_merger.", self.FW),
                        ("proj", MERGER + "mlp.0.", 4 * self.C)):
            d = getattr(self, m)
            vecs += [(d["pre_w"], W[p + "pre_norm.weight"]), (d["pre_b"], W[p + "pre_norm.bias"]),
                     (d["b1"], pad(W[p + "linear_1.bias"], F)),
                     (d["b2"], W[p + "linear_2.bias"])]
        vecs += [(self.post[0], W[TOWER + "post_layernorm.weight"]),
                 (self.post[1], W[TOWER + "post_layernorm.bias"]),
                 (self.emb["bp"], W[TOWER + "embeddings.patch_embedding.bias"])]
        return vecs

    def used(self) -> list:
        """The pieces (pieces() indices) the programs read: all but the window merger's block's
        MLP (its attention block has the layers' layout; vit_out_mlp runs it without the MLP)."""
        return [i for i, (key, _, _) in enumerate(self.pieces())
                if not (key[0] == "l" and key[1] == self.vs.layers and key[2] in ("w1", "w2"))]

    def piece_sizes(self) -> list:
        """Each piece's bytes (pieces() order)."""
        return [lay.piece_bytes(i) for _, lay, i in self.pieces()]

    def piece(self, W, i: int) -> np.ndarray:
        """Piece i's bytes (uint8, pieces() order): its matrix quantized (the last one's kept
        for the next piece of it)."""
        from ..kernels import pieces as PC
        key, lay, pi = self.pieces()[i]
        c = getattr(self, "_packed", None)
        if c is None or c[0] != key:
            q, s = Q.quantize_mxu(np.asarray(self._matrix(W, key), np.float32), lay.fmt,
                                  self.cfg.D)
            c = self._packed = (key, PC.pack(lay, q, s))
        return c[1][pi]

    def piece_bytes(self, W: dict) -> list:
        """Every piece's bytes in pieces() order (uint8, each its Layout's piece_bytes): the
        pool's vision entries, or the resident pieces."""
        out = [self.piece(W, i) for i in range(self.npieces)]
        self._packed = None
        return out

    def build(self, W: dict, put) -> None:
        """Write the vectors and (resident) the pieces and their table (W: load_vision's
        tensors) with put(addr, array); keep the position table for the host's rows
        (host_rows)."""
        for addr, v in self._vectors(W):
            put(addr, np.asarray(v, np.float32))
        if self.resident:
            from ..kernels import pieces as PC
            chash = self.wg > 1 and self.cfg.CHASH
            for (_, lay, i), a, p in zip(self.pieces(), self.resident_addresses(),
                                         self.piece_bytes(W)):
                put(int(a), PC.hash_piece(lay, i, p, int(a)) if chash else p)
            put(self.table, self.resident_addresses())
        self.tables["pos"] = np.asarray(W[TOWER + "embeddings.position_embedding.weight"],
                                        np.float32)

    # ---- the runs
    def programs(self, N: int, block: int = 256, R: int | None = None,
                 proj_gelu=None) -> tuple:
        """One crop of N patches (h x w, both multiples of 4, its rows in card_order): ({phase:
        (program, run_args)}, the runs in order [(phase, run argument values)]): a program per
        phase and patch count (not grid: the order makes the merges' windows consecutive), the
        encoder layers' at a run-time layer index ("layer"), the projector's at a run-time
        output row ("vis_row")."""
        from ..compiler import Affine, RunVar, Tensor
        from ..kernels import vit as VK
        from ..kernels.lib import gelu_tanh
        vs, cfg = self.vs, self.cfg
        R = R or min(16, cfg.act_rows)
        RM = max(1, R // 2)          # the merges' windows a group: their [R, chunk] MLP tiles
        G = min(16, cfg.act_rows) // 4 * 4  # the window attention's rows an MM (G / 4 windows)
        N2 = N // 4
        if N > self.nmax:
            raise ValueError(f"a crop of {N} patches: the image's work area holds {self.nmax}")
        if N % 16:
            raise ValueError(f"a crop of {N} patches: h and w multiples of 4 (two merges)")
        eps, sc = vs.eps, 1.0 / math.log(2.0) / math.sqrt(vs.head_dim)
        v = self.descriptors()
        lw = self.layer(RunVar("layer", bound=vs.layers))
        mlw = self.layer(vs.layers)

        def tr(kernel, **kw):
            b = kernel.trace(cfg, 0, {"v": v, **kw})
            return b.finish(), list(b.run_args)
        row = RunVar("vis_row", bound=self.out_rows)
        vis = Tensor(Affine.of(v.vis.base) + row * (4 * vs.out), v.vis.shape, v.vis.strides)
        P = {"embed": tr(VK.vit_embed, N=N, R=R),
             "qkv": tr(VK.vit_qkv, lw=lw, src=v.X, N=N, R=R, eps=eps),
             "attend": tr(VK.vit_attend, N=N, scale=sc, block=block),
             "out": tr(VK.vit_out_mlp, lw=lw, dst=v.X, N=N, R=R, eps=eps, chunk=self.CHUNK),
             "mqkv": tr(VK.vit_qkv, lw=mlw, src=v.X, N=N, R=R, eps=eps),
             "wattend": tr(VK.vit_window_attend, N=N, G=G, scale=sc),
             "mout": tr(VK.vit_out_mlp, lw=mlw, dst=v.X, N=N, R=R, eps=eps, chunk=self.CHUNK,
                        mlp=False),
             "wmerge": tr(VK.vit_merge, mw=self.merge_weights("wm"), src=v.X, dst=v.X2,
                          N4=N2, R=RM, eps=eps, chunk=self.WCHUNK, mean=True, part=self.part),
             "qkv2": tr(VK.vit_qkv, lw=lw, src=v.X2, N=N2, R=R, eps=eps),
             "attend2": tr(VK.vit_attend, N=N2, scale=sc, block=block),
             "out2": tr(VK.vit_out_mlp, lw=lw, dst=v.X2, N=N2, R=R, eps=eps, chunk=self.CHUNK),
             "last": tr(VK.vit_out_mlp, lw=lw, dst=v.X2, N=N2, R=R, eps=eps, chunk=self.CHUNK,
                        post=v.post),
             "proj": tr(VK.vit_merge, mw=self.merge_weights("proj"), src=v.X2, dst=vis,
                        N4=N2 // 4, R=RM, eps=eps, chunk=self.CHUNK, mean=False,
                        part=self.part, act=proj_gelu or gelu_tanh)}
        runs = [("embed", {})]
        for i in range(vs.insert + 1):
            runs += [("qkv", {"layer": i}), ("attend", {}), ("out", {"layer": i})]
        runs += [("mqkv", {}), ("wattend", {}), ("mout", {}), ("wmerge", {})]
        for i in range(vs.insert + 1, vs.layers):
            runs += [("qkv2", {"layer": i}), ("attend2", {}),
                     ("last" if i == vs.layers - 1 else "out2", {"layer": i})]
        runs.append(("proj", {"vis_row": 0}))
        return P, runs

    def host_rows(self, P: np.ndarray, size) -> tuple:
        """The host's rows of a crop (P raster order): its patch rows padded to KP columns, and
        its position rows (the position table's rows at its NaViT ids: a table lookup, no
        arithmetic), both in card_order."""
        h, w = size
        o = card_order(h, w)
        p = np.zeros((h * w, self.KP), np.float32)
        p[:, :P.shape[1]] = P[o]
        return p, self.tables["pos"][position_ids(h, w, self.vs.side)[o]]

    def encode(self, backend, P: np.ndarray, size, vis_row: int = 0, proj_gelu=None,
               cache: dict | None = None, s: int = 0) -> np.ndarray:
        """One crop through the tower and projector on `backend` (a qwen3 backend: write, run
        or start / wait, read; one that runs words gets each program assembled once); its
        rows go to vis[vis_row:]; returns them ([h w / 16, out] fp32). cache: the programs by
        patch count (and the projector's GELU)."""
        from ..compiler import Words, arg_words
        from .. import isa as I
        h, w = size
        key = (h * w, proj_gelu)
        progs = None if cache is None else cache.get(key)
        if progs is None:
            P_, runs = self.programs(h * w, proj_gelu=proj_gelu)
            if getattr(backend, "runs_words", False):
                P_ = {k: (Words.of(I.assemble(p), ra), ra) for k, (p, ra) in P_.items()}
            else:
                P_ = {k: ([p], ra) for k, (p, ra) in P_.items()}
            progs = (P_, runs)
            if cache is not None:
                cache[key] = progs
        P_, runs = progs
        p, e = self.host_rows(P, size)
        backend.write(s, self.work["P"], p)
        backend.write(s, self.work["E"], e)
        start = getattr(backend, "start", None)
        for name, vals in runs:
            prog, ra = P_[name]
            if "vis_row" in vals:
                vals = {"vis_row": vis_row}
            if start is None:
                backend.run(prog, args=arg_words(ra, vals))
            else:
                start(prog, args=arg_words(ra, vals))
                backend.wait()
        n = h * w // 16
        return backend.read(s, self.work["vis"] + 4 * vis_row * self.vs.out,
                            4 * n * self.vs.out).view(np.float32).reshape(n, self.vs.out)


# =============================================================================== the model's image
from . import qwen35 as QW              # noqa: E402  (qwen35 does not import this module)
from .. import language as ol           # noqa: E402
# the family's module-level names, which tools and tiled.py look up by the image's module
from .qwen35 import ATTN_BLOCK, RUN_ROWS, emulated_logits, reference_logits  # noqa: E402
FAMILY = (ATTN_BLOCK, RUN_ROWS, emulated_logits, reference_logits)


@dataclass(frozen=True)
class Spec(QW.Spec):
    """MiniCPM-V 4.7: its language model qwen35's Spec (Qwen3.5-MoE), its vision side
    `vision` (VisionSpec) in the model's image before the expert slots (Image): the tower's
    weights in `vision_fmt` (VisionImage.formats), a crop's work area of `vision_patches`
    patches, `vision_rows` rows of image tokens (the projector's output rows a prompt's
    images take, docs/minicpmv.md 4.3). `vision_staged` (a MoE's; the default since session
    mcpv2, docs/minicpmv.md 6: decode +8.4%, an image +0.3-0.9 s): the tower's weights are the
    expert pool's entries past the experts, placed in expert slots for an image's runs (the
    expert server's place) instead of resident (docs/minicpmv.md 4.2)."""
    vision: VisionSpec | None = VisionSpec()   # None: the language model alone
    vision_fmt: str = "int8"
    vision_patches: int = 2048
    vision_rows: int = 2048
    vision_staged: bool = True

    @staticmethod
    def from_hf(model_dir) -> "Spec":
        from dataclasses import fields
        base = QW.Spec.from_hf(model_dir)
        return Spec(**{f.name: getattr(base, f.name) for f in fields(QW.Spec)},
                    vision=VisionSpec.from_hf(model_dir))

    def image(self, cfg, cap: int, batch: int = 1, rows: int = 1, wformat: str = "int8",
              head_format: str | None = None, lookup: bool | str = False,
              experts: int | None = None, embed_host: bool | None = None,
              formats: str | None = None, prefill_rows: int | None = None,
              high: bool | None = None) -> "Image":
        return QW.fit_high(lambda h: Image(self, cfg, cap, batch, rows, wformat, head_format,
                                           lookup, experts, embed_host, formats, prefill_rows,
                                           h), cfg, high)


class Image(QW.Image):
    """qwen35's image with MiniCPM-V's vision side (`vision`: a VisionImage, its weights and a
    crop's work area) after the model's DRAM and before the expert slots; its `vis` rows are
    the prompt's image rows, which vis_copy_run puts in a prefill chunk's residual rows
    (layer 0's input, qwen35_layer_run embedded). The text path's programs are qwen35's: a
    text-only prompt runs exactly qwen35-35b's."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        v = self.vision
        if v is not None and not v.resident:        # staged: the pieces in expert slots
            L = self.offload
            if L is None:
                raise ValueError("staged vision weights need the expert slots (a MoE model)")
            if max(v.piece_sizes()) > L.slot_bytes:
                raise ValueError(f"vision pieces of up to {max(v.piece_sizes())} bytes, slots "
                                 f"of {L.slot_bytes}")
            v.set_bound(*L.slot_bounds)
            if v.wg > 1 and self.cfg.CHASH:     # (the server hashes a piece's data in a slot)
                G, pcs = L.layers * L.E, v.pieces()
                self.entry_lanes = lambda g: pcs[g - G][1].data_regions(pcs[g - G][2])

    def check_room(self) -> None:
        """Staged vision's pieces against the expert slots past k a layer (ExpertServer.place's
        room): a ValueError when they do not fit. An engine's image is checked (streamed.engine
        after it opens, Vision); not every image: streamed.probe's has k slots a layer, the
        layout alone (session mcpv3's first run, 2026-10-13: the check in __init__ refused the
        probe, 438 pieces against 0 slots, before the engine's 36 a layer)."""
        v, L = self.vision, self.offload
        if v is None or v.resident:
            return
        free = sum(max(0, n - L.k) for _, n in L.slots)
        if len(v.used()) > free:
            raise ValueError(f"staged vision: {len(v.used())} pieces, {free} expert slots past "
                             f"k = {L.k} a layer (more slots, or vision_staged False)")

    def before_slots(self, b) -> None:
        sp = self.spec
        mp = None                       # staged: pieces of at most an expert slot's bytes
        staged = sp.vision is not None and sp.vision_staged and sp.moe is not None
        if staged:
            from . import moe as MO
            from ..compiler import gall_wg
            mp = min(VisionImage.MAX_PIECE,
                     MO.ExpertFormat(sp.hidden, sp.moe.ffn, self.cfg.D, self.wformat,
                                     gall_wg(self.cfg)).nbytes)
        self.vision = None if sp.vision is None else \
            VisionImage(sp.vision, self.cfg, b, nmax=sp.vision_patches, out_rows=sp.vision_rows,
                        fmt=sp.vision_fmt, resident=not staged, max_piece=mp)
        if self.vision is not None:     # (an attribute of the image's own: the image cache keeps
            self.vis_tables = self.vision.tables    # its entries, opentpu/imgcache.py)

    @property
    def pool_extra(self):
        """Staged vision: (the pool's entries past the experts, their key: moe.pool_key's
        "vision"), else None. Entry layers x E + i is piece i (VisionImage.pieces())."""
        v = self.vision
        if v is None or v.resident:
            return None
        h = hashlib.blake2b(digest_size=16)
        h.update(json.dumps([v.fmts, v.max_piece, v.piece_sizes(),
                             [[list(k), lay.n, lay.k, i] for k, lay, i in v.pieces()]]).encode())
        return v.npieces, {"entries": v.npieces, "format": self.spec.vision_fmt,
                           "pieces": h.hexdigest()}

    def expert(self, W, g: int) -> np.ndarray:
        """Global expert g in its slot's bytes; past the experts (staged vision), piece
        g - layers x E zero-padded to a slot (the vision tensors read as asked for)."""
        L = self.offload
        G = L.layers * L.E
        if g < G:
            return super().expert(W, g)
        if getattr(self, "_vis_w", None) is None:
            self._vis_w = VisionWeights(W.model_dir) if hasattr(W, "model_dir") else W
        out = np.zeros(L.slot_bytes, np.uint8)
        p = self.vision.piece(self._vis_w, g - G)
        out[:p.size] = p
        return out

    def build(self, W: dict) -> list:
        """qwen35's DRAM image, then the vision side's weights: load_vision's tensors of the
        checkpoint W reads (LazyWeights' model_dir), or W's own under their names."""
        imgs = super().build(W)
        if self.vision is None:
            return imgs
        Wv = (load_vision(W.model_dir) if self.vision.resident else
              VisionWeights(W.model_dir)) if hasattr(W, "model_dir") else W

        def put(addr, a):
            v = np.ascontiguousarray(a).view(np.uint8).reshape(-1)
            imgs[0][addr:addr + v.size] = v
        self.vision.build(Wv, put)
        return imgs

    def descriptors(self, sid: int, slot: int = 0):
        m = super().descriptors(sid, slot)
        if self.vision is not None:
            m.vis = self.vision.descriptors().vis
        return m

    def compile_vis_run(self, n: int, em: bool = False):
        """(programs, run_args): vis_copy_run of n rows, the chunk's row ("row") and the vis
        row ("src") run arguments."""
        from ..compiler import RunVar
        b = vis_copy_run.trace(self.cfg, 0, {"m": self.descriptors(0), "n": n,
                                             "row": RunVar("row", self.prefill_rows),
                                             "src": RunVar("src", self.spec.vision_rows),
                                             "em": em})
        return [b.finish()], list(b.run_args)

    def rope_table(self) -> tuple:
        """The image's RoPE tables as the host built them (qwen3.rope_tables at slots 0 .. cap
        - 1): cos, sin [cap, rope_dim / 2]."""
        if not hasattr(self, "_rope"):
            from .qwen3 import rope_tables
            cs = [rope_tables(self.spec, p) for p in range(self.cap)]
            self._rope = (np.stack([c for c, _ in cs]), np.stack([x for _, x in cs]))
        return self._rope


@ol.jit
def vis_copy_run(m, n: int, row, src, em: bool = False):
    """n image rows of a layer-major prefill chunk: the projector's rows m.vis[src:src + n] ->
    the chunk's residual rows m.xbuf[row:row + n] (em: their records' X, moe.em_x), layer 0's
    input (qwen35_layer_run with embedded): a DMA through TMEM, no arithmetic."""
    from . import moe as MO
    X = MO.em_x(m.moe_dev, row, n, m.spec.hidden) if em else m.xbuf[row:row + n, :]
    ol.store(X, ol.load(m.vis[src:src + n, :]))


def mrope_rows(table: tuple, pos: np.ndarray) -> tuple:
    """The RoPE rows of positions pos [3, n] (t, h, w; rope_index's) from the image's tables
    (Image.rope_table): Qwen3.5's interleaved M-RoPE (mrope_section [11, 11, 10]) takes
    frequency i from h's position where i % 3 == 1, w's where i % 3 == 2, t's else, so each
    entry is the table's entry of that position and frequency: a gather, no arithmetic. A text
    position (t = h = w) gives the table's row itself, bit for bit."""
    cos, sin = table
    half = cos.shape[1]
    sec = np.arange(half) % 3
    rows = np.asarray(pos)[sec]                     # [half, n]: each frequency's position
    if rows.min() < 0 or rows.max() >= len(cos):
        raise ValueError(f"M-RoPE positions {rows.min()} .. {rows.max()} outside the table "
                         f"({len(cos)})")
    cols = np.arange(half)[:, None]
    return np.ascontiguousarray(cos[rows, cols].T), np.ascontiguousarray(sin[rows, cols].T)


# =============================================================================== the host's images
class Vision:
    """MiniCPM-V's images on an Engine whose image is a minicpmv.Image (docs/minicpmv.md 4.3):
    each image's crops through the card's tower and projector into the image's vis rows
    (encode), each prompt token's vis row (rows), and the RoPE table's rows at every slot's
    M-RoPE positions (positions). The host only moves data: the patch rows it cut, the vis
    rows the card computed (by their index), the table entries the image was built with."""

    def __init__(self, eng):
        img = eng.image
        if getattr(img, "vision", None) is None:
            raise ValueError("the engine's image has no vision side (minicpmv.Spec.vision)")
        img.check_room()
        self.eng, self.img, self.vs = eng, img, eng.spec.vision
        self.table = img.rope_table()
        self.cache = {}                     # VisionImage's programs by patch count
        self.dev = np.tile(np.arange(img.cap), (3, 1))  # the device table's positions, a slot
        self.seconds = 0.0                  # the encodes' wall time

    def encode(self, images: list, row: int = 0) -> int:
        """Each ImageInput's crops in order (thumbnail, slices) into vis rows from `row`: their
        rows follow one another as the prompt's image tokens do. Returns the rows."""
        import time
        t, r = time.time(), row
        n = sum(im.tokens for im in images)
        v = self.img.vision
        if row + n > v.out_rows:
            raise ValueError(f"{n} image rows from {row}: the image holds "
                             f"{v.out_rows} (Spec.vision_rows)")
        if not images:
            return 0
        if not v.resident:
            self.stage()
        try:
            for im in images:
                for P, size in zip(im.patches, im.sizes):
                    v.encode(self.eng.backend, P, size, vis_row=r, cache=self.cache)
                    r += size[0] * size[1] // 16
        finally:
            if not v.resident:
                self.eng.server.release()
        self.seconds += time.time() - t
        return r - row

    def stage(self) -> None:
        """Staged weights (Spec.vision_staged): the pieces the programs read placed in expert
        slots by the expert server (the pool's entries layers x E + i, place), and their table
        written: the address of each (the first's for a piece the programs never read)."""
        v, srv = self.img.vision, self.eng.server
        G = srv.L.layers * srv.L.E
        used, sizes = v.used(), v.piece_sizes()
        at = srv.place([G + i for i in used], {G + i: sizes[i] for i in used})
        table = np.full(v.npieces, at[0], np.uint32)
        table[used] = at
        self.eng.backend.write(0, v.table, table)

    def prompt(self, ids, images: list, n: int = 0) -> list | None:
        """A prompt's image side (ids: the whole sequence's token ids, its images in order):
        the RoPE table's rows at its positions (positions), the images with tokens in ids[n:]
        (what is fed) through the tower into vis rows from 0 (an image that started before n
        whole), and each of ids[n:]'s vis row (-1: text): prefill_layers' vis. Without images
        the table's own rows (pristine) and None."""
        if not images:
            self.pristine()
            return None
        self.positions(ids, [s for im in images for s in im.sizes])
        at, j, k = [], 0, 0                 # each image token's (image, its k-th row)
        for t in ids:
            if int(t) == self.vs.image_token:
                if k == images[j].tokens:
                    j, k = j + 1, 0
                at.append((j, k))
                k += 1
            else:
                at.append(None)
        new = sorted({a[0] for a in at[n:] if a is not None})
        base, row = {}, 0
        for j in new:
            base[j], row = row, row + images[j].tokens
        self.encode([images[j] for j in new])
        return [-1 if a is None else base[a[0]] + a[1] for a in at[n:]]

    def rows(self, ids, row: int = 0) -> list:
        """Each token's vis row (prefill_layers' vis): the image tokens' in order from `row`,
        -1 for the others."""
        out, r = [], row
        for t in ids:
            if int(t) == self.vs.image_token:
                out.append(r)
                r += 1
            else:
                out.append(-1)
        return out

    def positions(self, ids, sizes) -> int:
        """The RoPE table's rows for a sequence (every token since slot 0, every crop's (h, w)
        in order): slot s < T at rope_index's position, s >= T at s + delta (the decode's).
        Writes the rows that differ from what the table holds (none for text after text: the
        image's own rows); returns delta."""
        pos, delta = rope_index(ids, sizes, self.vs)
        cap = self.img.cap
        T = pos.shape[1]
        if T > cap:
            raise ValueError(f"{T} tokens: the cache holds {cap}")
        self._write(np.concatenate([pos, np.tile(np.arange(T, cap) + delta, (3, 1))], axis=1))
        return int(delta)

    def pristine(self) -> None:
        """The table's own rows back at every slot (a text-only sequence: a chat's reset)."""
        self._write(np.tile(np.arange(self.img.cap), (3, 1)))

    def _write(self, want: np.ndarray) -> None:
        cap = self.img.cap
        diff = (want != self.dev).any(0)
        lk = self.img.lookup
        half = lk["half"]
        i = 0
        while i < cap:
            if not diff[i]:
                i += 1
                continue
            j = i
            while j < cap and diff[j]:
                j += 1
            c, s_ = mrope_rows(self.table, want[:, i:j])
            for sl in range(self.eng.cfg.S):
                self.eng.backend.write(sl, lk["cos_t"] + 4 * half * i, c)
                self.eng.backend.write(sl, lk["sin_t"] + 4 * half * i, s_)
            i = j
        self.dev = want
