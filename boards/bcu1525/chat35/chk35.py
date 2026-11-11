"""chk35.py B: bucket B's decode step in the program cache: its key, whether it loads, and its wait entries (computed
or from the disk: waits.stats)."""
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warm35 as W5

b = int(sys.argv[1])
args = pickle.loads((W5.HERE / "warm35.pkl").read_bytes())
(spec, cfg, cap, batch, rows, image_kw, stream_fill, layout, block, conv_lo, chain, debug, split, waitc, samps) = args
from opentpu import progcache as PC
from opentpu import waits as WC
from opentpu.llm import qwen3 as Q
Q._worker_init(spec, cfg, cap, batch, rows, block, image_kw, stream_fill)
image = Q._WORKER[0]
lo = max((b - 1) * block, conv_lo)
what = ("decode", b, lo) + (("fill",) if stream_fill else ())
k = PC.key(layout, what)
f = PC.cache_dir() / k[:2] / f"{k}.npz"
print("what", what, "key", k, "exists", f.exists(), "waitc", waitc)
v = PC._load(f)
print("loaded", None if v is None else type(v[0]), None if v is None else len(v[0]))
t = time.perf_counter()
W5.decode_waits(layout, what, image.cfg)
print("decode_waits", round(time.perf_counter() - t, 3), "s", dict(WC.stats))
