"""purge35x2.py [PKL] [--dry]: every bucket's resident decode step out of the program cache (OTPU_PROG_CACHE), for the
image of PKL's arguments (default warm35x2.pkl beside this file): warm35.py before 11-03 cached _worker_decode's words
of slice 0 there at S > 1, which the cards ran with HALT on slices 1-3 (the chat's first prompt token step with its
position as arguments never ended). warm35.py (w35x2.sh) then compiles them again, every slice's. Run it from the tree's
root with PYTHONPATH=. and the chat's keyed environment (OTPU_TARGET bcu1525-x2:st16: w35x2.sh's)."""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> int:
    dry = "--dry" in sys.argv
    paths = [a for a in sys.argv[1:] if a != "--dry"]
    args = pickle.loads(Path(paths[0] if paths else HERE / "warm35x2.pkl").read_bytes())
    from opentpu import progcache as PC
    from opentpu.llm import generate as G
    cap, stream_fill, layout, block, conv_lo = args[2], args[6], args[7], args[8], args[9]
    d = PC.cache_dir()
    if d is None:
        raise SystemExit("purge35x2: no program cache (OTPU_PROG_CACHE)")
    found = 0
    for b in range(1, G.buckets(cap, block) + 1):
        what = ("decode", b, max((b - 1) * block, conv_lo)) + (("fill",) if stream_fill else ())
        k = PC.key(layout, what)
        f = d / k[:2] / f"{k}.npz"
        if f.is_file():
            found += 1
            if not dry:
                f.unlink()
    print(f"purge35x2: {found} of {G.buckets(cap, block)} buckets' decode steps {'found' if dry else 'removed'} "
          f"in {d} (S {args[1].S})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
