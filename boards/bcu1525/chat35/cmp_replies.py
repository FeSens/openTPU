"""cmp_replies.py A.out B.out: the --plain chat replies of two otpu-chat35 / otpu-chat35x2 sessions, word for word.
CHAT35_TIMES' "[chat35] t name s" fragments are cut out; a reply is the text between a "> " prompt and its
"[TTFT ...]" line; CHAT35_SNAP's notice lines go first (with the timing line above cut, one would end the reply).
Prints each turn's match and the timing lines side by side."""
import re
import sys

TIMED = re.compile(r"\[chat35\]\s+[0-9.]+ \S+ [0-9.]+ s\n?")
SNAP = re.compile(r"(?m)^chat35x2: reply \d+'s context before and after it in \S+, g\d+b\.snap\.npz(\n|$)")


def turns(path):
    text = TIMED.sub("", SNAP.sub("", open(path, encoding="utf-8", errors="replace").read()))
    text = re.sub(r"\[chat35[^\]]*\] [^\d\s][^\n]*\n", "", text)         # (its warm lines)
    # the wrappers' own lines (otpu-chat35 / otpu-chat35x2), the one after the closing "> " prompt too
    # (its SAFE lines: no turn); nothing else goes
    text = re.sub(r"(?m)^(> )?otpu-chat35(x2)?: .*(\n|$)", "", text)
    out, stats = [], []
    for part in text.split("\n> ")[1:]:
        reply, _, rest = part.partition("\n[TTFT")
        out.append(reply.strip())
        stats.append(("[TTFT" + rest.split("\n", 1)[0]) if rest else "(no TTFT line)")
    return out, stats


a, sa = turns(sys.argv[1])
b, sb = turns(sys.argv[2])
ok = len(a) == len(b)
print(f"turns: {len(a)} vs {len(b)}")
for i, (x, y) in enumerate(zip(a, b)):
    same = x == y
    ok &= same
    wx, wy = x.split(), y.split()
    k = next((j for j, (p, q) in enumerate(zip(wx, wy)) if p != q), min(len(wx), len(wy)))
    print(f"turn {i + 1}: {'SAME' if same else f'DIFFER at word {k}'} ({len(wx)} / {len(wy)} words)")
    if not same:
        print("  A:", " ".join(wx[max(0, k - 5):k + 8]))
        print("  B:", " ".join(wy[max(0, k - 5):k + 8]))
    print("  A", sa[i])
    print("  B", sb[i])
print("ALL SAME" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
