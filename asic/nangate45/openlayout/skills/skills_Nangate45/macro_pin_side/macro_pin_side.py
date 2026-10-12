#!/usr/bin/env python3
"""macro_pin_side.py — detect & highlight the signal-pin side(s) of every macro.

The pin sides are taken from OpenROAD DB *geometry* (iterm getBBox), never inferred
visually, so they always reflect the macro's current placed orientation. A macro may
carry pins on MORE THAN ONE side (e.g. an SRAM with pins on both L and R) — every
such side is highlighted, not just the single dominant one. The flow:

  1. OpenROAD loads the placement (--odb, or --def + LEFs) and, for every BLOCK
     instance, counts its SIGNAL iterms by the macro edge (L/R/T/B) nearest each
     pin center.  The dominant side + per-edge counts + bbox are written as text:

         MACRO_PIN_SIDE name=<inst> orient=<orient> side=<L/R/T/B/NONE> dominant=<n> \
             counts=L:<n>,R:<n>,T:<n>,B:<n> bbox=(x0,y0,x1,y1)

  2. OpenROAD saves a whole-die PNG (save_image) and records the exact micron area.
  3. Python derives each macro's pin side(s) from the per-edge counts via pin_sides()
     (EVERY edge with count >= --min-pins; default 1), then maps each macro bbox
     micron->pixel and draws a thick red bar (black shadow) on EACH pin side -- a
     macro with pins on all four edges gets four bars. The numbered label is drawn
     once, on the dominant side.

Reuse:
  conda activate coder
  python macro_pin_side.py \
      --def /…/bp_fe/def/u65_placed.def \
      --lef-dir /…/bp_fe/lef \
      --out-dir /…/work_images --design bp_fe

  # or from a checkpoint ODB (tech+LEF+placement restored by read_db):
  python macro_pin_side.py --odb run/floorplan.odb --out-dir work_images --design foo
"""
import argparse
import glob
import os
import re
import subprocess
import sys

# ----------------------------------------------------------------------------- TCL
# Generated and run by OpenROAD.  Emits the structured MACRO_PIN_SIDE lines into
# <txt>, an IMAGE_AREA line, and the whole-die PNG via save_image.
_TCL = r"""
# ---- load design -----------------------------------------------------------
{load}

set block [[[ord::get_db] getChip] getBlock]
set dbu [$block getDbUnitsPerMicron]
proc um {{v}} {{ global dbu; return [expr {{double($v)/$dbu}}] }}

# ---- per-macro signal-pin side ---------------------------------------------
set fh [open "{txt}" w]
foreach inst [$block getInsts] {{
  set m [$inst getMaster]
  if {{![$m isBlock]}} {{ continue }}
  set bb [$inst getBBox]
  set x0 [$bb xMin] ; set y0 [$bb yMin]
  set x1 [$bb xMax] ; set y1 [$bb yMax]
  array set cnt {{L 0 R 0 T 0 B 0}}
  foreach it [$inst getITerms] {{
    if {{[$it getSigType] ne "SIGNAL"}} {{ continue }}
    set sb [$it getBBox]
    set cx [expr {{([$sb xMin]+[$sb xMax])/2.0}}]
    set cy [expr {{([$sb yMin]+[$sb yMax])/2.0}}]
    set dL [expr {{$cx-$x0}}] ; set dR [expr {{$x1-$cx}}]
    set dB [expr {{$cy-$y0}}] ; set dT [expr {{$y1-$cy}}]
    set best L ; set bd $dL
    if {{$dR<$bd}} {{ set best R ; set bd $dR }}
    if {{$dT<$bd}} {{ set best T ; set bd $dT }}
    if {{$dB<$bd}} {{ set best B ; set bd $dB }}
    incr cnt($best)
  }}
  set side NONE ; set dom 0
  foreach s {{L R T B}} {{ if {{$cnt($s)>$dom}} {{ set dom $cnt($s) ; set side $s }} }}
  puts $fh [format "MACRO_PIN_SIDE name=%s orient=%s side=%s dominant=%d counts=L:%d,R:%d,T:%d,B:%d bbox=(%g,%g,%g,%g)" \
      [$inst getName] [$inst getOrient] $side $dom \
      $cnt(L) $cnt(R) $cnt(T) $cnt(B) [um $x0] [um $y0] [um $x1] [um $y1]]
}}
close $fh

# ---- top-level IO pins (bterms) + union save area --------------------------
# Whole-die-WITH-pins convention (skills_Nangate45/image_snapshot): OpenROAD
# rasterizes thin pins to near-invisibility, so we record the bterm boxes to
# redraw in Python, and union them into the save area so edge pins aren't clipped.
set da [$block getDieArea]
set ux0 [um [$da xMin]] ; set uy0 [um [$da yMin]]
set ux1 [um [$da xMax]] ; set uy1 [um [$da yMax]]
set fhp [open "{iopins}" w]
foreach bt [$block getBTerms] {{
  set st [$bt getSigType]
  if {{$st ne "SIGNAL" && $st ne "CLOCK"}} {{ continue }}
  foreach bp [$bt getBPins] {{
    foreach bx [$bp getBoxes] {{
      set px0 [um [$bx xMin]] ; set py0 [um [$bx yMin]]
      set px1 [um [$bx xMax]] ; set py1 [um [$bx yMax]]
      puts $fhp [format "IO_PIN sig=%s box=(%g,%g,%g,%g)" $st $px0 $py0 $px1 $py1]
      if {{$px0<$ux0}} {{ set ux0 $px0 }} ; if {{$py0<$uy0}} {{ set uy0 $py0 }}
      if {{$px1>$ux1}} {{ set ux1 $px1 }} ; if {{$py1>$uy1}} {{ set uy1 $py1 }}
    }}
  }}
}}
close $fhp

# pad the union (die + out-of-die pins) by margin
set pad [expr {{max($ux1-$ux0,$uy1-$uy0)*{margin}}}]
set ax0 [expr {{$ux0-$pad}}] ; set ay0 [expr {{$uy0-$pad}}]
set ax1 [expr {{$ux1+$pad}}] ; set ay1 [expr {{$uy1+$pad}}]
puts "IMAGE_AREA $ax0 $ay0 $ax1 $ay1"

save_image -area [list $ax0 $ay0 $ax1 $ay1] -width {width} "{png}"
exit
"""


def _load_block(odb, def_path, lefs):
    if odb:
        return f"read_db {odb}"
    lines = [f"read_lef {l}" for l in lefs] + [f"read_def {def_path}"]
    return "\n".join(lines)


def _order_lefs(lef_dir, explicit):
    if explicit:
        return explicit
    lefs = sorted(glob.glob(os.path.join(lef_dir, "*.lef")))
    # tech LEF must be read first
    lefs.sort(key=lambda p: 0 if "tech" in os.path.basename(p).lower() else 1)
    return lefs


def run_openroad(tcl_path, openroad="/usr/bin/openroad"):
    cmd = ["xvfb-run", "-a", openroad, "-no_init", "-exit", tcl_path]
    print("[run]", " ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r


def parse_sides(txt_path):
    """Parse MACRO_PIN_SIDE lines -> list of dicts."""
    pat = re.compile(
        r"MACRO_PIN_SIDE name=(?P<name>\S+) orient=(?P<orient>\S+) "
        r"side=(?P<side>\S+) dominant=(?P<dom>\d+) "
        r"counts=L:(?P<L>\d+),R:(?P<R>\d+),T:(?P<T>\d+),B:(?P<B>\d+) "
        r"bbox=\((?P<x0>[-\d.eE+]+),(?P<y0>[-\d.eE+]+),(?P<x1>[-\d.eE+]+),(?P<y1>[-\d.eE+]+)\)"
    )
    out = []
    with open(txt_path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            d = m.groupdict()
            out.append({
                "name": d["name"], "orient": d["orient"], "side": d["side"],
                "dom": int(d["dom"]),
                "counts": {k: int(d[k]) for k in "LRTB"},
                "bbox": tuple(float(d[k]) for k in ("x0", "y0", "x1", "y1")),
            })
    return out


def parse_iopins(txt_path):
    """Parse IO_PIN lines -> list of (sig, (x0,y0,x1,y1)) micron boxes."""
    pat = re.compile(
        r"IO_PIN sig=(?P<sig>\S+) "
        r"box=\((?P<x0>[-\d.eE+]+),(?P<y0>[-\d.eE+]+),"
        r"(?P<x1>[-\d.eE+]+),(?P<y1>[-\d.eE+]+)\)")
    out = []
    if not os.path.exists(txt_path):
        return out
    with open(txt_path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                d = m.groupdict()
                out.append((d["sig"],
                            tuple(float(d[k]) for k in ("x0", "y0", "x1", "y1"))))
    return out


def _load_font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def pin_sides(counts, min_pins=1):
    """Return EVERY macro edge that carries signal pins (count >= min_pins), ordered
    by count (dominant first). A macro often has pins on more than one edge -- an
    SRAM with address/data buses on the left & right AND control pins on top & bottom
    yields all four sides [L,R,T,B]. Each pin is binned to its geometrically nearest
    edge upstream, so a count here means real pins on that edge; min_pins (default 1)
    only guards against a lone stray pin. Returns [] for a pinless macro."""
    sides = [s for s in "LRTB" if counts.get(s, 0) >= min_pins]
    sides.sort(key=lambda s: -counts[s])
    return sides


def draw_overlay(png_in, png_out, macros, area, iopins=(),
                 line_frac=0.004, min_px=3, label=True):
    """Map macro/pin bboxes micron->pixel; draw IO pins, then a red bar hugging
    EVERY pin-bearing edge of each macro (a macro with pins on >1 side gets a bar
    per side). The numbered label is drawn once, on the dominant side. Each
    macro's pin sides come from mac["sides"] (falls back to the single mac["side"]
    if not set), so it is unambiguous which macro each bar belongs to."""
    from PIL import Image, ImageDraw
    ax0, ay0, ax1, ay1 = area
    img = Image.open(png_in).convert("RGB")
    W, H = img.size
    sx = W / (ax1 - ax0)
    sy = H / (ay1 - ay0)

    def to_px(xu, yu):
        return (xu - ax0) * sx, (ay1 - yu) * sy  # flip Y: image row0 = top

    draw = ImageDraw.Draw(img)

    # IO (bterm) pins first, so macro side bars stay on top.
    pin_clr = (0, 220, 255)   # cyan
    for _sig, (x0u, y0u, x1u, y1u) in iopins:
        px0, py0 = to_px(x0u, y0u)
        px1, py1 = to_px(x1u, y1u)
        left, right = min(px0, px1), max(px0, px1)
        top, bot = min(py0, py1), max(py0, py1)
        if right - left < 2:
            right = left + 2
        if bot - top < 2:
            bot = top + 2
        draw.rectangle((left, top, right, bot), fill=pin_clr)

    lw = max(min_px, int(min(W, H) * line_frac))
    shadow = lw + max(4, lw // 2)
    # inset the bar INSIDE its own macro, hugging the pin edge, so it is never
    # confused with the neighbouring macro across the channel.
    gap = shadow // 2 + 2
    fg = (255, 0, 0)             # bright red
    sh = (0, 0, 0)              # black shadow
    font = _load_font(max(12, int(min(W, H) * 0.016)))

    drawn = 0
    for idx, mac in enumerate(macros, 1):
        mac["idx"] = idx
        sides = mac.get("sides")
        if sides is None:
            sides = [] if mac["side"] == "NONE" else [mac["side"]]
        if not sides:
            continue
        x0u, y0u, x1u, y1u = mac["bbox"]
        px0, py0 = to_px(x0u, y0u)
        px1, py1 = to_px(x1u, y1u)
        left, right = min(px0, px1), max(px0, px1)
        top, bot = min(py0, py1), max(py0, py1)
        # inset the bar to ~70% of the edge, centred, so it reads as "this macro"
        ins_v = (bot - top) * 0.15
        ins_h = (right - left) * 0.15
        # one bar per pin-bearing side (multi-side macros get several); the label
        # is drawn once, on the dominant side (mac["side"]).
        for side in sides:
            if side in ("L", "R"):
                # bar just INSIDE the left/right edge; label toward centre
                x = (left + gap) if side == "L" else (right - gap)
                seg = (x, top + ins_v, x, bot - ins_v)
                lx = x + (1 if side == "L" else -1) * (lw + 2)
                ly = (top + bot) / 2
                anchor = "lm" if side == "L" else "rm"
            else:  # T / B   (T edge is the smaller py row; inside = larger py)
                y = (top + gap) if side == "T" else (bot - gap)
                seg = (left + ins_h, y, right - ins_h, y)
                lx = (left + right) / 2
                ly = y + (1 if side == "T" else -1) * (lw + 2)
                anchor = "mt" if side == "T" else "mb"
            draw.line(seg, fill=sh, width=shadow)
            draw.line(seg, fill=fg, width=lw)
            if label and side == mac["side"]:
                _label(draw, str(idx), lx, ly, anchor, font, fg)
        drawn += 1
    img.save(png_out)
    return drawn, (W, H)


def _label(draw, text, x, y, anchor, font, fg):
    """Draw a small index label with a dark background box for legibility."""
    bb = draw.textbbox((x, y), text, font=font, anchor=anchor)
    pad = 2
    draw.rectangle((bb[0] - pad, bb[1] - pad, bb[2] + pad, bb[3] + pad),
                   fill=(0, 0, 0))
    draw.text((x, y), text, font=font, anchor=anchor, fill=(255, 255, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--def", dest="def_path", help="placed DEF")
    src.add_argument("--odb", help="checkpoint ODB (restores tech+LEF+placement)")
    ap.add_argument("--lef-dir", help="dir to glob lef/*.lef (DEF mode)")
    ap.add_argument("--lef", nargs="+", help="explicit LEFs, tech first (DEF mode)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--design", default="design", help="basename for outputs")
    ap.add_argument("--width", type=int, default=1600, help="save_image px width")
    ap.add_argument("--margin", type=float, default=0.03, help="die pad fraction")
    ap.add_argument("--no-pins", dest="pins", action="store_false",
                    help="skip the IO-pin (bterm) overlay")
    ap.add_argument("--no-label", dest="label", action="store_false",
                    help="skip the numbered macro labels next to each bar")
    ap.add_argument("--min-pins", type=int, default=1,
                    help="a macro edge is highlighted when it carries at least this "
                         "many signal pins (default 1 = every edge with any pins). "
                         "A macro with pins on all four edges gets four bars. Raise "
                         "it (e.g. 3) to suppress edges with only a stray pin or two")
    ap.add_argument("--openroad", default="/usr/bin/openroad")
    args = ap.parse_args()

    if args.def_path and not (args.lef_dir or args.lef):
        ap.error("--def needs --lef-dir or --lef")

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.design)
    txt = base + "_pin_side.txt"
    iopins_txt = base + "_iopins.txt"
    raw_png = base + "_raw.png"
    out_png = base + "_pin_side.png"
    tcl = base + "_pin_side.tcl"

    lefs = _order_lefs(args.lef_dir, args.lef)
    load = _load_block(args.odb, args.def_path, lefs)
    tcl_src = _TCL.format(load=load, txt=txt, iopins=iopins_txt, png=raw_png,
                          width=args.width, margin=args.margin)
    with open(tcl, "w") as f:
        f.write(tcl_src)

    r = run_openroad(tcl, args.openroad)
    area = None
    for line in r.stdout.splitlines():
        if line.startswith("IMAGE_AREA"):
            area = tuple(float(v) for v in line.split()[1:5])
    if r.returncode != 0 or area is None or not os.path.exists(raw_png):
        sys.stderr.write(r.stdout[-3000:] + "\n" + r.stderr[-3000:] + "\n")
        sys.exit(f"OpenROAD failed (rc={r.returncode}, area={area}, "
                 f"png_exists={os.path.exists(raw_png)})")

    macros = parse_sides(txt)
    for m in macros:
        m["sides"] = pin_sides(m["counts"], args.min_pins)
    iopins = parse_iopins(iopins_txt) if args.pins else []
    drawn, (W, H) = draw_overlay(raw_png, out_png, macros, area,
                                 iopins=iopins, label=args.label)

    combos = {}
    for m in macros:
        key = "".join(m["sides"]) or "NONE"
        combos[key] = combos.get(key, 0) + 1
    print(f"[ok] macros={len(macros)} highlighted={drawn} io_pins={len(iopins)} "
          f"side_combos={combos} image={W}x{H}")
    print(f"[ok] structured -> {txt}")
    print(f"[ok] image      -> {out_png}")
    for m in macros:
        c = m["counts"]
        sides_str = ",".join(m["sides"]) or "NONE"
        print(f"  #{m.get('idx', '-'):>2} sides={sides_str:<7} dom={m['side']:<2} "
              f"L:{c['L']} R:{c['R']} T:{c['T']} B:{c['B']}  {m['name']}")


if __name__ == "__main__":
    main()
