"""Model images on a DRAM wider than 4 GiB (Config.DAW 33..36, docs/wide_dram_hbm.md).

A DRAM byte address is {REG, (R + imm) mod 2^32}: REG is an instruction's 4 GiB region, every
DRAM operand of the instruction lies in it, and no instruction's DRAM range crosses a region
boundary (isa.py "DRAM address width"; compiler.Builder.region checks every address). So:
- region 0 holds what a register or a table word names absolutely, or what the programs address
  with a run-time value of unknown range: the I/O area, the lookup tables (the embedding rows
  by token id), the generate area, the Engine's snapshot and batch areas, and the program area
  (PROG_ADDR and HALT CHAIN's register are 32 bits);
- a layer block (its norms, every matrix with its scale rows, its KV caches with their scale
  rows, its DeltaNet state and convolution window) is one piece: a matrix and its scales, a
  cache and its scales share their block's region;
- a run of layer blocks (one hardware loop of the programs, stepping one address register by
  the unit's stride) lies whole in one region: the run planner (Regions.place) splits a run
  that does not fit what is left of a region at a unit, so each piece is a loop of its own.

An image that fits below 4 GiB with the program area is laid out as at DAW 32, bit for bit
(regions_for returns None): nothing changes for it. A bigger one keeps the low areas in region
0 and puts its layer runs in regions 1, 2, ... (each run whole where it fits); its `prog_at` is
the program area's address in region 0 (past the low areas), and its `nbytes` the extent, up to
the last region's end. The images an Engine hands its backend are then ImageArrays: `prog_at`,
and the byte ranges that hold anything (`spans`: the board writes those, not the gaps).

The relocation switches of the tests (Image(..., high_base=, low_limit=)): every layer block
from `high_base` on (any 4096-aligned address of region 1 or above: each run fills its region
and is split at the next boundary), the blocks below `low_limit` in region 0.
"""
from __future__ import annotations

REGION = 1 << 32                # a DRAM region: 4 GiB
PAGE = 4096                     # layer runs and the program area are page-aligned


def wide(cfg) -> bool:
    """A configuration whose DRAM reaches past 4 GiB with REG to address it (DAW > 32)."""
    return getattr(cfg, "DAW", 32) > 32 and cfg.DRAM_BYTES > REGION


def dram_limit(cfg) -> int:
    """The bytes a layout may use: DRAM_BYTES (at DAW > 32 at most 2^DAW; at DAW 32 the
    layout-only probes' DRAM_BYTES of 2^33 or 2^40 stay what they say)."""
    daw = getattr(cfg, "DAW", 32)
    return min(cfg.DRAM_BYTES, 1 << daw) if daw > 32 else cfg.DRAM_BYTES


def dram_low(cfg) -> int:
    """Region 0's bytes, which hold what the programs address with 32-bit device or run-time
    values (the I/O area, the lookup tables, the generate, snapshot, batch and program areas):
    all of DRAM at DAW 32, at most 4 GiB above it."""
    return min(cfg.DRAM_BYTES, REGION) if getattr(cfg, "DAW", 32) > 32 else cfg.DRAM_BYTES


def tail_at(image) -> int:
    """Where the areas after an image go (the Engine's snapshots and batch area, then the
    board's program area): past the image, page-aligned, or for an image laid out over regions
    (Image.prog_at) past its region-0 areas."""
    at = getattr(image, "prog_at", None)
    return -(-image.nbytes // PAGE) * PAGE if at is None else at


def prog_area(cfg) -> int:
    from .. import waits as W
    return W.area(cfg.IMEM_WORDS, getattr(cfg, "SEQ_WC", False))


class Regions:
    """The layer runs of an image over the regions of a wide DRAM. Region 0 from `at` (the
    first layer block's address) up to `limit`; the high regions from `high` (region 1's start
    by default) up to `end` (the DRAM's). `whole` (the default layout): each run goes whole into
    the first high region with room for it, else below `limit` where it fits; only a run that
    fits no region whole (it alone exceeds a region) is split at units, filling the region with
    the most room first. Not `whole` (the relocation switches): region 0 is filled up to `limit`
    first (a run that crosses it split at a unit), then the high part from `high` in address
    order, each run up to the next region boundary and the rest of it after it.
    `alloc` places a static region (an LM head) in the high part the same way."""

    def __init__(self, at: int, limit: int, end: int, high: int = REGION, whole: bool = True):
        if high < REGION or high % PAGE:
            raise ValueError(f"high {high:#x}: a page of region 1 or above")
        self.low, self.limit, self.end, self.whole = at, max(at, limit), end, whole
        self.high = high
        # each high region's next free address (region r: [r * 4 GiB, (r + 1) * 4 GiB))
        self.next = {r: max(r * REGION, high) for r in range(high // REGION,
                                                              -(-end // REGION))}
        self.spilled = False

    def _room(self, r: int) -> int:
        return max(0, min((r + 1) * REGION, self.end) - self.next[r])

    def place(self, runs, size) -> tuple[list, list]:
        """The runs [(first layer, unit of keys, repeats)] placed (size(key): a block's bytes,
        page multiples): (the runs after the splits, each one's base address)."""
        out, bases = [], []
        for first, unit, reps in runs:
            us, n = sum(size(k) for k in unit), len(unit)
            while reps:
                lo_room = max(0, self.limit - self.low)
                if not self.whole:
                    k = 0 if self.spilled else min(reps, lo_room // us)
                    if k:
                        r = None
                    else:
                        self.spilled = True
                        r = next((r for r in sorted(self.next) if self._room(r) >= us), None)
                        if r is None:
                            raise MemoryError(self._full(reps * us))
                        k = min(reps, self._room(r) // us)
                else:
                    fit = [r for r in sorted(self.next) if self._room(r) >= reps * us]
                    if fit:
                        r, k = fit[0], reps
                    elif reps * us <= lo_room:
                        r, k = None, reps
                    else:                               # (a run that alone exceeds a region)
                        rooms = sorted(((self._room(r), -r) for r in self.next), reverse=True)
                        if rooms and rooms[0][0] >= us:
                            r = -rooms[0][1]
                            k = self._room(r) // us
                        elif lo_room >= us:
                            r, k = None, lo_room // us
                        else:
                            raise MemoryError(self._full(reps * us))
                out.append((first, unit, k))
                if r is None:
                    bases.append(self.low)
                    self.low += k * us
                else:
                    bases.append(self.next[r])
                    self.next[r] += k * us
                first += k * n
                reps -= k
        return out, bases

    def _full(self, need: int) -> str:
        free = {r: self._room(r) >> 20 for r in self.next}
        return (f"model image: {need / 2**20:.0f} MiB of layer blocks fit no DRAM region (MiB "
                f"free: region 0 {max(0, self.limit - self.low) >> 20}, the others {free})")

    def alloc(self, nbytes: int) -> int:
        """A static region in the high part (page-aligned), in one region."""
        need = -(-nbytes // PAGE) * PAGE
        r = next((r for r in sorted(self.next) if self._room(r) >= need), None)
        if r is None:
            raise MemoryError(self._full(need))
        a = self.next[r]
        self.next[r] += need
        return a

    @property
    def top(self) -> int:
        """The end of what the high regions hold (their extent), or 0."""
        used = [a for r, a in self.next.items() if a > max(r * REGION, self.high)]
        return max(used, default=0)

    def spans(self, low_end: int) -> list:
        """The byte ranges the image uses: [0, low_end), then each high region's part."""
        out = [(0, low_end)]
        for r, a in sorted(self.next.items()):
            s = max(r * REGION, self.high)
            if a > s:
                out.append((s, a))
        return out


def regions_for(cfg, at: int, runs, size, tail: int, high_base: int | None = None,
                low_limit: int | None = None, no: str | None = None) -> Regions | None:
    """How an image whose layer runs `runs` start at `at` (size(key): a block's bytes; `tail`:
    the bytes of the region-0 areas that follow them: the LM head, the lookup tables) uses the
    DRAM: None where it is one contiguous image from 0, as at DAW 32 (every image of a
    configuration that is not wide, and a wide one that fits below the program area in region
    0), else the Regions that place the layer runs (each run whole in one region, a high one
    where it fits). high_base / low_limit: the relocation switches (the module's doc). `no`:
    what the image has that is not placed over regions (an MTP layer, a MoE's expert slots,
    ...): it stays contiguous in region 0 (MemoryError where it does not fit, as the Engine's
    fallbacks take it), and the switches are refused."""
    wd = wide(cfg)
    if (high_base is not None or low_limit is not None) and not wd:
        raise ValueError("high_base / low_limit: layer blocks above 4 GiB need a configuration "
                         "with DAW > 32 and DRAM_BYTES over 4 GiB")
    if not wd:
        return None
    top = (REGION - prog_area(cfg)) // PAGE * PAGE      # region 0's areas end here at the latest
    end = at + sum(reps * sum(size(k) for k in unit) for _, unit, reps in runs)
    if high_base is None and low_limit is None:
        if end + tail <= top:
            return None
        if no:
            raise MemoryError(f"model image needs {(end + tail) / 2**20:.0f} MiB in region 0 "
                              f"with its program area: {no} is not placed over the DRAM's "
                              f"regions")
        return Regions(at, top - tail, dram_limit(cfg))
    if no:
        raise ValueError(f"high_base / low_limit: {no} is not placed over the DRAM's regions")
    hb = REGION if high_base is None else high_base
    if hb < REGION or hb % PAGE or hb >= dram_limit(cfg):
        raise ValueError(f"high_base {hb:#x}: a page of region 1 or above, below DRAM_BYTES "
                         f"{cfg.DRAM_BYTES:#x}")
    return Regions(at, at if low_limit is None else min(low_limit, top - tail),
                   dram_limit(cfg), hb, whole=False)


def place_runs(runs, size, at: int, rg: Regions | None = None) -> tuple[list, list, int]:
    """The layer runs' base addresses: each run's blocks after the one before from `at` (rg
    None: one image, as at DAW 32), or as the Regions place them. (the runs after the splits,
    each one's base, the end of the region-0 blocks)."""
    if rg is not None:
        runs, bases = rg.place(runs, size)
        return runs, bases, rg.low
    bases = []
    for first, unit, reps in runs:
        bases.append(at)
        at += reps * sum(size(k) for k in unit)
    return runs, bases, at


def extent(cfg, rg: Regions | None, low_end: int) -> tuple[int | None, int, list | None]:
    """(prog_at, nbytes, spans) of an image whose region-0 areas end at `low_end` (and, with
    `rg`, its high part): the program area's address (None: after the image, as at DAW 32), the
    image's extent and the ranges it uses (None: all of it). MemoryError when the program area
    does not fit region 0, or the extent the DRAM."""
    prog_at, nbytes, spans = None, low_end, None
    if rg is not None:
        prog_at = -(-low_end // PAGE) * PAGE
        if prog_at + prog_area(cfg) > REGION:
            raise MemoryError(f"the I/O area, the LM head and the lookup tables take "
                              f"{low_end / 2**20:.0f} MiB: no room for the program area in "
                              f"region 0")
        nbytes = max(low_end, rg.top)
        spans = rg.spans(low_end)
    elif wide(cfg) and -(-low_end // PAGE) * PAGE + prog_area(cfg) > REGION:
        raise MemoryError(f"model image needs {low_end / 2**20:.0f} MiB in region 0 with its "
                          f"program area: what is in it is not placed over the DRAM's regions")
    if nbytes > dram_limit(cfg):
        raise MemoryError(f"model image needs {nbytes / 2**20:.0f} MiB per slice, "
                          f"DRAM_BYTES is {cfg.DRAM_BYTES / 2**20:.0f} MiB")
    return prog_at, nbytes, spans


def check_low(cfg, nbytes: int, what: str = "model image") -> None:
    """An image laid out contiguously from 0 (one not placed over regions): at DAW > 32 it and
    the program area after it must fit region 0 (its programs address it with REG 0)."""
    if wide(cfg) and -(-nbytes // PAGE) * PAGE + prog_area(cfg) > REGION:
        raise MemoryError(f"{what} needs {nbytes / 2**20:.0f} MiB in region 0 with the program "
                          f"area: it is not placed over the DRAM's regions")


class ImageArrays(list):
    """The DRAM images an Engine hands its backend (one array per slice) of an image laid out
    over regions: `prog_at`, where the board backend puts the program area (in region 0, past
    the image's region-0 areas and the Engine's areas after them), and `spans`, the byte ranges
    that hold anything (the board writes those only). A plain list otherwise."""

    def __init__(self, images, prog_at: int, spans=None):
        super().__init__(images)
        self.prog_at = prog_at
        self.spans = spans
