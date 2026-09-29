"""The LiteDRAM core's own calibration CPU (tools/litedram/calcpu.py, docs/litedram.md section 10),
seen from the host: is it there, what did it do, and the override.

At reset the core's CPU (VexRiscv, firmware tools/litedram/selfcal_fw) calibrates its channels as
opentpu.host.ddrcal.calibrate_channel would, then raises each channel's cal_ready (STATUS
CALIB0 / CALIB1), so memcal finds the channels calibrated. The CSRs, in csr.csv next to the
controllers' (all through `csr`, ddrcal's w / r on CSR names):

    selfcal_hold      1: the CPU held in reset (taken between two of its bus accesses; `held`
                      says when). Released, the firmware starts again from the top and
                      recalibrates, first dropping each channel's cal_ready.
    selfcal_config    bits 1:0 the channels to calibrate (reset 0b11), 15:8 the CK phase scan's
                      stride in fine steps (reset 1); read by the firmware when it starts.
    selfcal_status    bits 31:16 MAGIC (the core has the CPU), 4:1 log2 of the firmware
                      memory's words, bit 0 held.
    selfcal_state     written by the firmware: 2 bits per channel (STATES), bit 7 done, each
                      channel's error (ERRORS) in bits 15:8 / 23:16.
    selfcal_mbox_adr / selfcal_mbox_dat   word `adr` of the result mailbox (layout: cal.h).
    selfcal_mem_adr / selfcal_mem_dat / selfcal_mem_rdat   while the CPU is held, word `adr` of
                      its firmware memory, written / read (load(): a new firmware without a new
                      bitstream).

Stdlib only, as ddrcal (and like it, usable as a copy next to tools/litedram/ld_host.py).
"""
import time

try:
    from . import ddrcal
except ImportError:         # a copy next to ld_host.py, with ddrcal's
    import ddrcal

MAGIC = 0x5CA1
MB_MAGIC_V = 0x53435231
STATES = ("idle", "running", "ok", "failed")
DONE = 1 << 7
ERRORS = {0: "", 1: "write clock DRP timeout", 2: "write clock MMCM not locked",
          3: "BIST timeout", 4: "write clock MMCM phases not as assumed",
          5: "no CK phase common to every lane", 6: "calibration at the chosen phase failed",
          7: "CK phase shift stuck busy"}
MB_CH, MB_WORDS = (16, 80), 256


def present(csr):
    """The core has the calibration CPU (its CSRs in the map and the magic in the bitstream)."""
    return ddrcal.has_csr(csr, "selfcal_status") and csr.r("selfcal_status") >> 16 == MAGIC


def held(csr):
    return bool(csr.r("selfcal_status") & 1)


def hold(csr, timeout=1.0):
    """Stop the CPU (between two of its accesses; it stays in reset until release)."""
    csr.w("selfcal_hold", 1)
    t0 = time.time()
    while not held(csr):
        if time.time() - t0 > timeout:
            raise TimeoutError("the calibration CPU is not held")


def release(csr):
    """Restart the CPU's calibration from the top (it drops each channel's cal_ready first).
    selfcal_state is cleared first: the firmware writes it only once it starts (a millisecond
    on), and until then wait() would find the last run done."""
    csr.w("selfcal_state", 0)
    csr.w("selfcal_hold", 0)


def mem_words(csr):
    """The firmware memory's size, 32-bit words."""
    return 1 << (csr.r("selfcal_status") >> 1 & 0xF)


def load(csr, image, verify=True):
    """A new firmware for the CPU without a new bitstream: `image` is fw.py's selfcal.bin
    (bytes, run from address 0), built against this core's csr.csv and sdram_init.py
    (`fw.py target BUILD_DIR OUT_DIR`). The CPU is held and stays held (release() runs the new
    firmware); every word of its memory is written, the image then zeros, and read back. The
    bitstream's own firmware comes back only with the FPGA's next configuration. Returns the
    memory's size in words."""
    if not ddrcal.has_csr(csr, "selfcal_mem_dat"):
        raise RuntimeError("this core's CPU takes no firmware from the host (no selfcal_mem_dat)")
    n = mem_words(csr)
    if len(image) > 4 * n:
        raise ValueError(f"firmware {len(image)} bytes > the CPU's memory ({4 * n} bytes)")
    data = bytes(image) + bytes(4 * n - len(image))
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, 4 * n, 4)]
    hold(csr)
    for i, v in enumerate(words):
        csr.w("selfcal_mem_adr", i)
        csr.w("selfcal_mem_dat", v)
    if verify:
        bad = []
        for i, v in enumerate(words):
            csr.w("selfcal_mem_adr", i)
            if csr.r("selfcal_mem_rdat") != v:
                bad.append(i)
        if bad:
            raise RuntimeError(f"firmware upload: {len(bad)} of {n} words read back wrong "
                               f"(the first: word {bad[0]})")
    return n


def state(csr, value=None):
    """selfcal_state decoded: {"done": bool, "channels": {ch: (state, error text)}}."""
    v = csr.r("selfcal_state") if value is None else value
    return {"done": bool(v & DONE),
            "channels": {ch: (STATES[v >> 2 * ch & 3], ERRORS.get(v >> 8 + 8 * ch & 0xFF, "?"))
                         for ch in (0, 1)}}


def wait(csr, timeout=60.0, poll=0.05):
    """Wait for the CPU's run to finish; its state (TimeoutError if it does not)."""
    t0 = time.time()
    while True:
        v = csr.r("selfcal_state")
        if v & DONE:
            return state(csr, v)
        if time.time() - t0 > timeout:
            raise TimeoutError(f"the calibration CPU still running after {timeout} s "
                               f"(state {v:#x}, step {mailbox(csr, 3, 4)[0] & 0xFFFF})")
        time.sleep(poll)


def mailbox(csr, start=0, stop=MB_WORDS):
    out = []
    for i in range(start, stop):
        csr.w("selfcal_mbox_adr", i)
        out.append(csr.r("selfcal_mbox_dat"))
    return out


def s16(v):
    v &= 0xFFFF
    return None if v == 0xFFFF else v


def decode(mb, phy, groups=None):
    """The mailbox as {ch: result} for the channels the CPU ran, each result shaped as
    ddrcal.calibrate_channel's (dqs_steps, window_steps, window_ps, traffic_checked,
    write_latency, read, lanes, and for a WL7DDRPHY channel group1_eighths / group_runs_steps),
    plus state, error, seconds, bit_offsets (per lane, each bit's read bitslip offset), pick,
    run (first, last step from the scan's start) and start (the CK phase the scan started at)."""
    if mb[0] != MB_MAGIC_V:
        return {}
    nm = phy["modules"]
    period = round(56 * phy["vco_hz"] / (4 * phy["sys_hz"]))
    step_ps = 1e12 / phy["vco_hz"] / 56                  # ddrcal.dqs_phase's step
    out = {}
    for ch, base in enumerate(MB_CH):
        w = mb[base:base + 64]
        st = w[0] & 3
        if st == 0:
            continue
        stride, err = w[0] >> 24, w[0] >> 8 & 0xFF
        nk = w[2] >> 16
        table = {j * stride: [ddrcal.MIN_WINDOW if w[24 + 4 * m + j // 32] >> j % 32 & 1 else 0
                              for m in range(nm)] for j in range(nk)}
        wl = [(w[5] >> 4 * m & 0xF) if m < 8 else w[6] & 0xF for m in range(nm)]
        read = []
        for m in range(nm):
            r = w[7 + m]
            read.append({"taps": r & 0xFF, "bitslip": r >> 8 & 0xFF, "tap": r >> 24 & 0xFF})
        boff = [[{0: 0, 1: 2, 3: -2}.get(w[16 + (8 * m + i) // 16] >> 2 * ((8 * m + i) % 16) & 3, 0)
                 for i in range(8)] for m in range(nm)]
        steps = w[1] - (1 << 32) if w[1] >> 31 else w[1]
        start = w[23] - (1 << 32) if w[23] >> 31 else w[23]
        res = {"state": STATES[st], "error": ERRORS.get(err, f"error {err}"),
               "dqs_steps": steps, "window_steps": w[2] & 0xFFFF,
               "window_ps": round((w[2] & 0xFFFF) * step_ps), "traffic_checked": bool(w[0] >> 17 & 1),
               "write_latency": [-1 if x == 0xF else x for x in wl], "read": read,
               "lanes": ddrcal.pass_map(table, nm, period) if nk else [],
               "bit_offsets": boff, "pick": s16(w[3]), "run": (s16(w[3] >> 16), s16(w[4])),
               "start": start, "stride": stride, "seconds": round(w[22] / phy["sys_hz"], 2)}
        g = (groups or {}).get(str(ch)) if phy.get("phy") == "wl" else None
        if g:
            res["group1_eighths"] = 0
            res["group_runs_steps"] = {k: s16(w[21] >> 16 * k) for k in sorted(set(g))}
        out[ch] = res
    return out


def result(csr, config):
    """The CPU's last run, decoded (decode) with the build's PHY settings (sdram_init.py)."""
    ns = {}
    exec(open(ddrcal.load_config(config)).read(), ns)
    return decode(mailbox(csr), ns["phy"], ns["phy"].get("groups"))
