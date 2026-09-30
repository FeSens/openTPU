#!/usr/bin/env python3
"""The calibration CPU in simulation (Verilator): tools/litedram/calcpu.py's RTL (VexRiscv minimal,
its memory, mailbox, timer, SoC bus window and hold logic) running a core's RV32I firmware, in a
CPU-less LiteX SoC with the core's cal / cal1 / selfcal CSR blocks at the core's addresses, behind
the core's CSR bridge and bus (csr_pipe.py: a register stage per group of banks). Every
other CSR access of the CPU (the PHYs, controllers, BISTs, write clocks) leaves the SoC on a
Wishbone port that the testbench serves from ddrcal's simulated PHY (FakeBoard), word by word, as
the core's CSR bus would. A second port is the host's (BAR0): it reads the mailbox and holds or
releases the CPU. ddrcal itself runs on an identical FakeBoard; the two CSR write sequences must be
the same, and c0_ready / c1_ready must rise.

    python3 selfcal_sim.py CORE_DIR [--stride 8] [--seed 0] [--hold-test] [--upload-test]

CORE_DIR: a gen_core.py --selfcal output (csr.csv, sdram_init.py, selfcal_fw/selfcal.bin). Needs
LiteX (the SoC's Verilog), Verilator and a C++ compiler; ddrcal from this tree.
"""
import argparse
import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "selfcal_fw"))

HARNESS = r"""
#include <cstdint>
#include "Vselfcal_sim.h"
#include "verilated.h"

typedef uint32_t (*rd_fn)(uint32_t);
typedef void (*wr_fn)(uint32_t, uint32_t);
static VerilatedContext *ctx;
static Vselfcal_sim *top;
static uint64_t cycles, model_ops, fetches, prof[4096], soc_req, soc_cyc, dreq;
static uint32_t last_pc;
static rd_fn rd_cb;
static wr_fn wr_cb;

double sc_time_stamp() { return (double)cycles; }

static void cycle()
{
    top->sys_clk = 0;
    top->eval();
    if (top->model_ack)                             /* one cycle of ack per access */
        top->model_ack = 0;
    else if (top->model_cyc && top->model_stb) {
        uint32_t a = top->model_adr << 2;
        if (top->model_we)
            wr_cb(a, top->model_dat_w);
        else
            top->model_dat_r = rd_cb(a);
        top->model_ack = 1;
        model_ops++;
    }
    top->eval();
    top->sys_clk = 1;
    top->eval();
    cycles++;
    if (top->prof_fetch) {                          /* each cycle to the last fetched word */
        last_pc = top->prof_pc;
        fetches++;
    }
    prof[last_pc]++;
    soc_req += top->prof_soc_req;
    soc_cyc += top->prof_soc_cyc;
    dreq += top->prof_dreq;
}

extern "C" {
void sim_init(rd_fn rd, wr_fn wr)
{
    ctx = new VerilatedContext;
    top = new Vselfcal_sim{ctx};
    rd_cb = rd;
    wr_cb = wr;
    top->sys_rst = 1;
    for (int i = 0; i < 16; i++)
        cycle();
    top->sys_rst = 0;
}
void sim_run(uint64_t n) { for (uint64_t i = 0; i < n; i++) cycle(); }
uint32_t sim_host(int we, uint32_t addr, uint32_t v)   /* one BAR0 access (a Wishbone master) */
{
    top->host_adr = addr >> 2;
    top->host_we = we;
    top->host_dat_w = v;
    top->host_sel = 0xF;
    top->host_cyc = top->host_stb = 1;
    uint64_t t0 = cycles;
    do
        cycle();
    while (!top->host_ack && cycles - t0 < 1000000);
    uint32_t d = top->host_dat_r;
    cycle();                                        /* the edge that takes the ack */
    top->host_cyc = top->host_stb = 0;
    return d;
}
uint64_t sim_cycles() { return cycles; }
uint64_t sim_model_ops() { return model_ops; }
uint64_t sim_fetches() { return fetches; }
uint64_t sim_stat(int i) { return i == 0 ? soc_req : i == 1 ? soc_cyc : dreq; }
uint32_t sim_ready() { return top->c0_ready | top->c1_ready << 1; }
void sim_prof(uint64_t *out) { for (int i = 0; i < 4096; i++) { out[i] = prof[i]; prof[i] = 0; } }
}
"""


def soc_verilog(firmware, out):
    """The simulation SoC's Verilog (out/selfcal_sim.v) and its other sources (the CPU)."""
    from migen.genlib.io import CRG
    from litex.build.generic_platform import Pins, Subsignal
    from litex.build.sim import SimPlatform
    from litex.soc.integration.soc_core import SoCCore
    from litex.soc.interconnect import wishbone
    import calcpu
    from calcpu import Cal, one_file
    from csr_pipe import PipelinedCSR

    wb = [Subsignal(k, Pins(w)) for k, w in (("adr", 30), ("dat_w", 32), ("dat_r", 32), ("sel", 4),
                                             ("cyc", 1), ("stb", 1), ("ack", 1), ("we", 1))]
    io = [("sys_clk", 0, Pins(1)), ("sys_rst", 0, Pins(1)), ("model", 0, *wb), ("host", 0, *wb),
          ("c0_ready", 0, Pins(1)), ("c1_ready", 0, Pins(1)),
          ("prof", 0, Subsignal("pc", Pins(12)), Subsignal("fetch", Pins(1)),
           Subsignal("soc_req", Pins(1)), Subsignal("soc_cyc", Pins(1)), Subsignal("dreq", Pins(1)))]
    platform = SimPlatform("SIM", io)

    class SimSoC(PipelinedCSR, SoCCore):       # the core's CSR bridge and bus (csr_pipe.py)
        def __init__(self):
            self.crg = CRG(platform.request("sys_clk"), platform.request("sys_rst"))
            SoCCore.__init__(self, platform, 133.333e6, ident="", cpu_type=None,
                             integrated_rom_size=0, integrated_sram_size=0, with_uart=False,
                             with_timer=False, csr_data_width=32, with_ctrl=False)
            # the core's blocks at the core's places (csr.csv): cal 2, cal1 3, selfcal 16
            for name, loc in (("cal", 2), ("cal1", 3)):
                setattr(self, name, Cal())
                self.csr.add(name, n=loc)
            self.comb += [platform.request("c0_ready").eq(self.cal.ready.storage),
                          platform.request("c1_ready").eq(self.cal1.ready.storage)]
            self.selfcal = sc = calcpu.SelfCal(platform, firmware)
            self.csr.add("selfcal", n=calcpu.LOCATION)
            # the CPU's window: those three blocks' pages to the SoC bus, the rest to the model
            inside = wishbone.Interface(data_width=32, address_width=32, addressing="word")
            model = wishbone.Interface(data_width=32, address_width=32, addressing="word")
            page = lambda a: a[9:26]
            self.decoder = wishbone.Decoder(sc.bus, [
                (lambda a: (page(a) == 2) | (page(a) == 3) | (page(a) == calcpu.LOCATION), inside),
                (lambda a: ~((page(a) == 2) | (page(a) == 3) | (page(a) == calcpu.LOCATION)), model)])
            self.bus.add_master(name="selfcal", master=inside)
            m = platform.request("model")
            self.comb += [m.adr.eq(model.adr), m.dat_w.eq(model.dat_w), m.sel.eq(model.sel),
                          m.cyc.eq(model.cyc), m.stb.eq(model.stb), m.we.eq(model.we),
                          model.dat_r.eq(m.dat_r), model.ack.eq(m.ack)]
            host = wishbone.Interface(data_width=32, address_width=32, addressing="word")
            h = platform.request("host")
            self.comb += [host.adr.eq(h.adr), host.dat_w.eq(h.dat_w), host.sel.eq(h.sel),
                          host.cyc.eq(h.cyc), host.stb.eq(h.stb), host.we.eq(h.we),
                          h.dat_r.eq(host.dat_r), h.ack.eq(host.ack)]
            self.bus.add_master(name="host", master=host)
            prof = platform.request("prof")         # the fetches, for the harness's profile
            d = sc.cpu.dbus
            self.comb += [prof.pc.eq(sc.cpu.ibus.adr[:12]), prof.fetch.eq(sc.cpu.ibus.ack),
                          prof.soc_req.eq(d.cyc & d.stb & (d.adr[26:30] == 0xF)),
                          prof.soc_cyc.eq(sc.bus.cyc), prof.dreq.eq(d.cyc & d.stb)]

    soc = SimSoC()
    v = platform.get_verilog(soc, name="selfcal_sim")
    assert soc.csr.locs["cal"] == 2 and soc.csr.locs["cal1"] == 3 and soc.csr.locs["selfcal"] == 16
    out.mkdir(parents=True, exist_ok=True)
    for name, content in v.data_files.items():           # the memories' .init files
        (out / name).write_text(content)
    # one file, as gen_core.py writes the core: initial values inlined, the CPU appended
    (out / "selfcal_sim.v").write_text(one_file(v.main_source, out, [s for s, *_ in platform.sources]))
    return [str(out / "selfcal_sim.v")]


def build(core, out):
    """The simulation library (Verilator + the harness): out/libselfcal_sim.{so,dylib}."""
    sys.path.insert(0, str(HERE / "selfcal_fw"))
    import fw
    out = Path(out).resolve()
    data = (Path(core) / "selfcal_fw" / "selfcal.bin").read_bytes()
    srcs = soc_verilog(fw.words(data), out)
    (out / "harness.cpp").write_text(HARNESS)
    obj = out / "obj_dir"
    subprocess.run(["verilator", "--cc", "--build", "-O3", "--top-module", "selfcal_sim",
                    "-Wno-fatal", "-Wno-lint", "-Wno-style", "-Wno-MULTIDRIVEN",
                    "-CFLAGS", "-fPIC -O2", "--Mdir", str(obj), *srcs],
                   cwd=out, check=True, capture_output=True, text=True)
    lib = out / ("libselfcal_sim.dylib" if sys.platform == "darwin" else "libselfcal_sim.so")
    vroot = subprocess.run(["verilator", "--getenv", "VERILATOR_ROOT"], capture_output=True,
                           text=True, check=True).stdout.strip()
    subprocess.run(["c++", "-std=c++17", "-O2", "-shared", "-fPIC", f"-I{obj}",
                    f"-I{vroot}/include", f"-I{vroot}/include/vltstd", "-o", str(lib),
                    str(out / "harness.cpp"), str(obj / "Vselfcal_sim__ALL.a"),
                    str(obj / "libverilated.a"), "-lpthread"], check=True)
    return lib


class Sim:
    RD = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_uint32)
    WR = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32)

    def __init__(self, lib, core, csr):
        """csr: the model (ddrcal's w / r on names) behind the CPU's window."""
        from opentpu.host import ddrcal as C
        self.lib = ctypes.CDLL(str(lib))
        for f in (self.lib.sim_cycles, self.lib.sim_model_ops, self.lib.sim_fetches, self.lib.sim_stat):
            f.restype = ctypes.c_uint64
        self.lib.sim_run.argtypes = [ctypes.c_uint64]
        self.lib.sim_host.argtypes = [ctypes.c_int, ctypes.c_uint32, ctypes.c_uint32]
        self.lib.sim_host.restype = ctypes.c_uint32
        self.regs = C.csr_map(Path(core) / "csr.csv")
        words = {a + 4 * i: (n, i, k) for n, (a, k) in self.regs.items() for i in range(k)}
        pending, self.errors = {}, []

        def rd(a):
            try:
                n, i, k = words[a]
                return (csr.r(n) >> 32 * (k - 1 - i)) & 0xFFFFFFFF
            except Exception as e:           # noqa: BLE001 (raised by check())
                self.errors.append(e)
                return 0

        def wr(a, v):
            try:
                n, i, k = words[a]
                acc = (pending.pop(n, 0) << 32) | v
                if i < k - 1:
                    pending[n] = acc
                else:
                    csr.w(n, acc)
            except Exception as e:           # noqa: BLE001
                self.errors.append(e)

        self._cbs = (self.RD(rd), self.WR(wr))
        self.lib.sim_init(*self._cbs)

    def check(self):
        if self.errors:
            raise self.errors[0]

    # the host's side (BAR0): ddrcal's w / r on names, for opentpu.host.selfcal
    def w(self, name, v):
        a, k = self.regs[name]
        for i in range(k):
            self.lib.sim_host(1, a + 4 * i, (v >> 32 * (k - 1 - i)) & 0xFFFFFFFF)

    def r(self, name):
        a, k = self.regs[name]
        v = 0
        for i in range(k):
            v = (v << 32) | self.lib.sim_host(0, a + 4 * i, 0)
        return v

    def run(self, n):
        self.lib.sim_run(n)
        self.check()

    def cycles(self):
        return self.lib.sim_cycles()

    def ready(self):
        return self.lib.sim_ready()

    def profile(self, elf, nm="riscv64-elf-nm"):
        """Cycles per firmware function since the last call (the last fetched word's)."""
        buf = (ctypes.c_uint64 * 4096)()
        self.lib.sim_prof(buf)
        syms = []
        for line in subprocess.run([nm, "-n", str(elf)], capture_output=True, text=True).stdout.splitlines():
            a, k, name = line.split()[:3]
            if k.lower() == "t":
                syms.append((int(a, 16), name))
        out = {}
        for w, c in enumerate(buf):
            if c:
                name = max((s for s in syms if s[0] <= 4 * w), default=(0, "?"))[1]
                out[name] = out.get(name, 0) + c
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def run_until_done(sim, limit, chunk=2_000_000, log=print):
    """Run until selfcal_state says done (polled by the host every `chunk` cycles)."""
    t0 = time.time()
    while sim.cycles() < limit:
        sim.run(chunk)
        st = sim.r("selfcal_state")
        if st & 0x80:
            log(f"  done at {sim.cycles() / 133.333e6:.3f} s simulated ({sim.cycles()} cycles, "
                f"{time.time() - t0:.0f} s wall), state {st:#x}")
            return st
    raise TimeoutError(f"not done after {limit} cycles (state {sim.r('selfcal_state'):#x})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("core", type=Path, help="gen_core.py --selfcal output directory")
    ap.add_argument("--out", type=Path, default=None, help="build directory (default CORE/sim)")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="0: ldtest3e's channel 1 framing")
    ap.add_argument("--hold-test", action="store_true",
                    help="also: hold the CPU mid-scan, check it stops, release, calibrate again")
    ap.add_argument("--upload-test", action="store_true",
                    help="also: a second firmware loaded by the host (selfcal.load), then run")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tests"))
    from opentpu.host import ddrcal as C, selfcal as S
    import selfcal_harness as H
    out = a.out or a.core / "sim"
    t0 = time.time()
    lib = build(a.core, out)
    print(f"built {lib} in {time.time() - t0:.0f} s")

    def board():
        b = C.FakeBoard(a.core)
        if a.seed == 0:                     # ldtest3e: channel 1's misframed bits
            f = b.ch[1]
            f.RB[3], f.RB[8] = 3, 4
            f.BOFF = {(3, 3): 2, (8, 1): 2, (8, 2): 2, (8, 5): 2, (8, 7): 2}
        else:
            for k, f in enumerate(b.ch):
                g = C.FakeCsr(a.core, seed=10 * a.seed + k)
                f.RB, f.WB, f.LO, f.HI, f.WLO, f.WHI = g.RB, g.WB, g.LO, g.HI, g.WLO, g.WHI
        return b

    ok = True
    ref_log, ref = H.reference(a.core, board(), a.stride)
    # cal_ready / cal1_ready are the simulated SoC's own CSRs (c0_ready / c1_ready), not the model's
    ref_log = [x for x in ref_log if not x[0].startswith(("cal_", "cal1_"))]
    b = board()
    t = H.Trace(b)
    sim = Sim(lib, a.core, t)
    if a.stride != 1:                       # selfcal_config's stride, before the CPU reads it
        sim.w("selfcal_hold", 1)
        sim.w("selfcal_config", 3 | a.stride << 8)
        sim.w("selfcal_hold", 0)
    print(f"status {sim.r('selfcal_status'):#x} (magic {S.MAGIC:#x}), config {sim.r('selfcal_config'):#x}")
    run_until_done(sim, 60 * 133_333_000)
    prof = sim.profile(a.core / "selfcal_fw" / "selfcal.elf")
    tot = sum(prof.values())
    print("profile: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in list(prof.items())[:8])
          + f"; {sim.lib.sim_fetches() / sim.cycles():.2f} fetches per cycle, "
          f"{sim.cycles() / max(sim.lib.sim_model_ops(), 1):.0f} cycles per PHY CSR access; "
          f"cycles with a SoC request {sim.lib.sim_stat(0) / sim.cycles():.0%}, on the SoC bus "
          f"{sim.lib.sim_stat(1) / sim.cycles():.0%}, any data request {sim.lib.sim_stat(2) / sim.cycles():.0%}")
    same = t.log == ref_log
    ok &= same
    print(f"CSR writes: firmware on the CPU {len(t.log)}, ddrcal {len(ref_log)}: "
          + ("identical" if same else "DIFFERENT: " + H.first_diff(ref_log, t.log)))
    ready = sim.ready()
    print(f"c0_ready / c1_ready: {ready & 1} / {ready >> 1 & 1}")
    ok &= ready == 3
    import fw
    p = fw.config(a.core)[1].phy
    dec = S.decode(S.mailbox(sim), p, p.get("groups"))
    for ch, r in ref.items():
        d = dec.get(ch, {})
        eq = not isinstance(r, Exception) and {k: d.get(k) for k in r} == r
        ok &= eq
        print(f"channel {ch}: {d.get('state')}, CK {d.get('dqs_steps')}, window {d.get('window_steps')} "
              f"steps, wl {d.get('write_latency')}, {d.get('seconds')} s; result "
              + ("equal to ddrcal's" if eq else f"DIFFERS: ddrcal {r}"))
    first = list(t.log)
    if a.upload_test:
        ok &= upload_test(sim, a.core, out, t, first, dec, S)
    if a.hold_test:
        ok &= hold_test(sim, t, S)
    print("selfcal_sim: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def upload_test(sim, core, out, t, first, dec, S, fw_id=0xB0B0CAFE):
    """A new firmware without a new bitstream: the same source built with another FW_ID. A write
    to the memory while the CPU runs is dropped; selfcal.load writes the image into the held
    CPU's memory and reads it back; released, the CPU runs it (the mailbox's firmware word) and
    calibrates again: the same CSR writes as its first run, the same result."""
    import fw
    words = fw.words((Path(core) / "selfcal_fw" / "selfcal.bin").read_bytes())
    sim.w("selfcal_mem_adr", 5)
    sim.w("selfcal_mem_dat", 0xDEADBEEF)                  # not held: dropped
    S.hold(sim)
    sim.w("selfcal_mem_adr", 5)
    kept = sim.r("selfcal_mem_rdat") == words[5]
    data = fw.target(core, Path(out) / "fw_upload", fw_id=fw_id)
    t0 = time.time()
    n = S.load(sim, data)
    print(f"upload: a write while the CPU ran {'dropped' if kept else 'NOT dropped'}; "
          f"{len(data)} bytes into {4 * n}, read back, in {time.time() - t0:.1f} s wall "
          f"({sim.cycles()} cycles)")
    n0 = len(t.log)
    S.release(sim)
    run_until_done(sim, sim.cycles() + 60 * 133_333_000)
    again = t.log[n0:]
    mb = S.mailbox(sim)
    p = fw.config(core)[1].phy
    dec2 = S.decode(mb, p, p.get("groups"))
    same_log, same_res = again == first, dec2.keys() == dec.keys() and all(
        {k: v for k, v in dec2[ch].items() if k != "seconds"} ==
        {k: v for k, v in dec[ch].items() if k != "seconds"} for ch in dec)
    print(f"the new firmware: id {mb[1]:#x} (want {fw_id:#x}), CSR writes "
          + ("identical to its first run's" if same_log else "DIFFERENT: " + __import__(
              "selfcal_harness").first_diff(first, again))
          + f", result {'the same' if same_res else 'DIFFERENT'}, c0/c1_ready {sim.ready():#x}")
    return kept and mb[1] == fw_id and same_log and same_res and sim.ready() == 3


def hold_test(sim, t, S):
    """Release the CPU: it starts over (channel 0's cal_ready drops, the scan writes again). Hold
    it mid-scan: it stops at once (no more writes) and stays held. Release it: it finishes."""
    sim.w("selfcal_hold", 1)
    ok = S.held(sim)
    sim.w("selfcal_hold", 0)
    n0 = len(t.log)
    sim.run(3_000_000)
    n = len(t.log)
    ok &= not sim.ready() & 1 and not sim.r("selfcal_state") & 0x80 and n > n0
    sim.w("selfcal_hold", 1)
    held = S.held(sim)
    n1 = len(t.log)
    sim.run(2_000_000)
    ok &= held and len(t.log) == n1
    print(f"released: {n - n0} writes in 3M cycles, c0_ready {sim.ready() & 1}; held mid-scan: "
          f"{held}, {len(t.log) - n1} writes in the 2M cycles after")
    sim.w("selfcal_hold", 0)
    run_until_done(sim, sim.cycles() + 60 * 133_333_000)
    ok &= sim.ready() == 3
    print(f"after release: c0/c1_ready {sim.ready():#x}")
    return ok


if __name__ == "__main__":
    sys.exit(main())
