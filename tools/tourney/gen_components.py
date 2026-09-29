"""Regenerates tools/tourney/components/*.yaml (the component definitions). The YAML has since
been edited by hand (test_perf.py and the DSTEP RTL test in the gates, notes on known timing
paths; otpu_impl and otpu_full are hand-written): edit the YAML, and port those edits
here before regenerating (docs/tourney.md, "Config drift")."""
from pathlib import Path

import yaml

OUT = Path(__file__).resolve().parent / "components"
BASE = ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv", "rtl/top/otpu_pkg.sv"]
FAST = ["tests/test_rtl.py::test_mlp_rtl[1]",
        "tests/test_rtl.py::test_attention_decode_rtl[1-8-2-100]",
        "tests/test_rtl.py::test_design_size_block_128_rtl",
        "tests/test_rtl.py::test_fuzz_single_slice[0]",
        "tests/test_rtl.py::test_fuzz_single_slice[1]",
        "tests/test_rtl.py::test_scoreboard_stress_single_slice[0]",
        "tests/test_rtl.py::test_scoreboard_stress_single_slice[1]",
        "tests/test_rtl.py::test_scoreboard_stress_single_slice[2]"]
BOARD = ["tests/test_rtl.py::test_design_size_block_128_rtl",
         "tests/test_rtl.py::test_board_memory_path_stress[1-30]",
         "tests/test_rtl.py::test_board_memory_path_stress[4-80]",
         "tests/test_rtl.py::test_board_memory_path_fuzz[0-40]"]

C = {
    "otpu_seq": dict(
        files=["rtl/seq/otpu_seq.sv"], top="otpu_seq",
        params=dict(IMEM_WORDS=32768, SID=0, S=1, D=128, WIN=16),
        desc="Sequencer: instruction fetch (IMEM block RAM), decode, loop stack, dispatch window "
             "with the scoreboard (footprint conflicts), per-unit oldest-ready start, MXU two-phase "
             "release.",
        extra=["tests/test_rtl.py::test_scoreboard_stress_single_slice[3]",
               "tests/test_rtl.py::test_scoreboard_stress_single_slice[4]",
               "tests/test_rtl.py::test_fuzz_two_slices[0]",
               "tests/test_rtl.py::test_scoreboard_stress_two_slices[0]"]),
    "otpu_mxu": dict(
        files=["rtl/mxu/otpu_mxu.sv"], top="otpu_mxu",
        params=dict(D=128, MCOLS=4, DEPTH=1024, LANES=8, IMPL=2, SID=0),
        desc="MXU: streams int8 weight chunks (D bytes/cycle) through a prefetch FIFO against up to "
             "MCOLS stationary int8 rows in ACT RAM; products in the systolic array (IMPL 2: a "
             "DSP48E1 per position, weights hopping column to column), i2f, scale multiplies, "
             "isum_4 partial loop, result FIFO, drain with ACC/ASCALE read-modify-write, RMAX.",
        extra=["tests/test_rtl.py::test_attention_layer_rtl[1]",
               "tests/test_rtl.py::test_mlp_rtl[2]"]),
    "otpu_quant": dict(
        files=["rtl/vpu/otpu_quant.sv"], top="otpu_quant",
        params=dict(D=128, LANES=8, SID=0),
        desc="Quantizer: QACT (fp32 TMEM rows -> int8 ACT RAM rows with per-block or per-row "
             "scales, streaming or two-pass) and QST (int8 bytes + scales to DRAM); prescale, amax "
             "trees, reciprocal (Newton), q8 rounding.",
        extra=["tests/test_rtl.py::test_attention_layer_rtl[1]",
               "tests/test_rtl.py::test_scoreboard_stress_single_slice[3]"]),
    "otpu_vpu": dict(
        files=["rtl/vpu/otpu_vtree.sv", "rtl/vpu/otpu_se_comp.sv", "rtl/vpu/otpu_se_tail.sv",
               "rtl/vpu/otpu_vpu.sv"],
        top="otpu_vpu",
        params=dict(LANES=8, SID=0),
        desc="VPU: elementwise fp32 ops on LANES lanes (split lanes: CL composite lanes with 10 "
             "multiply-add slots for EXP2/RECIP/RSQRT/LOG2, simple lanes with 1 slot; slot 0 is "
             "a*b + c*e for OUTER, with per-lane column buffers), RMAX, RSUM/RSSQ/RDOT (isum_64 "
             "partial loop and folding trees; RDOT buffers its row sums).",
        extra=["tests/test_rtl.py::test_lane_count_does_not_change_results[4]",
               "tests/test_rtl.py::test_lane_count_does_not_change_results[16]",
               "tests/test_rtl.py::test_attention_layer_rtl[1]"]),
    "otpu_tmem": dict(
        files=["rtl/mem/otpu_tmem.sv"], top="otpu_tmem",
        params=dict(WORDS=65536, LANES=8, NRP=8, NWP=4, WPB=1, GEN_R=1, GEN_W=2, SID=0),
        desc="TMEM: fp32 scratchpad, LANES banks, one copy per read port (simple dual-port block "
             "RAM), writes broadcast (WPB per bank), per-lane read data held until the lane reads "
             "again; contiguous ports through barrel rotators, the MXU ports (GEN_R/GEN_W) through "
             "a lane crossbar.",
        extra=["tests/test_rtl.py::test_lane_count_does_not_change_results[4]",
               "tests/test_rtl.py::test_lane_count_does_not_change_results[16]",
               "tests/test_rtl.py::test_scoreboard_stress_two_slices[0]"]),
    "otpu_dma": dict(
        files=["rtl/dma/otpu_dma.sv"],
        top="otpu_dma",
        params=dict(D=128, LANES=8),
        desc="DMA: LD/ST between DRAM port B (chunk-aligned requests, backpressure, variable "
             "latency) and TMEM: each chunk requested once (LD chunk buffer, ST gather), one W-word segment per cycle on TMEM; ST completes on write acknowledge.",
        extra=["tests/test_rtl.py::test_fuzz_single_slice[2]",
               "tests/test_rtl.py::test_fuzz_single_slice[3]"],
        board_extra=["tests/test_rtl.py::test_board_memory_path_stress[2-60]"]),
    "otpu_actram": dict(
        files=["rtl/mem/otpu_actram.sv"], top="otpu_actram",
        params=dict(D=128, MCOLS=4, BLOCKS=128, LANES=8),
        desc="ACT RAM: MCOLS rows x BLOCKS blocks of D int8 plus block scales; written by the "
             "quantizer, read one block per cycle by the MXU.",
        extra=["tests/test_rtl.py::test_attention_layer_rtl[1]"]),
    "otpu_coll": dict(
        files=["rtl/top/otpu_coll.sv"], top="otpu_coll",
        params=dict(S=1, LANES=8),
        desc="Collective unit: all-gather / gather across slices through TMEM (S=1 on the board, "
             "but the RTL is generic in S and the multi-slice tests gate it).",
        extra=["tests/test_rtl.py::test_mlp_rtl[2]",
               "tests/test_rtl.py::test_fuzz_two_slices[0]",
               "tests/test_rtl.py::test_scoreboard_stress_two_slices[0]",
               "tests/test_rtl.py::test_attention_decode_rtl[2-8-2-100]"]),
    "512-bit AXI4 channels, 64-byte interleave, per-channel queues and response FIFOs, "
             "A-beat reuse, write acknowledges (wr_idle).",
        extra=[],
        board_extra=["tests/test_rtl.py::test_board_memory_path_stress[0-0]",
                     "tests/test_rtl.py::test_board_memory_path_stress[2-60]",
                     "tests/test_rtl.py::test_board_memory_path_stress[3-30]",
                     "tests/test_rtl.py::test_board_memory_path_stress[5-50]",
                     "tests/test_rtl.py::test_board_memory_path_fuzz[1-70]",
                     "tests/test_board.py"]),
    "otpu_native_dram": dict(
        files=["rtl/mem/otpu_native_dram.sv"], top="otpu_native_dram",
        params=dict(D=128),
        desc="Native DRAM adapter (the board's): slice ports A (words), B (chunks) and "
             "the QST write port onto the two channels' native ports, one command per 64-byte "
             "beat, 64-byte interleave (hashed), per-channel queues, in-order read merge, A-beat "
             "reuse, the SW queue's read-fill with its hazard buckets, write acknowledges "
             "(n_wdone, wr_idle).",
        extra=[],
        board_extra=["tests/test_rtl.py::test_native_memory_path[hazard-1-50-20]",
                     "tests/test_rtl.py::test_native_memory_path[hazard-2-80-120]",
                     "tests/test_rtl.py::test_native_memory_path[random-3-40-20]",
                     "tests/test_rtl.py::test_native_memory_path[random-4-70-400]",
                     "tests/test_rtl.py::test_board_memory_path_stress[2-60]",
                     "tests/test_rtl.py::test_board_memory_path_fuzz[1-70]",
                     "tests/test_board.py"]),
}

FP = {
    "name": "otpu_fp",
    "description": "Shared fp32 operators: the staged functions of otpu_fp (unpack, multiply, "
                   "align, normalize with tree LZC, round) and the pipelined otpu_fmul (2 stages) / "
                   "otpu_fadd (4 stages) / otpu_fmadd built from them. Instantiated ~140 times "
                   "across the VPU, MXU and quantizer, so every LUT here counts many times.",
    "allowed": ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv"],
    "synth": {"parts": [
        {"top": "otpu_fadd", "weight": 64, "params": {"LAT": 4},
         "sources": ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv"]},
        {"top": "otpu_fmul", "weight": 78, "params": {"LAT": 2},
         "sources": ["rtl/vpu/otpu_fp.sv", "rtl/vpu/otpu_fpipe.sv"]}],
        "note": "weights = approximate instance counts at the board configuration (VPU ~42 fadd / "
                "34 fmul, MXU ~16 / 12, quantizer ~6 / 32); otpu_fp functions used inline elsewhere "
                "(fp_gt, i2f, q8, ...) are not in the area figure."},
    "target_mhz": 110,
    "tests": {"fast": ["tests/test_fp.py", "tests/test_rtl.py"], "board": ["tests/test_rtl.py"]},
    "perf": True,
}


# The cross-unit component of the fmax tournament (--objective fmax): the wiring between the
# units, the slice and board tops and the timing constraints. It has no out-of-context part (its
# paths only exist in the whole design), so a candidate goes straight to the full build. The pin
# constraint files are not in it (pins are fixed by the card).
XUNIT = {
    "name": "otpu_xunit",
    "description": "Cross-unit paths: otpu_slice (the units and their wiring: scoreboard grants, "
                   "TMEM port muxes, DRAM ports), otpu_top, otpu_pkg, the board wrappers "
                   "(otpu_board: PCIe bridge, DRAM adapter, control registers, trace; "
                   "otpu_native_sys: the channels' bridges (otpu_mem_ch, otpu_afifo) and XDMA's "
                   "split (otpu_axi_split2); otpu_fpga_top_ld: clocks and resets) and the core timing "
                   "constraints (otpu_top_ld.xdc: pblocks, "
                   "placement, no timing exceptions or clock changes). Changes here target paths that cross "
                   "unit boundaries: register slices on inter-unit buses, fanout of broadcast "
                   "signals, placement.",
    "allowed": ["rtl/top/otpu_slice.sv", "rtl/top/otpu_top.sv", "rtl/top/otpu_pkg.sv",
                "rtl/boards/ypcb-00338/otpu_board.sv", "rtl/boards/ypcb-00338/otpu_ctrl.sv",
                "rtl/boards/ypcb-00338/otpu_native_sys.sv", "rtl/boards/ypcb-00338/otpu_mem_ch.sv",
                "rtl/boards/ypcb-00338/otpu_afifo.sv", "rtl/boards/ypcb-00338/otpu_axi_split2.sv",
                "rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv", "rtl/boards/ypcb-00338/otpu_trace.sv",
                "boards/ypcb-00338/constraints/otpu_top_ld.xdc"],
    "synth": {"parts": []},
    "objectives": ["fmax"],
    "target_mhz": 133.33,
    "tests": {"fast": ["tests/test_perf.py"] + FAST + ["tests/test_rtl.py::test_fuzz_two_slices[0]",
                              "tests/test_rtl.py::test_scoreboard_stress_two_slices[0]",
                              "tests/test_rtl.py::test_attention_layer_rtl[1]"],
              "board": BOARD + ["tests/test_rtl.py::test_board_memory_path_stress[2-60]",
                                "tests/test_rtl.py::test_board_memory_path_fuzz[1-70]",
                                "tests/test_board.py"]},
    "perf": True,
}


def main():
    OUT.mkdir(exist_ok=True)
    for name, c in C.items():
        y = {
            "name": name,
            "description": c["desc"],
            "allowed": c["files"],
            "synth": {"parts": [{"top": c["top"], "weight": 1, "params": c["params"],
                                 "sources": BASE + c["files"]}]},
            "target_mhz": 110,
            "tests": {"fast": FAST + c["extra"], "board": BOARD + c.get("board_extra", [])},
            "perf": True,
        }
        (OUT / f"{name}.yaml").write_text(yaml.safe_dump(y, sort_keys=False, width=100))
    (OUT / "otpu_fp.yaml").write_text(yaml.safe_dump(FP, sort_keys=False, width=100))
    (OUT / "otpu_xunit.yaml").write_text(yaml.safe_dump(XUNIT, sort_keys=False, width=100))
    print("wrote", len(C) + 2, "components to", OUT)


if __name__ == "__main__":
    main()
