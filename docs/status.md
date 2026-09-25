# Status: 2026-09-24, before the first Vivado build

## Where things stand

- **Model:** Qwen3-0.6B runs with real weights.
  - On the ISA simulator it follows Hugging Face closely but not exactly: it is int8 (W8A8), and
    greedy decoding picks a different token when the top two are nearly tied (README, Accuracy).
  - On the RTL at the board configuration it is bit-exact with the ISA simulator.
- **Bring-up rehearsal:** from a clean clone, `otpu-selftest --sim --qwen models/Qwen3-0.6B` passed all 9 stages on the board model.
  - The model is `sim/verilator/tb_board.sv`: AXI-Lite registers, program loader, AXI adapter, two DDR3 channels with random stalls.
  - The run went through the same host driver as the card (`opentpu/host/board.py`).
  - Greedy decoding matched the ISA simulator token for token, at 6.38 Mcycles/token (a projected 15.7 tok/s at an assumed 100 MHz, without host time).
  - The rehearsal ran before the night's timing changes. They add a few cycles per instruction, and the per-token cycle count the tournaments track stayed within 0.2%.
  - What the board model does not cover: DDR3 calibration is hardwired to succeed, the MIGs are
    replaced by an AXI memory model, and PCIe, clocks and resets are not simulated. A review
    found that `rst_core` in bd.tcl took the MMCM `locked` signal on an active-high reset input,
    which would have held the core in reset on hardware. Fixed (C_EXT_RESET_HIGH 0), and
    `make lint` now requires every proc_sys_reset to state its polarity.
- **Accelerator and control logic, yosys estimate** (excludes the XDMA and MIG IP, roughly 45K LUT more):

  | | Start of night | Now |
  |---|---|---|
  | LUT | 131,915 | 82,224 (27.5% of the xc7k480t) |
  | FF | 55,890 | 35,944 |
  | DSP | 416 | 267 |
  | BRAM36 | 603 | 603 |
  | Logic delay | 14.94 ns | 5.56 ns |
  | Est. fmax | 41 MHz | 106 MHz |

  The estimate is `1000 / (1.6 * logic + 0.5)`, not signoff; Vivado gives the real number.

  Since then (v0.3) the board has register map v2: free-running counters, die temperature and
  a 16K-record hardware trace buffer (docs/observability.md). With them it is 85,594 LUT,
  42,393 FF, 635 BRAM36 (the trace uses 32) and still an estimated 106 MHz (logic 5.61 ns).

## What changed overnight

- **Cross-unit timing**, each fix verified in all RTL modes (default, cascade MXU, board with AXI and boot, heavy AXI stalls) plus the board, perf and Qwen3 tests:
  - TMEM arbiter: write-mask arbitration.
  - Grants reach only the BRAM enables. Read addresses are merged from requests, and write lanes are merged per port with the grant applied at the last level.
  - Registered TMEM data at the inputs of the VPU, the quantizer prescale and the MXU accumulate path.
  - Registered VPU writes.
  - Registered request addresses and masks in the DMA (LD mask, address, write), the collective unit (plus a write stage) and the MXU drain, plus the RMAX base and an in-flight counter.
  - The arbiter computes pairwise conflicts in parallel instead of a serial chain.
  - MXU scale FIFO read.
  - One-cycle loops removed: IMEM fetch → decode → PC in the sequencer; the fp multiply's flush-to-zero in front of the DSPs.
- **Component tournaments** (`tools/tourney`, Opus agents, yosys evaluation):
  - Units tuned: fp operators, quantizer, sequencer, MXU, VPU, AXI adapter.
  - 32 accepted winners. The per-unit estimated fmax of every one now clears 110 MHz; the MXU and quantizer are 40–60% smaller.
  - Harness fixes made during the night:
    - Pareto accept (an area win may stay below target as long as it does not lose fmax).
    - Full patches are kept.
    - Champions sync with main every round.
    - A gate against private copies of fp internals.
    - The kernel perf tests are part of every gate. This came after a VPU winner cut batched-attention speed.
- **Build switches:**
  - `make bit CORE_MHZ=80` (or 75, 90): a slower accelerator clock if 100 MHz does not close.
  - `make bit MCOLS=4`: faster prefill and batched decode. The host reads MCOLS from the bitstream.
- **Host:** `opentpu/host/setup_pcie.sh` covers the XDMA driver, udev, rescan after JTAG and the ID check.
- **Vivado front-end audit:** no construct that is sure to break the build. The risky ones were fixed.

## Today, in order

1. **Install Vivado** into the vivado-docker volume. This needs your AMD login.
2. **Build:**
   ```sh
   cd boards/ypcb-00338
   make lint
   VIVADO_DOCKER=vivado:2026.1 make bit
   ```
   Expect 1.5–3 h, then read `build/vivado/reports/SUMMARY.txt`.
3. **If core_clk fails at 100 MHz:**
   - For a working board first: `make bit CORE_MHZ=80`. Decode is DRAM-bound, so the loss is small.
   - Then look at `timing_worst.rpt`. By the yosys estimate the limit is now inside the quantizer (its writer into the ACT RAM). The next cross-unit limit is the TMEM arbiter: a combinational grant that drives the units' clock enables in the same cycle.
   - The structural fix for the arbiter is to arbitrate one cycle ahead (registered grants).
4. **Program and bring up:**
   - `make program`, then `opentpu/host/setup_pcie.sh --rescan` on the PC.
   - `otpu-selftest`, then `--model qwen3`.
   - `otpu-chat --backend board`.
5. **First things to check on hardware** (docs/board.md section 6):
   - MIG calibration on both channels (channel 1 lanes 6–7 had read-capture trouble on the old SoC).
   - That the ECC mode does read-modify-write for partial writes (the self-test `pattern` stage checks it).
   - PCIe link width.
