# Status: 2026-09-24, before the first Vivado build

## Where things stand

- **Model:** Qwen3-0.6B runs with real weights.
  - On the ISA simulator it matches Hugging Face.
  - On the RTL at the board configuration it is bit-exact with the ISA simulator.
- **Bring-up rehearsal:** from a clean clone, `tools/board_selftest.py --sim --qwen models/Qwen3-0.6B` passed all 9 stages on the board model.
  - The model is `sim/verilator/tb_board.sv`: AXI-Lite registers, program loader, AXI adapter, two DDR3 channels with random stalls.
  - The run went through the same host driver as the card (`host/board.py`).
  - Greedy decoding matched the ISA simulator token for token, at 6.38 Mcycles/token (about 15.7 tok/s at 100 MHz).
  - The rehearsal ran before the night's timing changes. They add a few cycles per instruction, and the per-token cycle count the tournaments track stayed within 0.2%.
- **Full board, yosys estimate:**

  | | Start of night | Now |
  |---|---|---|
  | LUT | 131,915 | 87,763 |
  | FF | 55,890 | 38,404 |
  | DSP | 416 | 283 |
  | BRAM36 | 603 | 603 |
  | Logic delay | 14.94 ns | 6.28 ns |
  | Est. fmax | 41 MHz | 95 MHz |

  The estimate is `1000 / (1.6 * logic + 0.5)`, not signoff; Vivado gives the real number.

## What changed overnight

- **Cross-unit timing**, each fix verified in all RTL modes (default, cascade MXU, board with AXI and boot, heavy AXI stalls) plus the board, perf and Qwen3 tests:
  - TMEM arbiter: write-mask arbitration.
  - Grants reach only the BRAM enables. Read addresses are merged from requests, and write lanes are merged per port with the grant applied at the last level.
  - Registered TMEM data at the inputs of the VPU, the quantizer prescale and the MXU accumulate path.
  - Registered VPU writes.
  - Registered request addresses and masks in the DMA (LD mask, address, write), the collective unit and the MXU drain, plus the RMAX base.
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
  - `make bit MCOLS=4`: faster prefill and batched decode. Run the host with `OTPU_MCOLS=4`.
- **Host:** `host/setup_pcie.sh` covers the XDMA driver, udev, rescan after JTAG and the ID check.
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
   - Then look at `timing_worst.rpt`. By the yosys estimate the next limit is the TMEM arbiter itself: a combinational grant that drives the units' clock enables in the same cycle.
   - The structural fix is to arbitrate one cycle ahead (registered grants).
4. **Program and bring up:**
   - `make program`, then `host/setup_pcie.sh --rescan` on the PC.
   - `python3 tools/board_selftest.py`, then `--qwen models/Qwen3-0.6B`.
   - `tools/chat.py --backend board`.
5. **First things to check on hardware** (docs/board.md section 6):
   - MIG calibration on both channels (channel 1 lanes 6–7 had read-capture trouble on the old SoC).
   - That the ECC mode does read-modify-write for partial writes (the self-test `pattern` stage checks it).
   - PCIe link width.
