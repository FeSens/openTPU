// The accelerator as built for the PCIe boards: S slices (one on the YPCB-00338, one or two on
// the BCU1525), each with its DRAM adapter otpu_native_dram onto NCH memory channels'
// native ports n_* (one command per CB-byte beat; the cards' two channels of 64-byte beats for
// otpu_mem_ch in front of LiteDRAM's native port, the BCU1525 WG targets' eight channels of 128: slice
// s on channels NCH s .. NCH s + NCH - 1), the collective unit, and the host control registers
// (AXI4-Lite slave). Everything runs on the core clock; the board's top does the clock crossings to the
// memory controllers and the PCIe bridge (otpu_native_sys on the YPCB). The slices share one
// control block: RUN, LOAD (each slice loads the program from its own DRAM) and the arguments
// go to every slice; HALTED when every slice halted, ERROR / WAIT_TO when any did; ICOUNT is
// slice 0's, ICOUNT_s each slice's (0xC0 + 4s); the trace and the activity counters are slice
// 0's, except DRAM_RD / DRAM_WR (every channel) and INSTR (every slice). Slice 0's instances
// keep the one-slice names (u_slice, u_mem), the others are g_slice[s]. The control
// block also holds the free-running activity counters and reads out the hardware trace
// (otpu_trace; docs/observability.md); it also drives the I2C pins low when the host asks and
// reads their levels back (the host bit-bangs I2C).
// At S > 1 every slice sits in its own SLR (the BCU1525's), a crossing away from the control
// block and the collective unit: everything between a slice and them goes through two register
// stages each way (otpu_xr: one on each side of the crossing; g_xs), the slices' control too (so
// they keep starting together), and the collective runs through otpu_coll_hub and a port per
// slice (otpu_coll_port: each slice grants the collective's TMEM accesses on its own). At S = 1
// the signals are wired as before (g_x1: the same logic, tools/synth/board_equiv.sh). XRS sets
// the register stages each way (2: one on each side of the crossing; more: an otpu_xr chain
// between them), COLL_PIPE the hub's reads: one in flight (0), or as many as the round trip
// through the stages takes (1: otpu_coll_hub NR, the writes' credits F with them).
// NAT_XSTG > 0 puts that many register stages each way between each slice and its adapter
// (otpu_xstg u_xs: the slice's ports, its reads and idles, the posted W beats and their counts),
// for a floorplan with the adapters and the channels' core halves an SLR away from their slices;
// 0 wires them (no cells).
module otpu_board #(
  parameter int NC         = 1,         // cards (1: the original board; > 1: remote hub)
  parameter int S          = 1,         // slices, each on NCH memory channels
  parameter int D          = 128,
  parameter int MCOLS      = 2,
  parameter int ACT_BLOCKS = 128,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int TMEM_IMPL  = 0,         // TMEM: 0 a copy per read port, 1 banked (three copies;
                                         //   otpu_slice: RPB 1, WPB 1)
  parameter bit TMEM_RQ    = 1'b0,      // TMEM_IMPL 1: request-priority bank addresses
  parameter int TMEM_NSB   = 8,         // TMEM_IMPL 3 (one copy, docs/tmem_nocopy.md): RAMs a bank,
  parameter int TMEM_SBP   = 3,         //   the sub-bank's word-address bit, its hash, replays,
  parameter int TMEM_SBH   = 0,         //   true dual-port RAMs, QUANT 3 and COLL the DMA's guests,
  parameter bit TMEM_RPL   = 1'b0,      //   the RAMs pumped at 2x on clk_f (1), the read ports'
  parameter bit TMEM_TDP   = 1'b0,      //   ranks and RAM sides (otpu_slice)
  parameter bit TMEM_GST   = 1'b0,
  parameter int TMEM_PMP   = 0,
  parameter int TMEM_PRK   = 0,
  parameter int TMEM_SDM   = -1,
  parameter string TMEM_STYLE = "block", // the pumped RAMs' (otpu_slice; "ultra": the WG targets')
  parameter string IMEM_STYLE = "auto",  // IMEM's RAMs (otpu_slice; "ultra": the WG targets')
  parameter string IMEM_WC_STYLE = "",   // its wait entries', if not IMEM_STYLE's (otpu_slice)
  parameter string MXU_RF_STYLE = "block", // the MXU's result FIFO (otpu_slice; the WG targets'
                                          //   "distributed")
  parameter string MXU_FIFO_STYLE = "block", // the MXU's chunk FIFO lanes (otpu_slice; the WG targets'
                                          //   "ultra")
  parameter bit MXU_WASK   = 1'b1,      // the MXU's UltraRAM lanes read stage by stage (otpu_slice)
  parameter int MXU_RF_ROWS = 0,       // the MXU's result FIFO rows (otpu_slice; 0: its rule)
  parameter bit MXU_BRAW   = 1'b1,      // the MXU's column ends' B from raw scale (otpu_slice)
  parameter bit MXU_AASK   = 1'b1,      // the ACT RAM read by the MXU's chain stage (otpu_slice)
  parameter int ACT_UCOLS  = 0,         // the ACT RAM's columns in UltraRAM (otpu_slice)
  parameter int RPB        = (TMEM_IMPL == 1) ? 1 : 8 * LANES,  // every read lane (TMEM is
                                          // replicated per read port): the slice's shallow
                                          // write-mask arbiter
  parameter int WPB        = 1,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int MXU_QD     = 2,       // MXU commands queued (otpu_mxu QD; 2..7)
  parameter bit QUANT_R1   = 1'b0,    // RSCALE's row factor read once a row (otpu_quant R1; timing only)
  parameter int QUANT_PW   = 1,       // a buffered QACT pass 1's chunks a cycle (otpu_quant PW; timing only)
  parameter bit QUANT_SPLIT = 1'b0,   // QACT / QST flag LO: the split's lo part (otpu_quant SPLIT; Config.SPLITQ)
  parameter bit QUANT_HL    = 1'b0,   // QACT flag HL: q and lo from one read (otpu_quant HL; Config.QHL)
  parameter int QUANT_SWW  = 1,       // the SW port's words (otpu_quant SWW; timing only)
  parameter bit QUANT_QSTB  = 1'b0,   // QST rows through the row buffer (otpu_quant QSTB; timing only)
  parameter bit QUANT_QSTHL = 1'b0,   // QST HL: a GRP ROW QST's hi and lo from one read (otpu_quant QSTHL; Config.QSTHL)
  parameter bit QONCE      = 1'b0,    // QACT flag ST and LDA (otpu_slice; Config.QONCE, CAPS2 bit24)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with exp2/recip/rsqrt
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int CORE_KHZ    = 100000,  // the core clock (CORE_KHZ register)
  parameter int CORE_KHZ1   = CORE_KHZ, // the fallback clock (otpu_clk_k), or CORE_KHZ: none
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS     = 0,       // DDR3 data rate (DDR_MTS register; 0: not given)
  parameter int TRACE_DEPTH = 16384,   // trace records (a power of two; 0: no trace buffer)
  parameter int TRACE_QD    = 32,      // trace capture queue (cycles with events)
  parameter int PQ_WIN      = 1024,    // cycles per P/Q counter window
  parameter bit HAS_I2C     = 1'b1,    // the I2C pins are wired (CAPS bit2)
  parameter bit CHASH       = 1'b1,    // hashed channel interleave (otpu_native_dram; CAPS bit7)
  parameter bit DSTEP       = 1'b1,    // the DMA's DSTEP datapath (CAPS bit6; 0 leaves it out)
  parameter bit HOSTCAL     = 1'b0,    // the host calibrates the DDR3 controllers (CAPS bit27: LiteDRAM)
  parameter bit COLL_GATHER = 1'b1,    // 0 (S = 1 only): the collective unit runs BAR alone, no
                                       // GATHER (otpu_coll_bar; CAPS2 bit0 = 0, docs/infra_hunt.md 5)
  parameter bit COLL_PIPE  = 1'b0,    // S > 1: the hub's reads in flight, the round trip's
                                       //   (otpu_coll_hub NR > 1); 0: one at a time
  parameter int XRS        = 2,       // S > 1: register stages each way between a slice and the
                                       //   control block / the collective unit (>= 2: otpu_xr)
  parameter bit HAS_BMC     = 1'b0,    // the board's top has the BMC's UART bridge (CAPS2 bit4)
  parameter int BHW         = 0,       // the board's memory's bank-half window, the slice's bytes'
                                       //   log2 (the BCU1525's pairs; CAPS2 bits 11:6; 0: none)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int PLATFORM    = 0,       // the target's platform id (CFG_PLATFORM; opentpu/target.py)
  parameter logic [31:0] ARCH_HASH = 32'h0, // the target's arch_hash (CFG_HASH)
  parameter int NCH         = 2,       // memory channels per slice, their beat bytes (otpu_native_dram;
  parameter int CB          = 64,      //   CFG_DRAM): a slice's 4 GiB is its channels' 4 GiB / NCH
  parameter int NAT_RD      = 256 / NCH, // otpu_native_dram's per-channel depths in beats: B's, A's
  parameter int NAT_AD      = 64,      //   and W's reads in flight, the SW queue (the target's
  parameter int NAT_XD      = 64,      //   [uarch] NAT_*; these: the adapter's defaults)
  parameter int NAT_WQD     = 32,
  parameter int NAT_APF     = 32,      //   A's read runs (otpu_native_dram APF; AD >= 2 APF - 1)
  parameter int NAT_RBB     = 0,       //   a mask by slice: slice s's B FIFO in block RAM (RB_BLOCK)
  parameter bit NAT_FILL    = 1'b1,    //   partial SW beats read and written whole (SW_FILL)
  parameter bit NAT_WPOST   = 1'b0,    //   write data posted at the take (WPOST: the WG targets' bridges)
  parameter bit NAT_SWG     = 1'b0,    //   WPOST: one SW gather beat and one posted beat a cycle (SWG)
  parameter bit NAT_XPOST   = 1'b0,    //   port W's beats posted into the MXU's chunk FIFO by the
  parameter int NAT_XQB     = 0,       //   channels (XPOST, WG > 1; 2^NAT_XQB writes a beat on
                                       //   p_clk: the WG targets' bridges 2, aclk's 256-bit quarters, the
                                       //   simulation's model 0)
  parameter int NAT_XSTG    = 0,       //   register stages each way between a slice and its
                                       //   adapter (otpu_xstg; 0: wired), and the posts' and their
  parameter int NAT_XSTG_P  = NAT_XSTG, //  counts' on p_clk (otpu_xstg NP)
  parameter bit NAT_RHB     = 1'b0,    //   read data from the channels in half beats (RHB: the
  parameter bit NAT_WHB     = 1'b0,    //   the WG targets' bridges, R1), write data to them (WHB, R2)
  parameter int WG          = 1,       // weight rows a group (otpu_slice; port W's P = WG x D;
                                       //   above 1 every MM in groups: no column reuse, CAPS bit5)
  parameter int SE_R        = 0,       // streams on an engine of their own (otpu_slice)
  parameter int DMA_DEPTH   = 128,     // otpu_dma DEPTH: a DSTEP's read-ahead (otpu_slice)
  parameter int DMA_LDQ     = 1,       // otpu_dma LDQ: 2, an LD's DRAM reads under the command before it
  parameter int MXU_PUMP    = 0,       // the MXU's array pumped on clk_f (otpu_mxu PUMP)
  parameter int VPU_FSUM    = 0,       // otpu_vpu FSUM: EXP2SUB's fused row sums (Config.FSUM)
  parameter int VPU_SIG     = 0,       // otpu_vpu VSIG: VOP SIGMOID / SILU (Config.VSIG)
  parameter int VPU_MAC     = 0,       // otpu_vpu MAC: VOP MAC (Config.VMAC)
  parameter int VPU_OUTER1  = 0,       // otpu_vpu OUTER1: VOP OUTER decay one on MAC's (Config.OUTER1, CAPS2 bit23)
  parameter int VPU_CNS     = 3,       // otpu_vpu CNS: the composites' stages per lane (3, or 5: two passes)
  parameter bit OUTER       = 1'b1,    // VOP OUTER (otpu_slice; 0: a VPU without it, CAPS2 bit1)
  parameter bit DSTEP2      = 1'b0,    // DSTEP's pair (otpu_slice; CAPS2 bit2)
  parameter int ST16        = 0,       // a stream's DRAM state in 16 bits: 1 e6m9, 2 bf16 (otpu_slice; CAPS2 bits 14:13)
  parameter bit ST16S       = 1'b0,    // with ST16: STREAM's flags ST16R / ST16W, one side alone (otpu_slice; CAPS2 bit18)
  parameter bit KT          = 1'b0,    // MM's K tail (otpu_slice, WG > 1; CAPS2 bit5)
  parameter int MXU_SPL     = 0,       // MM SPL, a split's terms as one MM (otpu_slice, WG > 1; CAPS2 bit25)
  parameter bit ICACHE      = 1'b0,    // IMEM an instruction cache over the program (otpu_slice;
                                       //   CAPS2 bit12, CFG_UARCH bit28)
  parameter int IC_LINES    = 16,      // ICACHE: its lines (otpu_slice)
  parameter int IC_PF       = 2,       // ICACHE: the lines prefetched ahead (otpu_slice)
  parameter int AW          = 32,      // DRAM byte address bits (otpu_slice; otpu_seq AW)
  parameter int DRAM_LOG2   = 32,      // a slice's DRAM, 2^DRAM_LOG2 bytes (<= 2^AW; CFG_DRAM):
  parameter int CH_LOG2     = DRAM_LOG2 - $clog2(NCH)  //   its NCH channels' 2^CH_LOG2 each
) (
  input  logic         clk,
  input  logic         clk_f,          // TMEM_PMP 1: 2 x clk, rising with it (the MMCM's); else unused
  input  logic         rst,            // synchronous, active high
  input  logic         ctl_mrst,       // the control master's reset (the SmartConnect's, in clk:
                                       //   PCIe hot reset or link down too): otpu_axil_iso
  input  logic [NCH*S-1:0] calib,      // memory controllers calibrated (any clock domain)
  input  logic [NCH*S-1:0] ded,        // a channel's ECC counted an uncorrectable word (any clock
                                       //   domain; STATUS ECC_DED)
  input  logic [11:0]  temp,           // XADC die-temperature code (any clock domain, slow)
  output logic [2:0]   led,
  // ---- I2C pins (otpu_fpga_top_ld's IOBUFs): 1 drives the line low; the levels (any clock)
  output logic [3:0]   i2c_lo,         // SCL0 SDA0 (LM73 bus), SCL1 SDA1 (PCIe SMBus)
  input  logic [4:0]   i2c_pin,        // the same four lines, then ALERT0 (LM73, active low)
  // Two link ports on the core clock (otpu_link in the FPGA top). All outputs are tied
  // off at NC 1. x2_rst resets the link and the hub's link side independently of RUN.
  output logic [1:0][255:0] x2_s_tdata,
  output logic [1:0]        x2_s_tvalid, x2_s_tlast,
  input  logic [1:0]        x2_s_tready,
  input  logic [1:0][255:0] x2_m_tdata,
  input  logic [1:0]        x2_m_tvalid, x2_m_tlast, x2_m_thdr,
  output logic [1:0]        x2_m_tready,
  input  logic [1:0]        x2_up, x2_rx_err, x2_abort_in,
  output logic [1:0]        x2_abort,
  output logic             x2_rst,
  output logic [7:0]       x2_card, x2_ncards,
  output logic             x2_c_we,
  output logic [7:0]       x2_c_addr,
  output logic [31:0]      x2_c_wd,
  input  logic [31:0]      x2_c_rd,
  // ---- control: AXI4-Lite slave
  input  logic [11:0]  s_ctl_awaddr,
  input  logic         s_ctl_awvalid,
  output logic         s_ctl_awready,
  input  logic [31:0]  s_ctl_wdata,
  input  logic [3:0]   s_ctl_wstrb,
  input  logic         s_ctl_wvalid,
  output logic         s_ctl_wready,
  output logic [1:0]   s_ctl_bresp,
  output logic         s_ctl_bvalid,
  input  logic         s_ctl_bready,
  input  logic [11:0]  s_ctl_araddr,
  input  logic         s_ctl_arvalid,
  output logic         s_ctl_arready,
  output logic [31:0]  s_ctl_rdata,
  output logic [1:0]   s_ctl_rresp,
  output logic         s_ctl_rvalid,
  input  logic         s_ctl_rready,
  // ---- memory channels 0 .. NCH S - 1: native masters (otpu_native_dram; slice s on NCH s ..)
  output logic [NCH*S-1:0]        n_cvalid,
  input  logic [NCH*S-1:0]        n_cready,
  output logic [NCH*S-1:0]        n_cwe,
  output logic [NCH*S-1:0][DRAM_LOG2-1-$clog2(NCH*CB):0] n_caddr,  // the beat in the channel
                                             //   (the card's: address bits 30:6)
  output logic [NCH*S-1:0]        n_wvalid,  // write data: a beat per write command, in their order
  input  logic [NCH*S-1:0]        n_wready,
  output logic [NCH*S-1:0][CB*8-1:0] n_wdata,
  output logic [NCH*S-1:0][CB-1:0] n_wmask,  // 1 = write the byte
  input  logic [NCH*S-1:0]        n_rvalid,  // read data in read-command order, no backpressure
  input  logic [NCH*S-1:0][CB*8-1:0] n_rdata,
  input  logic [NCH*S-1:0][15:0]  n_wdone,   // write beats the controller has taken (mod 2^16)
  input  logic [NCH*S-1:0][5:0]   n_wfree,   // NAT_WPOST: a channel's write-data FIFO's free beats
  input  logic [NCH*S-1:0]        n_err,     // a channel's controller broke its port contract (sticky)
  // NAT_XPOST (otpu_native_dram, otpu_mxu XPOST): a read command's source and chunk FIFO entry; the
  // channels' port W beats posted into lane c of entry p_slot of their slice's MXU on p_clk
  // (quarter q of the beat at NAT_XQB 2: p_q one-hot, bit q with p_we, else 0; p_slot and p_q in
  // copies of equal registers, a RAM group's each: otpu_mxu XPOST), and the beats so posted, a
  // count a channel (gray)
  output logic [NCH*S-1:0][1:0]   n_csrc,
  output logic [NCH*S-1:0][$clog2(FIFO_DEPTH)-1:0] n_cslot,
  input  logic                    p_clk,
  input  logic [NCH*S-1:0]        p_we,
  input  logic [NCH*S-1:0][1:0][((NAT_XQB > 0) ? (1 << NAT_XQB) : 1)-1:0] p_q,
  input  logic [NCH*S-1:0][4:0][$clog2(FIFO_DEPTH)-1:0] p_slot,
  input  logic [NCH*S-1:0][(D*8 >> NAT_XQB)-1:0] p_d,
  input  logic [NCH*S-1:0][$clog2(FIFO_DEPTH)+1:0] p_cntg
);
  import otpu_pkg::*;
  localparam int CL = $clog2(NCH), P = NCH * CB;
  localparam int WA = waw(AW);                       // the slices' word addresses (otpu_slice)
  localparam int NCAW = DRAM_LOG2 - $clog2(P);       // a channel's beat (n_caddr)
  initial if (AW < 32 || AW > AW_MAX || DRAM_LOG2 > AW || NCAW < 1 || CH_LOG2 != DRAM_LOG2 - CL)
    $fatal(1, "otpu_board: AW %0d, DRAM_LOG2 %0d, CH_LOG2 %0d (DRAM_LOG2 <= AW, CH_LOG2 = %0d)",
           AW, DRAM_LOG2, CH_LOG2, DRAM_LOG2 - CL);
  // the adapters' channel beats (AW - log2(P) bits) on the channels' NCAW: a slice's DRAM below
  // 2^AW drops the bits above it (in simulation an address there is an error)
  logic [NCH*S-1:0][AW-1-$clog2(P):0] m_caddr;
  localparam int AB = (WG > 1) ? CB : 32;            // port A's read bytes (otpu_slice)
  // a QST's footprint beat (otpu_slice QB): the adapter's fill beat, CB, where it fills partial SW
  // beats (NAT_FILL) and CB > 64; else 64 bytes (the cards' CB, the WG targets' NAT_FILL 0)
  localparam int QB = (NAT_FILL && CB > 64) ? $clog2(CB) : 6;
  localparam int DCW = $clog2(NCH + 1);   // a slice's channels' beats in a cycle

  // ---- calibration and uncorrectable-ECC flags from the memory controllers' clock domains
  // (registers there; levels, each bit on its own)
  (* ASYNC_REG = "TRUE" *) logic [NCH*S-1:0] cal_s1, cal_s2, ded_s1, ded_s2;
  // CALIB0 / CALIB1: every slice's even / odd channels (the cards': its first / second)
  logic [1:0] cal_all;
  always_comb begin
    cal_all = 2'b11;
    for (int k = 0; k < S * NCH; k++) cal_all[k % 2] &= cal_s2[k];
  end
  always_ff @(posedge clk) begin
    cal_s1 <= calib;
    cal_s2 <= cal_s1;
    ded_s1 <= ded;
    ded_s2 <= ded_s1;
  end

  // ---- die temperature: two flip-flops per bit, then a code is taken only when two
  // consecutive samples agree (it changes slowly: a sample caught mid-change is skipped). Valid
  // once the XADC (the block design's, on the core clock) has reported a reading; it does not
  // wait for the memory (the MIG builds' XADC was channel 0's, valid once it was calibrated).
  (* ASYNC_REG = "TRUE" *) logic [11:0] tmp_s1, tmp_s2;
  logic [11:0] tmp_s3, temp_q;
  logic        temp_v;
  always_ff @(posedge clk) begin
    tmp_s1 <= temp;
    tmp_s2 <= tmp_s1;
    tmp_s3 <= tmp_s2;
    if (rst) begin
      temp_q <= '0;
      temp_v <= 1'b0;
    end else if (tmp_s2 == tmp_s3 && tmp_s3 != '0) begin
      temp_q <= tmp_s3;
      temp_v <= 1'b1;
    end
  end

  // ---- I2C pin levels: two flip-flops each (the host samples them at microsecond pace)
  (* ASYNC_REG = "TRUE" *) logic [4:0] i2c_s1, i2c_s2;
  always_ff @(posedge clk) begin
    i2c_s1 <= i2c_pin;
    i2c_s2 <= i2c_s1;
  end

  // ---- control (slice 0's signals; the others' in g_slice)
  logic run, ld_start, ld_busy, halted, error, wait_to, wr_idle, rd_idle, a_inval;
  logic [31:0] ld_addr, ld_n;
  logic [S-1:0][31:0] icount;        // each slice's
  // every slice's, as the control block sees it: [0] slice 0's, [s] g_slice[s]'s
  logic [S-1:0] sl_ld_busy, sl_halted, sl_error, sl_wait_to, sl_wr_idle, sl_rd_idle;
  logic [S-1:0][1:0] sl_ret;         // instructions retired this cycle
  // the slices' own (q_*) and what each slice is given (x_*): the same at S = 1, two register
  // stages apart at S > 1 (g_xs)
  logic [S-1:0] q_ld_busy, q_halted, q_error, q_wait_to, q_wr_idle, q_rd_idle;
  logic [S-1:0][1:0] q_ret;
  logic [S-1:0][31:0] q_icount;
  logic [S-1:0] x_rst, x_srst, x_ld_start;
  logic [S-1:0][31:0] x_ld_addr, x_ld_n;
  logic [31:0] x_arg [S][8];
  // the free-running counters' increments: every channel's beats, every slice's instructions
  localparam int CW = $clog2(S * NCH + 1), IW = $clog2(2 * S + 1);
  logic [CW-1:0] dram_rd, dram_wr;
  logic [IW-1:0] instr;
  logic [31:0] arg [8];               // the run's arguments (ARG0..7: R8..R15 at the start)
  logic a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_rvalid, b_rtag, b_rdy;
  logic [CL-1:0] b_par;
  logic pw_rdy, pw_req, pw_one;
  logic [WA-1:0] pw_addr;
  logic [NCH-1:0] pw_bv;
  logic [P*8-1:0] pw_bd;
  logic [$clog2(FIFO_DEPTH)-1:0] pw_slot;
  logic [NCH-1:0][$clog2(FIFO_DEPTH)+1:0] p_cnt;
  logic [WA-1:0] a_addr, b_addr;
  logic [31:0] a_wdata;
  logic [AB*8-1:0] a_rdata;
  logic [3:0]  a_be;
  logic [4*QUANT_SWW-1:0] sw_be;
  logic        sw_req, sw_rdy;
  logic        m_wr_idle, m_rd_idle;   // slice 0's adapter's wr_idle and rd_idle (q_wr_idle[0],
  logic        m_rv;                   //   q_rd_idle[0]: with its SW and port registers, below);
                                       //   m_rv: a b_rvalid the cycle before
  logic [WA-1:0] sw_addr;
  logic [32*QUANT_SWW-1:0] sw_wdata;
  logic [D/4-1:0] b_wmask;
  logic [D*8-1:0] b_wdata, b_rdata;
  // slice 0's posted W beats as it takes them, and its adapter's side of u_xs (d*: the same
  // signals at NAT_XSTG 0)
  logic [NCH-1:0] sp_we;
  logic [NCH-1:0][1:0][((NAT_XQB > 0) ? (1 << NAT_XQB) : 1)-1:0] sp_q;
  logic [NCH-1:0][4:0][$clog2(FIFO_DEPTH)-1:0] sp_slot;
  logic [NCH-1:0][(D*8 >> NAT_XQB)-1:0] sp_d;
  logic d_rst, d_flush, da_rdy, da_req, da_we, da_rvalid, dsw_rdy, dsw_req, db_rdy, db_req, db_tag;
  logic db_we, db_rvalid, db_rtag, dw_rdy, dw_req, dw_one, d_wr_idle, d_rd_idle;
  logic [WA-1:0] da_addr, dsw_addr, db_addr, dw_addr;
  logic [31:0] da_wdata;
  logic [32*QUANT_SWW-1:0] dsw_wdata;
  logic [3:0] da_be;
  logic [4*QUANT_SWW-1:0] dsw_be;
  logic [AB*8-1:0] da_rdata;
  logic [D/4-1:0] db_wmask;
  logic [D*8-1:0] db_wdata, db_rdata;
  logic [CL-1:0] db_par;
  logic [$clog2(FIFO_DEPTH)-1:0] dw_slot;
  logic [NCH-1:0] dw_bv;
  logic [P*8-1:0] dw_bd;
  logic [NCH-1:0][$clog2(FIFO_DEPTH)+1:0] d_p_cnt, d_p_cntg;

  perf_t pf, pf_c;                   // slice 0's, and as the control block sees it
  logic        tr_en, tr_stop, tr_clear, tr_busy;
  logic [31:0] tr_addr, tr_count, tr_drop;
  logic [63:0] tr_rdata;

  // the registers reset with rst (PERST#), the master also on a hot reset: an access in flight
  // then is completed here and its response dropped (otpu_axil_iso)
  logic [11:0] c_awaddr, c_araddr;
  logic [31:0] c_wdata, c_rdata;
  logic [3:0]  c_wstrb;
  logic [1:0]  c_bresp, c_rresp;
  logic c_awvalid, c_awready, c_wvalid, c_wready, c_bvalid, c_bready;
  logic c_arvalid, c_arready, c_rvalid, c_rready;
  otpu_axil_iso #(.AW(12)) u_iso (
    .clk, .m_rst(ctl_mrst), .s_rst(rst),
    .m_awaddr(s_ctl_awaddr), .m_awvalid(s_ctl_awvalid), .m_awready(s_ctl_awready),
    .m_wdata(s_ctl_wdata), .m_wstrb(s_ctl_wstrb), .m_wvalid(s_ctl_wvalid),
    .m_wready(s_ctl_wready), .m_bresp(s_ctl_bresp), .m_bvalid(s_ctl_bvalid),
    .m_bready(s_ctl_bready), .m_araddr(s_ctl_araddr), .m_arvalid(s_ctl_arvalid),
    .m_arready(s_ctl_arready), .m_rdata(s_ctl_rdata), .m_rresp(s_ctl_rresp),
    .m_rvalid(s_ctl_rvalid), .m_rready(s_ctl_rready),
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready), .s_wdata(c_wdata),
    .s_wstrb(c_wstrb), .s_wvalid(c_wvalid), .s_wready(c_wready), .s_bresp(c_bresp),
    .s_bvalid(c_bvalid), .s_bready(c_bready), .s_araddr(c_araddr), .s_arvalid(c_arvalid),
    .s_arready(c_arready), .s_rdata(c_rdata), .s_rresp(c_rresp), .s_rvalid(c_rvalid),
    .s_rready(c_rready));

  logic x2_lrst;
  logic [31:0] coll_wd;
  logic [7:0] coll_status;
  logic [15:0] x2_seq;
  logic coll_xerr;
  assign x2_rst = NC > 1 && (rst || x2_lrst);
  if (NC == 1) begin : g_no_x2
    assign {x2_s_tdata, x2_s_tvalid, x2_s_tlast, x2_m_tready, x2_abort,
            coll_status, coll_xerr, x2_seq} = '0;
  end
  otpu_ctrl #(.NC(NC), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .LANES(LANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID),
              .CORE_KHZ1(CORE_KHZ1), .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_TEMP(1'b1),
              .HAS_I2C(HAS_I2C), .CHASH(CHASH), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL),
              .COLL_GATHER(COLL_GATHER), .HAS_BMC(HAS_BMC), .BHW(BHW), .OUTER(OUTER), .PAIR(WG == 1), .DSTEP2(DSTEP2 && DSTEP),
              .KT(KT && WG > 1),
              .SE_OWN(SE_R > 0 && DSTEP && LANES == 8 && D >= 32),   // (otpu_slice SE_OWN)
              .FSUM(VPU_FSUM != 0 && DSTEP && LANES == 8 && D >= 32), // (otpu_slice's VPU FSUM)
              .SPLITQ(QUANT_SPLIT), .QHL(QUANT_HL), .QSTHL(QUANT_QSTHL),
              .VSIG(VPU_SIG != 0 && DSTEP && LANES == 8 && D >= 32),
              .VMAC(VPU_MAC != 0), .OUTER1(VPU_OUTER1 != 0), .QONCE(QONCE),
              .SPLMM(MXU_SPL != 0 && WG > 1),
              .LDP(S > 1 ? 2 + 2 * XRS : 2),  // LOAD to a slice's busy and back
              // each slice on its NCH channels' 2^CH_LOG2 bytes each (otpu_native_dram)
              .S(S), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .ACT_BLOCKS(ACT_BLOCKS),
              .DRAM_LOG2(DRAM_LOG2), .NCH(NCH), .CH_LOG2(CH_LOG2), .CB(CB), .MXU_IMPL(MXU_IMPL),
              .TMEM_IMPL(TMEM_IMPL),
              .VPU_CL(VPU_CL), .ULANES(ULANES), .SEQ_WC(SEQ_WC), .WG(WG), .ICACHE(ICACHE),
              .ST16(DSTEP ? ST16 : 0), .ST16S(DSTEP && ST16 != 0 && ST16S), .PLATFORM(PLATFORM),
              .ARCH_HASH(ARCH_HASH)) u_ctrl (
    .clk, .rst,
    .s_awaddr(c_awaddr), .s_awvalid(c_awvalid), .s_awready(c_awready),
    .s_wdata(c_wdata), .s_wstrb(c_wstrb), .s_wvalid(c_wvalid),
    .s_wready(c_wready), .s_bresp(c_bresp), .s_bvalid(c_bvalid),
    .s_bready(c_bready), .s_araddr(c_araddr), .s_arvalid(c_arvalid),
    .s_arready(c_arready), .s_rdata(c_rdata), .s_rresp(c_rresp),
    .s_rvalid(c_rvalid), .s_rready(c_rready),
    .run, .ld_start, .arg, .ld_addr, .ld_n, .ld_busy, .halted, .error, .wait_to, .icount, .wr_idle,
    .axi_err(|n_err),                  // the channels' bridges (otpu_mem_ch)
    .calib(cal_all), .ecc_ded(|ded_s2),
    .b_rd(b_req && b_rdy && !b_we), .b_wr(b_req && b_rdy && b_we),
    .a_rd(a_req && a_rdy && !a_we), .a_wr(sw_req && sw_rdy), .b_wait(pf_c.bwait),
    .temp_v, .temp(temp_q),
    .mxu_busy(pf_c.sq.busy[U_MXU]), .mxu_mac(pf_c.mac), .mxu_starve(pf_c.starve),
    .vpu_busy(pf_c.sq.busy[U_VPU]),
    .qnt_busy(pf_c.sq.busy[U_Q]), .dma_busy(pf_c.sq.busy[U_DMA]), .tmem_deny(pf_c.deny),
    .dram_rd, .dram_wr,
    .dram_wait(pf_c.dwait),           // the units' waits (otpu_slice), not a parked register's
    .instr,
    .tr_en, .tr_stop, .tr_clear, .tr_addr, .tr_count, .tr_drop, .tr_busy, .tr_rdata,
    .i2c_lo, .i2c_in(i2c_s2),
    .x2_card, .x2_ncards, .x2_lrst, .coll_wd, .coll_status, .x2_seq,
    .x2_c_we, .x2_c_addr, .x2_c_wd, .x2_c_rd);

  // ---- hardware trace
  if (TRACE_DEPTH != 0) begin : g_trace
    otpu_trace #(.DEPTH(TRACE_DEPTH), .QD(TRACE_QD), .WIN(WIN)) u_trace (
      .clk, .rst, .pf(pf_c), .en(tr_en && run), .stop(tr_stop), .clear(tr_clear), .raddr(tr_addr),
      .rdata(tr_rdata), .count(tr_count), .drop(tr_drop), .busy(tr_busy));
  end else begin : g_no_trace
    assign tr_rdata = '0;
    assign tr_count = '0;
    assign tr_drop = '0;
    assign tr_busy = 1'b0;
  end

  // ---- the slices (held in reset while RUN is 0) and the collective unit.
  // core_rst reaches ~15k flip-flops across the die: synthesis replicates it (a single copy's
  // net took 9.6 ns, the worst core_clk path at 100 MHz). A run starts from a quiet memory
  // adapter (go_ok: no read in flight, every write taken, sampled while RUN is 0 and until the
  // run starts): the adapter does not reset with RUN, so a run the host stopped leaves its reads
  // and writes going, and the next run's units would take the old reads' data
  (* max_fanout = 256 *) logic core_rst;
  logic go_ok;
  always_ff @(posedge clk) begin
    go_ok <= !rst && ((run && go_ok) || (rd_idle && wr_idle));
    core_rst <= rst || !run || !go_ok;
  end

  // the run's status from every slice
  assign ld_busy = |sl_ld_busy;
  assign halted  = &sl_halted;
  logic  coll_gerr;                    // a GATHER on the BAR-only collective unit (COLL_GATHER 0)
  assign error   = |sl_error || coll_gerr || coll_xerr;
  assign wait_to = |sl_wait_to;
  assign wr_idle = &sl_wr_idle;
  assign rd_idle = &sl_rd_idle;
  assign q_ret[0] = pf.sq.ret;

  // the collective unit's ports (coll_*) and each slice's (sc_*): wired together at S = 1, the
  // hub and a port per slice between them at S > 1
  logic [S-1:0] coll_req;
  logic         coll_ack, coll_gnt;
  cmd_t         coll_cmds [S];
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  logic [S-1:0]                  sc_req, sc_ack, sc_gl, sc_gnt, sc_wtop;
  cmd_t                          sc_cmd [S];
  logic [S-1:0][LANES-1:0]       sc_ren, sc_wen;
  logic [S-1:0][LANES-1:0][31:0] sc_raddr, sc_rdata, sc_waddr, sc_wdata;

  otpu_slice #(.SID(0), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .MXU_E(MXU_E), .MXU_QD(MXU_QD), .QUANT_R1(QUANT_R1), .QUANT_PW(QUANT_PW), .QUANT_SPLIT(QUANT_SPLIT), .QUANT_HL(QUANT_HL), .QUANT_SWW(QUANT_SWW), .QUANT_QSTB(QUANT_QSTB), .QUANT_QSTHL(QUANT_QSTHL), .QONCE(QONCE), .VPU_CL(VPU_CL), .ULANES(ULANES), .PQ_WIN(PQ_WIN),
               .SEQ_WC(SEQ_WC), .HAS_DSTEP(DSTEP), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
               .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
               .TMEM_STYLE(TMEM_STYLE), .IMEM_STYLE(IMEM_STYLE), .ACT_UCOLS(ACT_UCOLS), .NCH(NCH), .CB(CB), .QB(QB),
               .MXU_RF_STYLE(MXU_RF_STYLE), .MXU_FIFO_STYLE(MXU_FIFO_STYLE), .MXU_WASK(MXU_WASK),
               .MXU_RF_ROWS(MXU_RF_ROWS), .MXU_BRAW(MXU_BRAW), .MXU_AASK(MXU_AASK),
               .IMEM_WC_STYLE(IMEM_WC_STYLE),
               .WG(WG), .SE_R(SE_R), .DMA_DEPTH(DMA_DEPTH), .DMA_LDQ(DMA_LDQ), .MXU_PUMP(MXU_PUMP), .VPU_FSUM(VPU_FSUM), .VPU_SIG(VPU_SIG), .VPU_MAC(VPU_MAC), .VPU_OUTER1(VPU_OUTER1), .VPU_CNS(VPU_CNS), .OUTER(OUTER), .XPOST(NAT_XPOST), .XQB(NAT_XQB),
               .CHASH(CHASH), .DSTEP2(DSTEP2), .ST16(ST16), .ST16S(ST16S), .KT(KT), .MXU_SPL(MXU_SPL), .ICACHE(ICACHE), .IC_LINES(IC_LINES),
               .IC_PF(IC_PF), .AW(AW)) u_slice (
    .clk, .clk_f, .sys_rst(x_srst[0]), .rst(x_rst[0]), .rinit(x_arg[0]), .ld_start(x_ld_start[0]),
    .ld_addr(x_ld_addr[0]), .ld_n(x_ld_n[0]),
    .ld_busy(q_ld_busy[0]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(m_wr_idle),
    .rd_idle(m_rd_idle),
    .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
    .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
    .pw_rdy, .pw_req, .pw_addr, .pw_bv, .pw_bd, .pw_slot, .pw_one,
    .p_clk, .p_we(sp_we), .p_q(sp_q), .p_slot(sp_slot), .p_d(sp_d), .p_cnt,
    .coll_req(sc_req[0]), .coll_cmd(sc_cmd[0]), .coll_ack(sc_ack[0]),
    .coll_ren(sc_ren[0]), .coll_raddr(sc_raddr[0]), .coll_rdata(sc_rdata[0]),
    .coll_wen(sc_wen[0]), .coll_waddr(sc_waddr[0]), .coll_wdata(sc_wdata[0]),
    .coll_wtop(sc_wtop[0]), .coll_gnt_local(sc_gl[0]), .coll_gnt(sc_gnt[0]),
    .halted(q_halted[0]), .error(q_error[0]), .wait_to(q_wait_to[0]), .a_inval,
    .icount(q_icount[0]), .pf, .dump(1'b0));

  // the collective's reset through a register of its own, next to it: it leaves reset a cycle
  // after the slice, idle either way (133.33 MHz, 110ec6d: core_rst -> u_coll's state, 0 levels,
  // 97% route, +0.162 ns); a request can only come many cycles after reset (checked)
  logic coll_rst;
  always_ff @(posedge clk) coll_rst <= core_rst;
`ifndef SYNTHESIS
  always @(posedge clk)
    if (coll_rst && |coll_req) $fatal(1, "otpu_board: a collective request in reset");
`endif
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_board: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER) begin : g_coll
    // RQ: its reads from its own request register on the slice's port at S = 1; at S > 1 each
    // slice's otpu_coll_port presents them a cycle later (TMEM's input stage)
    otpu_coll #(.S(S), .LANES(LANES), .RQ(S == 1 && NC == 1)) u_coll (
      .clk, .rst(coll_rst), .req(coll_req), .cmds(coll_cmds), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst(coll_rst), .req(coll_req), .cmds(coll_cmds), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
  end

  // ---- memory: the native adapter
  // port A's held beats go where the host may have written: between runs, a load, a WAITW.
  // Every write the slice gave is taken (STATUS WR_IDLE, go_ok) once the adapter has none
  // pending and neither the slice's SW register nor its port B output register holds one (sw_req,
  // b_req: the registers' valid bits); no read of the slice's is in flight (go_ok) once the
  // adapter has none, neither port A's nor port B's output register holds one (a_req: port A
  // reads only) and port B's input register holds no data (a b_rvalid the cycle before, m_rv). A
  // request or data there is in neither the slice's units nor the adapter (otpu_slice)
  assign q_wr_idle[0] = m_wr_idle && !sw_req && !(b_req && b_we);
  assign q_rd_idle[0] = m_rd_idle && !a_req && !(b_req && !b_we) && !m_rv;
  always_ff @(posedge clk) m_rv <= !x_srst[0] && b_rvalid;
  otpu_xstg #(.N(NAT_XSTG), .NP(NAT_XSTG_P), .D(D), .NCH(NCH), .CB(CB), .AB(AB),
              .XSW($clog2(FIFO_DEPTH)), .XQB(NAT_XQB), .AW(AW), .SWW(QUANT_SWW)) u_xs (
    .clk, .p_clk, .s_rst(x_srst[0]), .s_flush({x_rst[0], x_ld_start[0], a_inval}),
    .s_a_rdy(a_rdy), .s_a_req(a_req), .s_a_we(a_we), .s_a_addr(a_addr), .s_a_wdata(a_wdata),
    .s_a_be(a_be), .s_a_rvalid(a_rvalid), .s_a_rdata(a_rdata),
    .s_sw_rdy(sw_rdy), .s_sw_req(sw_req), .s_sw_addr(sw_addr), .s_sw_wdata(sw_wdata),
    .s_sw_be(sw_be),
    .s_b_rdy(b_rdy), .s_b_req(b_req), .s_b_tag(b_tag), .s_b_we(b_we), .s_b_wmask(b_wmask),
    .s_b_wdata(b_wdata), .s_b_addr(b_addr), .s_b_par(b_par), .s_b_rvalid(b_rvalid),
    .s_b_rtag(b_rtag), .s_b_rdata(b_rdata),
    .s_w_rdy(pw_rdy), .s_w_req(pw_req), .s_w_addr(pw_addr), .s_w_slot(pw_slot),
    .s_w_one(pw_one), .s_w_bv(pw_bv), .s_w_bd(pw_bd),
    .s_wr_idle(m_wr_idle), .s_rd_idle(m_rd_idle), .s_p_cnt(p_cnt),
    .s_p_we(sp_we), .s_p_q(sp_q), .s_p_slot(sp_slot), .s_p_d(sp_d),
    .m_rst(d_rst), .m_flush(d_flush),
    .m_a_rdy(da_rdy), .m_a_req(da_req), .m_a_we(da_we), .m_a_addr(da_addr),
    .m_a_wdata(da_wdata), .m_a_be(da_be), .m_a_rvalid(da_rvalid), .m_a_rdata(da_rdata),
    .m_sw_rdy(dsw_rdy), .m_sw_req(dsw_req), .m_sw_addr(dsw_addr), .m_sw_wdata(dsw_wdata),
    .m_sw_be(dsw_be),
    .m_b_rdy(db_rdy), .m_b_req(db_req), .m_b_tag(db_tag), .m_b_we(db_we), .m_b_wmask(db_wmask),
    .m_b_wdata(db_wdata), .m_b_addr(db_addr), .m_b_par(db_par), .m_b_rvalid(db_rvalid),
    .m_b_rtag(db_rtag), .m_b_rdata(db_rdata),
    .m_w_rdy(dw_rdy), .m_w_req(dw_req), .m_w_addr(dw_addr), .m_w_slot(dw_slot),
    .m_w_one(dw_one), .m_w_bv(dw_bv), .m_w_bd(dw_bd),
    .m_wr_idle(d_wr_idle), .m_rd_idle(d_rd_idle), .m_p_cnt(d_p_cnt), .m_p_cntg(d_p_cntg),
    .c_p_we(p_we[NCH-1:0]), .c_p_q(p_q[NCH-1:0]), .c_p_slot(p_slot[NCH-1:0]),
    .c_p_d(p_d[NCH-1:0]), .c_p_cntg(p_cntg[NCH-1:0]));
  otpu_native_dram #(.D(D), .CHASH(CHASH), .NCH(NCH), .CB(CB), .AB(AB), .RD(NAT_RD),
                     .RB_BLOCK(NAT_RBB[0]),
                     .AD(NAT_AD), .XD(NAT_XD), .WQD(NAT_WQD), .APF(NAT_APF), .SW_FILL(NAT_FILL),
                     .WPOST(NAT_WPOST), .SWG(NAT_SWG), .XPOST(NAT_XPOST), .XSW($clog2(FIFO_DEPTH)),
                     .RHB(NAT_RHB), .WHB(NAT_WHB), .AW(AW), .SWW(QUANT_SWW)) u_mem (
    .clk, .rst(d_rst), .a_flush(d_flush),
    .a_rdy_x(da_rdy), .a_req_x(da_req), .a_we_x(da_we), .a_addr_x(da_addr),
        .a_wdata_x(da_wdata), .a_be_x(da_be), .a_rvalid(da_rvalid), .a_rdata(da_rdata),
    .sw_rdy(dsw_rdy), .sw_req(dsw_req), .sw_addr(dsw_addr), .sw_wdata(dsw_wdata),
    .sw_be(dsw_be),
    .b_rdy(db_rdy), .b_req(db_req), .b_tag(db_tag), .b_we(db_we), .b_wmask(db_wmask),
    .b_wdata(db_wdata), .b_addr(db_addr), .b_par(db_par), .b_rvalid(db_rvalid),
    .b_rtag(db_rtag), .b_rdata(db_rdata), .w_rdy(dw_rdy), .w_req(dw_req), .w_addr(dw_addr),
    .w_slot(dw_slot), .w_one(dw_one), .w_bv(dw_bv), .w_bd(dw_bd),
    .wr_idle(d_wr_idle), .rd_idle(d_rd_idle),
    .n_cvalid(n_cvalid[NCH-1:0]), .n_cready(n_cready[NCH-1:0]), .n_cwe(n_cwe[NCH-1:0]),
    .n_caddr(m_caddr[NCH-1:0]), .n_csrc(n_csrc[NCH-1:0]), .n_cslot(n_cslot[NCH-1:0]),
    .n_wvalid(n_wvalid[NCH-1:0]), .n_wready(n_wready[NCH-1:0]),
    .n_wdata(n_wdata[NCH-1:0]), .n_wmask(n_wmask[NCH-1:0]), .n_rvalid(n_rvalid[NCH-1:0]),
    .n_rdata(n_rdata[NCH-1:0]), .n_wdone(n_wdone[NCH-1:0]), .n_wfree(n_wfree[NCH-1:0]),
    .p_cntg(d_p_cntg), .p_cnt(d_p_cnt));
  for (genvar k = 0; k < NCH * S; k++) begin : g_ca
    assign n_caddr[k] = NCAW'(m_caddr[k]);
`ifndef SYNTHESIS
    if (DRAM_LOG2 < AW) begin : g_chk
      always @(posedge clk)
        if (!rst && n_cvalid[k] && (m_caddr[k] >> NCAW) != 0)
          $fatal(1, "otpu_board: channel %0d beat %h beyond its 2^%0d bytes", k, m_caddr[k],
                 CH_LOG2);
    end
`endif
  end

  // ---- slices 1 .. S-1, each on its NCH channels
  for (genvar s = 1; s < S; s++) begin : g_slice
    logic           a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_rvalid, b_rtag;
    logic [CL-1:0]  b_par;
    logic           pw_rdy, pw_req, pw_one;
    logic [WA-1:0]  pw_addr;
    logic [NCH-1:0] pw_bv;
    logic [P*8-1:0] pw_bd;
    logic [$clog2(FIFO_DEPTH)-1:0] pw_slot;
    logic [NCH-1:0][$clog2(FIFO_DEPTH)+1:0] p_cnt;
    logic           b_rdy, a_inval, sw_req, sw_rdy, m_wr_idle, m_rd_idle, m_rv;
    logic [WA-1:0]  a_addr, b_addr, sw_addr;
    logic [31:0]    a_wdata;
    logic [32*QUANT_SWW-1:0] sw_wdata;
    logic [AB*8-1:0] a_rdata;
    logic [3:0]     a_be;
    logic [4*QUANT_SWW-1:0] sw_be;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata, b_rdata;
    perf_t          pf;
    assign q_ret[s] = pf.sq.ret;
    logic [NCH-1:0] sp_we;                       // (as slice 0's)
    logic [NCH-1:0][1:0][((NAT_XQB > 0) ? (1 << NAT_XQB) : 1)-1:0] sp_q;
    logic [NCH-1:0][4:0][$clog2(FIFO_DEPTH)-1:0] sp_slot;
    logic [NCH-1:0][(D*8 >> NAT_XQB)-1:0] sp_d;
    logic d_rst, d_flush, da_rdy, da_req, da_we, da_rvalid, dsw_rdy, dsw_req, db_rdy, db_req;
    logic db_tag, db_we, db_rvalid, db_rtag, dw_rdy, dw_req, dw_one, d_wr_idle, d_rd_idle;
    logic [WA-1:0] da_addr, dsw_addr, db_addr, dw_addr;
    logic [31:0] da_wdata;
    logic [32*QUANT_SWW-1:0] dsw_wdata;
    logic [3:0] da_be;
    logic [4*QUANT_SWW-1:0] dsw_be;
    logic [AB*8-1:0] da_rdata;
    logic [D/4-1:0] db_wmask;
    logic [D*8-1:0] db_wdata, db_rdata;
    logic [CL-1:0] db_par;
    logic [$clog2(FIFO_DEPTH)-1:0] dw_slot;
    logic [NCH-1:0] dw_bv;
    logic [P*8-1:0] dw_bd;
    logic [NCH-1:0][$clog2(FIFO_DEPTH)+1:0] d_p_cnt, d_p_cntg;

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS),
                 .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB),
                 .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .MXU_QD(MXU_QD), .QUANT_R1(QUANT_R1), .QUANT_PW(QUANT_PW), .QUANT_SPLIT(QUANT_SPLIT), .QUANT_HL(QUANT_HL), .QUANT_SWW(QUANT_SWW), .QUANT_QSTB(QUANT_QSTB), .QUANT_QSTHL(QUANT_QSTHL), .QONCE(QONCE), .VPU_CL(VPU_CL),
                 .ULANES(ULANES), .PQ_WIN(PQ_WIN), .SEQ_WC(SEQ_WC), .HAS_DSTEP(DSTEP),
                 .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
                 .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
                 .TMEM_STYLE(TMEM_STYLE), .IMEM_STYLE(IMEM_STYLE), .ACT_UCOLS(ACT_UCOLS), .NCH(NCH), .CB(CB), .QB(QB),
                 .MXU_RF_STYLE(MXU_RF_STYLE), .MXU_FIFO_STYLE(MXU_FIFO_STYLE), .MXU_WASK(MXU_WASK),
                 .MXU_RF_ROWS(MXU_RF_ROWS), .MXU_BRAW(MXU_BRAW), .MXU_AASK(MXU_AASK),
                 .IMEM_WC_STYLE(IMEM_WC_STYLE),
                 .WG(WG), .SE_R(SE_R), .DMA_DEPTH(DMA_DEPTH), .DMA_LDQ(DMA_LDQ), .MXU_PUMP(MXU_PUMP), .VPU_FSUM(VPU_FSUM), .VPU_SIG(VPU_SIG), .VPU_MAC(VPU_MAC), .VPU_OUTER1(VPU_OUTER1), .VPU_CNS(VPU_CNS), .OUTER(OUTER), .XPOST(NAT_XPOST), .XQB(NAT_XQB),
                 .CHASH(CHASH), .DSTEP2(DSTEP2), .ST16(ST16), .ST16S(ST16S), .KT(KT), .MXU_SPL(MXU_SPL), .ICACHE(ICACHE), .IC_LINES(IC_LINES),
                 .IC_PF(IC_PF), .AW(AW)) u_slice (
      .clk, .clk_f, .sys_rst(x_srst[s]), .rst(x_rst[s]), .rinit(x_arg[s]), .ld_start(x_ld_start[s]),
      .ld_addr(x_ld_addr[s]), .ld_n(x_ld_n[s]),
      .ld_busy(q_ld_busy[s]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(m_wr_idle),
      .rd_idle(m_rd_idle),
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
      .pw_rdy, .pw_req, .pw_addr, .pw_bv, .pw_bd, .pw_slot, .pw_one,
      .p_clk, .p_we(sp_we), .p_q(sp_q), .p_slot(sp_slot), .p_d(sp_d), .p_cnt,
      .coll_req(sc_req[s]), .coll_cmd(sc_cmd[s]), .coll_ack(sc_ack[s]),
      .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
      .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
      .coll_wtop(sc_wtop[s]), .coll_gnt_local(sc_gl[s]), .coll_gnt(sc_gnt[s]),
      .halted(q_halted[s]), .error(q_error[s]), .wait_to(q_wait_to[s]), .a_inval,
      .icount(q_icount[s]), .pf, .dump(1'b0));

    otpu_xstg #(.N(NAT_XSTG), .NP(NAT_XSTG_P), .D(D), .NCH(NCH), .CB(CB), .AB(AB),
                .XSW($clog2(FIFO_DEPTH)), .XQB(NAT_XQB), .AW(AW), .SWW(QUANT_SWW)) u_xs (
      .clk, .p_clk, .s_rst(x_srst[s]), .s_flush({x_rst[s], x_ld_start[s], a_inval}),
      .s_a_rdy(a_rdy), .s_a_req(a_req), .s_a_we(a_we), .s_a_addr(a_addr), .s_a_wdata(a_wdata),
      .s_a_be(a_be), .s_a_rvalid(a_rvalid), .s_a_rdata(a_rdata),
      .s_sw_rdy(sw_rdy), .s_sw_req(sw_req), .s_sw_addr(sw_addr), .s_sw_wdata(sw_wdata),
      .s_sw_be(sw_be),
      .s_b_rdy(b_rdy), .s_b_req(b_req), .s_b_tag(b_tag), .s_b_we(b_we), .s_b_wmask(b_wmask),
      .s_b_wdata(b_wdata), .s_b_addr(b_addr), .s_b_par(b_par), .s_b_rvalid(b_rvalid),
      .s_b_rtag(b_rtag), .s_b_rdata(b_rdata),
      .s_w_rdy(pw_rdy), .s_w_req(pw_req), .s_w_addr(pw_addr), .s_w_slot(pw_slot),
      .s_w_one(pw_one), .s_w_bv(pw_bv), .s_w_bd(pw_bd),
      .s_wr_idle(m_wr_idle), .s_rd_idle(m_rd_idle), .s_p_cnt(p_cnt),
      .s_p_we(sp_we), .s_p_q(sp_q), .s_p_slot(sp_slot), .s_p_d(sp_d),
      .m_rst(d_rst), .m_flush(d_flush),
      .m_a_rdy(da_rdy), .m_a_req(da_req), .m_a_we(da_we), .m_a_addr(da_addr),
      .m_a_wdata(da_wdata), .m_a_be(da_be), .m_a_rvalid(da_rvalid), .m_a_rdata(da_rdata),
      .m_sw_rdy(dsw_rdy), .m_sw_req(dsw_req), .m_sw_addr(dsw_addr), .m_sw_wdata(dsw_wdata),
      .m_sw_be(dsw_be),
      .m_b_rdy(db_rdy), .m_b_req(db_req), .m_b_tag(db_tag), .m_b_we(db_we),
      .m_b_wmask(db_wmask), .m_b_wdata(db_wdata), .m_b_addr(db_addr), .m_b_par(db_par),
      .m_b_rvalid(db_rvalid), .m_b_rtag(db_rtag), .m_b_rdata(db_rdata),
      .m_w_rdy(dw_rdy), .m_w_req(dw_req), .m_w_addr(dw_addr), .m_w_slot(dw_slot),
      .m_w_one(dw_one), .m_w_bv(dw_bv), .m_w_bd(dw_bd),
      .m_wr_idle(d_wr_idle), .m_rd_idle(d_rd_idle), .m_p_cnt(d_p_cnt), .m_p_cntg(d_p_cntg),
      .c_p_we(p_we[NCH*s +: NCH]), .c_p_q(p_q[NCH*s +: NCH]), .c_p_slot(p_slot[NCH*s +: NCH]),
      .c_p_d(p_d[NCH*s +: NCH]), .c_p_cntg(p_cntg[NCH*s +: NCH]));
    otpu_native_dram #(.D(D), .CHASH(CHASH), .NCH(NCH), .CB(CB), .AB(AB), .RD(NAT_RD),
                       .RB_BLOCK(NAT_RBB[s]),
                       .AD(NAT_AD), .XD(NAT_XD), .WQD(NAT_WQD), .APF(NAT_APF), .SW_FILL(NAT_FILL),
                       .WPOST(NAT_WPOST), .SWG(NAT_SWG), .XPOST(NAT_XPOST), .XSW($clog2(FIFO_DEPTH)),
                       .RHB(NAT_RHB), .WHB(NAT_WHB), .AW(AW), .SWW(QUANT_SWW)) u_mem (
      .clk, .rst(d_rst), .a_flush(d_flush),
      .a_rdy_x(da_rdy), .a_req_x(da_req), .a_we_x(da_we), .a_addr_x(da_addr),
          .a_wdata_x(da_wdata), .a_be_x(da_be), .a_rvalid(da_rvalid), .a_rdata(da_rdata),
      .sw_rdy(dsw_rdy), .sw_req(dsw_req), .sw_addr(dsw_addr), .sw_wdata(dsw_wdata),
      .sw_be(dsw_be),
      .b_rdy(db_rdy), .b_req(db_req), .b_tag(db_tag), .b_we(db_we), .b_wmask(db_wmask),
      .b_wdata(db_wdata), .b_addr(db_addr), .b_par(db_par), .b_rvalid(db_rvalid),
      .b_rtag(db_rtag), .b_rdata(db_rdata), .w_rdy(dw_rdy), .w_req(dw_req), .w_addr(dw_addr),
      .w_slot(dw_slot), .w_one(dw_one), .w_bv(dw_bv), .w_bd(dw_bd),
      .wr_idle(d_wr_idle), .rd_idle(d_rd_idle),
      .n_cvalid(n_cvalid[NCH*s +: NCH]), .n_cready(n_cready[NCH*s +: NCH]),
      .n_cwe(n_cwe[NCH*s +: NCH]), .n_caddr(m_caddr[NCH*s +: NCH]),
      .n_csrc(n_csrc[NCH*s +: NCH]), .n_cslot(n_cslot[NCH*s +: NCH]),
      .n_wvalid(n_wvalid[NCH*s +: NCH]), .n_wready(n_wready[NCH*s +: NCH]),
      .n_wdata(n_wdata[NCH*s +: NCH]), .n_wmask(n_wmask[NCH*s +: NCH]),
      .n_rvalid(n_rvalid[NCH*s +: NCH]), .n_rdata(n_rdata[NCH*s +: NCH]),
      .n_wdone(n_wdone[NCH*s +: NCH]), .n_wfree(n_wfree[NCH*s +: NCH]),
      .p_cntg(d_p_cntg), .p_cnt(d_p_cnt));
    assign q_wr_idle[s] = m_wr_idle && !sw_req && !(b_req && b_we);     // as slice 0's
    assign q_rd_idle[s] = m_rd_idle && !a_req && !(b_req && !b_we) && !m_rv;
    always_ff @(posedge clk) m_rv <= !x_srst[s] && b_rvalid;
  end

  if (S == 1 && NC == 1) begin : g_x1
    // ---- one slice: wired
    assign x_rst[0] = core_rst;
    assign x_srst[0] = rst;
    assign x_ld_start[0] = ld_start;
    assign x_ld_addr[0] = ld_addr;
    assign x_ld_n[0] = ld_n;
    assign x_arg[0] = arg;
    assign sl_ld_busy = q_ld_busy;
    assign sl_halted = q_halted;
    assign sl_error = q_error;
    assign sl_wait_to = q_wait_to;
    assign sl_wr_idle = q_wr_idle;
    assign sl_rd_idle = q_rd_idle;
    assign sl_ret = q_ret;
    assign icount = q_icount;
    assign pf_c = pf;
    always_comb begin
      dram_rd = '0;
      dram_wr = '0;
      instr = '0;
      for (int c = 0; c < NCH * S; c++) begin
        dram_rd += CW'(n_rvalid[c]);
        dram_wr += CW'(n_wvalid[c] && n_wready[c]);
      end
      for (int k = 0; k < S; k++) instr += IW'(sl_ret[k]);
    end
    assign coll_req = sc_req;
    assign coll_cmds = sc_cmd;
    assign sc_ack = coll_ack;
    assign sc_ren = coll_ren;
    assign sc_raddr = coll_raddr;
    assign coll_rdata = sc_rdata;
    assign sc_wen[0] = coll_wen;
    assign sc_waddr[0] = coll_waddr;
    assign sc_wdata[0] = coll_wdata;
    assign sc_wtop = '1;
    assign coll_gnt = &sc_gl;           // every slice's TMEM grant
    assign sc_gnt = coll_gnt;
  end else begin : g_xs
    // ---- S slices, each in its own SLR: XRS register stages each way between a slice and the
    // control block / the collective unit (otpu_xr: u_*_tx on the sending side, u_*_rx on the
    // receiving one, bcu1525_floorplan.tcl places them; above 2 the others between them, g_*m),
    // so every signal crosses from a flip-flop into one. Every slice's control goes through
    // them, slice 0's too (the slices start together); what comes back is XRS cycles late, which
    // the control block's LOAD window (LDP) covers. The collective: otpu_coll_hub next to the
    // unit, an otpu_coll_port per slice. COLL_PIPE: CNR reads in flight, the round trip's (the
    // hub's read to the port XRS, its FIFO and flip-flops 2, the TMEM 1, dd 1, back XRS, the
    // hub's FIFO 1 and the drain 1: 2 XRS + 6 cycles) and CF writes a port, the credit's loop
    // (the hub's write stage 1, to the port XRS, its FIFO 1, the pop 1, back XRS, the count 1:
    // 2 XRS + 4), each with two cycles to spare, rounded up to a power of two (16 at XRS 2-4).
    localparam int L = LANES, CMB = $bits(cmd_t), PB = $bits(perf_t);
    localparam int CNR = (COLL_PIPE || NC > 1) ? 2 ** $clog2(2 * XRS + 8) : 1;
    localparam int CF = (COLL_PIPE || NC > 1) ? 2 ** $clog2(2 * XRS + 6) : 8;
    initial if (XRS < 2) $fatal(1, "otpu_board: XRS %0d (>= 2: a stage on each side)", XRS);
    // down, per slice: pulses (ack, read, write; cleared by reset), the collective's read and
    // write, the control
    localparam int DW = (NC > 1 ? 1 : 0) + 3 + L + 32 * L + L + 64 * L + 3 + 64 + 256;
    // up, per slice: levels / pulses (request, read data, pop; cleared by reset), the data, the
    // command, the status, the channels' beats this cycle
    localparam int UW = 3 + 32 * L + CMB + 6 + 2 + 32 + 2 * DCW;
    logic                   h_ack, h_wv, h_wt;
    logic [S-1:0]           h_rv, h_req, h_dv, h_pop;
    logic [L-1:0]           h_rm, h_wm;
    logic [L-1:0][31:0]     h_ra, h_wa, h_wd;
    logic [S-1:0][L-1:0][31:0] h_dd;
    cmd_t                   h_cmd [S];
    logic [S-1:0][DCW-1:0]  sl_drd, sl_dwr;

    logic [1:0] l_tv, l_tlast, l_rcred, l_tcred, l_rv, l_rhdr, l_rlast;
    logic [1:0] l_up, l_rerr, l_abin;
    logic [1:0][255:0] l_td, l_rd;
    logic l_abort;
    logic [7:0] h_status;
    logic h_xerr;
    logic [15:0] h_seq;
    if (NC > 1) begin : g_hstatus
      assign coll_status = h_status;
      assign coll_xerr = h_xerr;
      assign x2_seq = h_seq;
    end
    if (NC > 1) begin : g_link
      for (genvar p = 0; p < 2; p++) begin : g_l
        lk_in_t li;
        lk_out_t lo;
        assign li = {x2_s_tready[p], x2_m_tvalid[p], x2_m_thdr[p], x2_m_tlast[p],
                     x2_m_tdata[p], x2_up[p], x2_rx_err[p], x2_abort_in[p]};
        assign x2_s_tdata[p] = lo.s_tdata;
        assign x2_s_tvalid[p] = lo.s_tvalid;
        assign x2_s_tlast[p] = lo.s_tlast;
        assign x2_m_tready[p] = lo.m_tready;
        assign x2_abort[p] = lo.abort_req;
        (* keep_hierarchy = "yes" *) otpu_coll_lx #(.XRS(XRS)) u_lx (
          .clk, .rst(x2_rst), .ren(!coll_rst), .t_v(l_tv[p]), .t_last(l_tlast[p]), .t_d(l_td[p]),
          .r_cred(l_rcred[p]), .abort(l_abort), .card(x2_card), .t_cred(l_tcred[p]),
          .r_v(l_rv[p]), .r_hdr(l_rhdr[p]), .r_last(l_rlast[p]), .r_d(l_rd[p]),
          .up(l_up[p]), .r_err(l_rerr[p]), .ab_in(l_abin[p]), .li, .lo);
      end
    end else begin : g_no_link
      assign {l_tcred, l_rv, l_rhdr, l_rlast, l_rd, l_up, l_rerr, l_abin} = '0;
    end

    otpu_coll_hub #(.S(S), .LANES(L), .F(CF), .NR(CNR), .NC(NC)) u_chub (
      .clk, .rst(coll_rst),
      .u_req(coll_req), .u_cmds(coll_cmds), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop), .x_wt(h_wt), .lrst(x2_rst), .cid(x2_card), .ncards(x2_ncards),
      .wdog(coll_wd), .l_tv, .l_tlast, .l_td, .l_rcred, .l_abort, .l_tcred, .l_rv,
      .l_rhdr, .l_rlast, .l_rd, .l_up, .l_rerr, .l_abin,
      .xst(h_status), .xerr(h_xerr), .xseq(h_seq));

    for (genvar s = 0; s < S; s++) begin : g_x
      logic [DW-1:0] dn_d, dn_q;
      logic [UW-1:0] up_d, up_q;
      logic [XRS-2:0][DW-1:0] dn_m;     // after u_dn_tx and each stage between
      logic [XRS-2:0][UW-1:0] up_m;
      logic [255:0]  arg_d, arg_q;
      logic          p_dv, p_pop;
      logic [L-1:0][31:0] p_dd;
      logic [L-1:0]       p_rm, p_wm;
      logic [L-1:0][31:0] p_ra, p_wa, p_wd;
      logic          p_rv, p_wv, p_wt, rx_crst;
      logic [DCW-1:0] drd, dwr;
      for (genvar k = 0; k < 8; k++) begin : g_arg
        assign arg_d[32*k +: 32] = arg[k];
        assign x_arg[s][k] = arg_q[32*k +: 32];
      end
      if (NC > 1) begin : g_tag
        assign dn_d = {h_wt, arg_d, ld_n, ld_addr, ld_start, core_rst, rst, h_wd, h_wa, h_wm,
                       h_ra, h_rm, h_wv, h_rv[s], h_ack};
        assign p_wt = dn_q[DW-1];
      end else begin : g_no_tag
        assign dn_d = {arg_d, ld_n, ld_addr, ld_start, core_rst, rst, h_wd, h_wa, h_wm, h_ra, h_rm,
                       h_wv, h_rv[s], h_ack};
        assign p_wt = 1'b1;
      end
      otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst(coll_rst), .d(dn_d), .q(dn_m[0]));
      for (genvar i = 1; i < XRS - 1; i++) begin : g_dm
        otpu_xr #(.W(DW), .RW(3)) u_dn (.clk, .rst(coll_rst), .d(dn_m[i-1]), .q(dn_m[i]));
      end
      otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(rx_crst), .d(dn_m[XRS-2]), .q(dn_q));
      assign {arg_q, x_ld_n[s], x_ld_addr[s], x_ld_start[s], rx_crst, x_srst[s], p_wd, p_wa,
              p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q[DW-(NC > 1 ? 1 : 0)-1:0];
      assign x_rst[s] = rx_crst;

      otpu_coll_port #(.LANES(L), .F(CF), .NR(CNR), .WT(NC > 1)) u_cport (
        .clk, .rst(x_rst[s]), .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
        .wd(p_wd), .wt(p_wt), .coll_wtop(sc_wtop[s]), .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
        .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
        .gnt_local(sc_gl[s]), .gnt(sc_gnt[s]));

      // the slice's channels' beats this cycle (DRAM_RD / DRAM_WR), counted on its side
      always_comb begin
        drd = '0;
        dwr = '0;
        for (int c = NCH * s; c < NCH * s + NCH; c++) begin
          drd += DCW'(n_rvalid[c]);
          dwr += DCW'(n_wvalid[c] && n_wready[c]);
        end
      end
      assign up_d = {drd, dwr, q_icount[s], q_ret[s], q_ld_busy[s], q_halted[s], q_error[s],
                     q_wait_to[s], q_wr_idle[s], q_rd_idle[s], sc_cmd[s], p_dd, p_pop, p_dv,
                     sc_req[s]};
      otpu_xr #(.W(UW), .RW(3)) u_up_tx (.clk, .rst(x_rst[s]), .d(up_d), .q(up_m[0]));
      for (genvar i = 1; i < XRS - 1; i++) begin : g_um
        otpu_xr #(.W(UW), .RW(3)) u_up (.clk, .rst(x_rst[s]), .d(up_m[i-1]), .q(up_m[i]));
      end
      otpu_xr #(.W(UW), .RW(3)) u_up_rx (.clk, .rst(coll_rst), .d(up_m[XRS-2]), .q(up_q));
      assign {sl_drd[s], sl_dwr[s], icount[s], sl_ret[s], sl_ld_busy[s], sl_halted[s],
              sl_error[s], sl_wait_to[s], sl_wr_idle[s], sl_rd_idle[s], h_cmd[s], h_dd[s],
              h_pop[s], h_dv[s], h_req[s]} = up_q;

      if (s == 0) begin : g_pf          // slice 0's activity (the counters, the trace)
        logic [XRS-2:0][PB-1:0] pf_m;
        otpu_xr #(.W(PB)) u_pf_tx (.clk, .rst(1'b0), .d(pf), .q(pf_m[0]));
        for (genvar i = 1; i < XRS - 1; i++) begin : g_pm
          otpu_xr #(.W(PB)) u_pf (.clk, .rst(1'b0), .d(pf_m[i-1]), .q(pf_m[i]));
        end
        otpu_xr #(.W(PB)) u_pf_rx (.clk, .rst(1'b0), .d(pf_m[XRS-2]), .q(pf_c));
      end
    end

    always_comb begin
      dram_rd = '0;
      dram_wr = '0;
      instr = '0;
      for (int k = 0; k < S; k++) begin
        dram_rd += CW'(sl_drd[k]);
        dram_wr += CW'(sl_dwr[k]);
        instr += IW'(sl_ret[k]);
      end
    end
  end

  // ---- LEDs: heartbeat, running, halted/error
  logic [26:0] hb;
  always_ff @(posedge clk) hb <= hb + 1;
  assign led = {halted && !error, run && !halted, hb[26]};
endmodule
