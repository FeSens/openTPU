// The accelerator as built for the PCIe boards: S slices (one on the YPCB-00338, two on the
// BCU1525), each with its DRAM adapter otpu_native_dram onto two memory channels' native ports
// n_* (one command per 64-byte beat, for otpu_mem_ch in front of LiteDRAM's native port: slice
// s on channels 2s and 2s + 1), the collective unit, and the host control registers (AXI4-Lite
// slave). Everything runs on the core clock; the board's top does the clock crossings to the
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
// the signals are wired as before (g_x1: the same logic, tools/synth/board_equiv.sh).
module otpu_board #(
  parameter int S          = 1,         // slices, each on two memory channels
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
  parameter int RPB        = (TMEM_IMPL == 1) ? 1 : 8 * LANES,  // every read lane (TMEM is
                                          // replicated per read port): the slice's shallow
                                          // write-mask arbiter
  parameter int WPB        = 1,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,  // VPU lanes with exp2/recip/rsqrt
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int CORE_KHZ    = 100000,  // the core clock (CORE_KHZ register)
  parameter logic [31:0] BUILD_ID = 32'h0,
  parameter int DDR_MTS     = 0,       // DDR3 data rate (DDR_MTS register; 0: not given)
  parameter int TRACE_DEPTH = 16384,   // trace records (a power of two; 0: no trace buffer)
  parameter int TRACE_QD    = 32,      // trace capture queue (cycles with events)
  parameter int PQ_WIN      = 1024,    // cycles per P/Q counter window
  parameter bit HAS_I2C     = 1'b1,    // the I2C pins are wired (CAPS bit2)
  parameter bit CHASH       = 1'b1,    // hashed channel interleave (otpu_native_dram; CAPS bit7)
  parameter int NRD         = 128,     // port B read beats in flight per channel (otpu_native_dram RD)
  parameter bit DSTEP       = 1'b1,    // the DMA's DSTEP datapath (CAPS bit6; 0 leaves it out)
  parameter bit HOSTCAL     = 1'b0,    // the host calibrates the DDR3 controllers (CAPS bit27: LiteDRAM)
  parameter bit COLL_GATHER = 1'b1,    // 0 (S = 1 only): the collective unit runs BAR alone, no
                                       // GATHER (otpu_coll_bar; CAPS2 bit0 = 0, docs/infra_hunt.md 5)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int PLATFORM    = 0,       // the target's platform id (CFG_PLATFORM; opentpu/target.py)
  parameter logic [31:0] ARCH_HASH = 32'h0, // the target's arch_hash (CFG_HASH)
  parameter bit DUAL        = 1'b0,    // S = 1: the slice on four channels (otpu_native_dram2), its
                                       // port B in chunk pairs (int8 MM PAIR; CFG_DRAM channels 4)
  parameter int NCPS        = DUAL ? 4 : 2,  // channels a slice
  // log2 bytes a slice addresses on each channel (CFG_DRAM CH_LOG2): 31 at DAW 32 (otpu_pkg; the
  // 4 GiB slice's map, DUAL too), else the slice's 2^DAW bytes over its NCPS channels; and the
  // channels' beat address bits (n_caddr), 25 up to 2 GiB a channel. A board whose channels are
  // smaller keeps BAW (their high beat bits stay 0) and the host keeps to DRAM_BYTES
  parameter int CH_LOG2     = (otpu_pkg::DAW > 32) ? otpu_pkg::DAW - $clog2(NCPS) : 31,
  parameter int BAW         = (CH_LOG2 - 6 > 25) ? CH_LOG2 - 6 : 25
) (
  input  logic         clk,
  input  logic         clk_f,          // TMEM_PMP 1: 2 x clk, rising with it (the MMCM's); else unused
  input  logic         rst,            // synchronous, active high
  input  logic         ctl_mrst,       // the control master's reset (the SmartConnect's, in clk:
                                       //   PCIe hot reset or link down too): otpu_axil_iso
  input  logic [NCPS*S-1:0] calib,        // memory controllers calibrated (any clock domain)
  input  logic [NCPS*S-1:0] ded,          // a channel's ECC counted an uncorrectable word (any clock
                                       //   domain; STATUS ECC_DED)
  input  logic [11:0]  temp,           // XADC die-temperature code (any clock domain, slow)
  output logic [2:0]   led,
  // ---- I2C pins (otpu_fpga_top_ld's IOBUFs): 1 drives the line low; the levels (any clock)
  output logic [3:0]   i2c_lo,         // SCL0 SDA0 (LM73 bus), SCL1 SDA1 (PCIe SMBus)
  input  logic [4:0]   i2c_pin,        // the same four lines, then ALERT0 (LM73, active low)
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
  // ---- memory channels 0 .. 2S-1: native masters (otpu_native_dram; slice s on 2s, 2s + 1; DUAL:
  // channels 0 .. 3, otpu_native_dram2)
  output logic [NCPS*S-1:0]        n_cvalid,
  input  logic [NCPS*S-1:0]        n_cready,
  output logic [NCPS*S-1:0]        n_cwe,
  output logic [NCPS*S-1:0][BAW-1:0] n_caddr, // the 64-byte beat in the channel (address bits BAW+5:6)
  output logic [NCPS*S-1:0]        n_wvalid,  // write data: a beat per write command, in their order
  input  logic [NCPS*S-1:0]        n_wready,
  output logic [NCPS*S-1:0][511:0] n_wdata,
  output logic [NCPS*S-1:0][63:0]  n_wmask,   // 1 = write the byte
  input  logic [NCPS*S-1:0]        n_rvalid,  // read data in read-command order, no backpressure
  input  logic [NCPS*S-1:0][511:0] n_rdata,
  input  logic [NCPS*S-1:0][15:0]  n_wdone,   // write beats the controller has taken (mod 2^16)
  input  logic [NCPS*S-1:0]        n_err      // a channel's controller broke its port contract (sticky)
);
  import otpu_pkg::*;
  localparam int XR = 2;              // register stages each way across an SLR (S > 1)

  // ---- the reset tree: rst through a register per destination, each next to its loads (250
  // MHz, FK35: the block design's reset -> the control block's counters and the trace's pointers
  // in one net, 1 level, 95% route, -0.65 ns; otpu_bcu_sys registers it first): r_ctl the control
  // block and its AXI-Lite isolator (one reset: the isolator hands the control block no access
  // while it is in reset), r_trc the trace, r_sh the rest of the shell and the slices' system
  // reset. keep: not merged back into one register. Everything here leaves reset a cycle after rst.
  (* keep = "true", shreg_extract = "no", max_fanout = 128 *) logic r_ctl;
  (* keep = "true", shreg_extract = "no", max_fanout = 64 *) logic r_trc, r_sh;
  always_ff @(posedge clk) begin
    r_ctl <= rst;
    r_trc <= rst;
    r_sh  <= rst;
  end

  // ---- calibration and uncorrectable-ECC flags from the memory controllers' clock domains
  // (registers there; levels, each bit on its own)
  (* ASYNC_REG = "TRUE" *) logic [NCPS*S-1:0] cal_s1, cal_s2, ded_s1, ded_s2;
  // CALIB0 / CALIB1: every slice's first / second channel
  logic [1:0] cal_all;
  always_comb begin
    cal_all = 2'b11;
    for (int c = 0; c < NCPS * S; c++) cal_all[c % 2] &= cal_s2[c];   // (DUAL: 0, 2 and 1, 3)
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
    if (r_sh) begin
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
  logic [S-1:0] x_frst, x_mrst;     // x_rst and x_srst for the adapter (its flush, its reset; at
                                    //   S > 1 copies of their own, the same cycles)
  logic [S-1:0][31:0] x_ld_addr, x_ld_n;
  logic [31:0] x_arg [S][8];
  // the free-running counters' increments: every channel's beats, every slice's instructions
  localparam int CW = $clog2(S * NCPS + 1), IW = $clog2(2 * S + 1);
  logic [CW-1:0] dram_rd, dram_wr;
  logic [IW-1:0] instr;
  logic [31:0] arg [8];               // the run's arguments (ARG0..7: R8..R15 at the start)
  logic a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag, b_rdy;
  logic [AWW-1:0] a_addr, b_addr, sw_addr;      // word addresses (otpu_pkg AWW)
  logic [31:0] a_wdata;
  logic [255:0] a_rdata;
  logic [3:0]  a_be, sw_be;
  logic        sw_req, sw_rdy;
  logic        m_wr_idle, m_rd_idle;   // slice 0's adapter's wr_idle and rd_idle (q_wr_idle[0],
  logic        m_rv;                   //   q_rd_idle[0]: with its SW and port registers, below);
                                       //   m_rv: a b_rvalid the cycle before
  logic [31:0] sw_wdata;
  logic [D/4-1:0] b_wmask;
  logic [D*8-1:0] b_wdata, b_rdata;
  logic           b_two;               // DUAL: slice 0's port B pairs
  logic [D*8-1:0] b_rdata2;
  initial if (DUAL && S != 1) $fatal(1, "otpu_board: DUAL is one slice's (S = 1)");

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
    .clk, .m_rst(ctl_mrst), .s_rst(r_ctl),
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

  otpu_ctrl #(.D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .LANES(LANES), .CORE_KHZ(CORE_KHZ), .BUILD_ID(BUILD_ID),
              .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .PQ_WIN(PQ_WIN), .HAS_TEMP(1'b1),
              .HAS_I2C(HAS_I2C), .CHASH(CHASH), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL),
              .COLL_GATHER(COLL_GATHER),
              .LDP(S > 1 ? 2 + 2 * XR : 2),  // LOAD to a slice's busy and back
              // each slice on its NCPS channels (otpu_native_dram / 2): 2^DAW bytes (otpu_pkg),
              // 2^CH_LOG2 a channel
              .S(S), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .ACT_BLOCKS(ACT_BLOCKS),
              .DRAM_LOG2(DAW), .NCH(NCPS), .CH_LOG2(CH_LOG2), .MXU_IMPL(MXU_IMPL), .TMEM_IMPL(TMEM_IMPL),
              .VPU_CL(VPU_CL), .ULANES(ULANES), .SEQ_WC(SEQ_WC), .PLATFORM(PLATFORM),
              .ARCH_HASH(ARCH_HASH)) u_ctrl (
    .clk, .rst(r_ctl),
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
    .i2c_lo, .i2c_in(i2c_s2));

  // ---- hardware trace
  if (TRACE_DEPTH != 0) begin : g_trace
    otpu_trace #(.DEPTH(TRACE_DEPTH), .QD(TRACE_QD), .WIN(WIN)) u_trace (
      .clk, .rst(r_trc), .pf(pf_c), .en(tr_en && run), .stop(tr_stop), .clear(tr_clear), .raddr(tr_addr),
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
    go_ok <= !r_sh && ((run && go_ok) || (rd_idle && wr_idle));
    core_rst <= r_sh || !run || !go_ok;
  end

  // the run's status from every slice
  assign ld_busy = |sl_ld_busy;
  assign halted  = &sl_halted;
  logic  coll_gerr;                    // a GATHER on the BAR-only collective unit (COLL_GATHER 0)
  assign error   = |sl_error || coll_gerr;
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
  logic [S-1:0]                  sc_req, sc_ack, sc_gl, sc_gnt;
  cmd_t                          sc_cmd [S];
  logic [S-1:0][LANES-1:0]       sc_ren, sc_wen;
  logic [S-1:0][LANES-1:0][31:0] sc_raddr, sc_rdata, sc_waddr, sc_wdata;

  otpu_slice #(.SID(0), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
               .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS), .FIFO_DEPTH(FIFO_DEPTH),
               .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB), .MXU_IMPL(MXU_IMPL),
               .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL), .ULANES(ULANES), .PQ_WIN(PQ_WIN),
               .SEQ_WC(SEQ_WC), .HAS_DSTEP(DSTEP), .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
               .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
               .DUAL(DUAL)) u_slice (
    .clk, .clk_f, .sys_rst(x_srst[0]), .rst(x_rst[0]), .rinit(x_arg[0]), .ld_start(x_ld_start[0]),
    .ld_addr(x_ld_addr[0]), .ld_n(x_ld_n[0]),
    .ld_busy(q_ld_busy[0]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(m_wr_idle),
    .rd_idle(m_rd_idle),
    .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
    .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
    .b_two, .b_rdata2,
    .coll_req(sc_req[0]), .coll_cmd(sc_cmd[0]), .coll_ack(sc_ack[0]),
    .coll_ren(sc_ren[0]), .coll_raddr(sc_raddr[0]), .coll_rdata(sc_rdata[0]),
    .coll_wen(sc_wen[0]), .coll_waddr(sc_waddr[0]), .coll_wdata(sc_wdata[0]),
    .coll_gnt_local(sc_gl[0]), .coll_gnt(sc_gnt[0]),
    .halted(q_halted[0]), .error(q_error[0]), .wait_to(q_wait_to[0]), .a_inval,
    .icount(q_icount[0]), .pf, .dump(1'b0));

  // the collective's reset through a register of its own, next to it: it leaves reset a cycle
  // after the slice, idle either way (133.33 MHz, 110ec6d: core_rst -> u_coll's state, 0 levels,
  // 97% route, +0.162 ns); a request can only come many cycles after reset (checked). keep: the
  // crossings' u_dn_tx register core_rst too, and merged with theirs it drove them from the unit
  // (250 MHz, FK35: coll_rst -> g_x[].u_dn_rx, 0 levels, 98% route, -0.38 ns)
  (* keep = "true" *) logic coll_rst;
  always_ff @(posedge clk) coll_rst <= core_rst;
`ifndef SYNTHESIS
  always @(posedge clk)
    if (coll_rst && |coll_req) $fatal(1, "otpu_board: a collective request in reset");
`endif
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_board: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER) begin : g_coll
    // RQ: its reads from its own request register on the slice's port at S = 1; at S > 1 each
    // slice's otpu_coll_port presents them a cycle later (TMEM's input stage). UG at S > 1: the
    // hub's crossings take the addresses and data from the unit's registers as they are, valid
    // with their enables (otpu_coll)
    otpu_coll #(.S(S), .LANES(LANES), .RQ(S == 1), .UG(S > 1)) u_coll (
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
  if (DUAL) begin : g_mem2
    otpu_native_dram2 #(.D(D), .CHASH(CHASH), .RD(NRD), .AW(AWW), .BAW(BAW)) u_mem (
      .clk, .rst(x_mrst[0]), .a_flush(x_frst[0] || x_ld_start[0] || a_inval),
      .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
      .a_be_x(a_be), .a_rvalid, .a_rdata,
      .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_two, .b_rvalid,
      .b_rtag, .b_rdata, .b_rdata2, .wr_idle(m_wr_idle), .rd_idle(m_rd_idle),
      .n_cvalid(n_cvalid[3:0]), .n_cready(n_cready[3:0]), .n_cwe(n_cwe[3:0]),
      .n_caddr(n_caddr[3:0]), .n_wvalid(n_wvalid[3:0]), .n_wready(n_wready[3:0]),
      .n_wdata(n_wdata[3:0]), .n_wmask(n_wmask[3:0]), .n_rvalid(n_rvalid[3:0]),
      .n_rdata(n_rdata[3:0]), .n_wdone(n_wdone[3:0]));
  end else begin : g_mem
    assign b_rdata2 = '0;
    otpu_native_dram #(.D(D), .CHASH(CHASH), .RD(NRD), .AW(AWW), .BAW(BAW)) u_mem (
      .clk, .rst(x_mrst[0]), .a_flush(x_frst[0] || x_ld_start[0] || a_inval),
      .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
          .a_be_x(a_be), .a_rvalid, .a_rdata,
      .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
      .b_rdata, .b_ravail(), .b_rrdy(1'b1), .a_ravail(), .a_rrdy(1'b1),
      .wr_idle(m_wr_idle), .rd_idle(m_rd_idle),
      .n_cvalid(n_cvalid[1:0]), .n_cready(n_cready[1:0]), .n_cwe(n_cwe[1:0]),
      .n_caddr(n_caddr[1:0]), .n_wvalid(n_wvalid[1:0]), .n_wready(n_wready[1:0]),
      .n_wdata(n_wdata[1:0]), .n_wmask(n_wmask[1:0]), .n_rvalid(n_rvalid[1:0]),
      .n_rdata(n_rdata[1:0]), .n_wdone(n_wdone[1:0]));
  end

  // ---- slices 1 .. S-1, each on its two channels
  for (genvar s = 1; s < S; s++) begin : g_slice
    logic           a_req, a_we, a_rvalid, a_rdy, b_req, b_tag, b_we, b_par, b_rvalid, b_rtag;
    logic           b_rdy, a_inval, sw_req, sw_rdy, m_wr_idle, m_rd_idle, m_rv;
    logic [AWW-1:0] a_addr, b_addr, sw_addr;
    logic [31:0]    a_wdata, sw_wdata;
    logic [255:0]   a_rdata;
    logic [3:0]     a_be, sw_be;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata, b_rdata;
    perf_t          pf;
    assign q_ret[s] = pf.sq.ret;

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS),
                 .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB),
                 .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL),
                 .ULANES(ULANES), .PQ_WIN(PQ_WIN), .SEQ_WC(SEQ_WC), .HAS_DSTEP(DSTEP),
                 .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP),
                 .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM)) u_slice (
      .clk, .clk_f, .sys_rst(x_srst[s]), .rst(x_rst[s]), .rinit(x_arg[s]), .ld_start(x_ld_start[s]),
      .ld_addr(x_ld_addr[s]), .ld_n(x_ld_n[s]),
      .ld_busy(q_ld_busy[s]), .a_rdy, .b_rdy, .sw_rdy, .wr_idle(m_wr_idle),
      .rd_idle(m_rd_idle),
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
      .b_two(), .b_rdata2('0),
      .coll_req(sc_req[s]), .coll_cmd(sc_cmd[s]), .coll_ack(sc_ack[s]),
      .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
      .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
      .coll_gnt_local(sc_gl[s]), .coll_gnt(sc_gnt[s]),
      .halted(q_halted[s]), .error(q_error[s]), .wait_to(q_wait_to[s]), .a_inval,
      .icount(q_icount[s]), .pf, .dump(1'b0));

    otpu_native_dram #(.D(D), .CHASH(CHASH), .RD(NRD), .AW(AWW), .BAW(BAW)) u_mem (
      .clk, .rst(x_mrst[s]), .a_flush(x_frst[s] || x_ld_start[s] || a_inval),
      .a_rdy_x(a_rdy), .a_req_x(a_req), .a_we_x(a_we), .a_addr_x(a_addr), .a_wdata_x(a_wdata),
          .a_be_x(a_be), .a_rvalid, .a_rdata,
      .sw_rdy, .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_rdy, .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag,
      .b_rdata, .b_ravail(), .b_rrdy(1'b1), .a_ravail(), .a_rrdy(1'b1),
      .wr_idle(m_wr_idle), .rd_idle(m_rd_idle),
      .n_cvalid(n_cvalid[2*s +: 2]), .n_cready(n_cready[2*s +: 2]), .n_cwe(n_cwe[2*s +: 2]),
      .n_caddr(n_caddr[2*s +: 2]), .n_wvalid(n_wvalid[2*s +: 2]),
      .n_wready(n_wready[2*s +: 2]), .n_wdata(n_wdata[2*s +: 2]), .n_wmask(n_wmask[2*s +: 2]),
      .n_rvalid(n_rvalid[2*s +: 2]), .n_rdata(n_rdata[2*s +: 2]), .n_wdone(n_wdone[2*s +: 2]));
    assign q_wr_idle[s] = m_wr_idle && !sw_req && !(b_req && b_we);     // as slice 0's
    assign q_rd_idle[s] = m_rd_idle && !a_req && !(b_req && !b_we) && !m_rv;
    always_ff @(posedge clk) m_rv <= !x_srst[s] && b_rvalid;
  end

  if (S == 1) begin : g_x1
    // ---- one slice: wired
    assign x_rst[0] = core_rst;
    assign x_srst[0] = r_sh;
    assign x_frst[0] = core_rst;
    assign x_mrst[0] = r_sh;
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
      for (int c = 0; c < NCPS * S; c++) begin
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
    assign coll_gnt = &sc_gl;           // every slice's TMEM grant
    assign sc_gnt = coll_gnt;
  end else begin : g_xs
    // ---- S slices, each in its own SLR: two register stages each way between a slice and the
    // control block / the collective unit (otpu_xr: u_*_tx on the sending side, u_*_rx on the
    // receiving one; bcu1525_floorplan.tcl places them), so every signal crosses from a flip-flop
    // into one. Every slice's control goes through them, slice 0's too (the slices start
    // together); what comes back is XR cycles late, which the control block's LOAD window
    // (LDP) covers. The collective: otpu_coll_hub next to the unit, an otpu_coll_port per slice.
    localparam int L = LANES, CB = $bits(cmd_t), PB = $bits(perf_t);
    // down, per slice: pulses (ack, read, write; cleared by reset), the collective's read and
    // write, the control
    localparam int DW = 3 + L + 32 * L + L + 64 * L + 3 + 64 + 256;
    // up, per slice: levels / pulses (request, read data, pop; cleared by reset), the data, the
    // command, the status, the channels' beats this cycle
    localparam int UW = 3 + 32 * L + CB + 6 + 2 + 32 + 4;
    logic                   h_ack, h_wv;
    logic [S-1:0]           h_rv, h_req, h_dv, h_pop;
    logic [L-1:0]           h_rm, h_wm;
    logic [L-1:0][31:0]     h_ra, h_wa, h_wd;
    logic [S-1:0][L-1:0][31:0] h_dd;
    cmd_t                   h_cmd [S];
    logic [S-1:0][1:0]      sl_drd, sl_dwr;

    otpu_coll_hub #(.S(S), .LANES(L)) u_chub (
      .clk, .rst(coll_rst),
      .u_req(coll_req), .u_cmds(coll_cmds), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop));

    for (genvar s = 0; s < S; s++) begin : g_x
      logic [DW-1:0] dn_d, dn_m, dn_q;
      logic [UW-1:0] up_d, up_m, up_q;
      logic [255:0]  arg_d, arg_q;
      logic          p_dv, p_pop;
      logic [L-1:0][31:0] p_dd;
      logic [L-1:0]       p_rm, p_wm;
      logic [L-1:0][31:0] p_ra, p_wa, p_wd;
      logic          p_rv, p_wv, rx_crst, rx_srst;
      logic [1:0]    drd, dwr;
      for (genvar k = 0; k < 8; k++) begin : g_arg
        assign arg_d[32*k +: 32] = arg[k];
        assign x_arg[s][k] = arg_q[32*k +: 32];
      end
      assign dn_d = {arg_d, ld_n, ld_addr, ld_start, core_rst, r_sh, h_wd, h_wa, h_wm, h_ra, h_rm,
                     h_wv, h_rv[s], h_ack};
      otpu_xr #(.W(DW), .RW(3)) u_dn_tx (.clk, .rst(coll_rst), .d(dn_d), .q(dn_m));
      otpu_xr #(.W(DW), .RW(3)) u_dn_rx (.clk, .rst(rx_crst), .d(dn_m), .q(dn_q));
      assign {arg_q, x_ld_n[s], x_ld_addr[s], x_ld_start[s], rx_crst, rx_srst, p_wd, p_wa,
              p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q;
      // the slice's resets, a register per destination after the crossing, next to its loads
      // (250 MHz, FK35: u_dn_rx's reset bits -> the slice's activity registers, its adapter's
      // reply-tag RAM, 0-1 levels, 97% route, -0.59 ns): the core reset to the slice (x_rst), to its
      // adapter's flush (x_frst), to the collective port and the up crossing; the system reset to
      // the slice (x_srst) and to its adapter (x_mrst). They reach the slice a cycle after the rest
      // of the crossing (LOAD, the arguments): LOAD comes only with RUN 0, the core reset held, and
      // the system reset only at power-up; a stopped run's slice stops a cycle later (go_ok waits
      // for its adapter as before). keep: equivalent flip-flops (and the other slice's), kept apart.
      (* keep = "true", shreg_extract = "no", max_fanout = 64 *) logic [2:0] cr_q;
      (* keep = "true", shreg_extract = "no", max_fanout = 64 *) logic [1:0] sr_q;
      always_ff @(posedge clk) begin
        cr_q <= {3{rx_crst}};
        sr_q <= {2{rx_srst}};
      end
      assign x_rst[s]  = cr_q[0];
      assign x_frst[s] = cr_q[1];
      assign x_srst[s] = sr_q[0];
      assign x_mrst[s] = sr_q[1];

      otpu_coll_port #(.LANES(L)) u_cport (
        .clk, .rst(cr_q[2]), .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
        .wd(p_wd), .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
        .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
        .gnt_local(sc_gl[s]), .gnt(sc_gnt[s]));

      // the slice's channels' beats this cycle (DRAM_RD / DRAM_WR), counted on its side
      always_comb begin
        drd = '0;
        dwr = '0;
        for (int c = 2 * s; c < 2 * s + 2; c++) begin
          drd += 2'(n_rvalid[c]);
          dwr += 2'(n_wvalid[c] && n_wready[c]);
        end
      end
      assign up_d = {drd, dwr, q_icount[s], q_ret[s], q_ld_busy[s], q_halted[s], q_error[s],
                     q_wait_to[s], q_wr_idle[s], q_rd_idle[s], sc_cmd[s], p_dd, p_pop, p_dv,
                     sc_req[s]};
      otpu_xr #(.W(UW), .RW(3)) u_up_tx (.clk, .rst(cr_q[2]), .d(up_d), .q(up_m));
      otpu_xr #(.W(UW), .RW(3)) u_up_rx (.clk, .rst(coll_rst), .d(up_m), .q(up_q));
      assign {sl_drd[s], sl_dwr[s], icount[s], sl_ret[s], sl_ld_busy[s], sl_halted[s],
              sl_error[s], sl_wait_to[s], sl_wr_idle[s], sl_rd_idle[s], h_cmd[s], h_dd[s],
              h_pop[s], h_dv[s], h_req[s]} = up_q;

      if (s == 0) begin : g_pf          // slice 0's activity (the counters, the trace)
        logic [PB-1:0] pf_m;
        otpu_xr #(.W(PB)) u_pf_tx (.clk, .rst(1'b0), .d(pf), .q(pf_m));
        otpu_xr #(.W(PB)) u_pf_rx (.clk, .rst(1'b0), .d(pf_m), .q(pf_c));
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

// One register stage of an SLR crossing (otpu_board at S > 1: a pair per direction and slice, one
// stage on each side). The low RW bits (valid pulses, request levels) clear with rst; the others
// are data. keep: every slice's stages hold the same control bits (the arguments, LOAD, the
// resets, the collective's writes), and merged into one set they sat between the slices and drove
// both (250 MHz, FK35: g_x[].u_dn_rx -> the slices' resets, -0.59 ns), so each slice keeps its own.
// shreg_extract: two plain stages behind plain sources stay flip-flops (no SRL).
module otpu_xr #(
  parameter int W  = 1,
  parameter int RW = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  // r_reg: the BCU1525 floorplan's USER_SLL_REG cells
  (* keep = "true", shreg_extract = "no", max_fanout = 256 *) logic [W-1:0] r;
  assign q = r;
  always_ff @(posedge clk) begin
    r <= d;
    if (rst) for (int i = 0; i < RW; i++) r[i] <= 1'b0;
  end
endmodule
