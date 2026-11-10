// openTPU: S slices, each with its own DRAM, plus the collective unit (simulation top).
// AXI = 0: the behavioural fixed-latency DRAM. AXI = 1: the board's memory path -- the native
// adapter (otpu_native_dram) in front of an NCH-channel native memory model with random stalls
// and latency (sim/verilator/otpu_native_mem.sv: CB-byte beats, its rate and latency the
// target's, opentpu.target.mem_model; requires D = 128); the dump prints the adapter's counters.
// AXI = 2: the same adapter in front of the card's channels as they are (two of 64-byte beats;
// sim/verilator/otpu_ldc_mem.sv: per channel the bridge otpu_mem_ch and LiteDRAM's own
// controller, generated with the production core's settings, in the controller clock).
module otpu_top
  import otpu_pkg::*;
#(
  parameter int NC         = 1,         // cards; > 1 always uses the pipelined hub
  parameter int S          = 1,
  parameter int D          = 32,
  parameter int MCOLS      = 8,
  parameter int ACT_BLOCKS = 64,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 16,
  parameter int DRAM_WORDS = 1 << 18,
  parameter int DRAM_LAT   = 8,
  parameter int FIFO_DEPTH = 128,
  parameter int LANES      = 8,
  parameter int WIN        = 32,
  parameter int RPB        = 4,
  parameter int WPB        = 2,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,       // MXU epilogue lanes (0: MCOLS / 2; otpu_mxu E)
  parameter int MXU_QD     = 2,       // MXU commands queued (otpu_mxu QD; 2..7)
  parameter int QUANT_XD   = 1024,    // quantizer row buffer chunks (0: two passes over TMEM; otpu_quant XD)
  parameter bit QUANT_R1   = 1'b0,    // RSCALE's row factor read once a row (otpu_quant R1; timing only)
  parameter int QUANT_PW   = 1,       // a buffered QACT pass 1's chunks a cycle (otpu_quant PW; timing only)
  parameter bit QUANT_SPLIT = 1'b0,   // QACT / QST flag LO: the split's lo part (otpu_quant SPLIT; Config.SPLITQ)
  parameter bit QUANT_HL    = 1'b0,   // QACT flag HL: q and lo from one read (otpu_quant HL; Config.QHL)
  parameter int QUANT_SWW  = 1,       // the SW port's words (otpu_quant SWW; timing only)
  parameter bit QUANT_QSTB  = 1'b0,   // QST rows through the row buffer (otpu_quant QSTB; timing only)
  parameter bit QUANT_QSTHL = 1'b0,   // QST HL: a GRP ROW QST's hi and lo from one read (otpu_quant QSTHL; Config.QSTHL)
  parameter bit QONCE      = 1'b0,    // QACT flag ST and LDA (otpu_slice; Config.QONCE)
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1,
  parameter int ULANES     = LANES,   // TMEM lanes of the MXU and the quantizer
  parameter int AXI        = 0,       // memory path: see the top
  parameter int TMEM_IMPL  = 0,       // TMEM: 0 a copy per read port, 1 banked (otpu_slice)
  parameter bit TMEM_RQ    = 1'b0,    // TMEM_IMPL 1: request-priority bank addresses (otpu_slice)
  parameter int TMEM_NCP   = 3,       // TMEM_IMPL 1: copies (otpu_slice)
  parameter int TMEM_NSB   = 8,       // TMEM_IMPL 3: sub-banks per bank (otpu_slice)
  parameter int TMEM_SBP   = 3,       // TMEM_IMPL 3: the sub-bank field's word-address bit
  parameter int TMEM_SBH   = 0,       // TMEM_IMPL 3: the sub-bank hashed (otpu_slice)
  parameter bit TMEM_RPL   = 1'b0,    // TMEM_IMPL 3: lost reads replayed (otpu_slice)
  parameter bit TMEM_TDP   = 1'b0,    // TMEM_IMPL 3: true dual-port RAMs (otpu_slice)
  parameter bit TMEM_GST   = 1'b0,    // TMEM_IMPL 3: QUANT 3 and COLL the DMA's guests (otpu_slice)
  parameter int TMEM_PMP   = 0,       // TMEM_TDP: 2 the pumped RAMs' model (otpu_slice)
  parameter int TMEM_PRK   = 0,       // TMEM_IMPL 3: the read ports' ranks (otpu_slice)
  parameter int TMEM_SDM   = -1,      // TMEM_TDP: the read ports' RAM sides (otpu_slice)
  parameter string TMEM_STYLE = "block", // TMEM_PMP 1: the pumped RAMs' (otpu_slice)
  parameter string IMEM_STYLE = "auto",  // IMEM's RAMs (otpu_slice)
  parameter string IMEM_WC_STYLE = "",   // its wait entries', if not IMEM_STYLE's (otpu_slice)
  parameter string MXU_RF_STYLE = "block", // the MXU's result FIFO (otpu_slice)
  parameter string MXU_FIFO_STYLE = "block", // the MXU's chunk FIFO lanes (otpu_slice)
  parameter int MXU_RF_ROWS = 0,      // the MXU's result FIFO rows (otpu_slice; 0: its rule)
  parameter int ACT_UCOLS  = 0,       // the ACT RAM's columns in UltraRAM (otpu_slice)
  parameter bit SEQ_WC     = 1'b0,    // EXPERIMENTAL: otpu_seq's wait counts (otpu_slice)
  parameter bit COLL_GATHER = 1'b1,   // 0 (S = 1 only): the collective unit runs BAR alone
                                      // (otpu_coll_bar), as the card builds it (otpu_board)
  parameter bit COLL_HUB   = 1'b0,    // S > 1: the collective as otpu_board builds it there
                                      // (otpu_coll_hub, an otpu_coll_port a slice, XRS register
                                      // stages each way); 0: otpu_coll on the slices' own ports
  parameter int XRS        = 2,       // COLL_HUB: the stages each way (otpu_board XRS, >= 2)
  parameter bit COLL_PIPE  = 1'b0,    // COLL_HUB: the hub's reads in flight (otpu_board COLL_PIPE)
  parameter int NCH        = 2,       // the memory's channels and beat bytes (otpu_native_dram;
  parameter int CB         = 64,      //   port W's wide chunk NCH x CB in every memory path)
  parameter bit CHASH      = 1'b1,    // the channels' hash (otpu_native_dram; at WG > 1 the
                                      //   Config's, rtlsim: also otpu_dram's port W lanes)
  parameter int NAT_RD     = 256 / NCH, // otpu_native_dram's per-channel depths (otpu_board's)
  parameter int NAT_AD     = 64,
  parameter int NAT_XD     = 64,
  parameter int NAT_WQD    = 32,
  parameter int NAT_APF    = 32,
  parameter bit NAT_FILL   = 1'b1,
  parameter bit NAT_WPOST  = 1'b0,    // write data posted at the take (otpu_native_dram WPOST; the
                                      //   model's write-data FIFO 32 beats, as the WG targets' bridges)
  parameter bit NAT_SWG    = 1'b0,    // NAT_WPOST: one SW gather beat and one posted beat a cycle
                                      //   (otpu_native_dram SWG)
  parameter bit NAT_XPOST  = 1'b0,    // AXI 1, WG > 1: port W's beats posted into the MXU's chunk
                                      //   FIFO by the channels (otpu_native_dram / otpu_mxu XPOST,
                                      //   the model's p_* at the core clock: otpu_mxu QB 0)
  parameter int NAT_XQB    = 0,       // NAT_XPOST: each beat in 2^NAT_XQB writes (otpu_mxu QB; the
                                      //   the WG targets' 2: quarters, one a cycle, from a queue a channel)
  parameter int NAT_XSTG   = 0,       // AXI 1 / 2: register stages each way between a slice and
                                      //   its adapter (otpu_xstg, as otpu_board's; 0: wired),
  parameter int NAT_XSTG_P = NAT_XSTG, //  and the posts' and their counts' (otpu_xstg NP)
  parameter bit NAT_RHB    = 1'b0,    // AXI 1: read / write data on the channels in half beats
  parameter bit NAT_WHB    = 1'b0,    //   (otpu_native_dram RHB / WHB, the model's; WHB: its
                                      //   write-data FIFO 16 beats, as the WG targets' bridges)
  parameter int WG         = 1,       // weight rows a group (otpu_slice; port W's P = WG x D)
  parameter int SE_R       = 0,       // streams on an engine of their own (otpu_slice)
  parameter int DMA_DEPTH  = 128,     // otpu_dma DEPTH (otpu_slice)
  parameter int DMA_LDQ    = 1,       // otpu_dma LDQ (otpu_slice)
  parameter int MXU_PUMP   = 0,       // the MXU's array pumped on clk_f (otpu_mxu PUMP)
  parameter int VPU_FSUM   = 0,       // otpu_vpu FSUM: EXP2SUB's fused row sums (Config.FSUM)
  parameter int VPU_SIG    = 0,       // otpu_vpu VSIG: VOP SIGMOID / SILU (Config.VSIG)
  parameter int VPU_MAC    = 0,       // otpu_vpu MAC: VOP MAC (Config.VMAC)
  parameter int VPU_OUTER1 = 0,       // otpu_vpu OUTER1: VOP OUTER decay one on MAC's (Config.OUTER1)
  parameter int VPU_CNS    = 3,       // otpu_vpu CNS: the composites' stages per lane (3, or 5: two passes)
  parameter bit OUTER      = 1'b1,    // VOP OUTER (otpu_slice; 0: a VPU without it)
  parameter bit DSTEP2     = 1'b0,    // DSTEP's pair (otpu_slice)
  parameter int ST16       = 0,       // a stream's DRAM state in 16 bits (otpu_slice)
  parameter bit ST16S      = 1'b0,    // with ST16: STREAM's flags ST16R / ST16W (otpu_slice)
  parameter bit KT         = 1'b0,    // MM's K tail (otpu_slice; WG > 1)
  parameter int MXU_SPL    = 0,       // MM SPL, a split's terms as one MM (otpu_slice; WG > 1, Config.SPLMM)
  parameter bit ICACHE     = 1'b0,    // IMEM an instruction cache over the program (otpu_slice)
  parameter int IC_LINES   = 16,      // ICACHE: its lines (otpu_slice)
  parameter int IC_PF      = 2,       // ICACHE: the lines prefetched ahead (otpu_slice)
  parameter int AW         = 32       // DRAM byte address bits (otpu_slice; otpu_seq AW)
) (
  input  logic          clk,
  input  logic          clk_f,         // TMEM_PMP 1: 2 x clk, rising with it (otpu_slice)
  input  logic          sys_rst,
  input  logic          rst,
  input  logic [31:0]   rinit [8],     // the run's arguments (R8..R15 at the start)
  input  logic          ld_start,
  input  logic [31:0]   ld_addr,
  input  logic [31:0]   ld_n,
  output logic          ld_busy,
  output logic          all_halted,
  output logic          any_error,
  output logic [31:0]   icount [S],
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
  input  logic             x2_rst,
  input  logic [7:0]       x2_card, x2_ncards,
  input  logic [31:0]      coll_wd,
  output logic [7:0]       coll_status,
  output logic [15:0]      x2_seq,
  input  logic          dump
);
  logic [S-1:0] halted, error, coll_req, ldb;
  assign ld_busy = |ldb;
  cmd_t         coll_cmd [S];
  logic         coll_ack, coll_gnt;
  logic [S-1:0][LANES-1:0]       coll_ren;
  logic [S-1:0][LANES-1:0][31:0] coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;
  // each slice's collective port (sc_*): the unit's own (coll_*), or at COLL_HUB and S > 1 its
  // otpu_coll_port's, the hub between them and the unit (as otpu_board's g_xs)
  localparam bit HUB = (COLL_HUB && S > 1) || NC > 1;
  logic [S-1:0]                  sc_req, sc_ack, sc_gl, sc_gnt, sc_wtop;
  cmd_t                          sc_cmd [S];
  logic [S-1:0][LANES-1:0]       sc_ren, sc_wen;
  logic [S-1:0][LANES-1:0][31:0] sc_raddr, sc_rdata, sc_waddr, sc_wdata;

  localparam int CL = $clog2(NCH), P = NCH * CB, CAW = AW - $clog2(P);
  localparam int WA = waw(AW);                        // the slices' word addresses (otpu_slice)
  initial if (AXI == 2 && AW != 32) $fatal(1, "otpu_top: AW %0d with the card's DDR3 (AXI 2)", AW);
  localparam int AB = (WG > 1) ? CB : 32;            // port A's read bytes (otpu_slice)
  // a QST's footprint beat (otpu_slice QB, as otpu_board's): the native adapter's fill beat, CB,
  // where it fills partial SW beats (NAT_FILL) and CB > 64; else 64 bytes
  localparam int QB = (AXI != 0 && NAT_FILL && CB > 64) ? $clog2(CB) : 6;
  initial if (AXI == 2 && (NCH != 2 || CB != 64)) $fatal(1, "otpu_top: AXI 2 is the card's channels (NCH 2, CB 64)");
  initial if (NAT_XPOST && (AXI != 1 || WG < 2 || D != CB))
    $fatal(1, "otpu_top: NAT_XPOST needs the native model (AXI 1), WG > 1 and D = CB");
  localparam int XSW = $clog2(FIFO_DEPTH);           // a chunk FIFO entry (otpu_mxu w_slot)
  localparam int QH = (NAT_XQB > 0) ? 1 << NAT_XQB : 1;   // p_q's bits (one-hot)
  initial if (NAT_XQB != 0 && (!NAT_XPOST || NAT_XQB > 2))
    $fatal(1, "otpu_top: NAT_XQB %0d needs NAT_XPOST and at most 2 (otpu_mxu QB)", NAT_XQB);
  for (genvar s = 0; s < S; s++) begin : g_slice
    logic          a_req, a_we, a_rvalid, b_req, b_tag, b_we, b_rvalid, b_rtag;
    logic [CL-1:0] b_par;
    logic          pw_rdy, pw_req;
    logic [WA-1:0] pw_addr;
    logic [NCH-1:0] pw_bv;
    logic [P*8-1:0] pw_bd;
    logic [D/4-1:0] b_wmask;
    logic [D*8-1:0] b_wdata;
    logic [WA-1:0] a_addr, b_addr;
    logic [31:0]   a_wdata;
    logic [AB*8-1:0] a_rdata;
    logic [3:0]    a_be;
    logic [D*8-1:0] b_rdata;
    logic          a_rdy, b_rdy, wr_idle, rd_idle, sw_req, sw_rdy, a_inval;
    logic [WA-1:0] sw_addr;
    logic [32*QUANT_SWW-1:0] sw_wdata;
    logic [4*QUANT_SWW-1:0]  sw_be;
    logic [XSW-1:0] pw_slot;              // NAT_XPOST: the posted W beats (otpu_slice p_*)
    logic          pw_one;
    logic [NCH-1:0] p_we;
    logic [NCH-1:0][1:0][QH-1:0] p_q;     // (otpu_mxu's copies)
    logic [NCH-1:0][4:0][XSW-1:0] p_slot;
    logic [NCH-1:0][(D*8 >> NAT_XQB)-1:0] p_d;
    logic [NCH-1:0][XSW+1:0] p_cnt;
    // the posts as the slice takes them (through u_xs on the native paths)
    logic [NCH-1:0] sp_we;
    logic [NCH-1:0][1:0][QH-1:0] sp_q;
    logic [NCH-1:0][4:0][XSW-1:0] sp_slot;
    logic [NCH-1:0][(D*8 >> NAT_XQB)-1:0] sp_d;

    if (AXI == 0) begin : g_dram
      assign a_rdy = 1'b1;
      assign sw_rdy = 1'b1;
      assign b_rdy = 1'b1;
      assign pw_rdy = P > D;
      logic pw_wv;
      assign wr_idle = 1'b1;
      otpu_dram #(.WORDS(DRAM_WORDS), .D(D), .LAT(DRAM_LAT), .SID(s), .P(P), .AB(AB), .CB(CB),
                  .CHASH(CHASH), .AW(AW), .SWW(QUANT_SWW)) u_dram (
        .clk, .rst(sys_rst), .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
        .sw_req, .sw_addr, .sw_wdata, .sw_be,
        .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_rvalid, .b_rtag, .b_rdata,
        .w_req(pw_req), .w_addr(pw_addr), .w_rvalid(pw_wv), .w_rdata(pw_bd), .rd_idle, .dump);
      // port W's chunk in one cycle: every lane (lane c logical beat c ^ h, as the adapter's)
      assign pw_bv = {NCH{pw_wv}};
      assign p_we = '0;
      assign p_q = '0;
      assign p_slot = '0;
      assign p_d = '0;
      assign p_cnt = '0;
      assign sp_we = '0, sp_q = '0, sp_slot = '0, sp_d = '0;
    end else begin : g_native
      logic [NCH-1:0] cvalid, cready, cwe, wvalid, wready, rvalid;
      logic [NCH-1:0][CAW-1:0] caddr;
      logic [NCH-1:0][CB*8-1:0] wdata, rdata;
      logic [NCH-1:0][CB-1:0] wmask;
      logic [NCH-1:0][15:0] wdone;
      logic [NCH-1:0][5:0] wfree;
      logic [NCH-1:0][1:0] csrc;
      logic [NCH-1:0][XSW-1:0] cslot;
      logic [NCH-1:0][XSW+1:0] pcntg;
      // the adapter's side of u_xs (d*: the slice's signals at NAT_XSTG 0)
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
      logic [XSW-1:0] dw_slot;
      logic [NCH-1:0] dw_bv;
      logic [P*8-1:0] dw_bd;
      logic [NCH-1:0][XSW+1:0] d_p_cnt, d_p_cntg;
      otpu_xstg #(.N(NAT_XSTG), .NP(NAT_XSTG_P), .D(D), .NCH(NCH), .CB(CB), .AB(AB), .XSW(XSW), .XQB(NAT_XQB),
                  .AW(AW), .SWW(QUANT_SWW))
      u_xs (
        .clk, .p_clk(clk), .s_rst(sys_rst), .s_flush({rst, ld_start, a_inval}),   // as otpu_board
        .s_a_rdy(a_rdy), .s_a_req(a_req), .s_a_we(a_we), .s_a_addr(a_addr), .s_a_wdata(a_wdata),
        .s_a_be(a_be), .s_a_rvalid(a_rvalid), .s_a_rdata(a_rdata),
        .s_sw_rdy(sw_rdy), .s_sw_req(sw_req), .s_sw_addr(sw_addr), .s_sw_wdata(sw_wdata),
        .s_sw_be(sw_be),
        .s_b_rdy(b_rdy), .s_b_req(b_req), .s_b_tag(b_tag), .s_b_we(b_we), .s_b_wmask(b_wmask),
        .s_b_wdata(b_wdata), .s_b_addr(b_addr), .s_b_par(b_par), .s_b_rvalid(b_rvalid),
        .s_b_rtag(b_rtag), .s_b_rdata(b_rdata),
        .s_w_rdy(pw_rdy), .s_w_req(pw_req), .s_w_addr(pw_addr), .s_w_slot(pw_slot),
        .s_w_one(pw_one), .s_w_bv(pw_bv), .s_w_bd(pw_bd),
        .s_wr_idle(wr_idle), .s_rd_idle(rd_idle), .s_p_cnt(p_cnt),
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
        .c_p_we(p_we), .c_p_q(p_q), .c_p_slot(p_slot), .c_p_d(p_d), .c_p_cntg(pcntg));
      otpu_native_dram #(.D(D), .CHASH(CHASH), .NCH(NCH), .CB(CB), .AB(AB), .RD(NAT_RD),
                         .AD(NAT_AD), .XD(NAT_XD), .WQD(NAT_WQD), .APF(NAT_APF),
                         .SW_FILL(NAT_FILL), .WPOST(NAT_WPOST), .SWG(NAT_SWG), .XPOST(NAT_XPOST),
                         .XSW(XSW), .RHB(NAT_RHB), .WHB(NAT_WHB), .AW(AW), .SWW(QUANT_SWW)) u_adapt (
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
        .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
        .n_csrc(csrc), .n_cslot(cslot),
        .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
        .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .n_wfree(wfree),
        .p_cntg(d_p_cntg), .p_cnt(d_p_cnt));
      if (AXI == 2) begin : g_ldc
        initial if (NAT_WPOST || NAT_RHB || NAT_WHB)
          $fatal(1, "otpu_top: NAT_WPOST / NAT_RHB / NAT_WHB with LiteDRAM's controller (AXI 2)");
        assign wfree = '0;
        assign pcntg = '0;
        assign p_we = '0;
        assign p_q = '0;
        assign p_slot = '0;
        assign p_d = '0;
        otpu_ldc_mem #(.WORDS(DRAM_WORDS), .SID(s), .CHASH(CHASH)) u_mem (
          .clk, .rst(sys_rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .dump);
      end else begin : g_model
        logic [NCH-1:0][CB*8-1:0] pd;
        logic [NCH-1:0] mwe;                 // the model's posts, a beat each
        logic [NCH-1:0][XSW-1:0] mslot;
        logic [NCH-1:0][XSW+1:0] mcntg;
        otpu_native_mem #(.WORDS(DRAM_WORDS), .LAT(DRAM_LAT), .SID(s), .NCH(NCH), .CB(CB),
                          .CHASH(CHASH), .WFD(NAT_WPOST ? (NAT_WHB ? 16 : 32) : 0),
                          .XPOST(NAT_XPOST), .XSW(XSW), .RHB(NAT_RHB), .WHB(NAT_WHB), .AW(AW)) u_mem (
          .clk, .rst(sys_rst), .run_rst(rst),
          .n_cvalid(cvalid), .n_cready(cready), .n_cwe(cwe), .n_caddr(caddr),
          .n_csrc(csrc), .n_cslot(cslot),
          .n_wvalid(wvalid), .n_wready(wready), .n_wdata(wdata), .n_wmask(wmask),
          .n_rvalid(rvalid), .n_rdata(rdata), .n_wdone(wdone), .n_wfree(wfree),
          .p_we(mwe), .p_slot(mslot), .p_d(pd), .p_cntg(mcntg), .dump);
        if (NAT_XQB == 0) begin : g_beat
          assign p_we = mwe, p_q = '0, pcntg = mcntg;
          for (genvar c = 0; c < NCH; c++) begin : g_ps
            assign p_slot[c] = {5{mslot[c]}};
          end
          for (genvar c = 0; c < NCH; c++) begin : g_pd
            assign p_d[c] = (D*8)'(pd[c]);   // D = CB at NAT_XPOST (checked)
          end
        end else begin : g_qtr
          // each posted beat as 2^NAT_XQB writes of a quarter, in order, one a cycle from a queue
          // a channel (as a bridge's core half posts, otpu_hbm_ch): p_q one-hot with p_we, and
          // the count stepping on the edge of the beat's last quarter (the adapter's pcntg)
          localparam int QW = D * 8 >> NAT_XQB;
          for (genvar c = 0; c < NCH; c++) begin : g_c
            typedef struct packed {
              logic [XSW-1:0] slot;
              logic [D*8-1:0] d;
            } pbeat_t;
            pbeat_t bq [$];
            logic [NAT_XQB-1:0] qi;
            logic [XSW+1:0] nb;
            always_ff @(posedge clk)
              if (sys_rst) begin
                bq.delete();
                qi <= '0; nb <= '0; p_we[c] <= 1'b0; p_q[c] <= '0;
              end else begin
                if (mwe[c]) bq.push_back('{mslot[c], (D*8)'(pd[c])});
                p_we[c] <= 1'b0;
                p_q[c] <= '0;
                if (bq.size() != 0) begin
                  p_we[c] <= 1'b1;
                  p_q[c] <= {2{QH'(1) << qi}};
                  p_slot[c] <= {5{bq[0].slot}};
                  p_d[c] <= bq[0].d[QW * qi +: QW];
                  qi <= qi + 1'b1;
                  if (&qi) void'(bq.pop_front());
                end
                if (p_we[c] && p_q[c][0][QH-1]) nb <= nb + 1'b1;
              end
            assign pcntg[c] = nb ^ (nb >> 1);
          end
        end
      end
      // the adapter's counters (rtlsim: A runs and fill reads, the partial writes by source)
      always @(posedge clk) if (dump)
        for (int c = 0; c < NCH; c++) begin
          $write("NATIVE ch%0d a_runs=%0d a_rd=%0d b_rd=%0d sw_rd=%0d w_rd=%0d ", c,
                 u_adapt.st_arun[c], u_adapt.st_ard[c], u_adapt.st_brd[c], u_adapt.st_srd[c],
                 u_adapt.st_xrd[c]);
          $display("b_wr=%0d a_wr=%0d sw_wr=%0d part_b=%0d part_a=%0d part_sw=%0d",
                   u_adapt.st_bwr[c], u_adapt.st_awr[c], u_adapt.st_swr[c], u_adapt.st_pb[c],
                   u_adapt.st_pa[c], u_adapt.st_ps[c]);
        end
    end

    otpu_slice #(.SID(s), .S(S), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .ACT_ROWS(ACT_ROWS),
                 .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
                 .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB),
                 .WPB(WPB), .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .MXU_QD(MXU_QD), .QUANT_XD(QUANT_XD), .QUANT_R1(QUANT_R1), .QUANT_PW(QUANT_PW), .QUANT_SPLIT(QUANT_SPLIT), .QUANT_HL(QUANT_HL), .QUANT_SWW(QUANT_SWW), .QUANT_QSTB(QUANT_QSTB), .QUANT_QSTHL(QUANT_QSTHL), .QONCE(QONCE), .VPU_CL(VPU_CL), .ULANES(ULANES),
                 .TMEM_IMPL(TMEM_IMPL), .TMEM_RQ(TMEM_RQ), .TMEM_NCP(TMEM_NCP),
                 .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
                 .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
                 .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
                 .TMEM_STYLE(TMEM_STYLE), .IMEM_STYLE(IMEM_STYLE), .ACT_UCOLS(ACT_UCOLS),
                 .MXU_RF_STYLE(MXU_RF_STYLE), .MXU_FIFO_STYLE(MXU_FIFO_STYLE),
                 .MXU_RF_ROWS(MXU_RF_ROWS),
                 .IMEM_WC_STYLE(IMEM_WC_STYLE),
                 .SEQ_WC(SEQ_WC), .NCH(NCH), .CB(CB), .QB(QB), .WG(WG), .SE_R(SE_R), .OUTER(OUTER),
                 .DMA_DEPTH(DMA_DEPTH), .DMA_LDQ(DMA_LDQ), .MXU_PUMP(MXU_PUMP), .VPU_FSUM(VPU_FSUM), .VPU_SIG(VPU_SIG), .VPU_MAC(VPU_MAC), .VPU_OUTER1(VPU_OUTER1), .VPU_CNS(VPU_CNS), .XPOST(NAT_XPOST), .XQB(NAT_XQB), .CHASH(CHASH), .DSTEP2(DSTEP2), .ST16(ST16), .ST16S(ST16S),
                 .KT(KT), .MXU_SPL(MXU_SPL), .ICACHE(ICACHE), .IC_LINES(IC_LINES), .IC_PF(IC_PF), .AW(AW))
    u_slice (
      .clk, .clk_f, .sys_rst, .rst, .rinit, .ld_start, .ld_addr, .ld_n, .ld_busy(ldb[s]),
      .a_rdy, .b_rdy, .sw_rdy, .wr_idle, .rd_idle,
      .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
      .sw_req, .sw_addr, .sw_wdata, .sw_be,
      .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
      .pw_rdy, .pw_req, .pw_addr, .pw_bv, .pw_bd, .pw_slot, .pw_one,
      .p_clk(clk), .p_we(sp_we), .p_q(sp_q), .p_slot(sp_slot), .p_d(sp_d), .p_cnt,
      .coll_req(sc_req[s]), .coll_cmd(sc_cmd[s]), .coll_ack(sc_ack[s]),
      .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
      .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
      .coll_wtop(sc_wtop[s]), .coll_gnt_local(sc_gl[s]), .coll_gnt(sc_gnt[s]),
      .halted(halted[s]), .error(error[s]), .wait_to(), .a_inval, .icount(icount[s]), .pf(), .dump);
  end

  logic coll_gerr;                    // a GATHER on the BAR-only collective unit
  initial if (S > 1 && !COLL_GATHER) $fatal(1, "otpu_top: COLL_GATHER 0 needs S = 1");
  if (COLL_GATHER) begin : g_coll
    otpu_coll #(.S(S), .LANES(LANES), .RQ(!HUB)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(S), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gnt), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
  end

  if (!HUB) begin : g_cw                // the slices on the unit's ports
    for (genvar s = 0; s < S; s++) begin : g_s
      assign coll_req[s] = sc_req[s];
      assign coll_cmd[s] = sc_cmd[s];
      assign sc_ack[s] = coll_ack;
      assign sc_ren[s] = coll_ren[s];
      assign sc_raddr[s] = coll_raddr[s];
      assign coll_rdata[s] = sc_rdata[s];
      assign sc_wen[s] = coll_wen;
      assign sc_waddr[s] = coll_waddr;
      assign sc_wdata[s] = coll_wdata;
      assign sc_wtop[s] = 1'b1;
      assign sc_gnt[s] = coll_gnt;
    end
    assign coll_gnt = &sc_gl;
  end else begin : g_ch
    // otpu_board's S > 1 collective (g_xs): the hub beside the unit, a port a slice, between them
    // XRS register stages each way (otpu_board's u_dn_tx / u_dn_rx, u_up_tx / u_up_rx at 2; the
    // low three bits, the pulses and the request level, cleared by reset). The slices' control
    // is not staged here (otpu_board's is: a run's start and HALTED a few cycles apart)
    localparam int L = LANES, CMB = $bits(cmd_t);
    localparam int DW = (NC > 1 ? 1 : 0) + 3 + 2 * L + 96 * L, UW = 3 + 32 * L + CMB;
    initial if (XRS < 2) $fatal(1, "otpu_top: XRS %0d (>= 2: a stage on each side)", XRS);
    // the hub's reads in flight and the ports' writes as otpu_board's (g_xs CNR, CF)
    localparam int CNR = (COLL_PIPE || NC > 1) ? 2 ** $clog2(2 * XRS + 8) : 1;
    localparam int CF  = (COLL_PIPE || NC > 1) ? 2 ** $clog2(2 * XRS + 6) : 8;
    logic                      h_ack, h_wv, h_wt;
    logic [S-1:0]              h_rv, h_req, h_dv, h_pop;
    logic [L-1:0]              h_rm, h_wm;
    logic [L-1:0][31:0]        h_ra, h_wa, h_wd;
    logic [S-1:0][L-1:0][31:0] h_dd;
    cmd_t                      h_cmd [S];
    logic [1:0] l_tv, l_tlast, l_rcred, l_tcred, l_rv, l_rhdr, l_rlast;
    logic [1:0] l_up, l_rerr, l_abin;
    logic [1:0][255:0] l_td, l_rd;
    logic l_abort;
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
        otpu_coll_lx #(.XRS(XRS)) u_lx (
          .clk, .rst(x2_rst), .ren(!rst), .t_v(l_tv[p]), .t_last(l_tlast[p]), .t_d(l_td[p]),
          .r_cred(l_rcred[p]), .abort(l_abort), .card(x2_card), .t_cred(l_tcred[p]),
          .r_v(l_rv[p]), .r_hdr(l_rhdr[p]), .r_last(l_rlast[p]), .r_d(l_rd[p]),
          .up(l_up[p]), .r_err(l_rerr[p]), .ab_in(l_abin[p]), .li, .lo);
      end
    end else begin : g_no_link
      assign {l_tcred, l_rv, l_rhdr, l_rlast, l_rd, l_up, l_rerr, l_abin} = '0;
    end

    otpu_coll_hub #(.S(S), .LANES(L), .F(CF), .NR(CNR), .NC(NC)) u_chub (
      .clk, .rst,
      .u_req(coll_req), .u_cmds(coll_cmd), .u_gnt(coll_gnt), .u_ack(coll_ack),
      .u_ren(coll_ren), .u_raddr(coll_raddr), .u_rdata(coll_rdata),
      .u_wen(coll_wen), .u_waddr(coll_waddr), .u_wdata(coll_wdata),
      .x_req(h_req), .x_cmds(h_cmd), .x_ack(h_ack), .x_rv(h_rv), .x_rm(h_rm), .x_ra(h_ra),
      .x_dv(h_dv), .x_dd(h_dd), .x_wv(h_wv), .x_wm(h_wm), .x_wa(h_wa), .x_wd(h_wd),
      .x_pop(h_pop), .x_wt(h_wt), .lrst(x2_rst), .cid(x2_card), .ncards(x2_ncards),
      .wdog(coll_wd), .l_tv, .l_tlast, .l_td, .l_rcred, .l_abort, .l_tcred, .l_rv,
      .l_rhdr, .l_rlast, .l_rd, .l_up, .l_rerr, .l_abin,
      .xst(coll_status), .xerr(coll_xerr), .xseq(x2_seq));
    for (genvar s = 0; s < S; s++) begin : g_x
      logic [DW-1:0]      dn_d, dn_q;
      logic [UW-1:0]      up_d, up_q;
      logic [XRS-1:0][DW-1:0] dn_r;     // the stages, the last one dn_q / up_q
      logic [XRS-1:0][UW-1:0] up_r;
      logic               p_rv, p_wv, p_wt, p_dv, p_pop;
      logic [L-1:0]       p_rm, p_wm;
      logic [L-1:0][31:0] p_ra, p_wa, p_wd, p_dd;
      if (NC > 1) begin : g_tag
        assign dn_d = {h_wt, h_wd, h_wa, h_wm, h_ra, h_rm, h_wv, h_rv[s], h_ack};
        assign p_wt = dn_q[DW-1];
      end else begin : g_no_tag
        assign p_wt = 1'b1;
        assign dn_d = {h_wd, h_wa, h_wm, h_ra, h_rm, h_wv, h_rv[s], h_ack};
      end
      assign {p_wd, p_wa, p_wm, p_ra, p_rm, p_wv, p_rv, sc_ack[s]} = dn_q[DW-(NC > 1 ? 1 : 0)-1:0];
      assign up_d = {sc_cmd[s], p_dd, p_pop, p_dv, sc_req[s]};
      assign {h_cmd[s], h_dd[s], h_pop[s], h_dv[s], h_req[s]} = up_q;
      assign dn_q = dn_r[XRS-1];
      assign up_q = up_r[XRS-1];
      always_ff @(posedge clk) begin
        dn_r[0] <= dn_d;
        up_r[0] <= up_d;
        for (int i = 1; i < XRS; i++) begin
          dn_r[i] <= dn_r[i-1];
          up_r[i] <= up_r[i-1];
        end
        if (rst)
          for (int i = 0; i < XRS; i++) begin
            dn_r[i][2:0] <= '0;
            up_r[i][2:0] <= '0;
          end
      end
      otpu_coll_port #(.LANES(L), .F(CF), .NR(CNR), .WT(NC > 1)) u_cport (
        .clk, .rst, .rv(p_rv), .rm(p_rm), .ra(p_ra), .wv(p_wv), .wm(p_wm), .wa(p_wa),
        .wd(p_wd), .wt(p_wt), .coll_wtop(sc_wtop[s]), .dv(p_dv), .dd(p_dd), .pop(p_pop),
        .coll_ren(sc_ren[s]), .coll_raddr(sc_raddr[s]), .coll_rdata(sc_rdata[s]),
        .coll_wen(sc_wen[s]), .coll_waddr(sc_waddr[s]), .coll_wdata(sc_wdata[s]),
        .gnt_local(sc_gl[s]), .gnt(sc_gnt[s]));
    end
  end

  logic coll_xerr;
  if (NC == 1) begin : g_no_x2
    assign {x2_s_tdata, x2_s_tvalid, x2_s_tlast, x2_m_tready, x2_abort} = '0;
    if (!HUB) begin : g_status
      assign {coll_status, x2_seq, coll_xerr} = '0;
    end
  end
  assign all_halted = &halted;
  assign any_error  = |error || coll_gerr || coll_xerr;
endmodule
