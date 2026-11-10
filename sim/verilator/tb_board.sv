// Board-level simulation: otpu_board (control registers, S slices, each with the native DRAM
// adapter on NCH channels of CB-byte beats) in front of the channels' memory holding their
// physical images (ch0.bin .. ch<NCH-1>.bin; slice s's are ch<NCH s> .. ch<NCH s + NCH - 1>):
// MEM_NATIVE 2 (the default; the cards' two 64-byte channels only) the build's channels,
// otpu_mem_ch in front of a LiteDRAM native-port model (otpu_chmem); 1 the native memory model
// alone (otpu_native_mem: the WG targets' channels, their rate and latency +nat_* / +axi_lat,
// opentpu.target.mem_model) -- driven by a host script
// (+dir=<d>, <d>/host.txt) -- the same register and memory protocol the PCIe host driver uses (opentpu/host/board.py). Script lines:
//   W <addr> <value>          AXI-Lite write
//   P <addr> <mask> <value>   poll until (read(addr) & mask) == value
//   R <addr>                  read, printed as "REG <addr> <value>"
//   C <cycles>                wait
// Numbers are hex. At the end the channel memories are dumped (ch<c>_out.bin).
// +trace prints the slice's trace lines (as tb_top; the hardware trace records the same events).
// The I2C pins are four open-drain lines with pull-ups and nothing else on them; +i2c_hold=<hex>
// holds lines low from the outside (bits 0..3 as I2C_CTRL, bit4 ALERT0). +calib=<hex>: the
// channels' calibration flags (default all set: calibrated); +ded=<hex>: their uncorrectable-ECC
// flags (default 0; STATUS ECC_DED). +mrst_c=K: in every C wait longer than K + 4 cycles, the
// control master's reset (ctl_mrst: a hot reset's, otpu_axil_iso) for 4 cycles from its K-th.
module tb_board;
  parameter int NCARDS    = 1;           // remote hub; unconnected links for its abort test
  parameter int S          = 1;           // slices, NCH channels each
  parameter int WORDS      = 1 << 20;     // logical memory words (a slice's channels)
  parameter int NCH        = 2;           // channels per slice, their beat bytes (otpu_board)
  parameter int CB         = 64;
  parameter int NAT_RD     = 256 / NCH;   // the adapter's depths (otpu_board NAT_*)
  parameter int NAT_AD     = 64;
  parameter int NAT_XD     = 64;
  parameter int NAT_WQD    = 32;
  parameter int NAT_APF    = 32;
  parameter bit NAT_FILL   = 1'b1;
  parameter bit NAT_WPOST  = 1'b0;        // (otpu_board; the model's write-data FIFO of 32 beats)
  parameter bit NAT_SWG    = 1'b0;        // (otpu_board)
  parameter bit NAT_XPOST  = 1'b0;        // (otpu_board; the model posts W beats at the core clock)
  parameter int NAT_XSTG   = 0;           // (otpu_board: stages between a slice and its adapter)
  parameter int NAT_XSTG_P = NAT_XSTG;    // (otpu_board: the posts')
  parameter bit NAT_RHB    = 1'b0;        // (otpu_board; the model's read / write data in half
  parameter bit NAT_WHB    = 1'b0;        //   beats, WHB its write-data FIFO 16 beats)
  parameter int FIFO_DEPTH = 1024;        // the MXU's chunk FIFO (otpu_board's default)
  parameter int WG         = 1;           // weight rows a group (otpu_board)
  parameter bit CHASH      = 1'b1;        // the channels' hash (otpu_board; CAPS bit7)
  parameter int D          = 128;
  parameter int MCOLS      = 4;
  parameter int ACT_ROWS   = MCOLS;
  parameter int ACT_BLOCKS = 128;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int IMEM_WORDS = 1 << 15;
  parameter int LANES      = 8;
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int MXU_IMPL   = 2;      // MXU dot product: 2 systolic (docs/mxu_systolic.md), the only one
  parameter int MXU_E      = 0;      // MXU epilogue lanes (0: MCOLS / 2; the board's: its target's)
  parameter int MXU_QD     = 2;      // MXU commands queued (otpu_mxu QD; its target's)
  parameter bit QUANT_R1   = 1'b0;   // RSCALE's row factor read once a row (otpu_quant R1; its target's)
  parameter int QUANT_PW   = 1;      // a buffered QACT pass 1's chunks a cycle (otpu_quant PW; its target's)
  parameter bit QUANT_SPLIT = 1'b0;  // QACT / QST flag LO (otpu_quant SPLIT; Config.SPLITQ)
  parameter bit QUANT_HL    = 1'b0; // QACT flag HL (otpu_quant HL; Config.QHL)
  parameter int QUANT_SWW  = 1;      // the SW port's words (otpu_quant SWW; timing only)
  parameter bit QUANT_QSTB  = 1'b0;   // QST rows through the row buffer (otpu_quant QSTB; timing only)
  parameter bit QUANT_QSTHL = 1'b0;   // QST HL: a GRP ROW QST's hi and lo from one read (otpu_quant QSTHL; Config.QSTHL)
  parameter bit QONCE      = 1'b0;    // QACT flag ST and LDA (Config.QONCE, CAPS2 bit24)
  parameter int ULANES     = LANES;
  parameter int WIN        = 16;
  parameter int LAT        = 20;
  parameter int CORE_KHZ   = 100000;
  parameter logic [31:0] BUILD_ID = 32'h0B0A_4D00;
  parameter int DDR_MTS    = 0;
  parameter int TRACE_DEPTH = 16384;
  parameter bit COLL_GATHER = 1'b1;       // 0: the BAR-only collective unit (otpu_coll_bar) at
                                          // S = 1 (rtlsim.BOARD_UARCH); S > 1 has GATHER
  parameter bit COLL_PIPE  = 1'b0;        // S > 1: the hub's reads in flight (otpu_board)
  parameter int XRS        = 2;           // S > 1: the crossings' stages each way (otpu_board)
  parameter int TRACE_QD   = 32;
  parameter int PQ_WIN     = 1024;
  parameter bit DSTEP      = 1'b1;
  parameter int MEM_NATIVE = 2;            // the channels: 2 otpu_chmem (otpu_mem_ch and a LiteDRAM
                                           // model), 1 otpu_native_mem
  parameter bit HOSTCAL    = 1'b0;         // CAPS bit27 (the host calibrates the controllers)
  parameter bit SEQ_WC     = 1'b0;    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
  parameter int SE_R       = 0;       // streams on an engine of their own (otpu_slice)
  parameter int DMA_DEPTH  = 128;     // otpu_dma DEPTH (otpu_slice)
  parameter int DMA_LDQ    = 1;       // otpu_dma LDQ (otpu_slice)
  parameter int MXU_PUMP   = 0;       // the MXU's array pumped on clk_f (otpu_mxu PUMP)
  parameter int VPU_FSUM   = 0;       // otpu_vpu FSUM: EXP2SUB's fused row sums (Config.FSUM)
  parameter int VPU_SIG    = 0;       // otpu_vpu VSIG: VOP SIGMOID / SILU (Config.VSIG)
  parameter int VPU_MAC    = 0;       // otpu_vpu MAC: VOP MAC (Config.VMAC)
  parameter int VPU_OUTER1 = 0;       // otpu_vpu OUTER1: VOP OUTER decay one on MAC's (Config.OUTER1)
  parameter int VPU_CNS    = 3;       // otpu_vpu CNS: the composites' stages per lane (3, or 5: two passes)
  parameter bit OUTER      = 1'b1;    // VOP OUTER (0: a VPU without it, CAPS2 bit1)
  parameter bit DSTEP2     = 1'b0;    // DSTEP's pair (CAPS2 bit2)
  parameter int ST16       = 0;       // a stream's DRAM state in 16 bits (CAPS2 bits 14:13)
  parameter bit ST16S      = 1'b0;    // with ST16: STREAM's flags ST16R / ST16W (CAPS2 bit18)
  parameter bit KT         = 1'b0;    // MM's K tail (CAPS2 bit5)
  parameter int MXU_SPL    = 0;       // MM SPL, a split's terms as one MM (CAPS2 bit25)
  parameter int PLATFORM   = 1;            // CFG_PLATFORM (opentpu/target.py)
  parameter logic [31:0] ARCH_HASH = 32'h0;  // CFG_HASH
  parameter logic [11:0] TEMP = 12'hA1A;  // the XADC code of 45 C
  parameter int NREG       = 0;            // bit j: channel j through two otpu_native_reg (an SLR
                                           // crossing's register slices, otpu_bcu_sys SLL)
  // the TMEM (otpu_board; docs/tmem_nocopy.md): 0 a copy per read port, 3 one copy and its knobs
  // (the card's target: opentpu/host/board.py passes them; PMP 1 on clk_f, below)
  parameter int TMEM_IMPL  = 0;
  parameter int TMEM_NSB   = 8;
  parameter int TMEM_SBP   = 3;
  parameter int TMEM_SBH   = 0;
  parameter bit TMEM_RPL   = 1'b0;
  parameter bit TMEM_TDP   = 1'b0;
  parameter bit TMEM_GST   = 1'b0;
  parameter int TMEM_PMP   = 0;
  parameter int TMEM_PRK   = 0;
  parameter int TMEM_SDM   = -1;
  parameter string TMEM_STYLE = "block";
  parameter string IMEM_STYLE = "auto";
  parameter string IMEM_WC_STYLE = "";
  parameter string MXU_RF_STYLE = "block";
  parameter string MXU_FIFO_STYLE = "block";
  parameter int MXU_RF_ROWS = 0;
  parameter int ACT_UCOLS  = 0;
  parameter int AW         = 32;     // DRAM byte address bits (otpu_board AW; the DRAM stays 4 GiB
                                     //   a slice: an address above it stops the run, g_chk)
  parameter bit ICACHE     = 1'b0;   // IMEM an instruction cache (otpu_board)
  parameter int IC_LINES   = 16;
  parameter int IC_PF      = 2;

  localparam int CAW = 32 - $clog2(NCH * CB), NC = NCH * S;
  initial if (MEM_NATIVE == 2 && (NCH != 2 || CB != 64))
    $fatal(1, "tb_board: MEM_NATIVE 2 (otpu_mem_ch, LiteDRAM) is the cards' two 64-byte channels");
  logic clk = 1'b0, clk_f = 1'b0, rst = 1'b1, dump = 1'b0, ctl_mrst = 1'b0;
  // TMEM_PMP 1: clk_f (2 x clk) rises with each edge of clk, in clk's own process (as tb_top);
  // clk's edges stay where they are (the memory models' clocks are absolute)
  if (TMEM_PMP == 1 || MXU_PUMP != 0) begin : g_clkf         // (and the pumped MXU's)
    initial forever begin
      #2 clk_f = 1'b0;
      #3 begin clk = ~clk; clk_f = 1'b1; end
    end
  end else begin : g_clk
    always #5 clk = ~clk;
  end

  // AXI-Lite
  logic [11:0] awaddr, araddr;
  logic        awvalid = 0, awready, wvalid = 0, wready, bvalid, bready = 1;
  logic        arvalid = 0, arready, rvalid, rready = 1;
  initial begin awaddr = 0; araddr = 0; wdata = 0; end
  logic [31:0] wdata, rdata;
  logic [1:0]  bresp, rresp;
  logic [2:0]  led;

  // native channels (slice s on NCH s .. NCH s + NCH - 1)
  logic [NC-1:0] ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [NC-1:0][CAW-1:0] nca;
  logic [NC-1:0][CB*8-1:0] nwd, nrd;
  logic [NC-1:0][CB-1:0] nwm;
  logic [NC-1:0][15:0] nwdone;
  logic [NC-1:0] nerr;

  // I2C: wired AND of the board's drive-low bits and the outside's holds, pulled up
  logic [3:0] i2c_lo;
  logic [4:0] i2c_hold = '0;
  initial void'($value$plusargs("i2c_hold=%h", i2c_hold));
  logic [NC-1:0] calib = '1, ded = '0;
  initial begin
    void'($value$plusargs("calib=%h", calib));
    void'($value$plusargs("ded=%h", ded));
  end

  otpu_board #(.NC(NCARDS), .S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
               .IMEM_WORDS(IMEM_WORDS), .LANES(LANES), .VPU_CL(VPU_CL), .MXU_IMPL(MXU_IMPL), .MXU_E(MXU_E), .MXU_QD(MXU_QD), .QUANT_R1(QUANT_R1), .QUANT_PW(QUANT_PW), .QUANT_SPLIT(QUANT_SPLIT), .QUANT_HL(QUANT_HL), .QUANT_SWW(QUANT_SWW), .QUANT_QSTB(QUANT_QSTB), .QUANT_QSTHL(QUANT_QSTHL), .QONCE(QONCE), .ULANES(ULANES), .WIN(WIN), .CORE_KHZ(CORE_KHZ),
               .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .TRACE_QD(TRACE_QD),
               .PQ_WIN(PQ_WIN), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL), .SEQ_WC(SEQ_WC), .PLATFORM(PLATFORM),
               .ARCH_HASH(ARCH_HASH), .COLL_GATHER(COLL_GATHER || S > 1), .COLL_PIPE(COLL_PIPE), .XRS(XRS),
               .TMEM_IMPL(TMEM_IMPL),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL),
               .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK),
               .TMEM_SDM(TMEM_SDM), .TMEM_STYLE(TMEM_STYLE), .IMEM_STYLE(IMEM_STYLE),
               .MXU_RF_STYLE(MXU_RF_STYLE), .MXU_FIFO_STYLE(MXU_FIFO_STYLE),
               .MXU_RF_ROWS(MXU_RF_ROWS),
               .IMEM_WC_STYLE(IMEM_WC_STYLE),
               .ACT_UCOLS(ACT_UCOLS), .NCH(NCH), .CB(CB), .NAT_RD(NAT_RD),
               .NAT_AD(NAT_AD),
               .NAT_XD(NAT_XD), .NAT_WQD(NAT_WQD), .NAT_APF(NAT_APF), .NAT_FILL(NAT_FILL),
               .NAT_WPOST(NAT_WPOST), .NAT_SWG(NAT_SWG), .NAT_XPOST(NAT_XPOST), .NAT_XQB(0), .NAT_XSTG(NAT_XSTG), .NAT_XSTG_P(NAT_XSTG_P),
               .NAT_RHB(NAT_RHB), .NAT_WHB(NAT_WHB),
               .FIFO_DEPTH(FIFO_DEPTH),
               .WG(WG), .CHASH(CHASH), .SE_R(SE_R), .DMA_DEPTH(DMA_DEPTH), .DMA_LDQ(DMA_LDQ), .MXU_PUMP(MXU_PUMP), .VPU_FSUM(VPU_FSUM), .VPU_SIG(VPU_SIG), .VPU_MAC(VPU_MAC), .VPU_OUTER1(VPU_OUTER1), .VPU_CNS(VPU_CNS), .OUTER(OUTER),
               .DSTEP2(DSTEP2), .ST16(ST16), .ST16S(ST16S), .KT(KT), .MXU_SPL(MXU_SPL), .ICACHE(ICACHE), .IC_LINES(IC_LINES), .IC_PF(IC_PF),
               .AW(AW)) dut (
    .clk, .clk_f, .rst, .ctl_mrst, .calib, .ded, .temp(TEMP), .led,
    .x2_s_tdata(), .x2_s_tvalid(), .x2_s_tlast(), .x2_s_tready('0),
    .x2_m_tdata('0), .x2_m_tvalid('0), .x2_m_tlast('0), .x2_m_thdr('0), .x2_m_tready(),
    .x2_up('0), .x2_rx_err('0), .x2_abort_in('0), .x2_abort(),
    .x2_rst(), .x2_card(), .x2_ncards(), .x2_c_we(), .x2_c_addr(), .x2_c_wd(), .x2_c_rd('0),
    .i2c_lo, .i2c_pin(~({1'b0, i2c_lo} | i2c_hold)),
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(bready),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(rready),
    .n_cvalid(bcv), .n_cready(bcr), .n_cwe(bcwe), .n_caddr(bca), .n_wvalid(bwv), .n_wready(bwr),
    .n_wdata(bwd), .n_wmask(bwm), .n_rvalid(brv), .n_rdata(brd), .n_wdone(bwdone),
    .n_wfree(nwfree), .n_err(berr),
    .n_csrc(ncsrc), .n_cslot(ncslot), .p_clk(clk), .p_we(pwe), .p_q('0), .p_slot(pslot5),
    .p_d(pd), .p_cntg(pcntg));
  // (the model's free count straight to the board: no register slices with WPOST)
  logic [NC-1:0][5:0] nwfree;
  initial if (NAT_WPOST && (NREG != 0 || MEM_NATIVE != 1))
    $fatal(1, "tb_board: NAT_WPOST needs the native model (MEM_NATIVE 1) and no register slices");
  // NAT_XPOST: the read commands' source and entry, the model's posted W beats straight to the
  // board (no register slices; whole beats, D = CB)
  localparam int XSW = $clog2(FIFO_DEPTH);
  logic [NC-1:0][1:0] ncsrc;
  logic [NC-1:0][XSW-1:0] ncslot, pslot;
  logic [NC-1:0] pwe;
  logic [NC-1:0][D*8-1:0] pd;
  logic [NC-1:0][XSW+1:0] pcntg;
  // (otpu_board's p_slot in its copies, one a lane RAM group: the model's entry in each; p_q,
  // the quarters' one-hot, unused at NAT_XQB 0)
  logic [NC-1:0][4:0][XSW-1:0] pslot5;
  for (genvar c = 0; c < NC; c++) begin : g_ps
    assign pslot5[c] = {5{pslot[c]}};
  end
  initial if (NAT_XPOST && (NREG != 0 || MEM_NATIVE != 1 || D != CB))
    $fatal(1, "tb_board: NAT_XPOST needs the native model (MEM_NATIVE 1), no NREG and D = CB");

  // the board's side of the channels, and the register slices where NREG asks for them
  logic [NC-1:0] bcv, bcr, bcwe, bwv, bwr, brv, berr;
  logic [NC-1:0][CAW-1:0] bca;
  logic [NC-1:0][CB*8-1:0] bwd, brd;
  logic [NC-1:0][CB-1:0] bwm;
  logic [NC-1:0][15:0] bwdone;
  for (genvar j = 0; j < NC; j++) begin : g_nreg
    if (NREG >> j & 1) begin : g_reg
      logic cv, cr, cwe, wv, wr, rv, err;
      logic [CAW-1:0] ca;
      logic [CB*8-1:0] wd, rd;
      logic [CB-1:0] wm;
      logic [15:0] wdone;
      otpu_native_reg #(.CB(CB), .CAW(CAW)) u_tx (.clk, .rst,
        .s_cvalid(bcv[j]), .s_cready(bcr[j]), .s_cwe(bcwe[j]), .s_caddr(bca[j]),
        .s_wvalid(bwv[j]), .s_wready(bwr[j]), .s_wdata(bwd[j]), .s_wmask(bwm[j]),
        .s_rvalid(brv[j]), .s_rdata(brd[j]), .s_wdone(bwdone[j]), .s_err(berr[j]),
        .m_cvalid(cv), .m_cready(cr), .m_cwe(cwe), .m_caddr(ca), .m_wvalid(wv), .m_wready(wr),
        .m_wdata(wd), .m_wmask(wm), .m_rvalid(rv), .m_rdata(rd), .m_wdone(wdone), .m_err(err));
      otpu_native_reg #(.CB(CB), .CAW(CAW)) u_rx (.clk, .rst,
        .s_cvalid(cv), .s_cready(cr), .s_cwe(cwe), .s_caddr(ca), .s_wvalid(wv), .s_wready(wr),
        .s_wdata(wd), .s_wmask(wm), .s_rvalid(rv), .s_rdata(rd), .s_wdone(wdone), .s_err(err),
        .m_cvalid(ncv[j]), .m_cready(ncr[j]), .m_cwe(ncwe[j]), .m_caddr(nca[j]),
        .m_wvalid(nwv[j]), .m_wready(nwr[j]), .m_wdata(nwd[j]), .m_wmask(nwm[j]),
        .m_rvalid(nrv[j]), .m_rdata(nrd[j]), .m_wdone(nwdone[j]), .m_err(nerr[j]));
    end else begin : g_wire
      assign ncv[j] = bcv[j]; assign bcr[j] = ncr[j]; assign ncwe[j] = bcwe[j];
      assign nca[j] = bca[j]; assign nwv[j] = bwv[j]; assign bwr[j] = nwr[j];
      assign nwd[j] = bwd[j]; assign nwm[j] = bwm[j]; assign brv[j] = nrv[j];
      assign brd[j] = nrd[j]; assign bwdone[j] = nwdone[j]; assign berr[j] = nerr[j];
    end
  end

  for (genvar s = 0; s < S; s++) begin : g_slice
    localparam int C = NCH * s;
    if (MEM_NATIVE == 2) begin : g_ch
      otpu_chmem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1), .SID(s)) u_mem (
        .clk, .rst,
        .n_cvalid(ncv[C +: 2]), .n_cready(ncr[C +: 2]), .n_cwe(ncwe[C +: 2]),
        .n_caddr(nca[C +: 2]), .n_wvalid(nwv[C +: 2]), .n_wready(nwr[C +: 2]),
        .n_wdata(nwd[C +: 2]), .n_wmask(nwm[C +: 2]), .n_rvalid(nrv[C +: 2]),
        .n_rdata(nrd[C +: 2]), .n_wdone(nwdone[C +: 2]), .n_err(nerr[C +: 2]), .dump);
      assign nwfree[C +: 2] = '0;
      assign pwe[C +: 2] = '0;
      assign pslot[C +: 2] = '0;
      assign pd[C +: 2] = '0;
      assign pcntg[C +: 2] = '0;
    end else begin : g_native
      assign nerr[C +: NCH] = '0;
      logic [NCH-1:0][CB*8-1:0] mpd;
      otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1), .SID(s), .NCH(NCH), .CB(CB),
                        .CHASH(CHASH), .WFD(NAT_WPOST ? (NAT_WHB ? 16 : 32) : 0),
                        .XPOST(NAT_XPOST), .XSW(XSW), .RHB(NAT_RHB), .WHB(NAT_WHB)) u_mem (
        .clk, .rst, .run_rst(1'b0),     // (no pokeb lines: the host writes through the board)
        .n_cvalid(ncv[C +: NCH]), .n_cready(ncr[C +: NCH]), .n_cwe(ncwe[C +: NCH]),
        .n_caddr(nca[C +: NCH]), .n_csrc(ncsrc[C +: NCH]), .n_cslot(ncslot[C +: NCH]),
        .n_wvalid(nwv[C +: NCH]), .n_wready(nwr[C +: NCH]),
        .n_wdata(nwd[C +: NCH]), .n_wmask(nwm[C +: NCH]), .n_rvalid(nrv[C +: NCH]),
        .n_rdata(nrd[C +: NCH]), .n_wdone(nwdone[C +: NCH]), .n_wfree(nwfree[C +: NCH]),
        .p_we(pwe[C +: NCH]), .p_slot(pslot[C +: NCH]), .p_d(mpd), .p_cntg(pcntg[C +: NCH]), .dump);
      for (genvar c = 0; c < NCH; c++) begin : g_pd
        assign pd[C + c] = (D*8)'(mpd[c]);     // D = CB at NAT_XPOST (checked)
      end
    end
  end

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  // A request in a slice's SW, port A or port B output register, or read data in its port B
  // input register (otpu_slice), is in neither its units nor its adapter: the control block (STATUS
  // WR_IDLE, go_ok) must not see the slice's writes all taken, or its reads all back, then, nor
  // the slice's loader its reads all back (rd_idle_i, its start)
  for (genvar s = 0; s < S; s++) begin : g_swchk
    logic sw_v, a_v, b_v, b_w, b_rv, wi, ri, li;
    if (s == 0) begin : g_0
      assign sw_v = dut.u_slice.sw_v;
      assign a_v = dut.u_slice.a_req;
      assign b_v = dut.u_slice.b_req; assign b_w = dut.u_slice.b_we;
      assign b_rv = dut.u_slice.b_rvalid_i;
      assign li = dut.u_slice.rd_idle_i;
    end else begin : g_s
      assign sw_v = dut.g_slice[s].u_slice.sw_v;
      assign a_v = dut.g_slice[s].u_slice.a_req;
      assign b_v = dut.g_slice[s].u_slice.b_req; assign b_w = dut.g_slice[s].u_slice.b_we;
      assign b_rv = dut.g_slice[s].u_slice.b_rvalid_i;
      assign li = dut.g_slice[s].u_slice.rd_idle_i;
    end
    assign wi = dut.q_wr_idle[s];
    assign ri = dut.q_rd_idle[s];
    always @(posedge clk)
      if (!rst) begin
        if (wi && sw_v)
          $fatal(1, "tb_board: slice %0d write idle with a write in its SW register", s);
        if (wi && b_v && b_w)
          $fatal(1, "tb_board: slice %0d write idle with a write in its port B register", s);
        if (ri && a_v)
          $fatal(1, "tb_board: slice %0d read idle with a read in its port A register", s);
        if (ri && b_v && !b_w)
          $fatal(1, "tb_board: slice %0d read idle with a read in its port B register", s);
        if (ri && b_rv)
          $fatal(1, "tb_board: slice %0d read idle with read data in its port B input register", s);
        if (li && ((b_v && !b_w) || b_rv))
          $fatal(1, "tb_board: slice %0d's loader sees its reads back with one in port B's registers", s);
        if (li && a_v)
          $fatal(1, "tb_board: slice %0d's loader sees its reads back with one in port A's register", s);
      end
  end

  // port A's held beats (otpu_native_dram: the reused beat al_v, the runs pv) do not outlive a
  // flush (a_flush: a run's reset, a program load, a WAITW's word holding, where the host may have
  // written): the cycle after one, none is held (a read taken in it is decided with the old
  // state, its beat dropped with the rest)
  for (genvar s = 0; s < S; s++) begin : g_pachk
    logic fl, fl_q, al_v;
    logic [NCH-1:0] pv;
    if (s == 0) begin : g_0
      assign fl = dut.x_rst[0] || dut.x_ld_start[0] || dut.a_inval;
      assign al_v = dut.u_mem.al_v; assign pv = dut.u_mem.pv;
    end else begin : g_s
      assign fl = dut.x_rst[s] || dut.x_ld_start[s] || dut.g_slice[s].a_inval;
      assign al_v = dut.g_slice[s].u_mem.al_v; assign pv = dut.g_slice[s].u_mem.pv;
    end
    always @(posedge clk) begin
      if (!rst && fl_q && (al_v || pv != '0))
        $fatal(1, "tb_board: slice %0d's port A beats outlived a flush", s);
      fl_q <= !rst && fl;
    end
  end

  // Handshakes are driven and sampled on the falling edge (race-free): a ready seen there
  // means the transfer happens at the next rising edge. The write's ready depends on its valid
  // (combinationally), so it is sampled a little after the valid is driven -- otherwise the
  // stale ready sends the write twice (harmless for most registers, not for SNAP).
  task automatic lwrite(input logic [11:0] a, input logic [31:0] v);
    @(negedge clk);
    awaddr = a; wdata = v; awvalid = 1'b1; wvalid = 1'b1;
    #1;
    while (!(awready && wready)) begin
      @(negedge clk);
      #1;
    end
    @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) @(negedge clk);
  endtask

  task automatic lread(input logic [11:0] a, output logic [31:0] v);
    @(negedge clk);
    araddr = a; arvalid = 1'b1;
    while (!arready) @(negedge clk);
    @(negedge clk);
    arvalid = 1'b0;
    while (!rvalid) @(negedge clk);
    v = rdata;
  endtask

  string dir, line, op;
  integer fd, n;
  longint max_cycles = 64'd1 << 40;
  int mrst_c = 0;
  initial begin
    logic [31:0] a, m, v, r;
    if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= missing");
    void'($value$plusargs("max_cycles=%d", max_cycles));
    void'($value$plusargs("mrst_c=%d", mrst_c));
    repeat (4) @(negedge clk);
    rst = 1'b0;
    fd = $fopen($sformatf("%s/host.txt", dir), "r");
    if (fd == 0) $fatal(1, "no host script");
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%s", op);
      if (n != 1) break;
      case (op)
        "W": begin
          n = $fscanf(fd, "%h %h", a, v);
          lwrite(a[11:0], v);
        end
        "P": begin
          n = $fscanf(fd, "%h %h %h", a, m, v);
          do begin
            lread(a[11:0], r);
            if (cyc > max_cycles) begin
              $display("TIMEOUT polling %h (%h)", a, r);
              $finish;
            end
          end while ((r & m) != v);
        end
        "R": begin
          n = $fscanf(fd, "%h", a);
          lread(a[11:0], r);
          $display("REG %h %h", a, r);
        end
        "C": begin
          n = $fscanf(fd, "%h", v);
          if (mrst_c > 0 && v > mrst_c + 4) begin
            repeat (mrst_c) @(posedge clk);
            @(negedge clk) ctl_mrst = 1'b1;
            repeat (4) @(negedge clk);
            ctl_mrst = 1'b0;
            repeat (v - mrst_c - 4) @(posedge clk);
          end else repeat (v) @(posedge clk);
        end
        default: $fatal(1, "bad host op %s", op);
      endcase
    end
    $fclose(fd);
    @(negedge clk);
    dump = 1'b1;                 // exactly one rising edge sees it
    @(negedge clk);
    dump = 1'b0;
    @(negedge clk);
    $display("DONE cycles=%0d", cyc);
    $finish;
  end
endmodule
