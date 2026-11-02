// Board-level simulation: otpu_board (control registers, S slices, each with the native DRAM
// adapter) in front of the channels' memory holding their physical images (ch0.bin, ch1.bin; slice
// s's are ch<2s>.bin and ch<2s+1>.bin): MEM_NATIVE 2 (the
// default) the build's channels, otpu_mem_ch in front of a LiteDRAM native-port model
// (otpu_chmem); 1 the native memory model alone (otpu_native_mem) -- driven by a host script
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
// flags (default 0; STATUS ECC_DED).
module tb_board;
  parameter int S          = 1;           // slices, two channels each
  parameter int WORDS      = 1 << 20;     // logical memory words (a slice's two channels)
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
  parameter int ULANES     = LANES;
  parameter int WIN        = 16;
  parameter int LAT        = 20;
  parameter int CORE_KHZ   = 100000;
  parameter logic [31:0] BUILD_ID = 32'h0B0A_4D00;
  parameter int DDR_MTS    = 0;
  parameter int TRACE_DEPTH = 16384;
  parameter bit COLL_GATHER = 1'b1;       // 0: the BAR-only collective unit (otpu_coll_bar) at
                                          // S = 1 (rtlsim.BOARD_UARCH); S > 1 has GATHER
  parameter int TRACE_QD   = 32;
  parameter int PQ_WIN     = 1024;
  parameter bit DSTEP      = 1'b1;
  parameter int MEM_NATIVE = 2;            // the channels: 2 otpu_chmem (otpu_mem_ch and a LiteDRAM
                                           // model), 1 otpu_native_mem
  parameter bit HOSTCAL    = 1'b0;         // CAPS bit27 (the host calibrates the controllers)
  parameter bit SEQ_WC     = 1'b0;    // EXPERIMENTAL: otpu_seq's wait entries (WAITC, docs/seq_vpu.md; CFG_UARCH bit24)
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
  parameter bit DUAL       = 1'b0;         // S = 1 on four channels (otpu_board DUAL)
  localparam int NCPS      = DUAL ? 4 : 2; // channels a slice

  logic clk = 1'b0, clk_f = 1'b0, rst = 1'b1, dump = 1'b0;
  // TMEM_PMP 1: clk_f (2 x clk) rises with each edge of clk, in clk's own process (as tb_top);
  // clk's edges stay where they are (the memory models' clocks are absolute)
  if (TMEM_PMP == 1) begin : g_clkf
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

  // native channels (slice s on 2s and 2s + 1; DUAL: 0 .. 3)
  logic [NCPS*S-1:0] ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [NCPS*S-1:0][24:0] nca;
  logic [NCPS*S-1:0][511:0] nwd, nrd;
  logic [NCPS*S-1:0][63:0] nwm;
  logic [NCPS*S-1:0][15:0] nwdone;
  logic [NCPS*S-1:0] nerr;

  // I2C: wired AND of the board's drive-low bits and the outside's holds, pulled up
  logic [3:0] i2c_lo;
  logic [4:0] i2c_hold = '0;
  initial void'($value$plusargs("i2c_hold=%h", i2c_hold));
  logic [NCPS*S-1:0] calib = '1, ded = '0;
  initial begin
    void'($value$plusargs("calib=%h", calib));
    void'($value$plusargs("ded=%h", ded));
  end

  otpu_board #(.S(S), .D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
               .IMEM_WORDS(IMEM_WORDS), .LANES(LANES), .VPU_CL(VPU_CL), .MXU_IMPL(MXU_IMPL), .MXU_E(MXU_E), .ULANES(ULANES), .WIN(WIN), .CORE_KHZ(CORE_KHZ),
               .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .TRACE_QD(TRACE_QD),
               .PQ_WIN(PQ_WIN), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL), .SEQ_WC(SEQ_WC), .PLATFORM(PLATFORM),
               .ARCH_HASH(ARCH_HASH), .COLL_GATHER(COLL_GATHER || S > 1), .TMEM_IMPL(TMEM_IMPL),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH), .TMEM_RPL(TMEM_RPL),
               .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST), .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK),
               .TMEM_SDM(TMEM_SDM), .DUAL(DUAL)) dut (
    .clk, .clk_f, .rst, .ctl_mrst(1'b0), .calib, .ded, .temp(TEMP), .led,
    .i2c_lo, .i2c_pin(~({1'b0, i2c_lo} | i2c_hold)),
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(bready),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(rready),
    .n_cvalid(bcv), .n_cready(bcr), .n_cwe(bcwe), .n_caddr(bca), .n_wvalid(bwv), .n_wready(bwr),
    .n_wdata(bwd), .n_wmask(bwm), .n_rvalid(brv), .n_rdata(brd), .n_wdone(bwdone), .n_err(berr));

  // the board's side of the channels, and the register slices where NREG asks for them
  logic [NCPS*S-1:0] bcv, bcr, bcwe, bwv, bwr, brv, berr;
  logic [NCPS*S-1:0][24:0] bca;
  logic [NCPS*S-1:0][511:0] bwd, brd;
  logic [NCPS*S-1:0][63:0] bwm;
  logic [NCPS*S-1:0][15:0] bwdone;
  for (genvar j = 0; j < NCPS * S; j++) begin : g_nreg
    if (NREG >> j & 1) begin : g_reg
      logic cv, cr, cwe, wv, wr, rv, err;
      logic [24:0] ca;
      logic [511:0] wd, rd;
      logic [63:0] wm;
      logic [15:0] wdone;
      otpu_native_reg u_tx (.clk, .rst,
        .s_cvalid(bcv[j]), .s_cready(bcr[j]), .s_cwe(bcwe[j]), .s_caddr(bca[j]),
        .s_wvalid(bwv[j]), .s_wready(bwr[j]), .s_wdata(bwd[j]), .s_wmask(bwm[j]),
        .s_rvalid(brv[j]), .s_rdata(brd[j]), .s_wdone(bwdone[j]), .s_err(berr[j]),
        .m_cvalid(cv), .m_cready(cr), .m_cwe(cwe), .m_caddr(ca), .m_wvalid(wv), .m_wready(wr),
        .m_wdata(wd), .m_wmask(wm), .m_rvalid(rv), .m_rdata(rd), .m_wdone(wdone), .m_err(err));
      otpu_native_reg u_rx (.clk, .rst,
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
    localparam int C = NCPS * s;
    if (MEM_NATIVE == 2) begin : g_ch
      // (DUAL: a pair of channels each, as two slices' files: ch<4s + 2p + c>)
      for (genvar q = 0; q < NCPS / 2; q++) begin : g_q
        otpu_chmem #(.WORDS(WORDS / (NCPS / 2)), .LAT(LAT), .PHYS(1), .SID(NCPS / 2 * s + q)) u_mem (
          .clk, .rst,
          .n_cvalid(ncv[C + 2*q +: 2]), .n_cready(ncr[C + 2*q +: 2]), .n_cwe(ncwe[C + 2*q +: 2]),
          .n_caddr(nca[C + 2*q +: 2]), .n_wvalid(nwv[C + 2*q +: 2]), .n_wready(nwr[C + 2*q +: 2]),
          .n_wdata(nwd[C + 2*q +: 2]), .n_wmask(nwm[C + 2*q +: 2]), .n_rvalid(nrv[C + 2*q +: 2]),
          .n_rdata(nrd[C + 2*q +: 2]), .n_wdone(nwdone[C + 2*q +: 2]), .n_err(nerr[C + 2*q +: 2]),
          .dump);
      end
    end else begin : g_native
      assign nerr[C +: NCPS] = '0;
      otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1), .SID(s), .NC(NCPS)) u_mem (
        .clk, .rst, .run_rst(1'b0),     // (no pokeb lines: the host writes through the board)
        .n_cvalid(ncv[C +: NCPS]), .n_cready(ncr[C +: NCPS]), .n_cwe(ncwe[C +: NCPS]),
        .n_caddr(nca[C +: NCPS]), .n_wvalid(nwv[C +: NCPS]), .n_wready(nwr[C +: NCPS]),
        .n_wdata(nwd[C +: NCPS]), .n_wmask(nwm[C +: NCPS]), .n_rvalid(nrv[C +: NCPS]),
        .n_rdata(nrd[C +: NCPS]), .n_wdone(nwdone[C +: NCPS]), .dump);
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
    logic [1:0] pv;
    if (s == 0) begin : g_0
      assign fl = dut.x_rst[0] || dut.x_ld_start[0] || dut.a_inval;
      if (DUAL) begin : g_d
        assign al_v = dut.g_mem2.u_mem.g_ad[0].u_ad.al_v || dut.g_mem2.u_mem.g_ad[1].u_ad.al_v;
        assign pv = dut.g_mem2.u_mem.g_ad[0].u_ad.pv | dut.g_mem2.u_mem.g_ad[1].u_ad.pv;
      end else begin : g_1
        assign al_v = dut.g_mem.u_mem.al_v; assign pv = dut.g_mem.u_mem.pv;
      end
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
  initial begin
    logic [31:0] a, m, v, r;
    if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= missing");
    void'($value$plusargs("max_cycles=%d", max_cycles));
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
          repeat (v) @(posedge clk);
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
