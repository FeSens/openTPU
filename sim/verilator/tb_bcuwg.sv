// Board-level simulation of the BCU1525 at WG 8 (targets/bcu1525-wg.toml): otpu_bcu_wg_sys (the
// accelerator, otpu_board at one slice of NCH = 8 channels; its bridges, otpu_hbm_ch, and the
// DIMMs' pairs, otpu_mig_pair), each pair through its DIMM's otpu_mig_hmux (the top's; its host
// side idle) in front of models of the four DIMMs' AXI4 ports (otpu_dimm_model: the clock
// converter and the MIG behind each hmux) holding the channels' physical images
// (ch0.bin .. ch7.bin; DIMM d holds channels 2d and 2d + 1), driven by tb_board's host script
// (+dir=<d>, <d>/host.txt; opentpu/host/board.py SimTransport with bcuwg, which gives otpu_board's
// parameters as OTPU_BOARD, defined in a source before this one: opentpu.host.boardmacro board_defines)
// on the control port (AXI4-Lite in the core clock, as tb_board's: the block design's crossing
// and the top's gates are not modelled). Script lines:
//   W <addr> <value>          AXI-Lite write
//   P <addr> <mask> <value>   poll until (read(addr) & mask) == value
//   R <addr>                  read, printed as "REG <addr> <value>"
//   C <cycles>                wait (core cycles)
// Numbers are hex. At the end the channel images are dumped (ch<c>_out.bin; the beats past WORDS's
// in ch<c>_hi_out.txt, ch<c>_hi.txt's in: otpu_dimm_model). CAW: the channels' beat address bits
// (otpu_bcu_wg_sys's: 22 512 MiB a channel, 25 4 GiB, 26 8 GiB), the DIMMs' addresses CAW + 8 bits;
// WORDS stays the dense image's size (the beats above it the models hold sparsely).
// S 2 (targets/bcu1525-s2.toml): two slices of NCH channels, four channels a DIMM (otpu_bcu_wg_sys
// S, its DIMM slots on the DIMMs by DMAP); DIMM d's model holds the channels of the slot on it
// (ch<NC i>.bin .. : board channel g = slice g / NCH's channel g mod NCH), the DIMMs' addresses
// CAW + 9 bits. WORDS is still a slice's.
// Two clocks (+cp= / +ap=: half periods in ps; default 5000 / 1666: the core 100, aclk 300 MHz),
// clk_f at twice the core's, rising with it. The DIMM models' timing: +dimm_* (otpu_dimm_model).
module tb_bcuwg;
  parameter int S        = 1;             // slices (otpu_bcu_wg_sys S)
  parameter int NCH      = 8;
  parameter int WORDS    = 1 << 20;       // a slice's logical memory words (its NCH channels)
  parameter int CORE_KHZ = 100000;
  parameter logic [4*S*NCH-1:0] RLY = '0; // channel j's bridge pipeline N (otpu_bcu_wg_sys)
  parameter logic [7:0] DMAP = 8'hE4;     // the DIMM slots' DIMMs (otpu_bcu_wg_sys)
  parameter int MWAIT    = 8;             // otpu_bcu_wg_sys's: a partial burst's wait,
  parameter int ARD      = 32;            // the bridges' read credits,
  parameter int PRD      = 32;            // the pairs' R FIFOs (512-bit beats)
  parameter int CAW      = 22;            // the channels' beat address bits (otpu_bcu_wg_sys)
  parameter int BHW      = 0;             // the pairs' bank-half window (otpu_mig_pair; 0: none)

  localparam int ND = NCH / 2, CB = 128, NC = 2 * S, DAW = CAW + 7 + $clog2(NC);
  // the DIMM slot on DIMM d (its channels NC slot ..)
  function automatic int dimm_slot(input int d);
    for (int i = 0; i < ND; i++) if (int'(DMAP[2 * i +: 2]) == d) return i;
    return 0;
  endfunction
  localparam int BEATS = WORDS / NCH / 8;          // a channel's 32-byte beats
  int cp = 5000, ap = 1666;
  logic cclk = 1'b0, cclk_f = 1'b0, aclk = 1'b0, dump = 1'b0;
  initial begin
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("ap=%d", ap));
  end
  always #(ap) aclk = ~aclk;
  initial forever begin                   // clk_f rises with each edge of the core clock
    #(cp / 2) cclk_f = 1'b0;
    #(cp - cp / 2) begin cclk = ~cclk; cclk_f = 1'b1; end
  end
  logic rst = 1'b1, arst = 1'b1;
  logic [ND-1:0] calib = '0;
  longint cyc = 0, acyc = 0;
  always @(posedge cclk) cyc <= cyc + 1;
  always @(posedge aclk) begin
    acyc <= acyc + 1;
    if (acyc == 40) arst <= 1'b0;
    if (acyc == 200) calib <= '1;
  end

  // ---- the control port (core clock)
  logic [11:0] awaddr = '0, araddr = '0;
  logic [31:0] wdata = '0, rdata;
  logic        awvalid = 1'b0, wvalid = 1'b0, arvalid = 1'b0;
  logic        awready, wready, bvalid, arready, rvalid;
  logic [1:0]  bresp, rresp;
  // ---- the pairs' ports (into the hmuxes)
  logic [ND-1:0]        m_awvalid, m_awready, m_wvalid, m_wready, m_wlast, m_bvalid, m_bready;
  logic [ND-1:0]        m_arvalid, m_arready, m_rvalid, m_rready, m_rlast;
  logic [ND-1:0][3:0]   m_awid, m_arid, m_bid, m_rid;
  logic [ND-1:0][DAW-1:0] m_awaddr, m_araddr;
  logic [ND-1:0][7:0]   m_awlen, m_arlen;
  logic [ND-1:0][511:0] m_wdata, m_rdata;
  logic [ND-1:0][63:0]  m_wstrb;
  logic [ND-1:0][1:0]   m_bresp, m_rresp;
  // ---- the DIMMs' ports (the hmuxes' m_*)
  logic [ND-1:0]        x_awvalid, x_awready, x_wvalid, x_wready, x_wlast, x_bvalid, x_bready;
  logic [ND-1:0]        x_arvalid, x_arready, x_rvalid, x_rready, x_rlast;
  logic [ND-1:0][3:0]   x_awid, x_arid, x_bid, x_rid;
  logic [ND-1:0][DAW-1:0] x_awaddr, x_araddr;
  logic [ND-1:0][7:0]   x_awlen, x_arlen;
  logic [ND-1:0][511:0] x_wdata, x_rdata;
  logic [ND-1:0][63:0]  x_wstrb;
  logic [ND-1:0][1:0]   x_bresp, x_rresp;
  logic [2:0]           led;

  // (the FPGA top's copies at S 2: the bridges' W rings' enables and read selects, WCP / RCP 4;
  // the pairs' W order's head, WOC 4, and R stage, PRS; the hmuxes' rings' read selects and
  // enables, RCP / WCP 8)
  otpu_bcu_wg_sys #(.S(S), .NCH(NCH), .CB(CB), .CAW(CAW), .CORE_KHZ(CORE_KHZ), .RLY(RLY),
                   .DMAP(DMAP), .MWAIT(MWAIT), .ARD(ARD), .PRD(PRD), .BHW(BHW),
                   .WCP(S == 2 ? 4 : 0), .RCP(S == 2 ? 4 : 0), .WOC(S == 2 ? 4 : 1),
                   .PRS(S == 2)) dut (
    .clk(cclk), .clk_f(cclk_f), .rst, .ctl_mrst(1'b0), .aclk, .arst({ND{arst}}), .calib,
    .temp(12'd0), .led,
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(1'b1),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(1'b1),
    .m_awvalid, .m_awready, .m_awid, .m_awaddr, .m_awlen, .m_wvalid, .m_wready, .m_wdata,
    .m_wstrb, .m_wlast, .m_bvalid, .m_bready, .m_bid, .m_bresp, .m_arvalid, .m_arready, .m_arid,
    .m_araddr, .m_arlen, .m_rvalid, .m_rready, .m_rid, .m_rdata, .m_rresp, .m_rlast);

  // the dump request crosses into aclk as a level the models see for some edges; they write
  // their images at the first
  logic adump = 1'b0, adump_q = 1'b0;
  always @(posedge aclk) begin adump <= dump; adump_q <= adump; end
  int     merr [ND];
  longint mrd [ND], mwr [ND];
  for (genvar d = 0; d < ND; d++) begin : g_dimm
    otpu_mig_hmux #(.IDW(4), .AW(DAW), .HB(S == 1 ? 2 : 0), .RCP(S == 2 ? 8 : 0),
                    .WCP(S == 2 ? 8 : 0)) u_hmux (.clk(aclk), .rst(arst),
      .p_awvalid(m_awvalid[d]), .p_awready(m_awready[d]), .p_awid(m_awid[d]),
      .p_awaddr(m_awaddr[d]), .p_awlen(m_awlen[d]), .p_wvalid(m_wvalid[d]),
      .p_wready(m_wready[d]), .p_wdata(m_wdata[d]), .p_wstrb(m_wstrb[d]), .p_wlast(m_wlast[d]),
      .p_bvalid(m_bvalid[d]), .p_bready(m_bready[d]), .p_bid(m_bid[d]), .p_bresp(m_bresp[d]),
      .p_arvalid(m_arvalid[d]), .p_arready(m_arready[d]), .p_arid(m_arid[d]),
      .p_araddr(m_araddr[d]), .p_arlen(m_arlen[d]), .p_rvalid(m_rvalid[d]),
      .p_rready(m_rready[d]), .p_rid(m_rid[d]), .p_rdata(m_rdata[d]), .p_rresp(m_rresp[d]),
      .p_rlast(m_rlast[d]),
      .h_awvalid(1'b0), .h_awready(), .h_awaddr('0), .h_awlen('0), .h_wvalid(1'b0), .h_wready(),
      .h_wdata('0), .h_wstrb('0), .h_wlast(1'b0), .h_bvalid(), .h_bready(1'b1), .h_bresp(),
      .h_arvalid(1'b0), .h_arready(), .h_araddr('0), .h_arlen('0), .h_rvalid(), .h_rready(1'b1),
      .h_rdata(), .h_rresp(), .h_rlast(),
      .m_awvalid(x_awvalid[d]), .m_awready(x_awready[d]), .m_awid(x_awid[d]),
      .m_awaddr(x_awaddr[d]), .m_awlen(x_awlen[d]), .m_wvalid(x_wvalid[d]),
      .m_wready(x_wready[d]), .m_wdata(x_wdata[d]), .m_wstrb(x_wstrb[d]), .m_wlast(x_wlast[d]),
      .m_bvalid(x_bvalid[d]), .m_bready(x_bready[d]), .m_bid(x_bid[d]), .m_bresp(x_bresp[d]),
      .m_arvalid(x_arvalid[d]), .m_arready(x_arready[d]), .m_arid(x_arid[d]),
      .m_araddr(x_araddr[d]), .m_arlen(x_arlen[d]), .m_rvalid(x_rvalid[d]),
      .m_rready(x_rready[d]), .m_rid(x_rid[d]), .m_rdata(x_rdata[d]), .m_rresp(x_rresp[d]),
      .m_rlast(x_rlast[d]));
    otpu_dimm_model #(.BEATS(BEATS), .DAW(DAW), .DIMM(d), .NC(NC), .CH0(NC * dimm_slot(d)),
                      .BHW(BHW)) u_dimm (
      .clk(aclk), .rst(arst),
      .dump(adump && !adump_q),
      .awvalid(x_awvalid[d]), .awready(x_awready[d]), .awid(x_awid[d]), .awaddr(x_awaddr[d]),
      .awlen(x_awlen[d]), .wvalid(x_wvalid[d]), .wready(x_wready[d]), .wdata(x_wdata[d]),
      .wstrb(x_wstrb[d]), .wlast(x_wlast[d]), .bvalid(x_bvalid[d]), .bready(x_bready[d]),
      .bid(x_bid[d]), .bresp(x_bresp[d]), .arvalid(x_arvalid[d]), .arready(x_arready[d]),
      .arid(x_arid[d]), .araddr(x_araddr[d]), .arlen(x_arlen[d]), .rvalid(x_rvalid[d]),
      .rready(x_rready[d]), .rid(x_rid[d]), .rdata(x_rdata[d]), .rresp(x_rresp[d]),
      .rlast(x_rlast[d]), .nrd(mrd[d]), .nwr(mwr[d]), .nerr(merr[d]));
  end

  // Handshakes driven and sampled on the falling edge, as tb_board's
  task automatic lwrite(input logic [11:0] a, input logic [31:0] v);
    @(negedge cclk);
    awaddr = a; wdata = v; awvalid = 1'b1; wvalid = 1'b1;
    #1;
    while (!(awready && wready)) begin
      @(negedge cclk);
      #1;
    end
    @(negedge cclk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) @(negedge cclk);
  endtask
  task automatic lread(input logic [11:0] a, output logic [31:0] v);
    @(negedge cclk);
    araddr = a; arvalid = 1'b1;
    while (!arready) @(negedge cclk);
    @(negedge cclk);
    arvalid = 1'b0;
    while (!rvalid) @(negedge cclk);
    v = rdata;
  endtask

  task automatic dimm_check(input string when);
    for (int d = 0; d < ND; d++)
      if (merr[d] != 0) $fatal(1, "tb_bcuwg: %s: DIMM %0d: %0d errors", when, d, merr[d]);
    if (|dut.n_err) $fatal(1, "tb_bcuwg: %s: bridge errors %b", when, dut.n_err);
  endtask

  string dir, op;
  integer fd, n;
  longint max_cycles = 64'd1 << 40;
  initial begin
    logic [31:0] a, m, v, r;
    if (!$value$plusargs("dir=%s", dir)) $fatal(1, "+dir= missing");
    void'($value$plusargs("max_cycles=%d", max_cycles));
    repeat (4) @(negedge cclk);
    rst = 1'b0;
    // the script starts once the DIMMs are calibrated (the card's host waits for STATUS CALIB):
    // the control block ignores LOAD and RUN until then
    while (calib != '1) @(negedge cclk);
    repeat (16) @(negedge cclk);            // (through otpu_board's synchronizers)
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
          repeat (v) @(posedge cclk);
        end
        default: $fatal(1, "bad host op %s", op);
      endcase
    end
    $fclose(fd);
    dimm_check("end");
    @(negedge cclk);
    dump = 1'b1;
    repeat (4) @(negedge aclk);
    dump = 1'b0;
    repeat (4) @(negedge aclk);
    dimm_check("the dump");
    for (int d = 0; d < ND; d++)
      $display("dimm %0d: %0d R beats, %0d W beats", d, mrd[d], mwr[d]);
    $display("DONE cycles=%0d", cyc);
    $finish;
  end
endmodule
