// Board-level simulation: otpu_board (control registers, slice, DRAM adapter) in front of the
// two-channel AXI memory model holding the channels' physical images (ch0.bin, ch1.bin), driven
// by a host script (+dir=<d>, <d>/host.txt) -- the same register and memory protocol the PCIe
// host driver uses (host/board.py). Script lines:
//   W <addr> <value>          AXI-Lite write
//   P <addr> <mask> <value>   poll until (read(addr) & mask) == value
//   R <addr>                  read, printed as "REG <addr> <value>"
//   C <cycles>                wait
// Numbers are hex. At the end the channel memories are dumped (ch0_out.bin, ch1_out.bin).
module tb_board;
  parameter int WORDS      = 1 << 20;     // logical memory words (both channels)
  parameter int D          = 128;
  parameter int MCOLS      = 2;
  parameter int ACT_BLOCKS = 128;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int IMEM_WORDS = 1 << 15;
  parameter int LANES      = 8;
  parameter int WIN        = 16;
  parameter int LAT        = 20;

  logic clk = 1'b0, rst = 1'b1, dump = 1'b0;
  always #5 clk = ~clk;

  // AXI-Lite
  logic [7:0]  awaddr, araddr;
  logic        awvalid = 0, awready, wvalid = 0, wready, bvalid, bready = 1;
  logic        arvalid = 0, arready, rvalid, rready = 1;
  initial begin awaddr = 0; araddr = 0; wdata = 0; end
  logic [31:0] wdata, rdata;
  logic [1:0]  bresp, rresp;
  logic [2:0]  led;

  // AXI channels
  logic [1:0] awv, awr, awi, wv, wr, bv, br, bi, arv, arr, ari, rv, rr, ri, rl;
  logic [1:0][31:0] awa, ara;
  logic [1:0][511:0] wd, rd;
  logic [1:0][63:0] ws;
  logic [1:0][1:0] bre, rre;
  logic [7:0] unused8 [8];
  logic [2:0] unused3 [8];
  logic [1:0] unused2 [4];
  logic [3:0] unused4 [12];
  logic       unusedl [6];

  otpu_board #(.D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
               .IMEM_WORDS(IMEM_WORDS), .LANES(LANES), .WIN(WIN)) dut (
    .clk, .rst, .calib(2'b11), .led,
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(bready),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(rready),
    .m0_axi_awid(awi[0]), .m0_axi_awaddr(awa[0]), .m0_axi_awlen(unused8[0]),
    .m0_axi_awsize(unused3[0]), .m0_axi_awburst(unused2[0]), .m0_axi_awlock(unusedl[0]),
    .m0_axi_awcache(unused4[0]), .m0_axi_awprot(unused3[1]), .m0_axi_awqos(unused4[1]),
    .m0_axi_awvalid(awv[0]), .m0_axi_awready(awr[0]), .m0_axi_wdata(wd[0]),
    .m0_axi_wstrb(ws[0]), .m0_axi_wlast(unusedl[1]), .m0_axi_wvalid(wv[0]),
    .m0_axi_wready(wr[0]), .m0_axi_bid(bi[0]), .m0_axi_bresp(bre[0]), .m0_axi_bvalid(bv[0]),
    .m0_axi_bready(br[0]), .m0_axi_arid(ari[0]), .m0_axi_araddr(ara[0]),
    .m0_axi_arlen(unused8[1]), .m0_axi_arsize(unused3[2]), .m0_axi_arburst(unused2[1]),
    .m0_axi_arlock(unusedl[2]), .m0_axi_arcache(unused4[2]), .m0_axi_arprot(unused3[3]),
    .m0_axi_arqos(unused4[3]), .m0_axi_arvalid(arv[0]), .m0_axi_arready(arr[0]),
    .m0_axi_rid(ri[0]), .m0_axi_rdata(rd[0]), .m0_axi_rresp(rre[0]), .m0_axi_rlast(rl[0]),
    .m0_axi_rvalid(rv[0]), .m0_axi_rready(rr[0]),
    .m1_axi_awid(awi[1]), .m1_axi_awaddr(awa[1]), .m1_axi_awlen(unused8[2]),
    .m1_axi_awsize(unused3[4]), .m1_axi_awburst(unused2[2]), .m1_axi_awlock(unusedl[3]),
    .m1_axi_awcache(unused4[4]), .m1_axi_awprot(unused3[5]), .m1_axi_awqos(unused4[5]),
    .m1_axi_awvalid(awv[1]), .m1_axi_awready(awr[1]), .m1_axi_wdata(wd[1]),
    .m1_axi_wstrb(ws[1]), .m1_axi_wlast(unusedl[4]), .m1_axi_wvalid(wv[1]),
    .m1_axi_wready(wr[1]), .m1_axi_bid(bi[1]), .m1_axi_bresp(bre[1]), .m1_axi_bvalid(bv[1]),
    .m1_axi_bready(br[1]), .m1_axi_arid(ari[1]), .m1_axi_araddr(ara[1]),
    .m1_axi_arlen(unused8[3]), .m1_axi_arsize(unused3[6]), .m1_axi_arburst(unused2[3]),
    .m1_axi_arlock(unusedl[5]), .m1_axi_arcache(unused4[6]), .m1_axi_arprot(unused3[7]),
    .m1_axi_arqos(unused4[7]), .m1_axi_arvalid(arv[1]), .m1_axi_arready(arr[1]),
    .m1_axi_rid(ri[1]), .m1_axi_rdata(rd[1]), .m1_axi_rresp(rre[1]), .m1_axi_rlast(rl[1]),
    .m1_axi_rvalid(rv[1]), .m1_axi_rready(rr[1]));

  otpu_axi_mem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1)) u_mem (
    .clk, .rst,
    .s_awvalid(awv), .s_awready(awr), .s_awaddr(awa), .s_awid(awi),
    .s_wvalid(wv), .s_wready(wr), .s_wdata(wd), .s_wstrb(ws),
    .s_bvalid(bv), .s_bready(br), .s_bid(bi), .s_bresp(bre),
    .s_arvalid(arv), .s_arready(arr), .s_araddr(ara), .s_arid(ari),
    .s_rvalid(rv), .s_rready(rr), .s_rid(ri), .s_rdata(rd), .s_rresp(rre), .s_rlast(rl),
    .dump);

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  // Handshakes are driven and sampled on the falling edge (race-free): a ready seen there
  // means the transfer happens at the next rising edge.
  task automatic lwrite(input logic [7:0] a, input logic [31:0] v);
    @(negedge clk);
    awaddr = a; wdata = v; awvalid = 1'b1; wvalid = 1'b1;
    while (!(awready && wready)) @(negedge clk);
    @(negedge clk);
    awvalid = 1'b0; wvalid = 1'b0;
    while (!bvalid) @(negedge clk);
  endtask

  task automatic lread(input logic [7:0] a, output logic [31:0] v);
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
          lwrite(a[7:0], v);
        end
        "P": begin
          n = $fscanf(fd, "%h %h %h", a, m, v);
          do begin
            lread(a[7:0], r);
            if (cyc > max_cycles) begin
              $display("TIMEOUT polling %h (%h)", a, r);
              $finish;
            end
          end while ((r & m) != v);
        end
        "R": begin
          n = $fscanf(fd, "%h", a);
          lread(a[7:0], r);
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
