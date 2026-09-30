// Simulation model of the board memory behind the two channels' native ports n_*: per channel the
// board's bridge, otpu_mem_ch, in front of a model of LiteDRAM's two native ports (otpu_ldn_model),
// both channels in one controller clock (LiteDRAM's sys). otpu_native_mem's parameters and ports,
// so the board model (tb_board) takes either; this one runs the RTL between the accelerator and
// the controller as well.
//
// Clocks: the core clock is the caller's (tb_board: a period of 10 time units). A controller
// clock (133.33 MHz against a 100 MHz core: DDR3-1066) has half periods of 4, 4, 4, 3 units in
// turn (7.5 on average, so its edges drift against the core's and meet them every 30 units).
// XDMA's clock (125 MHz) has half periods of 4. XDMA's side of each bridge is idle; its reset and
// the controllers' follow the caller's rst through two flip-flops of their clocks.
// The controller models' arguments: +axi_stall=P (ready withheld, percent), +axi_lat=N (read
// latency), +ldn_busy=N (LiteDRAM: refresh / row-change pauses, per mille), +axi_seed=N.
// PHYS = 1 only: the files are the channels' own memories, as otpu_native_mem's with PHYS = 1
// (ch<c>.bin in, big-endian words; ch<c>_out.bin out on dump, little-endian), WORDS / 2 words each.
module otpu_chmem #(
  parameter int WORDS = 1 << 18,
  parameter int PHYS  = 0,
  parameter int LAT   = 20,              // minimum read latency (controller cycles)
  parameter int SID   = 0,               // (otpu_native_mem's; unused)
  parameter bit CHASH = 1'b1             // (otpu_native_mem's; unused: the images are physical)
) (
  input  logic                  clk,
  input  logic                  rst,
  input  logic [1:0]            n_cvalid,
  output logic [1:0]            n_cready,
  input  logic [1:0]            n_cwe,
  input  logic [1:0][24:0]      n_caddr,
  input  logic [1:0]            n_wvalid,
  output logic [1:0]            n_wready,
  input  logic [1:0][511:0]     n_wdata,
  input  logic [1:0][63:0]      n_wmask,
  output logic [1:0]            n_rvalid,
  output logic [1:0][511:0]     n_rdata,
  output logic [1:0][15:0]      n_wdone,
  output logic [1:0]            n_err,
  input  logic                  dump
);
  localparam int BEATS = WORDS / 32;     // per channel: WORDS / 2 words, 16 a beat
  initial if (PHYS != 1) $fatal(1, "otpu_chmem: PHYS = 1 only (the channels' own images)");

  logic uclk = 1'b0, xclk = 1'b0;
  always begin
    #4 uclk = 1'b1; #4 uclk = 1'b0; #4 uclk = 1'b1; #3 uclk = 1'b0;
  end
  always #4 xclk = ~xclk;
  logic xrst = 1'b1, xr1 = 1'b1;
  always @(posedge xclk) {xrst, xr1} <= {xr1, rst};

  initial begin
    int seed;
    if ($value$plusargs("axi_seed=%d", seed)) void'($urandom(seed));
  end

  for (genvar c = 0; c < 2; c++) begin : g_ch
    logic         urst = 1'b1, ur1 = 1'b1;
    logic [1:0]   cv, cr, cwe, wv, wr, rv;          // the channel's two ports
    logic [1:0][24:0]  ca;
    logic [1:0][511:0] wd, rd;
    logic [1:0][63:0]  we;
    always @(posedge uclk) {urst, ur1} <= {ur1, rst};
    otpu_mem_ch #(.XIDW(4)) u_ch (
      .clk, .rst,
      .n_cvalid(n_cvalid[c]), .n_cready(n_cready[c]), .n_cwe(n_cwe[c]), .n_caddr(n_caddr[c]),
      .n_wvalid(n_wvalid[c]), .n_wready(n_wready[c]), .n_wdata(n_wdata[c]), .n_wmask(n_wmask[c]),
      .n_rvalid(n_rvalid[c]), .n_rdata(n_rdata[c]), .n_wdone(n_wdone[c]), .n_err(n_err[c]),
      .xclk, .xrst,
      .x_awvalid(1'b0), .x_awready(), .x_awid(4'h0), .x_awaddr(32'h0), .x_awlen(8'h0),
      .x_wvalid(1'b0), .x_wready(), .x_wdata(128'h0), .x_wstrb(16'h0), .x_wlast(1'b0),
      .x_bvalid(), .x_bready(1'b1), .x_bid(), .x_bresp(),
      .x_arvalid(1'b0), .x_arready(), .x_arid(4'h0), .x_araddr(32'h0), .x_arlen(8'h0),
      .x_rvalid(), .x_rready(1'b1), .x_rid(), .x_rdata(), .x_rresp(), .x_rlast(),
      .uclk, .urst,
      .c_cmd_valid(cv), .c_cmd_ready(cr), .c_cmd_we(cwe), .c_cmd_addr(ca),
      .c_wdata_valid(wv), .c_wdata_ready(wr), .c_wdata_data(wd), .c_wdata_we(we),
      .c_rdata_valid(rv), .c_rdata_data(rd));
    otpu_ldn_model #(.BEATS(BEATS), .LAT(LAT), .CH(c), .IMG(1'b1)) u_mem (
      .clk(uclk), .rst(urst),
      .c_cmd_valid(cv), .c_cmd_ready(cr), .c_cmd_we(cwe), .c_cmd_addr(ca),
      .c_wdata_valid(wv), .c_wdata_ready(wr), .c_wdata_data(wd), .c_wdata_we(we),
      .c_rdata_valid(rv), .c_rdata_data(rd), .dump);
  end
endmodule
