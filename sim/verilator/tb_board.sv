// Board-level simulation: otpu_board (control registers, slice, the native DRAM adapter) in front
// of the channels' memory holding their physical images (ch0.bin, ch1.bin): MEM_NATIVE 2 (the
// default) the build's channels, otpu_mem_ch in front of a LiteDRAM native-port model
// (otpu_chmem); 1 the native memory model alone (otpu_native_mem) -- driven by a host script
// (+dir=<d>, <d>/host.txt) -- the same register and memory protocol the PCIe host driver uses (opentpu/host/board.py). Script lines:
//   W <addr> <value>          AXI-Lite write
//   P <addr> <mask> <value>   poll until (read(addr) & mask) == value
//   R <addr>                  read, printed as "REG <addr> <value>"
//   C <cycles>                wait
// Numbers are hex. At the end the channel memories are dumped (ch0_out.bin, ch1_out.bin).
// +trace prints the slice's trace lines (as tb_top; the hardware trace records the same events).
// The I2C pins are four open-drain lines with pull-ups and nothing else on them; +i2c_hold=<hex>
// holds lines low from the outside (bits 0..3 as I2C_CTRL, bit4 ALERT0).
module tb_board;
  parameter int WORDS      = 1 << 20;     // logical memory words (both channels)
  parameter int D          = 128;
  parameter int MCOLS      = 4;
  parameter int ACT_ROWS   = MCOLS;
  parameter int ACT_BLOCKS = 128;
  parameter int TMEM_WORDS = 1 << 16;
  parameter int IMEM_WORDS = 1 << 15;
  parameter int LANES      = 8;
  parameter int VPU_CL     = (LANES >= 8) ? LANES / 4 : 1;
  parameter int MXU_IMPL   = 2;      // MXU dot product: 2 systolic (docs/mxu_systolic.md), 0 adder tree
  parameter int ULANES     = LANES;
  parameter int WIN        = 16;
  parameter int LAT        = 20;
  parameter int CORE_KHZ   = 100000;
  parameter logic [31:0] BUILD_ID = 32'h0B0A_4D00;
  parameter int DDR_MTS    = 0;
  parameter int TRACE_DEPTH = 16384;
  parameter int TRACE_QD   = 32;
  parameter int PQ_WIN     = 1024;
  parameter bit DSTEP      = 1'b1;
  parameter int MEM_NATIVE = 2;            // the channels: 2 otpu_chmem (otpu_mem_ch and a LiteDRAM
                                           // model), 1 otpu_native_mem
  parameter bit HOSTCAL    = 1'b0;         // CAPS bit27 (the host calibrates the controllers)
  parameter logic [11:0] TEMP = 12'hA1A;  // the XADC code of 45 C

  logic clk = 1'b0, rst = 1'b1, dump = 1'b0;
  always #5 clk = ~clk;

  // AXI-Lite
  logic [11:0] awaddr, araddr;
  logic        awvalid = 0, awready, wvalid = 0, wready, bvalid, bready = 1;
  logic        arvalid = 0, arready, rvalid, rready = 1;
  initial begin awaddr = 0; araddr = 0; wdata = 0; end
  logic [31:0] wdata, rdata;
  logic [1:0]  bresp, rresp;
  logic [2:0]  led;

  // native channels
  logic [1:0] ncv, ncr, ncwe, nwv, nwr, nrv;
  logic [1:0][24:0] nca;
  logic [1:0][511:0] nwd, nrd;
  logic [1:0][63:0] nwm;
  logic [1:0][15:0] nwdone;
  logic [1:0] nerr;
  logic [31:0] xmon [32];                  // no DMA monitors here (otpu_native_sys's XMON)
  for (genvar k = 0; k < 32; k++) begin : g_xmon
    assign xmon[k] = '0;
  end

  // I2C: wired AND of the board's drive-low bits and the outside's holds, pulled up
  logic [3:0] i2c_lo;
  logic [4:0] i2c_hold = '0;
  initial void'($value$plusargs("i2c_hold=%h", i2c_hold));

  otpu_board #(.D(D), .MCOLS(MCOLS), .ACT_ROWS(ACT_ROWS), .ACT_BLOCKS(ACT_BLOCKS), .TMEM_WORDS(TMEM_WORDS),
               .IMEM_WORDS(IMEM_WORDS), .LANES(LANES), .VPU_CL(VPU_CL), .MXU_IMPL(MXU_IMPL), .ULANES(ULANES), .WIN(WIN), .CORE_KHZ(CORE_KHZ),
               .BUILD_ID(BUILD_ID), .DDR_MTS(DDR_MTS), .TRACE_DEPTH(TRACE_DEPTH), .TRACE_QD(TRACE_QD),
               .PQ_WIN(PQ_WIN), .DSTEP(DSTEP), .HOSTCAL(HOSTCAL)) dut (
    .clk, .rst, .calib(2'b11), .temp(TEMP), .led,
    .i2c_lo, .i2c_pin(~({1'b0, i2c_lo} | i2c_hold)),
    .s_ctl_awaddr(awaddr), .s_ctl_awvalid(awvalid), .s_ctl_awready(awready),
    .s_ctl_wdata(wdata), .s_ctl_wstrb(4'hF), .s_ctl_wvalid(wvalid), .s_ctl_wready(wready),
    .s_ctl_bresp(bresp), .s_ctl_bvalid(bvalid), .s_ctl_bready(bready),
    .s_ctl_araddr(araddr), .s_ctl_arvalid(arvalid), .s_ctl_arready(arready),
    .s_ctl_rdata(rdata), .s_ctl_rresp(rresp), .s_ctl_rvalid(rvalid), .s_ctl_rready(rready),
    .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_wvalid(nwv), .n_wready(nwr),
    .n_wdata(nwd), .n_wmask(nwm), .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .n_err(nerr),
    .xmon_snap(), .xmon_clr(), .xmon);

  if (MEM_NATIVE == 2) begin : g_ch
    otpu_chmem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1)) u_mem (
      .clk, .rst,
      .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_wvalid(nwv),
      .n_wready(nwr), .n_wdata(nwd), .n_wmask(nwm), .n_rvalid(nrv), .n_rdata(nrd),
      .n_wdone(nwdone), .n_err(nerr), .dump);
  end else begin : g_native
    assign nerr = '0;
    otpu_native_mem #(.WORDS(WORDS), .LAT(LAT), .PHYS(1)) u_mem (
      .clk, .rst,
      .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_wvalid(nwv),
      .n_wready(nwr), .n_wdata(nwd), .n_wmask(nwm), .n_rvalid(nrv), .n_rdata(nrd),
      .n_wdone(nwdone), .dump);
  end

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

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
