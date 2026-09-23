// Host control registers (AXI4-Lite slave, reached from the host through the PCIe bridge's
// BAR). All in the core clock domain.
//
//   0x00 ID        RO  0x4F545055 ("OTPU")
//   0x04 VERSION   RO  {D[15:0], MCOLS[7:0], LANES[7:0]}
//   0x08 CTRL      RW  bit0 RUN: 1 releases the slice from reset (write 0, then 1, per run)
//                      bit1 LOAD (write 1: copy PROG_N instructions from PROG_ADDR into IMEM;
//                      only while RUN = 0)
//                      bit2 CLEAR: zero the counters
//   0x0C STATUS    RO  bit0 HALTED, bit1 ERROR (illegal instruction), bit2 LOADING,
//                      bit3 WR_IDLE, bit4 AXI_ERR (sticky), bit5 CALIB0, bit6 CALIB1, bit7 RUN
//   0x10 PROG_ADDR RW  program byte address in the slice's DRAM (chunk aligned)
//   0x14 PROG_N    RW  program length in instructions
//   0x18 CYCLES    RO  core cycles since RUN rose, until HALTED (low 32 bits)
//   0x1C CYCLES_HI RO
//   0x20 ICOUNT    RO  instructions retired
//   0x24 B_RD      RO  port B read requests taken (chunks)
//   0x28 B_WR      RO  port B write requests taken
//   0x2C A_RD      RO  port A read requests taken
//   0x30 A_WR      RO  port A write requests taken
//   0x34 B_STALL   RO  cycles a port B request waited for the memory
//   0x38 SCRATCH   RW  (host bring-up test)
module otpu_ctrl #(
  parameter int D     = 128,
  parameter int MCOLS = 2,
  parameter int LANES = 8
) (
  input  logic        clk,
  input  logic        rst,
  // AXI4-Lite slave
  input  logic [7:0]  s_awaddr,
  input  logic        s_awvalid,
  output logic        s_awready,
  input  logic [31:0] s_wdata,
  input  logic [3:0]  s_wstrb,
  input  logic        s_wvalid,
  output logic        s_wready,
  output logic [1:0]  s_bresp,
  output logic        s_bvalid,
  input  logic        s_bready,
  input  logic [7:0]  s_araddr,
  input  logic        s_arvalid,
  output logic        s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  output logic        s_rvalid,
  input  logic        s_rready,
  // core
  output logic        run,
  output logic        ld_start,
  output logic [31:0] ld_addr,
  output logic [31:0] ld_n,
  input  logic        ld_busy,
  input  logic        halted,
  input  logic        error,
  input  logic [31:0] icount,
  input  logic        wr_idle,
  input  logic        axi_err,
  input  logic [1:0]  calib,
  input  logic        b_rd, b_wr, a_rd, a_wr, b_wait
);
  logic [63:0] cycles;
  logic [31:0] n_brd, n_bwr, n_ard, n_awr, n_bst, scratch;
  logic        clear;

  // ---- write channel: take address and data together
  logic       w_go;
  assign s_awready = s_awvalid && s_wvalid && !s_bvalid;
  assign s_wready  = s_awready;
  assign w_go      = s_awready;
  assign s_bresp   = 2'b00;

  always_ff @(posedge clk) begin
    ld_start <= 1'b0;
    clear <= 1'b0;
    if (rst) begin
      run <= 1'b0;
      s_bvalid <= 1'b0;
      ld_addr <= '0; ld_n <= '0; scratch <= '0;
    end else begin
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
      if (w_go) begin
        s_bvalid <= 1'b1;
        case (s_awaddr[7:2])
          6'h02: begin
            run <= s_wdata[0];
            ld_start <= s_wdata[1] && !s_wdata[0] && !run;
            clear <= s_wdata[2];
          end
          6'h04: ld_addr <= s_wdata;
          6'h05: ld_n <= s_wdata;
          6'h0E: scratch <= s_wdata;
          default: ;
        endcase
      end
    end
  end

  // ---- counters
  always_ff @(posedge clk) begin
    if (rst || clear) begin
      cycles <= '0; n_brd <= '0; n_bwr <= '0; n_ard <= '0; n_awr <= '0; n_bst <= '0;
    end else begin
      if (run && !halted) cycles <= cycles + 1;
      if (b_rd) n_brd <= n_brd + 1;
      if (b_wr) n_bwr <= n_bwr + 1;
      if (a_rd) n_ard <= n_ard + 1;
      if (a_wr) n_awr <= n_awr + 1;
      if (b_wait) n_bst <= n_bst + 1;
    end
  end

  // ---- read channel
  assign s_arready = !s_rvalid;
  assign s_rresp = 2'b00;
  always_ff @(posedge clk) begin
    if (rst) begin
      s_rvalid <= 1'b0;
    end else begin
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
      if (s_arvalid && s_arready) begin
        s_rvalid <= 1'b1;
        case (s_araddr[7:2])
          6'h00: s_rdata <= 32'h4F54_5055;
          6'h01: s_rdata <= {16'(D), 8'(MCOLS), 8'(LANES)};
          6'h02: s_rdata <= {31'd0, run};
          6'h03: s_rdata <= {24'd0, run, calib, axi_err, wr_idle, ld_busy, error, halted};
          6'h04: s_rdata <= ld_addr;
          6'h05: s_rdata <= ld_n;
          6'h06: s_rdata <= cycles[31:0];
          6'h07: s_rdata <= cycles[63:32];
          6'h08: s_rdata <= icount;
          6'h09: s_rdata <= n_brd;
          6'h0A: s_rdata <= n_bwr;
          6'h0B: s_rdata <= n_ard;
          6'h0C: s_rdata <= n_awr;
          6'h0D: s_rdata <= n_bst;
          6'h0E: s_rdata <= scratch;
          default: s_rdata <= 32'hDEAD_BEEF;
        endcase
      end
    end
  end
endmodule
