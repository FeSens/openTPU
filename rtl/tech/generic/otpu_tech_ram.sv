// The technology RAMs, behavioural (rtl/tech/generic): Verilator, yosys and the FPGA flows infer
// them; rtl/tech/asic/otpu_tech_ram.sv has the same modules on SRAM macros. Every large memory of
// the core goes through one of these, so a target's file list picks the cells
// (docs/superpowers/specs/2026-10-10-multi-target-design.md, D4 and D5).
//
// All reads are synchronous and registered with an enable: rd holds while re (en) is low. A read
// of the word written at the same edge returns the old word here (read-first); the macros do not
// promise it (a 1R1W macro's same-address read is undefined), so a user must not depend on it
// (otpu_tmem bypasses the write it would hit); otpu_ram_1rw1r is read-first on both families
// (below). STYLE is the FPGA's ram_style ("block",
// "distributed" or "auto"); the macros ignore it. In simulation every word starts at 0, unless
// ZERO is 0 (a user that loads the memory from its own initial block: their order is undefined).

// One read port and one write port (simple dual-port).
module otpu_ram_1r1w #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter STYLE = "block",        // untyped: Vivado 2019.2 has no string parameters
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  if (STYLE == "distributed") begin : g_lut
    (* ram_style = "distributed" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      if (re) rd <= mem[ra];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else if (STYLE == "auto") begin : g_auto
    logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      if (re) rd <= mem[ra];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      if (re) rd <= mem[ra];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule

// One read port and one write port, read ahead: rd is the word at the read address register (ra,
// ra_n's value of the last cycle) with every write up to the last edge in it, as a LUT RAM's
// asynchronous read of a registered address gives it, for the cycle-exact place of such a RAM.
// The user gives the address's next value (ra_n) a cycle early; the block RAM reads it at the
// edge, and the word written at that same edge (which its read misses) comes from a register (the
// bypass: wd and its hit, registered). A wide LUT RAM puts each address bit on every 6 data bits'
// cells (the card's DMA gather: ~6,500 pins a bit); here on one block RAM per 36 or 72 bits.
module otpu_ram_1r1w_ra #(
  parameter int    W     = 32,
  parameter int    N     = 64,
  parameter STYLE = "block",        // untyped: Vivado 2019.2 has no string parameters
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic [$clog2(N)-1:0] ra_n,
  output logic [W-1:0]         rd
);
  logic [W-1:0] q, wq;
  logic         by;
  if (STYLE == "distributed") begin : g_lut
    (* ram_style = "distributed" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      q <= mem[ra_n];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else if (STYLE == "ultra") begin : g_uram
    // UltraRAM (72 bits x 4096 a cell): the same one-cycle read, from its unregistered output
    (* ram_style = "ultra" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      q <= mem[ra_n];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      if (we) mem[wa] <= wd;
      q <= mem[ra_n];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
  always_ff @(posedge clk) begin
    by <= we && wa == ra_n;
    wq <= wd;
  end
  assign rd = by ? wq : q;
`ifndef SYNTHESIS
  initial begin q = '0; wq = '0; by = 1'b0; end
`endif
endmodule

// One read port and one write port with byte enables (wbe[k]: write byte k of wd).
module otpu_ram_1r1w_be #(
  parameter int    W     = 64,           // a multiple of 8
  parameter int    N     = 1024,
  parameter STYLE = "block",        // untyped: Vivado 2019.2 has no string parameters
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 we,
  input  logic [W/8-1:0]       wbe,
  input  logic [$clog2(N)-1:0] wa,
  input  logic [W-1:0]         wd,
  input  logic                 re,
  input  logic [$clog2(N)-1:0] ra,
  output logic [W-1:0]         rd
);
  initial if (W % 8 != 0) $fatal(1, "otpu_ram_1r1w_be: W %0d is not a multiple of 8", W);
  if (STYLE == "block") begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      for (int k = 0; k < W / 8; k++) if (we && wbe[k]) mem[wa][8*k +: 8] <= wd[8*k +: 8];
      if (re) rd <= mem[ra];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_auto
    logic [W-1:0] mem [N];
    always_ff @(posedge clk) begin
      for (int k = 0; k < W / 8; k++) if (we && wbe[k]) mem[wa][8*k +: 8] <= wd[8*k +: 8];
      if (re) rd <= mem[ra];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule

// One port: a read or a write per cycle (en: the access, we: it is a write; a write does not
// change rd).
module otpu_ram_1rw #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter STYLE = "block",        // untyped: Vivado 2019.2 has no string parameters
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 en,
  input  logic                 we,
  input  logic [$clog2(N)-1:0] a,
  input  logic [W-1:0]         wd,
  output logic [W-1:0]         rd
);
  if (STYLE == "block") begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk) if (en) begin
      if (we) mem[a] <= wd;
      else rd <= mem[a];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_auto
    logic [W-1:0] mem [N];
    always_ff @(posedge clk) if (en) begin
      if (we) mem[a] <= wd;
      else rd <= mem[a];
    end
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule

// Two ports (true dual-port, one writing): port A a read or a write per cycle (ena: the access,
// wea: a write), port B a read (enb). Both read first: port A's write cycle loads rda with the
// word before the write, and port B's read of the word port A writes at that edge returns the
// word before it (the FPGA's true dual-port block RAM in READ_FIRST; the ASIC's 1RW1R macros as
// tools/asic/fakeram.py models them). otpu_tmem IMPL 3 TDP's replays depend on it.
module otpu_ram_1rw1r #(
  parameter int    W     = 32,
  parameter int    N     = 1024,
  parameter STYLE = "block",        // untyped: Vivado 2019.2 has no string parameters
  parameter bit    ZERO  = 1'b1
) (
  input  logic                 clk,
  input  logic                 ena,
  input  logic                 wea,
  input  logic [$clog2(N)-1:0] aa,
  input  logic [W-1:0]         wda,
  output logic [W-1:0]         rda,
  input  logic                 enb,
  input  logic [$clog2(N)-1:0] ab,
  output logic [W-1:0]         rdb
);
  if (STYLE == "block") begin : g_bram
    (* ram_style = "block" *) logic [W-1:0] mem [N];
    always_ff @(posedge clk)
      if (ena) begin
        if (wea) mem[aa] <= wda;
        rda <= mem[aa];
      end
    always_ff @(posedge clk)
      if (enb) rdb <= mem[ab];
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end else begin : g_auto
    logic [W-1:0] mem [N];
    always_ff @(posedge clk)
      if (ena) begin
        if (wea) mem[aa] <= wda;
        rda <= mem[aa];
      end
    always_ff @(posedge clk)
      if (enb) rdb <= mem[ab];
`ifndef SYNTHESIS
    initial if (ZERO) for (int i = 0; i < N; i++) mem[i] = '0;
`endif
  end
endmodule
