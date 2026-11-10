// Clock-domain crossings of the x2 link port (docs/x2_link.md), from the QSFP28
// link test's lt_sync.sv and lt_afifo.sv (card-proven in s6), renamed; the register names are
// theirs; boards/bcu1525/x2/cdc_x2.tcl constrains them (d_sy, *_xs, g_sy0, wg_sy0, dd_sy).
//   otpu_link_sync_bit / _pulse / _count / _word   lt_sync_bit / _pulse / _count / _word
//   otpu_link_afifo                                lt_afifo (dual-clock FIFO, first word falls through)
// FPGA power-up initializers are intentional (in particular, pulse toggles never reset).
// This file collects the small CDC primitives under their common family name.
/* verilator lint_off DECLFILENAME */
/* verilator lint_off PROCASSINIT */
`default_nettype none

(* keep_hierarchy = "yes" *) module otpu_link_sync_bit #(parameter int N = 2, parameter bit INIT = 1'b0) (
  input  wire clk,
  input  wire d,
  output wire q
);
  (* ASYNC_REG = "TRUE" *) logic d_sy = INIT;            // the first stage (lt_cdc.tcl: d_sy_reg)
  (* ASYNC_REG = "TRUE" *) logic [N-2:0] sy = {(N - 1){INIT}};
  if (N == 2) begin : g_2
    always_ff @(posedge clk) {sy, d_sy} <= {d_sy, d};
  end else begin : g_n
    always_ff @(posedge clk) {sy, d_sy} <= {sy[N-3:0], d_sy, d};
  end
  assign q = sy[N-2];
endmodule

(* keep_hierarchy = "yes" *) module otpu_link_sync_pulse (
  input  wire sclk,
  input  wire sp,
  input  wire dclk,
  output logic dp
);
  logic tg_xs = 1'b0;
  always_ff @(posedge sclk) if (sp) tg_xs <= ~tg_xs;
  logic t, t_d = 1'b0;
  otpu_link_sync_bit #(.N(3)) u_s (.clk(dclk), .d(tg_xs), .q(t));
  always_ff @(posedge dclk) begin
    t_d <= t;
    dp <= t ^ t_d;
  end
endmodule

(* keep_hierarchy = "yes" *) module otpu_link_sync_count #(parameter int W = 16) (
  input  wire          sclk,
  input  wire          srst,
  input  wire  [W-1:0] sbin,   // the counter, in sclk (steps by 0 or 1 a cycle)
  input  wire          dclk,
  output logic [W-1:0] dbin    // a value sbin had, in dclk (registered)
);
  logic [W-1:0] g_xs = '0;
  always_ff @(posedge sclk)
    if (srst) g_xs <= '0;
    else g_xs <= sbin ^ (sbin >> 1);
  (* ASYNC_REG = "TRUE" *) logic [W-1:0] g_sy0 = '0, g_sy1 = '0;
  always_ff @(posedge dclk) begin
    g_sy0 <= g_xs;
    g_sy1 <= g_sy0;
  end
  logic [W-1:0] b;
  always_comb begin
    b[W-1] = g_sy1[W-1];
    for (int i = W - 2; i >= 0; i--) b[i] = b[i+1] ^ g_sy1[i];
  end
  always_ff @(posedge dclk) dbin <= b;
endmodule

(* keep_hierarchy = "yes" *) module otpu_link_sync_word #(parameter int W = 16) (
  input  wire          sclk,
  input  wire          srst,
  input  wire  [W-1:0] sd,
  input  wire          dclk,
  input  wire          drst,
  output wire  [W-1:0] dd
);
  logic [W-1:0] hold_xs = '0;
  logic req_xs = 1'b0, ack_s;
  always_ff @(posedge sclk)
    if (srst) begin
      hold_xs <= '0;
      req_xs <= 1'b0;
    end else if (ack_s == req_xs && sd != hold_xs) begin
      hold_xs <= sd;
      req_xs <= ~req_xs;
    end
  logic req_d, ack_xs = 1'b0;
  otpu_link_sync_bit #(.N(2)) u_req (.clk(dclk), .d(req_xs), .q(req_d));
  otpu_link_sync_bit #(.N(2)) u_ack (.clk(sclk), .d(ack_xs), .q(ack_s));
  logic [W-1:0] dd_sy = '0;
  always_ff @(posedge dclk)
    if (drst) begin
      dd_sy <= '0;
      ack_xs <= 1'b0;
    end else if (req_d != ack_xs) begin
      dd_sy <= hold_xs;
      ack_xs <= req_d;
    end
  assign dd = dd_sy;
endmodule

`default_nettype wire
`default_nettype none

(* keep_hierarchy = "yes" *) module otpu_link_afifo #(parameter int W = 256, parameter int AW = 9) (
  input  wire          wclk,
  input  wire          wrst,
  input  wire          we,
  input  wire  [W-1:0] wdata,
  output logic         wfull,
  output logic [AW:0]  wfree,
  input  wire          rclk,
  input  wire          rrst,
  input  wire          re,
  output logic [W-1:0] rdata,
  output logic         rvalid,
  output logic [AW:0]  rcount     // the words the read side sees (not counting the output word)
);
  localparam int D = 1 << AW;
  (* ram_style = "block" *) logic [W-1:0] mem [D];

  // write side
  logic [AW:0] wbin = '0, rbin_w;
  logic [AW:0] rbin = '0, wbin_r;
  // its MSB is wbin's (same logic): kept apart so the crossing's source keeps its _xs name (lt_cdc.tcl)
  (* DONT_TOUCH = "TRUE" *) logic [AW:0] wgray_xs = '0;
  always_ff @(posedge wclk) begin
    if (wrst) begin
      wbin <= '0;
      wgray_xs <= '0;
    end else begin
      if (we && !wfull) begin
        mem[wbin[AW-1:0]] <= wdata;
        wbin <= wbin + 1'b1;
        wgray_xs <= (wbin + 1'b1) ^ ((wbin + 1'b1) >> 1);
      end
    end
  end
  otpu_link_sync_count #(.W(AW + 1)) u_rp (.sclk(rclk), .srst(rrst), .sbin(rbin), .dclk(wclk), .dbin(rbin_w));
  logic [AW:0] used_w;
  assign used_w = wbin - rbin_w;
  always_comb begin
    wfull = used_w[AW];
    wfree = (AW + 1)'(D) - used_w;
  end

  // read side
  (* ASYNC_REG = "TRUE" *) logic [AW:0] wg_sy0 = '0, wg_sy1 = '0;
  always_ff @(posedge rclk) begin
    wg_sy0 <= wgray_xs;
    wg_sy1 <= wg_sy0;
  end
  logic [AW:0] wb_c;
  always_comb begin
    wb_c[AW] = wg_sy1[AW];
    for (int i = AW - 1; i >= 0; i--) wb_c[i] = wb_c[i+1] ^ wg_sy1[i];
  end
  always_ff @(posedge rclk) wbin_r <= rrst ? '0 : wb_c;
  assign rcount = wbin_r - rbin;
  logic load;
  assign load = (rcount != '0) && (!rvalid || re);
  always_ff @(posedge rclk) begin
    if (rrst) begin
      rbin <= '0;
      rvalid <= 1'b0;
    end else begin
      if (load) begin
        rbin <= rbin + 1'b1;
        rvalid <= 1'b1;
      end else if (re) begin
        rvalid <= 1'b0;
      end
    end
  end
  always_ff @(posedge rclk) if (load) rdata <= mem[rbin[AW-1:0]];
endmodule

`default_nettype wire

/* verilator lint_on PROCASSINIT */
/* verilator lint_on DECLFILENAME */
