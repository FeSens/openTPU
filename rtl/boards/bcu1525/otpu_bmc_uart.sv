// The BCU1525's board management controller (BMC, the board's MSP432) on its UART: FPGA BB19 is the
// BMC's RX, BA19 its TX (docs/bcu1525.md section 13). An AXI4-Lite slave in core_clk (BAR0 0x20000:
// bd.tcl M_AXI_BMC) that moves bytes between the host and the BMC through two FIFOs: the host
// frames the requests and checks the replies (opentpu/host/bmc.py, which sends only its read-only
// ones). CAPS2 bit4 announces it. With WATCH (ID "BMC2") the line is otpu_bmc_watch's between the
// host's holds: it reads VCCINT's regulator itself every POLL_MS (its temperatures, READ_PIN and
// READ_VIN) for the thermal governor (reg_hot, reg_age: otpu_bcu_gov) and for the host (W_*). The
// host asks for the line (HOLD), uses it once STATUS HELD is set and gives it back; HOLD_MS after
// the grant the watch takes it back by itself (STATUS REVOKED). Without the line the host's TX
// bytes wait in its FIFO and its RX FIFO takes nothing; the line back to the watch empties the TX
// FIFO (a byte on the line finishes).
//
// The registers keep AMD's AXI UART Lite layout (PG142) at 0x0-0xC. Every response is OKAY (a
// BAR read through XDMA never gets an error back); 6 address bits are decoded (the map repeats).
//   0x00 RX     RO  bit31 VALID: [7:0] is the oldest received byte, and this read takes it from the
//                   FIFO; 0 when the FIFO is empty (nothing taken)
//   0x04 TX     W   [7:0] into the TX FIFO; dropped when the FIFO is full (STATUS TX_DROP)
//   0x08 STATUS RO  bit0 RX valid, bit1 RX full, bit2 TX empty, bit3 TX full, bit4 0 (no interrupt),
//                   bit5 overrun (a byte arrived at a full RX FIFO and was lost), bit6 framing
//                   error, bit7 parity error, bit8 TX_DROP (a TX write at a full FIFO), bit9 TX
//                   idle (the FIFO empty and the last stop bit sent), bit10 the RX line's level,
//                   bit11 HELD (the host has the line: always without WATCH), bit12 REVOKED (the
//                   watch took the line back at HOLD_MS), bit13 the watch stopped (the BMC's
//                   firmware is not one the protocol is known for). Bits 5-8 and 12 are sticky;
//                   this read clears them (bits 5-7 as PG142's; 5-7 count only on the host's line)
//   0x0C CTRL   W   bit0 empties the TX FIFO (a byte on the line finishes), bit1 the RX FIFO
//   0x10 LEVEL  RO  [15:0] bytes in the RX FIFO, [31:16] in the TX FIFO
//   0x14 ID     RO  0x424D4332 ("BMC2") with WATCH, 0x424D4331 ("BMC1") without
//   0x18 DIV    RO  [23:0] core clock cycles per bit (CLK_HZ / BAUD, rounded; CLK_HZ1's once
//                   clk133.tcl set the fallback clock's select: otpu_ctrl.sv's otpu_clk_k)
//   0x1C HOLD   RW  bit0 the host asks for the line (W: 1 asks, 0 gives it back; cleared at
//                   REVOKED), bit1 HELD
//   With WATCH (0 without), each [15:0] a LINEAR11 word and [31:16] ms since it was read
//   (saturating; 0xFFFF before the first):
//   0x20 W_HOT  RO  [15:0] the hottest of the three temperatures below (signed, quarter degrees
//                   C), [31:16] the oldest of their ages: what otpu_bcu_gov sees
//   0x24 W_P0   RO  page 0's READ_TEMPERATURE_1
//   0x28 W_P1   RO  page 1's READ_TEMPERATURE_1
//   0x2C W_CTL  RO  READ_TEMPERATURE_2, the controller's die
//   0x30 W_PIN  RO  READ_PIN (the input power, W)
//   0x34 W_VIN  RO  READ_VIN (V)
//   0x38 W_FW   RO  the BMC firmware's version (0 until it answered)
//   0x3C W_CNT  RO  [15:0] replies taken, [31:16] exchanges failed (saturating)
// The line: 8 data bits LSB first, a parity bit (even with PARITY 2, odd with 1, none with 0), one
// stop bit. The BMC's is 115200 8E1 (measured on the card). The receiver samples each bit at its
// middle, drops a start bit that does not last half a bit, and after a framing error (a low stop
// bit: a break, or a byte from another rate) waits for the line to go high before the next start.
module otpu_bmc_uart #(
  parameter int CLK_HZ = 100_000_000,
  parameter int CLK_HZ1 = CLK_HZ,          // the fallback clock (otpu_clk_k), or CLK_HZ: none
  parameter int BAUD   = 115_200,
  parameter int PARITY = 2,                 // 0 none, 1 odd, 2 even
  parameter int DEPTH  = 256,               // each FIFO's bytes (a power of two)
  parameter bit WATCH  = 1'b1,              // otpu_bmc_watch on the line between the host's holds
  parameter int POLL_MS = 500,              // its round's period, a reply's limit, the host's hold
  parameter int REPLY_MS = 200,
  parameter int HOLD_MS = 1000
) (
  input  logic        clk,
  input  logic        rst,
  // AXI4-Lite slave (AW and W taken together, one read at a time)
  input  logic [15:0] s_awaddr,
  input  logic        s_awvalid,
  output logic        s_awready,
  input  logic [31:0] s_wdata,
  input  logic [3:0]  s_wstrb,
  input  logic        s_wvalid,
  output logic        s_wready,
  output logic [1:0]  s_bresp,
  output logic        s_bvalid,
  input  logic        s_bready,
  input  logic [15:0] s_araddr,
  input  logic        s_arvalid,
  output logic        s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  output logic        s_rvalid,
  input  logic        s_rready,
  // the BMC's UART (the pins)
  output logic        tx,
  input  logic        rx,
  // the watch's regulator (otpu_bcu_gov): the hottest temperature, quarter degrees C, and the
  // oldest reading's age in ms (0xFFFF: none; always without WATCH)
  output logic signed [15:0] reg_hot,
  output logic [15:0] reg_age
);
  localparam int DIV = (CLK_HZ + BAUD / 2) / BAUD;
  localparam int DIV1 = (CLK_HZ1 + BAUD / 2) / BAUD;
  localparam int PW  = $clog2(DEPTH);
  localparam int NB  = PARITY != 0 ? 11 : 10;       // bits a frame: start, 8 data, parity, stop
`ifndef SYNTHESIS
  initial begin
    if (DIV < 16 || DIV >= (1 << 24)) $fatal(1, "otpu_bmc_uart: CLK_HZ / BAUD = %0d", DIV);
    if (DIV1 < 16 || DIV1 >= (1 << 24)) $fatal(1, "otpu_bmc_uart: CLK_HZ1 / BAUD = %0d", DIV1);
    if (DEPTH != (1 << PW) || DEPTH < 4) $fatal(1, "otpu_bmc_uart: DEPTH %0d", DEPTH);
  end
`endif

  // ---- a bit's cycles: DIV, DIV - 1, DIV - 2 and DIV / 2 - 1 at the build's clock, or at the
  // fallback one (DIV1's) once clk133.tcl set otpu_clk_sel's INIT
  logic [23:0] div, div_m1, div_m2, div_h1;
  if (CLK_HZ1 != CLK_HZ) begin : g_clk1
    logic sel;
    otpu_clk_sel u_sel (.clk, .sel);
    otpu_clk_k #(.W(24), .K0(24'(DIV)), .K1(24'(DIV1))) u_div (.sel, .k(div));
    otpu_clk_k #(.W(24), .K0(24'(DIV - 1)), .K1(24'(DIV1 - 1))) u_div_m1 (.sel, .k(div_m1));
    otpu_clk_k #(.W(24), .K0(24'(DIV - 2)), .K1(24'(DIV1 - 2))) u_div_m2 (.sel, .k(div_m2));
    otpu_clk_k #(.W(24), .K0(24'(DIV / 2 - 1)), .K1(24'(DIV1 / 2 - 1))) u_div_h1 (.sel, .k(div_h1));
  end else begin : g_clk0
    assign div = 24'(DIV);
    assign div_m1 = 24'(DIV - 1);
    assign div_m2 = 24'(DIV - 2);
    assign div_h1 = 24'(DIV / 2 - 1);
  end

  // ---- AXI4-Lite
  assign s_awready = s_awvalid && s_wvalid && !s_bvalid;
  assign s_wready  = s_awready;
  assign s_bresp   = 2'b00;
  assign s_rresp   = 2'b00;
  assign s_arready = !s_rvalid;
  wire       w_go   = s_awready && s_wstrb[0];
  wire [3:0] w_a    = s_awaddr[5:2];
  wire       r_go   = s_arvalid && s_arready;
  wire [3:0] r_a    = s_araddr[5:2];

  // ---- FIFOs (distributed RAM, read as addressed)
  logic [7:0]  tx_mem [DEPTH], rx_mem [DEPTH];
  logic [PW-1:0] tx_wp, tx_rp, rx_wp, rx_rp;
  logic [PW:0]   tx_n, rx_n;
  logic        tx_busy;                      // a frame on the line
  logic        rx_take;                      // the receiver has a byte (rx_byte), one cycle
  logic [7:0]  rx_byte;
  logic        rx_ferr, rx_perr;             // one cycle each, at the stop bit
  logic        overrun, frame_err, parity_err, tx_drop;
  logic        rx_s1, rx_s2;                 // the line, synchronized
  // the watch: its line (own), the host's request, the readings
  logic        own, own_q, hold, revoke, revoked, w_tx_go, fw_bad;
  logic [7:0]  w_tx_byte;
  logic [4:0][15:0] w_word, w_age;
  logic [15:0] hot_age, n_ok, n_bad;
  logic [31:0] fw;
  wire         host = !own;                 // the host has the line

  wire tx_clr  = w_go && w_a == 4'd3 && s_wdata[0] || revoke || own && !own_q;
  wire rx_clr  = w_go && w_a == 4'd3 && s_wdata[1];
  wire tx_push = w_go && w_a == 4'd1 && tx_n != (PW + 1)'(DEPTH) && !tx_clr;
  wire tx_pop  = host && !tx_busy && tx_n != 0 && !tx_clr;
  wire tx_load = tx_pop || own && w_tx_go && !tx_busy;
  wire rx_pop  = r_go && r_a == 4'd0 && rx_n != 0 && !rx_clr;
  wire rx_push = host && rx_take && (rx_n != (PW + 1)'(DEPTH) || rx_pop) && !rx_clr;
  wire st_read = r_go && r_a == 4'd2;
  wire [31:0] status = {18'd0, fw_bad, revoked, host, rx_s2, !tx_busy && tx_n == 0, tx_drop,
                        parity_err, frame_err, overrun, 1'b0, tx_n == (PW + 1)'(DEPTH), tx_n == 0,
                        rx_n == (PW + 1)'(DEPTH), rx_n != 0};
  function automatic logic [31:0] reading(input int i);
    return {w_age[i], w_word[i]};
  endfunction

  always_ff @(posedge clk) begin
    if (tx_push) tx_mem[tx_wp] <= s_wdata[7:0];
    if (rx_push) rx_mem[rx_wp] <= rx_byte;
    if (rst) begin
      s_bvalid <= 1'b0;
      s_rvalid <= 1'b0;
      s_rdata <= '0;
      tx_wp <= '0; tx_rp <= '0; tx_n <= '0;
      rx_wp <= '0; rx_rp <= '0; rx_n <= '0;
      overrun <= 1'b0; frame_err <= 1'b0; parity_err <= 1'b0; tx_drop <= 1'b0;
      hold <= 1'b0; revoked <= 1'b0; own_q <= 1'b1;
    end else begin
      own_q <= own;
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
      if (s_awready) s_bvalid <= 1'b1;
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
      if (r_go) begin
        s_rvalid <= 1'b1;
        case (r_a)
          4'd0:    s_rdata <= rx_n != 0 ? {1'b1, 23'd0, rx_mem[rx_rp]} : 32'd0;
          4'd2:    s_rdata <= status;
          4'd4:    s_rdata <= {16'(tx_n), 16'(rx_n)};
          4'd5:    s_rdata <= WATCH ? 32'h424D_4332 : 32'h424D_4331;
          4'd6:    s_rdata <= 32'(div);
          4'd7:    s_rdata <= {30'd0, host, hold};
          4'd8:    s_rdata <= {hot_age, reg_hot};
          4'd9:    s_rdata <= reading(0);
          4'd10:   s_rdata <= reading(1);
          4'd11:   s_rdata <= reading(2);
          4'd12:   s_rdata <= reading(3);
          4'd13:   s_rdata <= reading(4);
          4'd14:   s_rdata <= fw;
          4'd15:   s_rdata <= {n_bad, n_ok};
          default: s_rdata <= 32'd0;
        endcase
      end
      if (tx_clr) begin
        tx_wp <= '0; tx_rp <= '0; tx_n <= '0;
      end else begin
        if (tx_push) tx_wp <= tx_wp + 1'b1;
        if (tx_pop) tx_rp <= tx_rp + 1'b1;
        tx_n <= tx_n + (PW + 1)'(tx_push) - (PW + 1)'(tx_pop);
      end
      if (rx_clr) begin
        rx_wp <= '0; rx_rp <= '0; rx_n <= '0;
      end else begin
        if (rx_push) rx_wp <= rx_wp + 1'b1;
        if (rx_pop) rx_rp <= rx_rp + 1'b1;
        rx_n <= rx_n + (PW + 1)'(rx_push) - (PW + 1)'(rx_pop);
      end
      // the sticky flags: a status read clears them, an event in the same cycle sets them again
      if (st_read) begin
        overrun <= 1'b0; frame_err <= 1'b0; parity_err <= 1'b0; tx_drop <= 1'b0; revoked <= 1'b0;
      end
      if (host && rx_take && !rx_push && !rx_clr) overrun <= 1'b1;
      if (host && rx_ferr) frame_err <= 1'b1;
      if (host && rx_perr) parity_err <= 1'b1;
      if (w_go && w_a == 4'd1 && tx_n == (PW + 1)'(DEPTH)) tx_drop <= 1'b1;
      // the host's request for the line; the watch's revocation clears it
      if (revoke) begin
        hold <= 1'b0;
        revoked <= 1'b1;
      end else if (w_go && w_a == 4'd7) begin
        hold <= s_wdata[0];
      end
    end
  end

  // ---- transmitter: the frame in a shift register, LSB first, the line from a flip-flop
  logic [10:0] tx_sh;
  logic [3:0]  tx_k;                         // bits sent of this frame
  logic [23:0] tx_t;                         // cycles left of this bit
  wire  [7:0]  tx_d = own ? w_tx_byte : tx_mem[tx_rp];
  wire         tx_par = ^tx_d ^ (PARITY == 1);
  always_ff @(posedge clk) begin
    if (rst) begin
      tx_busy <= 1'b0;
      tx_sh <= '1;
      tx_k <= '0;
      tx_t <= '0;
      tx <= 1'b1;
    end else begin
      if (tx_load) begin
        tx_busy <= 1'b1;
        tx_sh <= PARITY != 0 ? {1'b1, tx_par, tx_d, 1'b0} : {2'b11, tx_d, 1'b0};
        tx_k <= '0;
        tx_t <= div_m2;                        // (the line goes low a cycle early: below)
      end else if (tx_busy) begin
        if (tx_t != 0) begin
          tx_t <= tx_t - 1'b1;
        end else if (tx_k == 4'(NB - 1)) begin
          tx_busy <= 1'b0;
        end else begin
          tx_sh <= {1'b1, tx_sh[10:1]};
          tx_k <= tx_k + 1'b1;
          tx_t <= div_m1;
        end
      end
      tx <= tx_load ? 1'b0 : tx_busy ? tx_sh[0] : 1'b1;
    end
  end

  // ---- receiver
  typedef enum logic [2:0] {R_IDLE, R_START, R_DATA, R_PAR, R_STOP, R_BREAK} rx_st_e;
  rx_st_e      r_st;
  logic [2:0]  r_k;
  logic [7:0]  r_sh;
  logic [23:0] r_t;
  logic        r_pbad;
  always_ff @(posedge clk) begin
    rx_s1 <= rx;
    rx_s2 <= rx_s1;
    rx_take <= 1'b0;
    rx_ferr <= 1'b0;
    rx_perr <= 1'b0;
    if (rst) begin
      rx_s1 <= 1'b1;
      rx_s2 <= 1'b1;
      r_st <= R_IDLE;
      r_k <= '0;
      r_sh <= '0;
      r_t <= '0;
      r_pbad <= 1'b0;
      rx_byte <= '0;
    end else begin
      case (r_st)
        R_IDLE:
          if (!rx_s2) begin
            r_st <= R_START;
            r_t <= div_h1;
          end
        R_START:
          if (r_t != 0) r_t <= r_t - 1'b1;
          else if (rx_s2) r_st <= R_IDLE;           // shorter than half a bit: not a start
          else begin
            r_st <= R_DATA;
            r_k <= '0;
            r_t <= div_m1;
          end
        R_DATA:
          if (r_t != 0) r_t <= r_t - 1'b1;
          else begin
            r_sh <= {rx_s2, r_sh[7:1]};
            r_k <= r_k + 1'b1;
            r_t <= div_m1;
            if (r_k == 3'd7) r_st <= PARITY != 0 ? R_PAR : R_STOP;
          end
        R_PAR:
          if (r_t != 0) r_t <= r_t - 1'b1;
          else begin
            r_pbad <= rx_s2 != (^r_sh ^ (PARITY == 1));
            r_st <= R_STOP;
            r_t <= div_m1;
          end
        R_STOP:
          if (r_t != 0) r_t <= r_t - 1'b1;
          else if (!rx_s2) begin
            rx_ferr <= 1'b1;
            r_st <= R_BREAK;
          end else begin
            r_st <= R_IDLE;
            if (PARITY != 0 && r_pbad) rx_perr <= 1'b1;
            else begin
              rx_byte <= r_sh;
              rx_take <= 1'b1;
            end
          end
        R_BREAK:
          if (rx_s2) r_st <= R_IDLE;
        default: r_st <= R_IDLE;
      endcase
    end
  end

  // ---- the watch
  if (WATCH) begin : g_watch
    otpu_bmc_watch #(.CLK_HZ(CLK_HZ), .CLK_HZ1(CLK_HZ1), .POLL_MS(POLL_MS), .REPLY_MS(REPLY_MS),
                     .HOLD_MS(HOLD_MS)) u_watch (
      .clk, .rst, .own, .hold_req(hold), .revoke, .tx_idle(!tx_busy), .tx_go(w_tx_go),
      .tx_byte(w_tx_byte), .rx_take, .rx_byte, .rx_err(rx_ferr || rx_perr), .word(w_word),
      .age(w_age), .hot(reg_hot), .hot_age, .fw, .fw_bad, .n_ok, .n_bad);
  end else begin : g_no_watch
    assign own = 1'b0;
    assign revoke = 1'b0;
    assign w_tx_go = 1'b0;
    assign w_tx_byte = '0;
    assign w_word = '0;
    assign w_age = '0;
    assign reg_hot = '0;
    assign hot_age = '1;
    assign fw = '0;
    assign fw_bad = 1'b0;
    assign n_ok = '0;
    assign n_bad = '0;
  end
  assign reg_age = hot_age;
endmodule
