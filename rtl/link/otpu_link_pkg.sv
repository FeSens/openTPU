// The cross-card link (docs/x2_link.md): the wire format between the
// collective hub's remote port (bf16st, otpu_coll.sv) and the link port (otpu_link, SLR2 beside
// quads 230/231), and the port's registers.
//
// Hub <-> port: per link (port 0 = QSFP0 / quad 231, port 1 = QSFP1 / quad 230), lt_port's
// AXI-Stream ends in the core clock: TX s_tdata[255:0] / s_tvalid / s_tready / s_tlast, RX
// m_tdata / m_tvalid / m_tready / m_tlast / m_thdr. The hub's adapter (SLR2, beside the port)
// owns the crossing to SLR1. A message is a header word and LEN payload words (tlast on the
// last). The port takes a header only when the peer's RX FIFO has room for the whole message, so
// a message never waits for credits once begun. The RX FIFO (RXW words a link, 64 KB) is the
// receive buffer: the hub holds a collective's messages there (m_tready low) until its own slices
// have issued it.
//
// Header word, low 64 bits (lt_pkg): [63:56] TYPE, [55:48] FLAGS (the port's), [47:32] LEN,
// [31:16] ARG, [15:8] AUX, [7:0] CHK (the port's: 0xA5 ^ the other seven bytes). DATA, the hub's:
// ARG the collective's sequence number, AUX the kind (A_*), [71:64] GS (global slice), [79:72]
// SRC (origin card), [83:80] HOPS, [111:96] ROW and [127:112] COL of the first payload word, the
// rest of [223:64] zero. [255:224] HCRC (the port's): CRC-32 of [223:0], checked before the
// header reaches m_*. Payload: 8 x 32-bit TMEM words, lane l in [32l+31:32l].
// Trailer (the port's; LEN > 0 only): after the last payload word, [31:0] DCRC = CRC-32 of the
// payload words, [63:56] T_TRL. The receiving port checks and drops it (the hub never sees it).
// It and the header count against the peer's credits: a message takes LEN + 1 + (LEN > 0).
// Control words (CRED, IDENT, ABORT, PING / PONG) go out between messages ahead of the next one,
// take no credits and are consumed by the receiving port; CRED, IDENT and ABORT carry HCRC too.
`default_nettype none

// A public wire/CSR vocabulary: a particular elaboration uses only a subset, and the CRC
// helpers intentionally inspect only their documented fields of a complete word.
/* verilator lint_off UNUSEDPARAM */
/* verilator lint_off UNUSEDSIGNAL */
package otpu_link_pkg;
  localparam int WW   = 256;               // wire word: one Aurora 64B/66B user word, x4
  localparam int LMAX = 32;                // payload words a hub message
  localparam int RXW  = 2048;              // RX FIFO words a link (64 KB)
  localparam logic [7:0] PROTO = 8'd1;

  // TYPE (lt_pkg's, plus ABORT)
  localparam logic [7:0] T_PRBS  = 8'h50,  // LEN words of the per-lane xorshift64 pattern (qual)
                         T_DATA  = 8'h44,  // the hub's message
                         T_PING  = 8'h49,  // latency probe (LEN 0)
                         T_PONG  = 8'h4F,  // its echo
                         T_CRED  = 8'h43,  // ARG: the receiver's freed RX FIFO words, mod 2^16
                         T_IDENT = 8'h4E,  // hello: [71:64] card, [79:72] ncards, [80] port,
                                           // [81] the sender has heard the receiver's hello,
                                           // [95:88] PROTO, [127:96] cfg
                         T_ABORT = 8'h41;  // the sender is in error (ARG: its cause bits)
  localparam logic [7:0] T_TRL   = 8'h54;  // a trailer's [63:56] (positional, not a header)
  localparam int F_CRED = 0;               // FLAGS bit 0: the words count against credits
  // DATA's AUX
  localparam logic [7:0] A_PIECE = 8'd1,   // LEN payload words of slice GS's piece at ROW / COL
                         A_BAR   = 8'd2,   // a BAR token (LEN 0)
                         A_END   = 8'd3;   // reserved (P4): the sender's last message of
                                           // collective ARG on this link (LEN 0)
  // DATA's hub fields (bit positions in the header word)
  localparam int H_GS = 64, H_SRC = 72, H_HOPS = 80, H_ROW = 96, H_COL = 112;
  // PING / PONG stamps (lt_pkg's; bit ranges in the word), in user_clk cycles
  localparam int TS_ATX = 112;   // [143:112] the initiator's counter as the PING entered Aurora
  localparam int TS_BRX = 144;   // [175:144] the echoer's counter as the PING left Aurora
  localparam int T_TURN = 176;   // [207:176] the echoer's turnaround
  localparam int T_URTT = 208;   // [239:208] the initiator's round trip

  // xorshift64 (13, 7, 17): one lane's next PRBS word (lt_pkg's)
  function automatic logic [63:0] xs64(input logic [63:0] x);
    logic [63:0] y;
    y = x ^ (x << 13);
    y = y ^ (y >> 7);
    y = y ^ (y << 17);
    return y;
  endfunction
  function automatic logic [WW-1:0] xs256(input logic [WW-1:0] x);
    logic [WW-1:0] y;
    for (int l = 0; l < 4; l++) y[64*l +: 64] = xs64(x[64*l +: 64]);
    return y;
  endfunction

  function automatic logic [7:0] hdr_chk(input logic [63:0] h);
    return 8'hA5 ^ h[63:56] ^ h[55:48] ^ h[47:40] ^ h[39:32] ^ h[31:24] ^ h[23:16] ^ h[15:8];
  endfunction
  function automatic logic known(input logic [7:0] t);
    return t == T_PRBS || t == T_DATA || t == T_PING || t == T_PONG || t == T_CRED ||
           t == T_IDENT || t == T_ABORT;
  endfunction
  // CRC-32 (0x04C11DB7, MSB first, not reflected) of a wire word, continuing c
  function automatic logic [31:0] crc32w(input logic [31:0] c, input logic [WW-1:0] w);
    logic [31:0] x;
    x = c;
    for (int i = WW - 1; i >= 0; i--)
      x = {x[30:0], 1'b0} ^ ((x[31] ^ w[i]) ? 32'h04C1_1DB7 : 32'h0);
    return x;
  endfunction
  localparam logic [31:0] CRC_INIT = 32'hFFFF_FFFF;
  // a header's HCRC: [223:0]
  function automatic logic [31:0] hcrc(input logic [WW-1:0] w);
    logic [31:0] x;
    x = CRC_INIT;
    for (int i = 223; i >= 0; i--)
      x = {x[30:0], 1'b0} ^ ((x[31] ^ w[i]) ? 32'h04C1_1DB7 : 32'h0);
    return x;
  endfunction
  // the port's TX header: FLAGS, CHK and HCRC filled in
  function automatic logic [WW-1:0] seal(input logic [WW-1:0] w, input logic [7:0] flags);
    logic [WW-1:0] v;
    v = w;
    v[55:48] = flags;
    v[7:0] = hdr_chk(v[63:0]);
    v[255:224] = hcrc(v);
    return v;
  endfunction
  // hcrc(w) as two halves (CRC-32 is affine in the word): hcrc_hi(w) ^ hcrc_lo(w), for a check
  // over two register stages
  function automatic logic [31:0] hcrc_hi(input logic [WW-1:0] w);
    return crc32w(32'h0, {32'h0, w[223:112], 112'h0});
  endfunction
  function automatic logic [31:0] hcrc_lo(input logic [WW-1:0] w);
    return hcrc({144'h0, w[111:0]});
  endfunction
  // the types whose words carry HCRC (PING / PONG are stamped on the way out, PRBS is the qual's)
  function automatic logic sealed(input logic [7:0] t);
    return t == T_DATA || t == T_CRED || t == T_IDENT || t == T_ABORT;
  endfunction
  function automatic logic hdr_ok(input logic [WW-1:0] w);
    return w[7:0] == hdr_chk(w[63:0]) && known(w[63:56]) && w[255:224] == hcrc(w);
  endfunction
  function automatic logic [WW-1:0] mk_data(input logic [15:0] seq, input logic [7:0] aux,
                                            input logic [15:0] len, input logic [7:0] gs,
                                            input logic [7:0] src, input logic [3:0] hops,
                                            input logic [15:0] row, input logic [15:0] col);
    logic [WW-1:0] w;
    w = '0;
    w[63:56] = T_DATA; w[47:32] = len; w[31:16] = seq; w[15:8] = aux;
    w[H_GS +: 8] = gs; w[H_SRC +: 8] = src; w[H_HOPS +: 4] = hops;
    w[H_ROW +: 16] = row; w[H_COL +: 16] = col;
    return w;
  endfunction
  function automatic logic [WW-1:0] mk_trl(input logic [31:0] dcrc);
    logic [WW-1:0] w;
    w = '0;
    w[63:56] = T_TRL; w[31:0] = dcrc;
    return w;
  endfunction
  // words a message takes of the peer's RX FIFO
  function automatic logic [16:0] msg_words(input logic [15:0] len);
    return {1'b0, len} + 17'd1 + 17'(len != 0);
  endfunction

  // ---- registers: offsets in the control block's X2 window (otpu_ctrl 0x300-0x3FF) ----------
  localparam logic [7:0] R_CARD  = 8'h00,  // X2_CARD (otpu_ctrl's): [7:0] CARD_ID, [15:8] NCARDS
                         R_CTRL  = 8'h04,  // X2_CTRL: X_* bits, self-clearing
                         R_INFO  = 8'h08,  // RO [7:0] PROTO, [15:8] LMAX, [31:16] RXW
                         R_PORT  = 8'h10,  // port p at R_PORT + 8'h10 * p: +0 status (P_*),
                                           // +4 hello, +8 CRC errors, +C frames received
                         R_PING  = 8'h30,  // RO R_PING + 4p: last round trip in core cycles,
                                           // [31] valid (includes request/response crossings)
                         R_CNT   = 8'h40;  // RO R_CNT + 8'h10 * p: +0 words sent, +4 words
                                           // received, +8 credit-wait cycles, +C soft errors
  localparam int X_CLR = 0,                // clear the sticky bits
                 X_HELLO = 1,              // send a hello now on both ports
                 X_LRST = 2,               // link reset: FIFOs, credits, counters, stickies
                 X_PING = 3;               // a latency probe on both ports
  localparam int P_CHUP = 0,               // channel up
                 P_HELLO = 1,              // a hello came since the last X2_CARD write / LRST
                 P_CRC = 2,                // sticky: a message failed HCRC / CHK / LEN / DCRC,
                                           //   or the RX FIFO overflowed (rx_err)
                 P_WDOG = 3,               // sticky: a message waited for credits > 2^24 cycles
                 P_ABORT = 4,              // sticky: an ABORT came
                 P_LANES = 5,              // all four lanes up
                 P_DROP = 6,               // sticky: the channel or a lane dropped while up
                 P_UP = 7;                 // the hub's `up`: channel and lanes up, hello agrees
                                           // [31:16] credits available (words)
  localparam logic [7:0] HELLO_MAGIC = 8'hA5;   // hello register [31:24]; [7:0] card,
                                                // [11:8] port, [19:12] ncards
endpackage
/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on UNUSEDPARAM */

`default_nettype wire
