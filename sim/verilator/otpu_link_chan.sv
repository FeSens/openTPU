// Aurora streaming stand-in for the x2 port: two directions on a user clock, D cycles of
// registered delay, a periodic clock-compensation tready stall (CC), and UPD cycles to up.
// There is no RX back pressure, just as on Aurora. Injection is on accepted wire words:
// +chan<ID><a|b>_flip=N flips raw word N's bit 0; _hflip=N flips HCRC bit 224 of DATA ARG N;
// _pflip=N flips its first payload bit; _tflip=N flips its trailer's DCRC; _drop_hdr=N,
// _drop_payload=N and _drop_trl=N remove that word. Faults do not change the sender's framing.
// +chan<ID>_stall=T:C withholds tx_tready for C cycles from T; _down=T drops the channel,
// _lane_down=T drops lane 0 only; _down_cycles=C makes either outage temporary; _soft=T:C
// asserts soft_err for C cycles. All times are user-clock cycles since reset.
`default_nettype none
module otpu_link_chan import otpu_link_pkg::*; #(
  parameter int W = 256,
  parameter int D = 58,
  parameter int CC = 7,
  parameter int UPD = 64,
  parameter int ID = 0
) (
  input logic clk, rst,
  input logic [W-1:0] a_tx_tdata,
  input logic a_tx_tvalid,
  output logic a_tx_tready,
  output logic [W-1:0] a_rx_tdata,
  output logic a_rx_tvalid,
  input logic [W-1:0] b_tx_tdata,
  input logic b_tx_tvalid,
  output logic b_tx_tready,
  output logic [W-1:0] b_rx_tdata,
  output logic b_rx_tvalid,
  output logic channel_up,
  output logic [3:0] lane_up,
  output logic soft_err
);
  localparam int R = 256;
  initial if (D < 1 || D >= R || W != WW) $fatal(1, "otpu_link_chan: invalid width/delay");
  int cyc, down_t, lane_t, down_cycles, stall_t, stall_c, soft_t, soft_c;
  initial begin
    string v;
    down_t = -1; lane_t = -1; down_cycles = -1;
    stall_t = -1; stall_c = 0; soft_t = -1; soft_c = 0;
    void'($value$plusargs($sformatf("chan%0d_down=%%d", ID), down_t));
    void'($value$plusargs($sformatf("chan%0d_lane_down=%%d", ID), lane_t));
    void'($value$plusargs($sformatf("chan%0d_down_cycles=%%d", ID), down_cycles));
    if ($value$plusargs($sformatf("chan%0d_stall=%%s", ID), v))
      if ($sscanf(v, "%d:%d", stall_t, stall_c) != 2) $fatal(1, "stall=T:C");
    if ($value$plusargs($sformatf("chan%0d_soft=%%s", ID), v))
      if ($sscanf(v, "%d:%d", soft_t, soft_c) != 2) $fatal(1, "soft=T:C");
  end
  wire down = down_t >= 0 && cyc >= down_t && (down_cycles < 0 || cyc < down_t + down_cycles);
  wire lane_down = lane_t >= 0 && cyc >= lane_t && (down_cycles < 0 || cyc < lane_t + down_cycles);
  wire stalled = stall_t >= 0 && cyc >= stall_t && cyc < stall_t + stall_c;
  always_comb begin
    channel_up = !rst && cyc >= UPD && !down;
    lane_up = {4{channel_up}};
    if (lane_down) lane_up[0] = 1'b0;
    a_tx_tready = channel_up && !stalled && !(CC > 0 && cyc % CC == CC - 1);
    b_tx_tready = a_tx_tready;
    soft_err = soft_t >= 0 && cyc >= soft_t && cyc < soft_t + soft_c;
  end
  always_ff @(posedge clk) cyc <= rst ? 0 : cyc + 1;

  wire [1:0][W-1:0] td = {b_tx_tdata, a_tx_tdata};
  wire [1:0] tv = {b_tx_tvalid, a_tx_tvalid};
  wire [1:0] tr = {b_tx_tready, a_tx_tready};
  logic [1:0][W-1:0] rd;
  logic [1:0] rv;
  assign {a_rx_tdata, b_rx_tdata} = {rd[1], rd[0]};
  assign {a_rx_tvalid, b_rx_tvalid} = {rv[1], rv[0]};
  for (genvar e = 0; e < 2; e++) begin : g_d
    logic [W-1:0] data [R];
    logic valid [R];
    int nword, left, length, seq;
    int flip, hflip, pflip, tflip, dh, dp, dt, chk_flip, len_bad;
    initial begin
      string pre;
      pre = $sformatf("chan%0d%s_", ID, e == 0 ? "a" : "b");
      flip = -1; hflip = -1; pflip = -1; tflip = -1; dh = -1; dp = -1; dt = -1; chk_flip = -1; len_bad = -1;
      void'($value$plusargs({pre, "chk_flip=%d"}, chk_flip));
      void'($value$plusargs({pre, "len_bad=%d"}, len_bad));
      void'($value$plusargs({pre, "flip=%d"}, flip));
      void'($value$plusargs({pre, "hflip=%d"}, hflip));
      void'($value$plusargs({pre, "pflip=%d"}, pflip));
      void'($value$plusargs({pre, "tflip=%d"}, tflip));
      void'($value$plusargs({pre, "drop_hdr=%d"}, dh));
      void'($value$plusargs({pre, "drop_payload=%d"}, dp));
      void'($value$plusargs({pre, "drop_trl=%d"}, dt));
    end
    always_ff @(posedge clk) begin
      if (rst) begin
        nword <= 0; left <= 0; length <= 0; seq <= -1;
        rv[e] <= 1'b0;
        for (int i = 0; i < R; i++) valid[i] <= 1'b0;
      end else begin
        automatic int now = cyc % R, due = (cyc + D - 1) % R;
        rv[e] <= valid[now] && channel_up;
        rd[e] <= data[now];
        valid[now] <= 1'b0;
        if (tv[e] && tr[e]) begin
          automatic bit hdr = left == 0 && td[e][63:56] == T_DATA;
          automatic bit payload = left > 1 && left == length + 1;
          automatic bit trailer = left == 1;
          automatic int arg = hdr ? int'(td[e][31:16]) : seq;
          automatic bit remove_word = (hdr && arg == dh) || (payload && arg == dp) || (trailer && arg == dt);
          automatic logic [W-1:0] mask = '0;
          if (nword == flip || (payload && arg == pflip) || (trailer && arg == tflip)) mask[0] = 1'b1;
          if (hdr && arg == hflip) mask[224] = 1'b1;
          valid[due] <= !remove_word;
          if (hdr && arg == len_bad) begin
            automatic logic [W-1:0] bad_header = td[e];
            bad_header[47:32] = 16'(LMAX + 1);
            data[due] <= seal(bad_header, td[e][55:48]);
          end else data[due] <= td[e] ^ mask ^ W'(hdr && arg == chk_flip);
          if (remove_word || mask != 0)
            $display("CHAN%0d %0d word %0d type %h seq %0d hdr %0d payload %0d trl %0d drop %0d mask %h",
                     ID, e, nword, td[e][63:56], arg, hdr, payload, trailer, remove_word, mask);
          if (hdr) begin
            length <= int'(td[e][47:32]);
            left <= td[e][47:32] == 0 ? 0 : int'(td[e][47:32]) + 1;
            seq <= arg;
          end else if (left > 0) left <= left - 1;
          nword <= nword + 1;
        end
      end
    end
  end
endmodule
`default_nettype wire
