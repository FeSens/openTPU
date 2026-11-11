// A stand-in for a card-to-card link (qsfp-link's lt_port pair and the Aurora cable between
// them) until its model lands: two lt_port-style AXI-Stream ends on the core clock (otpu_pkg
// lk_in_t / lk_out_t), card a's and card b's. Each direction (otpu_link_dir) takes a message's
// header only when the receiver's RX FIFO has room for the whole message (the credits: +xdepth=
// words, 2040), delivers each word +xlat= cycles later (40) into that FIFO, and returns a credit
// the same latency after the receiver takes a word; a sender's word a cycle at most. +xstall=P:
// the sender's tready withheld P percent of the cycles. Error injection on link +xelink=ID's
// direction +xedir= (0 a to b, 1 b to a), message +xemsg=K (counted from 0 on that direction):
// +xekind=flip (a payload bit of the message flipped, rx_err as its last word arrives: a CRC
// failure), drop (the message never arrives; its credits come back), seq (its header's
// collective number changed). ABORT from a sender reaches the other end's abort_in xlat later.
// up rises 8 cycles after reset.
module otpu_link_stub
  import otpu_pkg::*;
#(
  parameter int ID = 0
) (
  input  logic    clk,
  input  logic    rst,
  input  lk_out_t ao,
  output lk_in_t  ai,
  input  lk_out_t bo,
  output lk_in_t  bi
);
  logic ab_rdy, ab_v, ab_h, ab_l, ab_e, ab_x, ba_rdy, ba_v, ba_h, ba_l, ba_e, ba_x;
  logic [255:0] ab_d, ba_d;
  otpu_link_dir #(.ID(ID), .DIR(0)) u_ab (
    .clk, .rst, .s_tvalid(ao.s_tvalid), .s_tlast(ao.s_tlast), .s_tdata(ao.s_tdata),
    .abort_req(ao.abort_req), .s_tready(ab_rdy), .m_tready(bo.m_tready), .m_tvalid(ab_v),
    .m_thdr(ab_h), .m_tlast(ab_l), .m_tdata(ab_d), .rx_err(ab_e), .abort_in(ab_x));
  otpu_link_dir #(.ID(ID), .DIR(1)) u_ba (
    .clk, .rst, .s_tvalid(bo.s_tvalid), .s_tlast(bo.s_tlast), .s_tdata(bo.s_tdata),
    .abort_req(bo.abort_req), .s_tready(ba_rdy), .m_tready(ao.m_tready), .m_tvalid(ba_v),
    .m_thdr(ba_h), .m_tlast(ba_l), .m_tdata(ba_d), .rx_err(ba_e), .abort_in(ba_x));
  logic [3:0] upc;
  always_ff @(posedge clk) upc <= rst ? '0 : (upc == 4'd8 ? upc : upc + 1'b1);
  always_comb begin
    ai.s_tready = ab_rdy;
    ai.m_tvalid = ba_v;
    ai.m_thdr = ba_h;
    ai.m_tlast = ba_l;
    ai.m_tdata = ba_d;
    ai.up = upc == 4'd8;
    ai.rx_err = ba_e;
    ai.abort_in = ba_x;
    bi.s_tready = ba_rdy;
    bi.m_tvalid = ab_v;
    bi.m_thdr = ab_h;
    bi.m_tlast = ab_l;
    bi.m_tdata = ab_d;
    bi.up = upc == 4'd8;
    bi.rx_err = ab_e;
    bi.abort_in = ab_x;
  end
endmodule

// One direction of otpu_link_stub: the sender's lt_port TX end, the cable, the receiver's RX
// FIFO (first word falls through) and lt_port RX end.
module otpu_link_dir
  import otpu_pkg::*;
#(
  parameter int ID  = 0,
  parameter int DIR = 0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic         s_tvalid,
  input  logic         s_tlast,
  input  logic [255:0] s_tdata,
  input  logic         abort_req,
  output logic         s_tready,
  input  logic         m_tready,
  output logic         m_tvalid,
  output logic         m_thdr,
  output logic         m_tlast,
  output logic [255:0] m_tdata,
  output logic         rx_err,
  output logic         abort_in
);
  int lat = 40, depth = 2040, stall = 0, elink = -1, edir = 0, emsg = -1;
  string ekind = "flip";
  initial begin
    void'($value$plusargs("xlat=%d", lat));
    void'($value$plusargs("xdepth=%d", depth));
    void'($value$plusargs("xstall=%d", stall));
    void'($value$plusargs("xelink=%d", elink));
    void'($value$plusargs("xedir=%d", edir));
    void'($value$plusargs("xemsg=%d", emsg));
    void'($value$plusargs("xekind=%s", ekind));
  end
  typedef struct {
    longint       t;
    logic [255:0] d;
    bit           hdr, last, err;
  } fl_t;
  fl_t    line [$];                       // on the cable: arrival time
  fl_t    rxq [$];                        // the receiver's RX FIFO
  longint cr [$];                         // credits on their way back: arrival time
  longint ab [$];                         // ABORTs on their way
  longint cyc = 0;
  int     out = 0;                        // words sent, their credits not back
  bit     first = 1'b1, stl = 1'b0, drop = 1'b0, flip = 1'b0, flipped = 1'b0;
  int     rem = 0, msg = 0, nmsg = 0, nerr = 0;
  logic   [15:0] len;
  assign len = s_tdata[LKH_LEN +: 16];
  always_comb
    s_tready = !rst && !stl && (!first || depth - out >= 1 + int'(len));
  always @(posedge clk) begin
    cyc <= cyc + 1;
    rx_err <= 1'b0;
    abort_in <= 1'b0;
    if (rst) begin
      line.delete(); rxq.delete(); cr.delete(); ab.delete();
      out = 0; first = 1'b1; rem = 0; stl <= 1'b0;
      m_tvalid <= 1'b0;
    end else begin
      if (m_tvalid && m_tready) begin       // the receiver took the head
        void'(rxq.pop_front());
        cr.push_back(cyc + longint'(lat));
      end
      if (s_tvalid && s_tready) begin
        fl_t f;
        f.t = cyc + longint'(lat);
        f.d = s_tdata;
        f.hdr = first;
        f.err = 1'b0;
        if (first) begin
          bit hit;
          hit = ID == elink && DIR == edir && msg == emsg;
          drop = hit && ekind == "drop";
          flip = hit && ekind == "flip";
          flipped = 1'b0;
          if (hit && ekind == "seq") f.d[LKH_SEQ] = ~f.d[LKH_SEQ];
          if (hit) $display("XSTUB link %0d dir %0d: message %0d %s", ID, DIR, msg, ekind);
          msg++;
          rem = int'(len);
          first = rem == 0;
        end else begin
          if (flip && !flipped) begin
            f.d[0] = ~f.d[0];
            flipped = 1'b1;
          end
          rem--;
          first = rem == 0;
        end
        f.last = first;
        f.err = flip && first;
        out++;
        if (drop) cr.push_back(cyc + longint'(lat));
        else line.push_back(f);
      end
      while (line.size() != 0 && line[0].t <= cyc) begin
        fl_t f;
        f = line.pop_front();
        if (f.err) begin rx_err <= 1'b1; nerr++; end
        if (f.last) nmsg++;
        rxq.push_back(f);
        if (rxq.size() > depth)
          $fatal(1, "otpu_link_dir %0d.%0d: RX FIFO over %0d words (credits)", ID, DIR, depth);
      end
      while (cr.size() != 0 && cr[0] <= cyc) begin
        void'(cr.pop_front());
        out--;
      end
      if (abort_req) ab.push_back(cyc + longint'(lat));
      while (ab.size() != 0 && ab[0] <= cyc) begin
        void'(ab.pop_front());
        abort_in <= 1'b1;
      end
      stl <= int'($urandom_range(99)) < stall;
      m_tvalid <= rxq.size() != 0;
      if (rxq.size() != 0) begin
        m_tdata <= rxq[0].d;
        m_thdr <= rxq[0].hdr;
        m_tlast <= rxq[0].last;
      end
    end
  end
endmodule
