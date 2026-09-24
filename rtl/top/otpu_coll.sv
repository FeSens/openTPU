// Collective unit: BAR and GATHER across S slices. When every slice is waiting on the same
// collective, GATHER streams each slice's segment, up to LANES consecutive words per cycle, and
// broadcasts it into the TMEM of all slices at dst + s*seg + r*drs + c; then all slices are
// released together. `gnt` is the AND of every slice's TMEM grant; without it the unit holds.
module otpu_coll
  import otpu_pkg::*;
#(
  parameter int S     = 1,
  parameter int LANES = 8
) (
  input  logic                            clk,
  input  logic                            rst,
  input  logic [S-1:0]                    req,
  input  cmd_t                            cmds [S],
  input  logic                            gnt,
  output logic                            ack,
  output logic [S-1:0][LANES-1:0]         r_en,
  output logic [S-1:0][LANES-1:0][31:0]   r_addr,
  input  logic [S-1:0][LANES-1:0][31:0]   r_data,
  output logic [LANES-1:0]                w_en,
  output logic [LANES-1:0][31:0]          w_addr,
  output logic [LANES-1:0][31:0]          w_data
);
  typedef enum logic [1:0] {C_IDLE, C_RUN, C_ACK, C_COOL} st_t;
  st_t st;
  logic [31:0] dst, drs, seg;
  logic [15:0] rows, cols;
  logic [31:0] s, r, c, ps, pr, pcl;
  logic        pv, issuing;
  // row base addresses kept by adding the strides (no multipliers):
  // rrow = w1[s] + r*w4[s], wrow = dst + s*seg + r*drs, wseg = dst + s*seg
  logic [31:0] rrow, wrow, wseg;
  // the write stage's address base and lane mask, registered with pcl (they feed the TMEM
  // arbiter, so no adder or compare sits in front of it)
  logic [31:0]      pwa;
  logic [LANES-1:0] pm;
  // write stage q: stage p's address, mask and TMEM read data, registered (no path from the
  // TMEM block RAMs to their write port in one cycle); qlast: it holds the command's last write
  logic                   qv, qlast;
  logic [31:0]            qwa;
  logic [LANES-1:0]       qm;
  logic [LANES-1:0][31:0] qd;

  always_comb begin
    ack = (st == C_ACK);
    r_en = '0; r_addr = '0;
    w_en = '0; w_addr = '0; w_data = '0;
    if (st == C_RUN && issuing) begin
      for (int k = 0; k < S; k++)
        if (k == int'(s))
          for (int l = 0; l < LANES; l++)
            if (c + 32'(l) < 32'(cols)) begin
              r_en[k][l] = 1'b1;
              r_addr[k][l] = rrow + c + 32'(l);
            end
    end
    if (st == C_RUN && qv) begin
      for (int l = 0; l < LANES; l++)
        if (qm[l]) begin
          w_en[l] = 1'b1;
          w_addr[l] = qwa + 32'(l);
          w_data[l] = qd[l];
        end
    end
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      st <= C_IDLE;
      pv <= 1'b0;
      qv <= 1'b0;
    end else begin
      case (st)
        C_IDLE: if (&req) begin
          if (cmds[0].op == OP_BAR || cmds[0].w3[15:0] == 0 || cmds[0].w3[31:16] == 0) begin
            st <= C_ACK;
          end else begin
            dst  <= cmds[0].w2;
            rows <= cmds[0].w3[15:0];
            cols <= cmds[0].w3[31:16];
            drs  <= cmds[0].w5;
            seg  <= cmds[0].w6;
            s <= '0; r <= '0; c <= '0;
            rrow <= cmds[0].w1;
            wrow <= cmds[0].w2; wseg <= cmds[0].w2;
            pv <= 1'b0;
            qv <= 1'b0;
            issuing <= 1'b1;
            st <= C_RUN;
          end
        end
        C_RUN: if (gnt) begin
          qv <= pv;
          qwa <= pwa;
          qm <= pm;
          for (int l = 0; l < LANES; l++)
            for (int k = 0; k < S; k++) if (k == int'(ps)) qd[l] <= r_data[k][l];
          qlast <= pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols);
          pv <= issuing;
          ps <= s; pr <= r; pcl <= c;
          pwa <= wrow + c;
          for (int l = 0; l < LANES; l++) pm[l] <= (c + 32'(l) < 32'(cols));
          if (issuing) begin
            if (c + LANES >= 32'(cols)) begin
              c <= '0;
              if (r + 1 == 32'(rows)) begin
                r <= '0;
                if (s + 1 == S) issuing <= 1'b0;
                else begin
                  s <= s + 1;
                  for (int k = 0; k < S; k++) if (k == int'(s) + 1) rrow <= cmds[k].w1;
                  wseg <= wseg + seg;
                  wrow <= wseg + seg;
                end
              end else begin
                r <= r + 1;
                for (int k = 0; k < S; k++) if (k == int'(s)) rrow <= rrow + cmds[k].w4;
                wrow <= wrow + drs;
              end
            end else begin
              c <= c + LANES;
            end
          end
          if (pv && ps + 1 == S && pr + 1 == 32'(rows) && pcl + LANES >= 32'(cols))
            pv <= 1'b0;                                 // the last read's data moves to q
          if (qv && qlast) begin                         // the last write is taken this cycle
            qv <= 1'b0;
            st <= C_ACK;
          end
        end
        C_ACK:  st <= C_COOL;
        C_COOL: st <= C_IDLE;
        default: st <= C_IDLE;
      endcase
    end
  end
endmodule
