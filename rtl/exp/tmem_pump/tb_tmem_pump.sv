// EXPERIMENTAL (docs/tmem_pump.md): otpu_tmem_pump against otpu_tmem on the same random traffic,
// read data compared every core cycle on every port and lane. The slice's TMEM ports (otpu_slice:
// 8 read ports, the MXU's with the crossbar, Q3 and COLL guests on the DMA's copy; 4 write ports)
// and its rules: a port's lanes in distinct banks (rotator ports contiguous runs), one write per
// bank, the guests only when their host does not ask (IMPL 0), one granted port per copy and bank
// (IMPL 1, the slice's gnt_k). Addresses mostly in a small window, so reads often meet the
// writes in flight (the bypass) and recent ones.
// K = 3 / 4 (one copy pumped at 3x / 4x, reads a cycle later): its read data are compared with
// otpu_tmem's of the cycle before.
// Built and run by run_tb.sh (Verilator --binary --timing, -GIMPL / -GNCP / -GCPYV / -GK).
// Prints TB_RESULT PASS / FAIL with the counts.
`timescale 1ns / 1ps
module tb_tmem_pump #(
  parameter int WORDS  = 4096,
  parameter int IMPL   = 0,
  parameter int NCP    = 3,
  parameter logic [23:0] CPYV = {3'd2, 3'd1, 3'd2, 3'd1, 3'd0, 3'd0, 3'd2, 3'd0},  // the slice's
  parameter int CYCLES = 200000,
  parameter int SEED   = 1,
  parameter int HOT    = 256,       // the hot window's words
  parameter bit BYP    = 1'b0,      // otpu_tmem_pump's (redundant) bypass
  parameter bit RQ     = 1'b0,      // IMPL 1: request-priority addresses (otpu_tmem RQ)
  parameter int K      = 2          // the pump factor (otpu_tmem_pump K)
);
  localparam int LANES = 8, NRP = 8, NWP = 4, BW = 3;
  localparam int P_MXU = 0, P_DMA = 1, P_Q = 2, P_VA = 3, P_VB = 4, P_Q3 = 5, P_Q2 = 6, P_COLL = 7;
  localparam int W_DMA = 0, W_MXU = 1, W_COLL = 2, W_VPU = 3;
  localparam logic [NRP-1:0] GEN_R = NRP'(1) << P_MXU;
  localparam logic [NWP-1:0] GEN_W = NWP'(1) << W_MXU;
  localparam logic [NRP-1:0] SH_MASK = (NRP'(1) << P_Q3) | (NRP'(1) << P_COLL);
  function automatic logic [NRP-1:0][2:0] cpy3();
    for (int p = 0; p < NRP; p++) cpy3[p] = CPYV[3 * p +: 3];
  endfunction
  localparam logic [NRP-1:0][2:0] CPY3 = cpy3();
  // the slice's priority ranks (otpu_slice PRANK): DMA, COLL, VPU, MXU, QUANT
  //                                       COLL  Q2    Q3    VB    VA    Q     DMA   MXU
  localparam logic [NRP-1:0][2:0] PRANK = {3'd1, 3'd4, 3'd4, 3'd2, 3'd2, 3'd4, 3'd0, 3'd3};

  // the core clock 24 ns (rises at 12, 36, ...), clk_f 24 / K ns with a rise at every core edge
  localparam int HF = 12 / K;
  localparam int OFF = (12 % (2 * HF) != 0) ? 12 % (2 * HF) : 2 * HF;
  logic clk = 1'b0, clk_f = 1'b0;
  always #12 clk = ~clk;
  initial begin
    #(OFF);
    forever begin clk_f = 1'b1; #(HF); clk_f = 1'b0; #(HF); end
  end
  initial if (12 % K != 0) $fatal(1, "tb_tmem_pump: K %0d", K);

  logic [NRP-1:0][LANES-1:0]       r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, rd_ref, rd_dut;
  logic [NWP-1:0][LANES-1:0]       w_en, w_req;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;

  otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(1), .GEN_R(GEN_R),
              .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK(IMPL == 0 ? SH_MASK : '0), .SID(0),
              .IMPL(IMPL), .NCP(NCP), .CPY(CPY3), .RQ(RQ), .PRANK(PRANK)) u_ref (
    .clk, .clk_f(1'b0), .rst(1'b0), .r_en, .r_req, .r_addr, .r_data(rd_ref), .w_en, .w_req, .w_gnt, .w_addr,
    .w_data, .dump(1'b0));
  otpu_tmem_pump #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .GEN_R(GEN_R),
                   .GEN_W(GEN_W), .SH_HOST(P_DMA), .SH_MASK(IMPL == 0 ? SH_MASK : '0),
                   .IMPL(IMPL), .NCP(NCP), .CPY(CPY3), .BYP(BYP), .RQ(RQ),
                   .PRANK(PRANK), .K(K)) u_dut (
    .clk, .clk_f, .r_en, .r_req, .r_addr, .r_data(rd_dut), .w_en, .w_req, .w_gnt, .w_addr,
    .w_data);

  // a random address: mostly in the hot window
  function automatic int unsigned raddr(int unsigned lim);
    if ($urandom % 8 != 0) return $urandom % (HOT < lim ? HOT : lim);
    return $urandom % lim;
  endfunction
  // a contiguous run (rotator port): lane l at a0 + l, a random mask of lanes, all below WORDS
  task automatic run(output logic [LANES-1:0] en, output logic [LANES-1:0][31:0] a, input int pct);
    int unsigned a0;
    a0 = raddr(WORDS - LANES + 1);
    for (int l = 0; l < LANES; l++) a[l] = a0 + l;
    en = ($urandom % 100 < pct) ? LANES'($urandom) : '0;
    if (($urandom % 2) && en != '0) en = '1;
  endtask
  // scattered lanes (the MXU's crossbar ports): distinct banks, any rows
  task automatic scat(output logic [LANES-1:0] en, output logic [LANES-1:0][31:0] a, input int pct);
    int perm [LANES];
    for (int l = 0; l < LANES; l++) perm[l] = l;
    for (int l = LANES - 1; l > 0; l--) begin
      int j, t;
      j = $urandom % (l + 1); t = perm[l]; perm[l] = perm[j]; perm[j] = t;
    end
    for (int l = 0; l < LANES; l++) a[l] = (raddr(WORDS) / LANES) * LANES + perm[l];
    en = ($urandom % 100 < pct) ? LANES'($urandom) : '0;
  endtask
  function automatic logic [LANES-1:0] banks(logic [LANES-1:0] en, logic [LANES-1:0][31:0] a);
    banks = '0;
    for (int l = 0; l < LANES; l++) if (en[l]) banks[a[l][BW-1:0]] = 1'b1;
  endfunction

  // the arbitration groups' read ports and grants
  logic [NRP-1:0] rg;
  logic [NRP-1:0][LANES-1:0][31:0] rd_prev;   // K > 2: the reference's data a cycle ago
  int unsigned cyc, mism, reads, rdx, wrs, bypass_cand;
  initial begin
    void'($urandom(SEED));
    r_en = '0; r_req = '0; r_addr = '0; w_en = '0; w_req = '0; w_gnt = '0; w_addr = '0;
    w_data = '0;
    mism = 0; reads = 0; wrs = 0;
    for (cyc = 0; cyc < CYCLES; cyc++) begin
      @(posedge clk);
      #1;                            // after the core edge, before the next fast edge (2 ns)
      // compare what the last edge produced (K > 2: with the reference's a cycle earlier)
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++)
          if ((K > 2 ? rd_prev[p][l] : rd_ref[p][l]) !== rd_dut[p][l] && (K == 2 || cyc > 0)) begin
            if (mism < 20)
              $display("MISMATCH cycle %0d port %0d lane %0d: ref %h pump %h", cyc, p, l,
                       K > 2 ? rd_prev[p][l] : rd_ref[p][l], rd_dut[p][l]);
            mism++;
          end
      rd_prev = rd_ref;
      // next cycle's requests
      begin
        logic [LANES-1:0] wbk, b;
        logic [NCP-1:0][LANES-1:0] occ;
        logic [NRP-1:0][LANES-1:0] rq;
        logic [NRP-1:0][LANES-1:0][31:0] ra;
        logic [NWP-1:0][LANES-1:0] wq;
        logic [NWP-1:0][LANES-1:0][31:0] wa;
        scat(rq[P_MXU], ra[P_MXU], 50);
        run(rq[P_DMA], ra[P_DMA], 30);
        run(rq[P_Q], ra[P_Q], 40);
        run(rq[P_VA], ra[P_VA], 50);
        run(rq[P_VB], ra[P_VB], 50);
        run(rq[P_Q2], ra[P_Q2], 30);
        run(rq[P_COLL], ra[P_COLL], 10);
        // Q3: one word (lane 0)
        rq[P_Q3] = '0; ra[P_Q3] = '0;
        if ($urandom % 100 < 20) begin rq[P_Q3][0] = 1'b1; ra[P_Q3][0] = raddr(WORDS); end
        for (int l = 1; l < LANES; l++) ra[P_Q3][l] = ra[P_Q3][0] + l;
        run(wq[W_DMA], wa[W_DMA], 30);
        scat(wq[W_MXU], wa[W_MXU], 40);
        run(wq[W_COLL], wa[W_COLL], 10);
        run(wq[W_VPU], wa[W_VPU], 50);
        // writes: the DMA always, then the others in priority order where their banks are free
        w_gnt = '0;
        wbk = '0;
        for (int k = 0; k < NWP; k++) begin
          int p;
          p = (k == 0) ? W_DMA : (k == 1) ? W_COLL : (k == 2) ? W_VPU : W_MXU;
          b = banks(wq[p], wa[p]);
          if (p == W_DMA || (b & wbk) == '0) begin
            w_gnt[p] = 1'b1;
            wbk |= b;
          end
        end
        // read grants per group: DMA, COLL, VPU (VA VB), MXU, Q (Q Q2 Q3)
        rg = '0;
        occ = '0;
        for (int g = 0; g < 5; g++) begin
          logic [NRP-1:0] ports;
          logic [NCP-1:0][LANES-1:0] need;
          logic ok;
          ports = (g == 0) ? NRP'(1) << P_DMA : (g == 1) ? NRP'(1) << P_COLL :
                  (g == 2) ? (NRP'(1) << P_VA) | (NRP'(1) << P_VB) :
                  (g == 3) ? NRP'(1) << P_MXU :
                  (NRP'(1) << P_Q) | (NRP'(1) << P_Q2) | (NRP'(1) << P_Q3);
          need = '0;
          ok = ($urandom % 100) < 90;
          for (int p = 0; p < NRP; p++)
            if (ports[p]) need[CPY3[p] % NCP] |= banks(rq[p], ra[p]);
          if (IMPL == 1)
            for (int c = 0; c < NCP; c++) if ((need[c] & occ[c]) != '0) ok = 1'b0;
          if (IMPL == 0) begin
            // the guests read only when no port before them on the DMA's copy asks
            if (ports[P_Q3] && rq[P_Q3] != '0 && rq[P_DMA] != '0) ok = 1'b0;
            if (ports[P_COLL] && rq[P_COLL] != '0 && (rq[P_DMA] != '0 || rq[P_Q3] != '0))
              ok = 1'b0;
          end
          if (ok) rg |= ports;
          // the banks taken for the groups after: the granted groups' (gnt_k), or with RQ
          // every asking group's, granted or not (otpu_slice gnt_q)
          if (ok || RQ)
            for (int c = 0; c < NCP; c++) occ[c] |= need[c];
        end
        for (int p = 0; p < NRP; p++) begin
          r_req[p] = rq[p];
          r_addr[p] = ra[p];
          r_en[p] = rg[p] ? rq[p] : '0;
          reads += $countones(r_en[p]);
        end
        for (int p = 0; p < NWP; p++) begin
          w_req[p] = wq[p];
          w_addr[p] = wa[p];
          w_en[p] = w_gnt[p] ? wq[p] : '0;
          for (int l = 0; l < LANES; l++) w_data[p][l] = $urandom;
          wrs += $countones(w_en[p]);
        end
      end
    end
    $display("TB_RESULT %s IMPL %0d NCP %0d RQ %0d WORDS %0d BYP %0d K %0d: %0d cycles, %0d lane reads, %0d lane writes, %0d mismatches",
             mism == 0 ? "PASS" : "FAIL", IMPL, NCP, RQ, WORDS, BYP, K, CYCLES, reads, wrs, mism);
    $finish;
  end
endmodule
