// TMEM against a reference model under random traffic. Each cycle every port requests a random
// lane set (distinct banks within a port), a random subset of units is granted (writers only
// while their banks stay free, as the slice's arbiter does), and the reads aim mostly at words
// written in the last two cycles, so the registered-write bypass is exercised. A read returns
// the word as of its own cycle (writes of the same cycle are not seen) the cycle after it is
// granted, and the lane holds that value until its next granted read.
module tb_tmem;
  localparam int WORDS = 256, LANES = 4, NRP = 3, NWP = 2, WPB = 1, CYCLES = 20000;
  localparam int BW = $clog2(LANES);

  logic clk = 0;
  logic [NRP-1:0][LANES-1:0]       r_en, r_req;
  logic [NRP-1:0][LANES-1:0][31:0] r_addr, r_data;
  logic [NWP-1:0][LANES-1:0]       w_en, w_req;
  logic [NWP-1:0]                  w_gnt;
  logic [NWP-1:0][LANES-1:0][31:0] w_addr, w_data;

  otpu_tmem #(.WORDS(WORDS), .LANES(LANES), .NRP(NRP), .NWP(NWP), .WPB(WPB)) dut (
    .clk, .r_en, .r_req, .r_addr, .r_data, .w_en, .w_req, .w_gnt, .w_addr, .w_data,
    .dump(1'b0));

  logic [31:0] ref_mem [WORDS];
  logic [31:0] expect_q [NRP][LANES];
  bit          valid_q [NRP][LANES];
  int unsigned recent [8];              // addresses written in the last cycles
  int unsigned last_wrote [$];          // the previous cycle's writes (still registered in TMEM)
  int          hits = 0, checks = 0;

  function automatic int unsigned pick_addr(int bank);
    int unsigned a;
    if ($urandom_range(3) != 0) begin
      a = recent[$urandom_range(7)];
      a = (a & ~(LANES - 1)) | bank;    // same row neighbourhood, forced to the lane's bank
    end else
      a = ($urandom_range(WORDS / LANES - 1) * LANES) | bank;
    return a % WORDS;
  endfunction

  initial begin
    for (int i = 0; i < WORDS; i++) ref_mem[i] = '0;
    for (int i = 0; i < 8; i++) recent[i] = 0;
    for (int p = 0; p < NRP; p++) for (int l = 0; l < LANES; l++) valid_q[p][l] = 0;
    r_en = '0; r_req = '0; r_addr = '0; w_en = '0; w_req = '0; w_gnt = '0; w_addr = '0; w_data = '0;

    for (int cyc = 0; cyc < CYCLES; cyc++) begin
      logic [LANES-1:0] busy;
      int unsigned      wrote [$];
      // ---- writes: each port's lanes take a random permutation of banks
      busy = '0;
      for (int p = 0; p < NWP; p++) begin
        int perm [LANES];
        logic [LANES-1:0] banks;
        for (int l = 0; l < LANES; l++) perm[l] = l;
        perm.shuffle();
        banks = '0;
        for (int l = 0; l < LANES; l++) begin
          w_req[p][l] = $urandom_range(2) != 0;
          w_addr[p][l] = pick_addr(perm[l]);
          w_data[p][l] = $urandom;
          if (w_req[p][l]) banks[perm[l]] = 1'b1;
        end
        w_gnt[p] = $urandom_range(3) != 0 && (banks & busy) == '0;
        if (w_gnt[p]) busy |= banks;
        w_en[p] = w_gnt[p] ? w_req[p] : '0;
      end
      // ---- reads
      for (int p = 0; p < NRP; p++) begin
        int perm [LANES];
        bit g;
        for (int l = 0; l < LANES; l++) perm[l] = l;
        perm.shuffle();
        g = $urandom_range(3) != 0;
        for (int l = 0; l < LANES; l++) begin
          r_req[p][l] = $urandom_range(2) != 0;
          r_addr[p][l] = pick_addr(perm[l]);
          r_en[p][l] = g && r_req[p][l];
        end
      end
      #1;
      // ---- the reads see the memory before this cycle's writes
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++)
          if (r_en[p][l]) begin
            foreach (last_wrote[i])
              if (last_wrote[i] == r_addr[p][l]) begin hits++; break; end
            expect_q[p][l] = ref_mem[r_addr[p][l]];
            valid_q[p][l] = 1;
          end
      for (int p = 0; p < NWP; p++)
        for (int l = 0; l < LANES; l++)
          if (w_en[p][l]) begin
            ref_mem[w_addr[p][l]] = w_data[p][l];
            wrote.push_back(w_addr[p][l]);
          end
      for (int i = 0; i < wrote.size() && i < 8; i++) recent[(cyc * 3 + i) % 8] = wrote[i];
      last_wrote = wrote;
      clk = 1; #1; clk = 0; #1;
      // ---- every lane shows its last granted read
      for (int p = 0; p < NRP; p++)
        for (int l = 0; l < LANES; l++)
          if (valid_q[p][l]) begin
            checks++;
            if (r_data[p][l] !== expect_q[p][l]) begin
              $display("FAIL cycle %0d port %0d lane %0d: got %08x want %08x", cyc, p, l,
                       r_data[p][l], expect_q[p][l]);
              $fatal(1);
            end
          end
    end
    if (hits < 1000) $fatal(1, "only %0d reads of the previous cycle's writes", hits);
    $display("PASS checks=%0d bypass_reads=%0d", checks, hits);
    $finish;
  end
endmodule
