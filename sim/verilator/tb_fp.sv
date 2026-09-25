// Checks rtl/vpu/otpu_fp.sv against vectors produced by opentpu/fp32.py.
// Vector file: one line per case, 4 hex words: op a b expected.
module tb_fp;
  import otpu_fp::*;
  int fd, n, op, errs, cases;
  logic [31:0] a, b, exp_v, got;
  string fname;
  initial begin
    if (!$value$plusargs("vec=%s", fname)) $fatal(1, "need +vec=");
    fd = $fopen(fname, "r");
    if (fd == 0) $fatal(1, "cannot open %s", fname);
    errs = 0; cases = 0;
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%h %h %h %h\n", op, a, b, exp_v);
      if (n != 4) break;
      case (op)
        0: got = fp_add(a, b);
        1: got = fp_sub(a, b);
        2: got = fp_mul(a, b);
        3: got = fp_max(a, b);
        4: got = fp_min(a, b);
        5: got = fp_exp2(a);
        6: got = fp_recip(a);
        7: got = fp_rsqrt(a);
        8: got = i2f(a);
        9: got = {24'd0, q8(a)};
        10: got = {31'd0, fp_gt(a, b)};
        11: got = fabs(a);
        12: got = fp_log2(a);
        default: got = 32'hDEADBEEF;
      endcase
      cases++;
      if (got !== exp_v) begin
        if (errs < 20) $display("MISMATCH op=%0d a=%08h b=%08h exp=%08h got=%08h", op, a, b, exp_v, got);
        errs++;
      end
    end
    $display("FPTEST cases=%0d errors=%0d", cases, errs);
    $finish;
  end
endmodule
