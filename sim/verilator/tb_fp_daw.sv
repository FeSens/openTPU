// otpu_pkg's footprint() (the sequencer's scoreboard ranges) of commands read from a file, for
// tests/test_rtl_daw.py to compare with opentpu.waits.footprint at every DRAM address width
// (build with -DOTPU_DAW=N: a DRAM range's ends are {REG, 32 bits}, DAW bits).
// +cmds=FILE: per line "op flags rg w1 w2 w3 w4 w5 w6 w7" (hex; rg the command's region REG, as
// otpu_seq decodes it). One line per command:
//   FP all=<0|1> R<i>=<v>,<sp>,<lo>,<hi> (i = 0..3) W<i>=<v>,<sp>,<lo>,<hi> (i = 0..2)
// (lo and hi in hex, DAW bits), then "FP DONE".
`timescale 1ns/1ps
module tb_fp_daw;
  import otpu_pkg::*;
  parameter int D = 128;
  parameter int S = 1;
  initial begin
    string fn;
    integer fd;
    logic [31:0] op, fl, rg, w1, w2, w3, w4, w5, w6, w7;
    cmd_t c;
    fp_t f;
    if (!$value$plusargs("cmds=%s", fn)) $fatal(1, "tb_fp_daw: +cmds=FILE");
    fd = $fopen(fn, "r");
    if (fd == 0) $fatal(1, "tb_fp_daw: cannot open %s", fn);
    while ($fscanf(fd, "%h %h %h %h %h %h %h %h %h %h", op, fl, rg, w1, w2, w3, w4, w5, w6, w7) == 10)
    begin
      c = '0;
      c.op = op[7:0]; c.flags = fl[7:0]; c.rg = RGW'(rg);
      c.w1 = w1; c.w2 = w2; c.w3 = w3; c.w4 = w4; c.w5 = w5; c.w6 = w6; c.w7 = w7;
      f = footprint(c, D, S);
      $write("FP all=%0d", f.all);
      for (int i = 0; i < 4; i++)
        $write(" R%0d=%0d,%0d,%h,%h", i, f.rd[i].v, f.rd[i].sp, 64'(f.rd[i].lo), 64'(f.rd[i].hi));
      for (int i = 0; i < 3; i++)
        $write(" W%0d=%0d,%0d,%h,%h", i, f.wr[i].v, f.wr[i].sp, 64'(f.wr[i].lo), 64'(f.wr[i].hi));
      $display("");
    end
    $fclose(fd);
    $display("FP DONE DAW=%0d", DAW);
    $finish;
  end
endmodule
