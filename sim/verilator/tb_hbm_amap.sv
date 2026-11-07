// otpu_hbm_amap (rtl/boards/sqrl-fk33/otpu_hbm_port.sv: a channel's beat -> the HBM's AXI byte address) for beats
// read from a file, for tests/test_rtl_daw.py: +beats=FILE (hex, one a line); one line "AMAP <beat> <addr>" each
// (hex), then "AMAP DONE". The parameters: the port's (AAW, CH_BASE, PCW, PCB, BAW, PC_GRAN, PC_XOR).
`timescale 1ns/1ps
module tb_hbm_amap #(
  parameter int  AAW     = 33,
  parameter logic [AAW-1:0] CH_BASE = '0,
  parameter int  PCW     = 22,
  parameter int  PCB     = 3,
  parameter int  BAW     = 25,
  parameter int  PC_GRAN = 0,
  parameter bit  PC_XOR  = 1'b1
);
  logic [BAW-1:0] beat;
  logic [AAW-1:0] addr;
  otpu_hbm_amap #(.AAW(AAW), .CH_BASE(CH_BASE), .PCW(PCW), .PCB(PCB), .BAW(BAW), .PC_GRAN(PC_GRAN),
                  .PC_XOR(PC_XOR)) u_amap (.beat, .addr);
  initial begin
    string fn;
    integer fd;
    logic [63:0] b;
    if (!$value$plusargs("beats=%s", fn)) $fatal(1, "tb_hbm_amap: +beats=FILE");
    fd = $fopen(fn, "r");
    if (fd == 0) $fatal(1, "tb_hbm_amap: cannot open %s", fn);
    while ($fscanf(fd, "%h", b) == 1) begin
      beat = BAW'(b);
      #1;
      $display("AMAP %h %h", 64'(beat), 64'(addr));
    end
    $fclose(fd);
    $display("AMAP DONE");
    $finish;
  end
endmodule
