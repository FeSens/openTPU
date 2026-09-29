// Simulation model of one MIG 7-series channel's native user interface (4:1, BL8, 512-bit, ECC:
// 64-byte beats, app_addr = {rank, beat, 3'b000}), as otpu_mig_ch drives it. app_rdy and
// app_wdf_rdy are randomly withheld (+axi_stall=percent, +axi_seed=N as otpu_axi_mem) and a read's
// data comes LAT to LAT + 7 cycles after its command (+axi_lat=N), in command order and never
// held back. A command is performed when it is accepted: a read returns the memory as it was
// then, a write lands then, so the order of accesses to one address is the command order (the
// MIG keeps it). Write commands must come with their data in the same cycle (otpu_mig_ch
// does); a write with any mask bit set must be wr_bytes (011: ECC read-modify-write), a plain
// write (000) whole.
// With IMG = 1 the channel's memory loads from <dir>/ch<CH>.bin (big-endian words, as $fread
// reads them) and dumps to <dir>/ch<CH>_out.bin (little-endian) on dump, as otpu_axi_mem's PHYS
// images; beat m word k is bits 32k + 31 .. 32k.
module otpu_mig_model #(
  parameter int BEATS = 1 << 12,
  parameter int LAT   = 20,
  parameter int CH    = 0,
  parameter bit IMG   = 1'b0
) (
  input  logic         clk,
  input  logic         rst,
  input  logic [28:0]  app_addr,
  input  logic [2:0]   app_cmd,
  input  logic         app_en,
  output logic         app_rdy,
  input  logic [511:0] app_wdf_data,
  input  logic [63:0]  app_wdf_mask,
  input  logic         app_wdf_wren,
  input  logic         app_wdf_end,
  output logic         app_wdf_rdy,
  output logic [511:0] app_rd_data,
  output logic         app_rd_data_valid,
  input  logic         dump
);
  logic [511:0] mem [BEATS];
  int stall = 0, lat = LAT;
  longint cyc = 0, n_rd = 0, n_wr = 0, n_wb = 0;
  typedef struct { longint t; logic [511:0] d; } rd_t;
  rd_t rq [$];
  longint tlast = 0;

  always_ff @(posedge clk) begin
    cyc <= cyc + 1;
    app_rdy     <= !rst && (($urandom % 100) >= stall);
    app_wdf_rdy <= !rst && (($urandom % 100) >= stall);
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      rq.delete();
      app_rd_data_valid <= 1'b0;
      tlast = 0;
    end else begin
      if (app_wdf_wren && app_wdf_rdy && !(app_en && app_rdy && app_cmd != 3'b001))
        $fatal(1, "otpu_mig_model ch%0d: write data without its command", CH);
      if (app_en && app_rdy) begin
        logic [24:0] b;
        b = app_addr[27:3];
        if (app_addr[28] || app_addr[2:0] != 0) $fatal(1, "otpu_mig_model ch%0d: bad address %h", CH, app_addr);
        if (b >= BEATS) $fatal(1, "otpu_mig_model ch%0d: beat %0d beyond the memory", CH, b);
        case (app_cmd)
          3'b001: begin
            longint t;
            t = cyc + lat + ($urandom % 8);
            if (t <= tlast) t = tlast + 1;
            tlast = t;
            rq.push_back('{t, mem[b]});
            n_rd++;
          end
          3'b000, 3'b011: begin
            if (!(app_wdf_wren && app_wdf_rdy && app_wdf_end))
              $fatal(1, "otpu_mig_model ch%0d: write command without its data", CH);
            if (app_cmd == 3'b000 && app_wdf_mask != '0)
              $fatal(1, "otpu_mig_model ch%0d: masked plain write (needs wr_bytes)", CH);
            for (int k = 0; k < 64; k++)
              if (!app_wdf_mask[k]) mem[b][8 * k +: 8] <= app_wdf_data[8 * k +: 8];
            n_wr++;
            if (app_cmd == 3'b011) n_wb++;
          end
          default: $fatal(1, "otpu_mig_model ch%0d: command %b", CH, app_cmd);
        endcase
      end
      if (rq.size() != 0 && rq[0].t <= cyc) begin
        app_rd_data_valid <= 1'b1;
        app_rd_data <= rq[0].d;
        void'(rq.pop_front());
      end else begin
        app_rd_data_valid <= 1'b0;
      end
    end
  end

  string dir;
  integer fd, nread;
  logic [31:0] words [BEATS * 16];
  initial begin
    void'($value$plusargs("axi_stall=%d", stall));
    void'($value$plusargs("axi_lat=%d", lat));
    for (int i = 0; i < BEATS; i++) mem[i] = '0;
    if (IMG && $value$plusargs("dir=%s", dir)) begin
      for (int i = 0; i < BEATS * 16; i++) words[i] = '0;
      fd = $fopen($sformatf("%s/ch%0d.bin", dir, CH), "rb");
      if (fd != 0) begin
        nread = $fread(words, fd);
        $fclose(fd);
      end
      for (int i = 0; i < BEATS * 16; i++) mem[i / 16][32 * (i % 16) +: 32] = words[i];
    end
  end
  // dump may span several of this clock's cycles (it is driven in the core clock): its first
  logic dump_q = 1'b0;
  always @(posedge clk) dump_q <= dump;
  always @(posedge clk) if (dump && !dump_q) begin
    $display("MIG ch%0d rd=%0d wr=%0d wr_bytes=%0d", CH, n_rd, n_wr, n_wb);
    if (IMG) begin
      fd = $fopen($sformatf("%s/ch%0d_out.bin", dir, CH), "wb");
      for (int i = 0; i < BEATS * 16; i++) $fwrite(fd, "%u", mem[i / 16][32 * (i % 16) +: 32]);
      $fclose(fd);
    end
  end
endmodule
