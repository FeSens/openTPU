// The generic controller port of otpu_mem_ch (LiteDRAM native-port semantics) onto one MIG
// 7-series channel's native user interface (4:1, BL8, 512-bit, ECC: one 64-byte beat per
// command). The MIG takes a write's data with its command here, so a write command goes out
// (app_en with app_wdf_wren and app_wdf_end) only in a cycle where both app_rdy and app_wdf_rdy
// are high and its data is at the head of the write-data stream -- otpu_mem_ch puts it there no
// later than the command. app_en and app_wdf_wren are high only when they are taken.
//   app_cmd: 001 read; 000 write; 011 wr_bytes (the ECC controller's read-modify-write) when
//     c_cmd_partial says some byte is not written.
//   app_wdf_mask: 1 = keep the byte (the inverse of c_wdata_we).
//   app_addr = {rank 0, beat, 3'b000}.
// Read data comes back in command order (the MIG's read buffer reorders) and is never held back.
module otpu_mig_native (
  // generic port (otpu_mem_ch), ui_clk
  input  logic         c_cmd_valid,
  output logic         c_cmd_ready,
  input  logic         c_cmd_we,
  input  logic [24:0]  c_cmd_addr,
  input  logic         c_cmd_partial,
  input  logic         c_wdata_valid,
  output logic         c_wdata_ready,
  input  logic [511:0] c_wdata_data,
  input  logic [63:0]  c_wdata_we,
  output logic         c_rdata_valid,
  output logic [511:0] c_rdata_data,
  // MIG native interface
  output logic [28:0]  app_addr,
  output logic [2:0]   app_cmd,
  output logic         app_en,
  input  logic         app_rdy,
  output logic [511:0] app_wdf_data,
  output logic [63:0]  app_wdf_mask,
  output logic         app_wdf_wren,
  output logic         app_wdf_end,
  input  logic         app_wdf_rdy,
  input  logic [511:0] app_rd_data,
  input  logic         app_rd_data_valid
);
  assign c_cmd_ready   = app_rdy && (!c_cmd_we || (c_wdata_valid && app_wdf_rdy));
  assign app_en        = c_cmd_valid && c_cmd_ready;
  assign app_cmd       = !c_cmd_we ? 3'b001 : c_cmd_partial ? 3'b011 : 3'b000;
  assign app_addr      = {1'b0, c_cmd_addr, 3'b000};
  assign app_wdf_wren  = app_en && c_cmd_we;
  assign app_wdf_end   = app_wdf_wren;
  assign app_wdf_data  = c_wdata_data;
  assign app_wdf_mask  = ~c_wdata_we;
  assign c_wdata_ready = app_wdf_wren;
  assign c_rdata_valid = app_rd_data_valid;
  assign c_rdata_data  = app_rd_data;
endmodule
