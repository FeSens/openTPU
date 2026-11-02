// openTPU as an ASIC block (docs/asic.md): one slice and its collective unit, with the slice's
// own memory ports at the boundary.
//
// The boundary is the slice's DRAM interface (otpu_slice: port A word reads and writes, the
// scalar SW writes, port B chunk reads and writes, with ready, read-valid and idle handshakes),
// not a memory controller's: which DRAM an ASIC has (LPDDR4 / DDR4 / HBM PHY and controller, a
// link to an FPGA that owns the DRAM) is the platform's, as LiteDRAM and the two-channel adapter
// (otpu_native_dram) are the YPCB-00338 card's. The block is what every openTPU chip shares; its
// area and clock are the core's PPA. The run control (reset, the loader, the run's arguments)
// and the status the board's control block (otpu_ctrl) reads are ports too.
//
// The ASIC configurations (targets/nangate45*.toml): the TMEM banked (TMEM_IMPL 1: three copies
// of 1R1W bank RAMs, one read and one write per copy and bank per cycle; nangate45-small) or one
// copy (TMEM_IMPL 3 with TMEM_TDP: two-port 1RW1R RAMs at the core clock, the card's arch since
// release r2, docs/tmem_nocopy.md 6.4); the technology RAMs on SRAM macros
// (rtl/tech/asic/otpu_tech_ram.sv); the MXU's cells behavioural (rtl/tech/generic/otpu_tech_dsp.sv).
// An ASIC does not pump its SRAM (TMEM_PMP 0): there is no 2x clock. COLL_GATHER 0: the collective
// unit without GATHER (otpu_coll_bar, the card's at one slice; a GATHER sets `error`).
module otpu_asic_top
  import otpu_pkg::*;
#(
  parameter int D          = 128,
  parameter int MCOLS      = 4,
  parameter int ACT_BLOCKS = 128,
  parameter int ACT_ROWS   = MCOLS,
  parameter int TMEM_WORDS = 1 << 16,
  parameter int IMEM_WORDS = 1 << 15,
  parameter int FIFO_DEPTH = 1024,
  parameter int LANES      = 8,
  parameter int WIN        = 16,
  parameter int MXU_IMPL   = 2,
  parameter int MXU_CL     = 16,
  parameter int MXU_E      = 0,         // the MXU's epilogue lanes (otpu_slice; 0: MCOLS / 2)
  parameter int VPU_CL     = 2,
  parameter int ULANES     = LANES,
  parameter bit HAS_DSTEP  = 1'b1,
  parameter int TMEM_IMPL  = 1,         // the TMEM (otpu_slice): 1 banked, 3 one copy
  parameter int RPB        = 1,
  parameter int WPB        = 1,
  parameter int TMEM_NSB   = 8,         // TMEM_IMPL 3's knobs (otpu_slice TMEM_*)
  parameter int TMEM_SBP   = 3,
  parameter int TMEM_SBH   = 0,
  parameter bit TMEM_RPL   = 1'b0,
  parameter bit TMEM_TDP   = 1'b0,
  parameter bit TMEM_GST   = 1'b0,
  parameter int TMEM_PMP   = 0,         // 0 only: no 2x clock
  parameter int TMEM_PRK   = 0,
  parameter int TMEM_SDM   = -1,
  parameter bit SEQ_WC     = 1'b0,      // the sequencer's wait entries (otpu_slice)
  parameter bit COLL_GATHER = 1'b1      // 0: the collective unit runs BAR alone (otpu_coll_bar)
) (
  input  logic                clk,
  input  logic                sys_rst,      // the loader's reset
  input  logic                rst,          // the run's reset (high: stopped)
  input  logic [7:0][31:0]    rinit,        // the run's arguments, R8..R15
  input  logic                ld_start,
  input  logic [31:0]         ld_addr,
  input  logic [31:0]         ld_n,
  output logic                ld_busy,
  // memory: port A (words), SW (QST byte writes), port B (chunks)
  input  logic                a_rdy,
  input  logic                b_rdy,
  input  logic                sw_rdy,
  input  logic                wr_idle,
  input  logic                rd_idle,
  output logic                a_req,        // port A's outputs come from a register (otpu_slice): a
                                            // read waits in it while a_req is high (reads only: a_we
                                            // is 0; the reads all back: see b_req)
  output logic                a_we,
  output logic [AWW-1:0]      a_addr,       // word addresses, otpu_pkg AWW bits (otpu_slice)
  output logic [31:0]         a_wdata,
  output logic [3:0]          a_be,
  output logic                sw_req,       // the SW outputs come from a register (otpu_slice), and
                                            // a write waits in it while sw_req is high (the writes
                                            // all taken: see b_req)
  output logic [AWW-1:0]      sw_addr,
  output logic [31:0]         sw_wdata,
  output logic [3:0]          sw_be,
  input  logic                a_rvalid,
  input  logic [255:0]        a_rdata,      // the aligned 32 bytes holding the word read
  output logic                b_req,        // port B's outputs come from a register, and its read
                                            // data enters through one (otpu_slice): a request waits
                                            // in it while b_req is high, read data is the block's a
                                            // cycle after its b_rvalid; every write the block gave
                                            // is taken once wr_idle && !sw_req && !(b_req && b_we),
                                            // no read is in flight once rd_idle && !a_req &&
                                            // !(b_req && !b_we) and there was no b_rvalid the
                                            // cycle before
  output logic                b_tag,
  output logic                b_we,
  output logic [D/4-1:0]      b_wmask,
  output logic [D*8-1:0]      b_wdata,
  output logic [AWW-1:0]      b_addr,
  output logic                b_par,
  input  logic                b_rvalid,
  input  logic                b_rtag,
  input  logic [D*8-1:0]      b_rdata,
  // status
  output logic                halted,
  output logic                error,
  output logic                wait_to,
  output logic                a_inval,
  output logic [31:0]         icount
);
  initial if (TMEM_PMP != 0) $fatal(1, "otpu_asic_top: TMEM_PMP %0d: an ASIC does not pump its SRAM", TMEM_PMP);
  logic [31:0] rinit_a [8];
  always_comb for (int k = 0; k < 8; k++) rinit_a[k] = rinit[k];

  logic                          coll_req, coll_ack, coll_gl, coll_gerr, s_error;
  cmd_t                          coll_cmd [1];
  logic [0:0][LANES-1:0]         coll_ren;
  logic [0:0][LANES-1:0][31:0]   coll_raddr, coll_rdata;
  logic [LANES-1:0]              coll_wen;
  logic [LANES-1:0][31:0]        coll_waddr, coll_wdata;

  otpu_slice #(.SID(0), .S(1), .D(D), .MCOLS(MCOLS), .ACT_BLOCKS(ACT_BLOCKS),
               .ACT_ROWS(ACT_ROWS), .TMEM_WORDS(TMEM_WORDS), .IMEM_WORDS(IMEM_WORDS),
               .FIFO_DEPTH(FIFO_DEPTH), .LANES(LANES), .WIN(WIN), .RPB(RPB), .WPB(WPB),
               .MXU_IMPL(MXU_IMPL), .MXU_CL(MXU_CL), .MXU_E(MXU_E), .VPU_CL(VPU_CL),
               .ULANES(ULANES), .HAS_DSTEP(HAS_DSTEP), .TMEM_IMPL(TMEM_IMPL),
               .TMEM_NSB(TMEM_NSB), .TMEM_SBP(TMEM_SBP), .TMEM_SBH(TMEM_SBH),
               .TMEM_RPL(TMEM_RPL), .TMEM_TDP(TMEM_TDP), .TMEM_GST(TMEM_GST),
               .TMEM_PMP(TMEM_PMP), .TMEM_PRK(TMEM_PRK), .TMEM_SDM(TMEM_SDM),
               .SEQ_WC(SEQ_WC)) u_slice (
    // clk_f: TMEM_PMP 0 (above), so no 2x clock
    .clk, .clk_f(1'b0), .sys_rst, .rst, .rinit(rinit_a), .ld_start, .ld_addr, .ld_n, .ld_busy,
    .a_rdy, .b_rdy, .sw_rdy, .wr_idle, .rd_idle,
    .a_req, .a_we, .a_addr, .a_wdata, .a_be, .a_rvalid, .a_rdata,
    .sw_req, .sw_addr, .sw_wdata, .sw_be,
    .b_req, .b_tag, .b_we, .b_wmask, .b_wdata, .b_addr, .b_par, .b_rvalid, .b_rtag, .b_rdata,
    .b_two(), .b_rdata2('0),
    .coll_req, .coll_cmd(coll_cmd[0]), .coll_ack,
    .coll_ren(coll_ren[0]), .coll_raddr(coll_raddr[0]), .coll_rdata(coll_rdata[0]),
    .coll_wen, .coll_waddr, .coll_wdata, .coll_gnt_local(coll_gl), .coll_gnt(coll_gl),
    .halted, .error(s_error), .wait_to, .a_inval, .icount, .pf(), .dump(1'b0));

  if (COLL_GATHER) begin : g_coll
    otpu_coll #(.S(1), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gl), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata));
    assign coll_gerr = 1'b0;
  end else begin : g_coll_bar
    otpu_coll_bar #(.S(1), .LANES(LANES)) u_coll (
      .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gl), .ack(coll_ack),
      .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
      .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata), .gerr(coll_gerr));
  end
  assign error = s_error || coll_gerr;
endmodule
