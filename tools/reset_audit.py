"""Power-up state audit (docs/asic.md "Power-up state"): the flip-flops a reset leaves unknown.

    python3 tools/reset_audit.py [CONFIG ...] [--rstlen N] [--all]

The RTL is read as synthesis reads it (yosys with slang, -D SYNTHESIS: no simulation-only
`initial` block), flattened, with every flip-flop and memory word at X (an ASIC's power-up), and
simulated with both resets (sys_rst, rst) held for --rstlen cycles (3, the shortest the
simulation top holds sys_rst: sim/verilator/tb_top.sv) and every other input idle (yosys `sim`,
four-state). A flip-flop still X after it is one whose power-up value can outlive the reset:
each must be in ALLOW below, the data whose first use writes it (class b), or the audit fails.
The RTL's arrays are flip-flops here too (yosys memory_map: delay lines, queues, the sequencer's
windows), all but the RAMs (the tech wrappers' `mem`: data).
Control state (valids, grants, enables, FSMs, counters, replay and age state) takes a synchronous
reset instead. Simulation-only checkers (class a) are not in the netlist.

A second check reads the sources: a register a simulation-only `initial` block sets is one whose
power-up value the simulations would hide. Each such register must be X-free after the reset
in the netlist (it has a reset of its own), a memory, or simulation-only itself.

The configurations (CONFIGS): the ASIC's small point (TMEM_IMPL 1), the card's micro-architecture
(TMEM_IMPL 3 with replays, true dual-port RAMs, guests; the stream engine; SEQ_WC; the BAR-only
collective unit) with its RAMs at the core clock (the ASIC's) and pumped (PMP 1: the YPCB-00338
card's), one copy per read port (TMEM_IMPL 0: the BCU1525's, with GATHER), and bcu1525-x2:wmx's
VPU (the composites at VPU_CNS 5 with FSUM and SIGMOID / SILU, the RAMs pumped), at small sizes
(the generate branches are the full sizes'). Each runs in ~10-60 s and < 1.5 GB.

yosys: YOSYS, else yosys on PATH, with the slang plugin (-m slang) or slang built in.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL = ROOT / "rtl"
CORE = ["vpu/otpu_fp.sv", "vpu/otpu_fpipe.sv", "top/otpu_pkg.sv", "mem/otpu_tmem.sv",
        "mem/otpu_actram.sv", "seq/otpu_seq.sv", "vpu/otpu_vtree.sv", "dma/otpu_dma.sv",
        "mxu/otpu_mxu.sv", "tech/generic/otpu_tech_dsp.sv", "tech/generic/otpu_tech_ram.sv",
        "vpu/otpu_quant.sv", "vpu/otpu_se_comp.sv", "vpu/otpu_se_tail.sv", "vpu/otpu_vpu.sv",
        "top/otpu_coll.sv", "top/otpu_slice.sv"]

_SMALL = dict(D=32, MCOLS=2, ACT_BLOCKS=32, ACT_ROWS=2, TMEM_WORDS=16384, IMEM_WORDS=8192,
              FIFO_DEPTH=64, LANES=8, WIN=16, MXU_CL=16, VPU_CL=2, ULANES=8, HAS_DSTEP=1,
              RPB=64, WPB=1)
_ONE = dict(TMEM_IMPL=3, TMEM_NSB=2, TMEM_SBP=3, TMEM_RPL=1, TMEM_TDP=1, TMEM_GST=1, SEQ_WC=1,
            COLL_GATHER=0)
CONFIGS = {
    # targets/nangate45-small.toml (tools/asic/flow.py params("small"))
    "small": dict(D=32, MCOLS=2, ACT_BLOCKS=32, ACT_ROWS=2, TMEM_WORDS=16384, IMEM_WORDS=8192,
                  FIFO_DEPTH=64, LANES=4, WIN=16, MXU_CL=16, VPU_CL=1, ULANES=4, HAS_DSTEP=0,
                  TMEM_IMPL=1, RPB=1, WPB=1),
    # targets/nangate45.toml's micro-architecture (the ASIC: RAMs at the core clock)
    "card": dict(_SMALL, **_ONE, TMEM_PMP=0),
    # targets/ypcb-00338.toml's (the RAMs pumped at 2x; clk_f is clk here: X, not timing)
    "pump": dict(_SMALL, **_ONE, TMEM_PMP=1),
    # targets/bcu1525.toml's (a copy per read port), with the GATHER collective unit and the
    # VPU's MAC (bcu1525-x2:wmx's)
    "copy": dict(_SMALL, TMEM_IMPL=0, COLL_GATHER=1, VPU_MAC=1),
    # bcu1525-x2:wmx's VPU (private: no OUTER but its decay one on MAC's datapath, the composites
    # at VPU_CNS 5 with FSUM, SIGMOID / SILU and MAC) and quantizer (QACT / QST flag LO, QACT HL,
    # QST's 4-word SW port, two chunks a pop, RSCALE once a row, QST rows through the row buffer,
    # QST HL) and the DMA's ST16R / ST16W
    "wmx": dict(_SMALL, **_ONE, TMEM_PMP=1, VPU_CNS=5, VPU_FSUM=1, VPU_SIG=1, VPU_MAC=1,
                OUTER=0, VPU_OUTER1=1, QUANT_SPLIT=1, QUANT_HL=1, QUANT_SWW=4, QUANT_PW=2,
                QUANT_R1=1, QUANT_QSTB=1, QUANT_QSTHL=1, ST16=1, ST16S=1),
}

# The flip-flops a reset may leave X: (file, register regex, class, why). Class b: data -- a
# register whose value is read only once a reset-cleared valid or state says it was written (a
# command's fields and counters, loaded at its start and read while the unit is busy; datapath
# pipelines, whose valids are reset; RAM read registers, taken only for a read). Class v: a
# struct whose valid alone resets (`if (rst) m.v <= 0`, v its first field: each word's top bit),
# its other fields data under it: the audit fails if a valid is X. A register is its leaf name
# (the generate scopes' indices dropped). Some copy a reset register every cycle and
# are X only while the units' resets (srst two registers late: otpu_slice) have had one edge.
ALLOW: list[tuple[str, str, str, str]] = [
    # TMEM, the ACT RAM, the RAMs
    (r"mem/otpu_tmem\.sv", r"wi_addr|wi_data", "b",
     "a write's address and data: taken only with its reset valid (wi_en)"),
    (r"mem/otpu_tmem\.sv", r"rpa|rpe|rpl", "b",
     "a read's replay copy: loaded while not replaying, read only under rpv/rpvb (reset)"),
    (r"mem/otpu_tmem\.sv", r"q|qa|qb|rq|pd|pss|bh|bk", "b",
     "read data, the banks' addresses and the lanes' selects: steer a read's word only"),
    (r"mem/otpu_tmem\.sv", r"fresh|held", "b",
     "a port's word (fresh: this cycle's read, else held): sampled only under the reader's valid"),
    (r"mem/otpu_actram\.sv", r"[sw]_\w+_[ih]|sb|sd|wb|wd", "b",
     "a write's row, index and bytes: taken only with the write enables (reset)"),
    (r"tech/generic/otpu_tech_ram\.sv", r"q|rd|rda|rdb|wq", "b",
     "RAM read registers and the bypass's word: taken only for a read"),
    (r"tech/generic/otpu_tech_dsp\.sv", r"acc|adr|ar|br|br2|fr|m|p|p3|p7|selr|wc|whr|wlr|wpe", "b",
     "the MXU's PE / column chains: load every cycle; a row's result is taken under the MXU's "
     "reset valids (tl_v), acc cleared at a row's first block (wpe: the weight registers, by "
     "the MXU's net)"),
    # the floating-point pipelines and delay lines (their valids in otpu_vdelay, reset)
    (r"vpu/otpu_fpipe\.sv", r"m|r|qr|ro|s1|s2|s3|alr|xo", "b",
     "fmul/fadd stages and otpu_delay lines: payload only, beside valids in reset registers"),
    (r"vpu/otpu_vtree\.sv", r"qa|qb|td|tr_y1", "b",
     "the adder tree's operands: a root is used only with root_v (rsr, reset)"),
    # sequencer, slice, collective unit
    (r"seq/otpu_seq\.sv", r"ir|ir_sync", "b",
     "the fetched instruction: decoded only under ir_v (reset)"),
    (r"seq/otpu_seq\.sv", r"[pqsc]_(cmd|unit|sync|fp|pp|pr)|c_cmd_w", "b",
     "the dispatch stages' payload: read only under their stage valids (p_v, s_v, q_v, c_v)"),
    (r"seq/otpu_seq\.sv", r"m|q", "b",
     "a unit's command store and its copy: read only at the unit's start (ustart, reset)"),
    (r"seq/otpu_seq\.sv", r"lp_cnt|lp_len|stk_rem|stk_start|stk_end", "b",
     "loop state: read only with lp / below sp (reset)"),
    (r"seq/otpu_seq\.sv", r"ch_n|rl_rd", "b",
     "a CHAIN's count, an RLD's register: read with ch_req / rl_v (reset)"),
    (r"seq/otpu_seq\.sv", r"older|soh|sdepd|sfp|uq|vix|vpend|wt|wv", "b",
     "the window's per-slot state: loaded at a slot's dispatch, read only for live slots (sv) or "
     "queued starts (uq_r != uq_t)"),
    (r"top/otpu_slice\.sv", r"ld_(ch|cmp|go|iss|rows|tot)", "b",
     "the loader's command: loaded with ld_busy (sys_rst), read only while busy"),
    (r"top/otpu_slice\.sv", r"c_n", "b", "a CHAIN's count: taken with c_ld (reset)"),
    (r"top/otpu_slice\.sv", r"rl_(a|raw|mul|m|w|x|pl|ph|val)", "b",
     "the RLD pipeline's payload: each stage read under the next stage's reset valid"),
    (r"top/otpu_slice\.sv", r"b_we", "b",
     "port B's write flag: payload, read only with b_req (sys_rst)"),
    (r"top/otpu_coll\.sv", r"rows|cols|drs|seg|s|r|c|ssel|rrow|ra|rm|wrow|wseg|issuing", "b",
     "GATHER's fields: loaded in C_IDLE, read only in C_RUN (st, reset)"),
    (r"top/otpu_coll\.sv", r"asel|am|aal|as_|ar|acl|awa|ps|pr|pcl|pwa|pm|qwl|qm|qd|qlast", "b",
     "the read stages' payload: read only with av / pv / qv (reset)"),
    # DMA
    (r"dma/otpu_dma\.sv", r"is_st|is_ds|dw|de|sw|so|sm|sleft|sl_nz|sl_one|cleft|cl_nz|cl_one|"
     r"occ|oc_nf|oc_lr|cnt|wp|rp|st_fin|cb|st_wn", "b",
     "a command's fields and counters: loaded at its start, read while busy (reset)"),
    (r"dma/otpu_dma\.sv", r"ww_\w+", "b", "WAITW's fields: read only under is_ww / ww_c* (reset)"),
    (r"dma/otpu_dma\.sv", r"sc|og|oi|ss_cfg|dsw|ds_\w+|fs_\w+|f_nk|f_kn|f_ka", "b",
     "a stream's descriptor, setup and fill tables: loaded at its start or setup (ds_su), read "
     "under ds_run / ds_fill / ds_out (reset)"),
    (r"dma/otpu_dma\.sv", r"dv_(a|m|p)|w_addr|w_data|r_addr|r_bad|fr_i|pe_pos|pe_zero|o_q|"
     r"pla0|pla|pl|t_rq|ob|ow_a", "b",
     "pipeline payload and the output buffer: taken only under their valids (dv_v, t_wen, t_ren, "
     "ss_fk, pe_in, ow_e, st_pend: reset)"),
    # MXU
    (r"mxu/otpu_mxu\.sv", r"h|n|pop_e|pwf|c_\w+|p_G|p_M|act_rhi", "b",
     "the command queue's entries (pwf: pop_q's format copy): read only for q_n's live entries "
     "(reset)"),
    (r"mxu/otpu_mxu\.sv", r"i_\w+|scale_addr|srow_addr|srow_next", "b",
     "the issuer's state: loaded at i_ini, read only while i_act (reset)"),
    (r"mxu/otpu_mxu\.sv", r"dad|dal|mxl|drow|r_addr|r_bad", "b",
     "drain / RMAX / TMEM read addresses: used only with their enables (drain_go, lvm, mx_go, "
     "t_ren: reset)"),
    (r"mxu/otpu_mxu\.sv", r"m0|M0|cu0|rs0|wf0|pr0|hi0|da1|hi1|bm2|sw1|sap1|wd|wc|s|as_k", "b",
     "S0-S2 and the TQ ring's payload: m0 cleared without a pop (m0.v), the rest beside valids"),
    (r"mxu/otpu_mxu\.sv", r"r", "b",
     "otpu_skew: the operands' skews and the column ends' deskew, data beside m0.v and tl_v"),
    (r"mxu/otpu_mxu\.sv", r"acc_al|rowi|rsa|rsw|rh0|rh1", "b",
     "result rows: pushed under rv (from tl_v, reset), read under drain_go"),
    (r"mxu/otpu_mxu\.sv", r"p1_x|p1_sa|p1_sw|lo|fv|g[123]|lz[23]|[sz][123]", "b",
     "the epilogue's payload: its valids p1.v and lvm (lo.v, lo.m) reset"),
    (r"mxu/otpu_mxu\.sv", r"r1", "v",
     "the ACC entry after r0 (lanes, addresses, new values): used under r1.v (rh; u_rwv's lanes "
     "beside its valid, read under rw.v)"),
    (r"mxu/otpu_mxu\.sv", r"r_sw|alpha|mk|mq_k", "b",
     "the Sw register, ASCALE's factors and RMAX's keys: written before read (mx_have, c_asc)"),
    (r"mxu/otpu_mxu\.sv", r"iq", "b", "IQ's entries: read only for iq_n's live entries (reset)"),
    (r"mxu/otpu_mxu\.sv", r"dw_n|dw_lst", "b",
     "the drain's weight-group count and last (GA): loaded at a drain's start, read while it "
     "drains (d_fin, ncnt: reset)"),
    (r"mxu/otpu_mxu\.sv", r"wpe", "b",
     "the weights' decode pipeline: data, into products beside m0.v / tl_v (reset)"),
    # quantizer
    (r"vpu/otpu_quant\.sv", r"G|GW|ab|badr|brs|csb|csf|dr|drs|dup|e|e1c|es|is_st|pass|r|r1|"
     r"rd_done|rd_wait|rem|row_src|rowm|rows|rsb|rsf|srs|sv|sw_i|swn|wide|xocc|xr|xw|"
     r"lom|wsw|hlm|abh|abk|sbase|sstep|hldr", "b",
     "a command's fields and counters: loaded at its start, read while busy (reset; LO, the "
     "SW port's wide write, HL and lo's blocks too; QST's scale row address and QSTHL's lo "
     "offset, which QSTB's pops' address line reads every cycle, taken only with mb2.v)"),
    (r"vpu/otpu_quant\.sv",
     r"gb|grb|gkp|gln|gno|gcol|gm_n|gm_ko|gm_k1|gm_k|gar|gm_eo|gm_e1|gm_e|glb",
     "b", "QST GRP's fields, addresses and start products: loaded at its start, read while busy "
     "(reset; the scale area gar and its product also by QSTB's pops' address line; QSTHL's lo "
     "pops' chunk glb, gb + HLD every cycle, read only at a lo pop)"),
    (r"vpu/otpu_quant\.sv", r"m0|mb1|mb2|mh|mc|wqp", "v",
     "the chunks' and the write's fields (wqp: wq_r): every use under its .v (gw_go, sq_iv, "
     "the maxes, the writer's selection, QSTB's m_ss / m_hl / baddr, QONCE's q_bw, "
     "otpu_vdelay's valid)"),
    (r"vpu/otpu_quant\.sv", r"amx_r|gacc", "b",
     "running maxes: restarted at a row's / group's first chunk"),
    (r"vpu/otpu_quant\.sv", r"r_addr|r_bad|t_raddr3", "b",
     "TMEM read addresses: used only with t_ren* (reset)"),
    (r"vpu/otpu_quant\.sv", r"amax_q|itag_q|ax|y0|kx|zero|big|ek|r3|x1|x2|cmax|ca|grd_d|dq|e1|"
     r"b1|dal|xb1|xb2|wx_r|winv_r|qm|lm|qb|hq|gw_a|gw_d", "b",
     "the scale unit's, prescale's and writer's datapath: beside the reset valids (iv_q, mh.v, "
     "mc.v, wq_r.v, otpu_vdelay; lm, hq: LO's lanes and HL's lo select beside u_wqv)"),
    (r"vpu/otpu_quant\.sv", r"amx_q|sc_q|inv_q|invl", "b",
     "the rows' scales and maxes: written under sq_ov before a pass 1 reads them"),
    # stream engine
    (r"vpu/otpu_se_comp\.sv", r"lp_m|c_om|c_ometa|uc_rt|lp_st|uco", "b",
     "the composite's chunk meta and lane state: read only under vpos / cpos / ret (reset)"),
    (r"vpu/otpu_se_comp\.sv", r"rr_e|rr_g|rr_f|e1|e2|g1|g2|f1|q0|q1|sti", "b",
     "the lanes' operands and their steering"),
    (r"vpu/otpu_se_comp\.sv", r"xr", "b",
     "SIG: SILU's A, a chunk's slot written as it enters, read at its pass 3"),
    (r"vpu/otpu_se_comp\.sv", r"ra|rb|rc", "b",
     "CNS 5's internal stages' operands (beside vpos, reset)"),
    (r"vpu/otpu_se_tail\.sv", r"ns|c_\w+|b1|gk|e|beta|b1_b|gk_b|e_b|beta_b", "b",
     "the stream's mode and K0 / K1 (DSTEP2: and its pair's head b's, _b): from cfg every cycle "
     "(ss_cfg loaded with ss_req) or the fill, before ss_act"),
    (r"vpu/otpu_se_tail\.sv", r"s_xa|s_xd|d1a|d2a|d2c|d3a|ra|rg|rc|re|qa|qb|qc|qc2|d_cur|fbq|dq|"
     r"gb|kb|kb2|vb|rb|qy", "b",
     "X / D / U / Q operands, the d FIFO and the fill's buffers: beside the reset valids (x0, "
     "otpu_vdelay, dq_n, warm)"),
    # VPU
    (r"vpu/otpu_vpu\.sv", r"dst|b|imm|rows|cols|drs|ars|brs|func|bmode|ir|ic|nch|ch|a_row|b_row|"
     r"d_row|nfill|oc_a|od_a|od_sc|od_one", "b",
     "a command's fields and counters: loaded at its start, read while issuing / filling or "
     "reducing (reset)"),
    (r"vpu/otpu_vpu\.sv", r"(func|tag|imm|rows|drs|dst|od_one|od_sc)_b", "b",
     "the back end's copies (FB): read under red_act_b / flushing (tag_b copies the reset tag "
     "every cycle)"),
    (r"vpu/otpu_vpu\.sv", r"r_addr|r_bad|w_addr|w_data|hrot", "b",
     "TMEM request addresses and write data: used only with their enables (reset)"),
    (r"vpu/otpu_vpu\.sv", r"ta_d|tb_d|xa|xb|xc|xd|ra|rb|re|sr|dr|nr|p0|p1|sti|[eg][12]|fbq|cq|"
     r"cbuf|dbuf|wq", "b",
     "read data, the lanes' datapath, and the queues and buffers behind reset counts / pointers"),
    (r"vpu/otpu_vpu\.sv", r"rdb|rows_done|wr_row|fl|mx_have|mx_run|mx_ri|mxh_[vhic]", "b",
     "the reductions' state and the RMAX tree's values: loaded at begr_b, read under red_act_b / "
     "flushing (the tree's meta, mxh_m, reset)"),
    (r"vpu/otpu_vpu\.sv", r"am[1-4]_[almisxz]", "b", "ARGMAX's payload: read under am4_v (reset)"),
    (r"vpu/otpu_vpu\.sv", r"mxh_m", "v",
     "the RMAX tree's chunk meta: read only through live() (its v and tag)"),
    (r"vpu/otpu_vpu\.sv", r"s1_fd|s1_fi|s1_in_d|s1_in_v|s_fd|s_in_d|s_in_v|s_cfg", "b",
     "the stream's inputs: from ss_* every cycle (the valids from reset ones), the tail reading "
     "them under pe / fk (reset)"),
    (r"vpu/otpu_vpu\.sv", r"fs_imm|fs_rows", "b",
     "FSUM's sums' address and rows: loaded with h_fs, read only under fs_act / fs_b (reset)"),
    (r"vpu/otpu_vpu\.sv", r"q_rl|q_sub|q_pd|kq", "b",
     "Q's partials and the roots' kinds: read under q_cap / root_v (reset)"),
    (r"vpu/otpu_vpu\.sv", r"acc|oa|ob|s1|ds|fs_pd|fs_sub|fs_rl", "b",
     "FSUM's partials: acc read only after a write in its row (in_f: +0 on the first chunks), "
     "the adders' operands and chunk beside s1v / vsr, the drain's sub and words under d_on / "
     "fs_cap (reset)"),
]


def yosys() -> str:
    y = os.environ.get("YOSYS") or shutil.which("yosys")
    if not y:
        raise RuntimeError("no yosys: set YOSYS or put yosys (with the slang plugin) on PATH")
    return y


def harness(p: dict) -> str:
    """A reset-only top: one slice and its collective unit (as rtl/asic/otpu_asic_top.sv), both
    resets on one input, every other input idle, clk_f on clk."""
    params = ", ".join(f".{k}({v})" for k, v in p.items() if k != "COLL_GATHER")
    coll = "otpu_coll" if p.get("COLL_GATHER", 1) else "otpu_coll_bar"
    gerr = "" if p.get("COLL_GATHER", 1) else ", .gerr()"
    lanes = p["LANES"]
    return f"""
module otpu_xrst_top import otpu_pkg::*; (input logic clk, input logic rst, output logic halted);
  logic coll_req, coll_ack, coll_gl;
  cmd_t coll_cmd [1];
  logic [0:0][{lanes}-1:0] coll_ren;
  logic [0:0][{lanes}-1:0][31:0] coll_raddr, coll_rdata;
  logic [{lanes}-1:0] coll_wen;
  logic [{lanes}-1:0][31:0] coll_waddr, coll_wdata;
  logic [31:0] rinit [8];
  always_comb for (int k = 0; k < 8; k++) rinit[k] = '0;
  otpu_slice #(.SID(0), .S(1), {params}) u_slice (
    .clk, .clk_f(clk), .sys_rst(rst), .rst, .rinit, .ld_start(1'b0), .ld_addr('0), .ld_n('0),
    .ld_busy(), .a_rdy(1'b1), .b_rdy(1'b1), .sw_rdy(1'b1), .wr_idle(1'b1), .rd_idle(1'b1),
    .a_req(), .a_we(), .a_addr(), .a_wdata(), .a_be(), .a_rvalid(1'b0), .a_rdata('0),
    .sw_req(), .sw_addr(), .sw_wdata(), .sw_be(),
    .b_req(), .b_tag(), .b_we(), .b_wmask(), .b_wdata(), .b_addr(), .b_par(), .b_rvalid(1'b0),
    .b_rtag(1'b0), .b_rdata('0),
    .coll_req, .coll_cmd(coll_cmd[0]), .coll_ack, .coll_ren(coll_ren[0]),
    .coll_raddr(coll_raddr[0]), .coll_rdata(coll_rdata[0]), .coll_wen, .coll_waddr, .coll_wdata,
    .coll_gnt_local(coll_gl), .coll_gnt(coll_gl), .halted, .error(), .wait_to(), .a_inval(),
    .icount(), .pf(), .dump(1'b0));
  {coll} #(.S(1), .LANES({lanes})) u_coll (
    .clk, .rst, .req(coll_req), .cmds(coll_cmd), .gnt(coll_gl), .ack(coll_ack),
    .r_en(coll_ren), .r_addr(coll_raddr), .r_data(coll_rdata),
    .w_en(coll_wen), .w_addr(coll_waddr), .w_data(coll_wdata){gerr});
endmodule
"""


# the arrays a configuration's netlist holds, but the RAMs (otpu_ram_* and the ACT RAM's: `mem`)
ARRAYS = "otpu_xrst_top/t:$mem_v2 otpu_xrst_top/*.mem %d"


def run_yosys(name: str, rstlen: int, work: Path) -> Path:
    """The netlist after the reset (flip-flops' init attributes: their state), as JSON."""
    p = CONFIGS[name]
    top = work / f"{name}_top.sv"
    top.write_text(harness(p))
    out = work / f"{name}.json"
    files = " ".join(str(RTL / f) for f in CORE) + f" {top}"
    script = (f"read_slang -D SYNTHESIS --ignore-assertions --top otpu_xrst_top {files}; "
              "hierarchy -top otpu_xrst_top; proc; flatten; "
              # the registers' names: slang's flip-flops before any optimisation ($driver$<name>)
              f"tee -q -o {work / (name + '_regs.txt')} select -list t:$dff t:$dffe t:$sdff "
              "t:$sdffe t:$adff t:$adffe t:$aldff; opt_expr; opt_clean; opt_dff; "
              "opt_clean; memory_collect; memory_nordff; opt_clean; "
              # every array but the RAMs as flip-flops (their sources: memory_map keeps none)
              f"tee -q -o {work / (name + '_mems.txt')} printattrs {ARRAYS}; memory_map {ARRAYS}; "
              f"opt_clean; sim -clock clk -reset rst -rstlen {rstlen} -n {rstlen} -w -q; "
              f"write_json {out}")
    _yosys(script, work / f"{name}.log", name)
    return out


def _yosys(script: str, log: Path, name: str) -> None:
    ys = yosys()
    r = subprocess.run([ys, "-m", "slang", "-q", "-l", str(log), "-p", script],
                       capture_output=True, text=True)
    if r.returncode != 0 and "Can't load module" in r.stdout + r.stderr:   # slang built in
        r = subprocess.run([ys, "-q", "-l", str(log), "-p", script], capture_output=True,
                           text=True)
    if r.returncode != 0:
        raise RuntimeError(f"yosys failed ({name}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


FF = {"$dff", "$dffe", "$sdff", "$sdffe", "$sdffce", "$adff", "$adffe", "$aldff", "$aldffe",
      "$dffsr", "$dffsre"}


def registers(path: Path, top: str = "otpu_xrst_top") -> dict:
    """{(file, register): {"x": X bits, "n": bits, "lines": {line}, "inst": {scope}}} of every
    flip-flop after the reset. A register is its RTL name's leaf (generate indices dropped);
    file is under rtl/."""
    m = json.loads(path.read_text())["modules"][top]
    regf = path.with_name(path.stem + "_regs.txt")
    rnames = {ln.split("$driver$", 1)[1].strip() for ln in regf.read_text().splitlines()
              if "$driver$" in ln} if regf.exists() else set()
    # the mapped arrays' sources (printattrs: the name, then its attributes)
    memf = path.with_name(path.stem + "_mems.txt")
    msrc, cur = {}, None
    for ln in memf.read_text().splitlines() if memf.exists() else []:
        if not ln.startswith(" "):
            cur = ln.strip().lstrip("\\")
        elif cur and "src=" in ln:
            msrc[cur] = ln.split('src="', 1)[1].split('"', 1)[0].split("|")[0]
    init, name, wsrc, top = {}, {}, {}, {}
    for n, nn in m["netnames"].items():
        a = nn.get("attributes", {})
        hid = nn.get("hide_name", 0)
        v = a.get("init")
        for i, b in enumerate(nn["bits"]):
            if not isinstance(b, int):
                continue
            if v is not None:
                init[b] = v[len(v) - 1 - i] if i < len(v) else "x"
            if not hid and "src" in a:
                wsrc.setdefault(b, a["src"].split("|")[0])
            # the register's own name (its variable's: a flip-flop's before optimisation), else
            # a public one, the shortest
            key = (n not in rnames, hid, "$" in n, len(n))
            if b not in name or key < name[b][1]:
                name[b] = (n, key)
                top[b] = i == len(nn["bits"]) - 1      # the word's top bit (a struct's v)
    regs: dict = {}
    for cn, c in m["cells"].items():
        if c["type"] not in FF:
            continue
        csrc = c.get("attributes", {}).get("src", "").split("|")[0]
        for b in c["connections"]["Q"]:
            if not isinstance(b, int):
                continue
            n = name.get(b, (cn, 1))[0]
            src = re.sub(r"^.*?rtl/", "", csrc or wsrc.get(b, "")
                         or msrc.get(re.sub(r"(\[\d+\])+$", "", n).lstrip("\\"), ""))
            f, _, ln = src.partition(":")
            leaf = re.sub(r"\[\d+\]", "", n.split(".")[-1])
            r = regs.setdefault((f, leaf), dict(x=0, n=0, xv=0, nv=0, lines=set(), inst=set()))
            r["n"] += 1
            r["nv"] += top.get(b, False)
            r["lines"].add(ln.split(".")[0])
            r["inst"].add(n.rsplit(".", 1)[0] if "." in n else "")
            if init.get(b, "x") == "x":
                r["x"] += 1
                r["xv"] += top.get(b, False)
    return regs


def allowed(f: str, reg: str) -> tuple[str, str] | None:
    for af, ar, cls, why in ALLOW:
        if re.fullmatch(af, f) and re.fullmatch(ar, reg):
            return cls, why
    return None


# a simulation-only block: `ifndef SYNTHESIS ... `else / `endif (one level: the RTL nests none)
_SIMONLY = re.compile(r"`ifndef\s+SYNTHESIS\b(.*?)(?:`else\b.*?)?`endif", re.S)
_INIT = re.compile(r"\binitial\b\s*(begin\b.*?\bend\b|[^;]*;)", re.S)
_LHS = re.compile(r"(?:^|[;\s(])([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^]=]*\]\s*)*=(?!=)")


def sim_inits() -> dict:
    """{file: {register}} that a simulation-only `initial` block assigns."""
    out: dict = defaultdict(set)
    for f in CORE:
        src = (RTL / f).read_text()
        src = re.sub(r"//[^\n]*", "", src)
        for blk in _SIMONLY.finditer(src):
            for ini in _INIT.finditer(blk.group(1)):
                for lhs in _LHS.finditer(ini.group(1)):
                    out[f].add(lhs.group(1))
    return out


def audit(names: list[str], rstlen: int, work: Path, show_all: bool = False) -> list[str]:
    """The failures (empty: the audit passes); prints the table."""
    fails = []
    seen: dict = {}
    for name in names:
        regs = registers(run_yosys(name, rstlen, work))
        for (f, reg), r in sorted(regs.items()):
            k = seen.setdefault((f, reg), dict(x=0, n=0, xv=0, nv=0, cfg=[], lines=set()))
            for c in ("x", "n", "xv", "nv"):
                k[c] += r[c]
            k["lines"] |= r["lines"]
            if r["x"]:
                k["cfg"].append(name)
    inits = sim_inits()
    for (f, reg), k in sorted(seen.items()):
        a = allowed(f, reg)
        lines = ",".join(sorted(k["lines"] - {""}, key=lambda s: int(s) if s.isdigit() else 0))
        if k["x"] and a is None:
            fails.append(f"{f}:{lines} {reg}: {k['x']}/{k['n']} bits X after the reset "
                         f"({' '.join(k['cfg'])}), not in ALLOW")
        if a is not None and a[0] == "v" and (k["xv"] or not k["nv"]):
            fails.append(f"{f}:{lines} {reg}: {k['xv']}/{k['nv']} valids (its words' top bits) X "
                         f"after the reset ({' '.join(k['cfg'])}): class v resets its .v")
        if k["x"] and reg in inits.get(f, ()):
            fails.append(f"{f}:{lines} {reg}: a simulation-only initial sets it, X after the "
                         f"reset in the netlist (give it a reset)")
        if show_all or (k["x"] and a is None):
            print(f"{'X' if k['x'] else '-'} {f}:{lines} {reg} {k['x']}/{k['n']} "
                  f"{a[0] if a else ''} {' '.join(k['cfg'])}")
    for af, ar, cls, why in ALLOW if set(names) >= set(CONFIGS) else []:   # (every config's)
        if not any(re.fullmatch(af, f) and re.fullmatch(ar, reg) and k["x"]
                   for (f, reg), k in seen.items()):
            fails.append(f"ALLOW {af} {ar}: matches no register X after the reset (stale)")
    return fails


def fpga_init_resets(name: str, work: Path, fpga_init: bool) -> tuple[dict, set]:
    """A configuration's slice read as synthesis reads it and optimised (yosys opt: a constant
    flip-flop and a constant reset go), with OTPU_FPGA_INIT (the FPGA flows' define: no memories'
    power-on reset, otpu_slice rst_mem) or without (the ASIC's and the simulations'):
    ({file under rtl/: synchronous-reset flip-flop bits (a reset, or any load of a constant yosys
    folds into one)}, {the memories' reset register left in the slice (rst_mem, which TMEM and the
    ACT RAM take as their rst), by leaf name})."""
    tag = f"{name}_{'fpga' if fpga_init else 'asic'}"
    top = work / f"{tag}_top.sv"
    top.write_text(harness(CONFIGS[name]))
    out = work / f"{tag}.json"
    files = " ".join(str(RTL / f) for f in CORE) + f" {top}"
    d = "-D OTPU_FPGA_INIT " if fpga_init else ""
    _yosys(f"read_slang -D SYNTHESIS {d}--ignore-assertions --top otpu_xrst_top {files}; "
           f"hierarchy -top otpu_xrst_top; proc; flatten; opt; write_json {out}",
           work / f"{tag}.log", tag)
    m = json.loads(out.read_text())["modules"]["otpu_xrst_top"]
    names: dict = defaultdict(set)
    for n, nn in m["netnames"].items():
        r = re.search(r"(?:^|\.)(rst_mem)$", n)
        for b in nn["bits"] if r else []:
            names[b].add(r.group(1))
    n: dict = defaultdict(int)
    regs = set()
    for c in m["cells"].values():
        if c["type"] not in FF:
            continue
        src = c.get("attributes", {}).get("src", "").split("|")[0]
        f = re.sub(r"^.*?rtl/", "", src).partition(":")[0]
        if c["type"] in ("$sdff", "$sdffe", "$sdffce"):
            n[f] += len(c["connections"]["Q"])
        regs |= {r for b in c["connections"]["Q"] for r in names.get(b, ())}
    return dict(n), regs


TB_TMEM = ROOT / "sim" / "yosys" / "tb_tmem_x.sv"


def tmem_traffic(work: Path, cycles: int = 600, no_rst: bool = False,
                 tmem: Path | None = None) -> dict:
    """TMEM under random traffic from an unknown power-up (sim/yosys/tb_tmem_x.sv, yosys's
    four-state sim): {"x": cycles after the reset with an X in lost / frz / blk_x / blk_s,
    "xd": with an X in a filled lane's word, "lost": lost reads, "frz": replays, "run": the
    traffic's cycles}. no_rst: TMEM's reset port left open (X: the RTL without it); tmem: another
    otpu_tmem.sv (with no_rst, one without the port)."""
    vcd = work / ("tmem_x_norst.vcd" if no_rst else "tmem_x.vcd")
    files = [RTL / "tech/generic/otpu_tech_ram.sv", tmem or RTL / "mem/otpu_tmem.sv", TB_TMEM]
    script = (f"read_slang -D SYNTHESIS {'-D NO_RST ' if no_rst else ''}"
              f"--ignore-assertions --top tb_tmem_x {' '.join(map(str, files))}; "
              "hierarchy -top tb_tmem_x; proc; flatten; opt_clean; memory_collect; "
              f"memory_nordff; opt_clean; sim -clock clk -reset rst -rstlen 4 -n {cycles} "
              f"-vcd {vcd} -q")
    _yosys(script, work / (vcd.stem + ".log"), "tmem_x")
    # the top's signals in the VCD: id -> name, then each time's values once its changes are in
    want = {"rst", "run", "lost", "frz", "blk_x", "blk_s", "rdx"}
    ids: dict = {}
    depth = 0
    val: dict = {}
    out = dict(x=0, xd=0, lost=0, frz=0, run=0)

    def step():
        if val.get("rst", "1") != "0" or len(val) < len(want):
            return
        if any("x" in val[k] or "z" in val[k] for k in ("lost", "frz", "blk_x", "blk_s")):
            out["x"] += 1
        if "x" in val["rdx"] or "z" in val["rdx"]:
            out["xd"] += 1
        if val["run"] == "1":
            out["run"] += 1
            out["lost"] += val["lost"].count("1")
            out["frz"] += val["frz"].count("1")
    with open(vcd) as fh:
        for ln in fh:
            t = ln.split()
            if not t:
                continue
            if t[0] == "$scope":
                depth += 1
            elif t[0] == "$upscope":
                depth -= 1
            elif t[0] == "$var" and depth == 1 and t[4] in want:
                ids[t[3]] = t[4]
            elif t[0].startswith("#"):
                step()
            elif t[0][0] in "bB" and len(t) == 2 and t[1] in ids:
                val[ids[t[1]]] = t[0][1:].lower()
            elif t[0][0] in "01xXzZ" and t[0][1:] in ids:
                val[ids[t[0][1:]]] = t[0][0].lower()
    step()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--rstlen", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="print every register, not only failures")
    ap.add_argument("--keep", help="keep the netlists in this directory")
    ap.add_argument("--tmem", action="store_true",
                    help="also TMEM under random traffic (sim/yosys/tb_tmem_x.sv): no X after the "
                         "reset, and X without it (the check sees the old RTL's fault)")
    a = ap.parse_args(argv)
    work = Path(a.keep) if a.keep else Path(tempfile.mkdtemp(prefix="otpu_rst_"))
    work.mkdir(parents=True, exist_ok=True)
    try:
        fails = audit(a.configs, a.rstlen, work, a.all)
        if a.tmem:
            t, t0 = tmem_traffic(work), tmem_traffic(work, no_rst=True)
            print(f"tmem traffic: {t} (without its reset: {t0})")
            if t["x"] or t["xd"] or t["lost"] < 100 or t["frz"] < 100:
                fails.append(f"TMEM traffic: {t}: X after the reset, or too few replays")
            if not t0["x"]:
                fails.append(f"TMEM traffic without its reset: {t0}: no X (the check is blind)")
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print(f"reset audit: {len(fails)} failures ({' '.join(a.configs)}, rstlen {a.rstlen})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
