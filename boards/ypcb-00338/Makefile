# openTPU on the Inspur YPCB-00338.
#   make lint       offline: Tcl syntax, XDC ports vs the top, Verilator lint of the build
#                   (otpu_fpga_top_ld; lint-ld is the same)
#   make bit        Vivado batch build (MCOLS=2|4, VPU_CL=2|4, LANES=8|16, CORE_MHZ=133.33|100, FAST=1) -> build/vivado/otpu.bit (VIVADO_DOCKER=image for Docker)
#                   the LiteDRAM core (DDR3-1066, host calibration), MCOLS=4, the systolic MXU
#   make impl       rerun implementation only on the existing project (constraint changes)
#   make program    JTAG load (openFPGALoader; BIT=path, default build/vivado/otpu.bit); make program-vivado with Vivado's hw_server
#   make flash      write the BPI flash (permanent; MCS=path, default build/vivado/otpu.mcs)
ROOT := $(abspath ../..)
DDR ?= 1066
# DDR: 1066, the speed the LiteDRAM core is generated for (the fastest in spec for these HR banks
# at 1.5 V, docs/board.md section 5)
# CORE_MHZ: accelerator clock, default 133.33 (800 / 6: the 128-byte port matches both DDR3-1066
#   channels, 17.07 GB/s; the production image closes it at full effort) or 100 with FAST=1 (a
#   development build: Vivado's default strategies)
# MCOLS: MXU columns, 2 or 4 (the host reads it from the bitstream's VERSION register)
# VPU_CL: VPU lanes with exp2/recip/rsqrt, 2 or 4 (timing only: the programs do not change)
# LANES: VPU lanes / TMEM banks, 8 or 16 (timing only; the host reads it from VERSION)
# ACT_ROWS: ACT RAM rows, the most rows of one MM (default MCOLS; 8: prefill's 8-row MMs stream
#   each weight once, the MXU replaying it per MCOLS rows)
# BUILD_ID: the BUILD_ID register (8 hex digits; default: the git commit)
ifeq ($(FAST),1)
CORE_MHZ ?= 100
else
CORE_MHZ ?= 133.33
endif
MCOLS ?= 4
VPU_CL ?= 2
LANES ?= 8
ACT_ROWS ?= $(MCOLS)
RTL := $(addprefix $(ROOT)/rtl/, vpu/otpu_fp.sv vpu/otpu_fpipe.sv top/otpu_pkg.sv \
  mem/otpu_tmem.sv mem/otpu_actram.sv seq/otpu_seq.sv vpu/otpu_vtree.sv \
  dma/otpu_dma.sv mxu/otpu_mxu.sv vpu/otpu_quant.sv vpu/otpu_se_comp.sv vpu/otpu_se_tail.sv \
  vpu/otpu_vpu.sv \
  top/otpu_coll.sv top/otpu_slice.sv \
  boards/ypcb-00338/otpu_ctrl.sv boards/ypcb-00338/otpu_trace.sv boards/ypcb-00338/otpu_board.sv \
  mem/otpu_native_dram.sv boards/ypcb-00338/otpu_afifo.sv boards/ypcb-00338/otpu_mem_ch.sv \
  boards/ypcb-00338/otpu_axi_split2.sv boards/ypcb-00338/otpu_xmon.sv \
  boards/ypcb-00338/otpu_native_sys.sv)
# MXU_IMPL: the MXU create_project.tcl builds (MXU systolic, the default: 2); the systolic cells'
# DSP48E1 comes from a lint stub
MXU_IMPL ?= 2
LINT := verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -DSYNTHESIS -GMCOLS=$(MCOLS) -GACT_ROWS=$(ACT_ROWS) -GVPU_CL=$(VPU_CL) -GLANES=$(LANES) \
  -GMXU_IMPL=$(MXU_IMPL) sim/unisim_dsp48e1_stub.sv

.PHONY: lint lint-ld bit impl program program-vivado flash
lint:
	python3 scripts/check_offline.py
	$(LINT) --top-module otpu_fpga_top_ld $(RTL) \
	  sim/otpu_bd_ld_wrapper_stub.sv sim/otpu_litedram_stub.sv $(ROOT)/rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv

lint-ld: lint

bit:
	MCOLS=$(MCOLS) ACT_ROWS=$(ACT_ROWS) VPU_CL=$(VPU_CL) LANES=$(LANES) CORE_MHZ=$(CORE_MHZ) $(if $(BUILD_ID),BUILD_ID=$(BUILD_ID)) ./run_vivado.sh $(DDR)

impl:
	STEP=impl ./run_vivado.sh $(DDR)

program:
	./program.sh $(BIT)

program-vivado:
	./program.sh --vivado $(BIT)

flash:
	./program.sh --flash $(MCS)
