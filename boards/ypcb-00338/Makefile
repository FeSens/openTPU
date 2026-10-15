# openTPU on the Inspur YPCB-00338.
#   make lint       offline: Tcl syntax, XDC ports vs the top, Verilator lint of the build
#                   (otpu_fpga_top_ld; lint-ld is the same)
#   make bit        Vivado batch build (TARGET=ypcb-00338:m4|m8, MCOLS=4|8, VPU_CL=2|4, LANES=8|16, CORE_MHZ=133.33|100, FAST=1) -> build/vivado/otpu.bit (VIVADO_DOCKER=image for Docker)
#                   the LiteDRAM core (DDR3-1066, host calibration), the LitePCIe core (Gen2 x8), the target's variant (its MCOLS, MXU_E, TMEM_WORDS), the systolic MXU
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
# TARGET: the target and its variant (targets/ypcb-00338.toml [variants]: m8, the default, 8 columns at E 8
#   and 128K TMEM words; m4, 4 columns at E 4 and 64K); the knobs below override it
# MCOLS: MXU columns, 4 or 8 (default the target's; the host reads it from the bitstream's VERSION register)
# VPU_CL: VPU lanes with exp2/recip/rsqrt, 2 or 4 (timing only: the programs do not change)
# LANES: VPU lanes / TMEM banks, 8 or 16 (timing only; the host reads it from VERSION)
# ACT_ROWS: ACT RAM rows, the most rows of one MM (default the target's: 16, the MX build's; the
#   MXU replays each streamed weight chunk per MCOLS rows; MCOLS: no replay)
# MXU_E: the MXU's epilogue lanes (timing only; default the target's: MCOLS, full rate)
# TMEM_IMPL: the TMEM (timing only; default the target's: 3), 0 a copy per read port (production
#   8b33206e), 1 banked (three copies, docs/tmem_pump.md; TMEM_RQ=1 with it: the banks' addresses
#   from the requests by priority), 3 one copy (docs/tmem_nocopy.md; TMEM_X its knobs over the
#   target's pick, vivado/tmem_knobs.tcl; pumped on core_clk2x: CORE_MHZ's divider even)
# TMEM_WORDS: the TMEM's words (default the target variant's: m4 65536, m8 131072)
# BUILD_ID: the BUILD_ID register (8 hex digits; default: the git commit)
# PCIE_GEN: 2 (Gen2 x8, LitePCIe's user clock 250 MHz), the only setting
ifeq ($(FAST),1)
CORE_MHZ ?= 100
else
CORE_MHZ ?= 133.33
endif
MCOLS ?=
VPU_CL ?=
LANES ?=
ACT_ROWS ?=
MXU_E ?=
RTL := $(addprefix $(ROOT)/rtl/, vpu/otpu_fp.sv vpu/otpu_fpipe.sv top/otpu_pkg.sv \
  mem/otpu_tmem.sv mem/otpu_actram.sv seq/otpu_seq.sv vpu/otpu_vtree.sv \
  dma/otpu_dma.sv mxu/otpu_mxu.sv tech/xilinx7/otpu_tech_dsp.sv tech/generic/otpu_tech_ram.sv vpu/otpu_quant.sv vpu/otpu_se_comp.sv vpu/otpu_se_tail.sv \
  vpu/otpu_vpu.sv \
  top/otpu_coll.sv top/otpu_slice.sv \
  shell/otpu_ctrl.sv shell/otpu_trace.sv shell/otpu_board.sv \
  mem/otpu_native_dram.sv shell/otpu_afifo.sv shell/otpu_mem_ch.sv \
  shell/otpu_axi_split2.sv shell/otpu_pcie_mm.sv boards/ypcb-00338/otpu_native_sys.sv)
# MXU_IMPL: the MXU create_project.tcl builds (MXU systolic, the default: 2); the systolic cells'
# DSP48E1 comes from a lint stub
MXU_IMPL ?= 2
LINT := verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -DSYNTHESIS $(if $(MCOLS),-GMCOLS=$(MCOLS)) $(if $(ACT_ROWS),-GACT_ROWS=$(ACT_ROWS)) $(if $(MXU_E),-GMXU_E=$(MXU_E)) \
  $(if $(VPU_CL),-GVPU_CL=$(VPU_CL)) $(if $(LANES),-GLANES=$(LANES)) $(if $(TMEM_WORDS),-GTMEM_WORDS=$(TMEM_WORDS)) \
  -GMXU_IMPL=$(MXU_IMPL) sim/unisim_dsp48e1_stub.sv

.PHONY: lint lint-ld bit impl program program-vivado flash
lint:
	python3 scripts/check_offline.py
	$(LINT) --top-module otpu_fpga_top_ld $(RTL) \
	  sim/otpu_bd_ld_wrapper_stub.sv sim/otpu_litedram_stub.sv sim/otpu_litepcie_stub.sv \
	  $(ROOT)/rtl/boards/ypcb-00338/otpu_fpga_top_ld.sv

lint-ld: lint

bit:
	$(if $(TARGET),TARGET=$(TARGET)) $(if $(MCOLS),MCOLS=$(MCOLS)) $(if $(ACT_ROWS),ACT_ROWS=$(ACT_ROWS)) $(if $(MXU_E),MXU_E=$(MXU_E)) $(if $(VPU_CL),VPU_CL=$(VPU_CL)) $(if $(LANES),LANES=$(LANES)) CORE_MHZ=$(CORE_MHZ) $(if $(TMEM_IMPL),TMEM_IMPL=$(TMEM_IMPL)) $(if $(TMEM_RQ),TMEM_RQ=$(TMEM_RQ)) $(if $(TMEM_X),TMEM_X=$(TMEM_X)) $(if $(TMEM_WORDS),TMEM_WORDS=$(TMEM_WORDS)) $(if $(BUILD_ID),BUILD_ID=$(BUILD_ID)) ./run_vivado.sh $(DDR)

impl:
	STEP=impl ./run_vivado.sh $(DDR)

program:
	./program.sh $(BIT)

program-vivado:
	./program.sh --vivado $(BIT)

flash:
	./program.sh --flash $(MCS)
