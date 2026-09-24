# openTPU on the Inspur YPCB-00338.
#   make lint       offline: MIG pin check, Tcl syntax, XDC ports vs the top, Verilator lint
#   make mig        regenerate the MIG .prj files and DDR3 pin XDC (DDR=800|1066)
#   make bit        Vivado batch build (MCOLS=2|4, CORE_MHZ=100|80|75) -> build/vivado/otpu.bit (VIVADO_DOCKER=image for Docker)
#   make program    JTAG load (openFPGALoader); make program-vivado with Vivado's hw_server
#   make flash      write the BPI flash (permanent)
ROOT := $(abspath ../..)
DDR ?= 800
# CORE_MHZ: accelerator clock (80 or 75 if 100 does not close timing)
# MCOLS: MXU columns, 2 or 4 (run the host with OTPU_MCOLS set to the same value)
# BUILD_ID: the BUILD_ID register (8 hex digits; default: the git commit)
CORE_MHZ ?= 100
MCOLS ?= 2
RTL := $(addprefix $(ROOT)/rtl/, vpu/otpu_fp.sv vpu/otpu_fpipe.sv top/otpu_pkg.sv \
  mem/otpu_tmem.sv mem/otpu_axi_dram.sv mem/otpu_actram.sv seq/otpu_seq.sv dma/otpu_dma.sv \
  mxu/otpu_mxu.sv vpu/otpu_quant.sv vpu/otpu_vpu.sv top/otpu_coll.sv top/otpu_slice.sv \
  boards/ypcb-00338/otpu_ctrl.sv boards/ypcb-00338/otpu_trace.sv boards/ypcb-00338/otpu_board.sv)

.PHONY: lint mig bit program program-vivado flash
mig:
	python3 scripts/gen_mig_prj.py --speed $(DDR)

lint: mig
	python3 scripts/check_offline.py
	verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
	  -Wno-UNUSEDPARAM -DSYNTHESIS --top-module otpu_fpga_top -GMCOLS=$(MCOLS) $(RTL) \
	  sim/otpu_bd_wrapper_stub.sv $(ROOT)/rtl/boards/ypcb-00338/otpu_fpga_top.sv

bit:
	MCOLS=$(MCOLS) CORE_MHZ=$(CORE_MHZ) $(if $(BUILD_ID),BUILD_ID=$(BUILD_ID)) ./run_vivado.sh $(DDR)

program:
	./program.sh

program-vivado:
	./program.sh --vivado

flash:
	./program.sh --flash
