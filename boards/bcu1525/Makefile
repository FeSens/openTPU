# openTPU on the BCU1525 (VCU1525 build: xcvu9p-fsgd2104-2L-e, 4 x DDR4 DIMM, PCIe Gen3 x16;
# docs/bcu1525.md; README.md here). DIMM=udimm (the default: x8 UDIMMs without ECC) or rdimm
# (the card's original x4 ECC RDIMMs).
#   make lint       offline: Tcl syntax, XDC pins / ports vs the DDR4 cores' pads and the top,
#                   core.json's hashes (scripts/check_offline.py, both DIMMs); Verilator lint of
#                   the top for both DIMMs (-DSYNTHESIS, the DSP48E2 cells of rtl/tech/xilinxus)
#                   with stand-ins for the block design, the DDR4 core and the unisim primitives
#   make pins       regenerate constraints/*.xdc from AMD's VCU1525 board file (scripts/gen_pins.py)
#   make core       regenerate litedram_udimm/ (DIMM=rdimm: litedram/; tools/litedram/
#                   gen_core_ddr4.py; LITEX_VENV: a venv with core.json's packages; RVSHIM: a
#                   directory whose riscv64-unknown-elf-* work)
#   make ipcheck    the block design and the DDR4 core in Vivado on PART (any UltraScale+ part with a
#                   PCIE4 block: the free tier's xcku5p-ffvb676-2-e until a VU9P license exists)
#   make synthcheck the whole project synthesized on PART (scripts/synth_check.tcl: the top against
#                   the generated block design wrapper, the DSP48E2 mapping, Vivado's utilization)
#   make bit        Vivado batch build (S, MCOLS, CORE_MHZ=100, PCIE_LANES=16, FAST=1: run_vivado.sh;
#                   IMAGE=mem the memory-test image, docs/bcu1525.md section 14)
#   make impl       rerun implementation on the existing project
#   make wg         the production image (docs/bcu1525_wg.md): IMAGE=wg at WG_TARGET (default
#                   bcu1525-wg:bhw16), WG_MHZ (150), FAST=0, the MIGs' b15 address map, in
#                   WG_OUT; synthesis and implementation as two Vivado runs (wg-synth, wg-impl)
#   make wg-dev     the same target as a development image: 100 MHz, FAST=1 (functional)
#   make s2         two slices (docs/bcu1525_s2.md): bcu1525-s2:st16, floorplan c, one job (the
#                   same two runs: s2-synth, s2-impl; S2_OUT)
#   make x2         one card's image of the two-card machine, the 100 MHz functional build
#                   (x2/build_x2.sh X2_OUT; x2-prepare: its generated inputs only, no Vivado)
#   make program    JTAG load over the board's FT4232H (openFPGALoader -b vcu1525; BIT=path)
#   make flash      write the SPI flash (MT25QU01G) -- permanent. openFPGALoader needs its
#                   spiOverJtag bitstream for xcvu9p-fsgd2104, which it does not ship (only the
#                   VCU118's flga2104): build one with Vivado, or use Vivado's hardware manager
#                   (program_hw_cfgmem, part mt25qu01g-spi-x1_x2_x4, MCS=path)
ROOT := $(abspath ../..)
LITEX_VENV ?= $(HOME)/eda/litex-venv
RVSHIM ?= $(HOME)/eda/rvshim
PART ?= xcku5p-ffvb676-2-e
DIMM ?= udimm
LD_DIR := $(if $(filter rdimm,$(DIMM)),litedram,litedram_udimm)
IPCHECK_STEPS ?= bd,bdsynth,ld4
OUT ?= $(ROOT)/build/vivado_bcu1525
BIT ?= $(OUT)/otpu.bit
MCS ?= $(OUT)/otpu.mcs

# the RTL: opentpu.target sources("bcu1525") (tests/test_target.py checks it), the top linted last
RTL := $(addprefix $(ROOT)/rtl/, vpu/otpu_fp.sv vpu/otpu_fpipe.sv top/otpu_pkg.sv mem/otpu_tmem.sv \
  mem/otpu_actram.sv seq/otpu_seq.sv vpu/otpu_vtree.sv dma/otpu_dma.sv mxu/otpu_mxu.sv \
  tech/xilinxus/otpu_tech_dsp.sv tech/generic/otpu_tech_ram.sv vpu/otpu_quant.sv vpu/otpu_se_comp.sv vpu/otpu_se_tail.sv \
  vpu/otpu_vpu.sv top/otpu_coll.sv top/otpu_slice.sv shell/otpu_ctrl.sv shell/otpu_trace.sv \
  shell/otpu_board.sv mem/otpu_native_dram.sv mem/otpu_xstg.sv shell/otpu_afifo.sv \
  shell/otpu_mem_ch.sv \
  shell/otpu_axi_split2.sv boards/bcu1525/otpu_bcu_sys.sv boards/bcu1525/otpu_bmc_watch.sv \
  boards/bcu1525/otpu_bmc_uart.sv boards/bcu1525/otpu_bcu_gov.sv boards/bcu1525/otpu_axil_off.sv \
  boards/bcu1525/otpu_axi_off.sv)
# the memory-test image's (IMAGE=mem: vivado/create_project_mem.tcl)
MEM_RTL := $(addprefix $(ROOT)/rtl/, shell/otpu_ctrl.sv boards/bcu1525/otpu_bmc_watch.sv \
  boards/bcu1525/otpu_bmc_uart.sv \
  boards/bcu1525/otpu_bcu_gov.sv boards/bcu1525/otpu_axil_off.sv)
LINT := verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -DSYNTHESIS -DOTPU_FPGA_INIT sim/unisim_dsp48e2_stub.sv sim/unisim_stubs.sv

.PHONY: lint pins core ipcheck synthcheck bit impl program flash wg wg-synth wg-impl wg-dev s2 s2-synth s2-impl x2 x2-prepare
lint:
	python3 scripts/check_offline.py
	$(LINT) --top-module otpu_fpga_top_bcu $(RTL) sim/otpu_bd_wrapper_stub.sv \
	  sim/otpu_ld4_stub.sv $(ROOT)/rtl/boards/bcu1525/otpu_fpga_top_bcu.sv
	$(LINT) -DOTPU_BCU_RDIMM --top-module otpu_fpga_top_bcu $(RTL) sim/otpu_bd_wrapper_stub.sv \
	  sim/otpu_ld4_stub.sv $(ROOT)/rtl/boards/bcu1525/otpu_fpga_top_bcu.sv
	$(LINT) --top-module otpu_fpga_top_bcu_mem $(MEM_RTL) sim/otpu_bdm_wrapper_stub.sv \
	  sim/otpu_ld4_stub.sv $(ROOT)/rtl/boards/bcu1525/otpu_fpga_top_bcu_mem.sv

pins:
	python3 scripts/gen_pins.py

core:
	PATH=$(RVSHIM):$(LITEX_VENV)/bin:$$PATH $(LITEX_VENV)/bin/python $(ROOT)/tools/litedram/gen_core_ddr4.py \
	  --cpu --dimm $(DIMM) --sys-mhz 200 --out $(ROOT)/build/ld4_$(DIMM)
	cp $(ROOT)/build/ld4_$(DIMM)/otpu_ld4.v $(ROOT)/build/ld4_$(DIMM)/otpu_ld4.xdc \
	  $(ROOT)/build/ld4_$(DIMM)/otpu_ld4_*.init $(ROOT)/build/ld4_$(DIMM)/csr.csv \
	  $(ROOT)/build/ld4_$(DIMM)/sdram_init.py $(ROOT)/build/ld4_$(DIMM)/ports.txt $(LD_DIR)/
	@echo "update $(LD_DIR)/core.json's sha256s (make lint checks them)"

ipcheck:
	vivado -mode batch -nojournal -log $(OUT)_ipcheck.log -source vivado/ip_check.tcl \
	  -tclargs $(PART) $(OUT)_ipcheck $(IPCHECK_STEPS)

synthcheck:
	vivado -mode batch -nojournal -log $(OUT)_synthcheck.log -source scripts/synth_check.tcl \
	  -tclargs $(PART) $(OUT)_synthcheck

bit:
	./run_vivado.sh

impl:
	STEP=impl ./run_vivado.sh

# The WG images (AMD's DDR4 MIG at DDR4-2400; run_vivado.sh IMAGE=wg). The b15 map patches a
# generated file, so the IP cache is off (OTPU_IP_CACHE empty)
WG_TARGET ?= bcu1525-wg:bhw16
WG_MHZ ?= 150
WG_OUT ?= $(ROOT)/build/vivado_bcu1525_wg
S2_TARGET ?= bcu1525-s2:st16
S2_OUT ?= $(ROOT)/build/vivado_bcu1525_s2
X2_OUT ?= $(ROOT)/build/vivado_bcu1525_x2
WG_ENV = IMAGE=wg WG_TARGET=$(WG_TARGET) CORE_MHZ=$(WG_MHZ) FAST=0 WG_MIGMAP=b15 OTPU_IP_CACHE= \
  OUT_DIR=$(WG_OUT)
S2_ENV = IMAGE=wg WG_TARGET=$(S2_TARGET) WG_PLAN=c WG_FP=soft CORE_MHZ=$(WG_MHZ) FAST=0 \
  WG_MIGMAP=b15 OTPU_IP_CACHE= JOBS=1 SYNTH_THREADS=1 OUT_DIR=$(S2_OUT)

wg: wg-synth wg-impl

wg-synth:
	$(WG_ENV) STEP=synth ./run_vivado.sh

wg-impl:
	$(WG_ENV) STEP=impl ./run_vivado.sh

wg-dev:
	IMAGE=wg WG_TARGET=$(WG_TARGET) CORE_MHZ=100 FAST=1 OUT_DIR=$(WG_OUT)_dev ./run_vivado.sh

s2: s2-synth s2-impl

s2-synth:
	$(S2_ENV) STEP=synth ./run_vivado.sh

s2-impl:
	$(S2_ENV) STEP=impl ./run_vivado.sh

x2:
	bash x2/build_x2.sh $(X2_OUT)

x2-prepare:
	bash x2/build_x2.sh --prepare $(X2_OUT)

program:
	openFPGALoader -b vcu1525 $(BIT)

flash:
	openFPGALoader -b vcu1525 -f $(BIT)
