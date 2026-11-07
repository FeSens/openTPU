# openTPU on the BCU1525 (VCU1525 build: xcvu9p-fsgd2104-2L-e, 4 x DDR4 RDIMM, PCIe Gen3 x16;
# docs/bcu1525.md).
#   make lint       offline: Tcl syntax, XDC pins / ports vs the DDR4 core's pads and the top,
#                   core.json's hashes (scripts/check_offline.py); Verilator lint of the top
#                   (-DSYNTHESIS, the DSP48E2 cells of rtl/tech/xilinxus) with stand-ins for the
#                   block design, the DDR4 core and the unisim primitives (sim/)
#   make pins       regenerate constraints/*.xdc from AMD's VCU1525 board file (scripts/gen_pins.py)
#   make core       regenerate litedram/ (tools/litedram/gen_core_ddr4.py; LITEX_VENV: a venv with
#                   core.json's packages; RVSHIM: a directory whose riscv64-unknown-elf-* work)
#   make ipcheck    the block design and the DDR4 core in Vivado on PART (any UltraScale+ part with a
#                   PCIE4 block: the free tier's xcku5p-ffvb676-2-e until a VU9P license exists)
#   make synthcheck the whole project synthesized on PART (scripts/synth_check.tcl: the top against
#                   the generated block design wrapper, the DSP48E2 mapping, Vivado's utilization)
#   make bit        Vivado batch build (S, MCOLS, CORE_MHZ=100, PCIE_LANES=16, FAST=1: run_vivado.sh)
#   make impl       rerun implementation on the existing project
#   make program    JTAG load over the board's FT4232H (openFPGALoader -b vcu1525; BIT=path)
#   make flash      write the SPI flash (MT25QU01G) -- permanent. openFPGALoader needs its
#                   spiOverJtag bitstream for xcvu9p-fsgd2104, which it does not ship (only the
#                   VCU118's flga2104): build one with Vivado, or use Vivado's hardware manager
#                   (program_hw_cfgmem, part mt25qu01g-spi-x1_x2_x4, MCS=path)
ROOT := $(abspath ../..)
LITEX_VENV ?= $(HOME)/eda/litex-venv
RVSHIM ?= $(HOME)/eda/rvshim
PART ?= xcku5p-ffvb676-2-e
IPCHECK_STEPS ?= bd,bdsynth,ld4
OUT ?= $(ROOT)/build/vivado_bcu1525
BIT ?= $(OUT)/otpu.bit
MCS ?= $(OUT)/otpu.mcs

# the RTL: opentpu.target sources("bcu1525") (tests/test_target.py checks it), the top linted last
RTL := $(addprefix $(ROOT)/rtl/, vpu/otpu_fp.sv vpu/otpu_fpipe.sv top/otpu_pkg.sv mem/otpu_tmem.sv \
  mem/otpu_actram.sv seq/otpu_seq.sv vpu/otpu_vtree.sv dma/otpu_dma.sv mxu/otpu_mxu.sv \
  tech/xilinxus/otpu_tech_dsp.sv tech/generic/otpu_tech_ram.sv vpu/otpu_quant.sv vpu/otpu_se_comp.sv vpu/otpu_se_tail.sv \
  vpu/otpu_vpu.sv top/otpu_coll.sv top/otpu_slice.sv shell/otpu_ctrl.sv shell/otpu_trace.sv \
  shell/otpu_board.sv mem/otpu_native_dram.sv mem/otpu_native_dram2.sv mem/otpu_native_xdram2.sv shell/otpu_afifo.sv shell/otpu_mem_ch.sv \
  shell/otpu_axi_split2.sv boards/bcu1525/otpu_bcu_sys.sv)
LINT := verilator --lint-only -Wno-fatal -Wno-WIDTHEXPAND -Wno-WIDTHTRUNC -Wno-UNUSEDSIGNAL \
  -Wno-UNUSEDPARAM -DSYNTHESIS sim/unisim_dsp48e2_stub.sv sim/unisim_stubs.sv

.PHONY: lint pins core ipcheck synthcheck bit impl program flash
lint:
	python3 scripts/check_offline.py
	$(LINT) --top-module otpu_fpga_top_bcu $(RTL) sim/otpu_bd_wrapper_stub.sv \
	  sim/otpu_ld4_stub.sv $(ROOT)/rtl/boards/bcu1525/otpu_fpga_top_bcu.sv

pins:
	python3 scripts/gen_pins.py

core:
	PATH=$(RVSHIM):$(LITEX_VENV)/bin:$$PATH $(LITEX_VENV)/bin/python $(ROOT)/tools/litedram/gen_core_ddr4.py \
	  --cpu --sys-mhz 200 --out $(ROOT)/build/ld4
	cp $(ROOT)/build/ld4/otpu_ld4.v $(ROOT)/build/ld4/otpu_ld4.xdc $(ROOT)/build/ld4/otpu_ld4_*.init \
	  $(ROOT)/build/ld4/csr.csv $(ROOT)/build/ld4/sdram_init.py $(ROOT)/build/ld4/ports.txt litedram/
	@echo "update litedram/core.json's sha256s (make lint checks them)"

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

program:
	openFPGALoader -b vcu1525 $(BIT)

flash:
	openFPGALoader -b vcu1525 -f $(BIT)
