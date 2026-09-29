# Architecture tournament (docs/tourney.md)
#   make tourney COMP=otpu_coll N=1 K=1 [AGENT=claude|codex] [EVAL=yosys|vivado-remote]
#                [MODEL_HYP=..] [MODEL_IMPL=..] [MODEL_SCRIBE=..]   (default claude-opus-5-5)
#                [EFFORT_HYP=high] [EFFORT_IMPL=high,xhigh] [EFFORT_SCRIBE=low] [ARGS=--keep]
#   make tourney-report [COMP=otpu_coll]
#   (any target: EXEC=remote (default) runs lint and the test gates on omarchy, EXEC=local here)
#   make tourney-fmax N=1 K=2 [TARGET_MHZ=133.33] [FMAX_COMPS="otpu_native_dram otpu_tmem ..."]
#        the whole-design fmax tournament: N passes over FMAX_COMPS, one round each, all on the
#        shared champion tourney/fmax, Vivado on the build host (EVAL=vivado-remote)
#   make tourney-fmax-baseline [TARGET_MHZ=133.33]   the champion's full build only
PYTHON ?= python3
COMP   ?=
N      ?= 1
K      ?= 1
AGENT  ?= claude
EVAL   ?= yosys
BASE   ?= main
ARGS   ?=
# 133.33 MHz (800 / 6, an MMCM step): 128 B x f = 17.07 GB/s, both DDR3-1066 channels' peak
TARGET_MHZ ?= 133.33
# where lint and the test gates run: remote (omarchy, tools/omarchy_test.sh) or local
EXEC   ?= remote
export EXEC
# the units on the measured worst path families first (docs/tourney.md)
FMAX_COMPS ?= otpu_native_dram otpu_tmem otpu_mxu otpu_seq otpu_dma otpu_xunit otpu_vpu \
              otpu_quant otpu_coll otpu_actram otpu_fp
export MODEL_HYP MODEL_IMPL MODEL_SCRIBE EFFORT_HYP EFFORT_IMPL EFFORT_SCRIBE

.PHONY: tourney tourney-baseline tourney-report test-tourney tourney-fmax tourney-fmax-baseline \
        tourney-forever

tourney:
	@test -n "$(COMP)" || (echo "COMP=<component> required; see tools/tourney/components/" && false)
	$(PYTHON) -m tools.tourney.orchestrator --comp $(COMP) --rounds $(N) --slots $(K) \
	  --agent $(AGENT) --eval $(EVAL) --base $(BASE) $(ARGS)

tourney-baseline:
	@test -n "$(COMP)" || (echo "COMP=<component> required" && false)
	$(PYTHON) -m tools.tourney.orchestrator --comp $(COMP) --eval $(EVAL) --base $(BASE) \
	  --baseline-only $(ARGS)

tourney-fmax:
	@for n in $$(seq 1 $(N)); do for c in $(FMAX_COMPS); do \
	  $(PYTHON) -m tools.tourney.orchestrator --objective fmax --target-mhz $(TARGET_MHZ) \
	    --comp $$c --rounds 1 --slots $(K) --agent $(AGENT) --eval vivado-remote --base $(BASE) \
	    $(ARGS) || exit $$?; done; done

# the fmax tournament without end: every component in turn, the whole-design component every
# WHOLE_EVERY rounds (tools/tourney/forever.sh; stop: touch /tmp/otpu-tourney-stop)
tourney-forever:
	K=$(K) TARGET_MHZ=$(TARGET_MHZ) bash tools/tourney/forever.sh

tourney-fmax-baseline:
	$(PYTHON) -m tools.tourney.orchestrator --objective fmax --target-mhz $(TARGET_MHZ) \
	  --comp otpu_xunit --eval vivado-remote --base $(BASE) --baseline-only $(ARGS)

tourney-report:
	$(PYTHON) -m tools.tourney.report $(if $(COMP),--comp $(COMP),)

test-tourney:
	$(PYTHON) -m pytest -q tests/test_tourney.py
