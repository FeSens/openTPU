# Architecture tournament (docs/tourney.md)
#   make tourney COMP=otpu_coll N=1 K=1 [AGENT=claude|codex] [EVAL=yosys|vivado-remote]
#                [MODEL_HYP=..] [MODEL_IMPL=..] [MODEL_SCRIBE=..]   (default claude-opus-5-5)
#                [EFFORT_HYP=high] [EFFORT_IMPL=high,xhigh] [EFFORT_SCRIBE=low] [ARGS=--keep]
#   make tourney-report [COMP=otpu_coll]
#   (any target: EXEC=remote (default) runs lint and the test gates on omarchy, EXEC=local here)
#   make tourney-fmax N=1 K=2 [TARGET_MHZ=125.49] [FMAX_COMPS="otpu_xunit otpu_tmem ..."]
#        the whole-design fmax tournament: N passes over FMAX_COMPS, one round each, all on the
#        shared champion tourney/fmax, Vivado on the build host (EVAL=vivado-remote)
#   make tourney-fmax-baseline [TARGET_MHZ=125.49]   the champion's full build only
PYTHON ?= python3
COMP   ?=
N      ?= 1
K      ?= 1
AGENT  ?= claude
EVAL   ?= yosys
BASE   ?= main
ARGS   ?=
TARGET_MHZ ?= 125.49
# where lint and the test gates run: remote (omarchy, tools/omarchy_test.sh) or local
EXEC   ?= remote
export EXEC
# the units on the current full build's worst paths first (docs/tourney.md)
FMAX_COMPS ?= otpu_xunit otpu_tmem otpu_vpu otpu_mxu otpu_coll otpu_seq otpu_quant otpu_dma \
              otpu_actram otpu_native_dram otpu_fp
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
