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
#   make tourney-units K=1 [UNIT_COMPS="otpu_tmem ..."]   the unit tournament, without end: each
#        unit alone out of context in Vivado, its own champion tourney/unit/<comp>, no full builds
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

# the unit tournament's components (the units with an out-of-context part) and build hosts
UNIT_COMPS ?= otpu_tmem otpu_quant otpu_seq otpu_actram otpu_coll otpu_vpu otpu_dma otpu_fp
UNIT_HOSTS ?= opentpu

.PHONY: tourney tourney-baseline tourney-report test-tourney tourney-fmax tourney-fmax-baseline \
        tourney-forever tourney-units

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

# the unit tournament beside it (docs/tourney.md, "The unit tournament"): forever.sh with
# OBJECTIVE=unit and its own control files, /tmp/otpu-tourney-units-{stop,pause,comps,hosts}
tourney-units:
	OBJECTIVE=unit WHOLE_EVERY=0 K=$(K) TARGET_MHZ=$(TARGET_MHZ) FOREVER_COMPS="$(UNIT_COMPS)" \
	  FOREVER_COMPS_FILE=/tmp/otpu-tourney-units-comps FOREVER_STOP=/tmp/otpu-tourney-units-stop \
	  FOREVER_PAUSE=/tmp/otpu-tourney-units-pause OTPU_HOSTS_FILE=/tmp/otpu-tourney-units-hosts \
	  OTPU_BUILD_HOSTS=$(UNIT_HOSTS) bash tools/tourney/forever.sh

tourney-fmax-baseline:
	$(PYTHON) -m tools.tourney.orchestrator --objective fmax --target-mhz $(TARGET_MHZ) \
	  --comp otpu_xunit --eval vivado-remote --base $(BASE) --baseline-only $(ARGS)

tourney-report:
	$(PYTHON) -m tools.tourney.report $(if $(COMP),--comp $(COMP),)

test-tourney:
	$(PYTHON) -m pytest -q tests/test_tourney.py
