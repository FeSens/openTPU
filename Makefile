# Architecture tournament (docs/tourney.md)
#   make tourney COMP=otpu_coll N=1 K=1 [AGENT=claude|codex] [EVAL=yosys|vivado]
#                [MODEL_HYP=..] [MODEL_IMPL=..] [MODEL_SCRIBE=..]   (default claude-opus-5-5)
#                [EFFORT_HYP=high] [EFFORT_IMPL=high,xhigh] [EFFORT_SCRIBE=low] [ARGS=--keep]
#   make tourney-report [COMP=otpu_coll]
PYTHON ?= python3
COMP   ?=
N      ?= 1
K      ?= 1
AGENT  ?= claude
EVAL   ?= yosys
BASE   ?= main
ARGS   ?=
export MODEL_HYP MODEL_IMPL MODEL_SCRIBE EFFORT_HYP EFFORT_IMPL EFFORT_SCRIBE

.PHONY: tourney tourney-baseline tourney-report test-tourney

tourney:
	@test -n "$(COMP)" || (echo "COMP=<component> required; see tools/tourney/components/" && false)
	$(PYTHON) -m tools.tourney.orchestrator --comp $(COMP) --rounds $(N) --slots $(K) \
	  --agent $(AGENT) --eval $(EVAL) --base $(BASE) $(ARGS)

tourney-baseline:
	@test -n "$(COMP)" || (echo "COMP=<component> required" && false)
	$(PYTHON) -m tools.tourney.orchestrator --comp $(COMP) --eval $(EVAL) --base $(BASE) \
	  --baseline-only $(ARGS)

tourney-report:
	$(PYTHON) -m tools.tourney.report $(if $(COMP),--comp $(COMP),)

test-tourney:
	$(PYTHON) -m pytest -q tests/test_tourney.py
