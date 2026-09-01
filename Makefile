PYTHON ?= python
GPU ?= 0
# Accept either the Make command-line variable (DATA_ROOT=...) or the
# environment variable documented for direct script execution.
DATA_ROOT ?= $(if $(RTDFL_DATA_ROOT),$(RTDFL_DATA_ROOT),$(CURDIR)/data/raw)
export PYTHONNOUSERSITE := 1

.PHONY: help smoke quick full full-dry-run plot validate table1 table2 fig8 fig9 clean-smoke

help:
	@printf '%s\n' \
	  'make smoke         Run core checks; validate evidence when available' \
	  'make quick         Run the 20-round CIFAR-10 five-method comparison' \
	  'make full          Run the complete simulation matrix (long)' \
	  'make full-dry-run  Print the complete simulation commands' \
	  'make plot          Process the bundled expected-result evidence' \
	  'make fig8          Recreate Figure 8 from physical-testbed traces' \
	  'make validate      Validate expected and generated results'

smoke:
	PYTHON="$(PYTHON)" bash scripts/smoke_test.sh

quick:
	PYTHON="$(PYTHON)" GPU="$(GPU)" RTDFL_DATA_ROOT="$(DATA_ROOT)" bash scripts/reproduce_quick.sh

full:
	PYTHON="$(PYTHON)" GPU="$(GPU)" RTDFL_DATA_ROOT="$(DATA_ROOT)" bash scripts/reproduce_all.sh

full-dry-run:
	PYTHON="$(PYTHON)" GPU="$(GPU)" RTDFL_DATA_ROOT="$(DATA_ROOT)" bash scripts/reproduce_all.sh --dry-run

plot: table1 table2 fig8 fig9

table1:
	$(PYTHON) scripts/reproduce_available_results.py --only table1

table2:
	$(PYTHON) scripts/reproduce_available_results.py --only table2

fig8:
	$(PYTHON) scripts/reproduce_fig8.py

fig9:
	$(PYTHON) scripts/reproduce_available_results.py --only fig9

validate:
	$(PYTHON) scripts/validate_results.py --expected-only
	$(PYTHON) scripts/reproduce_fig8.py --validate-only

clean-smoke:
	@find tmp/smoketest/latest -mindepth 1 -delete 2>/dev/null || true
