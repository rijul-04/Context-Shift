PY ?= python3

.PHONY: cohorts test h1 episodes selectors figures all clean

cohorts:
	$(PY) -m src.csh.data_p2012
	$(PY) -m src.csh.data_eicu

test:
	$(PY) tests/run_tests.py

h1:
	$(PY) scripts/02_source_models.py

episodes:
	$(PY) scripts/03_run_episodes.py

selectors:
	$(PY) scripts/05_selectors.py

figures:
	$(PY) scripts/08_figures.py

all: cohorts test h1 episodes selectors figures

clean:
	rm -rf __pycache__ src/csh/__pycache__ tests/__pycache__
