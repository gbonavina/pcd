PYTHON ?= python

.PHONY: all experimentos tune artigo clean

# Binario OpenMP usado em todos os experimentos do artigo.
all:
	$(MAKE) -C openmp

# Varredura 1..32 threads com energia RAPL: gera openmp_scaling.csv.
experimentos: all
	$(PYTHON) run_openmp_experiments.py

# Busca em grade dos hiperparametros (Iris, WDBC e Letter).
tune: all
	$(PYTHON) python/tune_params.py --threads 16 --export-csv python/sweep.csv

artigo:
	cd report/template && latexmk -pdf artigo.tex

clean:
	$(MAKE) -C openmp clean
	cd report/template && latexmk -c artigo.tex
