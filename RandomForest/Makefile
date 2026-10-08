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

# Figuras pgfplots (speedup e EDP) e depois o artigo.
artigo:
	cd report/template/figuras && lualatex -interaction=nonstopmode speedup.tex && lualatex -interaction=nonstopmode edp.tex
	cd report/template && latexmk -lualatex artigo.tex

clean:
	$(MAKE) -C openmp clean
	cd report/template && latexmk -c artigo.tex
