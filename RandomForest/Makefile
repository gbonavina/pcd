CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I src
LDFLAGS = -lm

# Sequential build (default). To try the commented OpenMP pragmas:
#   1. Uncomment the #pragma omp lines in src/rf.c
#   2. Build with: make CFLAGS="-O2 -std=c11 -Wall -Wextra -I src -fopenmp"

SRC = src/rf.c src/main.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean run test tune energy-test1 energy-test2 benchmarks

all: rf

rf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c src/rf.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: rf
	./rf --data data/iris.csv --trees 300 --max-depth 12 --min-samples 5 --mtry 2 --seed 42

# Busca em grade dos hiperparametros usando o binario OpenMP como referencia.
tune:
	python python/tune_params.py --threads 16 --export-csv python/sweep.csv

test:
	python python/benchmark_comparison.py --suite fast --repeats 3 --warmup --measure-idle

energy-test1:
	python python/benchmark_comparison.py --suite test1 --repeats 3 --warmup --measure-idle

# Cenario de alta demanda (covtype): ~13 min por rodada completa.
energy-test2:
	python python/benchmark_comparison.py --suite test2 --measure-idle

benchmarks:
	python python/benchmark_comparison.py --suite all --repeats 3 --warmup --measure-idle --export-csv benchmark_results.csv

clean:
	rm -f rf src/*.o
