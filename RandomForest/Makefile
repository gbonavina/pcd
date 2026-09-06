CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I src
LDFLAGS = -lm

# Sequential build (default). To try the commented OpenMP pragmas:
#   1. Uncomment the #pragma omp lines in src/rf.c
#   2. Build with: make CFLAGS="-O2 -std=c11 -Wall -Wextra -I src -fopenmp"

SRC = src/rf.c src/main.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean run test energy-test1 energy-test2 benchmarks

all: rf

rf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c src/rf.h
	$(CC) $(CFLAGS) -c -o $@ $<

run: rf
	./rf --data data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42

test:
	python python/benchmark_comparison.py --suite fast

energy-test1:
	python python/benchmark_comparison.py --suite test1

energy-test2:
	python python/benchmark_comparison.py --suite test2

benchmarks:
	python python/benchmark_comparison.py --suite all

clean:
	rm -f rf src/*.o
