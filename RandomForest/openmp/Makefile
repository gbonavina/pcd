CC = gcc
CFLAGS = -O2 -std=c11 -Wall -Wextra -I src -fopenmp
LDFLAGS = -lm -fopenmp

SRC = src/rf.c src/main.c
OBJ = $(SRC:.c=.o)

.PHONY: all clean run

all: rf

rf: $(OBJ)
	$(CC) $(CFLAGS) -o $@ $(OBJ) $(LDFLAGS)

src/%.o: src/%.c src/rf.h
	$(CC) $(CFLAGS) -c -o $@ $<

# Data lives in ../data (same CSVs as the sequential binary).
run: rf
	./rf --data ../data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42

clean:
	rm -f rf src/*.o
