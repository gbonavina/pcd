CC = gcc
CPPFLAGS += -I../common
CFLAGS = -O2 -std=c11 -Wall -Wextra -fopenmp
LDLIBS += -lm -fopenmp

OBJ = src/main.o rf.o

.PHONY: all run clean
all: rf

rf: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)

src/main.o: src/main.c ../common/rf.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

rf.o: ../common/rf.c ../common/rf.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

run: rf
	./rf --data ../data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42

clean:
	rm -f rf $(OBJ) src/rf.o