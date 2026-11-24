CC = mpicc
CPPFLAGS += -I../common
CFLAGS = -O2 -std=c11 -Wall -Wextra
LDLIBS += -lm
OBJ = src/main.o rf.o
.PHONY: all clean
all: rf
rf: $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $(OBJ) $(LDLIBS)
src/main.o: src/main.c ../common/rf.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
rf.o: ../common/rf.c ../common/rf.h
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@
clean:
	rm -f rf $(OBJ)
