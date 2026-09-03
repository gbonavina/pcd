NVCC ?= nvcc
CCBIN ?= "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64"
NVCCFLAGS = -O3 -std=c++14 -I src

SRC = src/rf_host.c src/rf_gpu.cu src/main.cu
HDR = src/rf_cuda.h

.PHONY: all clean run

all: rf_cuda.exe

rf_cuda.exe: $(SRC) $(HDR)
	$(NVCC) $(NVCCFLAGS) -ccbin $(CCBIN) -o $@ $(SRC)

run: rf_cuda.exe
	./rf_cuda.exe --data ../data/iris.csv --trees 100 --max-depth 8 --min-samples 2 --seed 42

clean:
	rm -f rf_cuda.exe rf_cuda.exp rf_cuda.lib rf.exe rf.exp rf.lib
