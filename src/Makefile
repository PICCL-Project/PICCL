# Simple build for piccl.so
# ------------------------------------------------------------
# Override CUDA_HOME / NCCL_HOME or CC as needed on your system

CUDA_HOME ?= /usr/local/cuda
NCCL_HOME ?= /usr
CC        := nvcc            # use nvcc so <cuda_runtime_api.h> works; gcc also ok if includes are present
CFLAGS    ?= -Xcompiler "-fPIC -O2 -Wall" -I$(CUDA_HOME)/include -I$(NCCL_HOME)/include -Iinclude
LDFLAGS   ?= -shared -ldl -L$(CUDA_HOME)/lib64 -lcudart -L$(NCCL_HOME)/lib -lnccl -lnvToolsExt
TARGET    ?= piccl.so
SRC       := piccl.c $(wildcard coll/*.c) $(wildcard kernels/*.cu)  # Add additional .c and .cu files here
INCLUDES  := include/*.h      # Optional: for clarity, list .h files

all : $(TARGET)

$(TARGET) : $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all cleansource