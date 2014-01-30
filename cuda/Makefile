include ./findcudalib.mk

UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	CUDA_PATH   := /usr/local/pkg/cuda/5.5/cuda
	MATLAB_LIB  := -I/opt/matlab/extern/include
	NVCC        := $(CUDA_PATH)/bin/nvcc $(MATLAB_LIB) -L$(CUDA_PATH)/lib64 -lcudart -lcublas --compiler-options -fPIC --output-file
	MEX 	    := mex -DUNIX $(MATLAB_LIB) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -lcublas
else
	CUDA_PATH   := /Developer/NVIDIA/CUDA-5.5
	MATLAB_LIB  := -I/Applications/MATLAB_R2013a_Student.app/extern/include/
	NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin /usr/bin/clang $(MATLAB_LIB) -L$(CUDA_PATH)/lib  --compiler-options -fPIC -Xcompiler -stdlib=libstdc++ --output-file
	MEX 	    := /Applications/MATLAB_R2013a_Student.app/bin/mex $(MATLAB_LIB) -L$(CUDA_PATH)/lib -lcudart -lcublas
endif


################################################################################

all: clean mex

nvcc:
	$(NVCC) C_.cpp --cuda src/C_.cu

mex: nvcc
	$(MEX) C_.cpp

clean:
	rm -rf *.o*
	rm -rf *.cpp*
