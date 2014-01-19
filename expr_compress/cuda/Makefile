include ./findcudalib.mk

UNAME := $(shell uname)
CONV_LIBS   := ../../cuda/src/
ifeq ($(UNAME), Linux)
	CUDA_PATH   := /usr/local/pkg/cuda/5.5/cuda
	MATLAB_LIB  := -I/opt/matlab/extern/include
	NVCC        := $(CUDA_PATH)/bin/nvcc $(MATLAB_LIB) -I$(CONV_LIBS) -L$(CUDA_PATH)/lib64 -lcudart -lcublas -arch=sm_30 --compiler-options -fPIC --output-file
	MEX 	    := mex -DUNIX $(MATLAB_LIB) -I$(CUDA_PATH)/include -L$(CUDA_PATH)/lib64 -lcudart -lcublas
else
	CUDA_PATH   := /Developer/NVIDIA/CUDA-5.5
	MATLAB_LIB  := -I/Applications/MATLAB_R2013a_Student.app/extern/include/
	NVCC        := $(CUDA_PATH)/bin/nvcc -ccbin /usr/bin/clang $(MATLAB_LIB) -I$(CONV_LIBS) -L$(CUDA_PATH)/lib  -arch=sm_30 --compiler-options -fPIC -Xcompiler  -stdlib=libstdc++ --output-file
	MEX 	    := /Applications/MATLAB_R2013a_Student.app/bin/mex $(MATLAB_LIB) -L$(CUDA_PATH)/lib -lcudart -lcublas
endif


################################################################################

all: clean mexapprox

nvccapprox:
	$(NVCC) Capprox_gen.cpp --cuda src/Capprox_gen.cu

mexapprox: nvccapprox
	$(MEX) Capprox_gen.cpp

clean:
	rm -rf *.mex*
	rm -rf *.o*
	rm -rf *.cpp*
	rm -rf src/*_gen.*
