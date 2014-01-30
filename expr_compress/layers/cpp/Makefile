UNAME := $(shell uname)
OPT := -v CC=gcc LD=gcc COPTIMFLAGS='-O3 -DNDEBUG' -largeArrayDims
ifeq ($(UNAME), Linux)
	MATLAB_LIB  := -I/opt/matlab/extern/include
	MEX 	    := mex -DUNIX $(MATLAB_LIB) $(OPT)
else
	MATLAB_LIB  := -I/Applications/MATLAB_R2013a_Student.app/extern/include/
	MEX 	    := /Applications/MATLAB_R2013a_Student.app/bin/mex $(MATLAB_LIB) $(OPT)
endif


################################################################################

all: clean
	$(MEX) ConvCpp.cpp
	$(MEX) MonoConvCpp.cpp

debug: clean
	$(MEX) ConvCpp.cpp -DDEBUG=1
	$(MEX) MonoConvCpp.cpp -DDEBUG=1

clean:
	rm -rf *.mex*
	rm -rf *.o*
