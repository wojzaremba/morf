UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	MATLAB_LIB  := -I/opt/matlab/extern/include
	MEX 	    := mex -DUNIX $(MATLAB_LIB) 
else
	MATLAB_LIB  := -I/Applications/MATLAB_R2013a_Student.app/extern/include/
	MEX 	    := /Applications/MATLAB_R2013a_Student.app/bin/mex $(MATLAB_LIB)
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
