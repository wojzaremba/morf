MKLROOT := /opt/intel/composer_xe_2013_sp1.1.103/
################################################################################

all: 
	/opt/intel/mkl/bin/mklvars.sh intel64 lp64
	icpc -c  -I../../../external/eigen -I/Applications/MATLAB_R2013a_Student.app/extern/include -I/Applications/MATLAB_R2013a_Student.app/simulink/include -DMATLAB_MEX_FILE -fno-common -fexceptions -arch x86_64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk -mmacosx-version-min=10.8  -DMX_COMPAT_32 -O3 -DNDEBUG  "ConvCpp.cpp"  -openmp -mkl=parallel  -I$(MKLROOT)/mkl/include 
	icpc -O -arch x86_64 -Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk -mmacosx-version-min=10.8 -bundle -Wl,-exported_symbols_list,/Applications/MATLAB_R2013a_Student.app/extern/lib/maci64/mexFunction.map -o  "ConvCpp.mexmaci64"  ConvCpp.o  -L/Applications/MATLAB_R2013a_Student.app/bin/maci64 -lmx -lmex -lmat -lstdc++ -L$(MKLROOT)/mkl/lib -lmkl_intel_lp64 -lmkl_core -lmkl_intel_thread -lpthread -parallel -lm -L$(MKLROOT)/compiler/lib/intel64

clean:
	rm -rf *.mex*
	rm -rf *.o*
