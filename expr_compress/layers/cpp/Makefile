MKLROOT := /opt/intel/composer_xe_2013_sp1.1.103/
################################################################################

%.o: %.cpp
	icpc -c  -I../../../external/eigen -I/Applications/MATLAB_R2013a_Student.app/extern/include -I/Applications/MATLAB_R2013a_Student.app/simulink/include -DMATLAB_MEX_FILE -fno-common -fexceptions -arch x86_64 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk -mmacosx-version-min=10.8  -DMX_COMPAT_32 -O3 -DNDEBUG  $<  -openmp -I$(MKLROOT)/mkl/include 

%.mexmaci64: %.o
	icpc -O -arch x86_64 -Wl,-syslibroot,/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.8.sdk -mmacosx-version-min=10.8 -bundle -Wl,-exported_symbols_list,/Applications/MATLAB_R2013a_Student.app/extern/lib/maci64/mexFunction.map -o "$@" $<  -L/Applications/MATLAB_R2013a_Student.app/bin/maci64 -lmx -lmex -lmat -lstdc++ -L$(MKLROOT)/mkl/lib $(MKLROOT)/mkl/lib/libmkl_intel_lp64.a $(MKLROOT)/mkl/lib/libmkl_core.a $(MKLROOT)/mkl/lib/libmkl_intel_thread.a -lpthread -lm -L$(MKLROOT)/compiler/lib/intel64 -parallel 

all: clean LRNormalCpp.mexmaci64 MaxPoolCpp.mexmaci64 ConvCpp.mexmaci64 MonoConvCpp.mexmaci64
	/opt/intel/mkl/bin/mklvars.sh intel64 lp64

clean:
	rm -rf *.mex*
	rm -rf *.o*
