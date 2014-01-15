#include "mex.h"
#include "#mock_gen.cuh"

const int fsize = 26;
static void (*func[fsize]) (int, mxArray **, int, const mxArray **) = 
	{CopyToGPU, CopyFromGPU, AddVector, Scale, 
	 ActRELU, dActRELU, ActLINEAR, dActLINEAR,
	 NULL, Reshape, NULL, PrintShape, ActEXP,
	 Sum, Max, EltwiseDivideByVector, Mult, NULL, CleanGPU,
	 StartTimer, StopTimer, EltwiseMult, Transpose, Add, Subtract, #mock};

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs >= 1);
	mexAtExit(cleanUp);
	int fid = (int)mxGetScalar(prhs[0]);
	assert_(fid < fsize);
	(*func[fid])(nlhs, plhs, nrhs, prhs);
}

