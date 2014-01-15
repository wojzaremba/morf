#include "mex.h"
#include "cpu_headers.h"

using namespace std;

const int fsize = 25;
static void (*func[fsize]) (int, mxArray **, int, const mxArray **) = 
	{CopyToGPU, CopyFromGPU, AddVector, Scale, 
	 ActRELU, dActRELU, ActLINEAR, dActLINEAR,
	 ConvAct, Reshape, MaxPooling, PrintShape, ActEXP,
	 Sum, Max, EltwiseDivideByVector, Mult, ConvResponseNormCrossMap, CleanGPU,
	 StartTimer, StopTimer, EltwiseMult, Transpose, Add, Subtract};

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs >= 1);
	mexAtExit(cleanUp);
	int fid = (int)mxGetScalar(prhs[0]);
	assert_(fid < fsize);
	(*func[fid])(nlhs, plhs, nrhs, prhs);
}

