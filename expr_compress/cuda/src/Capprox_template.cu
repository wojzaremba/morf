#include "mex.h"
#include "#mock_gen.cuh"
#include "C_.cu"

static void (*func_approx[fsize]) (int, mxArray **, int, const mxArray **) = 
	{NULL, NULL, NULL, NULL, NULL, 
	 NULL, NULL, NULL, NULL, NULL, 
	 NULL, NULL, NULL, NULL, NULL, 
	 NULL, NULL, NULL, NULL, NULL, 
	 NULL, #mock};

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs >= 1);
	mexAtExit(cleanUp);
	int fid = (int)mxGetScalar(prhs[0]);
	assert_(fid < fsize);
	if (fid == 26) {
		(*func_approx[fid])(nlhs, plhs, nrhs, prhs);
	} else {
		(*func[fid])(nlhs, plhs, nrhs, prhs);
	}
}

