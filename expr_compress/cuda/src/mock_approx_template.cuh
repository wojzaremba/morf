#include <stdio.h>
#include <algorithm>
#include <vector>

#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

using namespace std;

void mock_approx(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 10) && (nlhs == 0));
        mexPrintf('Executing mock_approx');
	// XXX: return something.
}
