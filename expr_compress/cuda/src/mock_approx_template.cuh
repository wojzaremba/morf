#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"

#include "common_conv.cuh"
#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

using namespace std;

void mock_approx(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
 //       mexPrintf('Executing mock_approx');
	// XXX: return something.
}
