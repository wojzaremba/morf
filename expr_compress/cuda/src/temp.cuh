#include <stdio.h>
#include <mex.h>

__device__ void temp_cuda(int x) {
	printf('x = %d\n', x);
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	int x = 1;
	temp_cuda<<<1, 1>>>(x);
}
                       


