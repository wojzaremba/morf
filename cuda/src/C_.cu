#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"

#include "common_conv.cuh"
#include "conv_util.cuh"
#include "cudaconv2.cuh"
#include "filter_acts.cuh"
#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "weight_acts.cuh"


void MaxPooling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 7) && (nlhs == 0));
	NVMatrix* images = getMatrix(prhs[1]);
	NVMatrix* targets = getMatrix(prhs[2]); 
	int channels = (int)mxGetScalar(prhs[3]);
	int sizeX = (int)mxGetScalar(prhs[4]);
	int start = 0;
	int stride = (int)mxGetScalar(prhs[5]);
	int outputsX = (int)mxGetScalar(prhs[6]); 
	images->transpose();
	targets->transpose();
	convLocalPool(*images, *targets, channels, sizeX, start, stride, outputsX, MaxPooler());
	images->transpose();
	targets->transpose();
}

void ConvAct(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 9) && (nlhs == 0));
	NVMatrix* images = getMatrix(prhs[1]);
	NVMatrix* filters = getMatrix(prhs[2]); 
	NVMatrix* targets = getMatrix(prhs[3]); 

	const mwSize* images_dims = mxGetDimensions(prhs[1]);
	int imgSizeY = (int)mxGetScalar(prhs[4]);
	int paddingStart = (int)mxGetScalar(prhs[8]);

	int filterSize = (int)mxGetScalar(prhs[6]);
	int moduleStride = (int)mxGetScalar(prhs[7]);
	int numModulesY = 1 + int(ceil((2 * paddingStart + imgSizeY - filterSize) / float(moduleStride)));
	int numModulesX = numModulesY;
	int numImgColors = (int)mxGetScalar(prhs[5]);
	int numGroups = 1;
	images->transpose();
	filters->transpose();
	targets->transpose();
	convFilterActs(*images, *filters, *targets,
                       imgSizeY, numModulesY, numModulesX, -paddingStart, moduleStride,
                       numImgColors, numGroups, 0, 1);
	images->transpose();
	filters->transpose();
	targets->transpose();
}

void ConvResponseNormCrossMap(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 9 && nlhs == 0);
	NVMatrix* inputs = getMatrix(prhs[1]);
	NVMatrix* denoms = getMatrix(prhs[2]);
	NVMatrix* targets = getMatrix(prhs[3]);
	inputs->transpose();
	denoms->transpose();
	targets->transpose();
	int numFilters = (int)mxGetScalar(prhs[4]);
	int n = (int)mxGetScalar(prhs[5]);
	float k = (float)mxGetScalar(prhs[6]);
	float alpha = (float)mxGetScalar(prhs[7]);
	float beta = (float)mxGetScalar(prhs[8]);
        convResponseNormCrossMap(*inputs, *denoms, *targets, numFilters, n, k, alpha, beta, false);
	inputs->transpose();
	denoms->transpose();
	targets->transpose();
}


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

