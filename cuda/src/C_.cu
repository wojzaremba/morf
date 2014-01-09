#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"

#include "conv_util.cuh"
#include "cudaconv2.cuh"
#include "filter_acts.cuh"
#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"
#include "weight_acts.cuh"


using namespace std;

static std::vector<NVMatrix*> p;
static cudaEvent_t start_event, end_event;


#define GET3()       assert_((nrhs == 4) && (nlhs == 0)); \
   	             NVMatrix* a = getMatrix(prhs[1]); \
	             NVMatrix* b = getMatrix(prhs[2]); \
	             NVMatrix* c = getMatrix(prhs[3]);

#define GET1X1(TYPE) assert_((nrhs == 4) && (nlhs == 0)); \
		     NVMatrix* a = getMatrix(prhs[1]); \
	             TYPE b = (TYPE)mxGetScalar(prhs[2]); \
	             NVMatrix* c = getMatrix(prhs[3]);
                 
#define GET2()       assert_((nrhs == 3) && (nlhs == 0)); \
	             NVMatrix* a = getMatrix(prhs[1]); \
	             NVMatrix* b = getMatrix(prhs[2]); 


NVMatrix* getMatrix(const mxArray *prhs) {
	int idx = (int)mxGetScalar(prhs);
	assert_(idx < p.size());
	assert_(p[idx] != NULL);
	return p[idx];
}

void cleanUp() {
	for (int i = 0; i < p.size(); ++i) {
		if (p[i] != NULL) {
			delete(p[i]);
		}
	}
	p.clear();
	cudaDeviceReset();
}

void CleanGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 1 && nlhs == 0);
	cleanUp();
}

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

void Reshape(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
        assert_((nrhs == 4) && (nlhs == 0));
        NVMatrix* a = getMatrix(prhs[1]);
	int b = (int)mxGetScalar(prhs[2]);
	int c = (int)mxGetScalar(prhs[3]);
	a->reshape(c, b);
}

void ActEXP(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
        a->apply(NVMatrixOps::Exp(), *b);
}

void ActRELU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
	a->maxWithScalar(0, *b); 
}

void dActRELU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
	a->biggerThanScalar(0, *b); 
}

void dActLINEAR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
	assert_(0);
}

void ActLINEAR(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
	b = a;
}

void AddVector(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET3();
	a->addVector(*b, *c);
}

void Add(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET3();
	a->add(*b, *c);
}

void Subtract(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET3();
	a->subtract(*b, *c);
}

void Scale(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET1X1(float);
	a->scale(b, *c);
}

void Mult(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET3();
	a->rightMult(*b, *c);
}

void EltwiseMult(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET3();
	a->eltwiseMult(*b, *c);
}

void Sum(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET1X1(int);
	a->sum(b, *c);
}

void Max(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET1X1(int);
	a->max(b, *c);
}

void Transpose(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 2) && (nlhs == 0));
	NVMatrix* a = getMatrix(prhs[1]);
	a->transpose();
}

void EltwiseDivideByVector(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	GET2();
	a->eltwiseDivideByVector(*b);
}

void PrintShape(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 2 && nlhs == 0);
	NVMatrix* nvmatrix = getMatrix(prhs[1]);
	nvmatrix->printShape("matrix");
}

void CopyFromGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 2 && nlhs == 1);
	NVMatrix* nvmatrix = getMatrix(prhs[1]);
	plhs[0] = nvmatrix->copyToHost();
}

void CopyToGPU(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 3) && (nlhs == 0));
	int pid = (int)mxGetScalar(prhs[1]);
	if (p.size() <= pid) {
		for (int i = p.size(); i <= pid; ++i) {
			p.push_back(NULL);
		}
	}
	if (p[pid] != NULL) {
		p[pid]->copyFromHost(prhs[2], true);
	} else { 
		p[pid] = new NVMatrix(prhs[2], true);
	}
	p[pid]->setTrans(true);
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

void StartTimer(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 1 && nlhs == 0);
	cudaEventCreate(&start_event);
	cudaEventCreate(&end_event);
	cudaEventRecord(start_event, 0);
}

void StopTimer(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_(nrhs == 1 && nlhs == 1);
	cudaEventRecord(end_event, 0);
	cudaEventSynchronize(start_event);
	cudaEventSynchronize(end_event);
    	plhs[0] = mxCreateNumericMatrix(1, 1, mxSINGLE_CLASS, mxREAL);
	float* lapse = (float*) mxGetData(plhs[0]);
	cudaEventElapsedTime(lapse, start_event, end_event);
	(*lapse) /= 1000.; // returned time is in ms.
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

