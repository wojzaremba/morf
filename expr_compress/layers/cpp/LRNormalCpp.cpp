#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>
#include "templates.h"
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <Eigen/Core>


using Eigen::MatrixXf;

template<int BS, int IN_D>
void lrnormal(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	float *i = (float*) mxGetData(prhs[0]);
	float *out = (float*) mxGetData(prhs[1]);
	float k = *(float*)mxGetData(prhs[2]);
	float n = *(float*)mxGetData(prhs[3]);
	float alpha = *(float*)mxGetData(prhs[4]);
	float beta = *(float*)mxGetData(prhs[5]);
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	const mwSize* out_size = mxGetDimensions(prhs[1]);
	assert(i_size[0] == BS);
	int in_s = i_size[1];
	assert(i_size[1] == i_size[2]);
	assert(i_size[3] == IN_D);
	assert_(out_size[0] == BS);
	assert_(out_size[3] == IN_D);
	assert_(out_size[1] == in_s);
	assert_(out_size[2] == in_s);
		
	Eigen::Map<MatrixXf>out_mat(out, BS * in_s * in_s, IN_D);
	out_mat.setConstant(0);
	Eigen::Map<MatrixXf>i_mat(i, BS * in_s * in_s, IN_D);
	#pragma omp parallel for
	for (int idx = 0; idx < IN_D; ++idx) {
		for (int idx_i = fmax(idx - ((n - 1) / 2), 0); idx_i <= fmin(idx + (n - 1) / 2, IN_D - 1); ++idx_i) {
			out_mat.col(idx) += i_mat.array().square().matrix().col(idx_i);
			
		} 
	}
	out_mat.array() = out_mat.array() * alpha + k;
	out_mat.array() = out_mat.array().pow(beta);
	out_mat.array() = i_mat.array() / out_mat.array();
}

void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs == 0) {
		return;
	}
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	int BS = i_size[0];
	int IN_D = i_size[3];
	if (BS == 1) { 
		if (IN_D == 3) {
			lrnormal<1, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			lrnormal<1, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			lrnormal<1, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			lrnormal<1, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			lrnormal<1, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else if (BS == 128) { 
		if (IN_D == 3) {
			lrnormal<128, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			lrnormal<128, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			lrnormal<128, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			lrnormal<128, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			lrnormal<128, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else {
		assert_(0);
	}
}
