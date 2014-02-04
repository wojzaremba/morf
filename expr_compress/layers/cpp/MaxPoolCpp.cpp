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
using Eigen::Stride;
using Eigen::Dynamic;
using Eigen::VectorXf;

template<int BS, int IN_D>
void maxpooling(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	float *i = (float*) mxGetData(prhs[0]);
	float *out = (float*) mxGetData(prhs[1]);
	int patch = (int)(*(double*)mxGetData(prhs[2]));
	int stride = (int)(*(double*)mxGetData(prhs[3]));
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	const mwSize* out_size = mxGetDimensions(prhs[1]);
	assert(i_size[0] == BS);
	int in_s = i_size[1];
	assert(i_size[1] == i_size[2]);
	assert(i_size[3] == IN_D);
	assert_(out_size[0] == BS);
	assert_(out_size[3] == IN_D);
	int out_s = ceil((float)(in_s - patch) / stride) + 1;
	assert_(out_size[1] == out_s);
	assert_(out_size[2] == out_s);
		
	Eigen::Map<MatrixXf>out_mat(out, BS * out_s * out_s, IN_D);
	out_mat.setConstant(-1e6); // -Inf.
	
	#pragma omp parallel for
	for (int y = 0; y < out_s; ++y) {
	  #pragma omp parallel for
	  for (int x = 0; x < out_s; ++x) {
	    Eigen::Map<MatrixXf, 1, Stride<Dynamic, 1> > out_stride_mat(out + BS * (x + out_s * y), BS, IN_D, Stride<Dynamic, 1>(BS * out_s * out_s, 1)); 
	    int y_offset = y * stride;
	    for (int py = fmax(0, -y_offset); py < fmin(patch, in_s - y_offset); ++py) {
              int y_idx = y_offset + py;
	      int x_offset = x * stride;
	      int x_idx = x_offset + fmax(0, -x_offset);
              for (int px = fmax(0, -x_offset); px < fmin(patch, in_s - x_offset); ++px) {
	    	Eigen::Map<MatrixXf, 1, Stride<Dynamic, 1> > i_stride_mat(i + BS * (x_idx + in_s * y_idx), BS, IN_D, Stride<Dynamic, 1>(BS * in_s * in_s, 1)); 
		out_stride_mat.array() = out_stride_mat.array().max(i_stride_mat.array());
		++x_idx;
	      }
	    }
	  }
	}
}

// ConvCpp(X, W, B, out, stride, pading);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	if (nrhs == 0) {
		return;
	}
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	int BS = i_size[0];
	int IN_D = i_size[3];
	if (BS == 1) { 
		if (IN_D == 3) {
			maxpooling<1, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			maxpooling<1, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			maxpooling<1, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			maxpooling<1, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			maxpooling<1, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else if (BS == 128) { 
		if (IN_D == 3) {
			maxpooling<128, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			maxpooling<128, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			maxpooling<128, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			maxpooling<128, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			maxpooling<128, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else {
		assert_(0);
	}
}
