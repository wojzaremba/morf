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


using Eigen::MatrixXf;
using Eigen::Stride;
using Eigen::Dynamic;
using Eigen::VectorXf;

template<int BS, int IN_D>
void conv(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	lookups = 0;
	float *i = (float*) mxGetData(prhs[0]);
	float *w = (float*) mxGetData(prhs[1]);
	float *bias = (float*) mxGetData(prhs[2]);
	float *out = (float*) mxGetData(prhs[3]);
	int stride = (int)(*(double*)mxGetData(prhs[4]));
	int padding = (int)(*(double*)mxGetData(prhs[5]));
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	const mwSize* w_size = mxGetDimensions(prhs[1]);
	const mwSize* bias_size = mxGetDimensions(prhs[2]);
	const mwSize* out_size = mxGetDimensions(prhs[3]);
	assert(i_size[0] == BS);
	int in_s = i_size[1];
	assert(i_size[1] == i_size[2]);
	assert(i_size[3] == IN_D);
	int out_depth = w_size[0];
	int patch = w_size[1];
	assert_(patch == w_size[2]);
	assert_(IN_D == w_size[3]);
	assert_(out_depth == bias_size[0]);
	assert_(out_size[0] == BS);
	assert_(out_size[3] == out_depth);
	int out_s = ceil((float)(in_s - patch + 2 * padding) / stride) + 1;
	assert_(out_size[1] == out_s);
	assert_(out_size[2] == out_s);
		
	Eigen::Map<MatrixXf>out_mat(out, BS * out_s * out_s, out_depth);
	out_mat.setZero();
	
	#pragma omp parallel for
	for (int y = 0; y < out_s; ++y) {
	  #pragma omp parallel for
	  for (int x = 0; x < out_s; ++x) {
	    Eigen::Map<MatrixXf, 1, Stride<Dynamic, 1> > out_stride_mat(out + BS * (x + out_s * y), BS, out_depth, Stride<Dynamic, 1>(BS * out_s * out_s, 1)); 
	    int y_offset = y * stride - padding;
	    for (int py = fmax(0, -y_offset); py < fmin(patch, in_s - y_offset); ++py) {
              int y_idx = y_offset + py;
	      int x_offset = x * stride - padding;
	      int x_idx = x_offset + fmax(0, -x_offset);
              for (int px = fmax(0, -x_offset); px < fmin(patch, in_s - x_offset); ++px) {
	    	Eigen::Map<MatrixXf, 1, Stride<Dynamic, 1> > i_stride_mat(i + BS * (x_idx + in_s * y_idx), BS, IN_D, Stride<Dynamic, 1>(BS * in_s * in_s, 1)); 
	    	Eigen::Map<MatrixXf, 1, Stride<Dynamic, 1> > w_stride_mat(w + out_depth * (px + patch * py), out_depth, IN_D, Stride<Dynamic, 1>(out_depth * patch * patch, 1)); 
		out_stride_mat += i_stride_mat * w_stride_mat.transpose();
		++x_idx;
	      }
	    }
	  }
	}

	Eigen::Map<VectorXf>bias_vec(bias, out_depth);
	out_mat.rowwise() += bias_vec.transpose();
	out_mat.array() /= 2.f; 
	out_mat.array() = (out_mat.array() + out_mat.array().abs()); // ReLU.
	print("lookups = %d\n", lookups);
}

// ConvCpp(X, W, B, out, stride, pading);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	int BS = i_size[0];
	int IN_D = i_size[3];
	if (BS == 1) { 
		if (IN_D == 3) {
			conv<1, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			conv<1, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			conv<1, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			conv<1, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			conv<1, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else if (BS == 128) { 
		if (IN_D == 3) {
			conv<128, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			conv<128, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			conv<128, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			conv<128, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			conv<128, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else {
		assert_(0);
	}
}
