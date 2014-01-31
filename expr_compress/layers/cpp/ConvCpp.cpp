#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>
#include "../../../external/eigen/Eigen/Dense"
#include "templates.h"

using namespace Eigen;

template<int BS>
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
	//int BS = i_size[0];
	assert(i_size[0] == BS);
	int in_rows = i_size[1];
	int in_cols = i_size[2];
	int in_depth = i_size[3];
	int out_depth = w_size[0];
	int patch = w_size[1];
	assert_(patch == w_size[2]);
	assert_(in_depth == w_size[3]);
	assert_(out_depth == bias_size[0]);
	assert_(out_size[0] == BS);
	assert_(out_size[3] == out_depth);
	int out_rows = ceil((float)(in_rows - patch + 2 * padding) / stride) + 1;
	int out_cols = ceil((float)(in_cols - patch + 2 * padding) / stride) + 1;
	assert_(out_size[1] == out_rows);
	assert_(out_size[2] == out_cols);
		
	Map<MatrixXf>out_mat(out, BS * out_rows * out_cols, out_depth);
	out_mat.setZero();
	for (int y = 0; y < out_cols; ++y) {
	  for (int x = 0; x < out_rows; ++x) {
	    Map<MatrixXf, 1, Stride<Dynamic, 1> > out_stride_mat(out + BS * (x + out_rows * y), BS, out_depth, Stride<Dynamic, 1>(BS * out_rows * out_cols, 1)); 
	    for (int py = 0; py < patch; ++py) {
              int y_idx = y * stride + py - padding;
	      if ((y_idx >= in_cols) || (y_idx < 0)) {
	        continue;
	      }
	      int x_offset = x * stride - padding;
              for (int px = fmax(0, -x_offset); px < fmin(patch, in_rows - x_offset); ++px) {
	        int x_idx = x_offset + px;
	    	Map<MatrixXf, 1, Stride<Dynamic, 1> > i_stride_mat(i + BS * (x_idx + in_rows * y_idx), BS, in_depth, Stride<Dynamic, 1>(BS * in_rows * in_cols, 1)); 
	    	Map<MatrixXf, 1, Stride<Dynamic, 1> > w_stride_mat(w + out_depth * (px + patch * py), out_depth, in_depth, Stride<Dynamic, 1>(out_depth * patch * patch, 1)); 
		out_stride_mat += i_stride_mat * w_stride_mat.transpose();
	      }
	    }
	  }
	}

	Map<VectorXf>bias_vec(bias, out_depth);
	out_mat.rowwise() += bias_vec.transpose();
	out_mat.array() /= 2.f; 
	out_mat.array() = (out_mat.array() + out_mat.array().abs()); // ReLU.
	print("lookups = %d\n", lookups);
}

// ConvCpp(X, W, B, out, stride, pading);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	int BS = i_size[0];
	if (BS == 1) { 
		conv<1>(nlhs, plhs, nrhs, prhs);
	} else if (BS == 128) { 
		conv<128>(nlhs, plhs, nrhs, prhs);
	} else {
		assert_(0);
	}
}

