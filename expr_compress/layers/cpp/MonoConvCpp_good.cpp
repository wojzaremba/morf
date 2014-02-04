#include <stdio.h>
#include <algorithm>
#include <iostream>
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
void monoconv(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	lookups = 0;
	float *i = (float*) mxGetData(prhs[0]);
	float *i_mono = (float*) mxGetData(prhs[1]);
	float *c = (float*) mxGetData(prhs[2]);
	float *w = (float*) mxGetData(prhs[3]);
	float *bias = (float*) mxGetData(prhs[4]);
	float *perm = (float*) mxGetData(prhs[5]);
	float *out = (float*) mxGetData(prhs[6]);
	int stride = (int)(*(double*)mxGetData(prhs[7]));
	int padding = (int)(*(double*)mxGetData(prhs[8]));
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	const mwSize* i_mono_size = mxGetDimensions(prhs[1]);
	const mwSize* c_size = mxGetDimensions(prhs[2]);
	const mwSize* w_size = mxGetDimensions(prhs[3]);
	const mwSize* bias_size = mxGetDimensions(prhs[4]);
	const mwSize* perm_size = mxGetDimensions(prhs[5]);
	const mwSize* out_size = mxGetDimensions(prhs[6]);
	assert(BS == i_size[0]);
	int in_s = i_size[1];
	assert(in_s == i_size[2]);
	assert(IN_D == i_size[3]);
	int out_depth = w_size[2];
	int patch = w_size[1];
	int inner_depth = c_size[1];
	assert_(c_size[0] == IN_D);
	assert_(patch == w_size[0]);
	assert_(out_depth == bias_size[0]);
	assert_(out_size[0] == BS);
	assert_(out_size[3] == out_depth);
	int out_s = ceil((float)(in_s - patch + 2 * padding) / stride) + 1;
	assert_(out_size[1] == out_s);
	assert_(out_size[2] == out_s);
	assert_(i_mono_size[0] == BS);
	assert_(i_mono_size[1] == in_s);
	assert_(i_mono_size[2] == in_s);
	assert_(i_mono_size[3] == inner_depth);

	memset(out, 0, sizeof(float) * BS * out_s * out_s * out_depth);
	// Color matrix multiplication.
	Eigen::Map<MatrixXf>i_mat(i, BS * in_s * in_s, IN_D);
	Eigen::Map<MatrixXf>i_mono_mat(i_mono, BS * in_s * in_s, inner_depth);
	Eigen::Map<MatrixXf>c_mat(c, IN_D, inner_depth);
	i_mono_mat = i_mat * c_mat;
	
	assert_(out_depth % inner_depth == 0);
        int num_image_colors = out_depth / inner_depth;
	#pragma omp parallel for
	for (int y = 0; y < out_s; ++y) {
	  #pragma omp parallel for
	  for (int x = 0; x < out_s; ++x) {
	    int y_offset = y * stride - padding;
	    for (int py = fmax(0, -y_offset); py < fmin(patch, in_s - y_offset); ++py) {
              int y_idx = y_offset + py;
	      int x_offset = x * stride - padding;
	      int x_idx = x_offset + fmax(0, -x_offset);
              for (int px = fmax(0, -x_offset); px < fmin(patch, in_s - x_offset); ++px) {
                for (int d = 0; d < out_depth / num_image_colors; ++d) {
		  int i_mono_offset = BS * (x_idx + in_s * (y_idx + d * in_s));
		  Eigen::Map<VectorXf>i_mono_vec(i_mono + i_mono_offset, BS); 
		  for (int n = 0; n < num_image_colors; ++n) { 
                    int in_d = (int)AT(perm, d * num_image_colors + n);
		    int w_idx = px + patch * (py + in_d * patch);
		    float w_val = AT(w, w_idx);
		    int out_offset = BS * (x + out_s * (y + in_d * out_s));
		    Eigen::Map<VectorXf>out_vec(out + out_offset, BS); 
		    out_vec += w_val * i_mono_vec;
                  }
		}
	      }
	    }
	  }
	}
	Eigen::Map<VectorXf>bias_vec(bias, out_depth);
	Eigen::Map<MatrixXf>out_mat(out, BS * out_s * out_s, out_depth);
	out_mat.rowwise() += bias_vec.transpose();
	out_mat.array() /= 2.f; 
	out_mat.array() = (out_mat.array() + out_mat.array().abs()); // ReLU.
	print("lookups = %d\n", lookups);
}

// MonoConvCpp(X, Xmono, Cmono, Wmono, B, perm, out, stride, obj.padding);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	const mwSize* i_size = mxGetDimensions(prhs[0]);
	int BS = i_size[0];
	int IN_D = i_size[3];
	if (BS == 1) { 
		if (IN_D == 3) {
			monoconv<1, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			monoconv<1, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			monoconv<1, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			monoconv<1, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			monoconv<1, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else if (BS == 128) { 
		if (IN_D == 3) {
			monoconv<128, 3>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 96) {
			monoconv<128, 96>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 256) {
			monoconv<128, 256>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 512) {
			monoconv<128, 512>(nlhs, plhs, nrhs, prhs);
		} else if (IN_D == 1024) {
			monoconv<128, 1024>(nlhs, plhs, nrhs, prhs);
		} else {
			mexPrintf("IN_D = %d\n", IN_D);
			assert_(0);
		}
	} else {
		assert_(0);
	}
}
