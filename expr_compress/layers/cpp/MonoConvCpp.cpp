#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>
#include "templates.h"
#include "../../../external/eigen/Eigen/Dense"

// MonoConvCpp(X, Xmono, Cmono, Wmono, B, perm, out, stride, obj.padding);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
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
	int bs = i_size[0];
	int in_rows = i_size[1];
	int in_cols = i_size[2];
	int in_depth = i_size[3];
	int out_depth = w_size[2];
	int patch = w_size[1];
	int inner_depth = c_size[1];
	assert_(c_size[0] == in_depth);
	assert_(patch == w_size[0]);
	assert_(out_depth == bias_size[0]);
	assert_(out_size[0] == bs);
	assert_(out_size[3] == out_depth);
	int out_rows = ceil((float)(in_rows - patch + 2 * padding) / stride) + 1;
	int out_cols = ceil((float)(in_cols - patch + 2 * padding) / stride) + 1;
	assert_(out_size[1] == out_rows);
	assert_(out_size[2] == out_cols);
	assert_(i_mono_size[0] == bs);
	assert_(i_mono_size[1] == in_rows);
	assert_(i_mono_size[2] == in_cols);
	assert_(i_mono_size[3] == inner_depth);

	memset(out, 0, sizeof(float) * bs * out_rows * out_cols * out_depth);
	// Color matrix multiplication.
	Eigen::Map<Eigen::MatrixXf>i_mat(i, bs * in_rows * in_cols, in_depth);
	Eigen::Map<Eigen::MatrixXf>i_mono_mat(i_mono, bs * in_rows * in_cols, inner_depth);
	Eigen::Map<Eigen::MatrixXf>c_mat(c, in_depth, inner_depth);
	i_mono_mat = i_mat * c_mat;
	
	assert_(out_depth % inner_depth == 0);
        int num_image_colors = out_depth / inner_depth;
	for (int y = 0; y < out_cols; ++y) {
	  for (int x = 0; x < out_rows; ++x) {
	    for (int py = 0; py < patch; ++py) {
	      int y_idx = y * stride + py;
	      if (y_idx >= in_cols) {
	        continue;
	      }
	      for (int px = 0; px < patch; ++px) {
		int x_idx = x * stride + px;
		if (x_idx >= in_rows) {
	          continue;
		}
                for (int d = 0; d < out_depth / num_image_colors; ++d) {
		  int i_mono_offset = bs * (x_idx + in_rows * (y_idx + d * in_cols));
		  Eigen::Map<Eigen::VectorXf>i_mono_vec(i_mono + i_mono_offset, bs); 
		  for (int n = 0; n < num_image_colors; ++n) { 
                    int in_d = (int)AT(perm, d * num_image_colors + n);
		    int w_idx = px + patch * (py + in_d * patch);
		    float w_val = AT(w, w_idx);
		    int out_offset = bs * (x + out_rows * (y + in_d * out_cols));
		    Eigen::Map<Eigen::VectorXf>out_vec(out + out_offset, bs); 
		    out_vec += w_val * i_mono_vec;
                  }
		}
	      }
	    }
	  }
	}

	Eigen::Map<Eigen::VectorXf>bias_vec(bias, out_depth);
	Eigen::Map<Eigen::MatrixXf>out_mat(out, bs * out_rows * out_cols, out_depth);
	out_mat.rowwise() += bias_vec.transpose();
	out_mat.array() = (out_mat.array() + out_mat.array().abs()) / 2.f; // ReLU.
	print("lookups = %d\n", lookups);
}

