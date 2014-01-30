#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>
#include "../../../external/eigen/Eigen/Dense"
#include "templates.h"

// ConvCpp(X, W, B, out, stride, pading);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
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
	int bs = i_size[0];
	int in_rows = i_size[1];
	int in_cols = i_size[2];
	int in_depth = i_size[3];
	int out_depth = w_size[0];
	int patch = w_size[1];
	assert_(patch == w_size[2]);
	assert_(in_depth == w_size[3]);
	assert_(out_depth == bias_size[0]);
	assert_(out_size[0] == bs);
	assert_(out_size[3] == out_depth);
	int out_rows = ceil((float)(in_rows - patch + 2 * padding) / stride) + 1;
	int out_cols = ceil((float)(in_cols - patch + 2 * padding) / stride) + 1;
	assert_(out_size[1] == out_rows);
	assert_(out_size[2] == out_cols);
		
	memset(out, 0, sizeof(float) * bs * out_rows * out_cols * out_depth);
	for (int y = 0; y < out_cols; ++y) {
	  for (int x = 0; x < out_rows; ++x) {
	    for (int py = 0; py < patch; ++py) {
              int y_idx = y * stride + py;
	      if (y_idx >= in_cols) {
	        break;
	      }
              for (int px = 0; px < patch; ++px) {
		int x_idx = x * stride + px;
		if (x_idx >= in_rows) {
	          break;
		}
	        for (int in_d = 0; in_d < in_depth; ++in_d) {
		  int w_offset = out_depth * (px + patch * (py + in_d * patch));
		  int i_offset = bs * (x_idx + in_rows * (y_idx + in_d * in_cols));
		  Eigen::Map<Eigen::VectorXf>i_vec(i + i_offset, bs); 
		  int out_offset = bs * (x + out_rows * y);
		  Eigen::Map<Eigen::VectorXf>w_vec(w + w_offset, out_depth);
	          for (int d = 0; d < out_depth; ++d) {
		    Eigen::Map<Eigen::VectorXf>out_vec(out + out_offset, bs); 
		    out_vec += w_vec(d) * i_vec;
		    out_offset += bs * out_rows * out_cols;
		  }
		}
	      }
	    }
	  }
	}

        for (int d = 0; d < out_depth; ++d) {
  	  float bias_val = AT(bias, d);
	  for (int y = 0; y < out_cols; ++y) {
	    for (int x = 0; x < out_rows; ++x) {
	      for (int b = 0; b < bs; ++b) {
	        int out_idx = b + bs * (x + out_rows * (y + d * out_cols));
	        AT(out, out_idx) += bias_val;
                // ReLU.
                if (AT(out, out_idx) < 0) {
                  AT(out, out_idx) = 0;
                }
              }
            }
          }
        }
        print("lookups = %d\n", lookups);
}

