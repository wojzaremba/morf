#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>

// MonoConvCpp(X, Xmono, Cmono, Wmono, B, perm, out, stride, obj.padding);
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
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
	int out_depth = w_size[0];
	int patch = w_size[1];
	int inner_depth = c_size[1];
	assert(c_size[0] == in_depth);
	assert(patch == w_size[2]);
	assert(out_depth == bias_size[0]);
	assert(out_size[0] == bs);
	assert(out_size[3] == out_depth);
	int out_rows = ceil((float)(in_rows - patch + 2 * padding) / stride) + 1;
	int out_cols = ceil((float)(in_cols - patch + 2 * padding) / stride) + 1;
	assert(out_size[1] == out_rows);
	assert(out_size[2] == out_cols);
	assert(i_mono_size[0] == bs);
	assert(i_mono_size[1] == in_rows);
	assert(i_mono_size[2] == in_cols);
	assert(i_mono_size[3] == inner_depth);

	// Color matrix multiplication.
	for (int b = 0; b < bs; ++b) {
	  for (int x = 0; x < in_rows; ++x) {
	    for (int y = 0; y < in_cols; ++y) {
	      for (int i_d = 0; i_d < inner_depth; ++i) {
		int i_mono_idx = b + bs * (x + in_rows * (y + in_cols * i_d));
		i_mono[i_mono_idx] = 0.;
	        for (int d = 0; d < in_depth; ++d) {
		  int i_idx = b + bs * (x + in_rows * (y + in_cols * d));
		  int c_idx = d + in_depth * i_d;
		  i_mono[i_mono_idx] += i[i_idx] * c[c_idx];
	        }
	      }
	    }
	  }
	}

//mexPrintf("in_depth = %d\n", in_depth);
//mexPrintf("patch = %d, padding = %d, stride = %d\n", patch, padding, stride);
//mexPrintf("bs = %d, out_rows = %d, out_cols = %d, out_depth = %d\n", bs, out_rows, out_cols, out_depth);	
	for (int b = 0; b < bs; ++b) {
	  for (int x = 0; x < out_rows; ++x) {
	    for (int y = 0; y < out_cols; ++y) {
	      for (int d = 0; d < out_depth; ++d) {
//mexPrintf("b = %d, x = %d, y = %d, d = %d\n", b, x, y, d);
		int out_idx = b + bs * (x + out_rows * (y + d * out_cols));
		out[out_idx] = 0.;
		int in_d = (int)perm[d];
		for (int px = 0; px < patch; ++px) {
		  for (int py = 0; py < patch; ++py) {
//mexPrintf("\tid_d = %d, px = %d, py = %d\n", in_d, px, py);
		    int w_idx = d + out_depth * (px + patch * (py + in_d * patch));
		    int x_idx = x * stride + px;
		    int y_idx = y * stride + py;
		    if ((x_idx >= in_rows) || (y_idx >= in_cols)) {
		      continue;
		    }
		    int i_idx = b + bs * (x_idx + in_rows * (y_idx + in_d * in_cols));
//mexPrintf("\tw = %f, i = %f\n", w[w_idx], i[i_idx]);
                    out[out_idx] += w[w_idx] * i[i_idx];
		  }
		}
		out[out_idx] += bias[d];
		// ReLU.
		if (out[out_idx] < 0) {
		  out[out_idx] = 0;
		}
	      }
	    }
	  }
	}
//	mexPrintf("out[0] = %f\n", out[0]);
}

