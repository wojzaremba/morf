#include <stdio.h>
#include <algorithm>
#include <vector>
#include "mex.h"
#include <assert.h>
#include <math.h>
#include "templates.h"

#define BS 1

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
	assert(i_size[0] == BS);
	int in_rows = i_size[1];
	int in_cols = i_size[2];
	int in_depth = i_size[3];
	int out_depth = w_size[2];
	int patch = w_size[1];
	int inner_depth = c_size[1];
	assert(c_size[0] == in_depth);
	assert(patch == w_size[0]);
	assert(out_depth == bias_size[0]);
	assert(out_size[0] == BS);
	assert(out_size[3] == out_depth);
	int out_rows = ceil((float)(in_rows - patch + 2 * padding) / stride) + 1;
	int out_cols = ceil((float)(in_cols - patch + 2 * padding) / stride) + 1;
	assert(out_size[1] == out_rows);
	assert(out_size[2] == out_cols);
	assert(i_mono_size[0] == BS);
	assert(i_mono_size[1] == in_rows);
	assert(i_mono_size[2] == in_cols);
	assert(i_mono_size[3] == inner_depth);

	// Color matrix multiplication.
	for (int b = 0; b < BS; ++b) {
	  for (int x = 0; x < in_rows; ++x) {
	    for (int y = 0; y < in_cols; ++y) {
	      for (int i_d = 0; i_d < inner_depth; ++i_d) {
		int i_mono_idx = b + BS * (x + in_rows * (y + in_cols * i_d));
		AT(i_mono, i_mono_idx) = 0.;
	        for (int d = 0; d < in_depth; ++d) {
		  int i_idx = b + BS * (x + in_rows * (y + in_cols * d));
		  int c_idx = d + in_depth * i_d;
		  AT(i_mono, i_mono_idx) += AT(i, i_idx) * AT(c, c_idx);
	        }
	      }
	    }
	  }
	}

	assert(out_depth % inner_depth == 0);
	int num_image_colors = out_depth / inner_depth;
        for (int d = 0; d < out_depth; ++d) {
          int in_d = (int)AT(perm, d);
	    for (int y = 0; y < out_cols; ++y) {
	      for (int x = 0; x < out_rows; ++x) {
	        for (int b = 0; b < BS; ++b) {
		int out_idx = b + BS * (x + out_rows * (y + in_d * out_cols));
		AT(out, out_idx) = 0.;
		for (int py = 0; py < patch; ++py) {
		  int y_idx = y * stride + py;
		  if (y_idx >= in_cols) {
		    continue;
		  }
		  int offset =  BS * (in_rows * (y_idx + floor(d / num_image_colors) * in_cols));
		  for (int px = 0; px < patch; ++px) {
		    int x_idx = x * stride + px;
		    if (x_idx >= in_rows) {
		      continue;
		    }
		    int w_idx = px + patch * (py + in_d * patch);
		    int i_mono_idx = b + BS * x_idx + offset;
                    AT(out, out_idx) += AT(w, w_idx) * AT(i_mono, i_mono_idx);
		  }
		}
		//AT(out, out_idx) += AT(bias, d);
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

