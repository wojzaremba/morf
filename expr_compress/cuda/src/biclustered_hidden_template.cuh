#include <stdio.h>
#include <algorithm>
#include <vector>

#include "common_conv.cuh"
#include "nvmatrix.cuh"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

using namespace std;

#ifndef COMMON_CUH
#define COMMON_CUH
enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};


void  convFilterActsClustered(NVMatrix& images, NVMatrix& F,  NVMatrix& C, NVMatrix& XY, NVMatrix& targets, NVMatrix& inPerm, NVMatrix& outPerm,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int newNumColors, int numFilters);

#endif  /* COMMON_CUH */

void biclustered_hidden(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
        assert_((nrhs == 14) && (nlhs == 0));
        NVMatrix* images = getMatrix(prhs[1]);
        NVMatrix* F = getMatrix(prhs[2]); 
        NVMatrix* C = getMatrix(prhs[3]); 
		NVMatrix* XY = getMatrix(prhs[4]); 
		NVMatrix* targets = getMatrix(prhs[5]); 
        NVMatrix* inPerm = getMatrix(prhs[6]);
        NVMatrix* outPerm = getMatrix(prhs[7]);
        
        const mwSize* images_dims = mxGetDimensions(prhs[1]);
        int imgSizeY = (int)mxGetScalar(prhs[8]);
		int numImgColors = (int)mxGetScalar(prhs[9]);
		int numFilters = (int)mxGetScalar(prhs[10]);
		int filterSize = (int)mxGetScalar(prhs[11]);
		int moduleStride = (int)mxGetScalar(prhs[12]);
        int paddingStart = (int)mxGetScalar(prhs[13]);
        
        int numModulesY = 1 + int(ceil((2 * paddingStart + imgSizeY - filterSize) / float(moduleStride)));
        int numModulesX = numModulesY;
        
        images->transpose();
        F->transpose();
		C->transpose();
		XY->transpose();
        targets->transpose();
        convFilterActsClustered(*images, *F, *C, *XY, *targets, *inPerm, *outPerm,
                       imgSizeY, numModulesY, numModulesX, -paddingStart, moduleStride,
                       numImgColors, numFilters);
        images->transpose();
        F->transpose();
		C->transpose();
		XY->transpose();
        targets->transpose();
}

/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of B_Y * filtersPerThread
 *
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * F: 			(sizeClustOut, rank, numClustIn, numClustOut)
 * C:			(sizeClustIn, rank, numClustIn, numClustOut) 
 * XY: 			(filterPixels, rank, numClustIn, numClustOut)  
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 * colorCache: how many colors to put into shmem
 *
 * numFilters should be divisible by B_Y * filtersPerThread
 * numImages be divisible by B_X * imgsPerThread
 * numFilterColors should be divisible by colorCache.
 * numImgColors must be even.
 * numFilters must be divisible by numGroups.
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int clustersPerBlock, int numClustIn, int numClustOut, int sizeClustIn, int sizeClustOut, int rank, int colorCache, bool checkImgBounds>
__global__ void filterActs_YxX(float* images, float* F, float* C, float* XY, float* targets, float* inPerm, float* outPerm,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors) {
	__shared__ float shC[sizeClustIn][rank];
	__shared__ float shXY[B_Y][rank];
    __shared__ float shImages[B_Y * sizeClustIn][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / sizeClustOut;
    const int moduleIdx = blockIdx.x / blocksPerModule;
    const int blockFilterIdx = sizeClustOut * (blockIdx.x % blocksPerModule);
    const int blockGroupIdx = blockFilterIdx / numFilters;
	const int outClustIdx = blockFilterIdx / numClustOut;
    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numImgColors * blockGroupIdx;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int myImgIdx = blockIdx.y * B_X * imgsPerThread + threadIdx.x;

    images += myImgIdx; //blockColorIdx * imgPixels * imgStride + myImgIdx;

	F += outClustIdx * numClustIn * rank * sizeClustOut; //numClustOut dimension
	  // + threadIdx.y * sizeClustOut;

	C += outClustIdx * numClustIn * rank * sizeClustIn //numClustOut dimension
	  + threadIdx.x * sizeClustIn; //rank dimension

	XY += outClustIdx * numClustIn * rank * filterPixels //numClustOut dimension
	   + threadIdx.x * filterPixels;

    targets += moduleIdx * numImages
            + blockFilterIdx * numImages * numModules
            + myImgIdx;


    float prod[imgsPerThread];
   	for (int ic = 0; ic < 1; ic ++) { // ic stands for input cluster
		
		#pragma unroll
		for(int g = 0; g < imgsPerThread; g++) {
		   prod[g] = 0;
		}
		
		// Increment F
		F += ic * rank * sizeClustOut;
	
		// Fill shC
		if (threadIdx.y < sizeClustIn && threadIdx.y < rank) {
			shC[threadIdx.y][threadIdx.x] = C[threadIdx.y];
		}	

   	 	for (int p = 0; p < filterPixels; p += B_Y) {
        	/*
			 * Load B_Y pixels from B_X*imgsPerThread images
			 */
			const int pixIdx = p + threadIdx.y;
			if (pixIdx < filterPixels) {
				const int x = imgLoadModPosX + pixIdx % filterSize;
				const int y = imgLoadModPosY + pixIdx / filterSize;
				if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
				   float* m = &images[imgStride * (ic * sizeClustIn * imgPixels + y * imgSizeX + x)];
					#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
						if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
							#pragma unroll
							for (int c = 0; c < sizeClustIn; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
							}
						} else {
							#pragma unroll
							for (int c = 0; c < sizeClustIn; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
							}
						}
					}
				} else { // Padding
					#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
						#pragma unroll
						for (int c = 0; c < sizeClustIn; c++) {
							shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
						}
					}
				}
			}
			__syncthreads();
			/*	
			 * Load B_Y * clustersPerBlock pixels from rankCache filters
			 */
			if (threadIdx.y < B_Y && threadIdx.x < rank) {
				shXY[threadIdx.y][threadIdx.x] = XY[threadIdx.y];
			}
			// Linearly combine input maps, multiply with XY  and store back in shImages
			for (int i = 0; i < imgsPerThread; i++) {
				#pragma unroll
				for (int p = 0; p < B_Y; p++) {
					float cimg = 0;	
					#pragma unroll
					for (int c = 0; c < sizeClustIn; c++) {
						cimg += shImages[p + c * B_Y][threadIdx.x + i * B_X] * shC[c][threadIdx.y];
					}
					prod[i] += cimg * shXY[p][threadIdx.y];
				}
			}
			__syncthreads();
		}
		for (int f = 0; f < sizeClustOut; f++) {
			int filterIdx = (threadIdx.y + f) % sizeClustOut;	
			float Fcoeff = F[filterIdx];
			#pragma unroll
			for (int i = 0; i < imgsPerThread; i++) {
				if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
					targets[i * B_X + filterIdx * numImages * numModules] += sizeClustOut;//filterIdx;// Fcoeff; //prod[i] * Fcoeff;
				}
			}
			__syncthreads();
		}
	}	
}



/*
 * images:      (numImgColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numFilterColors, filterPixels, numFilters)             if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModules, numImages)
 * 
 * Note: all of these convolution routines are optimized for the case when
 * the number of images (i.e. the minibatch size) is a multiple of 128. 
 * Other batch sizes will work, but but I made no attempt whatsoever
 * to make them work fast. 
 */
 void _filterActsClustered(NVMatrix& images, NVMatrix& F, NVMatrix& C, NVMatrix& XY, NVMatrix& targets, NVMatrix& inPerm, NVMatrix& outPerm,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numFilters) {
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    assert_(numImgColors > 3);
    assert_(numFilters % 16== 0);
    assert_(images.getNumRows() == imgPixels * numImgColors);
    assert_(imgSizeY * imgSizeX == imgPixels);

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix
	
    int filterPixels = XY.getNumCols();
    int filterSize = int(sqrt(filterPixels));
    assert_(filterSize * filterSize == filterPixels);

    assert_(F.getNumRows() == #numClustOut *  #numClustIn * #rank);
    assert_(C.getNumRows() == #numClustOut *  #numClustIn * #rank);
    assert_(XY.getNumRows() == #numClustOut *  #numClustIn * #rank);

	assert_(F.getNumCols() == #sizeClustOut);
	assert_(C.getNumCols() == #sizeClustIn);
	assert_(XY.getNumCols() == filterPixels);

	assert_(#numClustIn * #sizeClustIn == numImgColors);
	assert_(#numClustOut * #sizeClustOut == numFilters);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert_(paddingStart <= 0);
    assert_(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert_(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert_(moduleStride <= filterSize);
    
    assert_(!images.isTrans());
    assert_(!F.isTrans());
	assert_(!C.isTrans());
	assert_(!XY.isTrans());
    assert_(!targets.isTrans());

    assert_(F.isContiguous());
    assert_(C.isContiguous());
    assert_(XY.isContiguous());
    assert_(targets.isContiguous());
	
    dim3 blocks = dim3((numModules * numFilters) / (#clustersPerBlock * #sizeClustOut), DIVUP(numImages, #B_X * #imgsPerThread));
    dim3 threads(#B_X, #B_Y);
    targets.resize(numFilters * numModules, numImages);
 
	printf("(G_X, G_Y) = (%d, %d) \n", (numModules * numFilters) / (#clustersPerBlock * #sizeClustOut), DIVUP(numImages, #B_X * #imgsPerThread));
	cudaFuncSetCacheConfig(filterActs_YxX< #B_Y, #B_X, #imgsPerThread, #clustersPerBlock, #numClustIn, #numClustOut, #sizeClustIn, #sizeClustOut, #rank, #colorCache, #checkImgBounds >, cudaFuncCachePreferShared);
    filterActs_YxX< #B_Y, #B_X, #imgsPerThread, #clustersPerBlock, #numClustIn, #numClustOut, #sizeClustIn, #sizeClustOut, #rank, #colorCache, #checkImgBounds> <<<blocks, threads>>>(images.getDevData(), F.getDevData(), C.getDevData(), XY.getDevData(), targets.getDevData(), inPerm.getDevData(), outPerm.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors);

	getLastCudaError("filterActs: kernel execution failed");
}


void convFilterActsClustered(NVMatrix& images, NVMatrix& F, NVMatrix& C, NVMatrix& XY, NVMatrix& targets,  NVMatrix& inPerm, NVMatrix& outPerm ,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numFilters) { 
     _filterActsClustered(images, F, C, XY, targets, inPerm, outPerm, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, numFilters);
}

