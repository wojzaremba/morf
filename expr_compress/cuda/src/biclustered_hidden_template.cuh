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
    //__shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
	//__shared__ float shF[sizeClustOut * clustersPerBlock][rank];
	__shared__ float shC[sizeClustIn * clustersPerBlock][rank];
	__shared__ float shXY[B_Y * clustersPerBlock][rank];
    __shared__ float shImages[B_Y * clustersPerBlock * sizeClustIn][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
	//__shared__ float shOut[rank * clustersPerBlock][B_X * imgsPerThread];	
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
	const int filtersPerBlock = clustersPerBlock * sizeClustOut;
    const int blocksPerModule = numFilters / filtersPerBlock;
    const int moduleIdx = blockIdx.x / blocksPerModule;
    const int blockFilterIdx = clustersPerBlock * sizeClustOut * (blockIdx.x % blocksPerModule);
    const int blockGroupIdx = blockFilterIdx / numFilters;
	const int filtersPerThread = filtersPerBlock / B_Y;
	const int outClustIdx = blockFilterIdx / numClustOut;
    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numImgColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / rank;
    const int shFilterLoadX = tidx % rank; 
    const int myImgIdx = blockIdx.y * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;

    F += outClustIdx * numClustIn * rank * sizeClustOut // numClustOut dimension
      + shFilterLoadX * sizeClustOut; //rank dimension 

	C += outClustIdx * numClustIn * rank * sizeClustIn //numClustOut dimension
	  + shFilterLoadX * sizeClustIn; //rank dimension

	XY += outClustIdx * numClustIn * rank * filterPixels //numClustOut dimension
	   + shFilterLoadX * filterPixels;

    targets += moduleIdx * numImages
            + (blockFilterIdx + threadIdx.y) * numImages * numModules
            + myImgIdx;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
           prod[f][g] = 0;
        }
    }
//__shared__ float shOut[rank * clustersPerBlock][B_X * imgsPerThread];	
/*
	const int shOutLoadY = tidx / (B_X * imgsPerThread);
	const int shOutLoadX = tidx % (B_X * imgsPerThread);
	for (int i = 0; i < rank * clustersPerBlock; i++) {
		if (shOutLoadY + i <  rank * clustersPerBlock) {		
			shOut[shOutLoadY + i][shOutLoadX] = 0;
		}	
	}  	
*/
	#pragma unroll	
   	for (int ic = 0; ic < numClustIn; ic ++) { // ic stands for input cluster
		/*
		* Fill shF, shC and shXY.
		*/ 
	/*
		#pragma unroll
		for (int f = 0; f < filtersPerBlock; f += B_X * B_Y / rank) {
			if (shFilterLoadY + f < sizeClustOut * clustersPerBlock) {
				shF[shFilterLoadY + f][shFilterLoadX] = F[((shFilterLoadY + f) / sizeClustOut) + ((shFilterLoadY + f) % sizeClustOut)];
			}	
		}
	*/	
		F += ic * rank * sizeClustOut;	
		#pragma unroll
		for (int f = 0; f < sizeClustIn * clustersPerBlock; f += B_X * B_Y / rank) {
			if (shFilterLoadY + f < sizeClustIn * clustersPerBlock) {
				shC[shFilterLoadY + f][shFilterLoadX] = C[((shFilterLoadY + f) / sizeClustOut) + ((shFilterLoadY + f) % sizeClustOut)];
			}	
		}

   	 	for (int p = 0; p < filterPixels; p += B_Y) {
        	/*
         	 * Load B_Y * clustersPerBlock pixels from rank filters
        	 */
			#pragma unroll
			for (int f = 0; f < B_Y * clustersPerBlock; f += B_X * B_Y / rank) {
				if (shFilterLoadY + f < B_Y * clustersPerBlock) {
					shXY[shFilterLoadY + f][shFilterLoadX] = XY[((shFilterLoadY + f) / sizeClustOut) + ((shFilterLoadY + f) % sizeClustOut)];
				}	
			}
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
							for (int c = 0; c < clustersPerBlock * sizeClustIn; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
							}
						} else {
							#pragma unroll
							for (int c = 0; c < clustersPerBlock * sizeClustIn; c++) {
								shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
							}
						}
					}
				} else { // Padding
					#pragma unroll
					for (int i = 0; i < imgsPerThread; i++) {
						#pragma unroll
						for (int c = 0; c < clustersPerBlock * sizeClustIn; c++) {
							shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
						}
					}
				}
			}
			__syncthreads();
			//#pragma unroll
			for (int r = 0; r < rank; r++) {	
				for (int i = 0; i < imgsPerThread; i++) {
            	// First linearly combine input maps. Store them in shImages, but be careful!
					for (int c = 0; c < clustersPerBlock; c++) {
						float cimg = 0;
						#pragma unroll
						for (int cc = 0; cc < sizeClustIn; cc++) {
							cimg += shImages[threadIdx.y + (cc + c * sizeClustIn) * B_Y][threadIdx.x + i * B_X] * shC[cc + c * sizeClustIn][r]; 
						}
						shImages[threadIdx.y + c * clustersPerBlock][threadIdx.x + i * B_X] = cimg * shXY[threadIdx.y + c * B_Y][r];
					}
					if (threadIdx.y < clustersPerBlock) {
						for (int p = 0; p < B_Y; p++) {	
							//shImages[threadIdx.y + r * clustersPerBlock][threadIdx.x + i * B_X] += shImages[p + threadIdx.y * clustersPerBlock][threadIdx.x + i * B_X]; 
						}	
					}
					/*
					for (int f = 0; f < filtersPerThread; f++) {
						prod[f][i] += shImages[threadIdx.y ][threadIdx.x + i * B_X] * shF[threadIdx.y + f * B_Y][r];
					}
					*/	
				}	
			}
			__syncthreads();
		}
		__syncthreads();
		for (int i = 0; i < imgsPerThread; i++) {
			for (int f = 0; f < filtersPerThread; f++) {
				for (int r = 0; r < rank; r++) {
					prod[f][i] += F[threadIdx.y + f * B_Y + r * sizeClustOut];// * shOut[0][threadIdx.x + i * B_X];// * shF[threadIdx.y + f * B_Y][r];
				}
			}	
		}
	}	

    #pragma unroll
    for (int g = 0; g < imgsPerThread; g++) {
    	if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
    		#pragma unroll
    		for (int f = 0; f < filtersPerThread; f++) {
    			targets[g * B_X + f * B_Y * numImages * numModules] = prod[f][g];
    		}
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
	
    int filterPixels = XY.getNumRows() / (#numClustIn * #rank); 
    int filterSize = int(sqrt(filterPixels));
    assert_(filterSize * filterSize == filterPixels);

    assert_(F.getNumRows() == #numClustIn * #rank * #sizeClustOut);
    assert_(C.getNumRows() == #numClustIn * #rank * #sizeClustIn);
    assert_(XY.getNumRows() == #numClustIn * #rank * filterPixels);

	assert_(F.getNumCols() == #numClustOut);
	assert_(C.getNumCols() == #numClustOut);
	assert_(XY.getNumCols() == #numClustOut);

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

