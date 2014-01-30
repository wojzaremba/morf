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


void convFilterActsMonoLowrank(NVMatrix& images, NVMatrix& targets, NVMatrix& colors, NVMatrix& filter, NVMatrix& color_coeff, NVMatrix& perm, int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride, int newNumColors, float scaleTargets, float scaleOutput);

#endif  /* COMMON_CUH */

void monolowrank_input(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
        assert_((nrhs == 12) && (nlhs == 0));
        NVMatrix* images = getMatrix(prhs[1]);
        NVMatrix* targets = getMatrix(prhs[2]); 
		NVMatrix* colors = getMatrix(prhs[3]); 
		NVMatrix* filters = getMatrix(prhs[4]); 
		NVMatrix* filter_coeff = getMatrix(prhs[5]); 
		NVMatrix* perm = getMatrix(prhs[6]); 
        
        const mwSize* images_dims = mxGetDimensions(prhs[1]);
        int imgSizeY = (int)mxGetScalar(prhs[7]);
        int paddingStart = (int)mxGetScalar(prhs[11]);

        int filterSize = (int)mxGetScalar(prhs[9]);
        int moduleStride = (int)mxGetScalar(prhs[10]);
        int numModulesY = 1 + int(ceil((2 * paddingStart + imgSizeY - filterSize) / float(moduleStride)));
        int numModulesX = numModulesY;
        int numImgColors = (int)mxGetScalar(prhs[8]);

        

        images->transpose();
        targets->transpose();
		colors->transpose();
		filters->transpose();
		filter_coeff->transpose();	

		convFilterActsMonoLowrank(*images, *targets, *colors, *filters, *filter_coeff, *perm, 
        	               			imgSizeY, numModulesY, numModulesX, -paddingStart, moduleStride,
                      	 			numImgColors, 0, 1);
        images->transpose();
        targets->transpose();
		colors->transpose();
		filters->transpose();
		filter_coeff->transpose();	
}
/*
 * Block size B_YxB_X. Each block applies B_Y * filtersPerThread filters to B_X * imgsPerThread images.
 * threadIdx.x determines image
 * threadIdx.y determines filter
 *
 * blockIdx.x determines image batch of B_X * imgsPerThread
 * blockIdx.y determines filter batch of module and B_Y * filtersPerThread
 *
 * images:      (numColors, imgSizeY, imgSizeX, numImages) with stride given
 * filters:     (numColors, filterPixels, numFilters) if conv
 *              (numModules, numColors, filterPixels, numFilters) otherwise
 *
 * targets:     (numFilters, numModulesY, numModulesX, numImages)
 *
 * B_Y one of 4, 8, 16
 * B_X one of 16, 32
 * imgsPerThread one of 1, 2, 4
 * filtersPerThread one of 1, 2, 4, 8
 *
 * Number of filters per module should be divisible by B_Y * filtersPerThread
 * checkImgBounds indicates whether number of images is divisible by B_X * imgsPerThread
 *
 * The imgSize here is the size of the actual image without the padding.
 *
 */
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int numColors, int numClusters, int clusterSize, int rank,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_color(float* images, float* targets, float* colors, float* filters, float* filter_coeff, float* perm,
                                   const int numImages, const int numFilters,
                                   const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int imgStride,
                                   const float scaleTargets, const float scaleOutputs,
                                   const bool conv) {
    __shared__ float shFilters[B_Y][rank]; // pre-load B_Y pixels from rank filters
    __shared__ float shImages[B_Y * numColors][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    __shared__ float shColors[numColors][B_Y * filtersPerThread]; // Color transform coeffiecients 
	__shared__ float shFilterCoeff[clusterSize][rank]; // Filter transform coefficients

	const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int globalFilterIdx = (blockIdx.y % blocksPerModule) * B_Y * filtersPerThread;
	const int clusterIdx = globalFilterIdx / clusterSize;
    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

	const int shFilterCoeffLoadY = tidx / rank;
	const int shFilterCoeffLoadX = tidx % rank;

    const int shFilterLoadY = tidx / rank;
    const int shFilterLoadX = tidx % rank;
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += myImgIdx;
	filters += (int)perm[globalFilterIdx + shFilterLoadX] + shFilterLoadY * numFilters;
	filter_coeff += clusterIdx * rank * clusterSize + shFilterCoeffLoadX * clusterSize + shFilterCoeffLoadY;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages + myImgIdx;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

	/*
    * Fill shColorCoeff with the color coefficients
    */
    if (tidx < B_Y * filtersPerThread * numColors ) {
            const int shColorLoadY = tidx / (B_Y * filtersPerThread);
            const int shColorLoadX = tidx % (B_Y * filtersPerThread);
			const int filterIdx = (int)perm[globalFilterIdx + shColorLoadX];	
            shColors[shColorLoadY][shColorLoadX] = colors[filterIdx + shColorLoadY * numFilters];
    }
	if (shFilterCoeffLoadY < clusterSize) {
		shFilterCoeff[shFilterCoeffLoadY][shFilterCoeffLoadX] = *filter_coeff;
	}
    __syncthreads();
    
    for (int p = 0; p < filterPixels; p += B_Y) {
        /*
         * Load B_Y pixels from B_Y*filtersPerThread filters
         */
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/rank) {
                if (p + p2 + shFilterLoadY < filterPixels && p2 + shFilterLoadY < B_Y) {
                    shFilters[shFilterLoadY + p2][shFilterLoadX] = filters[(p + p2) * numFilters];
                } else {
                    shFilters[shFilterLoadY + p2][shFilterLoadX] = 0;
                }
            }
        }

        /*
         * Load B_Y pixels from B_X*imgsPerThread images
         */
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < filterPixels) {
            const int x = paddingStart + imgLoadModPosX + pixIdx % filterSize;
            const int y = paddingStart + imgLoadModPosY + pixIdx / filterSize;
            if (y >= 0 && y< imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * (c * imgPixels + y * imgSizeX + x) + i * B_X];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < numColors; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    #pragma unroll
                    for (int c = 0; c < numColors; c++) {
                        shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
        }
        __syncthreads();
        #pragma unroll
       for (int i = 0; i < B_Y; i++) {
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
				// Construct weight
				float weight = 0;
				#pragma unroll
				for (int r = 0; r < rank; r++) {
					weight += shFilters[i][r] * shFilterCoeff[threadIdx.y + f * B_Y][r];
				}
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
					float img = 0;
					#pragma unroll
					for (int c = 0; c < numColors; c++) {
						img += shImages[threadIdx.y + c * B_Y][g * B_X + threadIdx.x] * shColors[c][threadIdx.y + f * B_Y];
					}
					prod[f][g] += img * weight; 
                }
            }

       }
        __syncthreads();
    }
    
    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                        int targetIdx =  (int) perm[globalFilterIdx + threadIdx.y + f * B_Y];
                        targets[g * B_X + targetIdx * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + targetIdx * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    int targetIdx =  (int) perm[globalFilterIdx + threadIdx.y + f * B_Y];
                    targets[g * B_X + targetIdx * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                }
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
 void _filterActs(NVMatrix& images, NVMatrix& targets, NVMatrix& colors, NVMatrix& filters, NVMatrix& filter_coeff, NVMatrix& perm,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput, bool conv) {
    int numFilterColors = numImgColors / numGroups;      
    int numFilters = colors.getNumCols();
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
	
	printf("numFilters = %d \n", numFilters);

    assert_(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
    assert_(numGroups == 1 || numFilterColors % 2 == 0);
    assert_(numFilters % (16 * numGroups) == 0);
    assert_(numImgColors % numGroups == 0);
    assert_(images.getNumRows() == imgPixels * numImgColors);
    assert_(imgSizeY * imgSizeX == imgPixels);
	assert(filter_coeff.getNumCols() == #clusterSize);
	assert(filter_coeff.getNumRows() == #rank * #numClusters);
	assert(numFilters == #numClusters * #clusterSize);
    int imgStride = images.getStride(); // images does not need to be a contiguous matrix
    int filterPixels = filters.getNumCols();
    int filterSize = int(sqrt(filterPixels));

	printf("filterPixels = %d\n", filterPixels);

    assert_(filterSize * filterSize == filterPixels);
    assert_(filters.getNumRows() == #rank * #numClusters);

    // These routines don't handle the case when only part of the image is visited in the convolution
    assert_(paddingStart <= 0);
    assert_(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
    assert_(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
    assert_(moduleStride <= filterSize);
    
    assert_(!images.isTrans());
    assert_(!filters.isTrans());
    assert_(!targets.isTrans());

    assert_(filters.isContiguous());
    assert_(targets.isContiguous());

    dim3 blocks = dim3(DIVUP(numImages, #B_X * #imgsPerThread), (numModules * numFilters) / (#B_Y * #filtersPerThread));
    dim3 threads(#B_X, #B_Y);
    printf("(B_X, B_Y) = (%d, %d)\n(G_X, G_Y) = (%d, %d)\n", #B_X, #B_Y, DIVUP(numImages, #B_X * #imgsPerThread), (numModules * numFilters) / (#B_Y * #filtersPerThread));
    cudaFuncSetCacheConfig(filterActs_YxX_color< #B_Y, #B_X, #imgsPerThread, #filtersPerThread, #numImgColors, #numClusters, #clusterSize, #rank, #scale, #checkImgBounds >, cudaFuncCachePreferShared);
    filterActs_YxX_color <  #B_Y, #B_X, #imgsPerThread, #filtersPerThread, #numImgColors, #numClusters, #clusterSize, #rank, #scale, #checkImgBounds> <<<blocks, threads>>>(images.getDevData(), targets.getDevData(), colors.getDevData(), filters.getDevData(), filter_coeff.getDevData(), perm.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);

}

void convFilterActsMonoLowrank(NVMatrix& images, NVMatrix& targets, NVMatrix& colors, NVMatrix& filters, NVMatrix& filter_coeff, NVMatrix& perm, int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride, int numImgColors, float scaleTargets, float scaleOutput) {
     _filterActs(images, targets, colors, filters, filter_coeff, perm, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, 1, scaleTargets, scaleOutput, true);
}
