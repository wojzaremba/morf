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


void  convFilterActsClustered(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, NVMatrix& perm, NVMatrix& colors,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int newNumColors,
                   float scaleTargets, float scaleOutput);

#endif  /* COMMON_CUH */
/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 */

#include "nvmatrix.cuh"
#include "cudaconv2.cuh"


void biclustered_hidden(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
        assert_((nrhs == 11) && (nlhs == 0));
        NVMatrix* images = getMatrix(prhs[1]);
        NVMatrix* filters = getMatrix(prhs[2]); 
        NVMatrix* targets = getMatrix(prhs[3]); 
        NVMatrix* perm = getMatrix(prhs[9]);
        NVMatrix* colors = getMatrix(prhs[10]);
        
        const mwSize* images_dims = mxGetDimensions(prhs[1]);
        int imgSizeY = (int)mxGetScalar(prhs[4]);
        int paddingStart = (int)mxGetScalar(prhs[8]);

        int filterSize = (int)mxGetScalar(prhs[6]);
        int moduleStride = (int)mxGetScalar(prhs[7]);
        int numModulesY = 1 + int(ceil((2 * paddingStart + imgSizeY - filterSize) / float(moduleStride)));
        int numModulesX = numModulesY;
        int newNumColors = (int)mxGetScalar(prhs[5]);

        

        images->transpose();
        filters->transpose();
        targets->transpose();
        convFilterActsClustered(*images, *filters, *targets, *perm, *colors,
                       imgSizeY, numModulesY, numModulesX, -paddingStart, moduleStride,
                       newNumColors, 0, 1);
        images->transpose();
        filters->transpose();
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
 * filters:     (numFilterColors, filterPixels, numFilters) if conv
 *              (numModules, numFilterColors, filterPixels, numFilters) otherwise
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
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorCache,
          bool scale, bool checkImgBounds>
__global__ void filterActs_YxX_sparse(float* images, float* filters, float* targets,
                                       const int numImages, const int numFilters,
                                       const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                       const int moduleStride,
                                       const int numModulesY, const int numModulesX, const int imgStride, const int numImgColors,
                                       const int numGroups, 
                                       const float scaleTargets, const float scaleOutputs,
                                       const bool conv) {
    __shared__ float shFilters[B_Y*colorCache][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorCache][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int numFilterColors = numImgColors / numGroups;
    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    const int blockFilterIdx = filtersPerThread * B_Y * (blockIdx.y % blocksPerModule);
    const int numFiltersPerGroup = numFilters / numGroups;
    const int blockGroupIdx = blockFilterIdx / numFiltersPerGroup;

    const int numModules = numModulesX * numModulesY;
    const int blockColorIdx = numFilterColors * blockGroupIdx;

    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = paddingStart + (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = paddingStart + (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;

    images += blockColorIdx * imgPixels * imgStride + myImgIdx;
    filters +=blockFilterIdx
            + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numFilterColors * filterPixels * numFilters;
    }

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
//    __shared__ int imgPos[]
    for (int oc = 0; oc < numFilterColors; oc += colorCache) { // oc stands for outer color (loop)
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = filters[((oc+c) * filterPixels + p + p2) * numFilters];
                        }
                    } else {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shFilters[shFilterLoadY + p2 + c * B_Y][shFilterLoadX] = 0;
                        }
                    }
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
                    float* m = &images[imgStride * (oc * imgPixels + y * imgSizeX + x)];
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = m[c * imgStride * imgPixels + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorCache; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorCache; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < B_Y*colorCache; i++) {
                #pragma unroll
                for(int f = 0; f < filtersPerThread; f++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }
                }

            }
            __syncthreads();
        }
    }

    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModules] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModules] = scaleOutputs * prod[f][g];
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
 void _filterActsClustered(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, 
                   float scaleTargets, float scaleOutput, bool conv) {
    int numFilters = filters.getNumCols();
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;
    assert_(numImgColors > 3);
    assert_(numFilters % 16== 0);
    assert_(images.getNumRows() == imgPixels * numImgColors);
    assert_(imgSizeY * imgSizeX == imgPixels);

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters.getNumRows() / (filterModuleMult * numImgColors);
    int filterSize = int(sqrt(filterPixels));
    assert_(filterSize * filterSize == filterPixels);
    assert_(filters.getNumRows() == filterModuleMult * numImgColors * filterPixels);

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
    if (scaleTargets == 0) {
        targets.resize(numFilters * numModules, numImages);
    } else {
        assert_(targets.getNumRows() == numFilters * numModules);
        assert_(targets.getNumCols() == numImages);
    }
  
	cudaFuncSetCacheConfig(filterActs_YxX_sparse< #B_Y, #B_X, #imgsPerThread, #filtersPerThread, #colorCache, #scale, #checkImgBounds >, cudaFuncCachePreferShared);
    filterActs_YxX_sparse < #B_Y, #B_X, #imgsPerThread, #filtersPerThread, #colorCache, #scale, #checkImgBounds> <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, numImgColors, 1, scaleTargets, scaleOutput, conv);
    
	getLastCudaError("filterActs: kernel execution failed");
}


void convFilterActsClustered(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,  NVMatrix& perm, NVMatrix& colors,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, 
                   float scaleTargets, float scaleOutput) {
     _filterActsClustered(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, scaleTargets, scaleOutput, true);
}

