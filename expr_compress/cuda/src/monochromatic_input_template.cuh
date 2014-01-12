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


void convFilterActsMono(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, NVMatrix& perm);

void convFilterActsMono(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors,
                   float scaleTargets, float scaleOutput, NVMatrix& perm);

#endif  /* COMMON_CUH */

void monochromatic_input(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	assert_((nrhs == 10) && (nlhs == 0));
	NVMatrix* images = getMatrix(prhs[1]);
	NVMatrix* filters = getMatrix(prhs[2]); 
	NVMatrix* targets = getMatrix(prhs[3]); 

	const mwSize* images_dims = mxGetDimensions(prhs[1]);
	int imgSizeY = (int)mxGetScalar(prhs[4]);
	int paddingStart = (int)mxGetScalar(prhs[8]);

	int filterSize = (int)mxGetScalar(prhs[6]);
	int moduleStride = (int)mxGetScalar(prhs[7]);
	int numModulesY = 1 + int(ceil((2 * paddingStart + imgSizeY - filterSize) / float(moduleStride)));
	int numModulesX = numModulesY;
	int numImgColors = (int)mxGetScalar(prhs[5]);

	NVMatrix* perm = getMatrix(prhs[9]);

	images->transpose();
	filters->transpose();
	targets->transpose();
	convFilterActsMono(*images, *filters, *targets,
                       imgSizeY, numModulesY, numModulesX, -paddingStart, moduleStride,
                       numImgColors, 0, 1, *perm);
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
template <int B_Y, int B_X, int imgsPerThread, int filtersPerThread, int colorsPerBlock,
          bool scale, bool checkImgBounds>
__global__ void filterActsMonoEven_YxX_color(float* images, float* filters, float* targets, float* perm,
                                   const int numImages, const int numFilters, const int numColors,
                                   const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int imgStride,
                                   const float scaleTargets, const float scaleOutputs,
                                   const bool conv) {
    __shared__ float shFilters[B_Y][B_Y * filtersPerThread]; // pre-load B_Y pixels from B_Y*filtersPerThread filters
    __shared__ float shImages[B_Y*colorsPerBlock][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;

    const int blocksPerModule = numFilters / (B_Y*filtersPerThread);
    const int moduleIdx = blockIdx.y / blocksPerModule;
    //const int blockFilterIdx = blockIdx.y % blocksPerModule;
    const int globalFilterIdx = (blockIdx.y % blocksPerModule) * B_Y * filtersPerThread;
    const int colorStartIdx = globalFilterIdx * numColors / numFilters; //globalFilterIdx / filtersPerColor
    //const int colorStopIdx = colorStartIdx + numFilters / numColors;
    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;

    const int shFilterLoadY = tidx / (B_Y * filtersPerThread);
    const int shFilterLoadX = tidx % (B_Y * filtersPerThread);
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += globalFilterIdx
             + shFilterLoadY * numFilters + shFilterLoadX;
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (globalFilterIdx + threadIdx.y) * numImages * numModulesY * numModulesX
            + myImgIdx;


    float prod[filtersPerThread][imgsPerThread];
    #pragma unroll
    for(int f = 0; f < filtersPerThread; f++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[f][g] = 0;
        }
    }

    if (colorsPerBlock == 1) {
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
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
                            shImages[threadIdx.y][threadIdx.x + i * B_X] = images[imgStride * ( colorStartIdx  * imgPixels + y * imgSizeX + x) + i * B_X];
                        } else {
                            shImages[threadIdx.y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        shImages[threadIdx.y][threadIdx.x + i * B_X] = 0;
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                #pragma unroll
                for (int i = 0; i < B_Y; i++) {
                    #pragma unroll
                    for(int g = 0; g < imgsPerThread; g++) {
                        prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                    }                
                 }
            }
            __syncthreads();
        }
    } else { //colorsPerBlock == 2
        for (int p = 0; p < filterPixels; p += B_Y) {
            /*
             * Load B_Y pixels from B_Y*filtersPerThread filters
             */
            if (shFilterLoadY < B_Y) {
                #pragma unroll
                for (int p2 = 0; p2 < B_Y; p2 += B_X/filtersPerThread) {
                    if (p + p2 + shFilterLoadY < filterPixels) {
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
                            for (int c = 0; c < colorsPerBlock; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = images[imgStride * ( (colorStartIdx + c) * imgPixels + y * imgSizeX + x) + i * B_X];
                            }
                        } else {
                            #pragma unroll
                            for (int c = 0; c < colorsPerBlock; c++) {
                                shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                            }
                        }
                    }
                } else { // Padding
                    #pragma unroll
                    for (int i = 0; i < imgsPerThread; i++) {
                        #pragma unroll
                        for (int c = 0; c < colorsPerBlock; c++) {
                            shImages[threadIdx.y + c * B_Y][threadIdx.x + i * B_X] = 0;
                        }
                    }
                }
            }
            __syncthreads();
            #pragma unroll
            for(int f = 0; f < filtersPerThread; f++) {
                if (colorStartIdx + threadIdx.y + f * B_Y < colorStartIdx + numFilters / numColors) { 
                    #pragma unroll
                    for (int i = 0; i < B_Y*1; i++) {
                        #pragma unroll
                        for(int g = 0; g < imgsPerThread; g++) {
                            prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y + f * B_Y];
                        }                
                     }
                } else {
                    #pragma unroll
                    for (int i = B_Y; i < B_Y*2; i++) {
                        #pragma unroll
                        for(int g = 0; g < imgsPerThread; g++) {
                            prod[f][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i - B_Y][threadIdx.y + f * B_Y];
                        }                
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
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleTargets * targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] + scaleOutputs * prod[f][g];
                }
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                #pragma unroll
                for (int f = 0; f < filtersPerThread; f++) {
                    targets[g * B_X + f * B_Y * numImages * numModulesY * numModulesX] = scaleOutputs * prod[f][g];
                }
            }
        }
    }
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
template <int B_Y, int B_X, int imgsPerThread,
          bool scale, bool checkImgBounds>
__global__ void filterActsMonoEvenManyCol_YxX_color(float* images, float* filters, float* targets, float* perm,
                                   const int numImages, const int numFilters, const int numColors,
                                   const int imgSizeY, const int imgSizeX, const int filterSize, const int paddingStart,
                                   const int moduleStride,
                                   const int numModulesY, const int numModulesX, const int numKernelModulesX, const int imgStride,
                                   const float scaleTargets, const float scaleOutputs,
                                   const bool conv) {
    __shared__ float shFilters[B_Y][B_Y*4]; // pre-load B_Y pixels from B_Y*4 filters (4 modules per block)
    __shared__ float shImages[B_Y][B_X * imgsPerThread]; // pre-load B_Y pixels from B_X*imgsPerThread images
    const int imgPixels = imgSizeY * imgSizeX;
    const int filterPixels = filterSize * filterSize;
    const int blocksPerModule = numFilters / B_Y;
    const int kernelModuleIdx = blockIdx.y / blocksPerModule;
    const int moduleIdx = 2 * numModulesX * (kernelModuleIdx / numKernelModulesX) + 2 * (kernelModuleIdx % numKernelModulesX); //upper left index
    //const int blockFilterIdx = blockIdx.y % blocksPerModule;
    const int globalFilterIdx = (blockIdx.y % blocksPerModule) * B_Y;
    const int colorStartIdx = globalFilterIdx * numColors / numFilters; //globalFilterIdx / filtersPerColor
    //const int colorStopIdx = colorStartIdx + numFilters / numColors;
    const int tidx = threadIdx.y * B_X + threadIdx.x;

    const int imgLoadModPosY = (moduleIdx / numModulesX) * moduleStride;
    const int imgLoadModPosX = (moduleIdx % numModulesX) * moduleStride;
    
    const int imgPatchSizeX = (moduleIdx + 1) % numModulesX == 0 ? filterSize : filterSize + moduleStride;
    const int imgPatchSizeY = moduleIdx >= numModulesX * (numModulesY - 1) ? filterSize : filterSize + moduleStride;
    const int imgPatchPixels = imgPatchSizeX * imgPatchSizeY;

    const int shFilterLoadY = tidx / (B_Y*4) ;
    const int shFilterLoadX = tidx % (B_Y*4);

    const int thModuleIdx = shFilterLoadX % 4;
    
    const int myImgIdx = blockIdx.x * B_X * imgsPerThread + threadIdx.x;
    images += myImgIdx;
    filters += globalFilterIdx
             + (shFilterLoadX / 4);//+ shFilterLoadY * numFilters + (shFilterLoadX / 4);
    if (!conv) {
        filters += moduleIdx * numColors * filterPixels * numFilters;
    }

    targets += moduleIdx * numImages
            + (globalFilterIdx + threadIdx.y) * numImages * numModulesY * numModulesX
            + myImgIdx;


    float prod[4][imgsPerThread]; // starts, by row, from moduleIdx
    #pragma unroll
    for(int m = 0; m < 4; m++) {
        #pragma unroll
        for(int g = 0; g < imgsPerThread; g++) {
            prod[m][g] = 0;
        }
    }

    int thp; //where are we in respective modules
    for (int p = 0; p < imgPatchPixels; p += B_Y) {
        /*
         * Load B_Y pixels from B_Y*4 filters
         */
        if (shFilterLoadY < B_Y) {
            #pragma unroll
            for (int p2 = 0; p2 < B_Y; p2 += B_X/4) {
                int x = (shFilterLoadY + p + p2) % imgPatchSizeX;
                int y = (shFilterLoadY + p + p2) / imgPatchSizeX;

                if (thModuleIdx == 0) { //upper left
                    thp = (x < filterSize && y < filterSize) ? y*filterSize + x : -1;
                } else if (thModuleIdx == 1) { //upper right
                    thp = (x >= moduleStride && y < filterSize) ? y*filterSize + (x - moduleStride) : -1;
                } else if (thModuleIdx == 2) { //lower left
                    thp = (y >= moduleStride && x < filterSize) ? (y - moduleStride)*filterSize + x: -1;
                } else { //lower right
                    thp = (x >= moduleStride && y >= moduleStride) ? (y - moduleStride)*filterSize + (x - moduleStride) : -1;
                }

                if (thp < filterPixels && thp >= 0) {
                    shFilters[shFilterLoadY + p2][shFilterLoadX] = filters[(thp) * numFilters];
                } else {
                    shFilters[shFilterLoadY + p2][shFilterLoadX] = 0;
                }
            }
        }

        /*
         * Load B_Y pixels from B_X*imgsPerThread images
         */
        const int pixIdx = p + threadIdx.y;
        if (pixIdx < imgPatchPixels) {
            int x = paddingStart + imgLoadModPosX + pixIdx % imgPatchSizeX;
            int y = paddingStart + imgLoadModPosY + pixIdx / imgPatchSizeX;
            if (y >= 0 && y < imgSizeY && x >= 0 && x < imgSizeX) {
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    if (!checkImgBounds || myImgIdx + i * B_X < numImages) {
                        shImages[threadIdx.y][threadIdx.x + i * B_X] = images[imgStride * ( colorStartIdx  * imgPixels + y * imgSizeX + x) + i * B_X];
                    } else {
                        shImages[threadIdx.y][threadIdx.x + i * B_X] = 0;
                    }
                }
            } else { // Padding
                #pragma unroll
                for (int i = 0; i < imgsPerThread; i++) {
                    shImages[threadIdx.y][threadIdx.x + i * B_X] = 0;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for(int m = 0; m < 4; m++) {
            #pragma unroll
            for (int i = 0; i < B_Y; i++) {
                #pragma unroll
                for(int g = 0; g < imgsPerThread; g++) {
                    prod[m][g] += shImages[i][g * B_X + threadIdx.x] * shFilters[i][threadIdx.y*4 + m];
                }               
            }
        }
        __syncthreads();
    }
  
    if (scale) {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                targets[g * B_X + 0 * numImages] = scaleTargets * targets[g * B_X + 0 * numImages] + scaleOutputs * prod[0][g];
                if (imgPatchSizeX > filterSize) targets[g * B_X + 1 * numImages] = scaleTargets * targets[g * B_X + 1 * numImages] + scaleOutputs * prod[1][g];
                if (imgPatchSizeY > filterSize) targets[g * B_X + numModulesX * numImages] = scaleTargets * targets[g * B_X + numModulesX * numImages] + scaleOutputs * prod[2][g];
                if (imgPatchSizeX > filterSize && imgPatchSizeY > filterSize) targets[g * B_X + (numModulesX + 1) * numImages] = scaleTargets * targets[g * B_X + (numModulesX + 1) * numImages] + scaleOutputs * prod[3][g];
            }
        }
    } else {
        #pragma unroll
        for (int g = 0; g < imgsPerThread; g++) {
            if (!checkImgBounds || myImgIdx + g * B_X < numImages) {
                targets[g * B_X + 0 * numImages] = scaleOutputs * prod[0][g];
                if (imgPatchSizeX > filterSize) targets[g * B_X + 1 * numImages] = scaleOutputs * prod[1][g];
                if (imgPatchSizeY > filterSize) targets[g * B_X + numModulesX * numImages] = scaleOutputs * prod[2][g];
                if (imgPatchSizeX > filterSize && imgPatchSizeY > filterSize) targets[g * B_X + (numModulesX + 1) * numImages] = scaleOutputs * prod[3][g];                
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
 void _filterActsMono(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors,
                   float scaleTargets, float scaleOutput, bool conv, NVMatrix& perm) {    
    int numFilters = filters.getNumCols();
    int numModules = numModulesY * numModulesX;
    int numImages = images.getNumCols();
    int imgPixels = images.getNumRows()/numImgColors;
    int imgSizeX = imgPixels / imgSizeY;
    int filterModuleMult = conv ? 1 : numModules;
    assert_(numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0));
    assert_(numFilters % 16 == 0);
    assert_(images.getNumRows() == imgPixels * numImgColors);
    assert_(imgSizeY * imgSizeX == imgPixels);

    int imgStride = images.getStride(); // images does not need to be a contiguous matrix

    int filterPixels = filters.getNumRows() / filterModuleMult;
    int filterSize = int(sqrt(filterPixels));
    assert_(filterSize * filterSize == filterPixels);
    assert_(filters.getNumRows() == filterModuleMult * filterPixels);

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

    //int filtersPerColor = numFilters / numImgColors; // must be 32, 24, 16, 12, 8 or 6
    //int filtersPerThread = filtersPerColor % 32 == 0 ? 8 : filtersPerColor % 24 == 0 ? 6 : filtersPerColor % 16 == 0 ? 4 : filtersPerColor % 12 == 0 ? 3 : filtersPerColor % 8 == 0 ? 2 : filtersPerColor % 6 == 0 ? 2 : 1;
    //int B_Y = filtersPerColor == 3 || filtersPerColor == 6 ? 3 : 4;

    //int colorsPerBlock = filtersPerThread * B_Y / filtersPerColor;
    //int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
    //if (filtersPerColor < 24) imgsPerThread *= 2;


    if (numImgColors <= 4) {
        dim3 blocks = dim3(DIVUP(numImages, #B_X * #imgsPerThread), (numModules * numFilters) / (#B_Y * #filtersPerThread));
        dim3 threads(#B_X, #B_Y); // B_Y always 4
        //bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
        if (scaleTargets == 0) {
            targets.resize(numFilters * numModules, numImages);
        } else {
            assert_(targets.getNumRows() == numFilters * numModules);
            assert_(targets.getNumCols() == numImages);
        }

       // printf("(G_X, G_Y) = (%d, %d) \n", DIVUP(numImages, #B_X * #imgsPerThread), (numModules * numFilters) / (#B_Y * #filtersPerThread));
        filterActsMonoEven_YxX_color < #B_Y, #B_X, #imgsPerThread, #filtersPerThread, #colorsPerBlock, #scale, #checkImgBounds > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), perm.getDevData(),
            numImages, numFilters, numImgColors, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, imgStride, scaleTargets, scaleOutput, conv);
    
    } else {
        int numKernelModulesX = DIVUP(numModulesX, 2); // kernelModulesX == kernelModulesY
        dim3 blocks = dim3(DIVUP(numImages, #B_X * #imgsPerThread), (numKernelModulesX*numKernelModulesX * numFilters) / #B_Y);
        dim3 threads(#B_X, #B_Y); // B_Y always 4
        //bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
        if (scaleTargets == 0) {
            targets.resize(numFilters * numModules, numImages);
        } else {
            assert_(targets.getNumRows() == numFilters * numModules);
            assert_(targets.getNumCols() == numImages);
        }
        //printf("(G_X, G_Y) = (%d, %d) \n", DIVUP(numImages, #B_X * #imgsPerThread), (numKernelModulesX*numKernelModulesX * numFilters) / #B_Y);
        filterActsMonoEvenManyCol_YxX_color < #B_Y, #B_X, #imgsPerThread, #scale, #checkImgBounds > <<<blocks, threads>>>(images.getDevData(), filters.getDevData(), targets.getDevData(), perm.getDevData(),
            numImages, numFilters, numImgColors, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY, numModulesX, numKernelModulesX, imgStride, scaleTargets, scaleOutput, conv);
    
    }            
    
    getLastCudaError("filterActs: kernel execution failed");
}

void convFilterActsMono(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors,  NVMatrix& perm) {
    convFilterActsMono(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, 0, 1, perm);
}

void convFilterActsMono(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, 
                   float scaleTargets, float scaleOutput, NVMatrix& perm) {
     _filterActsMono(images, filters, targets, imgSizeY, numModulesY, numModulesX, paddingStart, moduleStride, numImgColors, scaleTargets, scaleOutput, true, perm);
}
