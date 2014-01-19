#ifndef GPU_HEADERS_H
#define GPU_HEADERS_H


#include <curand_kernel.h>
#include "nvmatrix_operations.cuh"

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                            mexPrintf("Error at %s:%d, x = %d\n",__FILE__,__LINE__, x);\
                            exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            mexPrintf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)

static cudaEvent_t start_event, end_event;

__global__ void kTile(const float* src, float* tgt, const unsigned int srcWidth, const unsigned int srcHeight, const unsigned int tgtWidth, const unsigned int tgtHeight);
__global__ void kDotProduct_r(float* a, float* b, float* target, const unsigned int numCols, const unsigned int numElements);
__global__ void kSetupCurand(curandState *state, unsigned long long seed);

inline void __getLastCudaError(const char *errorMsg, const char *file, const int line) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
                mexPrintf("%s ( %s ) : get:LastCudaError() CUDA error : %s : ( %d ) %s\n", file, line, errorMsg, (int)err, cudaGetErrorString(err));
        	mexErrMsgTxt("!!!! error\n");
        }
}

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};

void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX);
void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize);

void convLocalAvgUndo(NVMatrix& avgGrads, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, int imgSize,
                      float scaleTargets, float scaleOutput);
void convLocalMaxUndo(NVMatrix& images, NVMatrix& maxGrads, NVMatrix& maxActs, NVMatrix& target,
                      int subsX, int startX, int strideX, int outputsX, float scaleTargets, float scaleOutput);

void convResponseNorm(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
void convResponseNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters,
                         int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);
void convContrastNorm(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale);
void convContrastNormUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& meanDiffs, NVMatrix& acts, NVMatrix& target, int numFilters, int sizeX, float addScale, float powScale, float scaleTargets, float scaleOutput);

void convGaussianBlur(NVMatrix& images, NVMatrix& filter, NVMatrix& target, bool horiz, int numChannels,
                      float scaleTargets, float scaleOutputs);
void convBedOfNails(NVMatrix& images, NVMatrix& target, int numChannels, int imgSize, int startX,
                    int strideX, float scaleTargets, float scaleOutput);
void convBedOfNailsUndo(NVMatrix& actsGrad, NVMatrix& target, int numChannels, int imgSize,
                        int startX, int strideX, float scaleTargets, float scaleOutput);

void convResizeBilinear(NVMatrix& images, NVMatrix& target, int imgSize, int tgtSize, float scale);
void convRGBToYUV(NVMatrix& images, NVMatrix& target);
void convRGBToLAB(NVMatrix& images, NVMatrix& target, bool center);
void convCrop(NVMatrix& imgs, NVMatrix& target, int imgSize, int tgtSize, int startY, int startX);
void normalizeLocalWeights(NVMatrix& weights, int numModules, float norm);
void convTICAGrad(NVMatrix& images, NVMatrix& ticas, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convTICA(NVMatrix& images, NVMatrix& target, int numFilters, int sizeX, float scaleTarget, float scaleOutput);
void convContrastNormCrossMap(NVMatrix& images, NVMatrix& meanDiffs, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, float k, float addScale, float powScale, bool blocked);
void convResponseNormCrossMapUndo(NVMatrix& outGrads, NVMatrix& denoms, NVMatrix& inputs, NVMatrix& acts, NVMatrix& target, int numFilters, int sizeF, float addScale, float powScale, bool blocked, float scaleTargets, float scaleOutput);
void convResponseNormCrossMap(NVMatrix& images, NVMatrix& denoms, NVMatrix& target, int numFilters, int sizeF, float k, float addScale, float powScale, bool blocked);

void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups);
void convFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                   int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                   int numImgColors, int numGroups,
                   float scaleTargets, float scaleOutput);

void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups);
void localFilterActs(NVMatrix& images, NVMatrix& filters, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups,
                     float scaleTargets, float scaleOutput);

void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void convImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                 int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                 float scaleTargets, float scaleOutput);

void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups);
void localImgActs(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets,
                  int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numGroups,
                  float scaleTargets, float scaleOutput);

void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                    int moduleStride, int numImgColors, int numGroups, int partialSum);
void convWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                    int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                    int numImgColors, int numGroups, int partialSum,
                    float scaleTargets, float scaleOutput);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart,
                     int moduleStride, int numImgColors, int numGroups);

void localWeightActs(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets,
                     int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                     int numImgColors, int numGroups, float scaleTargets, float scaleOutput);

void convFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups);
void convFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups,
                          float scaleTargets, float scaleOutput);

void localFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups,
                          float scaleTargets, float scaleOutput);
void localFilterActsSparse(NVMatrix& images, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                          int imgSizeY, int numModulesY, int numModulesX, int paddingStart, int moduleStride,
                          int numImgColors, int numFilterColors, int numGroups);

void convWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                         int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                         int numImgColors, int numFilterColors, int numGroups);
void convWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numFilterColors,
                        int numGroups, int partialSum, float scaleTargets, float scaleOutput);

void localWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                         int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride,
                         int numImgColors, int numFilterColors, int numGroups);
void localWeightActsSparse(NVMatrix& images, NVMatrix& hidActs, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int numModulesY, int numModulesX, int filterSize, int paddingStart, int moduleStride, int numImgColors, int numFilterColors,
                        int numGroups, float scaleTargets, float scaleOutput);

void convImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups);
void convImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups,
                       float scaleTargets, float scaleOutput);

void localImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                        int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups);
void localImgActsSparse(NVMatrix& hidActs, NVMatrix& filters, NVMatrix& targets, int* dColorIndices,
                       int imgSizeY, int imgSizeX, int numModulesY, int paddingStart, int moduleStride, int numImgColors, int numFilterColors, int numGroups,
                       float scaleTargets, float scaleOutput);

class AvgPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return a + b;
    }
    __device__ inline float getBaseValue() const {
        return 0;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a / regionSize;
    }
};

class MaxPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fmaxf(a, b);
    }
    __device__ inline float getBaseValue() const {
        return -2e38; 
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};

class MaxAbsPooler {
public:
    __device__ inline float operator()(const float a, const float b) const {
        return fabsf(a) > fabsf(b) ? a : b;
    }
    __device__ inline float getBaseValue() const {
        return 0.0f;
    }
    __device__ inline float output(const float a, const int regionSize) const {
        return a;
    }
};



#endif
