/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 */

#ifndef COMMON_CUH
#define	COMMON_CUH

#include "nvmatrix.cuh"
#include "conv_util.cuh"

enum FILTER_OUTPUT_ORDER {MODULE_FILTER_IMAGE, FILTER_MODULE_IMAGE};

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


#endif	/* COMMON_CUH */

