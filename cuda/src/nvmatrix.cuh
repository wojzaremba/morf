/* 
 * Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
 * All rights reserved.
 */
#ifndef NVMATRIX_H_
#define NVMATRIX_H_

#ifndef RND_MULTIPLIERS_FILE
#define RND_MULTIPLIERS_FILE ("rnd_multipliers_32bit.txt")
#endif

#include <map>
#include <cublas.h>
#include <cuda.h>
#include <curand.h>
#include <time.h>
#include <curand_kernel.h>
#include <pthread.h>

#include "mex.h"
#include "cuda.h"
#include "nvmatrix_kernels.cuh"
#include "nvmatrix_operators.cuh"

#ifdef WARNINGS
#define WARN(msg) mexPrintf("WARN: File %s, line %d: %s\n", __FILE__, __LINE__, msg);
#else
#define WARN(msg) ;
#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
                            mexPrintf("Error at %s:%d, x = %d\n",__FILE__,__LINE__, x);\
                            exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
                            mexPrintf("Error at %s:%d\n",__FILE__,__LINE__);\
                            exit(EXIT_FAILURE);}} while(0)
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)
#define assert_(EXP) { if (!(bool)(EXP)) { \
                       mexPrintf("FILE : %s, LINE : %d, EXP: %s \n", __FILE__, __LINE__, #EXP); \
        	       mexErrMsgTxt("!!!! error\n"); \
                     } }
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define MAX(x, y) ((x) > (y) ? (x) : (y))

using namespace std;

inline void __assert_(bool state, const char *file, const int line) {
}

inline void __getLastCudaError(const char *errorMsg, const char *file, const int line) {
        cudaError_t err = cudaGetLastError();
        if (cudaSuccess != err) {
                mexPrintf("%s ( %s ) : get:LastCudaError() CUDA error : %s : ( %d ) %s\n", file, line, errorMsg, (int)err, cudaGetErrorString(err));
        	mexErrMsgTxt("!!!! error\n");
        }
}


class NVMatrix {
private:
    int _numCols, _numRows;
    int _numElements;
    int _stride;
    float* _devData;
    bool _isTrans;
    bool _ownsData;

    static std::map<int,curandState*> rndDevStates;
    static pthread_mutex_t *_rndMutex;

    static void checkCublasError(const char* msg) {
        cublasStatus status = cublasGetError();
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, msg, NULL);
            exit(EXIT_FAILURE);
        }
    }

    char getTransChar() const {
        /*
         * not a typo! return opposite character because a
         * non-transposed krizhevsky matrix is in row-major order while a non-transposed
         * cublas matrix is in column-major order.
         */
        return _isTrans ? 'n' : 't';
    }

    void _init(int numRows, int numCols);
    void _init(int numRows, int numCols, int stride, bool isTrans);
    void _sum_setParams(int n, dim3* blocks, dim3* threads, int* numCols);
    template<class Agg> float _totalAgg(Agg agg);
    template<class Agg, class BinaryOp> void _aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp op);
    template<class Agg, class BinaryOp> NVMatrix& _aggregate(int axis, Agg agg, BinaryOp op);
    template <class Randomizer> void _unaryRandomize(NVMatrix& target, Randomizer rnd);
    template <class Randomizer> void _binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd);   
public:
    NVMatrix();
    NVMatrix(bool isTrans);
    NVMatrix(int numRows, int numCols, bool isTrans=false);
    NVMatrix(const mxArray *like, bool copy);
    NVMatrix(const NVMatrix& like, bool copy);
    NVMatrix(const NVMatrix& like);
    NVMatrix(const mxArray *like);
    NVMatrix(float* devData, int numRows, int numCols, int stride, bool isTrans);
    ~NVMatrix();

    static void initRandom(unsigned long long seed);
    static void initRandom();
    static int getDeviceID();
    static bool isRndInitialized();
    static curandState* getCurandState();
    static void destroyRandom();
    static pthread_mutex_t* makeMutex();

    /*
     * DO NOT DEREFERENCE IN HOST CODE! This is a device memory pointer.
     */
    float* getCellPtr(int i, int j) const {
        if (_isTrans) {
            return &_devData[j * _numRows + i];
        }
        return &_devData[i * _numCols + j];
    }

    bool isSameDims(const mxArray *m) const {
        return ((int)mxGetM(m)) == _numRows && ((int)mxGetN(m)) == _numCols;
    }

    bool isSameDims(const NVMatrix& m) const {
        return m.getNumRows() == _numRows && m.getNumCols() == _numCols;
    }

    int getNumRows() const {
        return _numRows;
    }

    int getNumCols() const {
        return _numCols;
    }

    int getStride() const {
        return _stride;
    }

    int getLeadingDim() const {
        return _isTrans ? _numRows : _numCols;
    }

    int getFollowingDim() const {
        return !_isTrans ? _numRows : _numCols;
    }

    /*
     * FALSE:    Row-major order.
     * TRUE:     Column-major order.
     */
    bool isTrans() const {
        return _isTrans;
    }

    bool isView() const {
        return !_ownsData;
    }

    float* getDevData() const {
        return _devData;
    }

    unsigned int getNumElements() const {
        return _numElements;
    }

    /*
     * Only use if you know what you're doing!
     * Does not actually transpose matrix.
     */
    void setTrans(bool trans) {
        if (trans != _isTrans) {
            assert_(isContiguous());
            _isTrans = trans;
            _stride = getLeadingDim();
        }
    }
    
    /*
     * Only use if you know what you're doing!
     * This toggles whether this object will free its GPU memory when it's destroyed.
     */
    void setView(bool isView) {
        _ownsData = !isView;
    }

    bool isContiguous() const {
        return _stride == getLeadingDim() || getFollowingDim() == 1;
    }
    
    void truncate() {
        resize(0,0);
    }
   

    void copyFromHost(const mxArray* hostMatrix);
    void copyFromHost(const mxArray* hostMatrix, bool resizeDeviceMatrix);
    mxArray* copyToHost() const;
    void copy(NVMatrix& dest) const;
    NVMatrix& copy() const;
    void addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB);
    void addProduct(const NVMatrix& a, const NVMatrix &b);
    void rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, NVMatrix &target) const;
    void rightMult(const NVMatrix &b, float scaleAB);
    void randomizeUniform();
    void addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target);
    void addGaussianNoise(float stdev, NVMatrix& target);
    void addGaussianNoise(NVMatrix& stdevs, bool var);
    void addGaussianNoise(NVMatrix& stdevs);
    void addGaussianNoise(float stdev);
    void addGaussianNoise();
    void randomizeGaussian();
    void randomizeGaussian(float stdev);
    void randomizeGaussian(float mean, float stdev);
    void randomizeGaussian(NVMatrix& stdevs);
    void randomizeGaussian(NVMatrix& stdevs, NVMatrix& target);
    void binarizeProbs();
    void binarizeProbs(NVMatrix& target);

    void biggerThan(NVMatrix& m, NVMatrix& target);
    void biggerThan(NVMatrix& m);
    void biggerThanVector(NVMatrix& vec, NVMatrix& target);
    void biggerThanVector(NVMatrix& vec);
    void equals(NVMatrix& m, NVMatrix& target);
    void equals(NVMatrix& m);

    void _checkBounds(int startRow, int endRow, int startCol, int endCol) const;
    NVMatrix& slice(int startRow, int endRow, int startCol, int endCol) const;
    void slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const;
    NVMatrix& sliceRows(int startRow, int endRow) const;
    void sliceRows(int startRow, int endRow, NVMatrix& target) const;
    NVMatrix& sliceCols(int startCol, int endCol) const;
    void sliceCols(int startCol, int endCol, NVMatrix& target) const;

    template <class Op> void apply(Op op, NVMatrix& target) {
        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }
        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        if (target.isTrans() == isTrans()) {
            kEltwiseUnaryOp<Op><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            getLastCudaError("kEltwiseUnaryOp: Kernel execution failed");
        } else {
            bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
            if (checkBounds) {
                kEltwiseUnaryOpTrans<Op, true><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            } else {
                kEltwiseUnaryOpTrans<Op, false><<<blocks, threads>>>(_devData, target._devData, height, width, getStride(), target.getStride(), op);
            }
            getLastCudaError("kEltwiseUnaryOpTrans: Kernel execution failed");
        }
    }
    
    template <class Op> void apply(Op op) {
        apply(op, *this);
    }
    
    template <class Op> void applyBinary(Op op, NVMatrix& b) {
        applyBinary(op, b, *this);
    }

    template <class Op> void applyBinary(Op op, NVMatrix& b, NVMatrix& target) {
        assert_(this->isSameDims(b));

        if (!target.isSameDims(*this)) {
            target.resize(*this);
        }

        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                    std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        if (target.isTrans() == isTrans() && target.isTrans() == b.isTrans()) {
            kEltwiseBinaryOp<Op><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width, getStride(),
                                                      b.getStride(), target.getStride(), op);
            getLastCudaError("kEltwiseBinaryOp: Kernel execution failed");
        } else {
            //  both x here since y divides x
            bool checkBounds = !(width % ELTWISE_THREADS_X == 0 && height % ELTWISE_THREADS_X == 0);
            if (target.isTrans() == isTrans() && target.isTrans() != b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,false,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,false,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                }
            } else if (target.isTrans() != isTrans() && target.isTrans() != b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,true,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,true,false><<<blocks, threads>>>(_devData, b._devData, target._devData, height, width,getStride(),
                                                               b.getStride(), target.getStride(), op);
                }
            } else if (target.isTrans() != isTrans() && target.isTrans() == b.isTrans()) {
                if (checkBounds) {
                    kEltwiseBinaryOpTrans<Op,true,false,true><<<blocks, threads>>>(b._devData, _devData, target._devData, height, width,b.getStride(),
                                                               getStride(), target.getStride(), op);
                } else {
                    kEltwiseBinaryOpTrans<Op,false,false,true><<<blocks, threads>>>(b._devData, _devData, target._devData, height, width, b.getStride(),
                                                               getStride(), target.getStride(), op);
                }
            }
            getLastCudaError("kEltwiseBinaryOpTrans: Kernel execution failed");
        }
    }
    
    template <class Op> void applyTernary(Op op, NVMatrix& b, NVMatrix& c, NVMatrix& target) {
        assert_(this->isSameDims(b));
        assert_(this->isSameDims(c));
        // For now ternary ops are only supported for matrices of same transposedness
        assert_(isTrans() == b.isTrans());
        assert_(isTrans() == c.isTrans());
        if (!target.isSameDims(*this) || target.isTrans() != isTrans()) {
            target.resize(*this);
        }

        int height = target.getFollowingDim(), width = target.getLeadingDim();
        dim3 blocks(std::min(NUM_BLOCKS_MAX, DIVUP(width, ELTWISE_THREADS_X)),
                    std::min(NUM_BLOCKS_MAX, DIVUP(height, ELTWISE_THREADS_Y)));
        dim3 threads(ELTWISE_THREADS_X, ELTWISE_THREADS_Y);
        kEltwiseTernaryOp<Op><<<blocks, threads>>>(_devData, b._devData, c._devData, target._devData, height, width,
                                                   getStride(), b.getStride(), c.getStride(), target.getStride(), op);
        getLastCudaError("kEltwiseTernaryOp: Kernel execution failed");
    }

    bool resize(int numRows, int numCols);
    bool resize(const NVMatrix &like);
    bool resize(const mxArray *like);
    void reshape(int numRows, int numCols);
    NVMatrix& reshaped(int numRows, int numCols);
    void copy(NVMatrix &dest, int srcStartRow, int srcEndRow, int srcStartCol, int srcEndCol, int destStartRow, int destStartCol) const;
    void add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, float scaleB, NVMatrix& target);
    void add(NVMatrix& b, NVMatrix& target);
    void add(NVMatrix& b, float scaleB);
    void add(NVMatrix& b, float scaleA, float scaleB);
    void add(NVMatrix& b);
    void eltwiseMult(NVMatrix& b);
    void eltwiseMult(NVMatrix& b, NVMatrix& target);
    void eltwiseDivide(NVMatrix& b);
    void eltwiseDivide(NVMatrix& b, NVMatrix& target);
    void squaredDiff(NVMatrix& b);
    void squaredDiff(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b, NVMatrix& target);
    void subtract(NVMatrix& b);
    void addVector(NVMatrix& vec, float scaleVec, NVMatrix& target);
    void addVector(NVMatrix& vec);
    void addVector(NVMatrix& vec, float scaleVec);
    void addVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec, NVMatrix& target);
    void equalsVector(NVMatrix& vec);
    void eltwiseMultByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseMultByVector(NVMatrix& vec);
    void eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target);
    void eltwiseDivideByVector(NVMatrix& vec);
    void tile(int timesY, int timesX, NVMatrix& target);

    void addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum);
    void sum(int axis, NVMatrix& target);
    NVMatrix& sum(int axis);
    void max(int axis, NVMatrix& target);
    NVMatrix& max(int axis);
    void min(int axis, NVMatrix& target);
    NVMatrix& min(int axis);
    float mean();
    float sum();
    float max();
    float min();
    float norm2();
    float norm();
    
    void inRangeInc(float lower, float upper);
    void inRangeInc(float lower, float upper, NVMatrix& target);
    void inRangeExc(float lower, float upper);
    void inRangeExc(float lower, float upper, NVMatrix& target);
    void biggerThanScalar(float scalar);
    void biggerThanScalar(float scalar, NVMatrix& target);
    void smallerThanScalar(float scalar);
    void smallerThanScalar(float scalar, NVMatrix& target);
    void addScalar(float scaleThis, float scalar, NVMatrix& target);
    void addScalar(float scalar, NVMatrix& target);
    void addScalar(float scalar);
    void minWithScalar(float scalar, NVMatrix& target);
    void minWithScalar(float scalar);
    void maxWithScalar(float scalar, NVMatrix& target);
    void maxWithScalar(float scalar);
    void pow(float p, NVMatrix& target);
    void pow(float p);
    void scale(float _scale);
    void scale(float _scale, NVMatrix& target);

    float dotProduct(NVMatrix& b);

    /*
     * Does SOFT transpose and returns result, leaving this matrix unchanged
     */
    NVMatrix& getTranspose();

    /*
     * Does HARD transpose and puts result in target
     */
    void transpose(NVMatrix& target);

    /*
     * Does SOFT transpose
     */
    void transpose();
    bool transpose(bool trans);

    void flipTrans(NVMatrix& target);
    NVMatrix& flipTrans();

    void print(const char* name) const;
    void printShape(const char* name) const;

    template <class Op> void applyBinaryV(Op op, NVMatrix& vec, NVMatrix& target) {
        assert_(&target != &vec); // for now
        assert_(vec.getNumRows() == 1 || vec.getNumCols() == 1);
        assert_(vec.getNumRows() == _numRows || vec.getNumCols() == _numCols);
        assert_(vec.isContiguous());

        target.resize(*this); // target must be same orientation as me for now

        int width = getLeadingDim(); //_isTrans ? _numRows : _numCols;
        int height = getFollowingDim(); //_isTrans ? _numCols : _numRows;
        dim3 threads(ADD_VEC_THREADS_X, ADD_VEC_THREADS_Y);
        dim3 blocks(MIN(NUM_BLOCKS_MAX, DIVUP(width, ADD_VEC_THREADS_X)), MIN(NUM_BLOCKS_MAX, DIVUP(height, ADD_VEC_THREADS_Y)));

        if (vec.getNumRows() == _numRows && !isTrans() || vec.getNumCols() == _numCols && isTrans()) {
            kColVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
        } else {
            kRowVectorOp<Op><<<blocks,threads>>>(_devData, vec._devData, target._devData, width, height, getStride(), target.getStride(), op);
        }
        //getLastCudaError("Kernel execution failed");
        //cudaThreadSynchronize();
    }

    template<class UnaryOperator> float argMax(UnaryOperator u) {
       return _totalAgg(NVMatrixAggs::ArgMax<UnaryOperator>(u));
    }
};

/*
 * Device random number generator pointers.
 */
//map<int,curandGenerator_t> NVMatrix::rndGen;
map<int,curandState*> NVMatrix::rndDevStates;
pthread_mutex_t* NVMatrix::_rndMutex = makeMutex();

pthread_mutex_t* NVMatrix::makeMutex() {
    pthread_mutex_t* m = (pthread_mutex_t*) malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(m, NULL);
    return m;
}

void NVMatrix::_init(int numRows, int numCols, int stride, bool isTrans) {
    _numRows = numRows;
    _numCols = numCols;
    _numElements = numRows * numCols;
    _ownsData = true;

    _isTrans = isTrans;
    _devData = NULL;
    if (_numElements > 0) {
        cublasAlloc(_numElements, sizeof(float), (void**) &_devData);
        checkCublasError("!!!! device memory allocation error\n");
    }
    _stride = stride < 0 ? getLeadingDim() : stride;
}

NVMatrix::NVMatrix() {
    _init(0, 0, -1, false);
}

NVMatrix::NVMatrix(bool isTrans) {
    _init(0, 0, -1, isTrans);
}

NVMatrix::NVMatrix(int numRows, int numCols, bool isTrans) {
    _init(numRows, numCols, -1, isTrans);
}

NVMatrix::NVMatrix(const mxArray* like, bool copy) {
    _init((int)mxGetM(like), (int)mxGetN(like), -1, true);
    if (copy) {
        copyFromHost(like);
    }
}


NVMatrix::NVMatrix(const NVMatrix& like, bool copy) {
    _init(like.getNumRows(), like.getNumCols(), -1, like.isTrans());
    if (copy) {
        like.copy(*this);
    }
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const NVMatrix& like) {
    _init(like.getNumRows(), like.getNumCols(), -1, like.isTrans());
}

/*
 * Initializes NVMatrix with same dimensions as given matrix but
 * does not copy any data.
 */
NVMatrix::NVMatrix(const mxArray* like) {
    _init((int)mxGetM(like), (int)mxGetN(like), -1, true);
}

NVMatrix::NVMatrix(float* devData, int numRows, int numCols, int stride, bool isTrans) :
    _numRows(numRows),
    _numCols(numCols),
    _numElements(numRows*numCols),
    _ownsData(false),
    _devData(devData),
    _isTrans(isTrans) {
    _stride = stride < 0 ? getLeadingDim() : stride;
}

NVMatrix::~NVMatrix() {
    if(_ownsData && _numElements > 0) {
        cublasStatus status = cublasFree(_devData);
        if (status != CUBLAS_STATUS_SUCCESS) {
            mexErrMsgTxt("!!!! memory free error\n");
            exit(EXIT_FAILURE);
        }
    }
}

void NVMatrix::copyFromHost(const mxArray* hostMatrix, bool resizeDeviceMatrix) {
    if (resizeDeviceMatrix) {
        resize(hostMatrix);
    }
    copyFromHost(hostMatrix);
}

void NVMatrix::copyFromHost(const mxArray *hostMatrix) {
    assert_(getStride() == getLeadingDim());
    assert_(isSameDims(hostMatrix));
    setTrans(true);

    if (getNumElements() > 0) {
	float* data = (float*)mxGetData(hostMatrix);
	cublasStatus status;
        status = cublasSetMatrix(getLeadingDim(), getFollowingDim(), sizeof(float),
                                 data, getLeadingDim(), _devData, _stride);
        if (status != CUBLAS_STATUS_SUCCESS) {
  	    cudaError_t error = cudaGetLastError();
	    mexPrintf("ERROR : %s\n", cudaGetErrorString(error));
            mexErrMsgTxt("!!!! device access error (write)\n");
        }
    }
}

mxArray* NVMatrix::copyToHost() const {
    const mwSize dims[] = {_numRows, _numCols};
    assert_(_isTrans);
    mxArray* hostMatrix = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
    if (getNumElements() > 0) {
	float* data = (float*)mxGetPr(hostMatrix);
	cublasStatus status;
        	status = cublasGetMatrix(getLeadingDim(), getFollowingDim(), sizeof(float),
                	                 _devData, getStride(), data, getLeadingDim());
        if (status != CUBLAS_STATUS_SUCCESS) {
  	    cudaError_t error = cudaGetLastError();
	    mexPrintf("ERROR : %s\n", cudaGetErrorString(error));
            mexErrMsgTxt("!!!! device access error (read)\n");
            exit( EXIT_FAILURE);
        }
    }
    return hostMatrix;
}

void NVMatrix::copy(NVMatrix& dest) const {
    dest.resize(*this);
    copy(dest, 0, -1, 0, -1, 0, 0);
}

NVMatrix& NVMatrix::copy() const {
    NVMatrix* c = new NVMatrix();
    copy(*c);
    return *c;
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB, NVMatrix &target) const {
    assert_(isContiguous() && b.isContiguous() && target.isContiguous());
//    assert_(&target != &b);
    assert_(_numCols == b.getNumRows());
    if(&target != this) {
        target.resize(_numRows, b.getNumCols());
        target.setTrans(true);
    }
    assert_(target.getNumRows() == _numRows);
    assert_(target.getNumCols() == b.getNumCols());
    if(_numRows % 64 != 0 || _numCols % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(getTransChar(), b.getTransChar(), _numRows, b.getNumCols(), _numCols,
                scaleAB, _devData, getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                0, target.getDevData(), getNumRows());
    checkCublasError("cublasSgemm failed");
//    cudaThreadSynchronize();
}

void NVMatrix::rightMult(const NVMatrix &b, float scaleAB) {
    rightMult(b, scaleAB, *this);
}

void NVMatrix::rightMult(const NVMatrix &b, NVMatrix& target) const {
    rightMult(b, 1, target);
}

/*
 * This will only work if this matrix is in column-major order! In other words,
 * if isTrans() returns true.
 */
void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b, float scaleThis, float scaleAB) {
    if (scaleThis == 0) {
        a.rightMult(b, scaleAB, *this);
        return;
    }
    assert_(isContiguous());
    assert_(a.getNumCols() == b.getNumRows());
    assert_(this->getNumRows() == a.getNumRows());
    assert_(this->getNumCols() == b.getNumCols());
    assert_(_isTrans);
    if(a.getNumRows() % 64 != 0 || a.getNumCols() % 64 != 0 || b.getNumCols() % 64 != 0) {
        WARN("Matrix dimensions not divisible by 64 -- cublasSgemm performance may suffer.");
    }
    cublasSgemm(a.getTransChar(), b.getTransChar(), a.getNumRows(), b.getNumCols(), a.getNumCols(),
                scaleAB, a.getDevData(), a.getLeadingDim(), b.getDevData(), b.getLeadingDim(),
                scaleThis, _devData, getLeadingDim());
    checkCublasError("cublasSgemm failed");
//    cudaThreadSynchronize();
}

void NVMatrix::addProduct(const NVMatrix& a, const NVMatrix &b) {
    addProduct(a, b, 1, 1);
}

template <class Randomizer>
void NVMatrix::_unaryRandomize(NVMatrix& target, Randomizer rnd) {
    assert_(isRndInitialized());
    assert_(isContiguous() && target.isContiguous());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert_(isTrans() == target.isTrans());
    kUnaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kUnaryRandomize: Kernel execution failed");
}

template <class Randomizer>
void NVMatrix::_binaryRandomize(NVMatrix& data2, NVMatrix& target, Randomizer rnd) {
    assert_(isRndInitialized());
    assert_(isContiguous() && data2.isContiguous() && target.isContiguous());
    assert_(isSameDims(data2));
    assert_(isTrans() == data2.isTrans());
    if (!isSameDims(target)) {
        target.resize(*this);
    }
    assert_(isTrans() == target.isTrans());
    kBinaryRandomize<<<NUM_RND_BLOCKS,NUM_RND_THREADS_PER_BLOCK>>>(getDevData(), data2.getDevData(), target.getDevData(), getCurandState(), getNumElements(), rnd);
    getLastCudaError("kBinaryRandomize: Kernel execution failed");
}

void NVMatrix::initRandom(unsigned long long seed) {
    assert_(!isRndInitialized());
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    rndDevStates[d] = NULL;
    CUDA_CALL(cudaMalloc((void **)&rndDevStates[d], NUM_RND_STREAMS * sizeof(curandState)));
    pthread_mutex_unlock(_rndMutex);
    kSetupCurand<<<NUM_RND_BLOCKS, NUM_RND_THREADS_PER_BLOCK>>>(getCurandState(), 1 + seed*2); // so there's no chance it'll be correlated with the other one
    getLastCudaError("initRandom: Kernel execution failed");
}

void NVMatrix::initRandom() {
    NVMatrix::initRandom(time(0));
}

curandState* NVMatrix::getCurandState() {
    pthread_mutex_lock(_rndMutex);
    int d = getDeviceID();
    assert_(rndDevStates.count(d) != 0);
    curandState* r = rndDevStates[d];
    pthread_mutex_unlock(_rndMutex);
    return r;
}

int NVMatrix::getDeviceID() {
    int d;
    cudaGetDevice(&d);
    return d;
}

bool NVMatrix::isRndInitialized() {
    pthread_mutex_lock(_rndMutex);
    bool b = rndDevStates.count(getDeviceID()) != 0;
    pthread_mutex_unlock(_rndMutex);
    return b;
}

void NVMatrix::destroyRandom() {
    assert_(isRndInitialized());
    int d = getDeviceID();
    
    pthread_mutex_lock(_rndMutex);
    CUDA_CALL(cudaFree(rndDevStates[d]));
    rndDevStates.erase(d);
    pthread_mutex_unlock(_rndMutex);
}

void NVMatrix::binarizeProbs() {
    binarizeProbs(*this);
}

void NVMatrix::binarizeProbs(NVMatrix& target) {
    _unaryRandomize(target, BinarizeUnaryRandomizer());
}

void NVMatrix::randomizeUniform() {
    assert_(isContiguous());
    assert_(isRndInitialized());
//    CURAND_CALL(curandGenerateUniform(rndGen, _devData, getNumElements()));
    _unaryRandomize(*this, UniformUnaryRandomizer());
}

void NVMatrix::randomizeGaussian() {
    randomizeGaussian(1);
}

void NVMatrix::randomizeGaussian(float stdev) {
    randomizeGaussian(0, stdev);
}

void NVMatrix::randomizeGaussian(float mean, float stdev) {
    assert_(isContiguous());
    assert_(isRndInitialized());
//    CURAND_CALL(curandGenerateNormal(rndGen, _devData, getNumElements(), mean, stdev));
    _unaryRandomize(*this, GaussianUnaryRandomizer(mean, stdev));
}

/*
 * Kind of a hack since we don't actually need the contents of this matrix for it,
 * so we don't really need a binary randomizer.
 */
void NVMatrix::randomizeGaussian(NVMatrix& stdevs) {
    _binaryRandomize(stdevs, *this, GaussianBinaryRandomizer());
}

void NVMatrix::addGaussianNoise() {
    addGaussianNoise(1);
}

void NVMatrix::addGaussianNoise(float stdev) {
    addGaussianNoise(stdev, *this);
}

void NVMatrix::addGaussianNoise(float stdev, NVMatrix& target) {
    _unaryRandomize(target, AddGaussianUnaryRandomizer(stdev));
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var) {
    addGaussianNoise(stdevs, var, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs) {
    addGaussianNoise(stdevs, false, *this);
}

void NVMatrix::addGaussianNoise(NVMatrix& stdevs, bool var, NVMatrix& target) {
    if (var) {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<true>());
    } else {
        _binaryRandomize(stdevs, target, AddGaussianBinaryRandomizer<false>());
    }
}

void NVMatrix::biggerThan(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::BiggerThan(), b, target);
}

void NVMatrix::biggerThan(NVMatrix& b) {
    biggerThan(b, *this);
}

void NVMatrix::equals(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Equals(), b, target);
}

void NVMatrix::equals(NVMatrix& m) {
    equals(m, *this);
}

void NVMatrix::biggerThanVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::BiggerThan(), vec, target);
}

void NVMatrix::biggerThanVector(NVMatrix& vec) {
    biggerThanVector(vec, *this);
}

void NVMatrix::_checkBounds(int startRow, int endRow, int startCol, int endCol) const {
    assert_(startRow >= 0 && startRow < _numRows);
    assert_(endRow > startRow && endRow <= _numRows);
    assert_(startCol >= 0 && startCol < _numCols);
    assert_(endCol > startCol && endCol <= _numCols);
}

/*
 * The only place where stride is supported for now!
 * Will ALWAYS return a view of the original data, sometimes non-contiguous.
 */
NVMatrix& NVMatrix::slice(int startRow, int endRow, int startCol, int endCol) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);
    if (!isTrans()) {
        return *new NVMatrix(this->_devData + startRow * _stride + startCol, endRow - startRow, endCol - startCol, _stride, false);
    } 
    return *new NVMatrix(this->_devData + startCol * _stride + startRow, endRow - startRow, endCol - startCol, _stride, true);
}

/* this will NEVER return a view */
void NVMatrix::slice(int startRow, int endRow, int startCol, int endCol, NVMatrix& target) const {
    endRow = endRow < 0 ? this->_numRows : endRow;
    endCol = endCol < 0 ? this->_numCols : endCol;
    _checkBounds(startRow, endRow, startCol, endCol);

    int sliceRows = endRow - startRow, sliceCols = endCol - startCol;
    if (target.getNumRows() != sliceRows || target.getNumCols() != sliceCols) {
        target.resize(sliceRows, sliceCols);
    }
    this->copy(target, startRow, endRow, startCol, endCol, 0, 0);
}

NVMatrix& NVMatrix::sliceRows(int startRow, int endRow) const {
    return slice(startRow, endRow, 0, -1);
}

void NVMatrix::sliceRows(int startRow, int endRow, NVMatrix& target) const {
    slice(startRow, endRow, 0, -1, target);
}

NVMatrix& NVMatrix::sliceCols(int startCol, int endCol) const {
    return slice(0, -1, startCol, endCol);
}

void NVMatrix::sliceCols(int startCol, int endCol, NVMatrix& target) const {
    slice(0, -1, startCol, endCol, target);
}

/*
 * Guaranteed to not change the data if the number of elements doesn't change.
 * So you can use this to "reshape" a matrix.
 */
bool NVMatrix::resize(int numRows, int numCols) {
    bool reallocated = false;
    if (numRows != _numRows || numCols != _numCols) {
        assert_(_ownsData);
        if (_numElements != numRows * numCols) {
            if (_numElements > 0) { // free old memory
                cublasStatus status = cublasFree(_devData);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    mexErrMsgTxt("!!!! memory free error\n");
                    exit(EXIT_FAILURE);
                }
            }
            if (numRows * numCols > 0) { // allocate new memory
                cublasStatus status = cublasAlloc(numCols * numRows, sizeof(float), (void**) &_devData);
                if (status != CUBLAS_STATUS_SUCCESS) {
                    mexErrMsgTxt("!!!! device memory allocation error\n");
                    exit(EXIT_FAILURE);
                }
            } else {
                _devData = NULL;
            }
            reallocated = true;
        }
        _numRows = numRows;
        _numCols = numCols;
        _numElements = numRows * numCols;
        _stride = getLeadingDim();
    }
    return reallocated;
}

bool NVMatrix::resize(const NVMatrix& like) {
    setTrans(like.isTrans());
    return resize(like.getNumRows(), like.getNumCols());
}

bool NVMatrix::resize(const mxArray* like) {
    setTrans(true);
    return resize((int)mxGetM(like), (int)mxGetN(like));
}

void NVMatrix::reshape(int numRows, int numCols) {
    assert_(isContiguous());
    assert_(_numElements == numRows*numCols);
    _numRows = numRows;
    _numCols = numCols;
    _stride = getLeadingDim();
}

NVMatrix& NVMatrix::reshaped(int numRows, int numCols) {
    assert_(isContiguous());
    assert_(_numElements == numRows*numCols);
    return *new NVMatrix(_devData, numRows, numCols, -1, _isTrans);
}

void NVMatrix::copy(NVMatrix &dest, int srcStartRow, int srcEndRow,
                    int srcStartCol, int srcEndCol,
                    int destStartRow, int destStartCol) const {
    srcEndRow = srcEndRow < 0 ? _numRows : srcEndRow;
    srcEndCol = srcEndCol < 0 ? _numCols : srcEndCol;
    NVMatrix* srcSlice = &slice(srcStartRow, srcEndRow, srcStartCol, srcEndCol);
    NVMatrix* destSlice = &dest.slice(destStartRow, destStartRow + srcEndRow - srcStartRow, destStartCol, destStartCol + srcEndCol - srcStartCol);
    srcSlice->apply(NVMatrixOps::Identity(), *destSlice);
    delete srcSlice;
    delete destSlice;
}


NVMatrix& NVMatrix::getTranspose() {
    return *new NVMatrix(_devData, _numCols, _numRows, _stride, !_isTrans);;
}

void NVMatrix::transpose(NVMatrix& target) {
    flipTrans(target);
    target.setTrans(!target.isTrans());
    target.reshape(target.getNumCols(), target.getNumRows());
}

void NVMatrix::transpose() {
    int tmp = _numCols;
    _numCols = _numRows;
    _numRows = tmp;
    _isTrans = !_isTrans;
}

bool NVMatrix::transpose(bool trans) {
    bool oldTrans = _isTrans;
    if (oldTrans != trans) {
        transpose();
    }
    return oldTrans;
}

/*
 * Flips the ordering of the matrix from row-major to column-major and vice versa.
 * This creates temporary storage -- not a cheap operation.
 *
 * This is not equivalent to a "hard transpose". The resultant matrix still has
 * the same dimensions, its layout in memory just changes.
 */
NVMatrix& NVMatrix::flipTrans() {
    NVMatrix* meTrans = new NVMatrix(*this);
    flipTrans(*meTrans);
    return *meTrans;
}

void NVMatrix::flipTrans(NVMatrix& target) {
    assert_(&target != this);
    target.resize(_numRows, _numCols);
    target.setTrans(!isTrans());
    apply(NVMatrixOps::Identity(), target);
}

void NVMatrix::squaredDiff(NVMatrix& b) {
    squaredDiff(b, *this);
}

void NVMatrix::squaredDiff(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::SquaredDiff(), b, target);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB, NVMatrix& target) {
    if (scaleA == 0) {
        b.scale(scaleB, target);
        return;
    }
    if (scaleA == 1 && scaleB == 1) { // slight optimization
        applyBinary(NVMatrixBinaryOps::Add(), b, target);
    } else {
        applyBinary(NVMatrixBinaryOps::WeightedAdd(scaleA, scaleB), b, target);
    }
}

void NVMatrix::add(NVMatrix& b, float scaleB, NVMatrix& target) {
    add(b, 1, scaleB, target);
}

void NVMatrix::add(NVMatrix& b, NVMatrix& target) {
    add(b, 1, target);
}

void NVMatrix::add(NVMatrix& b, float scaleB) {
    add(b, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b, float scaleA, float scaleB) {
    add(b, scaleA, scaleB, *this);
}

void NVMatrix::add(NVMatrix& b) {
    add(b, 1, *this);
}

void NVMatrix::subtract(NVMatrix& b, NVMatrix& target) {
    add(b, -1, target);
}

void NVMatrix::subtract(NVMatrix& b) {
    add(b, -1);
}

void NVMatrix::eltwiseMult(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Multiply(), b, target);
}

void NVMatrix::eltwiseMult(NVMatrix& b) {
    eltwiseMult(b, *this);
}

void NVMatrix::eltwiseDivide(NVMatrix& b, NVMatrix& target) {
    applyBinary(NVMatrixBinaryOps::Divide(), b, target);
}

void NVMatrix::eltwiseDivide(NVMatrix& b) {
    eltwiseDivide(b, *this);
}

void NVMatrix::tile(int timesY, int timesX, NVMatrix& target) {
    assert_(isContiguous() && target.isContiguous());
    assert_(timesX > 0 && timesY > 0);
    target.resize(_numRows*timesY, _numCols*timesX);
    target.setTrans(_isTrans);
    if(!isTrans()) {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK>>>(_devData, target._devData, _numCols, _numRows, target._numCols, target._numRows);
    } else {
        kTile<<<NUM_TILE_BLOCKS,NUM_TILE_THREADS_PER_BLOCK>>>(_devData, target._devData, _numRows, _numCols, target._numRows, target._numCols);
    }
    getLastCudaError("Kernel execution failed");
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::WeightedAdd(1, scaleVec), vec, target);
}

void NVMatrix::addVector(NVMatrix& vec) {
    addVector(vec, 1, *this);
}

void NVMatrix::addVector(NVMatrix& vec, float scaleVec) {
    addVector(vec, scaleVec, *this);
}

void NVMatrix::addVector(NVMatrix& vec, NVMatrix& target) {
    addVector(vec, 1, target);
}

void NVMatrix::equalsVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Equals(), vec, target);
}

void NVMatrix::equalsVector(NVMatrix& vec) {
    equalsVector(vec, *this);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Multiply(), vec, target);
}

void NVMatrix::eltwiseMultByVector(NVMatrix& vec) {
    eltwiseMultByVector(vec, *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec) {
    eltwiseDivideByVector(vec,  *this);
}

void NVMatrix::eltwiseDivideByVector(NVMatrix& vec, NVMatrix& target) {
    applyBinaryV(NVMatrixBinaryOps::Divide(), vec, target);
}

/*
 * num threads per block is ignored when summing rows (axis=1) because
 * it has to be a power of 2.
 *
 * TODO: this is a mess, fix it. it works pretty fast but it's too ugly.
 * TODO: this function is _really_ bad for very long aggregations of few columns.
 */
template<class Agg, class BinaryOp>
void NVMatrix::_aggregate(int axis, NVMatrix& target, Agg agg, BinaryOp op) {
    assert_(axis == 0 || axis == 1);
    assert_(isContiguous()  && target.isContiguous());
    assert_(&target != this);
    int width = _isTrans ? _numRows : _numCols;
    int height = _isTrans ? _numCols : _numRows;

    target.setTrans(_isTrans);
    assert_(width > 0);
    assert_(height > 0);
    if(axis == 0 && !_isTrans || axis == 1 && _isTrans) { //col sum
        target.resize(!_isTrans ? 1 : _numRows, !_isTrans ? _numCols : 1);
        int numBlocks = DIVUP(width, NUM_SUM_COLS_THREADS_PER_BLOCK);
        assert_(numBlocks * NUM_SUM_COLS_THREADS_PER_BLOCK >= width);
        assert_(numBlocks < NUM_BLOCKS_MAX);
        kDumbAggCols<Agg, BinaryOp><<<numBlocks,NUM_SUM_COLS_THREADS_PER_BLOCK>>>(_devData, target._devData, width, height, agg, op);
        getLastCudaError("kDumbAggCols: Kernel execution failed");
    } else { // row sum
        target.resize(_isTrans ? 1 : _numRows, _isTrans ? _numCols : 1);
        if (width > 1) {
            if (height >= 16384) { // linear aggregation
                int numBlocksX = 1;
                int numBlocksY = DIVUP(height, AGG_SHORT_ROWS_THREADS_Y*AGG_SHORT_ROWS_LOOPS_Y);
                int numThreadsX = width <= 4 ? 4 : width <= 8 ? 8 : width <= 12 ? 12 : width <= 16 ? 16 : AGG_SHORT_ROWS_THREADS_X;
                int numThreadsY = AGG_SHORT_ROWS_THREADS_Y;
                while (numBlocksY > NUM_BLOCKS_MAX) {
                    numBlocksY = DIVUP(numBlocksY,2);
                    numBlocksX *= 2;
                }
                dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                if(width <= 16) {
                    if(width <= 4) {
                        kAggShortRows<Agg, BinaryOp, 1, 4><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else if(width <= 8) {
                        kAggShortRows<Agg, BinaryOp, 1, 8><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else if(width <= 12) {
                        kAggShortRows<Agg, BinaryOp, 1, 12><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    } else {
                        kAggShortRows<Agg, BinaryOp, 1, 16><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                    }
                } else if(width <= 32) {
                    kAggShortRows<Agg, BinaryOp, 2, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else if(width <= 48){
                    kAggShortRows<Agg, BinaryOp, 3, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else if(width <= 64){
                    kAggShortRows<Agg, BinaryOp, 4, AGG_SHORT_ROWS_THREADS_X><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                } else {
                    kAggShortRows2<Agg, BinaryOp><<<grid, threads>>>(_devData, target._devData,width, height, agg, op);
                }
            } else {
                if (width >= 512) {
                    dim3 threads(AWR_NUM_THREADS);
                    dim3 blocks(1, std::min(1024, height));
                    kAggRows_wholerow_nosync<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
//                    dim3 threads(AWR_NUM_THREADS);
//                    dim3 blocks(1, std::min(1024, height));
//                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
                    
                } else {
//                    dim3 threads(AWR_NUM_THREADS);
//                    dim3 blocks(1, std::min(1024, height));
//                    kAggRows_wholerow<<<blocks, threads>>>(_devData, target._devData, width, height, agg, op);
                    NVMatrix *prevSum = this;
                    while (prevSum->getLeadingDim() > 1) {
                        int numThreadsX = width <= 64 ? 32 : (width <= 128 ? 64 : (width <= 256 ? 128 : (width <= 512 ? 256 : 512)));
                        int numThreadsY = 1;
                        int numBlocksX = DIVUP(width, 2*numThreadsX);
                        int numBlocksY = std::min(height, NUM_BLOCKS_MAX);
                        NVMatrix *nvSumAccum = target.getFollowingDim() == height && target.getLeadingDim() == numBlocksX ? &target : new NVMatrix(height, numBlocksX, false);

                        dim3 grid(numBlocksX, numBlocksY), threads(numThreadsX, numThreadsY);
                        assert_(numBlocksX <= NUM_BLOCKS_MAX);
                        assert_(numBlocksY <= NUM_BLOCKS_MAX);

                        if(width <= 64) {
                            kAggRows<Agg, BinaryOp, 32><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 128) {
                            kAggRows<Agg, BinaryOp, 64><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 256) {
                            kAggRows<Agg, BinaryOp, 128><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else if(width <= 512) {
                            kAggRows<Agg, BinaryOp, 256><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        } else {
                            kAggRows<Agg, BinaryOp, 512><<<grid, threads>>>(prevSum->_devData, nvSumAccum->_devData,
                                                       width, height, nvSumAccum->getLeadingDim(), agg, op);
                        }
                        getLastCudaError("agg rows: Kernel execution failed");
                        cudaThreadSynchronize();
                        width = numBlocksX; // only true in reduction agg, but for linear agg this doesn't matter anyway

                        if (prevSum != this) {
                            delete prevSum;
                        }
                        prevSum = nvSumAccum;
                    }
                }
            }
        } else {
            copy(target);
        }
    }
}

void NVMatrix::inRangeInc(float lower, float upper) {
    inRangeInc(lower, upper, *this);
}
void NVMatrix::inRangeInc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<false>(lower, upper), target);
}

void NVMatrix::inRangeExc(float lower, float upper) {
    inRangeExc(lower, upper, *this);
}

void NVMatrix::inRangeExc(float lower, float upper, NVMatrix& target) {
    apply(NVMatrixOps::InRange<true>(lower, upper), target);
}

void NVMatrix::biggerThanScalar(float scalar) {
    biggerThanScalar(scalar, *this);
}

void NVMatrix::biggerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::BiggerThanScalar(scalar), target);
}

void NVMatrix::smallerThanScalar(float scalar) {
    smallerThanScalar(scalar, *this);
}

void NVMatrix::smallerThanScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::SmallerThanScalar(scalar), target);
}

void NVMatrix::addScalar(float scaleThis, float scalar, NVMatrix& target) {
    apply(NVMatrixOps::WeightedAddScalar(scaleThis, scalar), target);
}

void NVMatrix::addScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::AddScalar(scalar), target);
}

void NVMatrix::addScalar(float scalar) {
    addScalar(scalar, *this);
}

void NVMatrix::minWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MinWithScalar(scalar), target);
}

void NVMatrix::minWithScalar(float scalar) {
    minWithScalar(scalar, *this);
}

void NVMatrix::maxWithScalar(float scalar, NVMatrix& target) {
    apply(NVMatrixOps::MaxWithScalar(scalar), target);
}

void NVMatrix::maxWithScalar(float scalar) {
    maxWithScalar(scalar, *this);
}

void NVMatrix::pow(float p, NVMatrix& target) {
    apply(NVMatrixOps::Pow(p), target);
}

void NVMatrix::pow(float p) {
    pow(p, *this);
}

void NVMatrix::scale(float _scale) {
    scale(_scale, *this);
}

void NVMatrix::scale(float _scale, NVMatrix& target) {
    if (_scale != 1 || &target != this) { // optimize away scale by 1
        apply(NVMatrixOps::MultByScalar(_scale), target);
    }
}

template<class Agg, class BinaryOp>
NVMatrix& NVMatrix::_aggregate(int axis, Agg agg, BinaryOp op) {
    NVMatrix *sumVec = new NVMatrix();
    _aggregate<Agg, BinaryOp>(axis, *sumVec, agg, op);
    return *sumVec;
}


void NVMatrix::max(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

void NVMatrix::addSum(NVMatrix& a, int axis, float scaleThis, float scaleSum) {
    if (scaleThis != 0) {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::WeightedAdd(scaleThis, scaleSum));
    } else {
        a._aggregate(axis, *this, NVMatrixAggs::Sum(), NVMatrixBinaryOps::SecondScaled(scaleSum));
    }
}

void NVMatrix::sum(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

void NVMatrix::min(int axis, NVMatrix& target) {
    _aggregate(axis, target, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::max(int axis) {
    return _aggregate(axis, NVMatrixAggs::Max(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::sum(int axis) {
    return _aggregate(axis, NVMatrixAggs::Sum(), NVMatrixBinaryOps::Second());
}

NVMatrix& NVMatrix::min(int axis) {
    return _aggregate(axis, NVMatrixAggs::Min(), NVMatrixBinaryOps::Second());
}

void NVMatrix::_sum_setParams(int n, dim3* blocks, dim3* threads, int* numCols) {
    int logn = int(ceil(log(float(n)) / log(2)));
    *numCols = DIVUP(n, logn);
    int numThreads = *numCols;
    *blocks = dim3(DIVUP(numThreads, DP_BLOCKSIZE));
    *threads = dim3(DP_BLOCKSIZE);
}

float NVMatrix::mean() {
    return sum() / getNumElements();
}

float NVMatrix::sum() {
    return _totalAgg(NVMatrixAggs::Sum());
}

float NVMatrix::max() {
    return _totalAgg(NVMatrixAggs::Max());
}

float NVMatrix::min() {
    return _totalAgg(NVMatrixAggs::Min());
}

template<class Agg>
float NVMatrix::_totalAgg(Agg agg) {
    assert_(isContiguous());
    dim3 blocks, threads;
    int numCols;
    // Sum most of it on GPU
    NVMatrix* src = this;
    for (NVMatrix* target = NULL; src->getNumElements() > CPUSUM_MAX; src = target) {
        _sum_setParams(src->getNumElements(), &blocks, &threads, &numCols);
        target = new NVMatrix(1, blocks.x);
        kTotalAgg<<<blocks, threads>>>(src->getDevData(), target->getDevData(), numCols, src->getNumElements(), agg);
        getLastCudaError("kTotalAgg: Kernel execution failed");
        cudaThreadSynchronize(); // not really necessary?
        delete (src == this ? NULL : src);
    }

    assert_(0);
    return 0;
    /*Matrix srcCPU(src->getNumRows(), src->getNumCols());
    src->copyToHost(srcCPU);
    if (src->getNumElements() > 1) { // Sum remainder on CPU
        delete (src == this ? NULL : src);
        if (typeid(Agg) == typeid(NVMatrixAggs::Sum)) {
            return srcCPU.sum();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Max)) {
            return srcCPU.max();
        } else if (typeid(Agg) == typeid(NVMatrixAggs::Min)) {
            return srcCPU.min();
        } else {
            assert_(false);
        }
    }
    return srcCPU(0,0);*/
}

/*
 * Fast dot product only for matrices with same transposedness.
 */
float NVMatrix::dotProduct(NVMatrix& b) {
    assert_(isContiguous() && b.isContiguous());
    assert_(isSameDims(b));
    assert_(isTrans() == b.isTrans()); // see?
    dim3 blocks, threads;
    int numCols;
    _sum_setParams(getNumElements(), &blocks, &threads, &numCols);
    NVMatrix target(1, blocks.x);
    kDotProduct_r<<<blocks, threads>>>(getDevData(), b.getDevData(), target.getDevData(), numCols, getNumElements());
    getLastCudaError("kDotProduct: Kernel execution failed");
    cudaThreadSynchronize();
    return target.sum();
}

float NVMatrix::norm2() {
    return dotProduct(*this);
}

float NVMatrix::norm() {
    return sqrt(norm2());
}

void NVMatrix::printShape(const char* name) const {
    mexPrintf("%s: %dx%d%s, getLeadingDim() = %d, getFollowingDim() = %d \n", name, _numRows, _numCols, _isTrans ? "^T" : "", getLeadingDim(), getFollowingDim());
}


#endif /* NVMATRIX_H_ */
