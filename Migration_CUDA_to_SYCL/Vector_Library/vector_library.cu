#include "vector_library.h"

__global__ void VectorAddKernel(float* a, float* b, float* c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}