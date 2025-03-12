#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixAddKernel(float* A, float* B, float* C, size_t pitch, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index
    int y = blockIdx.y * blockDim.y + threadIdx.y;   // row index
    if (x < width && y < height) {
        // Compute pointer to the beginning of the y-th row using the pitch (in bytes)
        float* rowA = (float*)((char*)A + y * pitch);
        float* rowB = (float*)((char*)B + y * pitch);
        float* rowC = (float*)((char*)C + y * pitch);
        rowC[x] = rowA[x] + rowB[x];
    }
}

int main() {
    const int N = 8192;
    int width = N;
    int height = N;
    size_t pitch;
    float *d_A, *d_B, *d_C;

    // Allocate pitched device memory for matrices A, B, and C.
    cudaMallocPitch(&d_A, &pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_B, &pitch, width * sizeof(float), height);
    cudaMallocPitch(&d_C, &pitch, width * sizeof(float), height);

    // Allocate and initialize host memory.
    float* h_A = new float[width * height];
    float* h_B = new float[width * height];
    for (int i = 0; i < width * height; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Copy host data to device using cudaMemcpy2D.
    cudaMemcpy2D(d_A, pitch, h_A, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, pitch, h_B, width * sizeof(float), width * sizeof(float), height, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int total_runs = 10;
    float average_elapsed_ms = 0.0;
    for (int k = 0; k < total_runs; k++) {
        // Record the start event.
        cudaEventRecord(start);
        // Launch the kernel.
        matrixAddKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, pitch, width, height);
        // Record the stop event.
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Compute elapsed time.
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        average_elapsed_ms += milliseconds;
        std::cout << "CUDA kernel execution time: " << milliseconds << " ms" << std::endl;
    }
    average_elapsed_ms /= (float)total_runs;
    std::cout << "CUDA kernel average execution time: " << average_elapsed_ms << " ms" << std::endl;

    // Copy the result back to host memory.
    float* h_C = new float[width * height];
    cudaMemcpy2D(h_C, width * sizeof(float), d_C, pitch, width * sizeof(float), height, cudaMemcpyDeviceToHost);

    // Verify results.
    bool success = true;
    for (int i = 0; i < width * height; ++i) {
        if (h_C[i] != 3.0f) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "CUDA Matrix Addition successful!" : "CUDA Matrix Addition FAILED!") << std::endl;

    // Cleanup.
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
