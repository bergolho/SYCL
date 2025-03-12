#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <vector>

using namespace sycl;

int main() {
    constexpr int N = 8192;
    constexpr int alignment = 32; // Alignment requirement in terms of floats.
    int width = N;
    int height = N;
    // Compute padded width (acts as a pitch in number of elements).
    //int padded_width = ((width + alignment - 1) / alignment) * alignment;
    //size_t total_elements = height * padded_width;
    size_t total_elements = height * width;

    // Use a GPU selector if available.
    queue q{gpu_selector{}};

    // Allocate device memory (USM device allocations).
    float* d_a = malloc_device<float>(total_elements, q);
    float* d_b = malloc_device<float>(total_elements, q);
    float* d_c = malloc_device<float>(total_elements, q);

    // Prepare host vectors and initialize input data.
    float *h_a = (float*)calloc(total_elements,sizeof(float));
    float *h_b = (float*)calloc(total_elements,sizeof(float));
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;
            h_a[idx] = 1.0f;
            h_b[idx] = 2.0f;
        }
    }

    // Asynchronously copy host data to device memory.
    q.memcpy(d_a, h_a, total_elements * sizeof(float));
    q.memcpy(d_b, h_b, total_elements * sizeof(float));
    q.wait();  // Ensure copies complete.

    // Time the kernel execution using std::chrono.
    int total_runs = 10;
    float average_elapsed_ms = 0.0;
    for (int k = 0; k < total_runs; k++) {
        auto start = std::chrono::high_resolution_clock::now();
        q.parallel_for(range<2>(height, width), [=](id<2> idx) {
            int row = idx[0];
            int col = idx[1];
            int index = row * width + col;
            d_c[index] = d_a[index] + d_b[index];
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        average_elapsed_ms += elapsed_ms;
        std::cout << "SYCL device kernel execution time: " << elapsed_ms << " ms" << std::endl;
    }
    average_elapsed_ms /= (float)total_runs;
    std::cout << "SYCL device kernel average execution time: " << average_elapsed_ms << " ms" << std::endl;
    
    // Asynchronously copy the result back to host memory.
    float *h_c = (float*)calloc(total_elements,sizeof(float));
    q.memcpy(h_c, d_c, total_elements * sizeof(float)).wait();

    // Verify results (only check the valid region).
    bool success = true;
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            int idx = row * width + col;
            if (h_c[idx] != 3.0f) {
                success = false;
                break;
            }
        }
        if (!success) break;
    }
    std::cout << (success ? "SYCL Matrix Addition successful!" : "SYCL Matrix Addition FAILED!") << std::endl;

    // Free device memory
    free(d_a, q);
    free(d_b, q);
    free(d_c, q);
    // Free host memory
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
