#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>
#include <cstdlib>

using namespace sycl;

int main() {
    constexpr int N = 8192;
    int width  = N;
    int height = N;
    // Total number of elements in the matrix.
    size_t total_elements = static_cast<size_t>(height) * width;

    // Use a GPU selector if available.
    queue q{gpu_selector{}};

    // Allocate device memory using USM.
    float* d_a = malloc_device<float>(total_elements, q);
    float* d_b = malloc_device<float>(total_elements, q);
    float* d_c = malloc_device<float>(total_elements, q);

    // Allocate and initialize host memory.
    float* h_a = static_cast<float*>(malloc(total_elements * sizeof(float)));
    float* h_b = static_cast<float*>(malloc(total_elements * sizeof(float)));
    for (size_t i = 0; i < total_elements; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Asynchronously copy host data to device memory.
    q.memcpy(d_a, h_a, total_elements * sizeof(float));
    q.memcpy(d_b, h_b, total_elements * sizeof(float));
    q.wait();

    // Use an nd_range kernel with an explicit local size.
    // For our total_elements (8192x8192 = 67,108,864) and local size 256,
    // note that total_elements is divisible by 256.
    constexpr int local_size = 256;
    size_t global_size = total_elements; // Already a multiple of 256.

    int total_runs = 10;
    double average_elapsed_ms = 0.0;

    // Warmup run.
    q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), [=](nd_item<1> item) {
        size_t i = item.get_global_id(0);
        d_c[i] = d_a[i] + d_b[i];
    }).wait();

    // Time kernel execution over several runs.
    for (int run = 0; run < total_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        q.parallel_for(nd_range<1>(range<1>(global_size), range<1>(local_size)), [=](nd_item<1> item) {
            size_t i = item.get_global_id(0);
            d_c[i] = d_a[i] + d_b[i];
        }).wait();
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        average_elapsed_ms += elapsed_ms;
        std::cout << "SYCL kernel run " << run << ": " << elapsed_ms << " ms" << std::endl;
    }
    average_elapsed_ms /= total_runs;
    std::cout << "Average kernel execution time: " << average_elapsed_ms << " ms" << std::endl;

    // Copy result back to host.
    float* h_c = static_cast<float*>(malloc(total_elements * sizeof(float)));
    q.memcpy(h_c, d_c, total_elements * sizeof(float)).wait();

    // Verify results.
    bool success = true;
    for (size_t i = 0; i < total_elements; ++i) {
        if (h_c[i] != 3.0f) {
            success = false;
            break;
        }
    }
    std::cout << (success ? "SYCL Matrix Addition successful!" : "SYCL Matrix Addition FAILED!") << std::endl;

    // Free device and host memory.
    free(d_a, q);
    free(d_b, q);
    free(d_c, q);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
