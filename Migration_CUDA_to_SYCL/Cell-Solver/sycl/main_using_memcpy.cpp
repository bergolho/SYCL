#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    constexpr int NEQ = 2;
    constexpr int NCELLS = 32;
    constexpr int BLOCK_SIZE = 32;

    sycl::queue q(sycl::default_selector{});

    // Calculate pitch manually
    size_t row_size = NCELLS * sizeof(double);
    size_t pitch_h = ((row_size + 31) / 32) * 32;  // Align to 32 bytes for memory coalescing

    // Allocate memory on device
    double *d_sv = static_cast<double *>(sycl::malloc_device(NEQ * pitch_h, q));

    // Launch Kernel
    q.submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(NCELLS, BLOCK_SIZE), [=](sycl::nd_item<1> item) {
            int threadID = item.get_global_id(0);
            if (threadID < NCELLS) {
                double *row0 = reinterpret_cast<double *>(reinterpret_cast<char *>(d_sv) + 0 * pitch_h);
                double *row1 = reinterpret_cast<double *>(reinterpret_cast<char *>(d_sv) + 1 * pitch_h);
                row0[threadID] = 0.00000820413566106744;
                row1[threadID] = 0.8789655121804799;
            }
        });
    });

    q.wait(); // Ensure kernel execution completes

    // Allocate host memory
    double *sv = new double[NEQ * NCELLS];

    // TODO: Improve this section
    // Copy from device to host manually (row by row)
    for (int i = 0; i < NEQ; i++) {
        q.memcpy(sv + i * NCELLS, reinterpret_cast<char *>(d_sv) + i * pitch_h, row_size).wait();
    }

    // Print results
    for (int i = 0; i < NCELLS; i++) {
        for (int j = 0; j < NEQ; j++) {
            printf("%10.6lf ", sv[j * NCELLS + i]);  // Correct order for printing
        }
        printf("\n");
    }

    // Free memory
    sycl::free(d_sv, q);
    delete[] sv;

    return 0;
}
