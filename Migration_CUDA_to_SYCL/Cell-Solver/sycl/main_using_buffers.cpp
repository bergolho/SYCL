#include <sycl/sycl.hpp>
#include <iostream>

int main() {
    constexpr int NEQ = 2;
    constexpr int NCELLS = 32;
    constexpr int BLOCK_SIZE = 32;

    sycl::queue q(sycl::default_selector{});

    // Allocate host memory dynamically
    double *host_sv = static_cast<double*>(malloc(NEQ * NCELLS * sizeof(double)));

    // Check allocation success
    if (!host_sv) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return -1;
    }
    
    // Create SYCL buffer from dynamically allocated memory
    {
        sycl::buffer<double, 2> sv_buf(host_sv, sycl::range<2>(NEQ, NCELLS));

        // Submit Kernel
        q.submit([&](sycl::handler &h) {
            sycl::accessor sv(sv_buf, h, sycl::write_only, sycl::no_init);

            h.parallel_for(sycl::nd_range<1>(NCELLS, BLOCK_SIZE), [=](sycl::nd_item<1> item) {
                int threadID = item.get_global_id(0);
                if (threadID < NCELLS) {
                    sv[0][threadID] = 0.00000820413566106744; // First row
                    sv[1][threadID] = 0.8789655121804799;     // Second row
                }
            });
        }); // Buffer automatically transfers data back when out of scope
    }

    // No explicit memcpy needed! Data is already back in `host_sv`.

    // Print results
    for (int i = 0; i < NCELLS; i++) {
        for (int j = 0; j < NEQ; j++) {
            printf("%10.6lf ", host_sv[j * NCELLS + i]);  // Correct order for printing
        }
        printf("\n");
    }

    // Free dynamically allocated memory
    free(host_sv);

    return 0;
}
