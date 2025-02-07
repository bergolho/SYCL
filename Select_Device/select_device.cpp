#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>

int main (int argc, char *argv[]) {

    if (argc-1 != 1) {
        printf("-------------------------------------------\n");
        printf("Usage:> %s <option_id>\n", argv[0]);
        printf("-------------------------------------------\n");
        printf("<option_id> = || 1->CPU || 2->GPU ||\n");
        printf("-------------------------------------------\n");
        exit(EXIT_FAILURE);
    }

    int option_id = atoi(argv[1]);

    if (option_id == 1) {
        sycl::queue my_queue_cpu(sycl::cpu_selector_v);
        std::cout << "Running on '" << my_queue_cpu.get_device().get_info<sycl::info::device::name>() << "'\n";

    }
    else if (option_id == 2) {
        sycl::queue my_queue_gpu(sycl::gpu_selector_v);
        std::cout << "Running on '" << my_queue_gpu.get_device().get_info<sycl::info::device::name>() << "'\n";
    }
    else {
        fprintf(stderr, "[-] ERROR! Invalid 'option_id=%d'!\n", option_id);
        exit(EXIT_FAILURE);
    }

	return EXIT_SUCCESS;
}
