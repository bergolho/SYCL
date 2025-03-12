#include "sycl_library/vector_sycl.hpp"
#include "c_library/vector_c.h"

int main () {
    printf("Hello world from main program!\n");
    hello_sycl();
    hello_c();

    struct ode_system *os = (struct ode_system*)malloc(sizeof(struct ode_system));
    os->neq = 2;
    os->sv = (double*)malloc(sizeof(double)*2);
    free(os->sv);
    free(os);
}