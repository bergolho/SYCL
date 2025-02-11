#include <sycl/sycl.hpp>
#include <iostream>

using namespace sycl;

// Define constants
constexpr int NEQ = 2;
constexpr int NCELLS = 32;
constexpr double dt_ode = 0.02;
constexpr double final_time = 500.0;
constexpr double stim_start = 0.0;
constexpr double stim_duration = 1.0;
constexpr double stim_value = 1.0;
constexpr int max_num_cells_to_stim = 5;
constexpr int plot_cell_id = 1;
constexpr int print_rate = 10;

void write_result (double cur_time, double *sv) {
    char filename[200];
    sprintf(filename,"outputs/sv_sycl_history.txt");
    FILE *file = fopen(filename,"a");

    // Write results for the target cell id
    fprintf(file,"%10.6lf ", cur_time);
    for (int j = 0; j < NEQ; j++) {
        fprintf(file,"%10.6lf ", sv[j * NCELLS + plot_cell_id]);
    }
    fprintf(file,"\n");
    fclose(file);
}

// ODE system right-hand side
void RHS_sycl(double V, double h, double stim_current, double* rDY) {
    const double tau_in = 0.3;
    const double tau_out = 6.0;
    const double V_gate = 0.13;
    const double tau_open = 120.0;
    const double tau_close = 150.0;

    double J_stim = stim_current;
    double J_in = (h * (V * V) * (1.0 - V)) / tau_in;
    double J_out = - (V / tau_out);

    rDY[0] = J_out + J_in + J_stim;
    rDY[1] = (V < V_gate) ? (1.0 - h) / tau_open : -h / tau_close;
}

int main() {
    queue q;

    // Allocate memory for solution variables (using raw pointers instead of std::vector)
    double* sv_host = new double[NEQ * NCELLS]();  // Initialize to zero
    double* stim_currents_host = new double[NCELLS]();  // Initialize to zero

    // Create buffers for SYCL
    buffer<double, 2> sv_buf(sv_host, range<2>(NEQ, NCELLS));
    buffer<double, 1> stim_buf(stim_currents_host, range<1>(NCELLS));

    // Initialize solution variables
    q.submit([&](handler& h) {
        auto sv = sv_buf.get_access<access::mode::write>(h);
        h.parallel_for(range<1>(NCELLS), [=](id<1> i) {
            sv[0][i] = 0.00000820413566106744;
            sv[1][i] = 0.8789655121804799;
        });
    });

    double cur_time = 0.0;
    int count = 0;

    while (cur_time <= final_time) {
        if (count % print_rate == 0) {
            //printf("Time %10.2lf\n", cur_time);
            //if (cur_time < 3.0)
            //    printf("%10.2lf %10.2lf\n", sv_host[0], sv_host[NCELLS]);
            write_result(cur_time, sv_host);
        }
        
        q.submit([&](handler& h) {
            auto stim = stim_buf.get_access<access::mode::write>(h);
            h.parallel_for(range<1>(NCELLS), [=](id<1> i) {
                stim[i] = ((cur_time >= stim_start && cur_time <= stim_start + stim_duration && i < max_num_cells_to_stim) ? stim_value : 0.0);
            });
        }).wait();

        q.submit([&](handler& h) {
            auto sv = sv_buf.get_access<access::mode::read_write>(h);
            auto stim = stim_buf.get_access<access::mode::read>(h);

            h.parallel_for(range<1>(NCELLS), [=](id<1> i) {
                double rDY[NEQ];
                RHS_sycl(sv[0][i], sv[1][i], stim[i], rDY);
                sv[0][i] += dt_ode * rDY[0];
                sv[1][i] += dt_ode * rDY[1];
            });
        }).wait();

        cur_time += dt_ode;
        count++;
    }

    // Clean up allocated memory
    delete[] sv_host;
    delete[] stim_currents_host;

    return 0;
}
