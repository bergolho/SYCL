#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

// Define the number of equations of the ODE system and
// define the number of cells in the grid to solve
// The block size is for the CUDA kernel
#define NEQ  2
#define NCELLS 32
#define BLOCK_SIZE 32

__constant__ size_t pitch;
size_t pitch_h;

inline __device__ void RHS_gpu (double *sv_, double *rDY_, double stim_current, int threadID_) {

    //State variables
    const double V = *((double*)((char*)sv_ + pitch * 0) + threadID_);
    const double h = *((double*)((char*)sv_ + pitch * 1) + threadID_);

    // Constants
    const double tau_in = 0.3;
    const double tau_out = 6.0;
    const double V_gate = 0.13;
    const double tau_open = 120.0;
    const double tau_close = 150.0;

    // Algebraics
    double J_stim = stim_current;
    double J_in = ( h*( pow(V, 2.00000)*(1.00000 - V)))/tau_in;
    double J_out = - (V/tau_out);

    // Rates
    rDY_[0] = J_out + J_in + J_stim;
    rDY_[1] = (V < V_gate ? (1.00000 - h)/tau_open : - h/tau_close);

}

// Set the initial condition for the ODE system
__global__ void SetInitialConditionsKernel(double *sv) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < NCELLS) {
        // Correctly index into pitched memory
        double *row0 = (double*)((char*)sv + 0 * pitch);
        double *row1 = (double*)((char*)sv + 1 * pitch);
        
        row0[threadID] = 0.00000820413566106744;
        row1[threadID] = 0.8789655121804799;
    }
}

__global__ void SolveOdesKernel (double dt, double *sv, double* stim_currents) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int sv_id;

    // Each thread solves one cell model
    if(threadID < NCELLS) {
        sv_id = threadID;

        double rDY[NEQ];

        RHS_gpu(sv, rDY, stim_currents[threadID], sv_id);

        for(int i = 0; i < NEQ; i++) {
            *((double *) ((char *) sv + pitch * i) + sv_id) = dt * rDY[i] + *((double *) ((char *) sv + pitch * i) + sv_id);
        }            
    }
}

void print_vector (double *arr, int n) {
    for (int i = 0; i < n; i++) {
        printf("\t%d = %lf\n", i, arr[i]);
    }
    printf("\n");
}

void print_result (double *sv, double *d_sv, size_t width) {
    // Copy result from device to host
    cudaMemcpy2D(sv, width, d_sv, pitch_h, width, NEQ, cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < NCELLS; i++) {
        for (int j = 0; j < NEQ; j++) {
            printf("%10.6lf ", sv[j * NCELLS + i]);  // Correct order for printing
        }
        printf("\n");
    }
}

void write_result (double cur_time, double *sv, double *d_sv, size_t width, int cell_id) {
    char filename[200];
    sprintf(filename,"outputs/sv_history.txt");
    FILE *file = fopen(filename,"a");

    // Copy result from device to host
    cudaMemcpy2D(sv, width, d_sv, pitch_h, width, NEQ, cudaMemcpyDeviceToHost);

    // Write results for the target cell id
    fprintf(file,"%10.6lf ", cur_time);
    for (int j = 0; j < NEQ; j++) {
        fprintf(file,"%10.6lf ", sv[j * NCELLS + cell_id]);
    }
    fprintf(file,"\n");
    fclose(file);
}

int main() {
    
    // ODE solver parameters
    double final_time = 500.0;
    double dt_ode = 0.02;
    double cur_time = 0.0;
    uint32_t count = 0;
    uint32_t print_rate = 10;
    int plot_cell_id = 1;

    // Stimulation parameters
    double stim_start = 0.0;
    double stim_duration = 1.0;
    double stim_value = 1.0;

    // Allocate host memory correctly
    double *sv = (double*)malloc(sizeof(double) * NEQ * NCELLS);

    // Device pointer
    double *d_sv;

    // Set up CUDA kernel launch configuration
    const int GRID = (NCELLS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate pitched memory correctly
    size_t width = sizeof(double) * NCELLS; // Width in bytes
    cudaMallocPitch((void**)&d_sv, &pitch_h, width, NEQ);   // memory coalescing
    cudaMemcpyToSymbol(pitch, &pitch_h, sizeof(size_t));

    // Launch the kernel
    SetInitialConditionsKernel<<<GRID, BLOCK_SIZE>>>(d_sv);
    cudaDeviceSynchronize();

    // Stimulus configuration
    double *merged_stims = (double*)calloc(sizeof(double), NCELLS);
    int max_num_cells_to_stim = 5;

    // Main iteration loop
    while (cur_time - final_time <= dt_ode) {
        // Print solution of the ODE system
        if (count % print_rate == 0) {
            //printf("Time %10.2lf\n", cur_time);
            //if (cur_time < 3.0)
            //    print_result(sv, d_sv, width);
            write_result(cur_time, sv, d_sv, width, plot_cell_id);
        }
        
        // Set the stimulation
        memset(merged_stims, 0.0, sizeof(double)*NCELLS);
        if((cur_time >= stim_start) && (cur_time <= stim_start + stim_duration)) {
            for (int i = 0; i < max_num_cells_to_stim; i++) {
                merged_stims[i] += stim_value;
            }
        }

        // Solve ODEs
        size_t stim_currents_size = sizeof(double)*NCELLS;
        double *stims_currents_device;
        cudaMalloc((void **) &stims_currents_device, stim_currents_size);
        cudaMemcpy(stims_currents_device, merged_stims, stim_currents_size, cudaMemcpyHostToDevice);
        
        SolveOdesKernel<<<GRID, BLOCK_SIZE>>>(dt_ode, d_sv, stims_currents_device);

        cudaPeekAtLastError();
        cudaFree(stims_currents_device);

        cur_time += dt_ode;
        count++;
    }

    // Free memory
    cudaFree(d_sv);
    free(sv);
    free(merged_stims);

    return 0;
}
