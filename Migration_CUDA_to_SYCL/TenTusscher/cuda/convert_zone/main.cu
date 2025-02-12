// =============================================================================================
// Program that solves the TenTusscher cellular model on an array of cells using CUDA.
// Author: Lucas Arantes Berg
// Last update: 12/02/2025
// =============================================================================================

#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

// Define the number of equations of the ODE system and
// define the number of cells in the grid to solve
// The block size is for the CUDA kernel
#define NEQ 12
#define NCELLS 32
#define BLOCK_SIZE 32

// TenTusscher type of cell selection
#define ENDO 1
//#define EPI 2
//#define MID 3

__constant__ size_t pitch;
size_t pitch_h;

inline __device__ void RHS_gpu (double *sv_, double *rDY_, double stim_current, int threadID_, double dt, double fibrosis, double *extra_parameters) {

    //fibrosis = 0 means that the cell is fibrotic, 1 is not fibrotic. Anything between 0 and 1 means border zone
    const double svolt = *((double*)((char*)sv_ + pitch * 0) + threadID_);
    const double sm   = *((double*)((char*)sv_ + pitch * 1) + threadID_);
    const double sh   = *((double*)((char*)sv_ + pitch * 2) + threadID_);
    const double sj   = *((double*)((char*)sv_ + pitch * 3) + threadID_);
    const double sxr1 = *((double*)((char*)sv_ + pitch * 4) + threadID_);
    const double sxs  = *((double*)((char*)sv_ + pitch * 5) + threadID_);
    const double ss   = *((double*)((char*)sv_ + pitch * 6) + threadID_);
    const double sf  = *((double*)((char*)sv_ + pitch * 7) + threadID_);
    const double sf2  = *((double*)((char*)sv_ + pitch * 8) + threadID_);
    const double D_INF  = *((double*)((char*)sv_ + pitch * 9) + threadID_);
    const double R_INF  = *((double*)((char*)sv_ + pitch * 10) + threadID_);
    const double Xr2_INF  = *((double*)((char*)sv_ + pitch * 11) + threadID_);

    #include "ten_tusscher_3_RS_common.inc"

}

// Set the initial condition for the ODE system
__global__ void SetInitialConditionsKernel(double *sv) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;

    if (threadID < NCELLS) {
        // Correctly index into pitched memory
        double *row0 = (double*)((char*)sv + 0 * pitch);
        double *row1 = (double*)((char*)sv + 1 * pitch);
        double *row2 = (double*)((char*)sv + 2 * pitch);
        double *row3 = (double*)((char*)sv + 3 * pitch);
        double *row4 = (double*)((char*)sv + 4 * pitch);
        double *row5 = (double*)((char*)sv + 5 * pitch);
        double *row6 = (double*)((char*)sv + 6 * pitch);
        double *row7 = (double*)((char*)sv + 7 * pitch);
        double *row8 = (double*)((char*)sv + 8 * pitch);
        double *row9 = (double*)((char*)sv + 9 * pitch);
        double *row10 = (double*)((char*)sv + 10 * pitch);
        double *row11 = (double*)((char*)sv + 11 * pitch);
        
        row0[threadID] = -86.2;    // V
        row1[threadID] = 0.0;      // m
        row2[threadID] = 0.75;     // h
        row3[threadID] = 0.75;     // j
        row4[threadID] = 0.0;      // xr1
        row5[threadID] = 0.0;      // xs
        row6[threadID] = 1.0;      // s
        row7[threadID] = 1.0;      // f
        row8[threadID] = 1.0;      // f2
        row9[threadID] = 0.0;      // d_inf
        row10[threadID] = 0.0;     // r_inf
        row11[threadID] = 0.0;     // xr2_inf
    }
}

__global__ void SolveOdesKernel (double dt, double *sv, double* stim_currents, double *extra_parameters, double fibrosis) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    int sv_id;

    // Each thread solves one cell model
    if(threadID < NCELLS) {
        sv_id = threadID;

        double rDY[NEQ];

        RHS_gpu(sv, rDY, stim_currents[threadID], sv_id, dt, fibrosis, extra_parameters);

        // Explicit Euler for 'Vm'
        *((double*)((char*)sv) + sv_id) = dt*rDY[0] + *((double*)((char*)sv) + sv_id);

        // Rush-Larsen for the other state variables
        for(int i = 1; i < NEQ; i++) {
            *((double*)((char*)sv + pitch * i) + sv_id) = rDY[i];
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
    sprintf(filename,"outputs/sv_cuda_history.txt");
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
    double dt_ode = 0.01;
    double cur_time = 0.0;
    uint32_t count = 0;
    uint32_t print_rate = 10;
    int plot_cell_id = 1;
    int num_extra_parameters = 8;
    double fibrosis = 1.0;

    // Stimulation parameters
    double stim_start = 0.0;
    double stim_duration = 1.0;
    double stim_value = -53.0;

    // Allocate host memory correctly
    double *sv = (double*)malloc(sizeof(double) * NEQ * NCELLS);
    double *extra_parameters = (double*)malloc(sizeof(double)*num_extra_parameters);

    // Default extra parameters for the TenTusscher model
    extra_parameters[0] = 6.8;                 // atpi
    extra_parameters[1] = 5.4;                 // Ko
    extra_parameters[2] = 138.3;               // Ki
    extra_parameters[3] = 0.0;                  // Vm_modifier
    extra_parameters[4] = 1.0;                 // GNa_modifier
    extra_parameters[5] = 1.0;                 // GCaL_modifier
    extra_parameters[6] = 1.0;                 // INaCa_modifier
    extra_parameters[7] = 1.0;                 // Ikatp_modifier

    // Device pointers for the GPU
    double *d_sv;
    double *d_extra_params;

    // Set up CUDA kernel launch configuration
    const int GRID = (NCELLS + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Allocate pitched memory correctly
    size_t width = sizeof(double) * NCELLS; // Width in bytes
    cudaMallocPitch((void**)&d_sv, &pitch_h, width, NEQ);   // memory coalescing
    cudaMemcpyToSymbol(pitch, &pitch_h, sizeof(size_t));

    // Allocate memory for the extra parameters on the GPU
    size_t extra_parameters_size = num_extra_parameters * sizeof(double);
    cudaMalloc((void **) &d_extra_params, extra_parameters_size);
    cudaMemcpy(d_extra_params, extra_parameters, extra_parameters_size, cudaMemcpyHostToDevice);

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
        
        // Verify which cells will be stimulated
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
        
        SolveOdesKernel<<<GRID, BLOCK_SIZE>>>(dt_ode, d_sv, stims_currents_device, d_extra_params, fibrosis);

        cudaPeekAtLastError();
        cudaFree(stims_currents_device);

        cur_time += dt_ode;
        count++;
    }

    // Free memory
    cudaFree(d_sv);
    cudaFree(d_extra_params);
    free(sv);
    free(merged_stims);
    free(extra_parameters);

    return 0;
}
