#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <mpi.h>
#include <stdbool.h>
#include "error.h"
#include "setup_opencl.h"
#include "structs.h"

// ======= PROTOTYPES ===================================================================
/* allocates, loads and initializes paramsm, obstacles & cells */
int initialise(mpi_comm_info_t* comm_info, const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** final_cells_ptr,
                int** obstacles_ptr, float** av_vels_ptr);
/*
** The main calculation methods. Timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void swap_ptrs(cl_mem* A, cl_mem* B);
/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** final_cells_ptr,
              int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const mpi_comm_info_t* comm_info, const t_param params, float* cells);

/* calculate Reynolds number */
// float calc_reynolds(const t_param* params, t_speed* cells, int* obstacles, float final_av_vel);
// float calc_average_velocity(t_speed cell);

/* Writing average velocities and final state to files */
void write_state(const t_param* params, const int* obstacles, const float* all_cells);
void write_av_vels(const t_param* params, const float* av_vels);


/* MPI routines */
void mpi_initialize(int* argc, char*** argv, int *rank, int *size);
void mpi_finalize();
void halo_exchange(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, const MPI_Datatype* cell_type, float* h_edges, float* h_halos);
void create_and_commit_halo_exchange_type(const mpi_comm_info_t* comm_info, MPI_Datatype* halo_exchange_type);
void gather_local_cells_and_obstacles(const mpi_comm_info_t* comm_info, const t_param* params, const MPI_Datatype* cell_type, const float* cells, float* all_cells, const int* obstacles, int* all_obstacles);
void av_vels_reduce(const mpi_comm_info_t* comm_info, const t_param* params, float* av_vels);
void mpi_initialize(int* argc, char*** argv, int *rank, int *size);
void mpi_finalize();

// void debug_initialized(const mpi_comm_info_t* comm_info, const t_param* params, const t_speed* cells, const int* obstacles);
// void debug_write(const mpi_comm_info_t* comm_info, const t_param* params, const t_speed* all_cells, const float* av_vels);

void usage(const char* exe);

void gather_and_write_results(const mpi_comm_info_t* comm_info, const t_param* params, const MPI_Datatype *cell_type, const float* cells, const int* obstacles, const float* av_vels);
void output_results(const mpi_comm_info_t *comm_info, const t_param* params, float* cells, int* obstacles, float* av_vels, const float sys_time, const float user_time, const float delta_time);
void get_cpu_time(double* user_time, double* sys_time);
double get_time_sec();
void parse_args(char **param_file, char** obstacle_file, const int argc, char** const argv);