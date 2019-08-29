#ifndef D2Q9_BKG_H
#define D2Q9_BKG_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <stdint.h>
#include <mpi.h>
#include <stdbool.h>

// ======= CONSTANTS ===================================================================
#define NSPEEDS         9
#define FINALSTATEFILE  "final_state.dat"
#define AVVELSFILE      "av_vels.dat"
#define MASTER 0
#define IN_DEGREE 2
#define OUT_DEGREE 2
#define ALIGN 32

// ======= COLLISION CONSTANTS ===================================================================
#define C_SQ       (1.f / 3.f)                          /* square of speed of sound */
#define INV_C_SQ   (1.0f / C_SQ)
#define INV_2C_SQ  (1.0f / (2.f * C_SQ))
#define INV_2C_SQ2 (1.0f / (2.f * C_SQ * C_SQ))
#define W0         (4.f / 9.f)                          /* weighting factor */
#define W1         (1.f / 9.f)                          /* weighting factor */
#define W2         (1.f / 36.f)                         /* weighting factor */

// ======= STRUCTS ===================================================================
/* struct to hold the parameter values */
typedef struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
} t_param;

typedef struct {
    int rank;       //rank of current MPI process
    int size;       //group size of intracommunitor
    int local_x;    //cells in x direction for current rank (columns)
    int local_y;    //cells in y direction for current rank (rows)
    int local_n_w_halos;
    int offset_start_y;   //offset in y direction to start of y's chunk
    int offset_end_y;     //offset in y direction to end of y's chunk
    int non_obstacle_num; //number of total fluid particles (non obstacle cells) in nx.ny size grid
} mpi_comm_info_t;


// ======= PROTOTYPES ===================================================================
/* allocates, loads and initializes paramsm, obstacles & cells */
int initialise(mpi_comm_info_t* comm_info, const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** final_cells_ptr, 
               int** obstacles_ptr, float** av_vels_ptr);

/*
** The main calculation methods. Timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
void swap_ptrs(float** A, float** B);
float timestep(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, const t_param params, float* cells, float* final_cells, int* obstacles);
int accelerate_flow(const mpi_comm_info_t* comm_info, const t_param params, float* cells, int* obstacles);
float timestep_r(const mpi_comm_info_t* const comm_info, const t_param params, const float *const restrict cells, float *const restrict final_cells, const int *const restrict obstacles);

/* finalise, including freeing up allocated memory */
int finalise(const t_param* params, float** cells_ptr, float** final_cells_ptr,
              int** obstacles_ptr, float** av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
//float total_density(const mpi_comm_info_t* comm_info, const t_param params, t_speed* cells);

/* calculate Reynolds number */
//float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, float final_av_vel);

/* Writing average velocities and final state to files */
void write_state(const t_param* params, const int* obstacles, float* all_cells);
void write_av_vels(const t_param* params, float* av_vels);


/* MPI routines */
void mpi_initialize(int* argc, char*** argv, int *rank, int *size);
void mpi_finalize();
void halo_exchange(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, float* cells);
void gather_local_cells_and_obstacles(const mpi_comm_info_t* comm_info, const t_param* params, const float* cells, float* all_cells, const int* obstacles, int* all_obstacles);
void av_vels_reduce(const mpi_comm_info_t* comm_info, const t_param* params, float* av_vels);
void mpi_initialize(int* argc, char*** argv, int *rank, int *size);
void mpi_finalize();
void mpi_create_graph(int rank, int size, MPI_Comm* comm);

// void debug_initialized(const mpi_comm_info_t* comm_info, const t_param* params, const t_speed* cells, const int* obstacles);
// void debug_write(const mpi_comm_info_t* comm_info, const t_param* params, const t_speed* all_cells, const float* av_vels);

void die(const char* message, const int line, const char* file);
void usage(const char* exe);

#endif