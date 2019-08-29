#pragma once
#include "constants.h"

// ======= STRUCTS ===================================================================
/* struct to hold the parameter values */
typedef __attribute__ (()) struct
{
  int    nx;            /* no. of cells in x-direction */
  int    ny;            /* no. of cells in y-direction */
  int    maxIters;      /* no. of iterations */
  int    reynolds_dim;  /* dimension for Reynolds number */
  float density;       /* density per link */
  float accel;         /* density redistribution */
  float omega;         /* relaxation parameter */
  float w1;
  float w2;
} t_param;

/* struct to hold the 'speed' values */
typedef struct
{
  float *speeds;
} t_soa;

typedef __attribute__ (()) struct {
  int rank;       //rank of current MPI process
  int size;       //group size of intracommunitor
  int local_x;    //cells in x direction for current rank (columns)
  int local_y;    //cells in y direction for current rank (rows)
  int local_n;    //number of cells belonging to current rank
  int local_n_w_halos;
  int offset_start_y;   //offset in y direction to start of y's chunk
  int offset_end_y;     //offset in y direction to end of y's chunk
  int non_obstacle_num; //number of total fluid particles (non obstacle cells) in nx.ny size grid
} mpi_comm_info_t;

typedef struct {
  int workgroup_rows;
  int workgroup_cols;
  int workgroup_elems;
  int num_workgroups;
} workgroup_info_t;

