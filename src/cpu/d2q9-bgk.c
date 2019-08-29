#include "d2q9-bgk.h"

#define HALOS 2

// main program: initialise, timestep loop, finalise
int main(int argc, char* argv[])
{
  char*    paramfile = NULL;    /* name of the input parameter file */
  char*    obstaclefile = NULL; /* name of a the input obstacle file */
  t_param  params;              /* struct to hold parameter values */
  float* cells     = NULL;    /* grid containing fluid densities */
  float* final_cells = NULL;    /* scratch space */
  int*     obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */
  struct timeval timstr;        /* structure to hold elapsed time */
  struct rusage ru;             /* structure to hold CPU time--system and user */
  double tic, toc;              /* floating point numbers to calculate elapsed wallclock time */
  double usrtim;                /* floating point number to record elapsed user CPU time */
  double systim;                /* floating point number to record elapsed system CPU time */

  /* parse the command line */
  if (argc != 3) usage(argv[0]);
  else{
    paramfile = argv[1];
    obstaclefile = argv[2];
  }

  /* initialise our data structures and load values from file */
  mpi_comm_info_t comm_info = {};
  mpi_initialize(&argc, &argv, &comm_info.rank, &comm_info.size);
  MPI_Comm graph_comm;
  mpi_create_graph(comm_info.rank, comm_info.size, &graph_comm);
  initialise(&comm_info, paramfile, obstaclefile, &params, &cells, &final_cells, &obstacles, &av_vels);
  //if(comm_info.rank == 0) debug_initialized(&comm_info, &params, cells, obstacles);

  /* iterate for maxIters timesteps */
  gettimeofday(&timstr, NULL);
  tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  for (int tt = 0; tt < params.maxIters; tt++){
    av_vels[tt] = timestep(&comm_info, &graph_comm, params, cells, final_cells, obstacles);
    swap_ptrs(&cells, &final_cells);
  }

  gettimeofday(&timstr, NULL);
  toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  usrtim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  systim = timstr.tv_sec + (timstr.tv_usec / 1000000.0);

  /* Reduction (gather) operation on average velocities */
  av_vels_reduce(&comm_info ,&params, av_vels);

  /* write final values and free memory */
  if(comm_info.rank == MASTER){
    printf("==done==\n");
    printf("size:%dx%d\n", params.ny, params.nx);
    //printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, av_vels[params.maxIters-1]));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", toc - tic);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", usrtim);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", systim);
  }
  
  /* Gather operation on invidual sub-domains (sub-cells and sub-obstacles) */
  float* all_cells = NULL; int* all_obstacles = NULL;
  if(comm_info.rank == MASTER) all_cells = malloc(sizeof(float) * NSPEEDS * params.nx * params.ny);
  if(comm_info.rank == MASTER) all_obstacles = malloc(sizeof(int) * params.nx * params.ny);
  gather_local_cells_and_obstacles(&comm_info, &params, cells, all_cells, obstacles, all_obstacles); //collective
  //if(comm_info.rank == MASTER) debug_write(&comm_info, &params, all_cells, av_vels);

  /* Master writes these into files */
  if(comm_info.rank == MASTER) {
    write_state(&params, all_obstacles, all_cells);
    write_av_vels(&params, av_vels);
  }
  free(all_cells); free(all_obstacles);

  finalise(&params, &cells, &final_cells, &obstacles, &av_vels);
  mpi_finalize();

  return EXIT_SUCCESS;
}

void swap_ptrs(float** A, float** B){
  float* temp;
  temp = *A;
  *A = *B;
  *B = temp;
}

float timestep(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, const t_param params, float* cells, float* final_cells, int* obstacles)
{
  accelerate_flow(comm_info, params, cells, obstacles);
  halo_exchange(comm_info, comm, cells);
  float av_vel = timestep_r(comm_info, params, cells, final_cells, obstacles);
  return av_vel;
}

void av_vels_reduce(const mpi_comm_info_t* comm_info, const t_param* params, float* av_vels)
{
  if(comm_info->rank == 0) MPI_Reduce(MPI_IN_PLACE, av_vels, params->maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  else MPI_Reduce(av_vels, av_vels, params->maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

int accelerate_flow(const mpi_comm_info_t* comm_info, const t_param params, float* cells, int* obstacles)
{
  /* modify the 2nd to last (or one before the last) row of the grid */
  int accel_jj = params.ny - 2;

  if(accel_jj >= comm_info->offset_start_y && accel_jj < comm_info->offset_end_y){
    int jj = accel_jj - comm_info->offset_start_y + (HALOS - 1);

    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    int previous_row = (jj-1)*comm_info->local_x;
    int curret_row = jj*comm_info->local_x;
    for (int ii = 0; ii < comm_info->local_x; ii++) {
      /* If the cell is not occupied && we don't sent a negative density */
      if (!obstacles[ii + previous_row]
            && (cells[comm_info->local_n_w_halos*3 + ii + curret_row] - w1) > 0.f
            && (cells[comm_info->local_n_w_halos*6 + ii + curret_row] - w2) > 0.f
            && (cells[comm_info->local_n_w_halos*7 + ii + curret_row] - w2) > 0.f) {
          /* increase 'east-side' densities */
          cells[comm_info->local_n_w_halos*1 + ii + curret_row] += w1;
          cells[comm_info->local_n_w_halos*5 + ii + curret_row] += w2;
          cells[comm_info->local_n_w_halos*8 + ii + curret_row] += w2;
          /* decrease 'west-side' densities */
          cells[comm_info->local_n_w_halos*3 + ii + curret_row] -= w1;
          cells[comm_info->local_n_w_halos*6 + ii + curret_row] -= w2;
          cells[comm_info->local_n_w_halos*7 + ii + curret_row] -= w2;
      }
    }
  }
  return EXIT_SUCCESS;
}

void halo_exchange(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, float* cells){
  //edge indices and halo indices
  int bottom_cells[3] = { 6, 2, 5 };
  int top_cells[3] = { 7, 4, 8 };
  for(int speed = 0; speed < 3; speed++){
    int tag = 0;
    float* top_edge = comm_info->local_n_w_halos*top_cells[speed] + (HALOS - 1) * comm_info->local_x + cells;
    float* bottom_edge = comm_info->local_n_w_halos*bottom_cells[speed] + (comm_info->local_y + (HALOS - 1) - 1) * comm_info->local_x + cells;
    float* top_halo = comm_info->local_n_w_halos*bottom_cells[speed] + 0 + cells;
    float* bottom_halo = comm_info->local_n_w_halos*top_cells[speed] + (comm_info->local_y + (HALOS - 1)) * comm_info->local_x + cells;
    int rank_below = (comm_info->rank + 1) % comm_info->size;
    int rank_above = ((comm_info->rank - 1) < 0) ? (comm_info->rank - 1) + comm_info->size : (comm_info->rank - 1);
    if(comm_info->rank % 2 == 0){
      MPI_Sendrecv(top_edge, comm_info->local_x, MPI_FLOAT, rank_above, tag, top_halo, comm_info->local_x, MPI_FLOAT, rank_above, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);    
      MPI_Sendrecv(bottom_edge, comm_info->local_x, MPI_FLOAT, rank_below, tag, bottom_halo, comm_info->local_x, MPI_FLOAT, rank_below, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if(comm_info->rank % 2 == 1){
      MPI_Sendrecv(bottom_edge, comm_info->local_x, MPI_FLOAT, rank_below, tag, bottom_halo, comm_info->local_x, MPI_FLOAT, rank_below, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Sendrecv(top_edge, comm_info->local_x, MPI_FLOAT, rank_above, tag, top_halo, comm_info->local_x, MPI_FLOAT, rank_above, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
}

float timestep_r(const mpi_comm_info_t* const comm_info, const t_param params, const float *const restrict cells, float *const restrict final_cells, const int *const restrict obstacles)
{
  float tot_u = 0.0f;          /* accumulated magnitudes of velocity for each cell */

  for(int jj = HALOS - 1; jj < comm_info->local_y + (HALOS - 1); jj++){
    #pragma simd
    #pragma vector aligned
    for (int ii = 0; ii < comm_info->local_x; ii++) {

      //=============== PROPOGATE ============================================================    
      /* determine indices of axis-direction neighbours
      ** respecting periodic boundary conditions (wrap around) */
      int y_n = jj + 1; int y_s = jj - 1;
      int x_e = (ii + 1) % comm_info->local_x;
      int x_w = (ii - 1 < 0) ? (ii - 1 + comm_info->local_x) : (ii - 1);
      /* propagate densities from neighbouring cells, following appropriate 
      directions of travel and writing into scratch space grid */
      float temp_cell[NSPEEDS];
      temp_cell[0] = cells[comm_info->local_n_w_halos*0 + ii + jj*comm_info->local_x]; /* central cell, no movement */
      temp_cell[1] = cells[comm_info->local_n_w_halos*1 + x_w + jj*comm_info->local_x]; /* east */
      temp_cell[2] = cells[comm_info->local_n_w_halos*2 + ii + y_s*comm_info->local_x]; /* north */
      temp_cell[3] = cells[comm_info->local_n_w_halos*3 + x_e + jj*comm_info->local_x]; /* west */
      temp_cell[4] = cells[comm_info->local_n_w_halos*4 + ii + y_n*comm_info->local_x]; /* south */
      temp_cell[5] = cells[comm_info->local_n_w_halos*5 + x_w + y_s*comm_info->local_x]; /* north-east */
      temp_cell[6] = cells[comm_info->local_n_w_halos*6 + x_e + y_s*comm_info->local_x]; /* north-west */
      temp_cell[7] = cells[comm_info->local_n_w_halos*7 + x_e + y_n*comm_info->local_x]; /* south-west */
      temp_cell[8] = cells[comm_info->local_n_w_halos*8 + x_w + y_n*comm_info->local_x]; /* south-east */

      //===============  ============================================================
      int isObstacle = !temp_cell[0] ? 1 : 0;

      //=============== REBOUND & COLLISION ===========================================
      /* don't consider occupied cells */
      /* compute local density total */
      float local_density = 0.f;
      #pragma novector
      for (int kk = 0; kk < NSPEEDS; kk++) {
        local_density += temp_cell[kk];
      }

      /* compute x velocity component */
      float u_x = (temp_cell[1]
                    + temp_cell[5]
                    + temp_cell[8]
                    - (temp_cell[3]
                        + temp_cell[6]
                        + temp_cell[7]))
                    / local_density;
      /* compute y velocity component */
      float u_y = (temp_cell[2]
                    + temp_cell[5]
                    + temp_cell[6]
                    - (temp_cell[4]
                        + temp_cell[7]
                        + temp_cell[8]))
                    / local_density;

      /* velocity squared */
      float u_sq = u_x * u_x + u_y * u_y;

      /* directional velocity components */
      float u[NSPEEDS];
      u[1] =   u_x;        /* east */
      u[2] =         u_y;  /* north */
      u[3] = - u_x;        /* west */
      u[4] =       - u_y;  /* south */
      u[5] =   u_x + u_y;  /* north-east */
      u[6] = - u_x + u_y;  /* north-west */
      u[7] = - u_x - u_y;  /* south-west */
      u[8] =   u_x - u_y;  /* south-east */

      /* equilibrium densities */
      float d_equ[NSPEEDS];
      /* zero velocity density: weight w0 */
      d_equ[0] = W0 * local_density
                  * (1.f - u_sq * INV_2C_SQ);
      /* axis speeds: weight w1 */
      #pragma novector 
      d_equ[1] = W1 * local_density * (1.f + u[1] * INV_C_SQ
                                        + (u[1] * u[1]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[2] = W1 * local_density * (1.f + u[2] * INV_C_SQ
                                        + (u[2] * u[2]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[3] = W1 * local_density * (1.f + u[3] * INV_C_SQ
                                        + (u[3] * u[3]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[4] = W1 * local_density * (1.f + u[4] * INV_C_SQ
                                        + (u[4] * u[4]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      /* diagonal speeds: weight w2 */
      d_equ[5] = W2 * local_density * (1.f + u[5] * INV_C_SQ
                                        + (u[5] * u[5]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[6] = W2 * local_density * (1.f + u[6] * INV_C_SQ
                                        + (u[6] * u[6]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[7] = W2 * local_density * (1.f + u[7] * INV_C_SQ
                                        + (u[7] * u[7]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);
      d_equ[8] = W2 * local_density * (1.f + u[8] * INV_C_SQ
                                        + (u[8] * u[8]) * INV_2C_SQ2
                                        - u_sq * INV_2C_SQ);

      /* relaxation step */
      #pragma novector
      final_cells[comm_info->local_n_w_halos*0 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[0] 
                  : temp_cell[0] + params.omega * (d_equ[0] - temp_cell[0]);
      final_cells[comm_info->local_n_w_halos*1 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[3] 
                  : temp_cell[1] + params.omega * (d_equ[1] - temp_cell[1]);
      final_cells[comm_info->local_n_w_halos*2 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[4] 
                  : temp_cell[2] + params.omega * (d_equ[2] - temp_cell[2]);
      final_cells[comm_info->local_n_w_halos*3 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[1] 
                  : temp_cell[3] + params.omega * (d_equ[3] - temp_cell[3]);
      final_cells[comm_info->local_n_w_halos*4 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[2] 
                  : temp_cell[4] + params.omega * (d_equ[4] - temp_cell[4]);
      final_cells[comm_info->local_n_w_halos*5 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[7] 
                  : temp_cell[5] + params.omega * (d_equ[5] - temp_cell[5]);
      final_cells[comm_info->local_n_w_halos*6 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[8] 
                  : temp_cell[6] + params.omega * (d_equ[6] - temp_cell[6]);
      final_cells[comm_info->local_n_w_halos*7 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[5] 
                  : temp_cell[7] + params.omega * (d_equ[7] - temp_cell[7]);
      final_cells[comm_info->local_n_w_halos*8 + ii + jj*comm_info->local_x] =
        isObstacle ? temp_cell[6] 
                  : temp_cell[8] + params.omega * (d_equ[8] - temp_cell[8]);

      //========== AVERAGE VELOCITIES ============================================
      /* accumulate the norm of x- and y- velocity components */
      tot_u += isObstacle ? 0 : sqrt(u_sq);
    }
  }
  return tot_u / (float) comm_info->non_obstacle_num;
}

int initialise(mpi_comm_info_t* comm_info, const char* paramfile, const char* obstaclefile,
               t_param* params, float** cells_ptr, float** final_cells_ptr,
               int** obstacles_ptr, float** av_vels_ptr) {

  MPI_File fhandler_param;
  int failed = MPI_File_open(MPI_COMM_WORLD, paramfile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandler_param);
  if(failed) die("Could not open input parameter file", __LINE__, __FILE__);
  char file_content[UINT16_MAX];
  MPI_File_read_all(fhandler_param, file_content, UINT16_MAX, MPI_CHAR, MPI_STATUS_IGNORE);
  int retval = sscanf(file_content, "%d\n%d\n%d\n%d\n%f\n%f\n%f\n", &(params->nx), 
      &(params->ny), &(params->maxIters), &(params->reynolds_dim), &(params->density), 
      &(params->accel), &(params->omega));
  if(retval != 7) die("could not read param file", __LINE__, __FILE__);
  MPI_File_close(&fhandler_param);

  /* Set local_x and local_y */
  int extra_rows = (params->ny % comm_info->size);
  int average_rows = (params->ny / comm_info->size);
  if(comm_info->rank < extra_rows) {
    comm_info->local_y = average_rows + 1;
    comm_info->offset_start_y = (average_rows + 1) * comm_info->rank;
  }
  else {
    comm_info->local_y = average_rows;
    comm_info->offset_start_y = (average_rows + 1) * (extra_rows) + average_rows * (comm_info->rank - extra_rows);  
  }
  comm_info->local_x = params->nx;
  comm_info->offset_end_y = comm_info->offset_start_y + comm_info->local_y;
  comm_info->local_n_w_halos = (comm_info->local_y + HALOS) * comm_info->local_x;

  /* main grid */
  *cells_ptr = (float*) _mm_malloc(sizeof(float) * NSPEEDS * comm_info->local_n_w_halos, ALIGN);
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);
  /* 'helper' grid, used as scratch space */
  *final_cells_ptr = (float*) _mm_malloc(sizeof(float) * NSPEEDS * comm_info->local_n_w_halos, ALIGN);
  if (*final_cells_ptr == NULL) die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

  /* the map of obstacles */
  *obstacles_ptr = (int*) calloc((comm_info->local_y * comm_info->local_x), sizeof(int));
  if (*obstacles_ptr == NULL) die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

  /* initialise densities */
  float w0 = params->density * 4.f / 9.f;
  float w1 = params->density      / 9.f;
  float w2 = params->density      / 36.f;

  for (int jj = HALOS - 1; jj < comm_info->local_y + HALOS - 1; jj++) {
    for (int ii = 0; ii < comm_info->local_x; ii++) {
      /* centre */
      (*cells_ptr)[comm_info->local_n_w_halos*0 + ii + jj*comm_info->local_x] = w0;
      /* axis directions */
      (*cells_ptr)[comm_info->local_n_w_halos*1 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos*2 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos*3 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos*4 + ii + jj*comm_info->local_x] = w1;
      /* diagonals */
      (*cells_ptr)[comm_info->local_n_w_halos*5 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos*6 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos*7 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos*8 + ii + jj*comm_info->local_x] = w2;
    }
  }
  
  MPI_File fhandler_obstacles;
  MPI_File_open(MPI_COMM_WORLD, obstaclefile, MPI_MODE_RDONLY, MPI_INFO_NULL, &fhandler_obstacles);
  MPI_File_read_all(fhandler_obstacles, file_content, UINT16_MAX, MPI_CHAR, MPI_STATUS_IGNORE);
  int xx, yy, blocked, total_offset, offset, obstacle_num;
  total_offset = 0; comm_info->non_obstacle_num = 0; obstacle_num = 0;
  while ((retval = sscanf(file_content + total_offset, "%d %d %d\n%n", &xx, &yy, &blocked, &offset)) > 0) {
    if (retval != 3) die("expected 3 values per line in obstacle file", __LINE__, __FILE__);  
    if (xx < 0 || xx > params->nx - 1) die("obstacle x-coord out of range", __LINE__, __FILE__);
    if (yy < 0 || yy > params->ny - 1) die("obstacle y-coord out of range", __LINE__, __FILE__);
    if (blocked != 1) die("obstacle blocked value should be 1", __LINE__, __FILE__);

    obstacle_num++;
    total_offset += offset;
    if(yy >= comm_info->offset_start_y && yy < comm_info->offset_end_y){
        yy = yy - comm_info->offset_start_y;
        (*obstacles_ptr)[xx + yy*comm_info->local_x] = blocked;
    }
  }
  comm_info->non_obstacle_num = (params->ny*params->nx) - obstacle_num;
  MPI_File_close(&fhandler_obstacles);

  for(int jj = HALOS - 1; jj < comm_info->local_y + (HALOS - 1); jj++){
    for(int ii = 0; ii < comm_info->local_x; ii++){
      if((*obstacles_ptr)[ii + (jj-1)*comm_info->local_x]){
        (*cells_ptr)[ii + jj*comm_info->local_x] = 0;
        (*final_cells_ptr)[ii + jj*comm_info->local_x] = 0;
      }
    }
  }

  /*
  ** allocate space to hold a record of the avarage velocities computed
  ** at each timestep
  */
  *av_vels_ptr = (float*) calloc(params->maxIters, sizeof(float));
  // for(int i = 0; i < params->maxIters; i++){
  //   (*av_vels_ptr)[i] = comm_info->rank;
  // }

  return EXIT_SUCCESS;
}

int finalise(const t_param* params, float** cells_ptr, float** final_cells_ptr,
              int** obstacles_ptr, float** av_vels_ptr)
{
  /*
  ** free up allocated memory
  */
  // free(*cells_ptr);
  // *cells_ptr = NULL;

  // free(*final_cells_ptr);
  // *final_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


// float calc_reynolds(const t_param params, t_speed* cells, int* obstacles, float final_av_vel)
// {
//   const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

//   return final_av_vel * params.reynolds_dim / viscosity;
// }

// float total_density(const mpi_comm_info_t* comm_info, const t_param params, t_speed* cells)
// {
//   float total = 0.f;  /* accumulator */

//   for (int jj = HALOS - 1; jj < comm_info->local_y + HALOS - 1; jj++)
//   {
//     for (int ii = 0; ii < comm_info->local_x; ii++)
//     {
//       for (int kk = 0; kk < NSPEEDS; kk++)
//       {
//         total += cells[ii + jj*params.nx].speeds[kk];
//       }
//     }
//   }

//   return total;
// }

void gather_local_cells_and_obstacles(const mpi_comm_info_t* comm_info, const t_param* params, const float* cells, float* all_cells, const int* obstacles, int* all_obstacles){
  int *displs, *recvcounts;
  displs = malloc(comm_info->size * sizeof(int));
  recvcounts = malloc(comm_info->size * sizeof(int));
  int average_y = (params->ny / comm_info->size);
  int extra_rows = (params->ny % comm_info->size);
  for(int i = 0; i < comm_info->size; i++){
    if(i < extra_rows) {
      recvcounts[i] = (average_y + 1) * comm_info->local_x;
      displs[i] = ((average_y + 1) * i) * comm_info->local_x;
    }
    else {
      recvcounts[i] = average_y * comm_info->local_x;
      displs[i] = ((average_y + 1) * (extra_rows) + average_y * (i - extra_rows)) * comm_info->local_x;
    }
  }

  for(int speed = 0; speed < NSPEEDS; speed++){
    MPI_Gatherv(cells + comm_info->local_n_w_halos * speed + (HALOS - 1) * comm_info->local_x, 
      recvcounts[comm_info->rank], MPI_FLOAT, all_cells + (params->nx * params->ny) * speed, 
      recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  MPI_Gatherv(obstacles, recvcounts[comm_info->rank], MPI_INT,
     all_obstacles, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
}

void write_state(const t_param* params, const int* all_obstacles, float* all_cells){
  
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  FILE* fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  int params_n_w_host = (params->ny * params->nx);
  for (int jj = 0; jj < params->ny; jj++) {
    for (int ii = 0; ii < params->nx; ii++) {
      float u_x, u_y, u, pressure;

      /* an occupied cell */
      if (all_obstacles[ii + jj*params->nx]) {
        u_x = u_y = u = 0;
        pressure = params->density * c_sq;
      }
      /* no obstacle */
      else {
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++) {
          local_density += all_cells[params_n_w_host*kk + ii + jj*params->nx];
        }

        /* compute x velocity component for cell */
        u_x = (all_cells[params_n_w_host*1 + ii + jj*params->nx]
               + all_cells[params_n_w_host*5 + ii + jj*params->nx]
               + all_cells[params_n_w_host*8 + ii + jj*params->nx]
               - (all_cells[params_n_w_host*3 + ii + jj*params->nx]
                  + all_cells[params_n_w_host*6 + ii + jj*params->nx]
                  + all_cells[params_n_w_host*7 + ii + jj*params->nx]))
              / local_density;
        /* compute y velocity component for cell */
        u_y = (all_cells[params_n_w_host*2 + ii + jj*params->nx]
               + all_cells[params_n_w_host*5 + ii + jj*params->nx]
               + all_cells[params_n_w_host*6 + ii + jj*params->nx]
               - (all_cells[params_n_w_host*4 + ii + jj*params->nx]
                  + all_cells[params_n_w_host*7 + ii + jj*params->nx]
                  + all_cells[params_n_w_host*8 + ii + jj*params->nx]))
              / local_density;
        /* compute norm of velocity for cell */
        u = sqrt((u_x * u_x) + (u_y * u_y));
        /* compute pressure for cell */
        pressure = local_density * c_sq;
      }

      /* write to file */
      fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, all_obstacles[ii * params->nx + jj]);
    }
  }

  fclose(fp);
}


void write_av_vels(const t_param* params, float* av_vels) {

  FILE* fp = fopen(AVVELSFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  for (int ii = 0; ii < params->maxIters; ii++) {
    fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
  }

  fclose(fp);
}

void mpi_initialize(int* argc, char*** argv, int *rank, int *size){
    int flag;
    MPI_Init(argc, argv);
    MPI_Initialized(&flag);
    if(flag != true) die("MPI_Initialize failed", __LINE__, __FILE__);

    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, size);
}

void mpi_finalize(){
    int flag;
    MPI_Finalize();
    MPI_Finalized(&flag);
    if(flag != true) die("MPI_Finalize failed", __LINE__, __FILE__);
}

void mpi_create_graph(int my_rank, int size, MPI_Comm* comm){
    int right = (my_rank + 1) % size; 
    int left = (my_rank - 1 < 0) ? (my_rank - 1 + size) : (my_rank - 1);
    int sources[IN_DEGREE] = { left, right };
    int destinations[OUT_DEGREE] = { left, right };
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, IN_DEGREE, sources, MPI_UNWEIGHTED, 
        OUT_DEGREE, destinations, MPI_UNWEIGHTED, MPI_INFO_NULL, true, comm);
}

void die(const char* message, const int line, const char* file)
{
  fprintf(stderr, "Error at line %d of file %s:\n", line, file);
  fprintf(stderr, "%s\n", message);
  fflush(stderr);
  exit(EXIT_FAILURE);
}

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

