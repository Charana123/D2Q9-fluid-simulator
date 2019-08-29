#include "d2q9-bgk.h"

// main program: initialise, timestep loop, finalise
int main(int argc, char* argv[])
{
  char *param_file, *obstacle_file;    /* name of the input parameter and input obstacle file */
  float *cells, *final_cells; /* grid containing fluid densities and temporary grids*/
  int* obstacles = NULL;    /* grid indicating which cells are blocked */
  float* av_vels   = NULL;     /* a record of the av. velocity computed for each timestep */

  //============================= PARSE ARGS ====================================
  parse_args(&param_file, &obstacle_file, argc, argv);

  //============================= MPI INITIALIZE ====================================
  // MPI Initialize
  mpi_comm_info_t comm_info; t_param params; MPI_Comm graph_comm; MPI_Datatype halo_exchange_type;
  mpi_initialize(&argc, &argv, &comm_info.rank, &comm_info.size);
  initialise(&comm_info, param_file, obstacle_file, &params, &cells, &final_cells, &obstacles, &av_vels);
  create_and_commit_halo_exchange_type(&comm_info, &halo_exchange_type);

  //============================= OpenCL INITIALIZE ====================================
  cl_command_queue cmd_queue; cl_context context;
  cl_device_id device;
  if(comm_info.rank % 2 == 0)  device = setup_opencl(0, 1, &context, &cmd_queue);
  else device = setup_opencl(0, 1, &context, &cmd_queue);
  workgroup_info_t workgroup_info;
  make_workgroup_info(device, &comm_info, &params, &workgroup_info);
  // Setup device buffers
  float *h_edges = malloc(sizeof(float) * 3 * 2 * comm_info.local_x);
  float *h_halos = malloc(sizeof(float) * 3 * 2 * comm_info.local_x);
  cl_mem d_params = create_device_buffer_from_host(context, CL_MEM_READ_ONLY, sizeof(t_param), &params);
  cl_mem d_comm_info = create_device_buffer_from_host(context, CL_MEM_READ_ONLY, sizeof(mpi_comm_info_t), &comm_info);
  cl_mem d_cells = create_device_buffer_from_host(context, CL_MEM_READ_WRITE, sizeof(float) * NSPEEDS * comm_info.local_n_w_halos, cells);
  cl_mem d_final_cells = create_device_buffer(context, CL_MEM_READ_WRITE, sizeof(float) * NSPEEDS * comm_info.local_n_w_halos);
  cl_mem d_obstacles = create_device_buffer_from_host(context, CL_MEM_READ_ONLY, sizeof(int) * comm_info.local_n, obstacles);
  cl_mem d_workgroup_sum_velocities = create_device_buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * params.maxIters * workgroup_info.num_workgroups);
  cl_mem d_edges = create_device_buffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * 3 * 2 * comm_info.local_x);
  cl_mem d_halos = create_device_buffer(context, CL_MEM_READ_ONLY, sizeof(float) * 3 * 2 * comm_info.local_x);
  // Create & Build program and select appropriate kernel(s)
  cl_program program = build_program("d2q9-bgk.cl", context, device);
  cl_kernel accel_kernel = create_kernel("accelerate_flow", program);
  cl_kernel timestep_r_kernel = create_kernel("timestep_r", program);
  cl_kernel read_edges_kernel = create_kernel("read_edges_from_device", program);
  cl_kernel write_halos_kernel = create_kernel("write_halos_to_device", program);

  //============================= START TIMING ====================================
  double start_time, end_time, delta_time, user_time, sys_time; /* elapsed wallclock, user CPU and system CPU time */
  start_time = get_time_sec();

  
  //============================= RUN ====================================
  for (int itr = 0; itr < params.maxIters; itr++){
    run_accel_kernel(accel_kernel, cmd_queue, &comm_info, d_comm_info, d_params, d_cells, d_obstacles);
    replace_host_edges(read_edges_kernel, cmd_queue, &comm_info, d_comm_info, d_cells, d_edges, h_edges);
    halo_exchange(&comm_info, &graph_comm, &halo_exchange_type, h_edges, h_halos);
    replace_device_halos(write_halos_kernel, cmd_queue, &comm_info, d_comm_info, d_cells, d_halos, h_halos);
    run_timestep_kernel(timestep_r_kernel, cmd_queue, &comm_info, d_comm_info, d_params,
       &workgroup_info, d_cells, d_final_cells, d_obstacles, d_workgroup_sum_velocities, itr);
    swap_ptrs(&d_cells, &d_final_cells);
  }
  read_cells(cmd_queue, &comm_info, d_cells, cells);
  read_av_vels(cmd_queue, &params, &workgroup_info, d_workgroup_sum_velocities, av_vels);
  av_vels_reduce(&comm_info, &params, av_vels);

  //============================= END TIMING ====================================
  end_time = get_time_sec();
  delta_time = end_time - start_time;
  get_cpu_time(&user_time, &sys_time);

  //============================= WRITE RESULTS =================================
  output_results(&comm_info, &params, cells, obstacles, av_vels, sys_time, user_time, delta_time);
  gather_and_write_results(&comm_info, &params, NULL, cells, obstacles, av_vels);

  //============================= CLEAN UP ========================================
  finalise(&params, &cells, &final_cells, &obstacles, &av_vels);
  mpi_finalize();

  return EXIT_SUCCESS;
}

void parse_args(char **param_file, char** obstacle_file, const int argc, char** const argv)
{
  if (argc != 3) usage(argv[0]);
  else {
    *param_file = argv[1];
    *obstacle_file = argv[2];
  }
}

double get_time_sec()
{
  struct timeval timstr;        /* structure to hold elapsed time */
  gettimeofday(&timstr, NULL);
  return timstr.tv_sec + (timstr.tv_usec / 1000000.0);
}

void get_cpu_time(double* user_time, double* sys_time)
{
  struct timeval timstr;
  gettimeofday(&timstr, NULL);

  struct rusage ru;             /* structure to hold CPU time--system and user */
  getrusage(RUSAGE_SELF, &ru);
  timstr = ru.ru_utime;
  *user_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
  timstr = ru.ru_stime;
  *sys_time = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
}

void output_results(const mpi_comm_info_t *comm_info, const t_param* params, float* cells, int* obstacles, float* av_vels, const float sys_time, const float user_time, const float delta_time)
{
  if(comm_info->rank == MASTER){
    printf("==done==\n");
    printf("size:%dx%d\n", params->ny, params->nx);
    //printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, cells, obstacles, av_vels[params->maxIters-1]));
    printf("Elapsed time:\t\t\t%.6lf (s)\n", delta_time);
    printf("Elapsed user CPU time:\t\t%.6lf (s)\n", user_time);
    printf("Elapsed system CPU time:\t%.6lf (s)\n", sys_time);
  }
}

void swap_ptrs(cl_mem* A, cl_mem* B)
{
  cl_mem temp;
  temp = *A;
  *A = *B;
  *B = temp;
}

void av_vels_reduce(const mpi_comm_info_t* comm_info, const t_param* params, float* av_vels)
{
  if(comm_info->rank == 0) MPI_Reduce(MPI_IN_PLACE, av_vels, params->maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
  else MPI_Reduce(av_vels, av_vels, params->maxIters, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

  for(int i = 0; i < params->maxIters; i++){
    av_vels[i] = av_vels[i] / comm_info->non_obstacle_num;
  }
}

void halo_exchange(const mpi_comm_info_t* comm_info, const MPI_Comm* comm, const MPI_Datatype* halo_exchange_type, float* h_edges, float* h_halos)
{
  for(int speed = 0; speed < 3; speed++){
    int tag = 0;
    float* top_edge = comm_info->local_x*2*speed + h_edges;
    float* bottom_edge = comm_info->local_x*2*speed + comm_info->local_x + h_edges;
    float* top_halo = comm_info->local_x*2*speed + h_halos;
    float* bottom_halo = comm_info->local_x*2*speed + comm_info->local_x + h_halos;
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

void create_and_commit_halo_exchange_type(const mpi_comm_info_t* comm_info, MPI_Datatype* halo_exchange_type){
  MPI_Type_vector(3, comm_info->local_x, comm_info->local_x * 2, MPI_FLOAT, halo_exchange_type);
  MPI_Type_commit(halo_exchange_type);
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
  params->w1 = params->density * params->accel / 9.f;
  params->w2 = params->density * params->accel / 36.f;

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
  comm_info->local_n = comm_info->local_y * comm_info->local_x;
  comm_info->local_n_w_halos = (comm_info->local_y + HALOS) * comm_info->local_x;
  printf("comm_info->local_n[%d]\n", comm_info->local_n);
  printf("comm_info->local_n_w_halos[%d]\n", comm_info->local_n_w_halos);

  /* main grid */
  *cells_ptr = malloc(sizeof(float) * NSPEEDS * (comm_info->local_y + HALOS) * comm_info->local_x);
  if (*cells_ptr == NULL) die("cannot allocate memory for cells", __LINE__, __FILE__);

  /* 'helper' grid, used as scratch space */
  *final_cells_ptr = malloc(sizeof(float) * NSPEEDS * (comm_info->local_y + HALOS) * comm_info->local_x);
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
      (*cells_ptr)[comm_info->local_n_w_halos * 0 + ii + jj*comm_info->local_x] = w0;
      /* axis directions */
      (*cells_ptr)[comm_info->local_n_w_halos * 1 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos * 2 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos * 3 + ii + jj*comm_info->local_x] = w1;
      (*cells_ptr)[comm_info->local_n_w_halos * 4 + ii + jj*comm_info->local_x] = w1;
      /* diagonals */
      (*cells_ptr)[comm_info->local_n_w_halos * 5 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos * 6 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos * 7 + ii + jj*comm_info->local_x] = w2;
      (*cells_ptr)[comm_info->local_n_w_halos * 8 + ii + jj*comm_info->local_x] = w2;
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
  free(*cells_ptr);
  *cells_ptr = NULL;

  free(*final_cells_ptr);
  *final_cells_ptr = NULL;

  free(*obstacles_ptr);
  *obstacles_ptr = NULL;

  free(*av_vels_ptr);
  *av_vels_ptr = NULL;

  return EXIT_SUCCESS;
}


// float calc_reynolds(const t_param* params, float* cells, int* obstacles, float final_av_vel)
// {
//   const float viscosity = 1.f / 6.f * (2.f / params->omega - 1.f);
//   return final_av_vel * params->reynolds_dim / viscosity;
// }

// float calc_average_velocity(t_speed cell){
  
//   float local_density = 0.f;
//   for (int kk = 0; kk < NSPEEDS; kk++) {
//     local_density += cell.speeds[kk];
//   }

//   /* x-component of velocity */
//   float u_x = (cell.speeds[1]
//               + cell.speeds[5]
//               + cell.speeds[8]
//               - (cell.speeds[3]
//                   + cell.speeds[6]
//                   + cell.speeds[7]));
  
//   //printf("u_x[%f]\n", u_x);         
//   u_x = u_x / local_density;
//   //printf("u_x[%f]\n", u_x);
//   /* compute y velocity component */
//   float u_y = (cell.speeds[2]
//               + cell.speeds[5]
//               + cell.speeds[6]
//               - (cell.speeds[4]
//                   + cell.speeds[7]
//                   + cell.speeds[8]));

//   //printf("u_y[%f]\n", u_y);
//   u_y = u_y / local_density;
//   //printf("u_y[%f]\n", u_y);
//   /* accumulate the norm of x- and y- velocity components */
  
//   float speed = sqrt((u_x * u_x) + (u_y * u_y));
//   //printf("speed[%f]\n", speed);
//   return speed;
// }

// float total_density(const mpi_comm_info_t* comm_info, const t_param params, float* cells)
// {
//   float total = 0.f;  /* accumulator */

//   for (int jj = HALOS - 1; jj < comm_info->local_y + HALOS - 1; jj++)
//   {
//     for (int ii = 0; ii < comm_info->local_x; ii++)
//     {
//       for (int kk = 0; kk < NSPEEDS; kk++)
//       {
//         total += cells[comm_info->local_n*kk + ii + jj*params.nx];
//       }
//     }
//   }

//   return total;
// }

void gather_and_write_results(const mpi_comm_info_t* comm_info, const t_param* params, const MPI_Datatype *cell_type, const float* cells, const int* obstacles, const float* av_vels)
{
  /* Gather operation on invidual sub-domains (sub-cells and sub-obstacles) */
  float* all_cells = NULL; int* all_obstacles = NULL;
  if(comm_info->rank == MASTER) all_cells = malloc(sizeof(float) * NSPEEDS * params->nx * params->ny);
  if(comm_info->rank == MASTER) all_obstacles = malloc(sizeof(int) * params->nx * params->ny);
  gather_local_cells_and_obstacles(comm_info, params, cell_type, cells, all_cells, obstacles, all_obstacles); //collective
  
  /* Master writes them into files */
  if(comm_info->rank == MASTER) {
    write_state(params, all_obstacles, all_cells);
    write_av_vels(params, av_vels);
  }
  /* Clean up */
  free(all_cells); free(all_obstacles);
}

void gather_local_cells_and_obstacles(const mpi_comm_info_t* comm_info, const t_param* params, const MPI_Datatype* cell_type, const float* cells, float* all_cells, const int* obstacles, int* all_obstacles){

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
    MPI_Gatherv(cells + comm_info->local_n_w_halos * speed + comm_info->local_x, 
      recvcounts[comm_info->rank], MPI_FLOAT, all_cells + (params->nx * params->ny) * speed, recvcounts, displs, MPI_FLOAT, 0, MPI_COMM_WORLD);
  }

  MPI_Gatherv(obstacles, recvcounts[comm_info->rank], MPI_INT,
     all_obstacles, recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
}

void write_state(const t_param* params, const int* all_obstacles, const float* all_cells){
  
  const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
  FILE* fp = fopen(FINALSTATEFILE, "w");

  if (fp == NULL) {
    die("could not open file output file", __LINE__, __FILE__);
  }

  int params_local_n = params->ny * params->nx;
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
          local_density += all_cells[params_local_n*kk + ii + jj*params->nx];
        }

        /* compute x velocity component for cell */
        u_x = (all_cells[params_local_n*1 + ii + jj*params->nx]
               + all_cells[params_local_n*5 + ii + jj*params->nx]
               + all_cells[params_local_n*8 + ii + jj*params->nx]
               - (all_cells[params_local_n*3 + ii + jj*params->nx]
                  + all_cells[params_local_n*6 + ii + jj*params->nx]
                  + all_cells[params_local_n*7 + ii + jj*params->nx]))
              / local_density;
        /* compute y velocity component for cell */
        u_y = (all_cells[params_local_n*2 + ii + jj*params->nx]
               + all_cells[params_local_n*5 + ii + jj*params->nx]
               + all_cells[params_local_n*6 + ii + jj*params->nx]
               - (all_cells[params_local_n*4 + ii + jj*params->nx]
                  + all_cells[params_local_n*7 + ii + jj*params->nx]
                  + all_cells[params_local_n*8 + ii + jj*params->nx]))
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


void write_av_vels(const t_param* params, const float* av_vels) {

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

void usage(const char* exe)
{
  fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
  exit(EXIT_FAILURE);
}

