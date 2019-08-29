// ======= CONSTANTS =============================================================================
#define NSPEEDS 9
#define HALOS 2

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
typedef __attribute__ (()) struct
{
  float speeds[NSPEEDS];
} t_speed;

typedef struct {
  int rank;       //rank of current MPI process
  int size;       //group size of intracommunitor
  int local_x;    //cells in x direction for current rank (columns)
  int local_y;    //cells in y direction for current rank (rows)
  int local_n;    //number of cells belonging to current rank
  int local_n_w_halos;    //number of cells belonging to current rank
  int offset_start_y;   //offset in y direction to start of y's chunk
  int offset_end_y;     //offset in y direction to end of y's chunk
  int non_obstacle_num; //number of total fluid particles (non obstacle cells) in nx.ny size grid
} mpi_comm_info_t;


__kernel void accelerate_flow(__global mpi_comm_info_t* comm_info, __global t_param* params, 
                        __global float *restrict cells, __global const int *restrict obstacles){
    
    int j = get_global_id(0) + (HALOS - 1);
    int ii = get_global_id(1);

    if(j == HALOS - 1){
        /* modify the 2nd to last (or one before the last) row of the grid */
        int accel_jj = params->ny - 2;
        if(accel_jj >= comm_info->offset_start_y && accel_jj < comm_info->offset_end_y){
            int jj = accel_jj - comm_info->offset_start_y + (HALOS - 1);

            /* compute weighting factors */
            float w1 = params->w1;
            float w2 = params->w2;

            int previous_row = (jj-1)*comm_info->local_x;
            int current_row = jj*comm_info->local_x;

            /* If the cell is not occupied && we don't sent a negative density */
            if (!obstacles[ii + previous_row]
                && (cells[comm_info->local_n_w_halos*3+ ii + current_row] - w1) > 0.f
                && (cells[comm_info->local_n_w_halos*6+ ii + current_row] - w2) > 0.f
                && (cells[comm_info->local_n_w_halos*7+ ii + current_row] - w2) > 0.f) 
            {
                /* increase 'east-side' densities */
                cells[comm_info->local_n_w_halos*1+ ii + current_row] += w1;
                cells[comm_info->local_n_w_halos*5+ ii + current_row] += w2;
                cells[comm_info->local_n_w_halos*8+ ii + current_row] += w2;
                /* decrease 'west-side' densities */
                cells[comm_info->local_n_w_halos*3+ ii + current_row] -= w1;
                cells[comm_info->local_n_w_halos*6+ ii + current_row] -= w2;
                cells[comm_info->local_n_w_halos*7+ ii + current_row] -= w2;
            }
        }
    }
}


__kernel void timestep_r(int itr, __global mpi_comm_info_t* comm_info, __global t_param* params, 
                    __global float *restrict cells, __global float *restrict final_cells,
                     __global const int *restrict obstacles, __local float* workgroup_cell_velocities,
                     __global float* workgroup_sum_velocities){

    
    int jj = get_global_id(0) + (HALOS - 1);
    int ii = get_global_id(1);
    int group_j = get_group_id(0);
    int group_i = get_group_id(1);
    int num_groups_j = get_num_groups(0);
    int num_groups_i = get_num_groups(1);
    int local_jj = get_local_id(0);
    int local_ii = get_local_id(1);
    int local_height = get_local_size(0);
    int local_width = get_local_size(1);

    int current_row = jj*comm_info->local_x;
    float workitem_cell_velocity = 0.0f;
    t_speed temp_cell;
    //=============== PROPOGATE ============================================================    
    /* determine indices of axis-direction neighbours
    ** respecting periodic boundary conditions (wrap around) */
    int y_n = jj + 1; int y_s = jj - 1;
    int x_e = (ii + 1) % comm_info->local_x;
    int x_w = (ii - 1 < 0) ? (ii - 1 + comm_info->local_x) : (ii - 1);
    /* propagate densities from neighbouring cells, following appropriate 
    directions of travel and writing into scratch space grid */
    temp_cell.speeds[0] = cells[comm_info->local_n_w_halos*0 + ii + jj*comm_info->local_x]; /* central cell, no movement */
    temp_cell.speeds[1] = cells[comm_info->local_n_w_halos*1 + x_w + jj*comm_info->local_x]; /* east */
    temp_cell.speeds[2] = cells[comm_info->local_n_w_halos*2 + ii + y_s*comm_info->local_x]; /* north */
    temp_cell.speeds[3] = cells[comm_info->local_n_w_halos*3 + x_e + jj*comm_info->local_x]; /* west */
    temp_cell.speeds[4] = cells[comm_info->local_n_w_halos*4 + ii + y_n*comm_info->local_x]; /* south */
    temp_cell.speeds[5] = cells[comm_info->local_n_w_halos*5 + x_w + y_s*comm_info->local_x]; /* north-east */
    temp_cell.speeds[6] = cells[comm_info->local_n_w_halos*6 + x_e + y_s*comm_info->local_x]; /* north-west */
    temp_cell.speeds[7] = cells[comm_info->local_n_w_halos*7 + x_e + y_n*comm_info->local_x]; /* south-west */
    temp_cell.speeds[8] = cells[comm_info->local_n_w_halos*8 + x_w + y_n*comm_info->local_x]; /* south-east */

    //=============== REBOUND ============================================================
    if (obstacles[(jj-1)*comm_info->local_x + ii]){
        /* called after propagate, so taking values from scratch space
        ** mirroring, and writing into main grid */
        final_cells[comm_info->local_n_w_halos*1 + ii + current_row] = temp_cell.speeds[3];
        final_cells[comm_info->local_n_w_halos*2 + ii + current_row] = temp_cell.speeds[4];
        final_cells[comm_info->local_n_w_halos*3 + ii + current_row] = temp_cell.speeds[1];
        final_cells[comm_info->local_n_w_halos*4 + ii + current_row] = temp_cell.speeds[2];
        final_cells[comm_info->local_n_w_halos*5 + ii + current_row] = temp_cell.speeds[7];
        final_cells[comm_info->local_n_w_halos*6 + ii + current_row] = temp_cell.speeds[8];
        final_cells[comm_info->local_n_w_halos*7 + ii + current_row] = temp_cell.speeds[5];
        final_cells[comm_info->local_n_w_halos*8 + ii + current_row] = temp_cell.speeds[6];
    }
    //=============== COLLISION ============================================================
    /* don't consider occupied cells */
    else {
        /* compute local density total */
        float local_density = 0.f;
        for (int kk = 0; kk < NSPEEDS; kk++) {
            local_density += temp_cell.speeds[kk];
        }

        /* compute x velocity component */
        float u_x = (temp_cell.speeds[1]
                        + temp_cell.speeds[5]
                        + temp_cell.speeds[8]
                        - (temp_cell.speeds[3]
                            + temp_cell.speeds[6]
                            + temp_cell.speeds[7]))
                        / local_density;
        /* compute y velocity component */
        float u_y = (temp_cell.speeds[2]
                        + temp_cell.speeds[5]
                        + temp_cell.speeds[6]
                        - (temp_cell.speeds[4]
                            + temp_cell.speeds[7]
                            + temp_cell.speeds[8]))
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
        for (int kk = 0; kk < NSPEEDS; kk++)
        {
            final_cells[comm_info->local_n_w_halos*kk + ii + current_row] = temp_cell.speeds[kk]
                                                    + params->omega
                                                    * (d_equ[kk] - temp_cell.speeds[kk]);
        }

        //========== AVERAGE VELOCITIES ============================================
        /* accumulate the norm of x- and y- velocity components */
        workitem_cell_velocity = sqrt(u_sq);
    }

    //Average Velocities
    workgroup_cell_velocities[local_jj * local_width + local_ii] = workitem_cell_velocity;
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_jj == 0 && local_ii == 0){
        float workgroup_sum_velocities_i = 0.0f;
        for(int workgroup_elem = 0; workgroup_elem < local_height * local_width; workgroup_elem++){
            workgroup_sum_velocities_i += workgroup_cell_velocities[workgroup_elem];
        }
        workgroup_sum_velocities[itr*num_groups_j*num_groups_i + group_j*num_groups_i + group_i] = workgroup_sum_velocities_i;
    }
}

__kernel void read_edges_from_device(__global mpi_comm_info_t* comm_info, __global float* d_cells, __global float* d_edges){
    
    int jj = get_global_id(0) + (HALOS - 1);
    int ii = get_global_id(1);

    //Prep to copy edges from device to host
    int top_cells[] = { 6, 2, 5 };
    int bottom_cells[] = { 7, 4, 8 };
    if(jj == HALOS - 1){
        for(int speed=0; speed<3; speed++){
            d_edges[comm_info->local_x*2*speed + ii] = d_cells[comm_info->local_n_w_halos*bottom_cells[speed] + (HALOS - 1) * comm_info->local_x + ii];
        }
    }
    if (jj == comm_info->local_y + (HALOS - 1) - 1){
        for(int speed=0; speed<3; speed++){
            d_edges[comm_info->local_x*2*speed + comm_info->local_x + ii] = d_cells[comm_info->local_n_w_halos*top_cells[speed] + (comm_info->local_y + (HALOS - 1) - 1) * comm_info->local_x + ii];
        }
    }
}


__kernel void write_halos_to_device(__global mpi_comm_info_t* comm_info, __global float* d_cells, __global float* d_halos){

    int jj = get_global_id(0);
    int ii = get_global_id(1);

    int top_cells[] = { 6, 2, 5 };
    int bottom_cells[] = { 7, 4, 8 };
    if(jj == 0){
        for(int speed = 0; speed < 3; speed++){
            d_cells[comm_info->local_n_w_halos*top_cells[speed] + ii] = d_halos[comm_info->local_x*2*speed + ii];
        }
    }
    if(jj == comm_info->local_y + HALOS - 1){
        for(int speed = 0; speed < 3; speed++){
            d_cells[comm_info->local_n_w_halos*bottom_cells[speed] + (comm_info->local_y + HALOS - 1) * comm_info->local_x + ii] = d_halos[comm_info->local_x*2*speed + comm_info->local_x + ii];
        }
    }
}



