#pragma once
#include <math.h>
#include "select_device.h"
#include "structs.h"

cl_device_id setup_opencl(cl_uint platform_i, cl_uint device_i, cl_context* context, cl_command_queue *cmd_queue){

    cl_uint num_platforms, *devices_per_platform;
    cl_device_id **devices;
    get_devices_info(&num_platforms, &devices_per_platform, &devices);

    printf("number_of_platforms[%d]\n", num_platforms);
    for(int i = 0; i < num_platforms; i++){
        printf("platform[%d]-devices_per_platform[%d]\n", i, devices_per_platform[i]);
    }
    if(platform_i >= num_platforms || device_i >= devices_per_platform[platform_i]){
        printf("number_of_platforms[%d]\n", num_platforms);
        for(int i = 0; i < num_platforms; i++){
            printf("platform[%d]-devices_per_platform[%d]\n", i, devices_per_platform[i]);
        }

        char message[UINT16_MAX];
        sprintf(message, "invalid platform[%d] or device[%d]\n", platform_i, device_i);
        die(message, __LINE__, __FILE__);
    }

    cl_device_id current_device = devices[platform_i][device_i];
    print_device_info(current_device);

    cl_int err;
    *context = clCreateContext(0, 1, &current_device, NULL, NULL, &err);
    *cmd_queue = clCreateCommandQueue(*context, current_device, 0, &err);

    return current_device;
}

void make_workgroup_info(cl_device_id device, const mpi_comm_info_t* comm_info, 
        const t_param* params, workgroup_info_t *workgroup_info){

    //workgroup_info->workgroup_rows = comm_info->local_y;
    workgroup_info->workgroup_rows = 1;

    // Chose number of columns to support max workgroup size of system GPU
    // size_t max_workgroup_size;
    // clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_workgroup_size, NULL);
    // size_t max_workgroup_cols = max_workgroup_size / workgroup_info->workgroup_rows;
    // size_t l = 0;
    // while((max_workgroup_cols >> l) > 1) { l++; }
    // size_t max_cols = pow(2, l);
    // if(max_cols > comm_info->local_x) workgroup_info->workgroup_cols = params->nx;
    // else workgroup_info->workgroup_cols = max_cols;
    workgroup_info->workgroup_cols = 128;
    
    workgroup_info->workgroup_elems = workgroup_info->workgroup_rows * workgroup_info->workgroup_cols;
    workgroup_info->num_workgroups = (comm_info->local_y / workgroup_info->workgroup_rows)
                                * (comm_info->local_x / workgroup_info->workgroup_cols);

    printf("workgroup_rows[%d]\n", workgroup_info->workgroup_rows);
    printf("workgroup_cols[%d]\n", workgroup_info->workgroup_cols);
    printf("workgroup_elems[%d]\n", workgroup_info->workgroup_elems);
    printf("num_workgroups[%d]\n", workgroup_info->num_workgroups);
}

// return: a buffer on the device.
cl_mem create_device_buffer(cl_context context, cl_mem_flags flags, size_t buffer_device_size) {
    cl_int err;
    cl_mem m = clCreateBuffer(context, flags, buffer_device_size, NULL, &err);
    check_error(err, "Creating buffer", __FILE__, __LINE__);
    return m;
}

// return: a buffer on the device, populated with the given data from the host.
cl_mem create_device_buffer_from_host(cl_context context, cl_mem_flags flags, size_t buffer_device_size, void *h_buffer) {
    cl_int err;
    //cl_mem m = clCreateBuffer(context, flags, buffer_device_size, NULL, &err);
    cl_mem m = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, buffer_device_size, h_buffer, &err);
    check_error(err, "Creating buffer from host buffer", __FILE__, __LINE__);
    return m;
}

// effect: reads the contents of the file with name source_filename into source.
void readfile(char *filename, char **content) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Could not open file: %s\n", filename);
        exit(1);
    }
    /* Get the number of bytes */
    fseek(file, 0L, SEEK_END);
    int numbytes = ftell(file) + 1;
    /* reset the file position indicator to the beginning of the file */
    fseek(file, 0L, SEEK_SET);
    /* grab sufficient memory for the buffer to hold the text */
    *content = (char*)calloc(numbytes, sizeof(char));
    /* memory error */
    if(content == NULL) {
        printf("Could not allocate memory");
        exit(1);
    }
    /* copy all the text into the buffer */
    fread(*content, sizeof(char), numbytes, file);
    fclose(file);
}

cl_program build_program(char* source_filename, const cl_context context, cl_device_id device){
    cl_int err;
    char *program_source;
    readfile(source_filename, &program_source);
    cl_program program = clCreateProgramWithSource(context, 1, (const char**) &program_source, NULL, &err);
    checkError(err, "Error creating program");
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS){
        size_t build_error_message_len;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &build_error_message_len);
        char* build_error_message = malloc(sizeof(char) * build_error_message_len);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, build_error_message_len, build_error_message, NULL);
        die(build_error_message, __LINE__, __FILE__);
    }
    return program;
}

cl_kernel create_kernel(char *kernel_name, cl_program program) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    check_error(err, "Creating kernel", __FILE__, __LINE__);
    return kernel;
}


void run_accel_kernel(cl_kernel accel_kernel, cl_command_queue commands, const mpi_comm_info_t* comm_info, 
            cl_mem d_comm_info, cl_mem d_params, cl_mem d_cells, cl_mem d_obstacles) {

    cl_int err;
    err  = clSetKernelArg(accel_kernel, 0, sizeof(cl_mem), &d_comm_info);
    err |= clSetKernelArg(accel_kernel, 1, sizeof(cl_mem), &d_params);
    err |= clSetKernelArg(accel_kernel, 2, sizeof(cl_mem), &d_cells);
    err |= clSetKernelArg(accel_kernel, 3, sizeof(cl_mem), &d_obstacles);
    check_error(err, "Setting accel_kernel arguments", __FILE__, __LINE__);

    // Only 1 dimension because only one row is accelerated.
    const int num_dims = 2;
    size_t global[2] = { comm_info->local_y, comm_info->local_x };
    err = clEnqueueNDRangeKernel(commands, accel_kernel, num_dims, NULL, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueueing accelerate kernel", __FILE__, __LINE__);
}

void run_timestep_kernel(cl_kernel timestep_r_kernel, cl_command_queue cmd_queue, mpi_comm_info_t* comm_info,
        cl_mem d_comm_info, cl_mem d_params, const workgroup_info_t* workgroup_info, 
        const cl_mem d_cells, const cl_mem d_final_cells, const cl_mem d_obstacles, 
        const cl_mem d_workgroup_sum_velocities, int itr)
    {

    cl_int err;
    err = clSetKernelArg(timestep_r_kernel, 0, sizeof(int), &itr);
    err |= clSetKernelArg(timestep_r_kernel, 1, sizeof(cl_mem), &d_comm_info);
    err |= clSetKernelArg(timestep_r_kernel, 2, sizeof(cl_mem), &d_params);
    err |= clSetKernelArg(timestep_r_kernel, 3, sizeof(cl_mem), &d_cells);
    err |= clSetKernelArg(timestep_r_kernel, 4, sizeof(cl_mem), &d_final_cells);
    err |= clSetKernelArg(timestep_r_kernel, 5, sizeof(cl_mem), &d_obstacles);
    err |= clSetKernelArg(timestep_r_kernel, 6, sizeof(float) * workgroup_info->workgroup_elems , NULL);
    err |= clSetKernelArg(timestep_r_kernel, 7, sizeof(cl_mem), &d_workgroup_sum_velocities);
    check_error(err, "Setting timestep_r kernel arguments", __FILE__, __LINE__);

    const int num_dims = 2;
    size_t global[2] = { comm_info->local_y, comm_info->local_x };
    size_t local[2] = { workgroup_info->workgroup_rows, workgroup_info->workgroup_cols };
    //printf("workgroup_rows[%d]-workgroup_cols[%d]\n", workgroup_info->workgroup_rows, workgroup_info->workgroup_cols);
    err = clEnqueueNDRangeKernel(cmd_queue, timestep_r_kernel, num_dims, NULL, global, local, 0, NULL, NULL);
    check_error(err, "Enqueueing timestep_r kernel", __FILE__, __LINE__);
}

//Writes h_halos to the halos of d_cells through using d_halos as an intermediate 
void replace_device_halos(cl_kernel write_halos, cl_command_queue cmd_queue, 
    const mpi_comm_info_t *comm_info, cl_mem d_comm_info,
    const cl_mem d_cells, const cl_mem d_halos, const float* h_halos){

    // Write the new halos after propogation to the device
    cl_int err = 0;
    err = clEnqueueWriteBuffer(cmd_queue, d_halos, CL_TRUE, 0, sizeof(float) * 3 * 2 * comm_info->local_x, h_halos, 0, NULL, NULL);
    check_error(err, "Copying h_halos to device at d_halos", __FILE__, __LINE__);

    //Run kernel that writes d_halos to d_cells
    err  = clSetKernelArg(write_halos, 0, sizeof(cl_mem), &d_comm_info);
    err |= clSetKernelArg(write_halos, 1, sizeof(cl_mem), &d_cells);
    err |= clSetKernelArg(write_halos, 2, sizeof(cl_mem), &d_halos);
    check_error(err, "Setting write_halos kernel arguments", __FILE__, __LINE__);

    const int num_dims = 2;
    size_t global[2] = { comm_info->local_y + HALOS, comm_info->local_x };
    //if(comm_info->rank == 0) printf("RANK_0\n");
    err = clEnqueueNDRangeKernel(cmd_queue, write_halos, num_dims, NULL, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueueing write_halos kernel", __FILE__, __LINE__);

}

//Reads the edges of d_cells into h_edges using d_edges as an intermediate
void replace_host_edges(cl_kernel read_edges, cl_command_queue cmd_queue,
        const mpi_comm_info_t* comm_info, cl_mem d_comm_info, 
        const cl_mem d_cells, const cl_mem d_edges, float* h_edges){
    
    cl_int err = 0;
    err  = clSetKernelArg(read_edges, 0, sizeof(cl_mem), &d_comm_info);
    err |= clSetKernelArg(read_edges, 1, sizeof(cl_mem), &d_cells);
    err |= clSetKernelArg(read_edges, 2, sizeof(cl_mem), &d_edges);
    check_error(err, "Setting read_edges kernel arguments", __FILE__, __LINE__);

    const int num_dims = 2;
    size_t global[2] = { comm_info->local_y, comm_info->local_x };
    err = clEnqueueNDRangeKernel(cmd_queue, read_edges, num_dims, NULL, global, NULL, 0, NULL, NULL);
    check_error(err, "Enqueueing read_edges kernel", __FILE__, __LINE__);

    err = clEnqueueReadBuffer(cmd_queue, d_edges, CL_TRUE, 0, sizeof(float) * 3 * 2 * comm_info->local_x, h_edges, 0, NULL, NULL);
    check_error(err, "Copying d_edges from device to h_edges", __FILE__, __LINE__);
}

void read_cells(cl_command_queue cmd_queue, const mpi_comm_info_t* comm_info, const cl_mem d_cells, float* h_cells){
    cl_int err = 0;
    err = clEnqueueReadBuffer(cmd_queue, d_cells, CL_TRUE, 0, sizeof(float) * NSPEEDS * (comm_info->local_y + HALOS) * comm_info->local_x, h_cells, 0, NULL, NULL);
    check_error(err, "Copying d_cells from device to h_cells", __FILE__, __LINE__);
}


void read_av_vels(cl_command_queue cmd_queue, const t_param* params, const workgroup_info_t* workgroup_info, const cl_mem d_workgroup_sum_velocities, float* av_vels){

    cl_int err = 0;
    float* h_workgroup_sum_velocities = calloc(sizeof(float), params->maxIters * workgroup_info->num_workgroups);
    err = clEnqueueReadBuffer(cmd_queue, d_workgroup_sum_velocities, CL_TRUE, 0, sizeof(float) * params->maxIters * workgroup_info->num_workgroups, h_workgroup_sum_velocities, 0, NULL, NULL);
    check_error(err, "Copying d_workgroup_sum_velocities from device to h_workgroup_sum_velocities", __FILE__, __LINE__);

    for(int itr = 0; itr < params->maxIters; itr++){
        float av_vel_itr = 0.0f;
        for(int workgroup_i = 0; workgroup_i < workgroup_info->num_workgroups; workgroup_i++){
            av_vel_itr += h_workgroup_sum_velocities[itr * workgroup_info->num_workgroups + workgroup_i];
        }
        av_vels[itr] = av_vel_itr;
    }
}