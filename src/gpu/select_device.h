#pragma once
#include "error.h"
#include <stdlib.h>

void get_devices_info(cl_uint* num_platforms, cl_uint** devices_per_platform, cl_device_id*** devices)
{
    // Get number of platforms
    cl_int err;
    err = clGetPlatformIDs(0, NULL, num_platforms);
    checkError(err, "Gettings number of platforms\n");
    if(*num_platforms == 0){
        fprintf(stderr, "Found 0 platforms!\n");
        exit(EXIT_FAILURE);
    }

    // Get platforms IDs
    cl_platform_id* platforms = (cl_platform_id*) malloc(*num_platforms * sizeof(cl_platform_id));
    err = clGetPlatformIDs(*num_platforms, platforms, NULL);
    checkError(err, "Getting platforms IDs\n");

    // Get number of devices for each platform
    *devices_per_platform = (cl_uint *) malloc(*num_platforms * sizeof(cl_uint));
    for(int platform_i = 0; platform_i < *num_platforms; platform_i++){
        err = clGetDeviceIDs(platforms[platform_i], CL_DEVICE_TYPE_ALL, 0, NULL, (*devices_per_platform) + platform_i);
        char message[UINT16_MAX];
        sprintf(message, "Getting number of devices for platform[%d]\n", platform_i);
        checkError(err, message);
    }

    //Get device ids for each platform
    *devices = (cl_device_id **) malloc(*num_platforms * sizeof(cl_device_id*));
    for(int platform_i = 0; platform_i < *num_platforms; platform_i++){
        (*devices)[platform_i] = (cl_device_id*) malloc((*devices_per_platform)[platform_i] * sizeof(cl_device_id));
        err = clGetDeviceIDs(platforms[platform_i], CL_DEVICE_TYPE_ALL, (*devices_per_platform)[platform_i], (*devices)[platform_i], NULL);
        char message[UINT16_MAX];
        sprintf(message, "Getting device_ids for platform[%d]\n", platform_i);
        checkError(err, message);
    }
}

void print_device_info(cl_device_id device)
{
    cl_int err = 0;
    char string[UINT16_MAX]; cl_ulong ulong = 0; 

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(string), string, NULL);
    checkError(err, "Getting device name");
    printf("Device-Name[%s]\n", string);

    err = clGetDeviceInfo(device, CL_DEVICE_OPENCL_C_VERSION, sizeof(string), string, NULL);
    checkError(err, "Getting device OpenCL C version");
    printf("OpenCL-Version[%s]\n", string);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_ulong), &ulong, NULL);
    checkError(err, "Getting device max compute units");
    printf("OpenCL-Version[%llu]\n", ulong);

    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(cl_ulong), &ulong, NULL);
    checkError(err, "Getting device max work-group size");
    printf("Workgroup-Max-Items[%llu]\n", ulong);

    // Find the maximum dimensions of the work-groups
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_ulong), &ulong, NULL);
    checkError(err, "Getting device max workgroup dims");
    printf("Workgroup-Max-Dims[%llu]\n", ulong);

    // Get the max. dimensions of the work-groups
    cl_ulong dims = ulong;
    cl_ulong *dim_sizes = (cl_ulong*) malloc(sizeof(cl_ulong) * dims);
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(cl_ulong) * dims, dim_sizes, NULL);
    checkError(err, "Getting device max workgroup sizes");
    for(size_t k = 0; k < dims; k++) printf("dim[%zu]-max_size[%llu]\n", k, dim_sizes[k]);

}