#pragma once

// Taken from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500d/utils.c

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define check_for_error_and_return(status)                                     \
  if (status != CL_SUCCESS) {                                                  \
    return status;                                                             \
  }
#define check_for_error_and_continue(status)                                   \
  if (status != CL_SUCCESS) {                                                  \
    continue;                                                                  \
  }

cl_int
find_device(cl_device_id* device_id)
{
  // just reset all bytes for safety !
  memset(device_id, 0, sizeof(cl_device_id));

  cl_int status;

  cl_uint num_platforms;
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  check_for_error_and_return(status);

  if (num_platforms == 0) {
    return CL_DEVICE_NOT_FOUND;
  }

  cl_platform_id* platforms =
    (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms, NULL);
  check_for_error_and_return(status);

  // preferred device is either CPU, GPU
  cl_device_type dev_type = CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU;

  for (cl_uint i = 0; i < num_platforms; i++) {
    cl_uint num_devices;
    status = clGetDeviceIDs(*(platforms + i), dev_type, 0, NULL, &num_devices);
    check_for_error_and_continue(status);

    if (num_devices == 0) {
      continue;
    }

    cl_device_id* devices =
      (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
    status =
      clGetDeviceIDs(*(platforms + i), dev_type, num_devices, devices, NULL);
    if (status != CL_SUCCESS) {
      free(devices);
      continue;
    }

    *device_id = *devices;
    free(devices);
    return CL_SUCCESS;
  }

  free(platforms);

  return CL_DEVICE_NOT_FOUND;
}

cl_int
build_kernel(cl_context ctx,
             cl_device_id dev_id,
             char* kernel,
             cl_program* prgm)
{
  cl_int status;

  FILE* fd = fopen(kernel, "r");
  fseek(fd, 0, SEEK_END);
  const size_t size = ftell(fd);
  fseek(fd, 0, SEEK_SET);

  char* kernel_src = (char*)malloc(sizeof(char) * size);
  size_t n = fread(kernel_src, sizeof(char), size, fd);

  assert(n == size);
  fclose(fd);

  cl_program prgm_ = clCreateProgramWithSource(
    ctx, 1, (const char**)&kernel_src, &size, &status);
  check_for_error_and_return(status);

  *prgm = prgm_;
  free(kernel_src);

  status = clBuildProgram(*prgm, 1, &dev_id, "-cl-std=CL2.0 -w", NULL, NULL);
  if (status != CL_SUCCESS) {
    return status;
  }

  return CL_SUCCESS;
}

cl_int
show_build_log(cl_device_id dev_id, cl_program prgm)
{
  cl_int status;

  size_t log_size;
  status = clGetProgramBuildInfo(
    prgm, dev_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (status != CL_SUCCESS) {
    return status;
  }

  void* log = malloc(log_size);
  status = clGetProgramBuildInfo(
    prgm, dev_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
  if (status != CL_SUCCESS) {
    free(log);
    return status;
  }

  printf("\nkernel build log:\n%s\n\n", (char*)log);
  free(log);

  return CL_SUCCESS;
}

void
random_input(cl_uchar* in, size_t count)
{
  for (size_t i = 0; i < count; i++) {
    *(in + i) = (cl_uchar)rand();
  }
}
