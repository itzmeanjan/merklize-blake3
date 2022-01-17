#pragma once

// Taken from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500d/utils.c

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char ocl_kernel_flag_0[] = "-cl-std=CL2.0 -w\
  -DLE_BYTES_TO_WORDS -DWORDS_TO_LE_BYTES -DEXPOSE_BLAKE3_HASH";
const char ocl_kernel_flag_1[] = "-cl-std=CL2.0 -w -DEXPOSE_BLAKE3_HASH";

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
             const char* kernel,
             const char* flags,
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

  status = clBuildProgram(*prgm, 1, &dev_id, flags, NULL, NULL);
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

// compile time known input pattern, used for testing
void
static_input_0(cl_uchar* const in, size_t count)
{
#pragma unroll
  for (size_t i = 0; i < count; i++) {
    *(in + i) = (cl_uchar)(i % 256);
  }
}

// compile time known input pattern, used for testing
//
// note this produces same input byte array as `static_input_0`
// written above, with just an exception of 4 consequtive little
// endian bytes being interpreted as `cl_uint` --- 32 -bit unsigned
// integer
void
static_input_1(cl_uint* const in, size_t count)
{
#pragma unroll
  for (size_t i = 0; i < count; i++) {
    size_t i_ = i * 4;

    *(in + i) = ((cl_uint)(i_ + 3) << 24) | ((cl_uint)(i_ + 2) << 16) |
                ((cl_uint)(i_ + 1) << 8) | ((cl_uint)(i_ + 0) << 0);
  }
}

void
words_from_le_bytes(const cl_uchar* input,
                    size_t i_size,
                    cl_uint* const msg_words,
                    size_t m_cnt)
{
  // because each message word is of 4 -bytes width
  assert(i_size == m_cnt * 4);

  for (size_t i = 0; i < m_cnt; i++) {
    const cl_uchar* i_start = input + i * 4;

    *(msg_words + i) =
      ((cl_uint) * (i_start + 3) << 24) | ((cl_uint) * (i_start + 2) << 16) |
      ((cl_uint) * (i_start + 1) << 8) | ((cl_uint) * (i_start + 0) << 0);
  }
}

void
words_to_le_bytes(const cl_uint* msg_words,
                  size_t m_cnt,
                  cl_uchar* const output,
                  size_t o_size)
{
  // because each message word is of 4 -bytes width
  assert(o_size == m_cnt * 4);

  for (size_t i = 0; i < m_cnt; i++) {
    const cl_uint num = *(msg_words + i);
    cl_uchar* out = output + i * 4;

    *(out + 0) = (cl_uchar)(num >> 0) & 0xff;
    *(out + 1) = (cl_uchar)(num >> 8) & 0xff;
    *(out + 2) = (cl_uchar)(num >> 16) & 0xff;
    *(out + 3) = (cl_uchar)(num >> 24) & 0xff;
  }
}
