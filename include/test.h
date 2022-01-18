#pragma once
#include "hash.h"
#include "utils.h"

// Tests hash_0( ... ) i.e. when opencl kernel `hash` is compiled
// with flags `ocl_kernel_flag_0`
cl_int
test_hash_0(cl_context ctx, cl_command_queue cq, cl_kernel hash_krnl)
{
  const cl_uchar digest[32] = { 78,  237, 113, 65,  234, 74,  92,  212,
                                183, 136, 96,  107, 210, 63,  70,  226,
                                18,  175, 156, 172, 235, 172, 220, 125,
                                31,  76,  109, 199, 242, 81,  27,  152 };

  cl_int status;

  // allocate memory on heap
  cl_uchar* in = (cl_uchar*)malloc(sizeof(cl_uchar) * 64);
  check_mem_alloc(in);
  cl_uchar* out = (cl_uchar*)malloc(sizeof(cl_uchar) * 32);
  check_mem_alloc(out);

  // prepare 64 -bytes (known) input
  static_input_0(in, 64);
  status = hash_0(ctx, cq, hash_krnl, in, out);

  // compare result !
  for (size_t i = 0; i < 32; i++) {
    assert(*(out + i) == digest[i]);
  }

  // deallocate memory
  free(in);
  free(out);

  return status;
}

// Tests hash_1( ... ) i.e. when opencl kernel `hash` is compiled
// with flags `ocl_kernel_flag_1`
cl_int
test_hash_1(cl_context ctx, cl_command_queue cq, cl_kernel hash_krnl)
{
  const cl_uint digest[8] = { 1097985358, 3562818282, 1801488567, 3796254674,
                              2895949586, 2111614187, 3345828895, 2551927282 };

  cl_int status;

  // allocate memory on heap
  cl_uchar* i_bytes = (cl_uchar*)malloc(sizeof(cl_uchar) * 64);
  check_mem_alloc(i_bytes);
  cl_uint* in = (cl_uint*)malloc(sizeof(cl_uint) * 16);
  check_mem_alloc(in);
  cl_uint* out = (cl_uint*)malloc(sizeof(cl_uint) * 8);
  check_mem_alloc(out);

  // prepare 64 -bytes (known) input
  static_input_0(i_bytes, 64);
  // interpreting little endian bytes as `uint`
  words_from_le_bytes(i_bytes, 64, in, 16);
  status = hash_1(ctx, cq, hash_krnl, in, out);

  // compare result !
  for (size_t i = 0; i < 8; i++) {
    assert(*(out + i) == digest[i]);
  }

  // deallocate memory
  free(in);
  free(i_bytes);
  free(out);

  return status;
}
