#pragma once
#include "merklize.h"

cl_int
bench_merklize(cl_context ctx,
               cl_command_queue cq,
               cl_kernel merklize_krnl,
               size_t leaf_count,
               size_t wg_size,
               cl_ulong* const ts)
{
  assert(leaf_count >= 1 << 20);

  cl_int status;

  cl_uchar* in = (cl_uchar*)malloc(sizeof(cl_uchar) * leaf_count << 5);
  check_mem_alloc(in);
  cl_uchar* out = (cl_uchar*)malloc(sizeof(cl_uchar) * 32);
  check_mem_alloc(out);

  // generate random input data
  random_input(in, leaf_count << 5);

  status = merklize(
    ctx, cq, merklize_krnl, in, leaf_count << 5, leaf_count, out, wg_size, ts);

  // deallocate memory
  free(in);
  free(out);

  return status;
}
