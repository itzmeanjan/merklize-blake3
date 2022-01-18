#pragma once
#include "merklize.h"

// Benchmarks execution of `merklize` kernel on accelerator, with given input
// size & work-group size for ndrange kernel dispatch
//
// Excution time is in nanosecond level granularity
cl_int
bench_merklize(cl_context ctx,
               cl_command_queue cq,
               cl_kernel merklize_krnl,
               size_t leaf_count,
               size_t wg_size,
               cl_ulong* const ts)
{
  // only use this merklization implementation when relatively large number of
  // leaf nodes are to be processed
  assert(leaf_count >= 1 << 20);

  cl_int status;

  // allocate input/ output memory on host
  cl_uchar* in = (cl_uchar*)malloc(sizeof(cl_uchar) * leaf_count << 5);
  check_mem_alloc(in);
  cl_uchar* out = (cl_uchar*)malloc(sizeof(cl_uchar) * 32);
  check_mem_alloc(out);

  // generate random input bytes
  random_input(in, leaf_count << 5);

  // merklize leaf nodes
  status = merklize(
    ctx, cq, merklize_krnl, in, leaf_count << 5, leaf_count, out, wg_size, ts);

  // deallocate memory
  free(in);
  free(out);

  return status;
}
