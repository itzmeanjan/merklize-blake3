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

  const size_t i_size = leaf_count << 5;
  const size_t o_size = leaf_count << 5;

  // allocate input/ output memory on host
  cl_uchar* in = (cl_uchar*)malloc(i_size);
  check_mem_alloc(in);
  // all intermediate nodes of merkle tree to be
  // placed on this host allocation
  cl_uchar* out = (cl_uchar*)malloc(o_size);
  check_mem_alloc(out);

  // generate random input bytes i.e. leaf nodes of binary merkle tree
  random_input(in, leaf_count << 5);

  // merklize leaf nodes
  status = merklize(
    ctx, cq, merklize_krnl, in, i_size, leaf_count, out, o_size, wg_size, ts);

  // deallocate memory
  free(in);
  free(out);

  return status;
}
