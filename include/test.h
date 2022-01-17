#pragma once
#include "hash.h"
#include "utils.h"

cl_int
test_hash(cl_context ctx, cl_command_queue cq, cl_kernel hash_krnl)
{
  const cl_uchar digest[32] = { 78,  237, 113, 65,  234, 74,  92,  212,
                                183, 136, 96,  107, 210, 63,  70,  226,
                                18,  175, 156, 172, 235, 172, 220, 125,
                                31,  76,  109, 199, 242, 81,  27,  152 };

  cl_int status;

  cl_uchar* in = (cl_uchar*)malloc(sizeof(cl_uchar) * 64);
  cl_uchar* out = (cl_uchar*)malloc(sizeof(cl_uchar) * 32);

  // prepare known 64 -bytes input
  static_input(in, 64);

  // compute blake3 hash of 64 -bytes input
  status = hash(ctx, cq, hash_krnl, in, out);

  // compare result !
  for (size_t i = 0; i < 32; i++) {
    assert(*(out + i) == digest[i]);
  }

  // deallocate memory
  free(in);
  free(out);

  return status;
}
