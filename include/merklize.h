#pragma once
#include "utils.h"
#include <math.h>

// Given a N -many leaf nodes of some binary merkle tree, this function
// constructs all intermediate nodes of tree, including root of merkle tree
//
// Expects to get access to OpenCL queue which has enabled out of order
// execution of dispatched kernels
//
// This function also need to time execution of commands using OpenCL event
// profiling, which calls for profiling enabled queue
cl_int
merklize(cl_context ctx,
         cl_command_queue cq,
         cl_kernel krnl,
         const cl_uchar* input,
         size_t i_size, // in bytes
         size_t leaf_count,
         cl_uchar* const output,
         size_t o_size, // in bytes
         size_t wg_size,
         cl_ulong* const ts)
{
  // binary merkle tree with N leaf nodes is input, where N = 2 ^ i
  // there will be (N - 1) intermediate nodes, to be computed in this function
  //
  // for storing all these intermediate nodes, passing them back to caller
  // as return value, allocation size of output is same as input
  assert(i_size == o_size);

  // because each leaf node of Merkle Tree will be of width 32 -bytes
  assert(leaf_count << 5 == i_size);

  // power of 2 many leaf nodes in binary merkle tree
  assert((leaf_count & (leaf_count - 1)) == 0);
  // initially requested work group size should also be
  // power of 2
  assert((wg_size & (wg_size - 1)) == 0);

  // for small trees using this implementation doesn't benefit much !
  assert(leaf_count >= 1 << 20);

  // so that each work group has equal number of work-items
  assert((leaf_count >> 1) >= wg_size);
  assert((leaf_count >> 1) % wg_size == 0);

  cl_int status;

  // notice they are initialized !
  cl_ulong exec_tm = 0; // total kernel execution time
  cl_ulong h2d_tm = 0;  // total time spent in moving data from host to device
  cl_ulong d2h_tm = 0;  // total time spent in moving data to host from device

  // converting 4 contiguous little endian bytes to `cl_uint`
  const size_t i_buf_elm_cnt = i_size >> 2;
  const size_t itmd_buf_elm_cnt = i_size >> 2;
  const size_t itmd_buf_size = itmd_buf_elm_cnt << 2; // in bytes

  // input byte array to be stored as `cl_uint *` on host, which will
  // be later explicitly transferred to device
  cl_uint* i_buf_ptr = (cl_uint*)malloc(i_size);

  // converting large input byte array into `uint *`, where each resulting
  // `uint` is obtained by intepreting four contiguous bytes in little endian
  // order
  words_from_le_bytes(input, i_size, i_buf_ptr, i_buf_elm_cnt);

  // allocating enough memory on heap for storing all intermediate nodes of
  // merkle tree
  cl_uint* itmd_buf_ptr = (cl_uint*)malloc(itmd_buf_size);

  const size_t i_offset = 0;
  const size_t itmd_offset = itmd_buf_elm_cnt >> 1;

  // input leaf nodes to be transferred to this buffer, allocated on device
  cl_mem i_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, i_size, NULL, &status);
  // all intermediate nodes of merkle tree to be kept in this buffer, allocated
  // on device
  //
  // note, r/ w flag mentioned on this buffer, because in certain kernel
  // dispatch rounds I'll pass this same buffer as both input & output to
  // `merklize` kernel
  cl_mem itmd_buf =
    clCreateBuffer(ctx, CL_MEM_READ_WRITE, itmd_buf_size, NULL, &status);
  // following two buffers allocated on device to be kept in constant memory
  cl_mem i_offset_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
  cl_mem itmd_offset_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);

  // transfering input bytes ( actually it's `uint *` ) to device
  cl_event evt_0;
  clEnqueueWriteBuffer(
    cq, i_buf, CL_FALSE, 0, i_size, i_buf_ptr, 0, NULL, &evt_0);

  // transferring constant i.e. offset to input buffer, to device's constant
  // memory space
  cl_event evt_1;
  clEnqueueWriteBuffer(
    cq, i_offset_buf, CL_FALSE, 0, sizeof(size_t), &i_offset, 0, NULL, &evt_1);

  // transferring constant i.e. offset to intermediate nodes buffer, to device's
  // constant memory space
  cl_event evt_2;
  clEnqueueWriteBuffer(cq,
                       itmd_offset_buf,
                       CL_FALSE,
                       0,
                       sizeof(size_t),
                       &itmd_offset,
                       0,
                       NULL,
                       &evt_2);

  // preparing kernel dispatch, setting kernel arguments ( `merklize` kernel )
  clSetKernelArg(krnl, 0, sizeof(cl_mem), &i_buf);
  clSetKernelArg(krnl, 1, sizeof(cl_mem), &i_offset_buf);
  clSetKernelArg(krnl, 2, sizeof(cl_mem), &itmd_buf);
  clSetKernelArg(krnl, 3, sizeof(cl_mem), &itmd_offset_buf);

  size_t glb_work_items[] = { leaf_count >> 1 };
  size_t loc_work_items[] = { wg_size };
  cl_event evts_0[] = { evt_0, evt_1, evt_2 };

  // dispatching kernel for computing intermediate nodes living just above
  // merkle tree leaf nodes ( which are provided as input )
  cl_event evt_3;
  clEnqueueNDRangeKernel(
    cq, krnl, 1, NULL, glb_work_items, loc_work_items, 3, evts_0, &evt_3);

  // these many rounds of kernel dispatches are still required for computing
  // whole merkle tree
  const size_t rounds = (size_t)log2((double)(leaf_count >> 1));

  // these are events obtained as result of enqueuing kernel execution commands,
  // computing intermediate nodes of binary merkle tree
  //
  // these events will be used later, when all computation is done, for finding
  // out total execution time of kernel
  cl_event* round_evts = (cl_event*)malloc(sizeof(cl_event) * (rounds + 1));
  *(round_evts + 0) = evt_3;

  // these events are obtained as a result of enqueuing buffer writing
  // commands, inside body of following for-loop construct
  cl_event* tmp_evts = (cl_event*)malloc(sizeof(cl_event) * (rounds << 1));

  // these allocations hold those constant buffers which convey
  // input/ output buffer offset to kernel, executing on device
  cl_mem* tmp_bufs = (cl_mem*)malloc(sizeof(cl_mem) * (rounds << 1));

  for (size_t r = 0; r < rounds; r++) {
    const size_t i_offset_ = itmd_offset >> r;
    const size_t itmd_offset_ = itmd_offset >> (r + 1);

    cl_mem i_offset_buf_ =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
    cl_mem itmd_offset_buf_ =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);

    cl_event evt_0_;
    clEnqueueWriteBuffer(cq,
                         i_offset_buf_,
                         CL_FALSE,
                         0,
                         sizeof(size_t),
                         &i_offset_,
                         0,
                         NULL,
                         &evt_0_);

    cl_event evt_1_;
    clEnqueueWriteBuffer(cq,
                         itmd_offset_buf_,
                         CL_FALSE,
                         0,
                         sizeof(size_t),
                         &itmd_offset_,
                         0,
                         NULL,
                         &evt_1_);

    clSetKernelArg(krnl, 0, sizeof(cl_mem), &itmd_buf);
    clSetKernelArg(krnl, 1, sizeof(cl_mem), &i_offset_buf_);
    clSetKernelArg(krnl, 2, sizeof(cl_mem), &itmd_buf);
    clSetKernelArg(krnl, 3, sizeof(cl_mem), &itmd_offset_buf_);

    size_t glb_work_items[] = { (leaf_count >> 1) >> (r + 1) };
    size_t loc_work_items[] = { glb_work_items[0] >= wg_size
                                  ? wg_size
                                  : glb_work_items[0] };
    cl_event evts_0_[] = { evt_0_, evt_1_, round_evts[r] };

    cl_event evt_2_;
    clEnqueueNDRangeKernel(
      cq, krnl, 1, NULL, glb_work_items, loc_work_items, 3, evts_0_, &evt_2_);

    // so that compute dependency graph can be constructed
    *(round_evts + r + 1) = evt_2_;

    // so that these data transfer related event resources
    // can be released at end of computation
    *(tmp_evts + (r << 1) + 0) = evt_0_;
    *(tmp_evts + (r << 1) + 1) = evt_1_;

    *(tmp_bufs + (r << 1) + 0) = i_offset_buf_;
    *(tmp_bufs + (r << 1) + 1) = itmd_offset_buf_;
  }

  // all intermediate nodes of merkle tree being copied back to host
  cl_event evt_4;
  clEnqueueReadBuffer(cq,
                      itmd_buf,
                      CL_FALSE,
                      0,
                      itmd_buf_size,
                      itmd_buf_ptr,
                      1,
                      round_evts + rounds,
                      &evt_4);

  // let compute dependency chain finish its execution
  clWaitForEvents(1, &evt_4);

  // all intermediate nodes of merkle tree being interpreted
  // as little endian byte array, as input was provided, output being
  // converted to similar representation
  words_to_le_bytes(itmd_buf_ptr, itmd_buf_elm_cnt, output, o_size);

  // just a temporary variable for holding a specific opencl command execution
  // time
  cl_ulong tmp;

  // sum of execution time of kernels with nanosecond level of granularity
  for (size_t i = 0; i < rounds + 1; i++) {
    tmp = 0;
    status = time_event(*(round_evts + i), &tmp);

    exec_tm += tmp;
  }

  // calculating sum of time ( in nanosecond level granularity ) spent
  // transferring data from host to device
  tmp = 0;
  status = time_event(evt_0, &tmp); // input leaf nodes transfer cost
  h2d_tm += tmp;

  tmp = 0;
  status = time_event(evt_1, &tmp); // constant denoting buffer offset tx cost
  h2d_tm += tmp;

  tmp = 0;
  status = time_event(evt_2, &tmp); // constant denoting buffer offset tx cost
  h2d_tm += tmp;

  for (size_t i = 0; i < rounds << 1; i++) {
    cl_ulong tmp = 0;
    // during multiple rounds of kernel dispatch
    // constants needs to be set for denoting offset into
    // buffer from where input can be read or
    // output can be written to
    status = time_event(*(tmp_evts + i), &tmp);

    h2d_tm += tmp;
  }

  // calculating total device to host data transfer cost
  //
  // this is the only time when device to host data transfer is required !
  tmp = 0;
  status = time_event(evt_4, &tmp); // intermediate node tx cost
  d2h_tm += tmp;

  *(ts + 0) = exec_tm; // sum of kernel execution times
  *(ts + 1) = h2d_tm;  // sum of host to device data tx time
  *(ts + 2) = d2h_tm;  // sum of device to host data tx time

  // release all opencl resources acquired during course of execution of this
  // function
  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);
  clReleaseEvent(evt_4);

  for (size_t i = 0; i < rounds + 1; i++) {
    clReleaseEvent(*(round_evts + i));
  }

  for (size_t i = 0; i < (rounds << 1); i++) {
    clReleaseEvent(*(tmp_evts + i));
  }

  clReleaseMemObject(i_buf);
  clReleaseMemObject(itmd_buf);
  clReleaseMemObject(i_offset_buf);
  clReleaseMemObject(itmd_offset_buf);

  for (size_t i = 0; i < (rounds << 1); i++) {
    clReleaseMemObject(*(tmp_bufs + i));
  }

  // release all heap allocation
  free(i_buf_ptr);
  free(itmd_buf_ptr);
  free(round_evts);
  free(tmp_evts);
  free(tmp_bufs);

  return CL_SUCCESS;
}
