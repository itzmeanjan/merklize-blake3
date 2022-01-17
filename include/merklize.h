#pragma once
#include "utils.h"
#include <math.h>

cl_int
merklize(cl_context ctx,
         cl_command_queue cq,
         cl_kernel krnl,
         const cl_uchar* input,
         size_t i_size, // in bytes
         size_t leaf_count,
         cl_uchar* const output,
         size_t wg_size)
{
  // because each leaf node Merkle Tree will be of width 32 -bytes
  assert(leaf_count << 5 == i_size);

  // power of 2 many leaf nodes in binary merkle tree
  assert((leaf_count & (leaf_count - 1)) == 0);
  assert((wg_size & (wg_size - 1)) == 0);

  // for small trees using this implementation doesn't benefit much !
  assert(leaf_count >= 1 << 20);

  // so that each work group has equal number of work-items
  assert((leaf_count >> 1) >= wg_size);
  assert((leaf_count >> 1) % wg_size == 0);

  cl_int status;

  // converting 4 contiguous little endian bytes to `cl_uint`
  const size_t i_buf_elm_cnt = i_size >> 2;
  const size_t itmd_buf_elm_cnt = i_size >> 2;
  const size_t itmd_buf_size = itmd_buf_elm_cnt << 2; // in bytes

  cl_uint* i_buf_ptr = (cl_uint*)malloc(i_size);
  words_from_le_bytes(input, i_size, i_buf_ptr, i_buf_elm_cnt);

  cl_uint* itmd_buf_ptr = (cl_uint*)malloc(sizeof(cl_uint) * 8);

  const size_t i_offset = 0;
  const size_t itmd_offset = itmd_buf_elm_cnt >> 1;

  cl_mem i_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, i_size, NULL, &status);
  cl_mem itmd_buf =
    clCreateBuffer(ctx, CL_MEM_READ_WRITE, itmd_buf_size, NULL, &status);
  cl_mem i_offset_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);
  cl_mem itmd_offset_buf =
    clCreateBuffer(ctx, CL_MEM_READ_ONLY, sizeof(size_t), NULL, &status);

  cl_event evt_0;
  clEnqueueWriteBuffer(
    cq, i_buf, CL_FALSE, 0, i_size, i_buf_ptr, 0, NULL, &evt_0);

  cl_event evt_1;
  clEnqueueWriteBuffer(
    cq, i_offset_buf, CL_FALSE, 0, sizeof(size_t), &i_offset, 0, NULL, &evt_1);

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

  clSetKernelArg(krnl, 0, sizeof(cl_mem), &i_buf);
  clSetKernelArg(krnl, 1, sizeof(cl_mem), &i_offset_buf);
  clSetKernelArg(krnl, 2, sizeof(cl_mem), &itmd_buf);
  clSetKernelArg(krnl, 3, sizeof(cl_mem), &itmd_offset_buf);

  size_t glb_work_items[] = { leaf_count >> 1 };
  size_t loc_work_items[] = { wg_size };
  cl_event evts_0[] = { evt_0, evt_1, evt_2 };

  cl_event evt_3;
  clEnqueueNDRangeKernel(
    cq, krnl, 1, NULL, glb_work_items, loc_work_items, 3, evts_0, &evt_3);

  const size_t rounds = (size_t)log2((double)(leaf_count >> 1));

  // these are events obtained as result of enqueuing kernels
  // computing intermediate nodes of binary merkle tree
  cl_event* round_evts = (cl_event*)malloc(sizeof(cl_event) * (rounds + 1));
  *(round_evts + 0) = evt_3;

  // these events are obtained as a result of enqueuing buffer writing
  // commands
  cl_event* tmp_evts = (cl_event*)malloc(sizeof(cl_event) * (rounds << 1));

  // this allocation holds those temporary buffers which convey
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
    // can be released at end of this function
    *(tmp_evts + (r << 1) + 0) = evt_0_;
    *(tmp_evts + (r << 1) + 1) = evt_1_;

    *(tmp_bufs + (r << 1) + 0) = i_offset_buf_;
    *(tmp_bufs + (r << 1) + 1) = itmd_offset_buf_;
  }

  // 32 -bytes root of merkle tree being copied back to host
  cl_event evt_4;
  clEnqueueReadBuffer(cq,
                      itmd_buf,
                      CL_FALSE,
                      32,
                      sizeof(cl_uint) * 8,
                      itmd_buf_ptr,
                      1,
                      round_evts + rounds,
                      &evt_4);

  // let compute dependency chain finish its execution
  clWaitForEvents(1, &evt_4);

  // root of merkle tree being interpreted as little endian
  // byte array
  words_to_le_bytes(itmd_buf_ptr, 8, output, 32);

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

  free(i_buf_ptr);
  free(itmd_buf_ptr);
  free(round_evts);
  free(tmp_evts);
  free(tmp_bufs);

  return status;
}
