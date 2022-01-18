#pragma once
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This function is expected to test OpenCL kernel `hash`, when
// `ocl_kernel_flag_0` compilation flags are passed
cl_int
hash_0(cl_context ctx,
       cl_command_queue cq,
       cl_kernel krnl,
       const cl_uchar* input,
       cl_uchar* const output)
{
  cl_int status;

  const size_t i_size = 64 * sizeof(cl_uchar);
  const size_t o_size = 32 * sizeof(cl_uchar);

  // input is supplied to kernel by this buffer
  cl_mem i_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, i_size, NULL, &status);

  // output to be placed here, after kernel completes hash computation
  cl_mem o_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, o_size, NULL, &status);

  // setting up kernel arguments
  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &i_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &o_buf);

  // input being copied to device memory
  cl_event evt_0;
  status = clEnqueueWriteBuffer(
    cq, i_buf, CL_FALSE, 0, i_size, input, 0, NULL, &evt_0);

  // preparing for creating dependency in compute execution graph
  cl_event evts[1] = { evt_0 };

  // setting up work-item count; it's like `sycl::single_task( ... )`
  size_t global_size[1] = { 1 };
  size_t local_size[1] = { 1 };

  // kernel being dispatched for execution on device
  cl_event evt_1;
  status = clEnqueueNDRangeKernel(
    cq, krnl, 1, NULL, global_size, local_size, 1, evts, &evt_1);

  // hash output being copied back to host
  cl_event evt_2;
  status = clEnqueueReadBuffer(
    cq, o_buf, CL_FALSE, 0, o_size, output, 1, &evt_1, &evt_2);

  // host synchronization
  status = clWaitForEvents(1, &evt_2);

  // resource deallocation phase
  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);

  clReleaseMemObject(i_buf);
  clReleaseMemObject(o_buf);

  return status;
}

// This function is expected to test OpenCL kernel `hash`, when
// `ocl_kernel_flag_1` compilation flags are passed
cl_int
hash_1(cl_context ctx,
       cl_command_queue cq,
       cl_kernel krnl,
       const cl_uint* input,
       cl_uint* const output)
{
  cl_int status;

  const size_t i_size = 16 * sizeof(cl_uint);
  const size_t o_size = 8 * sizeof(cl_uint);

  // input is supplied to kernel by this buffer
  cl_mem i_buf = clCreateBuffer(ctx, CL_MEM_READ_ONLY, i_size, NULL, &status);

  // output to be placed here, after kernel completes hash computation
  cl_mem o_buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, o_size, NULL, &status);

  // setting up kernel arguments
  status = clSetKernelArg(krnl, 0, sizeof(cl_mem), &i_buf);
  status = clSetKernelArg(krnl, 1, sizeof(cl_mem), &o_buf);

  // input being copied to device memory
  cl_event evt_0;
  status = clEnqueueWriteBuffer(
    cq, i_buf, CL_FALSE, 0, i_size, input, 0, NULL, &evt_0);

  // preparing for creating dependency in compute execution graph
  cl_event evts[1] = { evt_0 };

  // setting up work-item count; it's like `sycl::single_task( ... )`
  size_t global_size[1] = { 1 };
  size_t local_size[1] = { 1 };

  // kernel being dispatched for execution on device
  cl_event evt_1;
  status = clEnqueueNDRangeKernel(
    cq, krnl, 1, NULL, global_size, local_size, 1, evts, &evt_1);

  // hash output being copied back to host
  cl_event evt_2;
  status = clEnqueueReadBuffer(
    cq, o_buf, CL_FALSE, 0, o_size, output, 1, &evt_1, &evt_2);

  // host synchronization
  status = clWaitForEvents(1, &evt_2);

  // resource deallocation phase
  clReleaseEvent(evt_0);
  clReleaseEvent(evt_1);
  clReleaseEvent(evt_2);

  clReleaseMemObject(i_buf);
  clReleaseMemObject(o_buf);

  return status;
}
