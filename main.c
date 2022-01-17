// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500d/main.c

#include "utils.h"

#define show_message_and_exit(status, msg)                                     \
  if (status != CL_SUCCESS) {                                                  \
    printf(msg);                                                               \
    return EXIT_FAILURE;                                                       \
  }

int
main(int argc, char** argv)
{
  cl_int status;

  cl_device_id dev_id;
  status = find_device(&dev_id);
  show_message_and_exit(status, "failed to find device !\n");

  size_t val_size;
  status = clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 0, NULL, &val_size);
  show_message_and_exit(status, "failed to get device name !\n");

  void* dev_name = malloc(val_size);
  status = clGetDeviceInfo(dev_id, CL_DEVICE_NAME, val_size, dev_name, NULL);
  show_message_and_exit(status, "failed to get device name !\n");

  printf("running on %s\n", (char*)dev_name);

  cl_context ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);
  show_message_and_exit(status, "failed to create context !\n");

  // enable profiling in queue, to get (precise) kernel execution time
  cl_queue_properties props[] = { CL_QUEUE_PROPERTIES,
                                  CL_QUEUE_PROFILING_ENABLE,
                                  0 };
  cl_command_queue c_queue =
    clCreateCommandQueueWithProperties(ctx, dev_id, props, &status);
  show_message_and_exit(status, "failed to create command queue !\n");

  cl_program prgm;
  status = build_kernel(ctx, dev_id, "kernel.cl", &prgm);
  if (status != CL_SUCCESS) {
    printf("failed to compile kernel !\n");

    show_build_log(dev_id, prgm);
    return EXIT_FAILURE;
  }

  status = show_build_log(dev_id, prgm);
  show_message_and_exit(status, "failed to obtain kernel build log !\n");

  cl_kernel krnl_0 = clCreateKernel(prgm, "hash", &status);
  show_message_and_exit(status, "failed to create `hash` kernel !\n");

  clReleaseKernel(krnl_0);
  clReleaseProgram(prgm);
  clReleaseCommandQueue(c_queue);
  clReleaseContext(ctx);
  clReleaseDevice(dev_id);

  free(dev_name);

  return EXIT_SUCCESS;
}
