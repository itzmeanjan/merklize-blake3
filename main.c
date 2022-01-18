// Adapted from
// https://github.com/itzmeanjan/vectorized-rescue-prime/blob/614500d/main.c

#include "bench.h"
#include "test.h"

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
  check_mem_alloc(dev_name);

  status = clGetDeviceInfo(dev_id, CL_DEVICE_NAME, val_size, dev_name, NULL);
  show_message_and_exit(status, "failed to get device name !\n");

  printf("running on %s\n", (char*)dev_name);

  cl_context ctx = clCreateContext(NULL, 1, &dev_id, NULL, NULL, &status);
  show_message_and_exit(status, "failed to create context !\n");

  // enable profiling in queue, to get (precise) kernel execution time
  cl_queue_properties props[] = { CL_QUEUE_PROPERTIES,
                                  CL_QUEUE_PROFILING_ENABLE |
                                    CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                  0 };
  cl_command_queue c_queue =
    clCreateCommandQueueWithProperties(ctx, dev_id, props, &status);
  show_message_and_exit(status, "failed to create command queue !\n");

  cl_program prgm_0;
  status = build_kernel(ctx, dev_id, "kernel.cl", ocl_kernel_flag_0, &prgm_0);
  if (status != CL_SUCCESS) {
    printf("failed to compile kernel !\n");

    show_build_log(dev_id, prgm_0);
    return EXIT_FAILURE;
  }

  cl_program prgm_1;
  status = build_kernel(ctx, dev_id, "kernel.cl", ocl_kernel_flag_1, &prgm_1);
  if (status != CL_SUCCESS) {
    printf("failed to compile kernel !\n");

    show_build_log(dev_id, prgm_1);
    return EXIT_FAILURE;
  }

  cl_program prgm_2;
  status = build_kernel(ctx, dev_id, "kernel.cl", ocl_kernel_flag_2, &prgm_2);
  if (status != CL_SUCCESS) {
    printf("failed to compile kernel !\n");

    show_build_log(dev_id, prgm_2);
    return EXIT_FAILURE;
  }

  cl_kernel krnl_0 = clCreateKernel(prgm_0, "hash", &status);
  show_message_and_exit(status, "failed to create `hash` kernel !\n");

  cl_kernel krnl_1 = clCreateKernel(prgm_1, "hash", &status);
  show_message_and_exit(status, "failed to create `hash` kernel !\n");

  cl_kernel krnl_2 = clCreateKernel(prgm_2, "merklize", &status);
  show_message_and_exit(status, "failed to create `merklize` kernel !\n");

  status = test_hash_0(ctx, c_queue, krnl_0);
  status = test_hash_1(ctx, c_queue, krnl_1);

  printf("\npassed blake3 hash test !\n");
  printf("\nBenchmarking Binary Merklization using BLAKE3\n\n");

  const size_t wg_size = 1 << 5;

  for (size_t i = 20; i <= 25; i++) {
    cl_ulong ts = 0;
    size_t leaf_count = 1 << i;

    status = bench_merklize(ctx, c_queue, krnl_2, leaf_count, wg_size, &ts);
    printf("merklize\t\t2 ^ %2zu leaves\t\tin %16.4lf ms\n", i, (double)ts * 1e-6);
  }

  clReleaseKernel(krnl_0);
  clReleaseKernel(krnl_1);
  clReleaseKernel(krnl_2);
  clReleaseProgram(prgm_0);
  clReleaseProgram(prgm_1);
  clReleaseProgram(prgm_2);
  clReleaseCommandQueue(c_queue);
  clReleaseContext(ctx);
  clReleaseDevice(dev_id);

  free(dev_name);

  return EXIT_SUCCESS;
}
