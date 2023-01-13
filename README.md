> **Warning** I've stopped maintaining this project !

# merklize-blake3
OpenCL powered Merklization using BLAKE3

## Motivation

Few weeks ago, when I completed writing Binary Merklization Kernel using Rescue Prime Hash function for performing 2-to-1 hashing, I also decided to take a look at much faster BLAKE3 hash; using it for 2-to-1 hashing, constructing all intermediate nodes of Binary Merkle Tree, when N -many leaf nodes are provided, where `N = 2 ^ i | i = {1, 2, 3 ...}`

And here I'm today.

I've implemented Binary Merklization logic using BLAKE3 as 2-to-1 hash function for computing intermediate node when two of its immediate children are provided. Output of BLAKE3 hash function is 32 -bytes. Meaning, if N -leaf nodes are to be used for constructing Merkle Tree, I've `N << 5` -bytes input to work with. Each OpenCL work-item will take 64 -bytes input i.e. two contiguous (leaf/ intermediate) nodes of Merkle Tree on some level concatenated, where each node of Merkle Tree is represented using 32 -bytes ( output of BLAKE3 digest ). Each work-item will compress 64 -bytes input message and produce 32 -bytes output digest, which is an intermediate node. Note, in BLAKE3 input is splitted into chunks ( each of 1024 -bytes width ) and each chunk is splitted into 16 blocks each of 64 -bytes width. That means, here in BLAKE3 2-to-1 hashing case, I'll be working with only one input chunk, which has single block of width 64 -bytes.

2-to-1 BLAKE3 hash function naturally lends itself well to parallelization efforts using OpenCL vector intrinsics. I suggest you read BLAKE3 [specification](https://github.com/BLAKE3-team/BLAKE3-specs/blob/ac78a717924dd9e6f16f547baa916c6f71470b1a/blake3.pdf)'s section 5.3 for understanding how SIMD benefits BLAKE3 implementation here.

If you happen to be interested in my 2-to-1 Rescue Prime hash based Binary Merklization implementation, consider taking a look [here](https://github.com/itzmeanjan/vectorized-rescue-prime).

During this implementation I took my ideas from my previous `blake3` implementation in SYCL, which is majorly targeted towards a usecase where you've relatively large input byte array ( say 1GB ) & you want to just compute single 32 -bytes digest, as fast as possible. See that implementation [here](https://github.com/itzmeanjan/blake3).

I've always taken help from BLAKE3 reference [implementation](https://github.com/BLAKE3-team/BLAKE3/blob/da4c792d8094f35c05c41c9aeb5dfe4aa67ca1ac/reference_impl/reference_impl.rs).

## Prerequisite(s)

- I'm using GNU/ Linux machine

```bash
$ lsb_release -d

Description:    Ubuntu 20.04.3 LTS
```

- As compiler I'm using `clang-14`

```bash
clang --version
clang version 14.0.0 (https://github.com/intel/llvm 04c18f2f5697bb334b4cae288254fafccc1f5b81)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /home/ubuntu/sycl_workspace/llvm/build/bin
```

- For ease of building source code, I use `make` utility
- For formatting source files, I use `clang-format` with Mozilla code format style
- You must have both OpenCL ICD & OpenCl development headers installed. You might want to [see](https://github.com/kenba/cl3/blob/78f04cb2d55fd313816daeb9d0bb33ea1820cb91/docs/opencl_installation.md).
- To quickly verify list of accessible OpenCL accelerators, consider using `clinfo`

```bash
sudo apt-get install clinfo
```

- I wanted to check what's supported OpenCL C version on my development machine, so I executed following command on bash

```bash
$ clinfo | grep -i 'device opencl c'

  Device OpenCL C Version                         OpenCL C 1.2
  Device OpenCL C Version                         OpenCL C 3.0
```

Two outputs because this machine has two platforms

```bash
$ clinfo -l

Platform #0: Intel(R) FPGA Emulation Platform for OpenCL(TM)
 `-- Device #0: Intel(R) FPGA Emulation Device
Platform #1: Intel(R) OpenCL
 `-- Device #0: Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz
```

## Usage

This is a header-only library, you can pretty easily use it in your project by just adding `./include` to your project's INCLUDE_PATH. You'll probably be interested in `./include/merklize.h` file, which has host portion of Binary Merklization implementation. Only `./kernel.cl` contains device executable code.

I suggest you go through code, for getting a sense of how to use this implementation. I've tried accompanying source with some comments, for helping you with that.

> Note, this implementation is only helpful when you've relatively large number of leaf nodes and you want to quickly compute all intermediate nodes of Binary Merkle Tree using BLAKE3 2-to-1 hashing.

> Just to enforce aforementioned fact, I've also put one check that # -of leaf nodes of Merkle Tree is at least 2 ^ 20.

## Benchmark(s)

For benchmarking OpenCL accelerated Binary Merklization implementation using 2-to-1 BLAKE3 hashing, I set up random input byte array of size {32 MB, 64 MB, ... 1GB}, which are interpreted as contiguous blocks of 32 little endian bytes, making {2 ^ 20, 2 ^ 21, ... 2 ^ 25} -many leaf nodes of Binary Merkle Tree. Now, with multiple kernel dispatch rounds all (N - 1) -many intermediate nodes of Binary Merkle Tree with N -many leaf nodes are computed, which are then interpreted as little endian bytes making `cl_uint` -> `cl_uchar[4]`.

In following results, I show how much time was spent on executing kernels ( because multiple rounds of them are dispatched ), how much time spent on transferring input bytes to device & also how long does it take to transfer back intermediate node bytes from device to host, all seperately. Their real life execution may overlap, because data dependency graph allows that & I've explicitly enabled **OUT_OF_ORDER** execution of commands.

Command used for running benchmark

```bash
make    # build executable, with JIT compiling kernel
./run   # execute
```

---

Note, it's possible to AOT compile kernel to **I**ntermediate **R**epresentation language ( read spirv ), using following command

```bash
make aot && ./run
```

As effect of successful execution of above command, 6 new IR files should be created.

```bash
$ file kernel_*

kernel_0.bc:  LLVM IR bitcode
kernel_0.spv: Khronos SPIR-V binary, little-endian, version 0x00010000, generator 0x0006000e
kernel_1.bc:  LLVM IR bitcode
kernel_1.spv: Khronos SPIR-V binary, little-endian, version 0x00010000, generator 0x0006000e
kernel_2.bc:  LLVM IR bitcode
kernel_2.spv: Khronos SPIR-V binary, little-endian, version 0x00010000, generator 0x0006000e
```

Among these 6 files three of them are LLVM bitcodes of compile time preprocessed variants of same `kernel.cl`, and other 3 are respective spirv 64 -bit IL forms of these LLVM IRs. These spirv64 files are of our interest, as they'll be consumed at runtime for constructing OpenCL program objects, which are then compiled using backend compiler, generating device specific binaries, before computation can be offloaded to accelerators.

> [This](https://github.com/KhronosGroup/OpenCL-Guide/blob/4079d208cdc73cf060a8bf7a03cd4ea44d199d64/chapters/os_tooling.md) may be useful guide for understanding various tools available for offline compilation of OpenCL kernels.

> Remove intermediate object files using `make clean`

## On Nvidia GPU

```bash
running on Tesla V100-SXM2-16GB

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in           3.7582 ms          with host to device data tx in           5.9588 ms              while device to host data tx took          25.7536 ms
merklized 2 ^ 21 leaves in           7.2841 ms          with host to device data tx in          12.4451 ms              while device to host data tx took          52.9835 ms
merklized 2 ^ 22 leaves in          14.2924 ms          with host to device data tx in          19.1883 ms              while device to host data tx took         102.1858 ms
merklized 2 ^ 23 leaves in          28.2952 ms          with host to device data tx in          35.7116 ms              while device to host data tx took         201.9709 ms
merklized 2 ^ 24 leaves in          52.5327 ms          with host to device data tx in          71.5775 ms              while device to host data tx took         405.8741 ms
merklized 2 ^ 25 leaves in         100.6062 ms          with host to device data tx in         143.8628 ms              while device to host data tx took         809.7079 ms
```

## On Intel CPU(s)

```bash
# number of available CPU(s)

$ lscpu | grep -i cpu\(s\) | head -1

CPU(s):                          128

---------------------------------------------------------------------

running on Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          59.3219 ms          with host to device data tx in           5.3638 ms              while device to host data tx took           3.4559 ms
merklized 2 ^ 21 leaves in          86.8682 ms          with host to device data tx in           9.6732 ms              while device to host data tx took           6.1324 ms
merklized 2 ^ 22 leaves in          50.9794 ms          with host to device data tx in          22.8543 ms              while device to host data tx took           5.5853 ms
merklized 2 ^ 23 leaves in          70.0997 ms          with host to device data tx in          33.7137 ms              while device to host data tx took          10.5920 ms
merklized 2 ^ 24 leaves in          98.0322 ms          with host to device data tx in          45.2110 ms              while device to host data tx took          20.6102 ms
merklized 2 ^ 25 leaves in         175.9160 ms          with host to device data tx in          64.6548 ms              while device to host data tx took          40.7421 ms
```

---

```bash
# number of available CPU(s)

$ lscpu | grep -i cpu\(s\) | head -1

CPU(s):                          24

---------------------------------------------------------------------

running on Intel(R) Core(TM) i9-10920X CPU @ 3.50GHz

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          19.3894 ms          with host to device data tx in           3.5966 ms              while device to host data tx took           2.7474 ms
merklized 2 ^ 21 leaves in          28.1566 ms          with host to device data tx in           6.8970 ms              while device to host data tx took           5.5447 ms
merklized 2 ^ 22 leaves in          40.1784 ms          with host to device data tx in          14.3582 ms              while device to host data tx took          11.0112 ms
merklized 2 ^ 23 leaves in          68.7388 ms          with host to device data tx in          25.1962 ms              while device to host data tx took          21.8632 ms
merklized 2 ^ 24 leaves in         134.6519 ms          with host to device data tx in          46.9249 ms              while device to host data tx took          43.4311 ms
merklized 2 ^ 25 leaves in         270.3394 ms          with host to device data tx in          87.3723 ms              while device to host data tx took          84.2431 ms
```

---

```bash
# number of available CPU(s)

$ lscpu | grep -i cpu\(s\) | head -1

CPU(s):                          24

---------------------------------------------------------------------

running on Intel(R) Xeon(R) Gold 6128 CPU @ 3.40GHz

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          23.2566 ms          with host to device data tx in           4.6335 ms              while device to host data tx took           2.3862 ms
merklized 2 ^ 21 leaves in          32.3493 ms          with host to device data tx in           8.9893 ms              while device to host data tx took           4.0934 ms
merklized 2 ^ 22 leaves in          48.7371 ms          with host to device data tx in          16.6954 ms              while device to host data tx took           7.9074 ms
merklized 2 ^ 23 leaves in          83.0458 ms          with host to device data tx in          27.9030 ms              while device to host data tx took          15.5969 ms
merklized 2 ^ 24 leaves in         156.7341 ms          with host to device data tx in          44.2649 ms              while device to host data tx took          31.1593 ms
merklized 2 ^ 25 leaves in         310.1746 ms          with host to device data tx in          75.0253 ms              while device to host data tx took          62.2082 ms
```

---

```bash
# number of available CPU(s)

$ lscpu | grep -i cpu\(s\) | head -1

CPU(s):                          12

---------------------------------------------------------------------

running on Intel(R) Xeon(R) E-2176G CPU @ 3.70GHz

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          13.5114 ms          with host to device data tx in          11.7123 ms              while device to host data tx took           2.3868 ms
merklized 2 ^ 21 leaves in          26.0523 ms          with host to device data tx in           5.3010 ms              while device to host data tx took           4.6691 ms
merklized 2 ^ 22 leaves in          50.5436 ms          with host to device data tx in           9.8895 ms              while device to host data tx took           9.2759 ms
merklized 2 ^ 23 leaves in          98.6148 ms          with host to device data tx in          19.0315 ms              while device to host data tx took          18.4084 ms
merklized 2 ^ 24 leaves in         196.5320 ms          with host to device data tx in          37.5133 ms              while device to host data tx took          36.7274 ms
merklized 2 ^ 25 leaves in         393.0303 ms          with host to device data tx in          74.4886 ms              while device to host data tx took          72.5261 ms
```

---

```bash
# number of available CPU(s)

$ lscpu | grep -i cpu\(s\) | head -1

CPU(s):                          4

---------------------------------------------------------------------

running on Intel(R) Xeon(R) CPU E5-2686 v4 @ 2.30GHz

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          63.8268 ms          with host to device data tx in           9.5565 ms              while device to host data tx took           9.7178 ms
merklized 2 ^ 21 leaves in         136.0081 ms          with host to device data tx in          18.2696 ms              while device to host data tx took          18.3540 ms
merklized 2 ^ 22 leaves in         263.3239 ms          with host to device data tx in          36.0957 ms              while device to host data tx took          36.8454 ms
merklized 2 ^ 23 leaves in         541.0337 ms          with host to device data tx in          73.5482 ms              while device to host data tx took          75.0540 ms
merklized 2 ^ 24 leaves in        1092.4356 ms          with host to device data tx in         146.4404 ms              while device to host data tx took         148.4563 ms
merklized 2 ^ 25 leaves in        2129.8265 ms          with host to device data tx in         292.8271 ms              while device to host data tx took         305.3257 ms
```

## On Intel GPU(s)

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in           9.1614 ms          with host to device data tx in           8.9756 ms              while device to host data tx took           5.9159 ms
merklized 2 ^ 21 leaves in          18.1686 ms          with host to device data tx in          17.8592 ms              while device to host data tx took          11.8276 ms
merklized 2 ^ 22 leaves in          36.1911 ms          with host to device data tx in          35.6105 ms              while device to host data tx took          23.6581 ms
merklized 2 ^ 23 leaves in          72.2722 ms          with host to device data tx in          71.1157 ms              while device to host data tx took          47.3124 ms
merklized 2 ^ 24 leaves in         144.4273 ms          with host to device data tx in         142.0968 ms              while device to host data tx took          94.6267 ms
merklized 2 ^ 25 leaves in         288.7912 ms          with host to device data tx in         284.0003 ms              while device to host data tx took         189.2569 ms
```

---

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklized 2 ^ 20 leaves in          55.6401 ms          with host to device data tx in           4.1337 ms              while device to host data tx took          11.5352 ms
merklized 2 ^ 21 leaves in          93.9523 ms          with host to device data tx in           8.1561 ms              while device to host data tx took           4.5567 ms
merklized 2 ^ 22 leaves in         122.8529 ms          with host to device data tx in          16.5308 ms              while device to host data tx took           7.9646 ms
merklized 2 ^ 23 leaves in         185.1349 ms          with host to device data tx in          32.7902 ms              while device to host data tx took          15.7615 ms
merklized 2 ^ 24 leaves in         301.7459 ms          with host to device data tx in          63.4848 ms              while device to host data tx took          31.5628 ms
merklized 2 ^ 25 leaves in         559.1572 ms          with host to device data tx in         111.6804 ms              while device to host data tx took          61.9394 ms
```
