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

## Usage

This is a header-only library, you can pretty easily use it in your project by just adding `./include` to your project's INCLUDE_PATH. You'll probably be interested in `./include/merklize.h` file, which has host portion of Binary Merklization implementation. Only `./kernel.cl` contains device executable code.

I suggest you go through code, for getting a sense of how to use this implementation. I've tried accompanying source with some comments, for helping you with that.

> Note, this implementation is only helpful when you've relatively large number of leaf nodes and you want to quickly compute all intermediate nodes of Binary Merkle Tree using BLAKE3 2-to-1 hashing.

> Just to enforce aforementioned fact, I've also put one check that # -of leaf nodes of Merkle Tree is at least 2 ^ 20.

## Benchmark(s)

For benchmarking OpenCL accelerated Binary Merklization implementation using 2-to-1 BLAKE3 hashing, I set up random input byte array of size {32 MB, 64 MB, ... 1GB}, which are interpreted as contiguous blocks of 32 little endian bytes, making {2 ^ 20, 2 ^ 21, ... 2 ^ 25} -many leaf nodes of Binary Merkle Tree. Now, with multiple kernel dispatch rounds all (N - 1) -many intermediate nodes of Binary Merkle Tree with N -many leaf nodes are computed, which are then interpreted as little endian bytes making `cl_uint` -> `cl_uchar[4]`. In following tables, time column denotes how long does it take to complete execution of all kernels which are dispatched in multiple rounds during Binary Merklization.

> Note, following time column don't include host to device or device to host data transfer cost.

Command used for running benchmark

```bash
make    # build executable
./run   # execute
```

## On Nvidia GPU

```bash
running on Tesla V100-SXM2-16GB

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklize                2 ^ 20 leaves           in           3.7568 ms
merklize                2 ^ 21 leaves           in           7.2783 ms
merklize                2 ^ 22 leaves           in          14.2994 ms
merklize                2 ^ 23 leaves           in          26.7212 ms
merklize                2 ^ 24 leaves           in          50.9345 ms
merklize                2 ^ 25 leaves           in         101.2877 ms
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

merklize                2 ^ 20 leaves           in          39.5624 ms
merklize                2 ^ 21 leaves           in          38.9113 ms
merklize                2 ^ 22 leaves           in          48.4649 ms
merklize                2 ^ 23 leaves           in          73.8763 ms
merklize                2 ^ 24 leaves           in         117.5491 ms
merklize                2 ^ 25 leaves           in         205.1310 ms
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

merklize                2 ^ 20 leaves           in          19.1502 ms
merklize                2 ^ 21 leaves           in          26.7991 ms
merklize                2 ^ 22 leaves           in          38.9452 ms
merklize                2 ^ 23 leaves           in          66.1401 ms
merklize                2 ^ 24 leaves           in         131.7589 ms
merklize                2 ^ 25 leaves           in         271.2020 ms
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

merklize                2 ^ 20 leaves           in          23.0606 ms
merklize                2 ^ 21 leaves           in          33.2353 ms
merklize                2 ^ 22 leaves           in          47.6613 ms
merklize                2 ^ 23 leaves           in          87.2088 ms
merklize                2 ^ 24 leaves           in         161.0918 ms
merklize                2 ^ 25 leaves           in         301.2635 ms
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

merklize                2 ^ 20 leaves           in          14.4223 ms
merklize                2 ^ 21 leaves           in          34.3838 ms
merklize                2 ^ 22 leaves           in          93.8188 ms
merklize                2 ^ 23 leaves           in         125.2276 ms
merklize                2 ^ 24 leaves           in         227.2035 ms
merklize                2 ^ 25 leaves           in         467.8344 ms
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

merklize                2 ^ 20 leaves           in          63.8715 ms
merklize                2 ^ 21 leaves           in         123.5039 ms
merklize                2 ^ 22 leaves           in         239.4590 ms
merklize                2 ^ 23 leaves           in         543.0369 ms
merklize                2 ^ 24 leaves           in        1008.3083 ms
merklize                2 ^ 25 leaves           in        2206.7236 ms
```

## On Intel GPU(s)

```bash
running on Intel(R) Iris(R) Xe MAX Graphics [0x4905]

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklize                2 ^ 20 leaves           in          29.4210 ms
merklize                2 ^ 21 leaves           in          39.0622 ms
merklize                2 ^ 22 leaves           in          34.0255 ms
merklize                2 ^ 23 leaves           in          67.9463 ms
merklize                2 ^ 24 leaves           in         135.8051 ms
merklize                2 ^ 25 leaves           in         271.5425 ms
```

---

```bash
running on Intel(R) UHD Graphics P630 [0x3e96]

passed blake3 hash test !

Benchmarking Binary Merklization using BLAKE3

merklize                2 ^ 20 leaves           in          17.8799 ms
merklize                2 ^ 21 leaves           in          35.3276 ms
merklize                2 ^ 22 leaves           in          70.1749 ms
merklize                2 ^ 23 leaves           in         139.7336 ms
merklize                2 ^ 24 leaves           in         278.7808 ms
merklize                2 ^ 25 leaves           in         556.8201 ms
```
