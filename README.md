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
