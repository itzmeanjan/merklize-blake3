constant size_t MSG_PERMUTATION[16] = { 2, 6,  3,  10, 7, 0,  4,  13,
                                        1, 11, 12, 5,  9, 14, 15, 8 };

constant uint IV[8] = { 0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
                        0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19 };

constant size_t OUT_LEN = 32;
constant size_t ROUNDS = 7;

constant uint BLOCK_LEN = 64;
constant uint CHUNK_START = 1 << 0;
constant uint CHUNK_END = 1 << 1;
constant uint PARENT = 1 << 2;
constant uint ROOT = 1 << 3;

void
permute(global uint* const msg)
{
private
  uint permuted[16];

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    permuted[i] = *(msg + MSG_PERMUTATION[i]);
  }

#pragma unroll 16
  for (size_t i = 0; i < 16; i++) {
    *(msg + i) = permuted[i];
  }
}

void
round(private uint4* const state, global const uint* msg)
{
  uint4 mx = (uint4)(*(msg + 0), *(msg + 2), *(msg + 4), *(msg + 6));
  uint4 my = (uint4)(*(msg + 1), *(msg + 3), *(msg + 5), *(msg + 7));
  uint4 mz = (uint4)(*(msg + 8), *(msg + 10), *(msg + 12), *(msg + 14));
  uint4 mw = (uint4)(*(msg + 9), *(msg + 11), *(msg + 13), *(msg + 15));

  const uint4 rrot_16 = (uint4)(16);
  const uint4 rrot_12 = (uint4)(20);
  const uint4 rrot_8 = (uint4)(24);
  const uint4 rrot_7 = (uint4)(25);

  // column-wise mixing
  *(state + 0) = *(state + 0) + *(state + 1) + mx;
  *(state + 3) = rotate(*(state + 3) ^ *(state + 0), rrot_16);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = rotate(*(state + 1) ^ *(state + 2), rrot_12);
  *(state + 0) = *(state + 0) + *(state + 1) + my;
  *(state + 3) = rotate(*(state + 3) ^ *(state + 0), rrot_8);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // state matrix diagonalization
  {
    uint4 tmp = *(state + 1);
    *(state + 1) = tmp.yzwx;
  }
  {
    uint4 tmp = *(state + 2);
    *(state + 2) = tmp.zwxy;
  }
  {
    uint4 tmp = *(state + 3);
    *(state + 3) = tmp.wxyz;
  }

  // row-wise mixing
  *(state + 0) = *(state + 0) + *(state + 1) + mz;
  *(state + 3) = rotate(*(state + 3) ^ *(state + 0), rrot_16);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = rotate(*(state + 1) ^ *(state + 2), rrot_12);
  *(state + 0) = *(state + 0) + *(state + 1) + mw;
  *(state + 3) = rotate(*(state + 3) ^ *(state + 0), rrot_8);
  *(state + 2) = *(state + 2) + *(state + 3);
  *(state + 1) = rotate(*(state + 1) ^ *(state + 2), rrot_7);

  // state matrix un-diagonalization
  {
    uint4 tmp = *(state + 1);
    *(state + 1) = tmp.wxyz;
  }
  {
    uint4 tmp = *(state + 2);
    *(state + 2) = tmp.zwxy;
  }
  {
    uint4 tmp = *(state + 3);
    *(state + 3) = tmp.yzwx;
  }
}

void
compress(global uint* const msg,
         ulong counter,
         uint block_len,
         uint flags,
         global uint* const out_cv)
{
private
  uint4 state[4] = { (uint4)(IV[0], IV[1], IV[2], IV[3]),
                     (uint4)(IV[4], IV[5], IV[6], IV[7]),
                     (uint4)(IV[0], IV[1], IV[2], IV[3]),
                     (uint4)((uint)(counter & 0xffffffff),
                             (uint)(counter >> 32),
                             block_len,
                             flags) };

  // round 1
  round(state, msg);
  permute(msg);

  // round 2
  round(state, msg);
  permute(msg);

  // round 3
  round(state, msg);
  permute(msg);

  // round 4
  round(state, msg);
  permute(msg);

  // round 5
  round(state, msg);
  permute(msg);

  // round 6
  round(state, msg);
  permute(msg);

  // round 7
  round(state, msg);

  // preparing 32 -bytes output chaining value
  state[0] ^= state[2];
  state[1] ^= state[3];

  // writing output chaining value
  vstore4(state[0], 0, out_cv);
  vstore4(state[1], 1, out_cv);
}
