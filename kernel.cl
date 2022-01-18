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
permute(

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
  private uint* const msg
#else
  global uint* const msg
#endif

)
{
private
  uint permuted[16];

  __attribute__((opencl_unroll_hint(16)))
  for (size_t i = 0; i < 16; i++)
  {
    permuted[i] = *(msg + MSG_PERMUTATION[i]);
  }

  __attribute__((opencl_unroll_hint(16)))
  for (size_t i = 0; i < 16; i++)
  {
    *(msg + i) = permuted[i];
  }
}

void
round(private uint4* const state,

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
      private const uint* msg
#else
      global const uint* msg
#endif

)
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
compress(

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
  private uint* const msg,
#else
  global uint* const msg,
#endif

  ulong counter,
  uint block_len,
  uint flags,

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
  private uint* const out_cv
#else
  global uint* const out_cv
#endif

)
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

void
words_from_le_bytes(global const uchar* input, private uint* const msg_words)
{
  __attribute__((opencl_unroll_hint(8)))
  for (size_t i = 0; i < 16; i++)
  {
    *(msg_words + i) = ((uint) * (input + i * 4 + 3) << 24) |
                       ((uint) * (input + i * 4 + 2) << 16) |
                       ((uint) * (input + i * 4 + 1) << 8) |
                       ((uint) * (input + i * 4 + 0) << 0);
  }
}

void
words_to_le_bytes(private const uint* msg_words, global uchar* const output)
{
  __attribute__((opencl_unroll_hint(8)))
  for (size_t i = 0; i < 8; i++)
  {
    const uint num = *(msg_words + i);

    *(output + i * 4 + 0) = (uchar)(num >> 0) & 0xff;
    *(output + i * 4 + 1) = (uchar)(num >> 8) & 0xff;
    *(output + i * 4 + 2) = (uchar)(num >> 16) & 0xff;
    *(output + i * 4 + 3) = (uchar)(num >> 24) & 0xff;
  }
}

#if defined(EXPOSE_BLAKE3_HASH)
kernel
#endif

  void
  hash(

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
    global const uchar* input,
    global uchar* const output
#else
  global uint* const input,
  global uint* const output
#endif

  )
{

#if defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES)
private
  uint msg_words[16];
private
  uint out_cv[8];

  words_from_le_bytes(input, msg_words);
  compress(msg_words, 0, BLOCK_LEN, CHUNK_START | CHUNK_END | ROOT, out_cv);
  words_to_le_bytes(out_cv, output);
#else
  compress(input, 0, BLOCK_LEN, CHUNK_START | CHUNK_END | ROOT, output);
#endif
}

#if !(defined(LE_BYTES_TO_WORDS) && defined(WORDS_TO_LE_BYTES))

kernel void
merklize(global uint* const restrict input,
         constant size_t* restrict i_offset,
         global uint* const restrict output,
         constant size_t* restrict o_offset)
{
private
  const size_t idx = get_global_id(0);

  // idx << 4 => because input being hashed is 64 -bytes wide
  // idx << 3 => because output of blake3 hash is 32 -bytes wide
  hash(input + *i_offset + (idx << 4), output + *o_offset + (idx << 3));
}

#endif
