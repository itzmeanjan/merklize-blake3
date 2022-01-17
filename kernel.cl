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
permute(uint* const msg)
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
