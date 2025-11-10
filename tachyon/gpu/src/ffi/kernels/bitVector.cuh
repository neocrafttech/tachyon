/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once
namespace bitVector {
__device__ inline bool is_valid(const uint32_t *bitVec, size_t idx) {
  if (bitVec == nullptr)
    return true;

  size_t word_idx = idx / sizeof(uint32_t);
  size_t bit_idx = idx % sizeof(uint32_t);
  return (bitVec[word_idx] & (1u << bit_idx)) != 0;
}

__device__ inline void set_valid(uint32_t *bitVec, size_t idx) {
  if (bitVec == nullptr)
    return;
  size_t word_idx = idx / sizeof(uint32_t);
  size_t bit_idx = idx % sizeof(uint32_t);
  atomicOr(&bitVec[word_idx], 1u << bit_idx);
}

__device__ inline void set_null(uint32_t *bitVec, size_t idx) {
  if (bitVec == nullptr)
    return;
  size_t word_idx = idx / sizeof(uint32_t);
  size_t bit_idx = idx % sizeof(uint32_t);
  atomicAnd(&bitVec[word_idx], ~(1u << bit_idx));
}

__host__ __device__ inline size_t bitmap_size_words(size_t n) {
  return (n + sizeof(uint32_t) - 1) / sizeof(uint32_t);
}
} // namespace bitVector
