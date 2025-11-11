/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once
namespace bitVector {
template <typename T>
__device__ inline bool is_valid(const T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return true;

  size_t word_idx = idx / sizeof(T);
  size_t bit_idx = idx % sizeof(T);
  return (bit_vec[word_idx] & (1u << bit_idx)) != 0;
}

template <typename T> __device__ inline void set_valid(T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return;
  size_t word_idx = idx / sizeof(T);
  size_t bit_idx = idx % sizeof(T);
  atomicOr(&bitVec[word_idx], 1u << bit_idx);
}

template <typename T> __device__ inline void set_null(T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return;
  size_t word_idx = idx / sizeof(T);
  size_t bit_idx = idx % sizeof(T);
  atomicAnd(&bitVec[word_idx], ~(1u << bit_idx));
}

template <typename T>
__host__ __device__ inline size_t bitmap_size_words(size_t n) {
  return (n + sizeof(T) - 1) / sizeof(T);
}
} // namespace bitVector
