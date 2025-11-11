/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace bitVector {
template <typename T> __host__ __device__ constexpr size_t bits_per_word() {
  return sizeof(T) * 8;
}

template <typename T>
__device__ inline bool is_valid(const T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return true;

  size_t word_idx = idx / bits_per_word<T>();
  size_t bit_idx = idx % bits_per_word<T>();

  return (bit_vec[word_idx] & (T(1) << bit_idx)) != 0;
}

template <typename T> __device__ inline void set_valid(T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return;

  size_t word_idx = idx / bits_per_word<T>();
  size_t bit_idx = idx % bits_per_word<T>();

  atomicOr(&bit_vec[word_idx], T(1) << bit_idx);
}

template <typename T> __device__ inline void set_null(T *bit_vec, size_t idx) {
  if (bit_vec == nullptr)
    return;

  size_t word_idx = idx / bits_per_word<T>();
  size_t bit_idx = idx % bits_per_word<T>();

  atomicAnd(&bit_vec[word_idx], ~(T(1) << bit_idx));
}
} // namespace bitVector
