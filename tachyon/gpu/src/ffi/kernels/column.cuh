/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "bitVector.cuh"
#include "utils.cuh"

struct Column {
  void *data;
  uint64_t *null_bits;
  size_t size;

  __host__ __device__ Column(void *data, size_t size,
                             uint64_t *bit_vec = nullptr)
      : data(data), null_bits(null_bits), size(size) {
    ASSERT(data != nullptr, "Column data pointer must not be null");
    ASSERT(size > 0, "Column size must be greater than zero");
  }

  __device__ __forceinline__ bool is_valid(size_t idx) const {
    ASSERT(idx < size, "is_valid(): index out of range");

    return bitVector::is_valid<uint64_t>(null_bits, idx);
  }

  __device__ __forceinline__ void set_valid(size_t idx, bool valid) const {
    ASSERT(idx < size, "set_valid(): index out of range");

    if (valid) {
      bitVector::set_valid<uint64_t>(null_bits, idx);
    } else {
      bitVector::set_null<uint64_t>(null_bits, idx);
    }
  }

  template <TypeKind K>
  __device__ __forceinline__ kind_to_wrapper_t<K> load(size_t idx) const {
    ASSERT(idx < size, "load(): index out of range");
    kind_to_wrapper_t<K> load_value;
    if (!is_valid(idx)) {
      load_value.valid = false;
      return load_value;
    }
    using NativeType = typename TypeTraits<K>::NativeType;
    load_value.value = reinterpret_cast<const NativeType *>(data)[idx];
    return load_value;
  }

  template <TypeKind K>
  __device__ __forceinline__ void
  store(size_t idx, const kind_to_wrapper_t<K> &store_value) {
    ASSERT(idx < size, "store(): index out of range");
    set_valid(idx, store_value.valid);
    if (store_value.valid) {
      using NativeType = typename TypeTraits<K>::NativeType;
      reinterpret_cast<NativeType *>(data)[idx] = store_value.value;
    }
  }
};
