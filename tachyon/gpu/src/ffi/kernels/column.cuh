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
  uint32_t *validity_bitmap;
  TypeKind type;
  size_t size;

  __host__ __device__ Column(void *data, TypeKind type, size_t size,
                             uint32_t *bitVec = nullptr)
      : data(data), type(type), size(size), validity_bitmap(bitVec) {
    ASSERT(data != nullptr, "Column data pointer must not be null");
    ASSERT(size > 0, "Column size must be greater than zero");
  }

  __device__ inline bool is_valid(size_t idx) const {
    ASSERT(idx < size, "is_valid(): index out of range");

    return bitVector::is_valid(validity_bitmap, idx);
  }

  __device__ inline bool set_valid(size_t idx) const {
    ASSERT(idx < size, "set_valid(): index out of range");

    return bitVector::set_valid(validity_bitmap, idx);
  }

  template <typename Traits>
  __device__ inline bool load(size_t idx, Traits &loadValue) const {
    ASSERT(idx < size, "load(): index out of range");

    if (!is_valid(idx))
      return false;

    loadValue.value = reinterpret_cast<const Traits::NativeType *>(data)[idx];
    return true;
  }

  template <typename Traits>
  __device__ inline void store(size_t idx, const typename Traits &storeValue,
                               bool valid = true) {
    ASSERT(idx < size, "store(): index out of range");

    set_valid(idx, valid);

    if valid {
      reinterpret_cast<typename Traits::NativeType *>(data)[idx] = value;
    }
  }
};
