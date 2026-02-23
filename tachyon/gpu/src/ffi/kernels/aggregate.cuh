/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "column.cuh"
#include "math.cuh"
#include "types.cuh"

namespace aggregate {

constexpr int AGGREGATE_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;
constexpr int MAX_WARPS_PER_BLOCK = AGGREGATE_BLOCK_SIZE / WARP_SIZE;

template <typename T>
__device__ __forceinline__ T shfl_down_value(T value, int offset,
                                             unsigned mask = 0xffffffffu) {
  if constexpr (sizeof(T) == 8) {
    union {
      T value;
      uint64_t bits;
    } payload;
    payload.value = value;
    payload.bits = __shfl_down_sync(mask, payload.bits, offset);
    return payload.value;
  } else if constexpr (sizeof(T) == 4) {
    union {
      T value;
      uint32_t bits;
    } payload;
    payload.value = value;
    payload.bits = __shfl_down_sync(mask, payload.bits, offset);
    return payload.value;
  } else if constexpr (sizeof(T) == 2) {
    union {
      T value;
      uint16_t bits;
    } payload;
    payload.value = value;
    uint32_t widened = static_cast<uint32_t>(payload.bits);
    widened = __shfl_down_sync(mask, widened, offset);
    payload.bits = static_cast<uint16_t>(widened);
    return payload.value;
  } else if constexpr (sizeof(T) == 1) {
    union {
      T value;
      uint8_t bits;
    } payload;
    payload.value = value;
    uint32_t widened = static_cast<uint32_t>(payload.bits);
    widened = __shfl_down_sync(mask, widened, offset);
    payload.bits = static_cast<uint8_t>(widened);
    return payload.value;
  } else {
    return value;
  }
}

template <>
__device__ __forceinline__ float16 shfl_down_value<float16>(float16 value,
                                                            int offset,
                                                            unsigned mask) {
  uint32_t widened = static_cast<uint32_t>(__half_as_ushort(value));
  widened = __shfl_down_sync(mask, widened, offset);
  return __ushort_as_half(static_cast<unsigned short>(widened));
}

template <>
__device__ __forceinline__ bfloat16 shfl_down_value<bfloat16>(bfloat16 value,
                                                              int offset,
                                                              unsigned mask) {
  uint32_t widened = static_cast<uint32_t>(__bfloat16_as_ushort(value));
  widened = __shfl_down_sync(mask, widened, offset);
  return __ushort_as_bfloat16(static_cast<unsigned short>(widened));
}

template <typename T>
__device__ __forceinline__ void
combine_min(bool &lhs_valid, typename T::NativeType &lhs_value, bool rhs_valid,
            typename T::NativeType rhs_value) {
  if (!rhs_valid) {
    return;
  }
  if (!lhs_valid) {
    lhs_valid = true;
    lhs_value = rhs_value;
    return;
  }

  if constexpr (T::is_floating) {
    const bool lhs_nan = cuda_utils::is_nan(lhs_value);
    const bool rhs_nan = cuda_utils::is_nan(rhs_value);
    if (lhs_nan && !rhs_nan) {
      lhs_value = rhs_value;
    } else if (!lhs_nan && !rhs_nan && rhs_value < lhs_value) {
      lhs_value = rhs_value;
    }
  } else if (rhs_value < lhs_value) {
    lhs_value = rhs_value;
  }
}

template <typename T>
__device__ __forceinline__ void
combine_max(bool &lhs_valid, typename T::NativeType &lhs_value, bool rhs_valid,
            typename T::NativeType rhs_value) {
  if (!rhs_valid) {
    return;
  }
  if (!lhs_valid) {
    lhs_valid = true;
    lhs_value = rhs_value;
    return;
  }

  if constexpr (T::is_floating) {
    const bool lhs_nan = cuda_utils::is_nan(lhs_value);
    const bool rhs_nan = cuda_utils::is_nan(rhs_value);
    if (lhs_nan && !rhs_nan) {
      lhs_value = rhs_value;
    } else if (!lhs_nan && !rhs_nan && rhs_value > lhs_value) {
      lhs_value = rhs_value;
    }
  } else if (rhs_value > lhs_value) {
    lhs_value = rhs_value;
  }
}

template <typename T>
__device__ __forceinline__ void
block_reduce_min(bool &value_valid, typename T::NativeType &value) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const unsigned warp_mask = __activemask();

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const bool other_valid =
        __shfl_down_sync(warp_mask, static_cast<unsigned int>(value_valid),
                         offset) != 0;
    const auto other_value = shfl_down_value(value, offset, warp_mask);
    combine_min<T>(value_valid, value, other_valid, other_value);
  }

  __shared__ bool shared_valid[MAX_WARPS_PER_BLOCK];
  __shared__ typename T::NativeType shared_value[MAX_WARPS_PER_BLOCK];

  if (lane == 0) {
    shared_valid[warp_id] = value_valid;
    shared_value[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    bool final_valid = (lane < num_warps) ? shared_valid[lane] : false;
    typename T::NativeType final_value =
        (lane < num_warps) ? shared_value[lane] : T::zero();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const bool other_valid =
          __shfl_down_sync(warp_mask, static_cast<unsigned int>(final_valid),
                           offset) != 0;
      const auto other_value = shfl_down_value(final_value, offset, warp_mask);
      combine_min<T>(final_valid, final_value, other_valid, other_value);
    }

    if (lane == 0) {
      shared_valid[0] = final_valid;
      shared_value[0] = final_value;
    }
  }
  __syncthreads();

  value_valid = shared_valid[0];
  value = shared_value[0];
}

template <typename T>
__device__ __forceinline__ void
block_reduce_max(bool &value_valid, typename T::NativeType &value) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const unsigned warp_mask = __activemask();

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const bool other_valid =
        __shfl_down_sync(warp_mask, static_cast<unsigned int>(value_valid),
                         offset) != 0;
    const auto other_value = shfl_down_value(value, offset, warp_mask);
    combine_max<T>(value_valid, value, other_valid, other_value);
  }

  __shared__ bool shared_valid[MAX_WARPS_PER_BLOCK];
  __shared__ typename T::NativeType shared_value[MAX_WARPS_PER_BLOCK];

  if (lane == 0) {
    shared_valid[warp_id] = value_valid;
    shared_value[warp_id] = value;
  }
  __syncthreads();

  if (warp_id == 0) {
    bool final_valid = (lane < num_warps) ? shared_valid[lane] : false;
    typename T::NativeType final_value =
        (lane < num_warps) ? shared_value[lane] : T::zero();

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const bool other_valid =
          __shfl_down_sync(warp_mask, static_cast<unsigned int>(final_valid),
                           offset) != 0;
      const auto other_value = shfl_down_value(final_value, offset, warp_mask);
      combine_max<T>(final_valid, final_value, other_valid, other_value);
    }

    if (lane == 0) {
      shared_valid[0] = final_valid;
      shared_value[0] = final_value;
    }
  }
  __syncthreads();

  value_valid = shared_valid[0];
  value = shared_value[0];
}

template <typename NativeT> __device__ __forceinline__ NativeT zero_value() {
  return static_cast<NativeT>(0);
}

template <> __device__ __forceinline__ float16 zero_value<float16>() {
  return __float2half(0.0f);
}

template <> __device__ __forceinline__ bfloat16 zero_value<bfloat16>() {
  return __float2bfloat16(0.0f);
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ bool
add_checked(C *ctx, bool &lhs_valid, typename T::NativeType &lhs_value,
            bool rhs_valid, typename T::NativeType rhs_value) {
  if (!rhs_valid) {
    return true;
  }
  if (!lhs_valid) {
    lhs_valid = true;
    lhs_value = rhs_value;
    return true;
  }

  if constexpr (ErrorMode && T::is_integral) {
    T lhs;
    lhs.valid = true;
    lhs.value = lhs_value;
    T rhs;
    rhs.valid = true;
    rhs.value = rhs_value;
    T out = math::add<true>(ctx, lhs, rhs);
    if (!out.valid) {
      lhs_valid = false;
      return false;
    }
    lhs_value = out.value;
    return true;
  } else {
    lhs_value += rhs_value;
    return true;
  }
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ void
block_reduce_sum_count(C *ctx, bool &sum_valid, typename T::NativeType &sum,
                       uint64_t &count) {
  const int lane = threadIdx.x & (WARP_SIZE - 1);
  const int warp_id = threadIdx.x / WARP_SIZE;
  const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  const unsigned warp_mask = __activemask();

  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    const bool other_valid =
        __shfl_down_sync(warp_mask, static_cast<unsigned int>(sum_valid),
                         offset) != 0;
    const auto other_sum = shfl_down_value(sum, offset, warp_mask);
    add_checked<ErrorMode, C, T>(ctx, sum_valid, sum, other_valid, other_sum);
    count += __shfl_down_sync(warp_mask, count, offset);
  }

  __shared__ bool shared_valid[MAX_WARPS_PER_BLOCK];
  __shared__ typename T::NativeType shared_sum[MAX_WARPS_PER_BLOCK];
  __shared__ uint64_t shared_count[MAX_WARPS_PER_BLOCK];

  if (lane == 0) {
    shared_valid[warp_id] = sum_valid;
    shared_sum[warp_id] = sum;
    shared_count[warp_id] = count;
  }
  __syncthreads();

  if (warp_id == 0) {
    bool final_valid = (lane < num_warps) ? shared_valid[lane] : false;
    typename T::NativeType final_sum =
        (lane < num_warps) ? shared_sum[lane]
                           : zero_value<typename T::NativeType>();
    uint64_t final_count = (lane < num_warps) ? shared_count[lane] : 0;

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
      const bool other_valid =
          __shfl_down_sync(warp_mask, static_cast<unsigned int>(final_valid),
                           offset) != 0;
      const auto other_sum = shfl_down_value(final_sum, offset, warp_mask);
      add_checked<ErrorMode, C, T>(ctx, final_valid, final_sum, other_valid,
                                   other_sum);
      final_count += __shfl_down_sync(warp_mask, final_count, offset);
    }

    if (lane == 0) {
      shared_valid[0] = final_valid;
      shared_sum[0] = final_sum;
      shared_count[0] = final_count;
    }
  }
  __syncthreads();

  sum_valid = shared_valid[0];
  sum = shared_sum[0];
  count = shared_count[0];
}

template <TypeKind K, typename B>
__device__ __forceinline__ kind_to_wrapper_t<K> min(const Column &col,
                                                    size_t num_rows) {
  using T = kind_to_wrapper_t<K>;
  bool local_valid = false;
  typename T::NativeType local_value = T::zero();

  for (size_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
    const auto v = col.load<K, B>(i);
    combine_min<T>(local_valid, local_value, v.valid, v.value);
  }

  block_reduce_min<T>(local_valid, local_value);

  T result;
  result.valid = local_valid;
  result.value = local_value;
  return result;
}

template <TypeKind K, typename B>
__device__ __forceinline__ kind_to_wrapper_t<K> max(const Column &col,
                                                    size_t num_rows) {
  using T = kind_to_wrapper_t<K>;
  bool local_valid = false;
  typename T::NativeType local_value = T::zero();

  for (size_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
    const auto v = col.load<K, B>(i);
    combine_max<T>(local_valid, local_value, v.valid, v.value);
  }

  block_reduce_max<T>(local_valid, local_value);

  T result;
  result.valid = local_valid;
  result.value = local_value;
  return result;
}

template <bool ErrorMode, TypeKind K, typename B, typename C>
__device__ __forceinline__ kind_to_wrapper_t<K> sum(C *ctx, const Column &col,
                                                    size_t num_rows) {
  using T = kind_to_wrapper_t<K>;
  typename T::NativeType local_sum = zero_value<typename T::NativeType>();
  bool local_valid = false;
  uint64_t local_count = 0;

  for (size_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
    const auto v = col.load<K, B>(i);
    if (v.valid) {
      add_checked<ErrorMode, C, T>(ctx, local_valid, local_sum, true, v.value);
      ++local_count;
    }
  }

  block_reduce_sum_count<ErrorMode, C, T>(ctx, local_valid, local_sum,
                                          local_count);

  T result;
  result.valid = (local_count > 0) && local_valid;
  result.value = local_sum;
  return result;
}

template <TypeKind K, typename B>
__device__ __forceinline__ UInt64 count(const Column &col, size_t num_rows) {
  uint64_t local_count = 0;

  for (size_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
    if (col.template is_valid<B>(i)) {
      ++local_count;
    }
  }

  bool sum_valid = true;
  uint64_t sum = local_count;
  uint64_t count_dummy = 0;
  block_reduce_sum_count<false, void, UInt64>(static_cast<void *>(nullptr),
                                              sum_valid, sum, count_dummy);

  UInt64 result;
  result.valid = true;
  result.value = sum;
  return result;
}

template <bool ErrorMode, TypeKind K, typename B, typename C>
__device__ __forceinline__ kind_to_wrapper_t<K> avg(C *ctx, const Column &col,
                                                    size_t num_rows) {
  using T = kind_to_wrapper_t<K>;
  typename T::NativeType local_sum = zero_value<typename T::NativeType>();
  bool local_valid = false;
  uint64_t local_count = 0;

  for (size_t i = threadIdx.x; i < num_rows; i += blockDim.x) {
    const auto v = col.load<K, B>(i);
    if (v.valid) {
      add_checked<ErrorMode, C, T>(ctx, local_valid, local_sum, true, v.value);
      ++local_count;
    }
  }

  block_reduce_sum_count<ErrorMode, C, T>(ctx, local_valid, local_sum,
                                          local_count);

  T result;
  if (local_count == 0 || !local_valid) {
    result.valid = false;
    result.value = zero_value<typename T::NativeType>();
    return result;
  }

  result.valid = true;
  result.value = local_sum / static_cast<typename T::NativeType>(local_count);
  return result;
}

} // namespace aggregate
