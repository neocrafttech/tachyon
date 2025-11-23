/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

// Custom numeric_limits implementation for CUDA kernels
// This avoids dependency on <limits> header

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
typedef __nv_bfloat16 bfloat16;
typedef __half float16;

namespace std {

template <typename T> class numeric_limits {
public:
  static constexpr bool is_specialized = false;
};

template <> class numeric_limits<int8_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr int8_t max() noexcept { return 127; }
  __device__ static constexpr int8_t min() noexcept { return -128; }
  __device__ static constexpr int8_t lowest() noexcept { return -128; }
  __device__ static constexpr int8_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<uint8_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr uint8_t max() noexcept { return 255; }
  __device__ static constexpr uint8_t min() noexcept { return 0; }
  __device__ static constexpr uint8_t lowest() noexcept { return 0; }
  __device__ static constexpr uint8_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<int16_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr int16_t max() noexcept { return 32767; }
  __device__ static constexpr int16_t min() noexcept { return -32768; }
  __device__ static constexpr int16_t lowest() noexcept { return -32768; }
  __device__ static constexpr int16_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<uint16_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr uint16_t max() noexcept { return 65535; }
  __device__ static constexpr uint16_t min() noexcept { return 0; }
  __device__ static constexpr uint16_t lowest() noexcept { return 0; }
  __device__ static constexpr uint16_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<int32_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr int32_t max() noexcept { return 2147483647; }
  __device__ static constexpr int32_t min() noexcept { return -2147483648; }
  __device__ static constexpr int32_t lowest() noexcept { return -2147483648; }
  __device__ static constexpr int32_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<uint32_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr uint32_t max() noexcept { return 4294967295U; }
  __device__ static constexpr uint32_t min() noexcept { return 0; }
  __device__ static constexpr uint32_t lowest() noexcept { return 0; }
  __device__ static constexpr uint32_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<int64_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr int64_t max() noexcept {
    return 9223372036854775807LL;
  }
  __device__ static constexpr int64_t min() noexcept {
    return -9223372036854775807LL - 1;
  }
  __device__ static constexpr int64_t lowest() noexcept {
    return -9223372036854775807LL - 1;
  }
  __device__ static constexpr int64_t epsilon() noexcept { return 0; }
};

template <> class numeric_limits<uint64_t> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr uint64_t max() noexcept {
    return 18446744073709551615ULL;
  }
  __device__ static constexpr uint64_t min() noexcept { return 0; }
  __device__ static constexpr uint64_t lowest() noexcept { return 0; }
  __device__ static constexpr uint64_t epsilon() noexcept { return 0; }
};

// Floating point types
template <> class numeric_limits<float> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr float max() noexcept { return 3.40282347e+38F; }
  __device__ static constexpr float min() noexcept { return 1.17549435e-38F; }
  __device__ static constexpr float lowest() noexcept {
    return -3.40282347e+38F;
  }
  __device__ static constexpr float epsilon() noexcept {
    return 1.19209290e-07F;
  }
};

template <> class numeric_limits<double> {
public:
  static constexpr bool is_specialized = true;
  __device__ static constexpr double max() noexcept {
    return 1.7976931348623157e+308;
  }
  __device__ static constexpr double min() noexcept {
    return 2.2250738585072014e-308;
  }
  __device__ static constexpr double lowest() noexcept {
    return -1.7976931348623157e+308;
  }
  __device__ static constexpr double epsilon() noexcept {
    return 2.2204460492503131e-16;
  }
};

template <> class numeric_limits<float16> {
public:
  static constexpr bool is_specialized = true;
  static __device__ __forceinline__ float16 max() noexcept {
    return __float2half(65504.0f);
  }
  static __device__ __forceinline__ float16 min() noexcept {
    return __float2half(6.103515625e-05f);
  }
  static __device__ __forceinline__ float16 lowest() noexcept {
    return __float2half(-65504.0f);
  }
  static __device__ __forceinline__ float16 epsilon() noexcept {
    return __float2half(0.00097656f);
  }
};

template <> class numeric_limits<bfloat16> {
public:
  static constexpr bool is_specialized = true;
  static __device__ __forceinline__ bfloat16 max() noexcept {
    return __float2bfloat16(3.38953139e+38f);
  }
  static __device__ __forceinline__ bfloat16 min() noexcept {
    return __float2bfloat16(1.17549435e-38f);
  }
  static __device__ __forceinline__ bfloat16 lowest() noexcept {
    return __float2bfloat16(-3.38953139e+38f);
  }
  static __device__ __forceinline__ bfloat16 epsilon() noexcept {
    return __float2bfloat16(0.0078125f);
  }
};

template <typename T> struct is_signed {
  static const bool value = false;
};

template <> struct is_signed<int8_t> {
  static const bool value = true;
};
template <> struct is_signed<int16_t> {
  static const bool value = true;
};
template <> struct is_signed<int32_t> {
  static const bool value = true;
};
template <> struct is_signed<int64_t> {
  static const bool value = true;
};

template <> struct is_signed<float16> {
  static const bool value = true;
};
template <> struct is_signed<bfloat16> {
  static const bool value = true;
};
template <> struct is_signed<float> {
  static const bool value = true;
};
template <> struct is_signed<double> {
  static const bool value = true;
};

template <typename T> struct is_unsigned {
  static const bool value = !is_signed<T>::value;
};
} // namespace std
