/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <limits>

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

enum class TypeKind : uint8_t {
  BOOL,
  INT8,
  UINT8,
  INT16,
  UINT16,
  INT32,
  UINT32,
  INT64,
  UINT64,
  BFLOAT16,
  FLOAT16,
  FLOAT32,
  FLOAT64,
};

namespace std {
template <> class numeric_limits<bf16> {
public:
  static constexpr bool is_specialized = true;
  static constexpr bfloat16 max() noexcept {
    return __float2bfloat16(3.389531e38f);
  }
  static constexpr bfloat16 min() noexcept {
    return __float2bfloat16(-3.389531e38f);
  }
  static constexpr bfloat16 lowest() noexcept {
    return __float2bfloat16(-3.389531e38f);
  }
  static constexpr bfloat16 epsilon() noexcept {
    return __float2bfloat16(0.0078125f);
  }
};
template <> class numeric_limits<f16> {
public:
  static constexpr bool is_specialized = true;
  static constexpr float16 max() noexcept { return __float2half(65504.0f); }
  static constexpr float16 min() noexcept { return __float2half(-65504.0f); }
  static constexpr float16 lowest() noexcept { return __float2half(-65504.0f); }
  static constexpr float16 epsilon() noexcept {
    return __float2half(0.00097656f);
  }
};
} // namespace std

template <TypeKind K> struct TypeTraits;

#define DEFINE_TYPE_TRAITS(ENUM_VAL, CPP_TYPE, SIZE, IS_SIGNED, IS_FLOAT,      \
                           MIN_EXPR, MAX_EXPR)                                 \
  template <> struct TypeTraits<TypeKind::ENUM_VAL> {                          \
    using NativeType = CPP_TYPE;                                               \
    static constexpr TypeKind kind = TypeKind::ENUM_VAL;                       \
    static constexpr uint8_t size = SIZE;                                      \
    static constexpr bool is_signed = IS_SIGNED;                               \
    static constexpr bool is_floating = IS_FLOAT;                              \
    static constexpr bool is_integral = !IS_FLOAT;                             \
    static constexpr unsigned int size_bytes = sizeof(CPP_TYPE);               \
    NativeType value;                                                          \
    __host__ __device__ static constexpr CPP_TYPE min_value() {                \
      return MIN_EXPR;                                                         \
    }                                                                          \
    __host__ __device__ static constexpr CPP_TYPE max_value() {                \
      return MAX_EXPR;                                                         \
    }                                                                          \
  };

DEFINE_TYPE_TRAITS(BOOL, bool, sizeof(bool), false, false, false, true)
DEFINE_TYPE_TRAITS(INT8, int8_t, sizeof(int8_t), true, false,
                   std::numeric_limits<int8_t>::min(),
                   std::numeric_limits<int8_t>::max())
DEFINE_TYPE_TRAITS(UINT8, uint8_t, sizeof(uint8_t), false, false,
                   std::numeric_limits<uint8_t>::min(),
                   std::numeric_limits<uint8_t>::max())
DEFINE_TYPE_TRAITS(INT16, int16_t, sizeof(int16_t), true, false,
                   std::numeric_limits<int16_t>::min(),
                   std::numeric_limits<int16_t>::max())
DEFINE_TYPE_TRAITS(UINT16, uint16_t, sizeof(uint16_t), false, false,
                   std::numeric_limits<uint16_t>::min(),
                   std::numeric_limits<uint16_t>::max())
DEFINE_TYPE_TRAITS(INT32, int32_t, sizeof(int32_t), true, false,
                   std::numeric_limits<int32_t>::min(),
                   std::numeric_limits<int32_t>::max())
DEFINE_TYPE_TRAITS(UINT32, uint32_t, sizeof(uint32_t), false, false,
                   std::numeric_limits<uint32_t>::min(),
                   std::numeric_limits<uint32_t>::max())
DEFINE_TYPE_TRAITS(INT64, int64_t, sizeof(int64_t), true, false,
                   std::numeric_limits<int64_t>::min(),
                   std::numeric_limits<int64_t>::max())
DEFINE_TYPE_TRAITS(UINT64, uint64_t, sizeof(uint64_t) false, false,
                   std::numeric_limits<uint64_t>::min(),
                   std::numeric_limits<uint64_t>::max())
DEFINE_TYPE_TRAITS(BFLOAT16, bf16, sizeof(bfloat16), true, true,
                   std::numeric_limits<bfloat16>::min(),
                   std::numeric_limits<bfloat16>::max())
DEFINE_TYPE_TRAITS(FLOAT16, f16, sizeof(float16), true, true,
                   std::numeric_limits<float16>::min(),
                   std::numeric_limits<float16>::max())
DEFINE_TYPE_TRAITS(FLOAT32, float, sizeof(float), true, true,
                   std::numeric_limits<float>::min(),
                   std::numeric_limits<float>::max())
DEFINE_TYPE_TRAITS(FLOAT64, double, sizeof(double), true, true,
                   std::numeric_limits<double>::min(),
                   std::numeric_limits<double>::max())

#undef DEFINE_TYPE_TRAITS

template <TypeKind K> struct KindToType;

#define DEFINE_KIND_TO_TYPE(ENUM_VAL, CPP_TYPE)                                \
  template <> struct KindToType<TypeKind::ENUM_VAL> {                          \
    using type = CPP_TYPE;                                                     \
  };

DEFINE_KIND_TO_TYPE(BOOL, bool)
DEFINE_KIND_TO_TYPE(INT8, int8_t)
DEFINE_KIND_TO_TYPE(UINT8, uint8_t)
DEFINE_KIND_TO_TYPE(INT16, int16_t)
DEFINE_KIND_TO_TYPE(UINT16, uint16_t)
DEFINE_KIND_TO_TYPE(INT32, int32_t)
DEFINE_KIND_TO_TYPE(UINT32, uint32_t)
DEFINE_KIND_TO_TYPE(INT64, int64_t)
DEFINE_KIND_TO_TYPE(UINT64, uint64_t)
DEFINE_KIND_TO_TYPE(FLOAT32, float)
DEFINE_KIND_TO_TYPE(FLOAT64, double)

#undef DEFINE_KIND_TO_TYPE

template <TypeKind K> using kind_to_type_t = typename KindToType<K>::type;

struct TypeDescriptor {
  TypeKind kind;
  uint8_t size_bytes;
  bool is_signed;
  bool is_floating;

  __host__ __device__ constexpr TypeDescriptor(TypeKind k, uint8_t sz, bool sgn,
                                               bool flt)
      : kind(k), size_bytes(sz), is_signed(sgn), is_floating(flt) {}

  template <typename T>
  __host__ __device__ static constexpr TypeDescriptor from_type() {
    return TypeDescriptor(TypeTraits<T>::kind, TypeTraits<T>::size_bytes,
                          TypeTraits<T>::is_signed, TypeTraits<T>::is_floating);
  }
};

__constant__ const TypeDescriptor TYPE_DESCRIPTORS[] = {
    TypeDescriptor::from_type<bool>(),     TypeDescriptor::from_type<int8_t>(),
    TypeDescriptor::from_type<uint8_t>(),  TypeDescriptor::from_type<int16_t>(),
    TypeDescriptor::from_type<uint16_t>(), TypeDescriptor::from_type<int32_t>(),
    TypeDescriptor::from_type<uint32_t>(), TypeDescriptor::from_type<int64_t>(),
    TypeDescriptor::from_type<uint64_t>(), TypeDescriptor::from_type<float>(),
    TypeDescriptor::from_type<double>(),
};

__host__ __device__ inline const TypeDescriptor &
get_type_descriptor(TypeKind kind) {
  return TYPE_DESCRIPTORS[static_cast<uint8_t>(kind)];
}

__host__ __device__ inline constexpr unsigned int type_size(TypeKind kind) {
  const unsigned int sizes[] = {
      sizeof(bool),     sizeof(int8_t),  sizeof(uint8_t),  sizeof(int16_t),
      sizeof(uint16_t), sizeof(int32_t), sizeof(uint32_t), sizeof(int64_t),
      sizeof(uint64_t), sizeof(float),   sizeof(double)};
  return sizes[static_cast<uint8_t>(kind)];
}

__host__ __device__ inline constexpr bool is_numeric_type(TypeKind kind) {
  return kind != TypeKind::BOOL;
}
