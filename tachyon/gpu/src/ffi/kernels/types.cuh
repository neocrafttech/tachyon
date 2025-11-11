/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "limits.cuh"

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
  STRING,
};

template <TypeKind K> struct TypeTraits;

#define DEFINE_TYPE(NAME, CPP_TYPE, SIZE, IS_SIGNED, IS_FLOAT, MIN_EXPR,       \
                    MAX_EXPR)                                                  \
  struct NAME {                                                                \
    using NativeType = CPP_TYPE;                                               \
    NativeType value;                                                          \
    bool valid = true;                                                         \
    __host__ __device__ NAME() = default;                                      \
    __host__ __device__ NAME(NativeType v) : value(v) {}                       \
                                                                               \
    static constexpr TypeKind kind = TypeKind::NAME;                           \
    static constexpr uint8_t size = SIZE;                                      \
    static constexpr bool is_signed = IS_SIGNED;                               \
    static constexpr bool is_floating = IS_FLOAT;                              \
    static constexpr bool is_integral = !IS_FLOAT;                             \
                                                                               \
    __host__ __device__ static constexpr CPP_TYPE min() { return MIN_EXPR; }   \
    __host__ __device__ static constexpr CPP_TYPE max() { return MAX_EXPR; }   \
                                                                               \
    __host__ __device__ operator NativeType() const { return value; }          \
  };                                                                           \
                                                                               \
  template <> struct TypeTraits<TypeKind::NAME> {                              \
    using WrapperType = NAME;                                                  \
    using NativeType = CPP_TYPE;                                               \
    static constexpr TypeKind kind = TypeKind::NAME;                           \
    static constexpr uint8_t size = SIZE;                                      \
    static constexpr bool is_signed = IS_SIGNED;                               \
    static constexpr bool is_floating = IS_FLOAT;                              \
    static constexpr unsigned int size_bytes = sizeof(CPP_TYPE);               \
    __host__ __device__ static constexpr CPP_TYPE min() { return MIN_EXPR; }   \
    __host__ __device__ static constexpr CPP_TYPE max() { return MAX_EXPR; }   \
  };

DEFINE_TYPE(BOOL, bool, sizeof(bool), false, false, false, true)
DEFINE_TYPE(INT8, int8_t, sizeof(int8_t), true, false,
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max())
DEFINE_TYPE(UINT8, uint8_t, sizeof(uint8_t), false, false,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max())
DEFINE_TYPE(INT16, int16_t, sizeof(int16_t), true, false,
            std::numeric_limits<int16_t>::min(),
            std::numeric_limits<int16_t>::max())
DEFINE_TYPE(UINT16, uint16_t, sizeof(uint16_t), false, false,
            std::numeric_limits<uint16_t>::min(),
            std::numeric_limits<uint16_t>::max())
DEFINE_TYPE(INT32, int32_t, sizeof(int32_t), true, false,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max())
DEFINE_TYPE(UINT32, uint32_t, sizeof(uint32_t), false, false,
            std::numeric_limits<uint32_t>::min(),
            std::numeric_limits<uint32_t>::max())
DEFINE_TYPE(INT64, int64_t, sizeof(int64_t), true, false,
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max())
DEFINE_TYPE(UINT64, uint64_t, sizeof(uint64_t), false, false,
            std::numeric_limits<uint64_t>::min(),
            std::numeric_limits<uint64_t>::max())
DEFINE_TYPE(BFLOAT16, bfloat16, sizeof(bfloat16), true, true,
            std::numeric_limits<bfloat16>::min(),
            std::numeric_limits<bfloat16>::max())
DEFINE_TYPE(FLOAT16, float16, sizeof(float16), true, true,
            std::numeric_limits<float16>::min(),
            std::numeric_limits<float16>::max())
DEFINE_TYPE(FLOAT32, float, sizeof(float), true, true,
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max())
DEFINE_TYPE(FLOAT64, double, sizeof(double), true, true,
            std::numeric_limits<double>::min(),
            std::numeric_limits<double>::max())

#undef DEFINE_TYPE

template <TypeKind K> struct KindToWrapper;
#define DEFINE_KIND_MAPPING(ENUM_VAL)                                          \
  template <> struct KindToWrapper<TypeKind::ENUM_VAL> {                       \
    using type = ENUM_VAL;                                                     \
  };

DEFINE_KIND_MAPPING(BOOL)
DEFINE_KIND_MAPPING(INT8)
DEFINE_KIND_MAPPING(UINT8)
DEFINE_KIND_MAPPING(INT16)
DEFINE_KIND_MAPPING(UINT16)
DEFINE_KIND_MAPPING(INT32)
DEFINE_KIND_MAPPING(UINT32)
DEFINE_KIND_MAPPING(INT64)
DEFINE_KIND_MAPPING(UINT64)
DEFINE_KIND_MAPPING(BFLOAT16)
DEFINE_KIND_MAPPING(FLOAT16)
DEFINE_KIND_MAPPING(FLOAT32)
DEFINE_KIND_MAPPING(FLOAT64)

#undef DEFINE_KIND_MAPPING

template <TypeKind K> using kind_to_wrapper_t = typename KindToWrapper<K>::type;
template <TypeKind K>
using kind_to_native_t = typename KindToWrapper<K>::type::NativeType;

struct TypeDescriptor {
  TypeKind kind;
  uint8_t size_bytes;
  bool is_signed;
  bool is_floating;

  __host__ __device__ constexpr TypeDescriptor(TypeKind k, uint8_t sz, bool sgn,
                                               bool flt)
      : kind(k), size_bytes(sz), is_signed(sgn), is_floating(flt) {}

  template <TypeKind K>
  __host__ __device__ static constexpr TypeDescriptor from_type() {
    return TypeDescriptor(TypeTraits<K>::kind, TypeTraits<K>::size_bytes,
                          TypeTraits<K>::is_signed, TypeTraits<K>::is_floating);
  }
};

__constant__ const TypeDescriptor TYPE_DESCRIPTORS[] = {
    TypeDescriptor::from_type<TypeKind::BOOL>(),
    TypeDescriptor::from_type<TypeKind::INT8>(),
    TypeDescriptor::from_type<TypeKind::UINT8>(),
    TypeDescriptor::from_type<TypeKind::INT16>(),
    TypeDescriptor::from_type<TypeKind::UINT16>(),
    TypeDescriptor::from_type<TypeKind::INT32>(),
    TypeDescriptor::from_type<TypeKind::UINT32>(),
    TypeDescriptor::from_type<TypeKind::INT64>(),
    TypeDescriptor::from_type<TypeKind::UINT64>(),
    TypeDescriptor::from_type<TypeKind::BFLOAT16>(),
    TypeDescriptor::from_type<TypeKind::FLOAT16>(),
    TypeDescriptor::from_type<TypeKind::FLOAT32>(),
    TypeDescriptor::from_type<TypeKind::FLOAT64>(),
};

__host__ __device__ inline const TypeDescriptor &
get_type_descriptor(TypeKind kind) {
  return TYPE_DESCRIPTORS[static_cast<uint8_t>(kind)];
}
