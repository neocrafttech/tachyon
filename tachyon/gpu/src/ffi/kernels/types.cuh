/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "limits.cuh"

enum class TypeKind : uint8_t {
  Bool,
  Int8,
  UInt8,
  Int16,
  UInt16,
  Int32,
  UInt32,
  Int64,
  UInt64,
  BFloat16,
  Float16,
  Float32,
  Float64,
  String,
};

template <TypeKind K> struct TypeTraits;

#define DEFINE_TYPE(NAME, CPP_TYPE, SIZE, IS_SIGNED, IS_FLOAT, MIN_EXPR,       \
                    MAX_EXPR, ZERO)                                            \
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
    __host__ __device__ static constexpr CPP_TYPE min() { return MIN_EXPR; }   \
    __host__ __device__ static constexpr CPP_TYPE max() { return MAX_EXPR; }   \
    __host__ __device__ static constexpr CPP_TYPE zero() { return ZERO; }      \
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
    __host__ __device__ static constexpr CPP_TYPE zero() { return ZERO; }      \
  };

DEFINE_TYPE(Bool, bool, sizeof(bool), false, false, false, true, false)
DEFINE_TYPE(Int8, int8_t, sizeof(int8_t), true, false,
            std::numeric_limits<int8_t>::min(),
            std::numeric_limits<int8_t>::max(), 0)
DEFINE_TYPE(UInt8, uint8_t, sizeof(uint8_t), false, false,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max(), 0)
DEFINE_TYPE(Int16, int16_t, sizeof(int16_t), true, false,
            std::numeric_limits<int16_t>::min(),
            std::numeric_limits<int16_t>::max(), 0)
DEFINE_TYPE(UInt16, uint16_t, sizeof(uint16_t), false, false,
            std::numeric_limits<uint16_t>::min(),
            std::numeric_limits<uint16_t>::max(), 0)
DEFINE_TYPE(Int32, int32_t, sizeof(int32_t), true, false,
            std::numeric_limits<int32_t>::min(),
            std::numeric_limits<int32_t>::max(), 0)
DEFINE_TYPE(UInt32, uint32_t, sizeof(uint32_t), false, false,
            std::numeric_limits<uint32_t>::min(),
            std::numeric_limits<uint32_t>::max(), 0)
DEFINE_TYPE(Int64, int64_t, sizeof(int64_t), true, false,
            std::numeric_limits<int64_t>::min(),
            std::numeric_limits<int64_t>::max(), 0)
DEFINE_TYPE(UInt64, uint64_t, sizeof(uint64_t), false, false,
            std::numeric_limits<uint64_t>::min(),
            std::numeric_limits<uint64_t>::max(), 0)
DEFINE_TYPE(BFloat16, bfloat16, sizeof(bfloat16), true, true,
            std::numeric_limits<bfloat16>::min(),
            std::numeric_limits<bfloat16>::max(), 0.0)
DEFINE_TYPE(Float16, float16, sizeof(float16), true, true,
            std::numeric_limits<float16>::min(),
            std::numeric_limits<float16>::max(), 0.0)
DEFINE_TYPE(Float32, float, sizeof(float), true, true,
            std::numeric_limits<float>::min(),
            std::numeric_limits<float>::max(), 0.0)
DEFINE_TYPE(Float64, double, sizeof(double), true, true,
            std::numeric_limits<double>::min(),
            std::numeric_limits<double>::max(), 0.0)

#undef DEFINE_TYPE

template <TypeKind K> struct KindToWrapper;
#define DEFINE_KIND_MAPPING(ENUM_VAL)                                          \
  template <> struct KindToWrapper<TypeKind::ENUM_VAL> {                       \
    using type = ENUM_VAL;                                                     \
  };

DEFINE_KIND_MAPPING(Bool)
DEFINE_KIND_MAPPING(Int8)
DEFINE_KIND_MAPPING(UInt8)
DEFINE_KIND_MAPPING(Int16)
DEFINE_KIND_MAPPING(UInt16)
DEFINE_KIND_MAPPING(Int32)
DEFINE_KIND_MAPPING(UInt32)
DEFINE_KIND_MAPPING(Int64)
DEFINE_KIND_MAPPING(UInt64)
DEFINE_KIND_MAPPING(BFloat16)
DEFINE_KIND_MAPPING(Float16)
DEFINE_KIND_MAPPING(Float32)
DEFINE_KIND_MAPPING(Float64)

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
    TypeDescriptor::from_type<TypeKind::Bool>(),
    TypeDescriptor::from_type<TypeKind::Int8>(),
    TypeDescriptor::from_type<TypeKind::UInt8>(),
    TypeDescriptor::from_type<TypeKind::Int16>(),
    TypeDescriptor::from_type<TypeKind::UInt16>(),
    TypeDescriptor::from_type<TypeKind::Int32>(),
    TypeDescriptor::from_type<TypeKind::UInt32>(),
    TypeDescriptor::from_type<TypeKind::Int64>(),
    TypeDescriptor::from_type<TypeKind::UInt64>(),
    TypeDescriptor::from_type<TypeKind::BFloat16>(),
    TypeDescriptor::from_type<TypeKind::Float16>(),
    TypeDescriptor::from_type<TypeKind::Float32>(),
    TypeDescriptor::from_type<TypeKind::Float64>(),
};

__host__ __device__ inline const TypeDescriptor &
get_type_descriptor(TypeKind kind) {
  return TYPE_DESCRIPTORS[static_cast<uint8_t>(kind)];
}
