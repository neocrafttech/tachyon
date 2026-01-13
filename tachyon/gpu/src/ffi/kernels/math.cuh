/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
#pragma once

#include "context.cuh"
#include "types.cuh"

namespace math {

template <typename T>
__device__ __forceinline__ bool __check_add_overflow(T a, T b, T *res) {
  *res = a + b;

  if constexpr (cuda_utils::is_unsigned<T>::value) {
    return *res < a;
  } else {
    return ((a ^ *res) & (b ^ *res)) < 0;
  }
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T add(C *__restrict__ ctx, const T &a, const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    if constexpr (ErrorMode && T::is_integral) {
      bool overflow = __check_add_overflow(a.value, b.value, &result.value);
      if (overflow) [[unlikely]] {
        result.valid = false;
        ctx[0].error_code = KernelError::ADD_OVERFLOW;

        return result;
      }
    } else {
      result.value = a.value + b.value;
    }
  }

  return result;
}

template <typename T>
__device__ __forceinline__ bool __check_sub_overflow(T a, T b, T *res) {
  *res = a - b;
  if constexpr (cuda_utils::is_unsigned<T>::value) {
    return *res > a;
  } else {
    return ((a ^ b) & (a ^ *res)) < 0;
  }
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T sub(C *__restrict__ ctx, const T &a, const T &b) {
  T result;

  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    if constexpr (ErrorMode && T::is_integral) {
      bool overflow = __check_sub_overflow(a.value, b.value, &result.value);
      if (overflow) [[unlikely]] {
        result.valid = false;
        ctx[0].error_code = KernelError::SUB_OVERFLOW;

        return result;
      }
    } else {
      result.value = a.value - b.value;
    }
  }

  return result;
}

template <typename T>
__device__ __forceinline__ bool __check_mul_overflow(T a, T b, T *res) {
  if constexpr (sizeof(T) <= 4) {
    if constexpr (cuda_utils::is_unsigned<T>::value) {
      unsigned long long wide_a = a;
      unsigned long long wide_b = b;
      unsigned long long wide_res = wide_a * wide_b;
      *res = (T)wide_res;
      return wide_res > cuda_utils::numeric_limits<T>::max();
    } else {
      long long wide_a = a;
      long long wide_b = b;
      long long wide_res = wide_a * wide_b;
      *res = (T)wide_res;
      return wide_res < cuda_utils::numeric_limits<T>::min() ||
             wide_res > cuda_utils::numeric_limits<T>::max();
    }
  } else {
    if (a == 0 || b == 0) {
      *res = 0;
      return false;
    }
    if (a == 1) {
      *res = b;
      return false;
    }
    if (b == 1) {
      *res = a;
      return false;
    }

    if constexpr (cuda_utils::is_unsigned<T>::value) {
      if (a > cuda_utils::numeric_limits<T>::max() / b) {
        *res = a * b;
        return true;
      }
      *res = a * b;
      return false;
    } else {
      if (a == -1 && b == cuda_utils::numeric_limits<T>::min()) {
        *res = cuda_utils::numeric_limits<T>::min();
        return true;
      }
      if (b == -1 && a == cuda_utils::numeric_limits<T>::min()) {
        *res = cuda_utils::numeric_limits<T>::min();
        return true;
      }

      long long abs_a = (a < 0) ? -(long long)a : a;
      long long abs_b = (b < 0) ? -(long long)b : b;
      if (abs_a > cuda_utils::numeric_limits<T>::max() / abs_b) {
        *res = a * b;
        return true;
      }
      *res = a * b;
      return false;
    }
  }
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T mul(C *__restrict__ ctx, const T &a, const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    if constexpr (ErrorMode && T::is_integral) {
      bool overflow = __check_mul_overflow(a.value, b.value, &result.value);

      if (overflow) [[unlikely]] {
        result.valid = false;
        ctx[0].error_code = KernelError::MUL_OVERFLOW;

        return result;
      }
    } else {
      result.value = a.value * b.value;
    }
  }
  return result;
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T div(C *__restrict__ ctx, const T &a, const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    if constexpr (!T::is_floating) {
      if (b.value == b.zero()) [[unlikely]] {
        result.valid = false;
        ctx[0].error_code = KernelError::DIV_BY_ZERO;
        return result;
      }
    }

    // Irrespective of error mode overflow is error for division, no wrapping
    // like other operations
    bool overflow = (a.value == a.min() && b.value == static_cast<T>(-1));

    if (overflow) [[unlikely]] {
      result.valid = false;
      ctx[0].error_code = KernelError::DIV_OVERFLOW;
      return result;
    }

    result.value = a.value / b.value;
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool eq(C *__restrict__ _ctx, const T1 &lhs,
                                   const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = false;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) == rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = false;
      } else {
        result.value = lhs.value == static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for Nan == Nan is different than pure
      // programming languages
      if (cuda_utils::is_nan(lhs.value) && cuda_utils::is_nan(rhs.value))
          [[unlikely]] {
        result.value = true;
      } else {
        result.value = lhs.value == rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value == rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool neq(C *__restrict__ _ctx, const T1 &lhs,
                                    const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = true;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) != rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = true;
      } else {
        result.value = lhs.value != static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for Nan != Nan is different than pure
      // programming languages
      if (cuda_utils::is_nan(lhs.value) && cuda_utils::is_nan(rhs.value))
          [[unlikely]] {
        result.value = false;
      } else {
        result.value = lhs.value != rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value != rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool lt(C *__restrict__ _ctx, const T1 &lhs,
                                   const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = true;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) < rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = false;
      } else {
        result.value = lhs.value < static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for valid_number < Nan is different than
      // pure programming languages
      if (!cuda_utils::is_nan(lhs.value) && cuda_utils::is_nan(rhs.value))
          [[unlikely]] {
        result.value = true;
      } else {
        result.value = lhs.value < rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value < rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool lteq(C *__restrict__ _ctx, const T1 &lhs,
                                     const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = true;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) <= rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = false;
      } else {
        result.value = lhs.value <= static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for valid_number <= Nan // is different
      // than pure programming languages
      if (cuda_utils::is_nan(rhs.value)) [[unlikely]] {
        result.value = true;
      } else {
        result.value = lhs.value <= rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value <= rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool gt(C *__restrict__ _ctx, const T1 &lhs,
                                   const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = false;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) > rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = true;
      } else {
        result.value = lhs.value > static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for Nan > valid_number is different than
      // pure programming languages
      if (cuda_utils::is_nan(lhs.value) && !cuda_utils::is_nan(rhs.value))
          [[unlikely]] {
        result.value = true;
      } else {
        result.value = lhs.value > rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value > rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T1, typename T2>
__device__ __forceinline__ Bool gteq(C *__restrict__ _ctx, const T1 &lhs,
                                     const T2 &rhs) {
  Bool result;
  result.valid = lhs.valid & rhs.valid;

  if (result.valid) [[likely]] {
    if constexpr (cuda_utils::is_same<T1, Int64>::value &&
                  cuda_utils::is_same<T2, UInt64>::value) {
      if (lhs.value < 0) {
        result.value = false;
      } else {
        result.value = static_cast<uint64_t>(lhs.value) >= rhs.value;
      }
    } else if constexpr (cuda_utils::is_same<T1, UInt64>::value &&
                         cuda_utils::is_same<T2, Int64>::value) {
      if (rhs.value < 0) {
        result.value = true;
      } else {
        result.value = lhs.value >= static_cast<uint64_t>(rhs.value);
      }
    } else if constexpr (T1::is_floating && T2::is_floating) {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      // Databases/dataframes behavior for Nan >= valid_number is different than
      // pure programming languages
      if (cuda_utils::is_nan(lhs.value)) [[unlikely]] {
        result.value = true;
      } else {
        result.value = lhs.value >= rhs.value;
      }
    } else {
      static_assert(cuda_utils::is_same<T1, T2>::value == true);
      result.value = lhs.value >= rhs.value;
    }
  }

  return result;
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T bit_and(C *__restrict__ _ctx, const T &a,
                                     const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    result.value = a.value & b.value;
  }

  return result;
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T bit_or(C *__restrict__ _ctx, const T &a,
                                    const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    result.value = a.value | b.value;
  }

  return result;
}

template <bool ErrorMode, typename C, typename T>
__device__ __forceinline__ T mod(C *__restrict__ ctx, const T &a, const T &b) {
  T result;
  result.valid = a.valid & b.valid;

  if (result.valid) [[likely]] {
    if (b.value == 0) [[unlikely]] {
      result.valid = false;
      ctx[0].error_code = KernelError::MOD_BY_ZERO;
      return result;
    }

    if constexpr (ErrorMode) {
      bool overflow = (a.value == a.min() && b.value == -1);

      if (overflow) [[unlikely]] {
        result.valid = false;
        ctx[0].error_code = KernelError::MOD_OVERFLOW;
        return result;
      }

      result.value = a.value % b.value;

    } else {
      result.value = a.value % b.value;
    }
  }

  return result;
}
} // namespace math
