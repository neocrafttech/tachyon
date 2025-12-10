/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

/**
 * @brief Custom error codes for Tachyon CUDA kernels.
 */

typedef enum {
  NONE = 0,
  ADD_OVERFLOW = 1,
  SUB_OVERFLOW = 2,
  MUL_OVERFLOW = 3,
  DIV_OVERFLOW = 4,
  DIV_BY_ZERO = 5,
  MOD_OVERFLOW = 6,
  MOD_BY_ZERO = 7,
} KernelError;

/**
 * @brief Converts a KernelErrorCode into a human-readable string.
 *
 * @param errorCode KernelError.
 * @return A constant C-style string describing the error.
 */
extern "C" {
#ifdef __CUDACC__
__host__ __device__
#endif
    const char *
    kernelGetErrorString(KernelError errorCode);
}
