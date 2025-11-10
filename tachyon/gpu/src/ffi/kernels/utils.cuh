/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifdef __CUDA_ARCH__
#define ASSERT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("CUDA ASSERT FAILED: %s\nFile: %s, Line: %d, Thread: %d\n", msg,  \
             __FILE__, __LINE__, threadIdx.x + blockIdx.x * blockDim.x);       \
      asm("trap;");                                                            \
    }                                                                          \
  } while (0)
#else
#define ASSERT(cond, msg)                                                      \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "HOST ASSERT FAILED: %s\nFile: %s, Line: %d\n", msg,     \
              __FILE__, __LINE__);                                             \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)
#endif
