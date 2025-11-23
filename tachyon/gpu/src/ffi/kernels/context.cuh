/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

enum class ErrorCode : uint32_t {
  NONE = 0,
  ADD_OVERFLOW = 1,
  SUB_OVERFLOW = 2,
  MUL_OVERFLOW = 3,
  DIV_OVERFLOW = 4,
  DIV_BY_ZERO = 5,
  MOD_OVERFLOW = 6,
  MOD_BY_ZERO = 7,
};

struct Context {
  ErrorCode error_code;
  __device__ Context() : error_code(ErrorCode::NONE) {}
};
