/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */
#include "error.h"

const char *kernelGetErrorString(KernelError errorCode) {
  switch (errorCode) {
  case KernelError::NONE:
    return "Kernel success";
  case KernelError::ADD_OVERFLOW:
    return "Addition overflow";
  case KernelError::SUB_OVERFLOW:
    return "Subtraction overflow";
  case KernelError::MUL_OVERFLOW:
    return "Multiplication overflow";
  case KernelError::DIV_OVERFLOW:
    return "Division overflow";
  case KernelError::DIV_BY_ZERO:
    return "Division by zero";
  case KernelError::MOD_OVERFLOW:
    return "Modulo overflow";
  case KernelError::MOD_BY_ZERO:
    return "Modulo by zero";
  default:
    return "Unknown custom kernel error code";
  }
}
