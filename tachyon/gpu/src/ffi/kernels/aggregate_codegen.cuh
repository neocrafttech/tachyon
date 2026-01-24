/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "aggregate.cuh"
#include "column.cuh"
#include "types.cuh"

namespace aggregate_codegen {

template <TypeKind InKind, TypeKind OutKind, typename B, typename C>
__device__ __forceinline__ void min(C *ctx, Column *input, Column *output,
                                    size_t num_rows, uint16_t col_idx) {
  (void)ctx;
  auto agg = aggregate::min<InKind, B>(input[col_idx], num_rows);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0].store<OutKind, B>(0, agg);
  }
}

template <TypeKind InKind, TypeKind OutKind, typename B, typename C>
__device__ __forceinline__ void max(C *ctx, Column *input, Column *output,
                                    size_t num_rows, uint16_t col_idx) {
  (void)ctx;
  auto agg = aggregate::max<InKind, B>(input[col_idx], num_rows);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0].store<OutKind, B>(0, agg);
  }
}

template <bool ErrorMode, TypeKind InKind, TypeKind OutKind, typename B,
          typename C>
__device__ __forceinline__ void sum(C *ctx, Column *input, Column *output,
                                    size_t num_rows, uint16_t col_idx) {
  auto agg =
      aggregate::sum<ErrorMode, InKind, B, C>(ctx, input[col_idx], num_rows);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0].store<OutKind, B>(0, agg);
  }
}

template <bool ErrorMode, TypeKind InKind, TypeKind OutKind, typename B,
          typename C>
__device__ __forceinline__ void avg(C *ctx, Column *input, Column *output,
                                    size_t num_rows, uint16_t col_idx) {
  auto agg =
      aggregate::avg<ErrorMode, InKind, B, C>(ctx, input[col_idx], num_rows);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0].store<OutKind, B>(0, agg);
  }
}

template <TypeKind InKind, typename B, typename C>
__device__ __forceinline__ void count(C *ctx, Column *input, Column *output,
                                      size_t num_rows, uint16_t col_idx) {
  (void)ctx;
  auto agg = aggregate::count<InKind, B>(input[col_idx], num_rows);
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    output[0].store<TypeKind::UInt64, B>(0, agg);
  }
}

} // namespace aggregate_codegen
