/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "column.cuh"
#include "types.cuh"

namespace string_ops {

__device__ __forceinline__ uint32_t pack_prefix(const char *data,
                                                uint32_t size) {
  uint32_t prefix = 0;
  const uint32_t take =
      size < STRING_INLINE_PREFIX_BYTES ? size : STRING_INLINE_PREFIX_BYTES;
  for (uint32_t i = 0; i < take; i++) {
    prefix |= (static_cast<uint32_t>(static_cast<uint8_t>(data[i])) << (i * 8));
  }
  return prefix;
}

__device__ __forceinline__ const char *
view_data_ptr(const StringView &sv, const Column &col, char *inline_buf) {
  if (sv.size <= STRING_INLINE_TOTAL_BYTES) {
    for (uint32_t i = 0; i < STRING_INLINE_PREFIX_BYTES; i++) {
      inline_buf[i] = static_cast<char>((sv.prefix >> (i * 8)) & 0xFF);
    }
    for (uint32_t i = 0; i < STRING_INLINE_DATA_BYTES; i++) {
      inline_buf[STRING_INLINE_PREFIX_BYTES + i] =
          static_cast<char>((sv.data >> (i * 8)) & 0xFF);
    }
    return inline_buf;
  }
  ASSERT(col.string_buffer != nullptr,
         "Missing string buffer for external string");
  return reinterpret_cast<const char *>(col.string_buffer) +
         static_cast<size_t>(sv.data);
}

__device__ __forceinline__ bool write_string_view(const char *src, uint32_t len,
                                                  Column &out_col,
                                                  size_t row_idx,
                                                  StringView &out_view) {
  out_view.size = len;
  out_view.prefix = pack_prefix(src, len);

  if (len <= STRING_INLINE_TOTAL_BYTES) {
    uint64_t payload = 0;
    for (uint32_t i = STRING_INLINE_PREFIX_BYTES; i < len; i++) {
      payload |= (static_cast<uint64_t>(static_cast<uint8_t>(src[i]))
                  << ((i - STRING_INLINE_PREFIX_BYTES) * 8));
    }
    out_view.data = payload;
    return true;
  }

  if (out_col.string_buffer == nullptr || out_col.string_buffer_size == 0 ||
      out_col.size == 0) {
    return false;
  }

  size_t stride = out_col.string_buffer_size / out_col.size;
  if (stride < len) {
    return false;
  }

  const size_t offset = row_idx * stride;
  char *dst = reinterpret_cast<char *>(out_col.string_buffer) + offset;
  for (uint32_t i = 0; i < len; i++) {
    dst[i] = src[i];
  }
  out_view.data = static_cast<uint64_t>(offset);
  return true;
}

__device__ __forceinline__ uint32_t utf8_sequence_len(uint8_t lead) {
  if ((lead & 0x80u) == 0)
    return 1;
  if ((lead & 0xE0u) == 0xC0u)
    return 2;
  if ((lead & 0xF0u) == 0xE0u)
    return 3;
  if ((lead & 0xF8u) == 0xF0u)
    return 4;
  return 1;
}

__device__ __forceinline__ bool utf8_is_cont(uint8_t b) {
  return (b & 0xC0u) == 0x80u;
}

__device__ __forceinline__ void utf8_decode_next(const char *src, uint32_t len,
                                                 uint32_t offset,
                                                 uint32_t &codepoint,
                                                 uint32_t &next_offset) {
  if (offset >= len) {
    codepoint = 0;
    next_offset = offset;
    return;
  }

  const uint8_t b0 = static_cast<uint8_t>(src[offset]);
  const uint32_t n = utf8_sequence_len(b0);

  if (n == 1 || offset + n > len) {
    codepoint = b0;
    next_offset = offset + 1;
    return;
  }

  if (n == 2) {
    const uint8_t b1 = static_cast<uint8_t>(src[offset + 1]);
    if (!utf8_is_cont(b1)) {
      codepoint = b0;
      next_offset = offset + 1;
      return;
    }
    codepoint = ((b0 & 0x1Fu) << 6) | (b1 & 0x3Fu);
    next_offset = offset + 2;
    return;
  }

  if (n == 3) {
    const uint8_t b1 = static_cast<uint8_t>(src[offset + 1]);
    const uint8_t b2 = static_cast<uint8_t>(src[offset + 2]);
    if (!utf8_is_cont(b1) || !utf8_is_cont(b2)) {
      codepoint = b0;
      next_offset = offset + 1;
      return;
    }
    codepoint = ((b0 & 0x0Fu) << 12) | ((b1 & 0x3Fu) << 6) | (b2 & 0x3Fu);
    next_offset = offset + 3;
    return;
  }

  const uint8_t b1 = static_cast<uint8_t>(src[offset + 1]);
  const uint8_t b2 = static_cast<uint8_t>(src[offset + 2]);
  const uint8_t b3 = static_cast<uint8_t>(src[offset + 3]);
  if (!utf8_is_cont(b1) || !utf8_is_cont(b2) || !utf8_is_cont(b3)) {
    codepoint = b0;
    next_offset = offset + 1;
    return;
  }
  codepoint = ((b0 & 0x07u) << 18) | ((b1 & 0x3Fu) << 12) |
              ((b2 & 0x3Fu) << 6) | (b3 & 0x3Fu);
  next_offset = offset + 4;
}

__device__ __forceinline__ uint32_t utf8_encode(uint32_t cp, char *dst) {
  if (cp <= 0x7F) {
    dst[0] = static_cast<char>(cp);
    return 1;
  }
  if (cp <= 0x7FF) {
    dst[0] = static_cast<char>(0xC0u | ((cp >> 6) & 0x1Fu));
    dst[1] = static_cast<char>(0x80u | (cp & 0x3Fu));
    return 2;
  }
  if (cp <= 0xFFFF) {
    dst[0] = static_cast<char>(0xE0u | ((cp >> 12) & 0x0Fu));
    dst[1] = static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
    dst[2] = static_cast<char>(0x80u | (cp & 0x3Fu));
    return 3;
  }

  if (cp > 0x10FFFF) {
    cp = static_cast<uint32_t>('?');
  }
  dst[0] = static_cast<char>(0xF0u | ((cp >> 18) & 0x07u));
  dst[1] = static_cast<char>(0x80u | ((cp >> 12) & 0x3Fu));
  dst[2] = static_cast<char>(0x80u | ((cp >> 6) & 0x3Fu));
  dst[3] = static_cast<char>(0x80u | (cp & 0x3Fu));
  return 4;
}

__device__ __forceinline__ uint32_t unicode_to_lower(uint32_t cp) {
  if (cp >= 'A' && cp <= 'Z')
    return cp + 32;
  if ((cp >= 0x00C0 && cp <= 0x00D6) || (cp >= 0x00D8 && cp <= 0x00DE))
    return cp + 32;
  if (cp >= 0x0391 && cp <= 0x03A1)
    return cp + 32;
  if (cp >= 0x03A3 && cp <= 0x03AB)
    return cp + 32;
  if (cp >= 0x0410 && cp <= 0x042F)
    return cp + 32;
  if (cp == 0x0401)
    return 0x0451;
  return cp;
}

__device__ __forceinline__ uint32_t unicode_to_upper(uint32_t cp) {
  if (cp >= 'a' && cp <= 'z')
    return cp - 32;
  if ((cp >= 0x00E0 && cp <= 0x00F6) || (cp >= 0x00F8 && cp <= 0x00FE))
    return cp - 32;
  if (cp >= 0x03B1 && cp <= 0x03C1)
    return cp - 32;
  if (cp >= 0x03C3 && cp <= 0x03CB)
    return cp - 32;
  if (cp == 0x03C2)
    return 0x03A3;
  if (cp >= 0x0430 && cp <= 0x044F)
    return cp - 32;
  if (cp == 0x0451)
    return 0x0401;
  return cp;
}

__device__ __forceinline__ uint32_t utf8_codepoint_count(const char *src,
                                                         uint32_t len) {
  uint32_t count = 0;
  uint32_t i = 0;
  while (i < len) {
    uint32_t cp = 0;
    uint32_t next = i;
    utf8_decode_next(src, len, i, cp, next);
    i = next;
    count++;
  }
  return count;
}

__device__ __forceinline__ uint32_t
utf8_byte_offset_for_char(const char *src, uint32_t len, uint32_t char_idx) {
  uint32_t i = 0;
  uint32_t c = 0;
  while (i < len && c < char_idx) {
    uint32_t cp = 0;
    uint32_t next = i;
    utf8_decode_next(src, len, i, cp, next);
    i = next;
    c++;
  }
  return i;
}

__device__ __forceinline__ UInt32 length(const String &s,
                                         const Column &input_col) {
  UInt32 out;
  out.valid = s.valid;
  if (out.valid) {
    char inline_buf[STRING_INLINE_TOTAL_BYTES];
    const StringView &sv = s.value;
    const char *src = view_data_ptr(sv, input_col, inline_buf);
    out.value = utf8_codepoint_count(src, sv.size);
  }
  return out;
}

__device__ __forceinline__ bool
transform_case_utf8(const String &s, const Column &input_col, Column &out_col,
                    size_t row_idx, bool to_upper, StringView &out_view) {
  char inline_buf[STRING_INLINE_TOTAL_BYTES];
  const char *src = view_data_ptr(s.value, input_col, inline_buf);
  const uint32_t in_len = s.value.size;

  char inline_dst[STRING_INLINE_TOTAL_BYTES];
  char *dst = inline_dst;
  size_t stride = 0;
  if (in_len > STRING_INLINE_TOTAL_BYTES) {
    if (out_col.string_buffer == nullptr || out_col.size == 0) {
      return false;
    }
    stride = out_col.string_buffer_size / out_col.size;
    if (stride < in_len) {
      return false;
    }
    dst = reinterpret_cast<char *>(out_col.string_buffer) + row_idx * stride;
  }

  uint32_t in_off = 0;
  uint32_t out_off = 0;
  while (in_off < in_len) {
    uint32_t cp = 0;
    uint32_t next = in_off;
    utf8_decode_next(src, in_len, in_off, cp, next);
    cp = to_upper ? unicode_to_upper(cp) : unicode_to_lower(cp);
    char enc[4];
    uint32_t enc_len = utf8_encode(cp, enc);
    if (out_off + enc_len > in_len) {
      return false;
    }
    for (uint32_t i = 0; i < enc_len; i++) {
      dst[out_off + i] = enc[i];
    }
    out_off += enc_len;
    in_off = next;
  }

  return write_string_view(dst, out_off, out_col, row_idx, out_view);
}

__device__ __forceinline__ String lower(const String &s,
                                        const Column &input_col,
                                        Column &out_col, size_t row_idx) {
  String out;
  out.valid = s.valid;
  if (!out.valid) {
    return out;
  }
  out.valid =
      transform_case_utf8(s, input_col, out_col, row_idx, false, out.value);
  return out;
}

__device__ __forceinline__ String upper(const String &s,
                                        const Column &input_col,
                                        Column &out_col, size_t row_idx) {
  String out;
  out.valid = s.valid;
  if (!out.valid) {
    return out;
  }
  out.valid =
      transform_case_utf8(s, input_col, out_col, row_idx, true, out.value);
  return out;
}

__device__ __forceinline__ String substring(const String &s, int32_t start,
                                            int32_t len,
                                            const Column &input_col,
                                            Column &out_col, size_t row_idx) {
  String out;
  out.valid = s.valid;
  if (!out.valid) {
    return out;
  }

  char inline_buf[STRING_INLINE_TOTAL_BYTES];
  const char *src = view_data_ptr(s.value, input_col, inline_buf);
  const uint32_t byte_len = s.value.size;
  const uint32_t char_count = utf8_codepoint_count(src, byte_len);

  int32_t sidx = start < 0 ? 0 : start;
  if (sidx > static_cast<int32_t>(char_count)) {
    sidx = static_cast<int32_t>(char_count);
  }
  int32_t take = len < 0 ? 0 : len;
  if (sidx + take > static_cast<int32_t>(char_count)) {
    take = static_cast<int32_t>(char_count) - sidx;
  }

  const uint32_t start_byte =
      utf8_byte_offset_for_char(src, byte_len, static_cast<uint32_t>(sidx));
  const uint32_t end_byte = utf8_byte_offset_for_char(
      src, byte_len, static_cast<uint32_t>(sidx + take));
  const uint32_t out_len = end_byte - start_byte;

  if (out_len <= STRING_INLINE_TOTAL_BYTES) {
    char local_buf[STRING_INLINE_TOTAL_BYTES];
    for (uint32_t i = 0; i < out_len; i++) {
      local_buf[i] = src[start_byte + i];
    }
    out.valid =
        write_string_view(local_buf, out_len, out_col, row_idx, out.value);
    return out;
  }

  if (out_col.string_buffer == nullptr || out_col.size == 0) {
    out.valid = false;
    return out;
  }
  size_t stride = out_col.string_buffer_size / out_col.size;
  if (stride < out_len) {
    out.valid = false;
    return out;
  }

  char *row_dst =
      reinterpret_cast<char *>(out_col.string_buffer) + row_idx * stride;
  for (uint32_t i = 0; i < out_len; i++) {
    row_dst[i] = src[start_byte + i];
  }
  out.valid = write_string_view(row_dst, out_len, out_col, row_idx, out.value);
  return out;
}

__device__ __forceinline__ String concat(const String &lhs,
                                         const Column &lhs_col,
                                         const String &rhs,
                                         const Column &rhs_col, Column &out_col,
                                         size_t row_idx) {
  String out;
  out.valid = lhs.valid & rhs.valid;
  if (!out.valid) {
    return out;
  }

  char lhs_inline[STRING_INLINE_TOTAL_BYTES];
  char rhs_inline[STRING_INLINE_TOTAL_BYTES];
  const char *lptr = view_data_ptr(lhs.value, lhs_col, lhs_inline);
  const char *rptr = view_data_ptr(rhs.value, rhs_col, rhs_inline);
  const uint32_t lsize = lhs.value.size;
  const uint32_t rsize = rhs.value.size;
  const uint32_t out_len = lsize + rsize;

  if (out_len <= STRING_INLINE_TOTAL_BYTES) {
    char local_buf[STRING_INLINE_TOTAL_BYTES];
    for (uint32_t i = 0; i < lsize; i++) {
      local_buf[i] = lptr[i];
    }
    for (uint32_t i = 0; i < rsize; i++) {
      local_buf[lsize + i] = rptr[i];
    }
    out.valid =
        write_string_view(local_buf, out_len, out_col, row_idx, out.value);
    return out;
  }

  size_t stride =
      out_col.size == 0 ? 0 : out_col.string_buffer_size / out_col.size;
  if (stride < out_len || out_col.string_buffer == nullptr) {
    out.valid = false;
    return out;
  }
  char *row_dst =
      reinterpret_cast<char *>(out_col.string_buffer) + row_idx * stride;
  for (uint32_t i = 0; i < lsize; i++) {
    row_dst[i] = lptr[i];
  }
  for (uint32_t i = 0; i < rsize; i++) {
    row_dst[lsize + i] = rptr[i];
  }
  out.valid = write_string_view(row_dst, out_len, out_col, row_idx, out.value);
  return out;
}

} // namespace string_ops
