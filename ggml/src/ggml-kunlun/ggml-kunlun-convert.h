// ggml-kunlun: host-side weight conversion helpers (pure host code, no XPU deps)
//
// KL1's fc() consumes weights as raw fp16, raw fp32 or per-tensor-scaled int8.
// The device quant helper (gpt_fp16_quant_2int8) returns rc=4 on KL1, so GGUF
// quantized weights are requantized to int8 on the host at load time:
// dequantize to fp32, take per-tensor max |w|, store int8 = round(w * 127 / max_abs).
//
// The registry is keyed by the tensor's device pointer (tensor->data) and lives
// in ggml-kunlun.cpp; this header only declares the format contract so op dispatch
// (ggml-kunlun.cpp graph_compute) and unit tests share one definition.

#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

enum kunlun_tensor_fmt {
    KUNLUN_FMT_NONE = 0,   // no registry entry: tensor->data holds ggml-native bytes
    KUNLUN_FMT_F32  = 1,   // tensor->data holds ggml_nbytes() raw fp32 (memcpy upload)
    KUNLUN_FMT_F16  = 2,   // tensor->data holds ggml_nbytes() raw fp16 (aliases GGUF bytes)
    KUNLUN_FMT_INT8 = 3,   // tensor->data holds kunlun_int8_payload_size() int8 quants
};

struct kunlun_weight_entry {
    enum kunlun_tensor_fmt fmt;
    enum ggml_type         src_type;   // original ggml type of the tensor
    float                  max_abs;    // per-tensor max |w| of the dequantized weights
    float                  scale;      // max_abs / 127, i.e. w ~= int8 * scale
    int64_t                n;          // ggml_nelements(tensor)
};

// true if the tensor type is converted to int8 at upload time
bool kunlun_type_is_converted(enum ggml_type type);

// device bytes reserved/stored for a tensor: element count for converted types,
// ggml_nbytes() otherwise. feeds buft get_alloc_size
size_t kunlun_weight_alloc_size(const struct ggml_tensor * tensor);

// payload actually uploaded to tensor->data for a converted tensor (int8 bytes)
size_t kunlun_int8_payload_size(const struct ggml_tensor * tensor);

// dequantize n elements of a quantized buffer to fp32; false for unsupported types
bool kunlun_dequantize_to_f32(const void * data, enum ggml_type type, float * out, int64_t n);

// symmetric per-tensor int8 quantization; returns max_abs, scale = max_abs / 127
float kunlun_quantize_f32_to_int8(const float * x, int64_t n, int8_t * out, float * scale);

// requantize int8 payload back into ggml-native blocks of orig_type (round-trip for
// get_tensor / model save); false if quantize_row_q8_0 does not cover orig_type
bool kunlun_requantize_int8(const int8_t * q, float scale, enum ggml_type orig_type,
                            int64_t n, void * dst, size_t dst_size);
