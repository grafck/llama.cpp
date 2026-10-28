// ggml-kunlun: host-side weight conversion (see ggml-kunlun-convert.h)

#include "ggml-kunlun-convert.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cmath>
#include <cstring>
#include <vector>

bool kunlun_type_is_converted(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_F64:
        case GGML_TYPE_BF16:
            return false;
        default:
            // quantized types are requantized to int8 on the host; the dispatcher
            // checks the registry entry (src_type) to confirm the conversion happened
            return true;
    }
}

size_t kunlun_int8_payload_size(const struct ggml_tensor * tensor) {
    return (size_t) ggml_nelements(tensor) * sizeof(int8_t);
}

size_t kunlun_weight_alloc_size(const struct ggml_tensor * tensor) {
    if (kunlun_type_is_converted(tensor->type)) {
        return kunlun_int8_payload_size(tensor);
    }
    return ggml_nbytes(tensor);
}

bool kunlun_dequantize_to_f32(const void * data, enum ggml_type type, float * out, int64_t n) {
    switch (type) {
        case GGML_TYPE_Q4_0: dequantize_row_q4_0((const block_q4_0 *) data, out, n); return true;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1((const block_q4_1 *) data, out, n); return true;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0((const block_q5_0 *) data, out, n); return true;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1((const block_q5_1 *) data, out, n); return true;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0((const block_q8_0 *) data, out, n); return true;
        case GGML_TYPE_Q2_K: dequantize_row_q2_K((const block_q2_K *) data, out, n); return true;
        case GGML_TYPE_Q3_K: dequantize_row_q3_K((const block_q3_K *) data, out, n); return true;
        case GGML_TYPE_Q4_K: dequantize_row_q4_K((const block_q4_K *) data, out, n); return true;
        case GGML_TYPE_Q5_K: dequantize_row_q5_K((const block_q5_K *) data, out, n); return true;
        case GGML_TYPE_Q6_K: dequantize_row_q6_K((const block_q6_K *) data, out, n); return true;
        case GGML_TYPE_Q8_K: dequantize_row_q8_K((const block_q8_K *) data, out, n); return true;
        case GGML_TYPE_IQ2_XXS: dequantize_row_iq2_xxs((const block_iq2_xxs *) data, out, n); return true;
        case GGML_TYPE_IQ2_XS:  dequantize_row_iq2_xs((const block_iq2_xs *) data, out, n); return true;
        case GGML_TYPE_IQ2_S:   dequantize_row_iq2_s((const block_iq2_s *) data, out, n); return true;
        case GGML_TYPE_IQ3_XXS: dequantize_row_iq3_xxs((const block_iq3_xxs *) data, out, n); return true;
        case GGML_TYPE_IQ1_S:   dequantize_row_iq1_s((const block_iq1_s *) data, out, n); return true;
        case GGML_TYPE_IQ1_M:   dequantize_row_iq1_m((const block_iq1_m *) data, out, n); return true;
        case GGML_TYPE_IQ4_NL:  dequantize_row_iq4_nl((const block_iq4_nl *) data, out, n); return true;
        case GGML_TYPE_IQ4_XS:  dequantize_row_iq4_xs((const block_iq4_xs *) data, out, n); return true;
        case GGML_TYPE_IQ3_S:   dequantize_row_iq3_s((const block_iq3_s *) data, out, n); return true;
        case GGML_TYPE_TQ1_0:   dequantize_row_tq1_0((const block_tq1_0 *) data, out, n); return true;
        case GGML_TYPE_TQ2_0:   dequantize_row_tq2_0((const block_tq2_0 *) data, out, n); return true;
        case GGML_TYPE_MXFP4:   dequantize_row_mxfp4((const block_mxfp4 *) data, out, n); return true;
        case GGML_TYPE_NVFP4:   dequantize_row_nvfp4((const block_nvfp4 *) data, out, n); return true;
        case GGML_TYPE_Q1_0:    dequantize_row_q1_0((const block_q1_0 *) data, out, n); return true;
        case GGML_TYPE_Q2_0:    dequantize_row_q2_0((const block_q2_0 *) data, out, n); return true;
        default:
            return false;
    }
}

float kunlun_quantize_f32_to_int8(const float * x, int64_t n, int8_t * out, float * scale) {
    float max_abs = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > max_abs) {
            max_abs = a;
        }
    }
    if (max_abs == 0.0f) {
        // all-zero tensor: any scale works, keep 1 to avoid divide-by-zero
        max_abs = 1.0f;
    }
    float s = 127.0f / max_abs;
    for (int64_t i = 0; i < n; i++) {
        // + copysign(roundf, 0) keeps -0.0 -> 0
        float v = x[i] * s;
        int32_t q = (int32_t) (v >= 0 ? roundf(v) : -roundf(-v));
        if (q > 127)  q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t) q;
    }
    if (scale) {
        *scale = max_abs / 127.0f;
    }
    return max_abs;
}

// block granularity each *_ref quantizer asserts on (ggml-quants.c); a wrong
// k corrupts the read-back blocks
static int kunlun_quant_block_size(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1: case GGML_TYPE_IQ4_NL: case GGML_TYPE_MXFP4:
            return QK8_0;   // 32
        case GGML_TYPE_Q1_0: return QK1_0;   // 128
        case GGML_TYPE_Q2_0: return QK2_0;   // 64
        case GGML_TYPE_Q2_K: case GGML_TYPE_Q3_K: case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS: case GGML_TYPE_TQ1_0: case GGML_TYPE_TQ2_0:
            return QK_K;    // 256
        case GGML_TYPE_NVFP4: return QK_NVFP4;  // 64
        default:
            return 0;  // no host ref quantizer for this type
    }
}

bool kunlun_requantize_int8(const int8_t * q, float scale, enum ggml_type orig_type,
                            int64_t n, void * dst, size_t dst_size) {
    // read-back path (get_tensor / model save): dequant int8 -> fp32 per block,
    // then block-quantize with the ggml host ref quantizer of orig_type. the
    // requantized blocks are close to the originals but not bit-identical
    if (orig_type == GGML_TYPE_F32 || orig_type == GGML_TYPE_F16) {
        return false;
    }
    const int k = kunlun_quant_block_size(orig_type);
    if (k == 0 || (n % k) != 0 || dst_size < ggml_row_size(orig_type, n)) {
        return false;
    }
    std::vector<float> row(k);
    for (int64_t blk = 0; blk < n / k; blk++) {
        for (int64_t i = 0; i < k; i++) {
            row[i] = (float) q[blk * k + i] * scale;
        }
        switch (orig_type) {
            case GGML_TYPE_Q1_0: quantize_row_q1_0_ref(row.data(), (block_q1_0 *) dst + blk, k); break;
            case GGML_TYPE_Q2_0: quantize_row_q2_0_ref(row.data(), (block_q2_0 *) dst + blk, k); break;
            case GGML_TYPE_Q4_0: quantize_row_q4_0_ref(row.data(), (block_q4_0 *) dst + blk, k); break;
            case GGML_TYPE_Q4_1: quantize_row_q4_1_ref(row.data(), (block_q4_1 *) dst + blk, k); break;
            case GGML_TYPE_Q5_0: quantize_row_q5_0_ref(row.data(), (block_q5_0 *) dst + blk, k); break;
            case GGML_TYPE_Q5_1: quantize_row_q5_1_ref(row.data(), (block_q5_1 *) dst + blk, k); break;
            case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(row.data(), (block_q8_0 *) dst + blk, k); break;
            case GGML_TYPE_Q8_1: quantize_row_q8_1_ref(row.data(), (block_q8_1 *) dst + blk, k); break;
            case GGML_TYPE_Q2_K: quantize_row_q2_K_ref(row.data(), (block_q2_K *) dst + blk, k); break;
            case GGML_TYPE_Q3_K: quantize_row_q3_K_ref(row.data(), (block_q3_K *) dst + blk, k); break;
            case GGML_TYPE_Q4_K: quantize_row_q4_K_ref(row.data(), (block_q4_K *) dst + blk, k); break;
            case GGML_TYPE_Q5_K: quantize_row_q5_K_ref(row.data(), (block_q5_K *) dst + blk, k); break;
            case GGML_TYPE_Q6_K: quantize_row_q6_K_ref(row.data(), (block_q6_K *) dst + blk, k); break;
            case GGML_TYPE_Q8_K: quantize_row_q8_K_ref(row.data(), (block_q8_K *) dst + blk, k); break;
            case GGML_TYPE_IQ3_XXS: quantize_row_iq3_xxs_ref(row.data(), (block_iq3_xxs *) dst + blk, k); break;
            case GGML_TYPE_IQ4_NL:  quantize_row_iq4_nl_ref(row.data(), (block_iq4_nl *) dst + blk, k); break;
            case GGML_TYPE_IQ4_XS:  quantize_row_iq4_xs_ref(row.data(), (block_iq4_xs *) dst + blk, k); break;
            case GGML_TYPE_IQ3_S:   quantize_row_iq3_s_ref(row.data(), (block_iq3_s *) dst + blk, k); break;
            case GGML_TYPE_IQ2_S:   quantize_row_iq2_s_ref(row.data(), (block_iq2_s *) dst + blk, k); break;
            case GGML_TYPE_TQ1_0:   quantize_row_tq1_0_ref(row.data(), (block_tq1_0 *) dst + blk, k); break;
            case GGML_TYPE_TQ2_0:   quantize_row_tq2_0_ref(row.data(), (block_tq2_0 *) dst + blk, k); break;
            case GGML_TYPE_MXFP4:   quantize_row_mxfp4_ref(row.data(), (block_mxfp4 *) dst + blk, k); break;
            case GGML_TYPE_NVFP4:   quantize_row_nvfp4_ref(row.data(), (block_nvfp4 *) dst + blk, k); break;
            default:
                return false;
        }
    }
    return true;
}
