// ggml-kunlun activation staging (host-side, device-free logic)
//
// Maps ggml MUL_MAT operands onto the K200 `fc<TX,TW,TY,TGEMM>` fp16 kernel
// with x_trans=false, w_trans=true:
//
//   ggml:  src0 [k,n] (ne0=k, ne1=n, column-major storage), src1 [k,m], dst [n,m]
//   fc:    x = row-major [m][k] fp16, w = row-major [n][k] fp16, y = row-major [m][n] fp16
//
// Because ggml stores tensors with ne[0] contiguous, a *contiguous* ggml F16
// src0 [k,n] is exactly n row-major rows of k halves = fc's W[n][k], and a
// contiguous ggml F16 dst [n,m] is exactly m row-major rows of n halves = fc's
// y[m][n]. Activations src1 [k,m] need row packing when they are strided
// views; when src1 is F32 each row is converted to fp16 (xpu float16 = IEEE
// binary16 with uint16_t storage, RNE — bit-compatible with ggml F16).
//
// Layout proof on hardware: ~/dev/kunlun_k200/benchmarks/llama_spike.cpp
// "fc layout check (w_trans=true, W=[n][k])": max_err < 0.05 vs CPU.
// fc rejects m=0 (rc=1, measured) — callers must early-return on empty batch.

#ifndef GGML_KUNLUN_STAGING_H
#define GGML_KUNLUN_STAGING_H

#include "ggml.h"

#include <xpu/refactor/util/float16.h>
#include <cstring>

// Derive fc dims from ggml MUL_MAT operands. Returns false when the shape is
// unsupported by the v1 fp16 path (broadcast batches, inner-dim mismatch).
// m == 0 is reported as a valid shape (empty batch) — the caller must skip fc.
static inline bool kunlun_mul_mat_dims(const struct ggml_tensor * src0,
                                       const struct ggml_tensor * src1,
                                       int64_t * m, int64_t * n, int64_t * k) {
    if (src0 == nullptr || src1 == nullptr) {
        return false;
    }
    // v1: no batched broadcast (ne2/ne3 > 1); matches the supports_op gate
    if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src1->ne[0] != src0->ne[0]) {
        return false;
    }
    *k = src0->ne[0];
    *n = src0->ne[1];
    *m = src1->ne[1];
    return *k > 0 && *n > 0;
}

// Stage a ggml MUL_MAT activation (src1, F32 or F16, possibly a strided row
// view) as contiguous row-major fp16 [m][k].
//   F16 input -> byte-identical row copy, no conversion.
//   F32 input -> per-element fp16 conversion (RNE) at staging time only.
static inline void kunlun_stage_activation_f16(const struct ggml_tensor * src1,
                                               int64_t m, int64_t k, float16 * out) {
    const char * base = (const char *) src1->data;
    if (src1->type == GGML_TYPE_F16) {
        // ggml F16 is IEEE binary16, bit-compatible with xpu float16: alias raw bytes
        for (int64_t i = 0; i < m; i++) {
            memcpy(out + i * k, base + i * src1->nb[1], (size_t) k * sizeof(float16));
        }
    } else {
        for (int64_t i = 0; i < m; i++) {
            const float * row = (const float *) (base + i * src1->nb[1]);
            for (int64_t t = 0; t < k; t++) {
                out[i * k + t] = float16(row[t]);
            }
        }
    }
}

#endif // GGML_KUNLUN_STAGING_H
