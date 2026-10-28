#pragma once

// Per-tensor symmetric int8 quantization for Kunlun fc weights.
//
// Contract shared with the KL1 fc<f16,int8,f16,int8> kernel path:
//   - q[i] = round(w[i] * 127 / scale), scale = max(|w|), clamped to [-127, 127]
//   - w_maxptr points to a DEVICE-side float holding scale; the fc kernel
//     dequantizes the int8 dot product using it (scale/127 convention is
//     pinned by tools/kunlun-int8-fc-verify.cpp on real hardware)
//   - weights are stored int8[n*k] row-major, same layout as the fp16
//     weights they replace (fc w_trans=true)

#include <cmath>
#include <cstddef>
#include <cstdint>

static inline void ggml_kunlun_quant_int8(const float * w, size_t n, int8_t * q, float * scale) {
    float mx = 0.f;
    for (size_t i = 0; i < n; i++) {
        const float v = std::fabs(w[i]);
        if (v > mx) {
            mx = v;
        }
    }
    if (!(mx > 0.f)) {
        mx = 1.f;  // all-zero tensor: scale 1 keeps dequant finite
    }

    const float s = 127.f / mx;
    for (size_t i = 0; i < n; i++) {
        float v = w[i] * s;
        if (v > 127.f) {
            v = 127.f;
        } else if (v < -127.f) {
            v = -127.f;
        }
        q[i] = (int8_t) std::lroundf(v);
    }
    *scale = mx;
}
