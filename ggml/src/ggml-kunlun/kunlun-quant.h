// Host-side fp16 -> int8 per-tensor symmetric quantization for the Kunlun KL1
// backend. The KL1 device quant helpers return rc=4 (UNSUPPORTED), so weight
// quantization runs on the host at load time. Pure functions, no device calls
// (see tests/kunlun-quant-test.cpp).
//
// Layout contract (stable, consumed by MUL_MAT):
//   ggml weight tensor is [ne0=n][ne1=k] column-major: elem (i,j) at i + j*n.
//   Stored int8 is row-major [n][k]: w_q[i*k + j] = round(w(i,j) / scale).
//   scale = max|w| / 127 per tensor; all-zero tensor -> scale = 1, codes 0.
//   w_maxptr = device float* holding scale (KL1 fc int8 needs a device-side
//   pointer; a host float* returns rc=1 INVALID_PARAM).
//   fc(a[m][k] f16 row-major, w[n][k] int8 row-major, w_trans=true) -> out[m][n].

#ifndef GGML_KUNLUN_QUANT_H
#define GGML_KUNLUN_QUANT_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

static inline float kunlun_f16_to_f32(uint16_t h) {
    uint32_t sign = ((uint32_t) h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t man  = h & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (man == 0) {
            f32 = sign;
        } else {
            // subnormal: value = man * 2^-24; normalize to 1.mant * 2^e
            exp = 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            f32 = sign | ((exp + 112) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        f32 = sign | 0x7F800000u | (man << 13);
    } else {
        f32 = sign | ((exp + 127 - 15) << 23) | (man << 13);
    }
    float out;
    memcpy(&out, &f32, 4);
    return out;
}

static inline uint16_t kunlun_f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint16_t sign = (uint16_t) ((x >> 16) & 0x8000);
    uint32_t exp  = (x >> 23) & 0xFF;
    uint32_t man  = x & 0x7FFFFF;
    if (exp == 0xFF) {
        return sign | (man ? 0x7C00 : 0x7800);
    }
    int32_t e = (int32_t) exp - 127 + 15;
    if (e >= 31) {
        return sign | 0x7C00;
    }
    if (e >= 31) {
        return sign | 0x7C00;
    }
    if (e <= 0) {
        if (e < -11) {
            return sign;  // below half of the min subnormal
        }
        // subnormal target: implicit bit shifts down too (24-bit man includes it)
        int shift = 14 - e;          // bits dropped, in [14, 25]
        man |= 0x800000u;
        uint32_t half    = 1u << (shift - 1);
        uint32_t dropped = man & ((1u << shift) - 1);
        uint16_t q = (uint16_t) (man >> shift);   // 10 or 11 bits, RNE below
        if (dropped > half || (dropped == half && (q & 1))) {
            q++;
        }
        return sign | q;
    }
    uint32_t half    = 1u << 12;
    uint32_t dropped = man & 0x1FFF;
    uint16_t q = (uint16_t) (man >> 13);
    if (dropped > half || (dropped == half && (q & 1))) {
        q++;
    }
    // q can carry into the exponent: 1024 -> exp++, mantissa 0, handled naturally
    return sign | (uint16_t) (((uint32_t) e << 10) + q);
}

// reads column-major fp16 [n][k], writes row-major int8 [n][k].
// returns the per-tensor scale, or NaN if the weights hold non-finite values.
static inline float kunlun_quant_f16_to_int8(const uint16_t * w, size_t n, size_t k, int8_t * out) {
    const size_t total = n * k;
    float absmax = 0.0f;
    for (size_t idx = 0; idx < total; idx++) {
        float v = kunlun_f16_to_f32(w[idx]);
        if (!std::isfinite(v)) {
            return std::nanf("");
        }
        const float a = std::fabs(v);
        if (a > absmax) {
            absmax = a;
        }
    }
    const float scale = absmax > 0.0f ? absmax / 127.0f : 1.0f;
    for (size_t j = 0; j < k; j++) {
        for (size_t i = 0; i < n; i++) {
            float v = kunlun_f16_to_f32(w[i + j * n]);
            int32_t q = (int32_t) std::lroundf(v / scale);
            if (q > 127) {
                q = 127;
            } else if (q < -127) {
                q = -127;
            }
            out[i * k + j] = (int8_t) q;
        }
    }
    return scale;
}

// reads row-major int8 [n][k], writes column-major fp16 [n][k].
static inline void kunlun_dequant_int8_to_f16(const int8_t * q, size_t n, size_t k, float scale, uint16_t * out) {
    for (size_t j = 0; j < k; j++) {
        for (size_t i = 0; i < n; i++) {
            out[i + j * n] = kunlun_f32_to_f16((float) q[i * k + j] * scale);
        }
    }
}

#endif // GGML_KUNLUN_QUANT_H
