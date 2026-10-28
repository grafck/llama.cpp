// Unit tests for kunlun-quant.h: scale correctness, layout, roundtrip error.
// Pure host-side, no device calls. Build: g++ -O2 -std=c++17 this file; run it.

#include "kunlun-quant.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int failures = 0;

#define CHECK(cond, msg)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);         \
            failures++;                                                     \
        }                                                                   \
    } while (0)

static void test_f16_codec() {
    CHECK(kunlun_f16_to_f32(0x0000) == 0.0f, "f16 +0");
    CHECK(kunlun_f16_to_f32(0x8000) == -0.0f, "f16 -0");
    CHECK(kunlun_f16_to_f32(0x3C00) == 1.0f, "f16 1.0");
    CHECK(kunlun_f16_to_f32(0xC000) == -2.0f, "f16 -2.0");
    CHECK(kunlun_f16_to_f32(0x0001) == 5.9604644775390625e-8f, "f16 min subnormal");
    CHECK(kunlun_f16_to_f32(0x03FF) == 6.097555160522461e-5f, "f16 max subnormal");
    CHECK(std::isinf(kunlun_f16_to_f32(0x7C00)), "f16 inf");
    CHECK(std::isnan(kunlun_f16_to_f32(0x7C01)), "f16 nan");
    CHECK(kunlun_f16_to_f32(0x3555) == 0.333251953125f, "f16 1/3-ish");

    CHECK(kunlun_f32_to_f16(0.0f) == 0x0000, "f32->f16 +0");
    CHECK(kunlun_f32_to_f16(-0.0f) == 0x8000, "f32->f16 -0");
    CHECK(kunlun_f32_to_f16(1.0f) == 0x3C00, "f32->f16 1.0");
    CHECK(kunlun_f32_to_f16(-2.5f) == 0xC100, "f32->f16 -2.5");
    CHECK(kunlun_f32_to_f16(1.0f / 3.0f) == 0x3555, "f32->f16 RNE 1/3");
    CHECK(kunlun_f32_to_f16(65504.0f) == 0x7BFF, "f32->f16 f16 max");
    CHECK(kunlun_f32_to_f16(70000.0f) == 0x7C00, "f32->f16 overflow -> inf");
    CHECK(kunlun_f32_to_f16(1e-9f) == 0x0000, "f32->f16 underflow -> 0");
    CHECK(kunlun_f32_to_f16(5.96046448e-8f) == 0x0001, "f32->f16 min subnormal");
    CHECK(kunlun_f32_to_f16(1e30f) == 0x7C00, "f32->f16 huge -> inf");

    // roundtrip over a wide range
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-30.0f, 30.0f);
    for (int i = 0; i < 100000; i++) {
        float v = dist(rng);
        const float r = kunlun_f16_to_f32(kunlun_f32_to_f16(v));
        if (!(std::fabs(r - v) <= std::fabs(v) * 4.9e-4f + 6.2e-6f)) {
            printf("FAIL tol: v=%.9g r=%.9g err=%.3e (file:%d)\n", v, r,
                   std::fabs(r - v), __LINE__);
            failures++;
        }
    }
}

static void test_scale_and_layout() {
    // n=4 (ne0), k=3 (ne1), column-major fp16 source
    const size_t n = 4, k = 3;
    std::vector<uint16_t> w(n * k);
    const float vals[n * k] = {
        0.5f, -0.25f, 3.0f, -7.62f,   // column j=0 (i=0..3)
        0.0f, 1.0f, 2.0f, -7.62f,     // column j=1
        -1.0f, 0.0f, 0.0f, 0.0f,      // column j=2
    };
    // absmax = 7.62 -> nearest f16 value:
    uint16_t max16 = kunlun_f32_to_f16(-7.62f);
    float absmax = std::fabs(kunlun_f16_to_f32(max16));
    for (size_t idx = 0; idx < n * k; idx++) {
        w[idx] = kunlun_f32_to_f16(vals[idx]);
    }

    std::vector<int8_t> q(n * k, 0);
    float scale = kunlun_quant_f16_to_int8(w.data(), n, k, q.data());
    CHECK(!std::isnan(scale), "scale finite");
    CHECK(std::fabs(scale - absmax / 127.0f) < 1e-6f * scale, "scale == absmax/127");

    // the max-magnitude entries quantize to +-127
    CHECK(q[3 * k + 0] == -127, "layout+sign: (i=3,j=0) -> q[3*k] == -127");
    CHECK(q[3 * k + 1] == -127, "layout+sign: (i=3,j=1) -> q[3*k+1] == -127");

    // every code within [-127,127]
    for (size_t idx = 0; idx < n * k; idx++) {
        CHECK(q[idx] >= -127 && q[idx] <= 127, "code range");
    }
    // spot-check codes
    CHECK(q[0 * k + 0] == (int8_t) std::lroundf(0.5f / scale), "code (0,0)");
    CHECK(q[1 * k + 0] == (int8_t) std::lroundf(-0.25f / scale), "code (1,0)");
    CHECK(q[2 * k + 0] == (int8_t) std::lroundf(3.0f / scale), "code (2,0)");
    CHECK(q[0 * k + 2] == (int8_t) std::lroundf(-1.0f / scale), "code (0,2)");

    // dequant -> column-major f16, matches source within scale/2 + f16 rounding
    std::vector<uint16_t> back(n * k, 0);
    kunlun_dequant_int8_to_f16(q.data(), n, k, scale, back.data());
    float tol = scale * 0.5f + scale * 0.002f;
    for (size_t idx = 0; idx < n * k; idx++) {
        float orig = kunlun_f16_to_f32(w[idx]);
        float got  = kunlun_f16_to_f32(back[idx]);
        CHECK(std::fabs(orig - got) <= tol, "roundtrip element tol");
    }
}

static void test_zero_and_uniform() {
    const size_t n = 8, k = 8;
    std::vector<uint16_t> w(n * k, 0x0000);
    std::vector<int8_t> q(n * k, 42);
    float scale = kunlun_quant_f16_to_int8(w.data(), n, k, q.data());
    CHECK(scale == 1.0f, "all-zero tensor -> scale 1");
    for (size_t idx = 0; idx < n * k; idx++) {
        CHECK(q[idx] == 0, "all-zero tensor -> codes 0");
    }

    // uniform constant tensor: absmax = |c|, single code
    std::vector<uint16_t> u(n * k, kunlun_f32_to_f16(-0.125f));
    scale = kunlun_quant_f16_to_int8(u.data(), n, k, q.data());
    CHECK(std::fabs(scale - 0.125f / 127.0f) < 1e-7f, "uniform scale");
    for (size_t idx = 0; idx < n * k; idx++) {
        CHECK(q[idx] == -127, "uniform codes");
    }
}

static void test_nonfinite_rejected() {
    const size_t n = 4, k = 4;
    std::vector<uint16_t> w(n * k, 0x3C00);
    w[5] = 0x7C00; // +inf
    std::vector<int8_t> q(n * k, 0);
    float scale = kunlun_quant_f16_to_int8(w.data(), n, k, q.data());
    CHECK(std::isnan(scale), "non-finite weights -> NaN scale");
}

static void test_random_roundtrip() {
    // realistic weight-ish distribution incl. outliers: absmax must dominate
    const size_t n = 512, k = 256;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    std::vector<uint16_t> w(n * k);
    float absmax = 0.0f;
    for (size_t idx = 0; idx < n * k; idx++) {
        float v = dist(rng);
        if (idx == 12345) {
            v = 0.9f; // outlier
        }
        uint16_t h = kunlun_f32_to_f16(v);
        w[idx] = h;
        float a = std::fabs(kunlun_f16_to_f32(h));
        if (a > absmax) {
            absmax = a;
        }
    }
    std::vector<int8_t> q(n * k, 0);
    float scale = kunlun_quant_f16_to_int8(w.data(), n, k, q.data());
    CHECK(std::fabs(scale - absmax / 127.0f) < 1e-5f * scale, "random scale == absmax/127");

    // MSE check: int8 per-tensor quant should keep relative MSE small
    std::vector<uint16_t> back(n * k, 0);
    kunlun_dequant_int8_to_f16(q.data(), n, k, scale, back.data());
    double se = 0.0, sw = 0.0;
    float max_err = 0.0f;
    for (size_t idx = 0; idx < n * k; idx++) {
        float orig = kunlun_f16_to_f32(w[idx]);
        float got  = kunlun_f16_to_f32(back[idx]);
        se += (orig - got) * (orig - got);
        sw += orig * orig;
        max_err = std::max(max_err, std::fabs(orig - got));
    }
    // per-tensor int8: uniform rounding, abs RMSE ~ scale/sqrt(12); the 0.9
    // outlier sets the scale so typical 0.02-class weights see ~10% rel err.
    double rmse_abs = std::sqrt(se / (double(n) * double(k)));
    CHECK(rmse_abs <= scale * 0.35, "abs RMSE ~ scale/sqrt(12)");
    CHECK(max_err <= scale * 0.6f, "max element error <= scale/2 (+f16 rounding)");
    printf("random n=%zu k=%zu: scale=%.6f RMSE(abs)=%.6f rel=%.5f max_err=%.6f\n",
           n, k, scale, rmse_abs, std::sqrt(se / sw), max_err);
}

static void test_codec_exhaustive() {
    for (uint32_t b = 0; b <= 0xFFFF; b++) {
        if ((b & 0x7C00) == 0x7C00) continue;  // inf/nan payloads do not round-trip
        float r = kunlun_f16_to_f32((uint16_t) b);
        uint16_t back = kunlun_f32_to_f16(r);
        if (b == 0x8000) {  // -0 roundtrips through +0 comparison edge
            CHECK(back == 0x8000 || back == 0x0000, "f16 -0 roundtrip");
            continue;
        }
        if (!(back == (uint16_t) b)) {
            printf("FAIL: roundtrip 0x%04X -> %g -> 0x%04X (codec_test:%d)\n",
                   b, r, back, __FILE__, __LINE__);
            failures++;
        }
    }
}

int main() {
    test_f16_codec();
    test_codec_exhaustive();
    test_scale_and_layout();
    test_zero_and_uniform();
    test_nonfinite_rejected();
    test_random_roundtrip();
    if (failures) {
        printf("%d FAILURES\n", failures);
        return 1;
    }
    printf("all kunlun-quant tests passed\n");
    return 0;
}
