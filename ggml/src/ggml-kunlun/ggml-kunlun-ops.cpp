// ggml-kunlun: xdnn compute helpers.
//
// Kept in a separate translation unit on purpose: xpu/refactor/util/float16.h
// defines fp32_from_bits/fp32_to_bits, which clash with the same names in
// ggml-impl.h. This file never includes ggml-impl.h.

#include "ggml.h"
#include "ggml-kunlun-ops.h"

#include <cstdint>
#include <vector>

#include <xpu/runtime.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;
using f16 = float16;

struct kunlun_ops_state {
    xdnn::Context * ctx;
    void * x16;
    void * y16;
    size_t x16_cap;
    size_t y16_cap;
};

static std::vector<kunlun_ops_state> g_ops;

static kunlun_ops_state * ops_state(int device) {
    if ((int) g_ops.size() <= device) {
        g_ops.resize(device + 1, { nullptr, nullptr, nullptr, 0, 0 });
    }
    kunlun_ops_state & s = g_ops[device];
    if (s.ctx == nullptr) {
        xpu_set_device(device);
        s.ctx = xdnn::create_context();
    }
    return s.ctx ? &s : nullptr;
}

static void * scratch(void ** ptr, size_t * cap, size_t bytes) {
    if (*cap >= bytes) {
        return *ptr;
    }
    if (*ptr) {
        xpu_free(*ptr);
    }
    *ptr = nullptr;
    *cap = 0;
    if (xpu_malloc(ptr, (uint64_t) bytes) != 0) {
        return nullptr;
    }
    *cap = bytes;
    return *ptr;
}

// y[m,n] = x[m,k] * w[n,k]^T. ggml weight tensors are [k,n] row-major, which is
// the [n,k] matrix fc reads with w_trans=true. int8 weights need w_maxptr: fc
// dequantizes as w = q * max_abs / 127. Returns 0 on success.
int kunlun_ops_mul_mat(int device, const void * x, const void * w, void * y,
                       int64_t m, int64_t n, int64_t k,
                       enum kunlun_tensor_fmt w_fmt, const float * w_maxptr,
                       enum ggml_type x_type, enum ggml_type y_type) {
    kunlun_ops_state * s = ops_state(device);
    if (s == nullptr) {
        return -1;
    }

    int rc;

    if (w_fmt == KUNLUN_FMT_INT8) {
        if (w_maxptr == nullptr) {
            return -2;
        }
        if (x_type == GGML_TYPE_F32 && y_type == GGML_TYPE_F32) {
            rc = xdnn::fc<float, int8_t, float, int8_t>(s->ctx, (const float *) x,
                    (const int8_t *) w, (float *) y, m, n, k, false, true, nullptr, w_maxptr, nullptr);
        } else if (x_type == GGML_TYPE_F16) {
            void * y16 = scratch(&s->y16, &s->y16_cap, (size_t) m * n * 2);
            if (y16 == nullptr) {
                return -3;
            }
            rc = xdnn::fc<f16, int8_t, f16, int8_t>(s->ctx, (const f16 *) x,
                    (const int8_t *) w, (f16 *) y16, m, n, k, false, true, nullptr, w_maxptr, nullptr);
            if (rc == 0 && y_type == GGML_TYPE_F32) {
                rc = xdnn::cast<f16, float>(s->ctx, (const f16 *) y16, (float *) y, m * n);
            } else if (rc == 0 && y_type == GGML_TYPE_F16) {
                rc = xdnn::copy<f16>(s->ctx, (const f16 *) y16, (f16 *) y, m * n);
            } else if (rc == 0) {
                return -4;
            }
        } else {
            return -4;
        }
    } else if (w_fmt == KUNLUN_FMT_F32) {
        if (x_type != GGML_TYPE_F32 || y_type != GGML_TYPE_F32) {
            return -4;
        }
        rc = xdnn::fc<float, float, float, float>(s->ctx, (const float *) x,
                (const float *) w, (float *) y, m, n, k, false, true, nullptr, nullptr, nullptr);
    } else if (w_fmt == KUNLUN_FMT_F16) {
        const f16 * x16;
        if (x_type == GGML_TYPE_F16) {
            x16 = (const f16 *) x;
        } else if (x_type == GGML_TYPE_F32) {
            void * xs = scratch(&s->x16, &s->x16_cap, (size_t) m * k * 2);
            if (xs == nullptr) {
                return -3;
            }
            rc = xdnn::cast<float, f16>(s->ctx, (const float *) x, (f16 *) xs, m * k);
            if (rc != 0) {
                return rc;
            }
            x16 = (const f16 *) xs;
        } else {
            return -4;
        }
        if (y_type == GGML_TYPE_F16) {
            rc = xdnn::fc<f16, f16, f16, short>(s->ctx, x16, (const f16 *) w,
                    (f16 *) y, m, n, k, false, true, nullptr, nullptr, nullptr);
        } else if (y_type == GGML_TYPE_F32) {
            void * ys = scratch(&s->y16, &s->y16_cap, (size_t) m * n * 2);
            if (ys == nullptr) {
                return -3;
            }
            rc = xdnn::fc<f16, f16, f16, short>(s->ctx, x16, (const f16 *) w,
                    (f16 *) ys, m, n, k, false, true, nullptr, nullptr, nullptr);
            if (rc == 0) {
                rc = xdnn::cast<f16, float>(s->ctx, (const f16 *) ys, (float *) y, m * n);
            }
        } else {
            return -4;
        }
    } else {
        return -5;
    }

    xpu_wait();
    return rc;
}
