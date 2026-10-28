// Direct test of the ggml-kunlun MUL_MAT activation staging paths.
//
// Bypasses the backend scheduler: weights and activations are placed in
// Kunlun device buffers by hand, the graph is run through
// ggml_backend_kunlun graph_compute, and the result is compared to a
// CPU fp32 reference.
//
// Cases (all f16 weights on PD 0):
//   A. x f32 on device  -> xdnn::cast f32->f16 into scratch (cast path)
//   B. x f16 contiguous on device -> rows alias src1->data (no conversion)
//   C. x f16 strided view on device -> host row-pack staging, H2D (pack path)
//   D. x f32 on host -> host f32->f16 staging, H2D (host path)
//   E. dst f32 on device -> fc writes scratch, D2H + scalar widen (scratch out)
//   F. m=1 decode shape and m=16 prefill shape
//
// Build: g++ tests/kunlun-staging-test.cpp -I build-kunlun -L build-kunlun/lib
//        -l ggml -o build-kunlun/kunlun-staging-test

#include "ggml.h"
#include "ggml-backend-impl.h"
#include "ggml-kunlun.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

static std::vector<float> frand(size_t n, unsigned seed) {
    std::vector<float> v(n);
    std::srand(seed);
    for (size_t i = 0; i < n; i++) {
        v[i] = (float) (std::rand() % 2000 - 1000) / 1000.0f;
    }
    return v;
}

static float f16_to_f32(uint16_t h) {
    // IEEE binary16 -> float
    uint32_t sign = (uint32_t) (h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1f, m = h & 0x3ff;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else {
            int sh = 0;
            while (!(m & 0x400)) { m <<= 1; sh++; }
            m &= 0x3ff;
            bits = sign | ((uint32_t) (127 - 15 - sh) << 23) | (m << 13);
        }
    } else if (e == 31) {
        bits = sign | 0x7f800000 | (m << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static uint16_t f32_to_f16_bits(float f) {
    _Float16 h = (_Float16) f;  // IEEE binary16, RNE - same rounding as xpu ctor
    uint16_t bits;
    memcpy(&bits, &h, 2);
    return bits;
}

struct ctx_holder {
    ggml_context * c = nullptr;
};

// alloc a kunlun device buffer for tensor t (host-filled later via set)
static bool upload_to_dev(ggml_context * ctx, ggml_backend_buffer_type_t buft,
                          ggml_backend_buffer_t * keep, struct ggml_tensor * t,
                          const void * host_data) {
    *keep = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(t));
    if (!*keep) {
        return false;
    }
    t->data = ggml_backend_buffer_get_base(*keep);
    t->buffer = *keep;
    if (host_data) {
        ggml_backend_tensor_set(t, host_data, 0, ggml_nbytes(t));
    }
    return true;
}

int main(int argc, char ** argv) {
    GGML_UNUSED(argc); GGML_UNUSED(argv);
    ggml_backend_t backend = ggml_backend_kunlun_init(0, nullptr);
    if (!backend) {
        fprintf(stderr, "kunlun device 0 unavailable\n");
        return 1;
    }
    ggml_backend_dev_t dev = backend->device;
    ggml_backend_buffer_type_t buft = dev->iface.get_buffer_type(dev);

    size_t mem = 64ull << 20;
    ggml_context * ctx;
    ggml_init_params p{ mem, nullptr, true };
    ctx = ggml_init(p);

    int fails = 0;
    const int64_t k = 1024, n = 512;
    std::vector<float> w32 = frand(k * n, 1);
    std::vector<uint16_t> w16(k * n);
    for (size_t i = 0; i < w16.size(); i++) w16[i] = f32_to_f16_bits(w32[i]);

    // weights: F16 [k,n] on PD 0 (row-major [n][k] = fc W, w_trans=true)
    ggml_backend_buffer_t buf_w;
    struct ggml_tensor * W = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
    if (!upload_to_dev(ctx, buft, &buf_w, W, w16.data())) return 2;

    for (int64_t m : {1, 16}) {
        std::vector<float> x32((size_t) k * m, 0.0f);
        x32 = frand((size_t) k * m, 42 + m);
        std::vector<uint16_t> x16(x32.size());
        for (size_t i = 0; i < x16.size(); i++) x16[i] = f32_to_f16_bits(x32[i]);

        // CPU fp32 reference (dst [n,m] col-major: elem (j,i) = dot W row j, x row i)
        std::vector<float> ref((size_t) n * m);
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                double acc = 0;
                for (int64_t t = 0; t < k; t++) {
                    acc += (double) f16_to_f32(w16[j * k + t]) * (double) x32[i * k + t];
                }
                ref[j + i * n] = (float) acc;
            }
        }

        struct case_spec {
            const char * name;
            ggml_type x_type;
            bool x_on_dev;
            bool x_strided;
            ggml_type dst_type;
        };
        const case_spec cases[] = {
            {"A f32-dev",   GGML_TYPE_F32, true,  false, GGML_TYPE_F16},
            {"B f16-dev",   GGML_TYPE_F16, true,  false, GGML_TYPE_F16},
            {"C f16-str",   GGML_TYPE_F16, true,  true,  GGML_TYPE_F16},
            {"D f32-host",  GGML_TYPE_F32, false, false, GGML_TYPE_F16},
            {"E f32-dst",   GGML_TYPE_F16, true,  false, GGML_TYPE_F32},
        };

        for (const case_spec & cs : cases) {
            // X (activation)
            ggml_context * cx = ctx;
            struct ggml_tensor * X, * Xsrc = nullptr;
            ggml_backend_buffer_t buf_x = nullptr;
            std::vector<uint16_t> xbig;
            int64_t rows = cs.x_strided ? m + 3 : m;
            if (cs.x_strided) {
                // [k, m+3] f16 device base, viewed with row stride skipping first 3 rows
                xbig.assign((size_t) k * rows, 0);
                for (int64_t i = 0; i < m; i++) {
                    for (int64_t t = 0; t < k; t++) {
                        xbig[(size_t) (i + 3) * k + t] = x16[i * k + t];
                    }
                }
                Xsrc = ggml_new_tensor_2d(cx, GGML_TYPE_F16, k, rows);
                if (!upload_to_dev(cx, buft, &buf_x, Xsrc, xbig.data())) return 2;
                X = ggml_view_2d(cx, Xsrc, k, m, 2 * k, 3 * 2 * k);
            } else {
                X = ggml_new_tensor_2d(cx, cs.x_type, k, m);
                const void * host_x = cs.x_type == GGML_TYPE_F32
                    ? (const void *) x32.data() : (const void *) x16.data();
                if (cs.x_on_dev) {
                    if (!upload_to_dev(cx, buft, &buf_x, X, host_x)) return 2;
                } else {
                    X->data = (void *) host_x;  // host activation: mul_mat reads rows directly
                }
            }

            // dst
            ggml_context * cd = ctx;
            struct ggml_tensor * dst = ggml_mul_mat(cx, W, X);
            if (!dst) return 3;
            if (cs.dst_type == GGML_TYPE_F32) {
                dst = ggml_new_tensor_2d(cd, GGML_TYPE_F32, n, m);
                dst->op = GGML_OP_MUL_MAT;
                dst->src[0] = W;
                dst->src[1] = X;
                dst->nb[0] = ggml_type_size(GGML_TYPE_F32);
                dst->nb[1] = n * dst->nb[0];
            }
            ggml_backend_buffer_t buf_d = nullptr;
            if (cs.x_on_dev) {
                if (!upload_to_dev(cx, buft, &buf_d, dst, nullptr)) return 2;
            }

            // run
            ggml_cgraph * g = ggml_new_graph(cx);
            ggml_graph_add_node(g, dst);
            enum ggml_status st = backend->iface.graph_compute(backend, g);

            // read back
            std::vector<float> got((size_t) n * m, 0.0f);
            if (st != GGML_STATUS_SUCCESS) {
                printf("  %s m=%lld: FAILED status=%d\n", cs.name, (long long) m, (int) st);
                fails++;
                continue;
            }
            ggml_backend_tensor_get(dst, got.data(), 0, ggml_nbytes(dst));
            double max_err = 0;
            for (size_t i = 0; i < got.size(); i++) {
                max_err = std::max(max_err, (double) std::fabs(got[i] - ref[i]));
            }
            bool ok = max_err < 0.1 * std::max(1.0, (double) k) * 0.02;  // fp16 GEMM tol
            printf("  %s m=%lld: max_err=%.4g %s\n", cs.name, (long long) m, max_err,
                   ok ? "OK" : "BAD");
            if (!ok) fails++;
            ggml_backend_buffer_free(buf_d);
            ggml_backend_buffer_free(buf_x);
        }
    }
    ggml_backend_buffer_free(buf_w);
    ggml_free(ctx);
    backend->iface.free(backend);
    printf("%s: %d failures\n", fails ? "FAIL" : "PASS", fails);
    return fails != 0;
}
