// 4D f16 KV-cache round-trip test for the Kunlun K200 GGML_OP_SET_ROWS write path.
//
// Every case allocates a 4D f16 cache [head_dim, n_slots, n_head, n_layer] on a
// Kunlun device buffer, seeds it with a sentinel pattern, builds a 2-column idx
// tensor (ne1 = 2, the shape the llama KV-update path uses), runs the SET_ROWS
// node through ggml_backend_graph_compute, then copies the whole cache back to
// host and compares it element by element against an independent IEEE-754
// round-to-nearest-even f32 -> f16 conversion. That covers, in one pass:
//   - the 4D [d, slots, head, layer] cache layout and its nb-based row addressing
//   - 2-column idx indexing (i02 % ne1, i03 % ne2 select the idx column)
//   - the f32 -> f16 row cast: the values fed are deliberately NOT exactly
//     representable in f16 (ties, fractions, subnormals), so truncation instead
//     of round-to-nearest-even is detected
//   - the device-to-host round trip (values/idx read D2H, whole cache read D2H)
//   - which rows are left untouched (they must still hold the sentinel)
//   - out-of-range idx: the write path must refuse the graph rather than write
//     outside the cache, and the device must stay usable afterwards.
//
// Standalone build (no llama needed):
//   c++ -std=gnu++17 -O2 -I ggml/include tests/test-kunlun-set-rows-4d-roundtrip.cpp -o rt
//       -L build-kunlun/bin -Wl,-rpath,build-kunlun/bin -lggml-base -lggml-kunlun
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <vector>

extern "C" ggml_backend_reg_t ggml_backend_kunlun_reg(void);

static int g_pass = 0;
static int g_fail = 0;

// device buffers are not zero-initialised, so every cache is seeded with this
// pattern before the round trip; untouched rows must still hold it afterwards
static const uint16_t k_sentinel = 0xBEEF;

static void report(bool ok, const char * name, const char * detail) {
    printf("%-4s %-56s %s\n", ok ? "PASS" : "FAIL", name, detail);
    (ok ? g_pass : g_fail)++;
}

// ---------------------------------------------------------------------------
// independent reference: IEEE-754 binary16, round-to-nearest-even
// ---------------------------------------------------------------------------
static uint16_t f16_bits(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));

    const int      sign = (int) ((u >> 31) & 1u);
    const int      e    = (int) ((u >> 23) & 0xffu);
    const uint32_t mant = u & 0x007fffffu;

    if (e == 0xff) {
        return (uint16_t) ((sign << 15) | (mant ? 0x7c01u : 0x7c00u));  // NaN / inf
    }
    if (e == 0) {
        return (uint16_t) (sign << 15);  // f32 subnormals are all < 2^-126 -> 0
    }

    const int exp = e - 127;
    if (exp > 15) {
        return (uint16_t) ((sign << 15) | 0x7c00u);  // overflow -> inf
    }

    // exact value as a double (an f32 mantissa always fits in a double)
    const double value = (1.0 + (double) mant / 8388608.0) * std::ldexp(1.0, exp);
    const double ulp   = (exp >= -14) ? std::ldexp(1.0, exp - 10) : 1.0 / 16777216.0; // 2^-24

    double scaled  = value / ulp;
    double fl      = std::floor(scaled);
    double frac    = scaled - fl;
    double rounded = (frac > 0.5) ? fl + 1.0
                                  : (frac == 0.5) ? fl + (std::fmod(fl, 2.0) == 1.0 ? 1.0 : 0.0)
                                  : fl;

    if (exp < -14) {
        if (rounded >= 1024.0) {
            return (uint16_t) ((sign << 15) | (1u << 10));  // exactly 2^-14
        }
        return (uint16_t) ((sign << 15) | (uint16_t) rounded);  // subnormal
    }

    if (rounded >= 2048.0) {
        if (exp + 1 > 15) {
            return (uint16_t) ((sign << 15) | 0x7c00u);
        }
        return (uint16_t) ((sign << 15) | (uint16_t) ((exp + 1 + 15) << 10));
    }
    return (uint16_t) ((sign << 15) | (uint16_t) ((exp + 15) << 10) | (uint16_t) (rounded - 1024.0));
}

static uint16_t host_cast_bits(float x) {
#if defined(__GNUC__)
    _Float16 h = (_Float16) x;
    uint16_t bits;
    memcpy(&bits, &h, sizeof(bits));
    return bits;
#else
    return 0;
#endif
}

static void self_check_reference(void) {
    struct { float v; uint16_t want; const char * what; } known[] = {
        { 0.0f,                     0x0000, "zero" },
        { -0.0f,                    0x8000, "negative zero" },
        { 1.0f,                     0x3C00, "one" },
        { -1.0f,                    0xBC00, "minus one" },
        { 0.1f,                     0x2E66, "0.1" },
        { 1.0f / 3.0f,              0x3555, "1/3" },
        { 123.456f,                 0x57B7, "rounds up" },
        { 65504.0f,                 0x7BFF, "largest finite f16" },
        { 65520.0f,                 0x7C00, "overflow threshold -> inf" },
        { 1.00048828125f,           0x3C00, "exact tie -> rounds to even (1.0)" },
        { 1.0009765625f,            0x3C01, "one ulp above 1.0" },
        { 2.0009765625f,            0x4000, "tie at 2.0 -> rounds to even" },
        { 0.00048828125f,           0x1000, "2^-11, exact" },
        { 0.00006103515625f,        0x0400, "2^-14, smallest f16 normal" },
        { 0.000000059604644775390625f, 0x0001, "2^-24, smallest f16 subnormal" },
        { 1e-6f,                    0x0011, "subnormal" },
        { 127.9375f,                0x57FF, "exact" },
        { -0.1f,                    0xAE66, "negative fraction" },
    };
    int bad = 0;
    for (const auto & k : known) {
        if (f16_bits(k.v) != k.want) {
            printf("     reference mismatch for %s: got 0x%04x want 0x%04x\n",
                   k.what, f16_bits(k.v), k.want);
            bad++;
        }
    }

    int sweep = 0, cast_mismatch = 0;
#if defined(__GNUC__)
    for (int i = -60000; i <= 60000; i += 3) {
        const float f = (float) i / 7.0f;
        sweep++;
        if (f16_bits(f) != host_cast_bits(f)) {
            cast_mismatch++;
        }
    }
#endif
    char detail[128];
    snprintf(detail, sizeof(detail), "%d known values + %d host casts, %d wrong",
             (int) (sizeof(known) / sizeof(known[0])), sweep, bad + cast_mismatch);
    report(bad == 0 && cast_mismatch == 0, "reference f32->f16 RN-even converter", detail);
}

// ---------------------------------------------------------------------------
// fixtures
// ---------------------------------------------------------------------------
struct Fixture {
    ggml_context * ctx;
    ggml_tensor  * cache;
    ggml_tensor  * values;
    ggml_tensor  * idx;
    std::vector<ggml_backend_buffer_t> bufs;
};

static void attach(ggml_tensor * t, ggml_backend_buffer_t buf, Fixture * f) {
    if (!buf) {
        printf("FAIL: buffer alloc failed for %s\n", ggml_type_name(t->type));
        exit(1);
    }
    t->buffer = buf;
    t->data   = ggml_backend_buffer_get_base(buf);
    f->bufs.push_back(buf);
}

static void free_fixture(Fixture & f) {
    for (ggml_backend_buffer_t b : f.bufs) {
        ggml_backend_buffer_free(b);
    }
    ggml_free(f.ctx);
}

static Fixture make_fixture(ggml_backend_buffer_type_t buft, bool operands_on_device,
                            int64_t d, int64_t n, int64_t h, int64_t l, int64_t nr,
                            enum ggml_type vt, enum ggml_type it) {
    Fixture f;
    f.ctx = ggml_init({.mem_size = 4 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true});

    f.cache  = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, d, n, h, l);
    f.values = ggml_new_tensor_4d(f.ctx, vt, d, nr, h, l);
    f.idx    = ggml_new_tensor_4d(f.ctx, it, nr, 2, 1, 1);

    attach(f.cache, ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.cache)), &f);
    if (operands_on_device) {
        attach(f.values, ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.values)), &f);
        attach(f.idx,    ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.idx)), &f);
    } else {
        ggml_backend_buffer_type_t hbuft = ggml_backend_cpu_buffer_type();
        attach(f.values, ggml_backend_buft_alloc_buffer(hbuft, ggml_nbytes(f.values)), &f);
        attach(f.idx,    ggml_backend_buft_alloc_buffer(hbuft, ggml_nbytes(f.idx)), &f);
    }

    // seed the cache with a recognisable pattern so "not written" is provable
    std::vector<uint16_t> seed(ggml_nbytes(f.cache) / 2, k_sentinel);
    ggml_backend_tensor_set(f.cache, seed.data(), 0, ggml_nbytes(f.cache));

    return f;
}

// the SET_ROWS node exactly as ggml_set_rows builds it: a view of the cache
static ggml_tensor * set_rows_node(ggml_context * ctx, ggml_tensor * cache,
                                   ggml_tensor * values, ggml_tensor * idx) {
    ggml_tensor * node = ggml_new_tensor_4d(ctx, cache->type,
                                           cache->ne[0], cache->ne[1], cache->ne[2], cache->ne[3]);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        node->nb[i] = cache->nb[i];
    }
    node->op        = GGML_OP_SET_ROWS;
    node->src[0]    = values;
    node->src[1]    = idx;
    node->src[2]    = cache;
    node->view_src  = cache;
    node->view_offs = 0;
    node->buffer    = cache->buffer;
    node->data      = cache->data;
    return node;
}

static ggml_status run_graph(ggml_backend_t backend, ggml_context * ctx, ggml_tensor * node) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_graph_add_node(gf, node);
    return ggml_backend_graph_compute(backend, gf);
}

// f32 values deliberately unrepresentable in f16: ties, fractions, subnormals
static const float k_delta[] = {
    0.0f, 0.1f, 1.0f / 3.0f, 1.0f / 7.0f, 1e-5f, 1e-6f,
    0.00048828125f,          // 2^-11: exact grid point
    1.00048828125f,          // exact tie -> rounds to even
    127.9375f,               // exact
    123.456f, -0.1f, -(1.0f / 3.0f),
    0.00006103515625f,       // 2^-14, smallest f16 normal
    0.000000059604644775390625f, // 2^-24, smallest f16 subnormal
};
static const int k_ndelta = (int) (sizeof(k_delta) / sizeof(k_delta[0]));

static float test_value(int64_t i, int64_t i02, int64_t i03, int64_t j) {
    const float base = (float) ((i * 23 + i02 * 11 + i03 * 5) % 64);
    return base + k_delta[(size_t) ((i * 31 + i02 * 7 + i03 * 3 + j) % k_ndelta)];
}

static void fill_values(ggml_tensor * values) {
    const int64_t d = values->ne[0], nr = values->ne[1], h = values->ne[2], l = values->ne[3];
    std::vector<float> v((size_t) d * nr * h * l);
    for (int64_t i03 = 0; i03 < l; i03++) {
        for (int64_t i02 = 0; i02 < h; i02++) {
            for (int64_t i = 0; i < nr; i++) {
                for (int64_t j = 0; j < d; j++) {
                    v[(((size_t) i03 * h + i02) * (size_t) nr + i) * (size_t) d + j] =
                        test_value(i, i02, i03, j);
                }
            }
        }
    }
    if (values->type == GGML_TYPE_F16) {
        // store the f32 values as the f16 the cache is expected to end up with
        std::vector<uint16_t> h16(v.size());
        for (size_t k = 0; k < v.size(); k++) h16[k] = f16_bits(v[k]);
        ggml_backend_tensor_set(values, h16.data(), 0, ggml_nbytes(values));
    } else {
        ggml_backend_tensor_set(values, v.data(), 0, ggml_nbytes(values));
    }
}

static std::vector<uint16_t> read_cache(ggml_tensor * cache) {
    std::vector<uint16_t> host(ggml_nbytes(cache) / 2, 0);
    ggml_backend_tensor_get(cache, host.data(), 0, ggml_nbytes(cache));
    return host;
}

int main(void) {
    ggml_backend_reg_t reg = ggml_backend_kunlun_reg();
    self_check_reference();

    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("SKIP: no Kunlun device available (driver not loaded?)\n");
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        printf("SKIP: could not initialise the Kunlun backend\n");
        return 0;
    }
    printf("device: %s\nbuffer type: %s\n\n", ggml_backend_dev_name(dev), ggml_backend_buft_name(buft));

    // realistic KV cache dims: [head_dim, n_slots, n_head, n_layer]
    const int64_t d = 128, n = 512, h = 8, l = 4;
    const int64_t nr = 64;
    char detail[160];

    // element offset of cache[i1][i02][i03][0] in the flat host copy
    auto row_base = [&](int64_t i1, int64_t i02, int64_t i03) {
        return (((size_t) i03 * h + i02) * n + i1) * d;
    };

    // ---- 1. plain 4D f16 device cache + 2-column i32 idx round trip ----
    {
        Fixture f = make_fixture(buft, true, d, n, h, l, nr, GGML_TYPE_F32, GGML_TYPE_I32);
        fill_values(f.values);

        std::vector<int32_t> iv((size_t) nr * 2);
        for (int64_t i = 0; i < nr; i++) {
            iv[(size_t) i]                = (int32_t) ((i * 7)  % n);  // column 0
            iv[(size_t) i + (size_t) nr]  = (int32_t) ((i * 13) % n);  // column 1
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));
        report(st == GGML_STATUS_SUCCESS, "4D f16 device cache: graph_compute",
               st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

        std::vector<uint16_t> host = read_cache(f.cache);

        int64_t checked = 0, bad_elems = 0, bad_rows = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr; i++) {
                    const int64_t i1 = (i02 % 2 == 0) ? (i * 7) % n : (i * 13) % n;
                    const size_t base = row_base(i1, i02, i03);
                    bool row_ok = true;
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(i, i02, i03, j))) {
                            bad_elems++;
                            row_ok = false;
                        }
                    }
                    if (!row_ok) bad_rows++;
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements compared, %lld bad rows, %lld bad elements",
                 (long long) checked, (long long) bad_rows, (long long) bad_elems);
        report(bad_rows == 0 && bad_elems == 0, "D2H read-back == f32->f16 RN-even cast", detail);

        // rows the idx never selected must still hold the sentinel
        std::vector<char> seen((size_t) h * l * n, 0);
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr; i++) {
                    seen[(size_t) (i03 * h + i02) * n + ((i02 % 2 == 0) ? (i * 7) % n : (i * 13) % n)] = 1;
                }
            }
        }
        int64_t untouched = 0, dirty = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i1 = 0; i1 < n; i1++) {
                    if (seen[(size_t) (i03 * h + i02) * n + i1]) continue;
                    untouched++;
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        if (host[base + j] != k_sentinel) { dirty++; break; }
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld untouched rows, %lld changed", (long long) untouched, (long long) dirty);
        report(dirty == 0, "rows never written keep the sentinel", detail);

        free_fixture(f);
    }

    // ---- 2. colliding idx rows: the last write to a row wins ----
    {
        const int64_t nr2 = 32;
        Fixture f = make_fixture(buft, false, d, n, h, l, nr2, GGML_TYPE_F32, GGML_TYPE_I32);
        fill_values(f.values);

        // both columns point at (i % 8): 4 different values rows land on the same
        // destination row inside the same head plane
        std::vector<int32_t> iv((size_t) nr2 * 2);
        for (int64_t i = 0; i < nr2; i++) {
            iv[(size_t) i]                = (int32_t) (i % 8);
            iv[(size_t) i + (size_t) nr2] = (int32_t) (i % 8);
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));
        report(st == GGML_STATUS_SUCCESS, "overwrite round trip: graph_compute",
               st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

        std::vector<uint16_t> host = read_cache(f.cache);
        int64_t bad = 0, checked = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i1 = 0; i1 < 8; i1++) {
                    // the row loop runs i ascending, so the largest i with i % 8 == i1 wins
                    const int64_t last_i = nr2 - 1 - ((nr2 - 1 - i1) % 8);
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(last_i, i02, i03, j))) bad++;
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements, %lld mismatched", (long long) checked, (long long) bad);
        report(bad == 0, "colliding rows keep the last written value", detail);

        free_fixture(f);
    }

    // ---- 3. i64 idx ----
    {
        const int64_t nr3 = 16;
        Fixture f = make_fixture(buft, false, d, n, h, l, nr3, GGML_TYPE_F32, GGML_TYPE_I64);
        fill_values(f.values);

        std::vector<int64_t> iv((size_t) nr3 * 2);
        for (int64_t i = 0; i < nr3; i++) {
            iv[(size_t) i]                = (int64_t) ((i * 11) % n);
            iv[(size_t) i + (size_t) nr3] = (int64_t) ((i * 17) % n);
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));
        report(st == GGML_STATUS_SUCCESS, "i64 2-column idx: graph_compute",
               st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

        std::vector<uint16_t> host = read_cache(f.cache);
        int64_t bad = 0, checked = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr3; i++) {
                    const int64_t i1 = (i02 % 2 == 0) ? (i * 11) % n : (i * 17) % n;
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(i, i02, i03, j))) bad++;
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements, %lld mismatched", (long long) checked, (long long) bad);
        report(bad == 0, "i64 idx D2H comparison", detail);

        free_fixture(f);
    }

    // ---- 4. f16 values -> f16 cache (the identity branch of the row cast) ----
    {
        Fixture f = make_fixture(buft, false, d, n, h, l, nr, GGML_TYPE_F16, GGML_TYPE_I32);
        fill_values(f.values);

        std::vector<int32_t> iv((size_t) nr * 2);
        for (int64_t i = 0; i < nr; i++) {
            iv[(size_t) i]                = (int32_t) ((i * 5)  % n);
            iv[(size_t) i + (size_t) nr]  = (int32_t) ((i * 11) % n);
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));
        report(st == GGML_STATUS_SUCCESS, "f16 values -> f16 cache: graph_compute",
               st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

        std::vector<uint16_t> host = read_cache(f.cache);
        int64_t bad = 0, checked = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr; i++) {
                    const int64_t i1 = (i02 % 2 == 0) ? (i * 5) % n : (i * 11) % n;
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(i, i02, i03, j))) bad++;
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements, %lld mismatched", (long long) checked, (long long) bad);
        report(bad == 0, "f16 values round trip bit-exact", detail);

        free_fixture(f);
    }

    // ---- 5. destination is a KV slice view of the device cache ----
    {
        Fixture f = make_fixture(buft, false, d, n, h, l, nr, GGML_TYPE_F32, GGML_TYPE_I32);
        fill_values(f.values);

        const int64_t ns = n / 2;
        std::vector<int32_t> iv((size_t) nr * 2);
        for (int64_t i = 0; i < nr; i++) {
            iv[(size_t) i]                = (int32_t) ((i * 7)  % ns);
            iv[(size_t) i + (size_t) nr]  = (int32_t) ((i * 13) % ns);
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        // [head_dim, n_slots/2, n_head, n_layer] starting one slot into the cache,
        // with the strides the parent cache actually has - how llama passes a KV
        // slice to the graph
        ggml_tensor * slice = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, d, ns, h, l);
        slice->nb[0] = ggml_row_size(GGML_TYPE_F16, 1);
        slice->nb[1] = slice->nb[0] * (size_t) d;
        slice->nb[2] = slice->nb[1] * (size_t) n;
        slice->nb[3] = slice->nb[2] * (size_t) h;
        slice->view_src  = f.cache;
        slice->view_offs = ggml_row_size(GGML_TYPE_F16, d);
        slice->buffer    = f.bufs[0];
        slice->data      = (char *) ggml_backend_buffer_get_base(f.bufs[0]) + slice->view_offs;

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, slice, f.values, f.idx));
        report(st == GGML_STATUS_SUCCESS, "KV-slice view round trip: graph_compute",
               st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");

        std::vector<uint16_t> host = read_cache(f.cache);
        int64_t bad = 0, checked = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr; i++) {
                    const int64_t i1 = (i02 % 2 == 0) ? (i * 7) % ns : (i * 13) % ns;
                    // slice slot i1 == full cache slot i1 + 1
                    const size_t base = row_base(i1 + 1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(i, i02, i03, j))) bad++;
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements, %lld mismatched", (long long) checked, (long long) bad);
        report(bad == 0, "slice rows land in the right cache slots", detail);

        // slots outside the view (slot 0 and slots >= n/2) must be untouched
        bool outside_clean = true;
        int64_t outside_rows = 0;
        for (int64_t i03 = 0; i03 < l && outside_clean; i03++) {
            for (int64_t i02 = 0; i02 < h && outside_clean; i02++) {
                for (int64_t i1 = 0; i1 < n; i1++) {
                    if (i1 >= 1 && i1 < ns + 1) continue;
                    outside_rows++;
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        if (host[base + j] != k_sentinel) { outside_clean = false; break; }
                    }
                    if (!outside_clean) break;
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld rows outside the view checked", (long long) outside_rows);
        report(outside_clean, "memory outside the slice view untouched", detail);

        free_fixture(f);
    }

    // ---- 6. out-of-range idx must be refused, not written ----
    {
        struct OOB { int64_t v; const char * name; };
        const OOB oob[] = {
            { n,            "idx == ne1 (one past the last slot)" },
            { -1,           "idx == -1" },
            { n + 1,        "idx > ne1" },
            { 2147483647LL, "idx = INT32_MAX" },
        };

        for (const OOB & o : oob) {
            Fixture f = make_fixture(buft, false, d, n, h, l, nr, GGML_TYPE_F32, GGML_TYPE_I32);
            fill_values(f.values);

            std::vector<int32_t> iv((size_t) nr * 2);
            for (int64_t i = 0; i < nr; i++) {
                iv[(size_t) i]                = (int32_t) ((i * 7)  % n);
                iv[(size_t) i + (size_t) nr]  = (int32_t) ((i * 13) % n);
            }
            iv[0] = (int32_t) o.v;  // first idx read -> the guard trips before any write
            ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

            ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));

            std::vector<uint16_t> host = read_cache(f.cache);
            int64_t overwritten = 0;
            for (uint16_t x : host) {
                if (x != k_sentinel) overwritten++;
            }

            snprintf(detail, sizeof(detail), "%s -> %s, %lld elements overwritten",
                     o.name, st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED", (long long) overwritten);
            report(st != GGML_STATUS_SUCCESS && overwritten == 0, "out-of-range idx rejected before writing", detail);

            free_fixture(f);
        }
    }

    // ---- 7. the device is still usable after a rejected graph ----
    {
        Fixture f = make_fixture(buft, true, d, n, h, l, nr, GGML_TYPE_F32, GGML_TYPE_I32);
        fill_values(f.values);

        std::vector<int32_t> iv((size_t) nr * 2);
        for (int64_t i = 0; i < nr; i++) {
            iv[(size_t) i]                = (int32_t) ((i * 7)  % n);
            iv[(size_t) i + (size_t) nr]  = (int32_t) ((i * 13) % n);
        }
        ggml_backend_tensor_set(f.idx, iv.data(), 0, ggml_nbytes(f.idx));

        ggml_status st = run_graph(backend, f.ctx, set_rows_node(f.ctx, f.cache, f.values, f.idx));
        std::vector<uint16_t> host = read_cache(f.cache);
        int64_t bad = 0, checked = 0;
        for (int64_t i03 = 0; i03 < l; i03++) {
            for (int64_t i02 = 0; i02 < h; i02++) {
                for (int64_t i = 0; i < nr; i++) {
                    const int64_t i1 = (i02 % 2 == 0) ? (i * 7) % n : (i * 13) % n;
                    const size_t base = row_base(i1, i02, i03);
                    for (int64_t j = 0; j < d; j++) {
                        checked++;
                        if (host[base + j] != f16_bits(test_value(i, i02, i03, j))) bad++;
                    }
                }
            }
        }
        snprintf(detail, sizeof(detail), "%lld elements, %lld mismatched", (long long) checked, (long long) bad);
        report(st == GGML_STATUS_SUCCESS && bad == 0, "device still healthy after rejected graphs", detail);

        free_fixture(f);
    }

    ggml_backend_free(backend);
    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
