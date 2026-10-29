// Contract tests for the Kunlun K200 backend's GGML_OP_SET_ROWS support check.
//
// ggml_backend_kunlun_device_supports_op must accept exactly the tensors that
// ggml_set_rows (ggml/src/ggml.c) + the SET_ROWS write path in
// ggml_backend_kunlun_graph_compute can handle, and reject everything else:
//   - idx must be I32 or I64
//   - values must be F32/F16 and row-convertible to the cache element type
//   - cache ne0/ne2/ne3 == values ne0/ne2/ne3, values ne1 == idx ne0, idx ne3 == 1,
//     idx ne1 | cache ne2 and idx ne2 | cache ne3 (zero idx dims rejected so the
//     modulo in the write path cannot divide by zero)
//   - cache and values rows must be contiguous in memory
//   - the destination must resolve to Kunlun device memory (KV cache slices are
//     views, so the view_src chain is walked); values/idx may have NULL buffers
//     because supports_op runs before the graph is allocated
//
// Standalone build (no llama needed):
//   c++ -std=gnu++17 -O2 -I ggml/include tests/test-kunlun-set-rows-contract.cpp -o test-kunlun-set-rows-contract
//       -L build-kunlun/bin -Wl,-rpath,build-kunlun/bin -lggml-base -lggml-kunlun
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

extern "C" ggml_backend_reg_t ggml_backend_kunlun_reg(void);

static int g_pass = 0;
static int g_fail = 0;

static void check(ggml_backend_dev_t dev, const char * name, const ggml_tensor * node, bool want) {
    bool got = ggml_backend_dev_supports_op(dev, node);
    bool ok  = got == want;
    printf("%-4s %-52s got=%s want=%s\n", ok ? "PASS" : "FAIL", name,
           got ? "true" : "false", want ? "true" : "false");
    (ok ? g_pass : g_fail)++;
}

struct Fixture {
    ggml_context * ctx;
    ggml_backend_buffer_t buf;
    ggml_tensor * cache;
    ggml_tensor * values;
    ggml_tensor * idx;
};

static Fixture make_fixture(ggml_backend_buffer_type_t buft,
                            enum ggml_type ct, int64_t cd, int64_t cn, int64_t ch, int64_t cl,
                            enum ggml_type vt, int64_t vd, int64_t vn, int64_t vh, int64_t vl,
                            enum ggml_type it, int64_t id0, int64_t id1, int64_t id2, int64_t id3) {
    Fixture f;
    f.ctx = ggml_init({.mem_size = 1 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true});

    f.cache  = ggml_new_tensor_4d(f.ctx, ct, cd, cn, ch, cl);
    f.values = ggml_new_tensor_4d(f.ctx, vt, vd, vn, vh, vl);
    f.idx    = ggml_new_tensor_4d(f.ctx, it, id0, id1, id2, id3);

    f.buf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.cache));
    if (!f.buf) {
        printf("FAIL: could not allocate a Kunlun device buffer\n");
        exit(1);
    }
    f.cache->buffer = f.buf;
    f.cache->data   = ggml_backend_buffer_get_base(f.buf);
    ggml_backend_buffer_init_tensor(f.buf, f.cache);

    return f;
}

// Build the SET_ROWS node the way ggml_set_rows does (node = view of the cache),
// but by hand so the tests can pass shapes the builder asserts would reject.
static ggml_tensor * set_rows_node(ggml_context * ctx, ggml_tensor * cache,
                                   ggml_tensor * values, ggml_tensor * idx) {
    ggml_tensor * node = ggml_new_tensor_4d(ctx, cache->type,
                                           cache->ne[0], cache->ne[1], cache->ne[2], cache->ne[3]);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        node->nb[i] = cache->nb[i];
    }
    node->op       = GGML_OP_SET_ROWS;
    node->src[0]   = values;
    node->src[1]   = idx;
    node->src[2]   = cache;
    node->view_src = cache;
    node->view_offs = 0;
    node->buffer   = cache->buffer;
    node->data     = cache->data;
    return node;
}

int main(void) {
    ggml_backend_reg_t reg = ggml_backend_kunlun_reg();
    if (!reg || ggml_backend_reg_dev_count(reg) == 0) {
        printf("SKIP: no Kunlun device available (driver not loaded?)\n");
        return 0;
    }
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    printf("device: %s\nbuffer type: %s\n\n",
           ggml_backend_dev_name(dev), ggml_backend_buft_name(buft));

    // realistic KV cache dims: [head_dim, n_slots, n_head, n_layer]
    const int64_t d = 128, n = 512, h = 8, l = 4;
    const int64_t nr = 32; // values rows

    // ---- valid ----
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "f16 cache <- f32 values, i32 2-col idx", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F16, d, nr, h, l, GGML_TYPE_I64, nr, 2, 1, 1);
        check(dev, "f16 cache <- f16 values, i64 idx", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F32, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "f32 cache <- f32 values (same type)", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_BF16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "bf16 cache <- f32 values", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        // idx planes match the cache planes exactly (no broadcast)
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, h, l, 1);
        check(dev, "idx planes == cache planes", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        // strided idx: nb[1] != nb[0]*ne[0], like a gathered/permuted index tensor
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        f.idx->nb[1] = (size_t) (nr + 1) * ggml_row_size(GGML_TYPE_I32, 1);
        check(dev, "strided idx tolerated (CPU impl uses nb)", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        // KV slice: destination is a view of a view of the device cache
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        // same layout as a llama KV slice: [head_dim, n_slots/2, n_head, n_layer]
        ggml_tensor * slice = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, d, n / 2, h, l);
        slice->nb[0] = ggml_row_size(GGML_TYPE_F16, 1);
        slice->nb[1] = slice->nb[0] * (size_t) d;
        slice->nb[2] = slice->nb[1] * (size_t) (n / 2);
        slice->nb[3] = slice->nb[2] * (size_t) h;
        slice->view_src = f.cache;
        slice->buffer   = f.buf;
        slice->data     = (char *) ggml_backend_buffer_get_base(f.buf) + ggml_row_size(GGML_TYPE_F16, d);
        ggml_tensor * node = set_rows_node(f.ctx, slice, f.values, f.idx);
        check(dev, "view-of-view cache resolves to device", node, true);
    }
    {
        // operands with NULL buffers: supports_op runs before the graph is allocated
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        printf("     (values->buffer=%p idx->buffer=%p)\n", (void *) f.values->buffer, (void *) f.idx->buffer);
        check(dev, "NULL-buffer values/idx tolerated", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
    }
    {
        // destination buffer not assigned yet, but the view chain reaches the device
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        ggml_tensor * node = set_rows_node(f.ctx, f.cache, f.values, f.idx);
        node->buffer = NULL;
        node->data   = NULL;
        check(dev, "unallocated destination with device view_src", node, true);
    }
    {
        // device-resident operands
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        ggml_backend_buffer_t vbuf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.values));
        ggml_backend_buffer_t ibuf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.idx));
        if (vbuf && ibuf) {
            f.values->buffer = vbuf;
            f.values->data   = ggml_backend_buffer_get_base(vbuf);
            ggml_backend_buffer_init_tensor(vbuf, f.values);
            f.idx->buffer = ibuf;
            f.idx->data   = ggml_backend_buffer_get_base(ibuf);
            ggml_backend_buffer_init_tensor(ibuf, f.idx);
            check(dev, "device-resident values/idx accepted", set_rows_node(f.ctx, f.cache, f.values, f.idx), true);
        } else {
            printf("SKIP device-resident operands (alloc failed)\n");
        }
    }

    // ---- invalid dtype ----
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_F32, nr, 2, 1, 1);
        check(dev, "idx f32 rejected", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I16, nr, 2, 1, 1);
        check(dev, "idx i16 rejected", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_Q4_0, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "values q4_0 rejected", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_Q8_0, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "q8_0 cache rejected (no row writer)", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_BF16, d, n, h, l, GGML_TYPE_F16, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "f16 values -> bf16 cache rejected", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }

    // ---- invalid ne relations ----
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d / 2, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "values ne0 != cache ne0", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h + 1, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "values ne2 != cache ne2", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l + 1, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "values ne3 != cache ne3", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr + 1, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "values ne1 != idx ne0", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 2);
        check(dev, "idx ne3 != 1", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 3, 1, 1);
        check(dev, "idx ne1 does not divide cache ne2", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 3, 1);
        check(dev, "idx ne2 does not divide cache ne3", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 0, 1, 1);
        check(dev, "idx ne1 == 0 rejected (no modulo by zero)", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }

    // ---- invalid strides / buffers ----
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        f.cache->nb[0] = 2 * ggml_row_size(GGML_TYPE_F16, 1); // padded rows
        check(dev, "cache row stride != type size", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        f.values->nb[0] = 2 * ggml_row_size(GGML_TYPE_F32, 1); // padded rows
        check(dev, "values row stride != type size", set_rows_node(f.ctx, f.cache, f.values, f.idx), false);
    }
    {
        // destination is neither allocated nor a view of a device tensor
        ggml_context * ctx = ggml_init({.mem_size = 1 * 1024 * 1024, .mem_buffer = NULL, .no_alloc = true});
        ggml_tensor * cache  = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, d, n, h, l);
        ggml_tensor * values = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nr, h, l);
        ggml_tensor * idx    = ggml_new_tensor_4d(ctx, GGML_TYPE_I32, nr, 2, 1, 1);
        check(dev, "host/no-buffer cache rejected, no crash", set_rows_node(ctx, cache, values, idx), false);
    }
    {
        // missing operand must not dereference NULL
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        ggml_tensor * node = set_rows_node(f.ctx, f.cache, NULL, f.idx);
        check(dev, "NULL values operand -> false", node, false);
    }
    {
        // non-SET_ROWS op is never claimed
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);
        ggml_tensor * node = ggml_new_tensor_4d(f.ctx, GGML_TYPE_F16, d, n, h, l);
        node->op     = GGML_OP_ADD;
        node->src[0] = f.cache;
        check(dev, "ADD not claimed by Kunlun", node, false);
    }

    // ---- every case supports_op accepts must actually run in the write path ----
    {
        Fixture f = make_fixture(buft, GGML_TYPE_F16, d, n, h, l, GGML_TYPE_F32, d, nr, h, l, GGML_TYPE_I32, nr, 2, 1, 1);

        ggml_backend_buffer_t vbuf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.values));
        ggml_backend_buffer_t ibuf = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, f.idx));
        if (!vbuf || !ibuf) {
            printf("SKIP execution check (device alloc failed)\n");
        } else {
            f.values->buffer = vbuf;
            f.values->data   = ggml_backend_buffer_get_base(vbuf);
            f.idx->buffer    = ibuf;
            f.idx->data      = ggml_backend_buffer_get_base(ibuf);

            // idx is [nr, 2, 1, 1]: column 0 and column 1 select different cache rows,
            // matching the 2-column idx of the llama KV update path
            std::vector<int32_t> idx_data((size_t) nr * 2);
            for (int64_t i = 0; i < nr; i++) {
                idx_data[i]          = (int32_t) ((i * 7)  % n);  // col 0, used by even heads
                idx_data[i + nr]     = (int32_t) ((i * 13) % n);  // col 1, used by odd heads
            }
            ggml_backend_tensor_set(f.idx, idx_data.data(), 0, ggml_nbytes(f.idx));

            std::vector<float> v_data((size_t) d * nr * h * l);
            for (int64_t i03 = 0; i03 < l; i03++) {
                for (int64_t i02 = 0; i02 < h; i02++) {
                    for (int64_t i = 0; i < nr; i++) {
                        for (int64_t j = 0; j < d; j++) {
                            v_data[(((i03 * h + i02) * nr + i) * d) + j] = (float) (i * 20 + i02 * 4 + i03 + j);
                        }
                    }
                }
            }
            ggml_backend_tensor_set(f.values, v_data.data(), 0, ggml_nbytes(f.values));

            ggml_tensor * node = set_rows_node(f.ctx, f.cache, f.values, f.idx);
            ggml_cgraph * gf = ggml_new_graph(f.ctx);
            ggml_graph_add_node(gf, node);

            ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
            ggml_status st = ggml_backend_graph_compute(backend, gf);
            ggml_backend_free(backend);

            bool ok = st == GGML_STATUS_SUCCESS;
            printf("%-4s %-52s graph_compute=%s\n", ok ? "PASS" : "FAIL", "accepted case executes on device",
                   st == GGML_STATUS_SUCCESS ? "SUCCESS" : "FAILED");
            (ok ? g_pass : g_fail)++;

            // every written row is read back and compared with the f32 -> f16 values
            std::vector<ggml_fp16_t> row(d);
            int64_t bad_rows = 0;
            for (int64_t i03 = 0; i03 < l; i03++) {
                for (int64_t i02 = 0; i02 < h; i02++) {
                    for (int64_t i = 0; i < nr; i++) {
                        int64_t i1 = (i02 % 2 == 0) ? (i * 7) % n : (i * 13) % n;
                        ggml_backend_tensor_get(f.cache, row.data(),
                                               (size_t) (i1 * d + i02 * (n * d) + i03 * (n * h * d)) * 2,
                                               d * 2);
                        for (int64_t j = 0; j < d; j++) {
                            float want = (float) (i * 20 + i02 * 4 + i03 + j);  // < 2048: exact in f16
                            if ((float) ggml_fp16_to_fp32(row[j]) != want) { bad_rows++; break; }
                        }
                    }
                }
            }
            printf("%-4s %-52s mismatched rows=%lld\n", bad_rows == 0 ? "PASS" : "FAIL",
                   "written rows match values (D2H spot check)", (long long) bad_rows);
            (bad_rows == 0 ? g_pass : g_fail)++;
        }
    }

    printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
