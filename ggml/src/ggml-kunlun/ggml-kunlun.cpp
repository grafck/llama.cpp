// ggml-kunlun: Baidu Kunlun K200 (KL1) backend for llama.cpp
//
// Design (Phase B first cut, see ~/kunlun-04-00-0/llama-backend/PROGRAM.md):
//   - Device buffers in K200 HBM (xpu_malloc), raw ggml byte layout (offsets stay consistent).
//   - On first use per weight tensor: convert to fp16 device weights (lazy, cached in a
//     registry keyed by the raw device pointer). int8 weights come later.
//   - Supported device ops: MUL_MAT (fp16 weights via xdnn fc). Everything else falls back
//     to CPU; the scheduler inserts the copies for us.
//   - The eager xdnn API is ASYNC: xpu_wait() before any D2H read and before any H2D
//     overwrite of a device buffer last written by queued compute.

#include "ggml-kunlun.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

// the vendored xpu refactor headers define fp32_from_bits/fp32_to_bits at global
// scope, colliding with ggml-impl.h - rename them in the vendor TU only
#define fp32_from_bits xpu_compat_fp32_from_bits
#define fp32_to_bits   xpu_compat_fp32_to_bits
#include <xpu/runtime.h>
#include <xpu/xdnn.h>
#undef fp32_from_bits
#undef fp32_to_bits

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xdnn = baidu::xpu::api;
using f16 = float16;

#define KUNLUN_LOG_ERR(fmt, ...) GGML_LOG_ERROR("kunlun: " fmt "\n", ##__VA_ARGS__)

//
// device registry
//

struct ggml_kunlun_device {
    int               index     = -1;
    size_t            capacity  = 0;
    size_t            allocated = 0;
    xdnn::Context *   ctx       = nullptr;
};

// converted fp16 weights, keyed by the tensor's raw device pointer
struct ggml_kunlun_buffer_context {
    void *              dev_ptr;
    size_t              size;
    int                 dev_index;
    std::vector<void *> keys;  // weight registry keys registered for tensors in this buffer
};

struct kunlun_weight {
    void * w16;    // device ptr, fp16 [n][k] (may alias the raw weight bytes for F16 type)
    size_t bytes;  // size of the converted allocation (0 when aliasing)
    bool   alias;  // true => no separate allocation, freed with the buffer
};

struct ggml_kunlun_state {
    bool                            probed = false;
    bool                            ok     = false;
    std::vector<ggml_kunlun_device> devices;

    std::mutex                                        wreg_mtx;
    std::unordered_map<void *, kunlun_weight>         wreg;
};

static ggml_kunlun_state & kunlun_state() {
    static ggml_kunlun_state state;
    if (!state.probed) {
        state.probed = true;
        int n = 0;
        if (xpu_device_count(&n) != 0 || n <= 0) {
            KUNLUN_LOG_ERR("no XPU devices found (driver not loaded?)");
            return state;
        }
        for (int i = 0; i < n; i++) {
            ggml_kunlun_device dev;
            dev.index = i;
            uint64_t cap = 0;
            if (xpu_device_get_attr(&cap, XPUATTR_MEM_MAIN_CAPACITY, i) == 0) {
                dev.capacity = (size_t) cap;
            }
            state.devices.push_back(dev);
        }
        state.ok = !state.devices.empty();
    }
    return state;
}

static ggml_kunlun_device * kunlun_device(int index) {
    ggml_kunlun_state & st = kunlun_state();
    if (!st.ok || index < 0 || (size_t) index >= st.devices.size()) {
        return nullptr;
    }
    return &st.devices[index];
}

static xdnn::Context * kunlun_device_ctx(ggml_kunlun_device * dev) {
    if (!dev->ctx) {
        if (xpu_set_device(dev->index) != 0) {
            return nullptr;
        }
        dev->ctx = xdnn::create_context();
        if (!dev->ctx) {
            KUNLUN_LOG_ERR("failed to create xdnn context on device %d", dev->index);
        }
    }
    return dev->ctx;
}

// RAII: xdnn runtime calls operate on the "current" device per-thread
struct kunlun_dev_guard {
    int prev = -1;
    kunlun_dev_guard(int index) {
        xpu_current_device(&prev);
        xpu_set_device(index);
    }
    ~kunlun_dev_guard() {
        if (prev >= 0) {
            xpu_set_device(prev);
        }
    }
};

// device index stored in ggml_backend_device.context by the reg
static int kunlun_dev_index(ggml_backend_dev_t dev) {
    return (int) (intptr_t) dev->context;
}

//
// buffer forward decls
//

static ggml_backend_buffer_type_t ggml_backend_kunlun_device_buffer_type(int dev_index);

static bool ggml_backend_buft_is_kunlun_buft(ggml_backend_buffer_type_t buft, int dev_index) {
    return buft == ggml_backend_kunlun_device_buffer_type(dev_index);
}

//
// converted-weight registry
//

// convert a weight tensor (raw ggml bytes already resident in device memory at src0->data)
// to fp16 device weights [n][k]; returns device ptr or nullptr on failure
static void * kunlun_weights_get(const ggml_tensor * src0, ggml_kunlun_device * dev) {
    ggml_kunlun_state & st = kunlun_state();
    void * key = src0->data;

    {
        std::lock_guard<std::mutex> lock(st.wreg_mtx);
        auto it = st.wreg.find(key);
        if (it != st.wreg.end()) {
            return it->second.w16;
        }
    }

    const size_t nbytes = ggml_nbytes(src0);
    const int64_t nelem = ggml_nelements(src0);
    kunlun_dev_guard guard(dev->index);

    // register the key with the owning buffer so the entry is dropped when it is freed
    ggml_kunlun_buffer_context * owner_buf = nullptr;
    if (src0->buffer) {
        owner_buf = (ggml_kunlun_buffer_context *) src0->buffer->context;
    }

    if (src0->type == GGML_TYPE_F16) {
        // raw bytes are already IEEE half on device - use in place
        std::lock_guard<std::mutex> lock(st.wreg_mtx);
        st.wreg[key] = kunlun_weight{ src0->data, 0, true };
        if (owner_buf) {
            owner_buf->keys.push_back(key);
        }
        return src0->data;
    }

    // D2H raw bytes
    std::vector<char> raw(nbytes);
    xpu_wait();
    if (xpu_memcpy(raw.data(), src0->data, nbytes, XPU_DEVICE_TO_HOST) != 0) {
        KUNLUN_LOG_ERR("D2H failed for weight %s (%zu bytes)", src0->name, nbytes);
        return nullptr;
    }

    // dequantize to fp32 on host (covers every ggml quant type via type traits)
    std::vector<float> f32(nelem);
    if (src0->type == GGML_TYPE_F32) {
        // already fp32 - no to_float trait exists for F32
        std::memcpy(f32.data(), raw.data(), nbytes);
    } else {
        const struct ggml_type_traits * tt = ggml_get_type_traits(src0->type);
        if (tt->to_float == nullptr) {
            KUNLUN_LOG_ERR("no to_float for type %d", (int) src0->type);
            return nullptr;
        }
        tt->to_float(raw.data(), f32.data(), (int32_t) nelem);
    }
    std::vector<char>().swap(raw);

    // fp32 -> fp16
    std::vector<uint16_t> h16(nelem);
    for (int64_t i = 0; i < nelem; i++) {
        h16[i] = ggml_fp32_to_fp16(f32[i]);
    }
    std::vector<float>().swap(f32);

    void * w16 = nullptr;
    const size_t w16_bytes = h16.size() * 2;
    if (xpu_malloc(&w16, w16_bytes) != 0) {
        KUNLUN_LOG_ERR("xpu_malloc failed for converted weights of %s (%zu bytes)", src0->name, w16_bytes);
        return nullptr;
    }
    if (xpu_memcpy(w16, h16.data(), w16_bytes, XPU_HOST_TO_DEVICE) != 0) {
        xpu_free(w16);
        KUNLUN_LOG_ERR("H2D failed for converted weights of %s", src0->name);
        return nullptr;
    }
    xpu_wait();
    dev->allocated += w16_bytes;
    std::vector<uint16_t>().swap(h16);

    std::lock_guard<std::mutex> lock(st.wreg_mtx);
    st.wreg[key] = kunlun_weight{ w16, w16_bytes, false };
    if (owner_buf) {
        owner_buf->keys.push_back(key);
    }
    return w16;
}

//
// buffer (K200 HBM)
//

static void ggml_backend_kunlun_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_kunlun_buffer_context * buf_ctx = (ggml_kunlun_buffer_context *) buffer->context;
    ggml_kunlun_device * dev = kunlun_device(buf_ctx->dev_index);
    kunlun_dev_guard guard(buf_ctx->dev_index);
    xpu_wait();
    xpu_free(buf_ctx->dev_ptr);
    if (dev) {
        dev->allocated -= std::min(dev->allocated, buf_ctx->size);
        ggml_kunlun_state & st = kunlun_state();
        std::lock_guard<std::mutex> lock(st.wreg_mtx);
        for (void * key : buf_ctx->keys) {
            auto it = st.wreg.find(key);
            if (it != st.wreg.end()) {
                if (!it->second.alias) {
                    xpu_free(it->second.w16);
                    dev->allocated -= std::min(dev->allocated, it->second.bytes);
                }
                st.wreg.erase(it);
            }
        }
    }
    delete buf_ctx;
}

static void * ggml_backend_kunlun_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_kunlun_buffer_context * buf_ctx = (ggml_kunlun_buffer_context *) buffer->context;
    return buf_ctx->dev_ptr;
}

static void ggml_backend_kunlun_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                  const void * data, size_t offset, size_t size) {
    ggml_kunlun_buffer_context * buf_ctx = (ggml_kunlun_buffer_context *) buffer->context;
    kunlun_dev_guard guard(buf_ctx->dev_index);
    // async pitfall: wait before an H2D overwriting device memory last written by queued compute
    xpu_wait();
    char * dst = (char *) tensor->data + offset;
    if (xpu_memcpy(dst, data, size, XPU_HOST_TO_DEVICE) != 0) {
        KUNLUN_LOG_ERR("set_tensor H2D failed (%zu bytes @%zu)", size, offset);
    }
}

static void ggml_backend_kunlun_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                  void * data, size_t offset, size_t size) {
    GGML_UNUSED(buffer);
    ggml_kunlun_buffer_context * buf_ctx = (ggml_kunlun_buffer_context *) tensor->buffer->context;
    kunlun_dev_guard guard(buf_ctx->dev_index);
    xpu_wait();  // async pitfall: wait before any host-consumed D2H
    const char * src = (const char *) tensor->data + offset;
    if (xpu_memcpy(data, src, size, XPU_DEVICE_TO_HOST) != 0) {
        KUNLUN_LOG_ERR("get_tensor D2H failed (%zu bytes @%zu)", size, offset);
    }
}

static void ggml_backend_kunlun_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                     uint8_t value, size_t offset, size_t size) {
    static thread_local std::vector<char> scratch;
    if (scratch.size() < size) {
        scratch.resize(size);
    }
    std::memset(scratch.data(), value, size);
    ggml_backend_kunlun_buffer_set_tensor(buffer, tensor, scratch.data(), offset, size);
}

static void ggml_backend_kunlun_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_kunlun_buffer_context * buf_ctx = (ggml_kunlun_buffer_context *) buffer->context;
    kunlun_dev_guard guard(buf_ctx->dev_index);
    xpu_wait();
    static thread_local std::vector<char> scratch;
    const size_t chunk = 4u << 20;
    if (scratch.size() < chunk) {
        scratch.resize(chunk);
    }
    std::memset(scratch.data(), value, chunk);
    for (size_t off = 0; off < buf_ctx->size; off += chunk) {
        size_t n = std::min(chunk, buf_ctx->size - off);
        xpu_memcpy((char *) buf_ctx->dev_ptr + off, scratch.data(), n, XPU_HOST_TO_DEVICE);
    }
    xpu_wait();
}

static struct ggml_backend_buffer_i ggml_backend_kunlun_buffer_i = {
    /* .free_buffer     = */ ggml_backend_kunlun_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_kunlun_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ ggml_backend_kunlun_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_kunlun_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_kunlun_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_kunlun_buffer_clear,
    /* .reset           = */ NULL,
};

//
// buffer type
//

struct ggml_kunlun_buft_context {
    int dev_index;
};

static const char * ggml_backend_kunlun_buft_get_name(ggml_backend_buffer_type_t buft) {
    ggml_kunlun_buft_context * ctx = (ggml_kunlun_buft_context *) buft->context;
    static thread_local std::string name;
    name = "KUNLUN[" + std::to_string(ctx->dev_index) + "] HBM";
    return name.c_str();
}

static ggml_backend_buffer_t ggml_backend_kunlun_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_kunlun_buft_context * ctx = (ggml_kunlun_buft_context *) buft->context;
    ggml_kunlun_device * dev = kunlun_device(ctx->dev_index);
    if (!dev) {
        return NULL;
    }
    kunlun_dev_guard guard(dev->index);
    void * dev_ptr = nullptr;
    if (xpu_malloc(&dev_ptr, size) != 0 || !dev_ptr) {
        return NULL;  // the allocator handles failure (retries with smaller buffers)
    }
    dev->allocated += size;

    ggml_kunlun_buffer_context * buf_ctx = new ggml_kunlun_buffer_context;
    buf_ctx->dev_ptr   = dev_ptr;
    buf_ctx->size      = size;
    buf_ctx->dev_index = ctx->dev_index;

    return ggml_backend_buffer_init(buft, ggml_backend_kunlun_buffer_i, buf_ctx, size);
}

static size_t ggml_backend_kunlun_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 64;
}

static size_t ggml_backend_kunlun_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    // single xpu_malloc is capped ~4 GB (measured: 4.0 GB ok, 4.5 GB fails)
    GGML_UNUSED(buft);
    return (size_t) 4u << 30;
}

static bool ggml_backend_kunlun_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_t ggml_backend_kunlun_device_buffer_type(int dev_index) {
    static std::mutex mtx;
    static std::unordered_map<int, ggml_backend_buffer_type_t> buf_types;
    std::lock_guard<std::mutex> lock(mtx);
    auto it = buf_types.find(dev_index);
    if (it != buf_types.end()) {
        return it->second;
    }
    ggml_kunlun_buft_context * ctx = new ggml_kunlun_buft_context{ dev_index };
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ {
            /* .get_name          = */ ggml_backend_kunlun_buft_get_name,
            /* .alloc_buffer      = */ ggml_backend_kunlun_buft_alloc_buffer,
            /* .alloc_buffer_n    = */ NULL,
            /* .get_alignment     = */ ggml_backend_kunlun_buft_get_alignment,
            /* .get_max_size      = */ ggml_backend_kunlun_buft_get_max_size,
            /* .get_alloc_size    = */ NULL,
            /* .get_alloc_size_n  = */ NULL,
            /* .is_host           = */ ggml_backend_kunlun_buft_is_host,
        },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_kunlun_reg(), dev_index),
        /* .context = */ ctx,
    };
    buf_types[dev_index] = buft;
    return buft;
}

//
// backend (graph compute)
//

struct ggml_backend_kunlun_context {
    int    dev_index = 0;
    void * x16_dev   = nullptr;  // fp16 activation scratch
    size_t x16_size  = 0;
    void * y16_dev   = nullptr;  // fc output scratch
    size_t y16_size  = 0;
    std::vector<char>     stage;  // host staging (src1 rows / outputs)
    std::vector<uint16_t> h16;    // host fp16 staging
};

static const char * ggml_backend_kunlun_get_name(ggml_backend_t backend) {
    ggml_backend_kunlun_context * ctx = (ggml_backend_kunlun_context *) backend->context;
    static thread_local std::string name;
    name = "KUNLUN[" + std::to_string(ctx->dev_index) + "]";
    return name.c_str();
}

static void ggml_backend_kunlun_free(ggml_backend_t backend) {
    ggml_backend_kunlun_context * ctx = (ggml_backend_kunlun_context *) backend->context;
    ggml_kunlun_device * dev = kunlun_device(ctx->dev_index);
    if (dev) {
        kunlun_dev_guard guard(ctx->dev_index);
        if (ctx->x16_dev) { xpu_free(ctx->x16_dev); dev->allocated -= std::min(dev->allocated, ctx->x16_size); }
        if (ctx->y16_dev) { xpu_free(ctx->y16_dev); dev->allocated -= std::min(dev->allocated, ctx->y16_size); }
    }
    delete ctx;
    delete backend;
}

static void ggml_kunlun_synchronize(ggml_backend_t backend) {
    ggml_backend_kunlun_context * ctx = (ggml_backend_kunlun_context *) backend->context;
    kunlun_dev_guard guard(ctx->dev_index);
    xpu_wait();
}

// grow a persistent device scratch
static bool kunlun_scratch(ggml_backend_kunlun_context * bctx, void ** p, size_t * psize, size_t need, ggml_kunlun_device * dev) {
    if (*psize >= need) {
        return true;
    }
    kunlun_dev_guard guard(dev->index);
    xpu_wait();
    if (*p) {
        xpu_free(*p);
        dev->allocated -= std::min(dev->allocated, *psize);
    }
    void * pnew = nullptr;
    if (xpu_malloc(&pnew, need) != 0) {
        KUNLUN_LOG_ERR("scratch xpu_malloc(%zu) failed", need);
        *p = nullptr;
        *psize = 0;
        return false;
    }
    *p     = pnew;
    *psize = need;
    dev->allocated += need;
    return true;
}

static void ggml_kunlun_mul_mat(ggml_backend_kunlun_context * bctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // weights [k, n] ggml-logical -> [n][k] row-major on device
    const ggml_tensor * src1 = dst->src[1];  // activations

    GGML_TENSOR_BINARY_OP_LOCALS

    ggml_kunlun_device * dev = kunlun_device(bctx->dev_index);
    xdnn::Context * ctx = kunlun_device_ctx(dev);
    GGML_ASSERT(ctx);

    const int64_t nk = ne00;                 // inner dim
    const int64_t nn = ne01;                 // output features
    const int64_t m  = ne11 * ne12 * ne13;  // flattened rows (src0 ne02==ne03==1 checked in supports_op)

    void * w16 = kunlun_weights_get(src0, dev);
    GGML_ASSERT(w16 && "weight conversion failed");

    kunlun_dev_guard guard(dev->index);

    // --- stage src1 rows to host
    const size_t row_bytes = ggml_row_size(src1->type, nk);
    bctx->stage.resize((size_t) m * row_bytes);
    if (ggml_is_contiguous(src1)) {
        ggml_backend_tensor_get(src1, bctx->stage.data(), 0, (size_t) m * row_bytes);
    } else {
        for (int64_t i3 = 0; i3 < ne13; i3++) {
            for (int64_t i2 = 0; i2 < ne12; i2++) {
                for (int64_t i1 = 0; i1 < ne11; i1++) {
                    size_t off = i1 * src1->nb[1] + i2 * src1->nb[2] + i3 * src1->nb[3];
                    size_t drow = ((size_t) i3 * ne12 + i2) * ne11 + i1;
                    ggml_backend_tensor_get(src1, bctx->stage.data() + drow * row_bytes, off, row_bytes);
                }
            }
        }
    }

    // --- H2D activations as fp16
    const size_t x16_bytes = (size_t) m * nk * 2;
    GGML_ASSERT(kunlun_scratch(bctx, &bctx->x16_dev, &bctx->x16_size, x16_bytes, dev));

    xpu_wait();  // scratch may hold output of queued compute from the previous node
    if (src1->type == GGML_TYPE_F16) {
        xpu_memcpy(bctx->x16_dev, bctx->stage.data(), x16_bytes, XPU_HOST_TO_DEVICE);
    } else {
        bctx->h16.resize((size_t) m * nk);
        const float * fsrc = (const float *) bctx->stage.data();
        for (size_t i = 0; i < (size_t) m * nk; i++) {
            bctx->h16[i] = ggml_fp32_to_fp16(fsrc[i]);
        }
        xpu_memcpy(bctx->x16_dev, bctx->h16.data(), x16_bytes, XPU_HOST_TO_DEVICE);
    }
    xpu_wait();

    // --- fc: y[m][n] = x[m][k] * W^T with W fp16 [n][k] -> w_trans=true (proven path, llama7b_tp)
    const bool dst_on_dev = ggml_backend_buft_is_kunlun_buft(ggml_backend_buffer_get_type(dst->buffer), bctx->dev_index);
    void * y16;
    if (dst_on_dev && dst->type == GGML_TYPE_F16 && ggml_is_contiguous(dst)) {
        y16 = dst->data;  // fc writes straight into dst
    } else {
        const size_t y16_bytes = (size_t) m * nn * 2;
        GGML_ASSERT(kunlun_scratch(bctx, &bctx->y16_dev, &bctx->y16_size, y16_bytes, dev));
        y16 = bctx->y16_dev;
    }

    int rc = xdnn::fc<f16, f16, f16, short>(ctx, (const f16 *) bctx->x16_dev, (const f16 *) w16,
                                            (f16 *) y16, m, nn, nk, false, true, nullptr, nullptr, nullptr);
    xpu_wait();
    GGML_ASSERT(rc == 0 && "kunlun fc failed");

    // --- output
    if (dst_on_dev && dst->type == GGML_TYPE_F32) {
        GGML_ASSERT(ggml_is_contiguous(dst));
        rc = xdnn::cast<f16, float>(ctx, (const f16 *) y16, (float *) dst->data, (int64_t) m * nn);
        xpu_wait();
        GGML_ASSERT(rc == 0 && "kunlun cast failed");
    } else if (dst_on_dev && dst->type == GGML_TYPE_F16) {
        // wrote directly into dst
    } else {
        // host dst: D2H + convert
        const size_t y_bytes = (size_t) m * nn * 2;
        bctx->stage.resize(y_bytes);
        xpu_memcpy(bctx->stage.data(), y16, y_bytes, XPU_DEVICE_TO_HOST);
        xpu_wait();
        if (dst->type == GGML_TYPE_F32) {
            float * fdst = (float *) dst->data;
            const uint16_t * h = (const uint16_t *) bctx->stage.data();
            for (size_t i = 0; i < (size_t) m * nn; i++) {
                fdst[i] = ggml_fp16_to_fp32(h[i]);
            }
        } else {
            GGML_ASSERT(ggml_is_contiguous(dst));
            std::memcpy(dst->data, bctx->stage.data(), y_bytes);
        }
    }
}

static void ggml_kunlun_set_rows(ggml_backend_kunlun_context * bctx, ggml_tensor * dst) {
    // dst[positions[i]] = src0[i] - KV append via H2D copies to row offsets (PROGRAM.md design).
    // src0 may be fp32 while the cache is fp16: convert on host like ggml's SET_ROWS does.
    const ggml_tensor * src0 = dst->src[0];  // new rows
    const ggml_tensor * src1 = dst->src[1];  // int32 positions

    const int64_t n_rows   = src1->ne[0];
    const int64_t ne_row   = dst->ne[0];
    const size_t  row_out  = ggml_row_size(dst->type, ne_row);
    const size_t  row_in   = ggml_row_size(src0->type, src0->ne[0]);

    std::vector<int32_t> pos(n_rows);
    ggml_backend_tensor_get(src1, pos.data(), 0, n_rows * sizeof(int32_t));

    ggml_kunlun_device * dev = kunlun_device(bctx->dev_index);
    kunlun_dev_guard guard(dev->index);

    std::vector<char> stage_in;
    if (src0->type != dst->type) {
        stage_in.resize(row_in);
    }
    bctx->stage.resize(row_out);

    const struct ggml_type_traits * tt = ggml_get_type_traits(dst->type);
    GGML_ASSERT(tt->from_float_ref || src0->type == dst->type);

    const size_t dst_stride = dst->nb[1];
    const size_t src_stride = src0->nb[1];
    for (int64_t i = 0; i < n_rows; i++) {
        GGML_ASSERT(pos[i] >= 0 && (int64_t) pos[i] < dst->ne[1]);
        void * dst_stage = bctx->stage.data();
        if (src0->type != dst->type) {
            ggml_backend_tensor_get(src0, stage_in.data(), (size_t) i * src_stride, row_in);
            if (src0->type == GGML_TYPE_F32) {
                tt->from_float_ref((const float *) stage_in.data(), dst_stage, (int32_t) ne_row);
            } else {
                const struct ggml_type_traits * tt_in = ggml_get_type_traits(src0->type);
                GGML_ASSERT(tt_in->to_float && tt->from_float_ref);
                std::vector<float> tmp(ne_row);
                tt_in->to_float(stage_in.data(), tmp.data(), (int32_t) ne_row);
                tt->from_float_ref(tmp.data(), dst_stage, (int32_t) ne_row);
            }
        } else {
            ggml_backend_tensor_get(src0, dst_stage, (size_t) i * src_stride, row_out);
        }
        xpu_wait();  // dst row may hold output of queued compute
        int rc = xpu_memcpy((char *) dst->data + (size_t) pos[i] * dst_stride,
                            dst_stage, row_out, XPU_HOST_TO_DEVICE);
        GGML_ASSERT(rc == 0);
    }
    xpu_wait();
}

static ggml_status ggml_backend_kunlun_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_kunlun_context * ctx = (ggml_backend_kunlun_context *) backend->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            continue;
        }

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                ggml_kunlun_mul_mat(ctx, node);
                break;
            case GGML_OP_SET_ROWS:
                ggml_kunlun_set_rows(ctx, node);
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }

    xpu_wait();
    return GGML_STATUS_SUCCESS;
}

static struct ggml_backend_i ggml_backend_kunlun_i = {
    /* .get_name                = */ ggml_backend_kunlun_get_name,
    /* .free                    = */ ggml_backend_kunlun_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ ggml_kunlun_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_kunlun_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_kunlun_guid(void) {
    static const char * guid_str = "kunlun xpu    ";
    return reinterpret_cast<ggml_guid_t>(const_cast<char *>(guid_str));
}

ggml_backend_t ggml_backend_kunlun_init(int device) {
    ggml_kunlun_device * dev = kunlun_device(device);
    if (!dev) {
        return NULL;
    }

    ggml_backend_kunlun_context * ctx = new ggml_backend_kunlun_context;
    ctx->dev_index = device;

    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_kunlun_guid(),
        /* .iface   = */ ggml_backend_kunlun_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_kunlun_reg(), device),
        /* .context = */ ctx,
    };

    return backend;
}

bool ggml_backend_is_kunlun(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_kunlun_guid());
}

//
// device interface
//

static const char * ggml_backend_kunlun_device_get_name(ggml_backend_dev_t dev) {
    static thread_local std::string name;
    name = "KUNLUN[" + std::to_string(kunlun_dev_index(dev)) + "]";
    return name.c_str();
}

static const char * ggml_backend_kunlun_device_get_description(ggml_backend_dev_t dev) {
    return "Kunlun K200 (KL1) accelerator via xdnn 4.33";
    GGML_UNUSED(dev);
}

static void ggml_backend_kunlun_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_kunlun_device * kdev = kunlun_device(kunlun_dev_index(dev));
    if (!kdev) {
        *free = *total = 0;
        return;
    }
    *total = kdev->capacity;
    *free  = kdev->capacity > kdev->allocated ? kdev->capacity - kdev->allocated : 0;
}

static enum ggml_backend_dev_type ggml_backend_kunlun_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;
    GGML_UNUSED(dev);
}

static void ggml_backend_kunlun_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_kunlun_device_get_name(dev);
    props->description = ggml_backend_kunlun_device_get_description(dev);
    props->type        = ggml_backend_kunlun_device_get_type(dev);
    ggml_backend_kunlun_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = NULL;
    props->caps = {
        /* .async                = */ false,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events               = */ false,
        /* .mmap_support         = */ false,
    };
}

static ggml_backend_t ggml_backend_kunlun_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    return ggml_backend_kunlun_init(kunlun_dev_index(dev));
}

static ggml_backend_buffer_type_t ggml_backend_kunlun_device_get_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_kunlun_device_buffer_type(kunlun_dev_index(dev));
}

static ggml_backend_buffer_type_t ggml_backend_kunlun_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static bool ggml_backend_kunlun_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const int dev_index = kunlun_dev_index(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            const ggml_tensor * src0 = op->src[0];
            const ggml_tensor * src1 = op->src[1];

            // weights must be resident on THIS kunlun device when the buffer is known
            // (buffer is NULL during graph-construction support probes — treat as OK, like ggml-cuda)
            if (src0->buffer &&
                !ggml_backend_buft_is_kunlun_buft(ggml_backend_buffer_get_type(src0->buffer), dev_index)) {
                return false;
            }

            // no weight-batched (MoE) matmul on this path; mul_mat_id is a separate op
            if (src0->ne[2] != 1 || src0->ne[3] != 1) {
                return false;
            }

            if (!ggml_is_contiguous(src0) || src0->nb[0] != ggml_type_size(src0->type)) {
                return false;
            }

            if (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) {
                return false;
            }
            // rows must be individually contiguous
            if (src1->nb[0] != ggml_type_size(src1->type)) {
                return false;
            }

            if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
                return false;
            }

            return true;
        }

        case GGML_OP_SET_ROWS: {
            const ggml_tensor * src0 = op->src[0];
            // dst (KV cache) must be on THIS kunlun device when known (NULL = pre-allocation probe)
            const ggml_tensor * dst_buf_src = op->view_src ? op->view_src : op;
            if (dst_buf_src->buffer &&
                !ggml_backend_buft_is_kunlun_buft(ggml_backend_buffer_get_type(dst_buf_src->buffer), dev_index)) {
                return false;
            }
            if (op->nb[1] != (int64_t) ggml_row_size(op->type, op->ne[0])) {
                return false;
            }
            if (src0->nb[0] != (int64_t) ggml_type_size(src0->type)) {
                return false;
            }
            if (src0->ne[0] != op->ne[0]) {
                return false;
            }
            // implicit type cast is handled in eval (f32 activations -> f16 cache etc.)
            if (op->type != GGML_TYPE_F16 && op->type != GGML_TYPE_F32) {
                return false;
            }
            if (src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_F32) {
                return false;
            }
            fprintf(stderr, "KSETROWS accept\n");
            return true;
        }

        default:
            return false;
    }
}

static bool ggml_backend_kunlun_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    int dev_index = kunlun_dev_index(dev);
    return ggml_backend_buft_is_kunlun_buft(buft, dev_index) || ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_kunlun_device_i = {
    /* .get_name               = */ ggml_backend_kunlun_device_get_name,
    /* .get_description        = */ ggml_backend_kunlun_device_get_description,
    /* .get_memory             = */ ggml_backend_kunlun_device_get_memory,
    /* .get_type               = */ ggml_backend_kunlun_device_get_type,
    /* .get_props              = */ ggml_backend_kunlun_device_get_props,
    /* .init_backend           = */ ggml_backend_kunlun_device_init_backend,
    /* .get_buffer_type        = */ ggml_backend_kunlun_device_get_buffer_type,
    /* .get_host_buffer_type   = */ ggml_backend_kunlun_device_get_host_buffer_type,
    /* .buffer_from_host_ptr   = */ NULL,
    /* .supports_op            = */ ggml_backend_kunlun_device_supports_op,
    /* .supports_buft          = */ ggml_backend_kunlun_device_supports_buft,
    /* .offload_op             = */ NULL,
    /* .event_new              = */ NULL,
    /* .event_free             = */ NULL,
    /* .event_synchronize      = */ NULL,
};

//
// backend reg
//

static const char * ggml_backend_kunlun_reg_get_name(ggml_backend_reg_t reg) {
    return "KUNLUN";
    GGML_UNUSED(reg);
}

static size_t ggml_backend_kunlun_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return kunlun_state().devices.size();
}

static ggml_backend_dev_t ggml_backend_kunlun_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    static std::vector<ggml_backend_device> devs;
    static std::once_flag once;
    std::call_once(once, [] {
        for (size_t i = 0; i < kunlun_state().devices.size(); i++) {
            ggml_backend_device d;
            d.iface   = ggml_backend_kunlun_device_i;
            d.reg     = ggml_backend_kunlun_reg();
            d.context = (void *) (intptr_t) i;  // device index
            devs.push_back(d);
        }
    });
    GGML_ASSERT(index < devs.size());
    return &devs[index];
}

static const struct ggml_backend_reg_i ggml_backend_kunlun_reg_i = {
    /* .get_name         = */ ggml_backend_kunlun_reg_get_name,
    /* .get_device_count = */ ggml_backend_kunlun_reg_get_device_count,
    /* .get_device       = */ ggml_backend_kunlun_reg_get_device,
    /* .get_proc_address = */ NULL,
};

ggml_backend_reg_t ggml_backend_kunlun_reg(void) {
    static struct ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_kunlun_reg_i,
        /* .context     = */ NULL,
    };
    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_kunlun_reg)
