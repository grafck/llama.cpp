// ggml-kunlun: backend for Baidu Kunlun K100/K200 (KL1, PCI 1d22:3684)
//
// First cut: device memory residency + tensor data movement (H2D/D2H/D2D),
// graph_compute supports device-to-device CPY. Ops (MUL_MAT int8 GEMV etc.)
// land incrementally on top of the eager xdnn API.
//
// Runtime facts (see skill kunlun-xpu-driver):
//   - MEM_MAIN_CAPACITY = 7.88 GiB per PD; single xpu_malloc capped ~4 GB.
//   - xpu_set_device() is thread-local -> serialize all device calls under
//     one mutex and set the device before every call.
//   - everything is async; xpu_wait() before host-consumed D2H reads.

#include "ggml-kunlun.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <xpu/runtime.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;
using f16_kunlun = float16;

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define GGML_KUNLUN_MAX_ALLOC  ((4ull << 30) - (128ull << 20))  // 4.0 GB ok, 4.5 GB fails (measured)

// forward decls (interfaces are defined further down)
static ggml_backend_buffer_type_i ggml_backend_kunlun_buffer_type_interface(void);
static ggml_backend_buffer_i      ggml_backend_kunlun_buffer_interface(void);
static ggml_backend_device_i      ggml_backend_kunlun_device_interface(void);
ggml_backend_reg_t                ggml_backend_kunlun_reg(void);

// per-buffer context: device pointer + owning PD
struct kunlun_buffer_ctx {
    void * dev_ptr;
    int    device;
};

static bool ggml_backend_buffer_is_kunlun(const struct ggml_tensor * tensor) {
    return tensor->buffer &&
           tensor->buffer->buft &&
           tensor->buffer->buft->device &&
           tensor->buffer->buft->device->reg == ggml_backend_kunlun_reg();
}

// ---------------------------------------------------------------------------
// global device state

struct ggml_backend_kunlun_device_context {
    int          device;    // xpu device id
    uint64_t     total;     // MEM_MAIN_CAPACITY bytes
    std::atomic<size_t> allocated{0};
    std::string  name;
    std::string  description;
    std::string  pci_id;
    xdnn::Context * xdnn_ctx = nullptr;  // eager op context, created once per PD
};

static std::vector<ggml_backend_device>      g_kunlun_devices;
static std::vector<ggml_backend_buffer_type> g_kunlun_buft;
static std::once_flag                        g_kunlun_init_flag;
static int                                   g_kunlun_n_dev = 0;

// serializes all device calls: xpu_set_device is thread-local
static std::mutex                            g_kunlun_mutex;

// ---------------------------------------------------------------------------
// fp16 weight registry
//
// MUL_MAT weights are kept as f16 device buffers. An f16 weight tensor is
// aliased: the registry hands out tensor->data itself, no copy, no conversion
// (ggml [k rows, n cols] packed rows are exactly the [n x k] row-major layout
// that xdnn fc wants with w_trans=true). An f32 weight tensor gets a lazy
// on-device cast<float,f16> at first use; the result lives in tensor->extra
// and is reused by every later MUL_MAT that sees the same tensor.
//
// tensor->extra is the authoritative hit marker (a fresh tensor starts with
// extra = NULL, so a reused device address never aliases a dead tensor's
// f16 copy). The map below is the ownership registry: it lets a conversion
// overwrite free the dead entry at the same address and lets free_buffer
// sweep entries whose weight buffer is going away.

struct kunlun_f16_weight {
    void *  dev_f16;   // f16 device bytes for fc
    int     device;
    size_t  nbytes;    // f16 bytes, for memory accounting
    bool    owned;     // allocated by the registry (f32->f16), else aliased
};

// key: device address of the f32 weight tensor
static std::unordered_map<const void *, kunlun_f16_weight *> g_f16_registry;
static std::atomic<uint64_t> g_f16_conversions{0};

static ggml_backend_kunlun_device_context * kunlun_device_ctx(int device) {
    if (device < 0 || device >= (int) g_kunlun_devices.size()) {
        return nullptr;
    }
    return (ggml_backend_kunlun_device_context *) g_kunlun_devices[device].context;
}

// free an owned conversion and drop it from the registry (key = weight addr).
// caller must hold g_kunlun_mutex and have the target device selected.
static void kunlun_f16_weight_release(const ggml_tensor * w, kunlun_f16_weight * we) {
    if (!we->owned) {
        return;
    }
    xpu_wait();  // queued kernels may still read this buffer
    xpu_free(we->dev_f16);
    if (kunlun_device_ctx(we->device)) {
        kunlun_device_ctx(we->device)->allocated -= we->nbytes;
    }
    if (w && w->extra == we) {
        w->extra = nullptr;
    }
    auto it = g_f16_registry.find(w ? w->data : nullptr);
    if (it != g_f16_registry.end() && it->second == we) {
        g_f16_registry.erase(it);
    }
    delete we;
}

// drop every registry entry whose weight lives inside a buffer being freed.
// caller must hold g_kunlun_mutex and have the device selected.
static void kunlun_registry_sweep(uintptr_t base, size_t size) {
    for (auto it = g_f16_registry.begin(); it != g_f16_registry.end();) {
        const uintptr_t addr = (uintptr_t) it->first;
        if (addr >= base && addr < base + size) {
            kunlun_f16_weight * we = it->second;
            if (we->owned && kunlun_device_ctx(we->device)) {
                kunlun_device_ctx(we->device)->allocated -= we->nbytes;
            }
            if (we->owned) {
                xpu_free(we->dev_f16);
            }
            delete we;
            it = g_f16_registry.erase(it);
        } else {
            ++it;
        }
    }
}

// f32 -> f16 on device; caller holds g_kunlun_mutex, device selected
static kunlun_f16_weight * kunlun_f32_to_f16(const ggml_tensor * w,
                                             ggml_backend_kunlun_device_context * ctx,
                                             xdnn::Context * xdnn_ctx) {
    const int64_t n_elts = ggml_nelements(w);
    const size_t   n_f16 = (size_t) n_elts * sizeof(f16_kunlun);

    kunlun_f16_weight * we = new kunlun_f16_weight();
    we->device = ctx->device;
    we->nbytes = n_f16;
    we->owned  = true;

    if (n_f16 > GGML_KUNLUN_MAX_ALLOC || ctx->total - ctx->allocated < n_f16) {
        delete we;
        return nullptr;
    }
    if (xpu_malloc(&we->dev_f16, (uint64_t) n_f16) != 0 || we->dev_f16 == nullptr) {
        delete we;
        return nullptr;
    }
    ctx->allocated += n_f16;

    const int rc = xdnn::cast<float, f16_kunlun>(xdnn_ctx, (const float *) w->data,
                                                 (f16_kunlun *) we->dev_f16, n_elts);
    if (rc != 0) {
        xpu_wait();
        xpu_free(we->dev_f16);
        ctx->allocated -= n_f16;
        delete we;
        return nullptr;
    }

    g_f16_registry[w->data] = we;
    const uint64_t n = ++g_f16_conversions;
    GGML_LOG_INFO("kunlun: f16 weight registry: converted '%s' [%zu x %zu] (#%llu)\n",
                  w->name, (size_t) w->ne[0], (size_t) w->ne[1],
                  (unsigned long long) n);
    return we;
}

static void ggml_backend_kunlun_lazy_init(void) {
    std::call_once(g_kunlun_init_flag, []() {
        int count = 0;
        if (xpu_device_count(&count) != 0 || count <= 0) {
            g_kunlun_n_dev = 0;
            return;
        }
        g_kunlun_n_dev = count;
        for (int i = 0; i < count; i++) {
            uint64_t cap = 0;
            if (xpu_device_get_attr(&cap, XPUATTR_MEM_MAIN_CAPACITY, i) != 0) {
                cap = 0;
            }
            ggml_backend_kunlun_device_context * ctx = new ggml_backend_kunlun_device_context();
            ctx->device      = i;
            ctx->total       = cap;
            ctx->name        = "KUNLUN" + std::to_string(i);
            ctx->description = "Baidu Kunlun K200 (PD " + std::to_string(i) + ")";

            uint64_t pci = 0;
            if (xpu_device_get_attr(&pci, XPUATTR_PCI_ADDRESS, i) == 0) {
                // bit fields per runtime.h: [31:16] domain [15:8] bus [7:3] device [2:0] function
                char buf[32];
                snprintf(buf, sizeof(buf), "%04x:%02x:%02x.%02x",
                         (unsigned)((pci >> 16) & 0xffff),
                         (unsigned)((pci >> 8)  & 0xff),
                         (unsigned)((pci >> 3)  & 0x1f),
                         (unsigned)(pci & 0x7));
                ctx->pci_id = buf;
            }

            if (xpu_set_device(i) != 0) {
                GGML_LOG_ERROR("%s: xpu_set_device(%d) failed\n", __func__, i);
            } else {
                ctx->xdnn_ctx = xdnn::create_context();
                if (!ctx->xdnn_ctx) {
                    GGML_LOG_ERROR("%s: xdnn create_context for PD %d failed\n", __func__, i);
                }
            }

            ggml_backend_device dev = {
                /* .iface   = */ ggml_backend_kunlun_device_interface(),
                /* .reg     = */ ggml_backend_kunlun_reg(),
                /* .context = */ ctx,
            };
            g_kunlun_devices.push_back(dev);

            ggml_backend_buffer_type buft = {
                /* .iface   = */ ggml_backend_kunlun_buffer_type_interface(),
                /* .device  = */ &g_kunlun_devices.back(),
                /* .context = */ nullptr,
            };
            g_kunlun_buft.push_back(buft);
        }
    });
}

// ---------------------------------------------------------------------------
// buffer type

static const char * ggml_backend_kunlun_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return "KUNLUN";
}

static ggml_backend_buffer_t ggml_backend_kunlun_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) buft->device->context;

    if (size > GGML_KUNLUN_MAX_ALLOC) {
        GGML_LOG_ERROR("%s: request %zu bytes exceeds per-alloc cap %llu\n",
                       __func__, size, (unsigned long long) GGML_KUNLUN_MAX_ALLOC);
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    if (xpu_set_device(ctx->device) != 0) {
        return nullptr;
    }

    void * dev_ptr = nullptr;
    int rc = xpu_malloc(&dev_ptr, (uint64_t) size);
    if (rc != 0 || dev_ptr == nullptr) {
        GGML_LOG_ERROR("%s: xpu_malloc(%zu) on PD %d failed rc=%d\n", __func__, size, ctx->device, rc);
        return nullptr;
    }
    ctx->allocated += size;

    kunlun_buffer_ctx * bctx = new kunlun_buffer_ctx{ dev_ptr, ctx->device };

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
        ggml_backend_kunlun_buffer_interface(), bctx, size);
    return buffer;
}

static size_t ggml_backend_kunlun_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return 256;
}

static size_t ggml_backend_kunlun_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_KUNLUN_MAX_ALLOC;
}

static bool ggml_backend_kunlun_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return false;
}

static ggml_backend_buffer_type_i ggml_backend_kunlun_buffer_type_interface(void) {
    return {
        /* .get_name         = */ ggml_backend_kunlun_buffer_type_get_name,
        /* .alloc_buffer     = */ ggml_backend_kunlun_buft_alloc_buffer,
        /* .alloc_buffer_n   = */ nullptr,
        /* .get_alignment    = */ ggml_backend_kunlun_buft_get_alignment,
        /* .get_max_size     = */ ggml_backend_kunlun_buft_get_max_size,
        /* .get_alloc_size   = */ nullptr,  // defaults to ggml_nbytes
        /* .get_alloc_size_n = */ nullptr,
        /* .is_host          = */ ggml_backend_kunlun_buft_is_host,
    };
}

// ---------------------------------------------------------------------------
// buffer

static void ggml_backend_kunlun_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) buffer->buft->device->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(bctx->device);
    kunlun_registry_sweep((uintptr_t) bctx->dev_ptr, buffer->size);
    xpu_wait();
    xpu_free(bctx->dev_ptr);
    ctx->allocated -= buffer->size;
    delete bctx;
}

static void * ggml_backend_kunlun_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((kunlun_buffer_ctx *) buffer->context)->dev_ptr;
}

static void ggml_backend_kunlun_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                  const void * data, size_t offset, size_t size) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(bctx->device);
    int rc = xpu_memcpy((char *) tensor->data + offset, data, size, XPU_HOST_TO_DEVICE);
    xpu_wait();
    if (rc != 0) {
        GGML_LOG_ERROR("%s: H2D copy failed rc=%d (%s, %zu bytes)\n", __func__, rc, tensor->name, size);
    }
    // tensor contents changed: any cached f16 conversion is stale
    if (kunlun_f16_weight * we = (kunlun_f16_weight *) tensor->extra) {
        xpu_set_device(we->device);
        kunlun_f16_weight_release(tensor, we);
    }
}

static void ggml_backend_kunlun_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                  void * data, size_t offset, size_t size) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(bctx->device);
    int rc = xpu_memcpy(data, (const char *) tensor->data + offset, size, XPU_DEVICE_TO_HOST);
    xpu_wait();
    if (rc != 0) {
        GGML_LOG_ERROR("%s: D2H copy failed rc=%d (%s, %zu bytes)\n", __func__, rc, tensor->name, size);
    }
}

static void ggml_backend_kunlun_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                     uint8_t value, size_t offset, size_t size) {
    std::vector<char> staging(size, (char) value);
    ggml_backend_kunlun_buffer_set_tensor(buffer, tensor, staging.data(), offset, size);
}

static void ggml_backend_kunlun_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    std::vector<char> staging(buffer->size, (char) value);
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(bctx->device);
    xpu_memcpy(bctx->dev_ptr, staging.data(), buffer->size, XPU_HOST_TO_DEVICE);
    xpu_wait();
}

static bool ggml_backend_kunlun_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                  const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_kunlun(src)) {
        kunlun_buffer_ctx * dst_ctx = (kunlun_buffer_ctx *) buffer->context;
        kunlun_buffer_ctx * src_ctx = (kunlun_buffer_ctx *) src->buffer->context;
        size_t nbytes = ggml_nbytes(src);
        std::lock_guard<std::mutex> lock(g_kunlun_mutex);
        int rc;
        if (src_ctx->device == dst_ctx->device) {
            xpu_set_device(dst_ctx->device);
            rc = xpu_memcpy((char *) dst->data,
                            (const char *) src->data, nbytes, XPU_DEVICE_TO_DEVICE);
        } else {
            // cross-PD hop via xpu link
            rc = xpu_memcpy_peer(dst_ctx->device, (char *) dst->data,
                                 src_ctx->device, (const char *) src->data, nbytes);
        }
        xpu_wait();
        return rc == 0;
    }
    return false;  // host src handled by the default set_tensor path
}

static ggml_backend_buffer_i ggml_backend_kunlun_buffer_interface(void) {
    return {
        /* .free_buffer     = */ ggml_backend_kunlun_buffer_free_buffer,
        /* .get_base        = */ ggml_backend_kunlun_buffer_get_base,
        /* .init_tensor     = */ nullptr,
        /* .memset_tensor   = */ ggml_backend_kunlun_buffer_memset_tensor,
        /* .set_tensor      = */ ggml_backend_kunlun_buffer_set_tensor,
        /* .get_tensor      = */ ggml_backend_kunlun_buffer_get_tensor,
        /* .set_tensor_2d   = */ nullptr,
        /* .get_tensor_2d   = */ nullptr,
        /* .cpy_tensor      = */ ggml_backend_kunlun_buffer_cpy_tensor,
        /* .clear           = */ ggml_backend_kunlun_buffer_clear,
        /* .reset           = */ nullptr,
    };
}

// ---------------------------------------------------------------------------
// backend (graph compute)

static const char * ggml_backend_kunlun_name(ggml_backend_t backend) {
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) backend->device->context;
    return ctx->name.c_str();
}

static void ggml_backend_kunlun_free(ggml_backend_t backend) {
    delete backend;  // device contexts are static
}

// fc layout predicate: f32/f16 tensor whose rows are packed in memory
// (elements contiguous within a row and rows contiguous in memory), and no
// batch dims - exactly what fc consumes for activations and aliased weights.
static bool kunlun_rows_ready(const ggml_tensor * t) {
    const size_t es = ggml_element_size(t);
    return t->ne[0] >= 1 &&
           t->ne[2] == 1 &&
           t->ne[3] == 1 &&
           t->nb[0] == es &&
           t->nb[1] == es * (size_t) t->ne[0];
}

static enum ggml_status kunlun_mul_mat_exec(ggml_backend_kunlun_device_context * ctx,
                                            ggml_cgraph * cgraph, int node_idx) {
    ggml_tensor * dst = cgraph->nodes[node_idx];
    ggml_tensor * w   = dst->src[0];
    ggml_tensor * a   = dst->src[1];

    const int64_t k = w->ne[0];
    const int64_t n = w->ne[1];
    const int64_t m = a->ne[1];

    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    if (xpu_set_device(ctx->device) != 0 || !ctx->xdnn_ctx) {
        return GGML_STATUS_FAILED;
    }
    if (m == 0 || n == 0 || k == 0) {
        return GGML_STATUS_SUCCESS;  // empty batch
    }

    // --- weights: alias f16 bytes, or lazy f32 -> f16 conversion ----------
    void * w_f16 = nullptr;
    if (w->type == GGML_TYPE_F16) {
        w_f16 = w->data;  // already f16 on this PD: alias, zero copy
    } else {
        kunlun_f16_weight * we = (kunlun_f16_weight *) w->extra;
        if (!we) {
            we = kunlun_f32_to_f16(w, ctx, ctx->xdnn_ctx);
            if (!we) {
                return GGML_STATUS_FAILED;
            }
            w->extra = we;
        }
        w_f16 = we->dev_f16;
    }

    // --- activations: f16 bytes for fc (staged H2D when host-side) --------
    void * a_f16 = nullptr;
    void * x_scratch = nullptr;  // staging scratch, freed below
    if (a->type == GGML_TYPE_F16 && ggml_backend_buffer_is_kunlun(a)) {
        a_f16 = a->data;
    } else if (a->type == GGML_TYPE_F32 && ggml_backend_buffer_is_kunlun(a)) {
        const size_t xbytes = (size_t) k * m * sizeof(f16_kunlun);
        if (xpu_malloc(&x_scratch, (uint64_t) xbytes) != 0 || !x_scratch) {
            return GGML_STATUS_FAILED;
        }
        ctx->allocated += xbytes;
        if (xdnn::cast<float, f16_kunlun>(ctx->xdnn_ctx, (const float *) a->data,
                                          (f16_kunlun *) x_scratch, k * m) != 0) {
            xpu_wait();
            xpu_free(x_scratch);
            ctx->allocated -= xbytes;
            return GGML_STATUS_FAILED;
        }
        a_f16 = x_scratch;
    } else {
        // host-side activations: convert on host, one H2D hop
        const size_t xbytes = (size_t) k * m * sizeof(f16_kunlun);
        std::vector<f16_kunlun> stage((size_t) k * m);
        if (a->type == GGML_TYPE_F32) {
            const float * src = (const float *) a->data;
            for (size_t i = 0; i < stage.size(); i++) {
                stage[i] = f16_kunlun(src[i]);
            }
        } else {
            std::memcpy(stage.data(), a->data, xbytes);
        }
        if (xpu_malloc(&x_scratch, (uint64_t) xbytes) != 0 || !x_scratch) {
            return GGML_STATUS_FAILED;
        }
        ctx->allocated += xbytes;
        xpu_memcpy(x_scratch, stage.data(), xbytes, XPU_HOST_TO_DEVICE);
        a_f16 = x_scratch;
    }

    // --- output: f16 dst direct when contiguous, f32 dst via scratch+cast --
    void * y_f16;
    void * y_scratch = nullptr;
    const bool dst_direct = (dst->type == GGML_TYPE_F16) && ggml_backend_buffer_is_kunlun(dst);
    if (dst_direct) {
        y_f16 = dst->data;
    } else {
        const size_t ybytes = (size_t) n * m * sizeof(f16_kunlun);
        if (xpu_malloc(&y_scratch, (uint64_t) ybytes) != 0 || !y_scratch) {
            if (x_scratch) {
                xpu_wait();
                xpu_free(x_scratch);
                ctx->allocated -= (size_t) k * m * sizeof(f16_kunlun);
            }
            return GGML_STATUS_FAILED;
        }
        ctx->allocated += ybytes;
        y_f16 = y_scratch;
    }

    const int rc = xdnn::fc<f16_kunlun, f16_kunlun, f16_kunlun, short>(
        ctx->xdnn_ctx, (const f16_kunlun *) a_f16, (const f16_kunlun *) w_f16,
        (f16_kunlun *) y_f16, m, n, k, false, true, nullptr, nullptr, nullptr);

    enum ggml_status st = GGML_STATUS_SUCCESS;
    if (rc != 0) {
        GGML_LOG_ERROR("%s: fc rc=%d node '%s' [%lld x %lld x %lld]\n", __func__, rc,
                       dst->name, (long long) m, (long long) n, (long long) k);
        st = GGML_STATUS_FAILED;
    } else if (!dst_direct) {
        // f32 dst: cast the f16 scratch into it on device (dst is contiguous
        // rows by the supports_op gate, so the element order matches)
        const int rc2 = xdnn::cast<f16_kunlun, float>(ctx->xdnn_ctx,
            (const f16_kunlun *) y_f16, (float *) dst->data, (int64_t) n * m);
        if (rc2 != 0) {
            GGML_LOG_ERROR("%s: cast rc=%d node '%s'\n", __func__, rc2, dst->name);
            st = GGML_STATUS_FAILED;
        }
    }

    // scratch is read by queued kernels: drain the stream before freeing
    if (x_scratch) {
        xpu_wait();
        xpu_free(x_scratch);
        ctx->allocated -= (size_t) k * m * sizeof(f16_kunlun);
    }
    if (y_scratch) {
        xpu_wait();
        xpu_free(y_scratch);
        ctx->allocated -= (size_t) n * m * sizeof(f16_kunlun);
    }
    return st;
}

static enum ggml_status ggml_backend_kunlun_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) backend->device->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        switch (node->op) {
            case GGML_OP_CPY:
            case GGML_OP_CONT:
                {
                    // device-to-device copy within this PD (host transfers are
                    // split out of the graph by the scheduler)
                    ggml_tensor * src = node->src[0];
                    ggml_tensor * dst = node;
                    if (!ggml_backend_buffer_is_kunlun(src) || !ggml_backend_buffer_is_kunlun(dst)) {
                        GGML_LOG_ERROR("%s: CPY with non-device tensor\n", __func__);
                        return GGML_STATUS_FAILED;
                    }
                    kunlun_buffer_ctx * src_ctx = (kunlun_buffer_ctx *) src->buffer->context;
                    kunlun_buffer_ctx * dst_ctx = (kunlun_buffer_ctx *) dst->buffer->context;
                    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
                    int rc;
                    if (src_ctx->device == dst_ctx->device) {
                        xpu_set_device(ctx->device);
                        rc = xpu_memcpy((char *) dst->data,
                                        (const char *) src->data,
                                        ggml_nbytes(src), XPU_DEVICE_TO_DEVICE);
                    } else {
                        rc = xpu_memcpy_peer(dst_ctx->device, (char *) dst->data,
                                             src_ctx->device, (const char *) src->data,
                                             ggml_nbytes(src));
                    }
                    xpu_wait();
                    if (rc != 0) {
                        GGML_LOG_ERROR("%s: CPY rc=%d\n", __func__, rc);
                        return GGML_STATUS_FAILED;
                    }
                }
                break;
            case GGML_OP_MUL_MAT:
                {
                    const enum ggml_status st = kunlun_mul_mat_exec(ctx, cgraph, i);
                    if (st != GGML_STATUS_SUCCESS) {
                        return st;
                    }
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_name(node->op));
                return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_kunlun_synchronize(ggml_backend_t backend) {
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) backend->device->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(ctx->device);
    xpu_wait();
}

static ggml_backend_i ggml_backend_kunlun_interface(void) {
    return {
        /* .get_name             = */ ggml_backend_kunlun_name,
        /* .free                 = */ ggml_backend_kunlun_free,
        /* .set_tensor_async     = */ nullptr,
        /* .get_tensor_async     = */ nullptr,
        /* .set_tensor_2d_async  = */ nullptr,
        /* .get_tensor_2d_async  = */ nullptr,
        /* .cpy_tensor_async     = */ nullptr,
        /* .synchronize          = */ ggml_backend_kunlun_synchronize,
        /* .graph_plan_create    = */ nullptr,
        /* .graph_plan_free      = */ nullptr,
        /* .graph_plan_update    = */ nullptr,
        /* .graph_plan_compute   = */ nullptr,
        /* .graph_compute        = */ ggml_backend_kunlun_graph_compute,
        /* .event_record         = */ nullptr,
        /* .event_wait           = */ nullptr,
        /* .graph_optimize       = */ nullptr,
    };
}

static ggml_guid_t ggml_backend_kunlun_guid(void) {
    static ggml_guid guid = { 0x91, 0x4c, 0x55, 0x4e, 0x4c, 0x55, 0x4e, 0x21,
                              0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07 };  // "KUNLUN!"
    return &guid;
}

ggml_backend_t ggml_backend_kunlun_init(int device, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_kunlun_lazy_init();
    if (device < 0 || device >= g_kunlun_n_dev) {
        return nullptr;
    }

    ggml_backend * backend = new ggml_backend{
        /* .guid    = */ ggml_backend_kunlun_guid(),
        /* .iface   = */ ggml_backend_kunlun_interface(),
        /* .device  = */ &g_kunlun_devices[device],
        /* .context = */ nullptr,
    };
    return backend;
}

// ---------------------------------------------------------------------------
// device

static const char * ggml_backend_kunlun_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_kunlun_device_context * ctx = (ggml_backend_kunlun_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_kunlun_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_kunlun_device_context * ctx = (ggml_backend_kunlun_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_kunlun_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_kunlun_device_context * ctx = (ggml_backend_kunlun_device_context *) dev->context;
    *total = (size_t) ctx->total;
    *free  = (size_t) ctx->total > ctx->allocated ? (size_t) ctx->total - ctx->allocated : 0;
}

static enum ggml_backend_dev_type ggml_backend_kunlun_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_kunlun_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_kunlun_device_context * ctx = (ggml_backend_kunlun_device_context *) dev->context;
    memset(props, 0, sizeof(*props));
    props->name          = ctx->name.c_str();
    props->description   = ctx->description.c_str();
    props->memory_total  = (size_t) ctx->total;
    props->memory_free   = (size_t) ctx->total > ctx->allocated ? (size_t) ctx->total - ctx->allocated : 0;
    props->type          = GGML_BACKEND_DEVICE_TYPE_GPU;
    props->device_id     = ctx->pci_id.empty() ? nullptr : ctx->pci_id.c_str();
}

static ggml_backend_t ggml_backend_kunlun_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_kunlun_device_context * ctx = (ggml_backend_kunlun_device_context *) dev->context;
    return ggml_backend_kunlun_init(ctx->device, params);
}

static ggml_backend_buffer_type_t ggml_backend_kunlun_device_get_buffer_type(ggml_backend_dev_t dev) {
    size_t idx = dev - g_kunlun_devices.data();
    return &g_kunlun_buft[idx];
}

static bool kunlun_dev_supports_mul_mat(ggml_backend_dev_t dev, const ggml_tensor * op) {
    const ggml_tensor * w = op->src[0];
    const ggml_tensor * a = op->src[1];
    if (!w || !a) {
        return false;
    }

    // weights must live on this very PD: f16 is aliased raw, f32 is converted
    // into a device buffer of this PD. Other backends' weights stay CPU-side.
    if (!ggml_backend_buffer_is_kunlun(w)) {
        return false;
    }
    const kunlun_buffer_ctx * wc = (const kunlun_buffer_ctx *) w->buffer->context;
    const ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) dev->context;
    if (wc->device != ctx->device || !ctx->xdnn_ctx) {
        return false;
    }

    if (w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_F16) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
        return false;
    }

    // activations: same PD, or plain host memory (staged H2D in the op)
    if (a->buffer) {
        if (!ggml_backend_buffer_is_kunlun(a)) {
            return false;
        }
        const kunlun_buffer_ctx * ac = (const kunlun_buffer_ctx *) a->buffer->context;
        if (ac->device != ctx->device) {
            return false;
        }
    }

    // single matrix pair only: no batched/broadcast MUL_MAT
    if (w->ne[2] != 1 || w->ne[3] != 1) {
        return false;
    }
    if (a->ne[2] != 1 || a->ne[3] != 1) {
        return false;
    }
    if (op->ne[2] != 1 || op->ne[3] != 1) {
        return false;
    }

    // shapes
    if (a->ne[0] != w->ne[0]) {
        return false;
    }
    if (op->ne[0] != w->ne[1] || op->ne[1] != a->ne[1]) {
        return false;
    }

    // fc consumes packed rows: elements contiguous within a row, rows
    // contiguous in memory; empty tensors are fine (early-return in op)
    const bool empty = w->ne[0] == 0 || w->ne[1] == 0 || a->ne[1] == 0;
    if (!empty) {
        if (!kunlun_rows_ready(w) || !kunlun_rows_ready(a)) {
            return false;
        }
        if (!ggml_is_contiguous(op)) {
            return false;
        }
    }
    return true;
}

static bool ggml_backend_kunlun_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
            return true;  // input/output tensors live on this backend
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_backend_buffer_is_kunlun(op->src[0]) && ggml_backend_buffer_is_kunlun(op);
        case GGML_OP_MUL_MAT:
            return kunlun_dev_supports_mul_mat(dev, op);
        default:
            return false;
    }
}

static bool ggml_backend_kunlun_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return buft->device == dev;
}

static ggml_backend_device_i ggml_backend_kunlun_device_interface(void) {
    return {
        /* .get_name          = */ ggml_backend_kunlun_device_get_name,
        /* .get_description   = */ ggml_backend_kunlun_device_get_description,
        /* .get_memory        = */ ggml_backend_kunlun_device_get_memory,
        /* .get_type          = */ ggml_backend_kunlun_device_get_type,
        /* .get_props         = */ ggml_backend_kunlun_device_get_props,
        /* .init_backend      = */ ggml_backend_kunlun_device_init_backend,
        /* .get_buffer_type   = */ ggml_backend_kunlun_device_get_buffer_type,
        /* .get_host_buffer_type = */ nullptr,
        /* .buffer_from_host_ptr = */ nullptr,
        /* .supports_op       = */ ggml_backend_kunlun_device_supports_op,
        /* .supports_buft     = */ ggml_backend_kunlun_device_supports_buft,
        /* .offload_op        = */ nullptr,
        /* .event_new         = */ nullptr,
        /* .event_free        = */ nullptr,
        /* .event_synchronize = */ nullptr,
    };
}

// ---------------------------------------------------------------------------
// reg

static const char * ggml_backend_kunlun_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "Kunlun";
}

static size_t ggml_backend_kunlun_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    ggml_backend_kunlun_lazy_init();
    return (size_t) g_kunlun_n_dev;
}

static ggml_backend_dev_t ggml_backend_kunlun_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_UNUSED(reg);
    ggml_backend_kunlun_lazy_init();
    GGML_ASSERT((int) index < g_kunlun_n_dev);
    return &g_kunlun_devices[index];
}

static ggml_backend_reg_i ggml_backend_kunlun_reg_interface(void) {
    return {
        /* .get_name         = */ ggml_backend_kunlun_reg_get_name,
        /* .get_device_count = */ ggml_backend_kunlun_reg_get_device_count,
        /* .get_device       = */ ggml_backend_kunlun_reg_get_device,
        /* .get_proc_address = */ nullptr,
    };
}

ggml_backend_reg_t ggml_backend_kunlun_reg(void) {
    static struct ggml_backend_reg kunlun_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_kunlun_reg_interface(),
        /* .context     = */ nullptr,
    };
    return &kunlun_reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_kunlun_reg)
