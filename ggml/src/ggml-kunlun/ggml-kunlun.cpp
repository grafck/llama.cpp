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
#include "ggml-kunlun-convert.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <xpu/runtime.h>
#include "ggml-kunlun-ops.h"

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

// ---------------------------------------------------------------------------
// weight registry: keyed by tensor device pointer (tensor->data), records the
// on-device format after host-side conversion at load time. op dispatch reads
// this to feed fc the right weight format + w_maxptr scale.
//
// invariant: kunlun_weight_entry.n == ggml_nelements(tensor) so a recycled
// device pointer from a smaller tensor never matches.

static std::unordered_map<void *, kunlun_weight_entry> g_kunlun_weights;

// host staging for chunked set_tensor writes of converted (quantized) tensors
struct kunlun_staging {
    std::vector<char> buf;
    size_t            filled;
};

static std::unordered_map<void *, kunlun_staging> g_kunlun_staging;

// caller must hold g_kunlun_mutex
static const kunlun_weight_entry * kunlun_weight_lookup(const struct ggml_tensor * tensor) {
    auto it = g_kunlun_weights.find(tensor->data);
    if (it == g_kunlun_weights.end()) {
        return nullptr;
    }
    if (it->second.n != ggml_nelements(tensor) || it->second.src_type != tensor->type) {
        return nullptr;
    }
    return &it->second;
}

// on-device format of a tensor with no registry entry
static enum kunlun_tensor_fmt kunlun_fmt_from_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return KUNLUN_FMT_F16;
        case GGML_TYPE_F32: return KUNLUN_FMT_F32;
        default:            return KUNLUN_FMT_NONE;
    }
}

// a device-to-device weight copy carries its format with the bytes; the copy gets
// its own maxptr so free_buffer never double-frees a shared scale
// caller must hold g_kunlun_mutex
static void kunlun_mirror_entry(const ggml_tensor * src, ggml_tensor * dst) {
    const kunlun_weight_entry * src_entry = kunlun_weight_lookup(src);
    if (src_entry == nullptr) {
        return;
    }
    kunlun_weight_entry dst_entry = *src_entry;
    dst_entry.src_type = dst->type;
    dst_entry.n        = ggml_nelements(dst);
    if (dst_entry.maxptr_dev != nullptr) {
        float * maxptr = nullptr;
        if (xpu_malloc((void **) &maxptr, sizeof(float)) == 0) {
            xpu_wait();
            maxptr[0] = dst_entry.max_abs;
            xpu_wait();
        } else {
            maxptr = nullptr;
        }
        dst_entry.maxptr_dev = maxptr;
    }
    g_kunlun_weights[dst->data] = dst_entry;
}

// convert a full quantized payload to int8 + H2D; caller holds g_kunlun_mutex
// env GGML_KUNLUN_VERIFY=1: read the payload back from the PD and check
// |w - q*scale| <= 0.5*scale per element (catches DMA corruption and bad scales)
static bool kunlun_upload_converted(ggml_tensor * tensor, const void * data) {
    const int64_t n = ggml_nelements(tensor);

    std::vector<float> f32(n);
    if (!kunlun_dequantize_to_f32(data, tensor->type, f32.data(), n)) {
        GGML_LOG_ERROR("%s: no host dequantize path for type %s (%s)\n",
                       __func__, ggml_type_name(tensor->type), tensor->name);
        return false;
    }

    std::vector<int8_t> q8(n);
    float scale = 0.0f;
    float max_abs = kunlun_quantize_f32_to_int8(f32.data(), n, q8.data(), &scale);

    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) tensor->buffer->context;
    xpu_set_device(bctx->device);
    int rc = xpu_memcpy(tensor->data, q8.data(), kunlun_int8_payload_size(tensor), XPU_HOST_TO_DEVICE);
    xpu_wait();
    if (rc != 0) {
        GGML_LOG_ERROR("%s: H2D int8 failed rc=%d (%s)\n", __func__, rc, tensor->name);
        return false;
    }

    if (getenv("GGML_KUNLUN_VERIFY")) {
        std::vector<int8_t> rb(n);
        rc = xpu_memcpy(rb.data(), tensor->data, kunlun_int8_payload_size(tensor), XPU_DEVICE_TO_HOST);
        xpu_wait();
        float max_err = 0.0f;
        if (rc == 0) {
            const float tol = 0.5f * scale + 1e-6f * (max_abs > 0 ? max_abs : 1.0f);
            for (int64_t i = 0; i < n; i++) {
                float err = fabsf(f32[i] - (float) rb[i] * scale);
                if (err > max_err) {
                    max_err = err;
                }
            }
            if (max_err > tol) {
                GGML_LOG_ERROR("%s: verify failed (%s): max_err=%g > tol=%g (scale=%g)\n",
                               __func__, tensor->name, (double) max_err, (double) tol, (double) scale);
                return false;
            }
        } else {
            GGML_LOG_ERROR("%s: verify D2H failed rc=%d (%s)\n", __func__, rc, tensor->name);
            return false;
        }
        GGML_LOG_INFO("%s: verify OK (%s, n=%lld, scale=%g, max_abs=%g, max_err=%g)\n",
                      __func__, tensor->name, (long long) n, (double) scale, (double) max_abs, (double) max_err);
    }

    // fc reads w_maxptr as a device float holding max_abs; KL1 device memory is
    // host-mapped but we still go through xpu_memcpy to keep the DMA path explicit
    float * maxptr_dev = nullptr;
    if (xpu_malloc((void **) &maxptr_dev, sizeof(float)) != 0) {
        GGML_LOG_ERROR("%s: maxptr alloc failed (%s)\n", __func__, tensor->name);
        return false;
    }
    xpu_wait();
    if (xpu_memcpy(maxptr_dev, &max_abs, sizeof(float), XPU_HOST_TO_DEVICE) != 0) {
        xpu_free(maxptr_dev);
        GGML_LOG_ERROR("%s: maxptr H2D failed (%s)\n", __func__, tensor->name);
        return false;
    }
    xpu_wait();

    g_kunlun_weights[tensor->data] = kunlun_weight_entry{
        /* .fmt        = */ KUNLUN_FMT_INT8,
        /* .src_type   = */ tensor->type,
        /* .max_abs    = */ max_abs,
        /* .scale      = */ scale,
        /* .n          = */ n,
        /* .maxptr_dev = */ maxptr_dev,
    };
    return true;
}

static bool ggml_backend_buffer_is_kunlun(const struct ggml_tensor * tensor) {
    return tensor->buffer &&
           tensor->buffer->buft &&
           tensor->buffer->buft->device &&
           tensor->buffer->buft->device->reg == ggml_backend_kunlun_reg();
}

// global device state

struct ggml_backend_kunlun_device_context {
    int          device;    // xpu device id
    uint64_t     total;     // MEM_MAIN_CAPACITY bytes
    std::atomic<size_t> allocated{0};
    std::string  name;
    std::string  description;
    std::string  pci_id;
};

static std::vector<ggml_backend_device>      g_kunlun_devices;
static std::vector<ggml_backend_buffer_type> g_kunlun_buft;
static std::once_flag                        g_kunlun_init_flag;
static int                                   g_kunlun_n_dev = 0;

// ---------------------------------------------------------------------------
// serializes all device calls: xpu_set_device is thread-local
static std::mutex                            g_kunlun_mutex;

static void ggml_backend_kunlun_lazy_init(void) {
    std::call_once(g_kunlun_init_flag, []() {
        int count = 0;
        if (xpu_device_count(&count) != 0 || count <= 0) {
            g_kunlun_n_dev = 0;
            return;
        }
        g_kunlun_n_dev = count;
        // addresses of elements are handed out (buft.device, &devices[i]): no
        // reallocation may happen after init -> reserve the full size up front
        g_kunlun_devices.reserve(count);
        g_kunlun_buft.reserve(count);
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

static size_t ggml_backend_kunlun_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_UNUSED(buft);
    // quantized weights are stored as int8 payloads after host conversion
    return kunlun_weight_alloc_size(tensor);
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
        /* .get_alloc_size   = */ ggml_backend_kunlun_buft_get_alloc_size,
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
    xpu_wait();
    xpu_free(bctx->dev_ptr);
    ctx->allocated -= buffer->size;
    // drop registry/staging entries pointing into this allocation
    char * base = (char *) bctx->dev_ptr;
    for (auto it = g_kunlun_weights.begin(); it != g_kunlun_weights.end();) {
        char * p = (char *) it->first;
        if (p >= base && p < base + buffer->size) {
            if (it->second.maxptr_dev) {
                xpu_free(it->second.maxptr_dev);
            }
            it = g_kunlun_weights.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = g_kunlun_staging.begin(); it != g_kunlun_staging.end();) {
        char * p = (char *) it->first;
        if (p >= base && p < base + buffer->size) {
            it = g_kunlun_staging.erase(it);
        } else {
            ++it;
        }
    }
    delete bctx;
}

static void * ggml_backend_kunlun_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ((kunlun_buffer_ctx *) buffer->context)->dev_ptr;
}

static void ggml_backend_kunlun_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor,
                                                  const void * data, size_t offset, size_t size) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);

    if (kunlun_type_is_converted(tensor->type)) {
        const size_t total = ggml_nbytes(tensor);

        if (offset == 0 && size == total) {
            g_kunlun_staging.erase(tensor->data);
            if (!kunlun_upload_converted(tensor, data)) {
                GGML_LOG_ERROR("%s: int8 conversion failed (%s, %s)\n",
                               __func__, tensor->name, ggml_type_name(tensor->type));
            }
            return;
        }

        // chunked write: stage host-side until the full payload has arrived
        kunlun_staging & st = g_kunlun_staging[tensor->data];
        if (st.buf.size() < total) {
            st.buf.resize(total);
            st.filled = 0;
        }
        memcpy(st.buf.data() + offset, data, size);
        st.filled += size;
        if (st.filled >= total) {
            g_kunlun_staging.erase(tensor->data);
            if (!kunlun_upload_converted(tensor, st.buf.data())) {
                GGML_LOG_ERROR("%s: int8 conversion failed (%s, %s)\n",
                               __func__, tensor->name, ggml_type_name(tensor->type));
            }
        }
        return;
    }

    xpu_set_device(bctx->device);
    int rc = xpu_memcpy((char *) tensor->data + offset, data, size, XPU_HOST_TO_DEVICE);
    xpu_wait();
    if (rc != 0) {
        GGML_LOG_ERROR("%s: H2D copy failed rc=%d (%s, %zu bytes)\n", __func__, rc, tensor->name, size);
        return;
    }
    if (offset == 0 && size == ggml_nbytes(tensor)) {
        enum kunlun_tensor_fmt fmt = tensor->type == GGML_TYPE_F16 ? KUNLUN_FMT_F16 : KUNLUN_FMT_F32;
        const kunlun_weight_entry * old = kunlun_weight_lookup(tensor);
        if (old && old->maxptr_dev) {
            xpu_free(old->maxptr_dev);
        }
        g_kunlun_weights[tensor->data] = kunlun_weight_entry{
            /* .fmt        = */ fmt,
            /* .src_type   = */ tensor->type,
            /* .max_abs    = */ 0.0f,
            /* .scale      = */ 0.0f,
            /* .n          = */ ggml_nelements(tensor),
            /* .maxptr_dev = */ nullptr,
        };
    }
}

static void ggml_backend_kunlun_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor,
                                                  void * data, size_t offset, size_t size) {
    kunlun_buffer_ctx * bctx = (kunlun_buffer_ctx *) buffer->context;
    std::lock_guard<std::mutex> lock(g_kunlun_mutex);
    xpu_set_device(bctx->device);

    const kunlun_weight_entry * entry = kunlun_weight_lookup(tensor);

    if (!entry || entry->fmt != KUNLUN_FMT_INT8) {
        int rc = xpu_memcpy(data, (const char *) tensor->data + offset, size, XPU_DEVICE_TO_HOST);
        xpu_wait();
        if (rc != 0) {
            GGML_LOG_ERROR("%s: D2H copy failed rc=%d (%s, %zu bytes)\n", __func__, rc, tensor->name, size);
        }
        return;
    }

    // device holds int8 payload: read it back, dequantize, requantize into the
    // tensor's ggml-native blocks, hand out the requested byte range
    const int64_t n   = entry->n;
    const size_t  nat = ggml_nbytes(tensor);

    std::vector<int8_t> q8(n);
    int rc = xpu_memcpy(q8.data(), tensor->data, kunlun_int8_payload_size(tensor), XPU_DEVICE_TO_HOST);
    xpu_wait();
    if (rc != 0) {
        GGML_LOG_ERROR("%s: D2H int8 failed rc=%d (%s)\n", __func__, rc, tensor->name);
        return;
    }

    std::vector<char> native(nat);
    if (!kunlun_requantize_int8(q8.data(), entry->scale, entry->src_type, n,
                                native.data(), nat)) {
        GGML_LOG_ERROR("%s: no host requantize path for %s (%s)\n",
                       __func__, ggml_type_name(entry->src_type), tensor->name);
        memset(native.data(), 0, nat);
    }

    if (offset == 0 && size == nat) {
        memcpy(data, native.data(), nat);
        return;
    }
    // partial reads: ggml byte ranges are block-aligned, so element indices are exact
    const size_t row = ggml_row_size(tensor->type, 1);
    int64_t e0 = row ? (int64_t) (offset / row) : 0;
    int64_t e1 = row ? (int64_t) ((offset + size + row - 1) / row) : n;
    const size_t b0 = ggml_row_size(tensor->type, e0);
    const size_t b1 = ggml_row_size(tensor->type, e1 > n ? n : e1);
    memcpy(data, native.data() + b0, b1 - b0);
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
        if (rc == 0) {
            // format travels with the bytes: mirror the src registry entry onto dst
            kunlun_mirror_entry(src, dst);
            return true;
        }
        return false;
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

// ---------------------------------------------------------------------------
// ops

// fc computes y[m,n] = x[m,k] * w[n,k]^T. ggml weight tensors are [k,n] row-major,
// which is exactly the [n,k] matrix fc reads with w_trans=true.
// caller must hold g_kunlun_mutex
static enum ggml_status ggml_backend_kunlun_mul_mat(ggml_backend_t backend, ggml_tensor * node) {
    ggml_backend_kunlun_device_context * dev =
        (ggml_backend_kunlun_device_context *) backend->device->context;

    const ggml_tensor * a = node->src[0];  // weights, ne = [k, n]
    const ggml_tensor * b = node->src[1];  // activations, ne = [k, m]

    const kunlun_weight_entry * w = kunlun_weight_lookup(a);
    enum kunlun_tensor_fmt w_fmt = (w && w->fmt == KUNLUN_FMT_INT8)
                                   ? KUNLUN_FMT_INT8
                                   : kunlun_fmt_from_type(a->type);

    int rc = kunlun_ops_mul_mat(dev->device, b->data, a->data, node->data,
                                b->ne[1], a->ne[1], a->ne[0],
                                w_fmt, w ? w->maxptr_dev : nullptr,
                                b->type, node->type);
    if (rc != 0) {
        GGML_LOG_ERROR("%s: fc rc=%d (m=%lld n=%lld k=%lld, %s %s)\n",
                       __func__, rc, (long long) b->ne[1], (long long) a->ne[1],
                       (long long) a->ne[0], a->name, ggml_type_name(a->type));
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

// read ne elements of a device tensor into host fp32, honoring the int8 registry
// caller must hold g_kunlun_mutex
static bool kunlun_read_f32(const ggml_tensor * tensor, std::vector<float> & out) {
    const int64_t n = ggml_nelements(tensor);
    out.resize(n);

    const kunlun_weight_entry * e = kunlun_weight_lookup(tensor);
    if (e && e->fmt == KUNLUN_FMT_INT8) {
        std::vector<int8_t> q8(n);
        if (xpu_memcpy(q8.data(), tensor->data, kunlun_int8_payload_size(tensor), XPU_DEVICE_TO_HOST) != 0) {
            return false;
        }
        xpu_wait();
        for (int64_t i = 0; i < n; i++) {
            out[i] = (float) q8[i] * e->scale;
        }
        return true;
    }

    if (tensor->type == GGML_TYPE_F32) {
        return xpu_memcpy(out.data(), tensor->data, n * sizeof(float), XPU_DEVICE_TO_HOST) == 0 &&
               (xpu_wait(), true);
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(n);
        if (xpu_memcpy(f16.data(), tensor->data, n * sizeof(ggml_fp16_t), XPU_DEVICE_TO_HOST) != 0) {
            return false;
        }
        xpu_wait();
        for (int64_t i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(f16[i]);
        }
        return true;
    }
    GGML_LOG_ERROR("%s: cannot read device tensor %s of type %s\n",
                   __func__, tensor->name, ggml_type_name(tensor->type));
    return false;
}

// write host fp32 values into device memory of a tensor's ggml type
// caller must hold g_kunlun_mutex
static bool kunlun_write_f32(const ggml_tensor * tensor, void * dst, const float * src, int64_t n) {
    if (tensor->type == GGML_TYPE_F32) {
        return xpu_memcpy(dst, src, n * sizeof(float), XPU_HOST_TO_DEVICE) == 0 &&
               (xpu_wait(), true);
    }
    if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> f16(n);
        for (int64_t i = 0; i < n; i++) {
            f16[i] = ggml_fp32_to_fp16(src[i]);
        }
        return xpu_memcpy(dst, f16.data(), n * sizeof(ggml_fp16_t), XPU_HOST_TO_DEVICE) == 0 &&
               (xpu_wait(), true);
    }
    GGML_LOG_ERROR("%s: writing into %s of type %s unsupported\n",
                   __func__, tensor->name, ggml_type_name(tensor->type));
    return false;
}

// KV append: dst = cache (this node), src0 = row values, src1 = row indices.
// Mirrors ggml_compute_forward_set_rows_impl: cache is [nc, ne1, ne02, ne03],
// idx is [ne11, ne12], the row for (i, i02, i03) is read from idx at
// (i, i02 % ne11, i03 % ne12). caller must hold g_kunlun_mutex
static enum ggml_status ggml_backend_kunlun_set_rows(ggml_backend_t backend, ggml_tensor * node) {
    ggml_backend_kunlun_device_context * dev =
        (ggml_backend_kunlun_device_context *) backend->device->context;

    ggml_tensor * cache = node;
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];

    if (!ggml_backend_buffer_is_kunlun(cache)) {
        GGML_LOG_ERROR("%s: SET_ROWS cache is not a device tensor\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (!src0 || !src1 ||
        (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16) ||
        (src1->type != GGML_TYPE_I32 && src1->type != GGML_TYPE_I64)) {
        GGML_LOG_ERROR("%s: SET_ROWS contract violated (values %s, idx %s)\n",
                       __func__, src0 ? ggml_type_name(src0->type) : "NULL",
                       src1 ? ggml_type_name(src1->type) : "NULL");
        return GGML_STATUS_FAILED;
    }
    if (cache->ne[0] != src0->ne[0] || cache->ne[2] % src1->ne[1] != 0 || cache->ne[3] % src1->ne[2] != 0) {
        GGML_LOG_ERROR("%s: SET_ROWS shape mismatch\n", __func__);
        return GGML_STATUS_FAILED;
    }

    xpu_set_device(dev->device);

    const int64_t nc = cache->ne[0];
    const int64_t nr = src0->ne[1];
    const int64_t n_idx = ggml_nelements(src1);

    std::vector<int64_t> idx(n_idx);
    if (src1->type == GGML_TYPE_I64) {
        if (ggml_backend_buffer_is_kunlun(src1)) {
            if (xpu_memcpy(idx.data(), src1->data, n_idx * sizeof(int64_t), XPU_DEVICE_TO_HOST) != 0) {
                GGML_LOG_ERROR("%s: SET_ROWS idx D2H failed\n", __func__);
                return GGML_STATUS_FAILED;
            }
            xpu_wait();
        } else {
            memcpy(idx.data(), src1->data, n_idx * sizeof(int64_t));
        }
    } else {
        std::vector<int32_t> idx32(n_idx);
        if (ggml_backend_buffer_is_kunlun(src1)) {
            if (xpu_memcpy(idx32.data(), src1->data, n_idx * sizeof(int32_t), XPU_DEVICE_TO_HOST) != 0) {
                GGML_LOG_ERROR("%s: SET_ROWS idx D2H failed\n", __func__);
                return GGML_STATUS_FAILED;
            }
            xpu_wait();
        } else {
            memcpy(idx32.data(), src1->data, n_idx * sizeof(int32_t));
        }
        for (int64_t i = 0; i < n_idx; i++) {
            idx[i] = idx32[i];
        }
    }

    std::vector<float> row(nc);
    const size_t row_bytes = ggml_row_size(src0->type, nc);

    for (int64_t i03 = 0; i03 < cache->ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < cache->ne[2]; i02++) {
            for (int64_t i = 0; i < nr; i++) {
                const int64_t i1 = idx[i +
                    (i02 % src1->ne[1]) * src1->ne[0] +
                    (i03 % src1->ne[2]) * src1->ne[0] * src1->ne[1]];
                if (i1 < 0 || i1 >= cache->ne[1]) {
                    GGML_LOG_ERROR("%s: SET_ROWS index %lld out of range %lld\n",
                                   __func__, (long long) i1, (long long) cache->ne[1]);
                    return GGML_STATUS_FAILED;
                }

                const char * src_ptr = (const char *) src0->data +
                    i * src0->nb[1] + i02 * src0->nb[2] + i03 * src0->nb[3];
                if (ggml_backend_buffer_is_kunlun(src0)) {
                    std::vector<char> stage(row_bytes);
                    if (xpu_memcpy(stage.data(), src_ptr, row_bytes, XPU_DEVICE_TO_HOST) != 0) {
                        GGML_LOG_ERROR("%s: SET_ROWS row D2H failed\n", __func__);
                        return GGML_STATUS_FAILED;
                    }
                    xpu_wait();
                    src_ptr = stage.data();
                }
                if (src0->type == GGML_TYPE_F32) {
                    memcpy(row.data(), src_ptr, nc * sizeof(float));
                } else {
                    const ggml_fp16_t * f16 = (const ggml_fp16_t *) src_ptr;
                    for (int64_t j = 0; j < nc; j++) {
                        row[j] = ggml_fp16_to_fp32(f16[j]);
                    }
                }

                char * dst_ptr = (char *) cache->data + i1 * cache->nb[1] + i02 * cache->nb[2] + i03 * cache->nb[3];
                if (!kunlun_write_f32(cache, dst_ptr, row.data(), nc)) {
                    GGML_LOG_ERROR("%s: SET_ROWS row H2D failed\n", __func__);
                    return GGML_STATUS_FAILED;
                }
            }
        }
    }
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_kunlun_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_kunlun_device_context * ctx =
        (ggml_backend_kunlun_device_context *) backend->device->context;

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // fc rejects m=0 with rc=1; views/reshapes carry no compute
        if (ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
            node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE) {
            if (ggml_is_empty(node)) {
                GGML_LOG_DEBUG("%s: empty node '%s' (%s) skipped\n",
                               __func__, node->name, ggml_op_name(node->op));
            }
            continue;
        }

        std::lock_guard<std::mutex> lock(g_kunlun_mutex);

        switch (node->op) {
            case GGML_OP_MUL_MAT:
                if (ggml_backend_kunlun_mul_mat(backend, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SET_ROWS:
                if (ggml_backend_kunlun_set_rows(backend, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
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

                    if (src->type != dst->type) {
                        // ggml uses CPY for casts; go through a host fp32 staging buffer
                        std::vector<float> host;
                        if (!kunlun_read_f32(src, host) ||
                            !kunlun_write_f32(dst, dst->data, host.data(), (int64_t) host.size())) {
                            GGML_LOG_ERROR("%s: CPY conversion %s -> %s failed\n",
                                          __func__, ggml_type_name(src->type), ggml_type_name(dst->type));
                            return GGML_STATUS_FAILED;
                        }
                        break;
                    }

                    kunlun_buffer_ctx * src_ctx = (kunlun_buffer_ctx *) src->buffer->context;
                    kunlun_buffer_ctx * dst_ctx = (kunlun_buffer_ctx *) dst->buffer->context;
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
                    kunlun_mirror_entry(src, dst);
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

static bool ggml_backend_kunlun_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        // view ops are folded by the graph builder; NONE is the pre-allocation
        // probe for tensors like the KV cache (must return true so the tensor
        // stays assigned to this device's buffer)
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_backend_buffer_is_kunlun(op->src[0]) && ggml_backend_buffer_is_kunlun(op);
        case GGML_OP_MUL_MAT:
            {
                const ggml_tensor * src0 = op->src[0];
                const ggml_tensor * src1 = op->src[1];
                if (!src0 || !src1) {
                    return false;
                }
                if (src0->ne[0] != src1->ne[0]) {
                    return false;
                }
                // fc is a plain 2D GEMM
                if (src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
                    return false;
                }
                if (!ggml_is_contiguous(src0) || !ggml_is_contiguous(src1)) {
                    return false;
                }
                // buffers are null during the pre-allocation probe
                if (src0->buffer && !ggml_backend_buffer_is_kunlun(src0)) {
                    return false;
                }
                if (src1->buffer && !ggml_backend_buffer_is_kunlun(src1)) {
                    return false;
                }
                if (kunlun_type_is_converted(src0->type)) {
                    return true;  // requantized to int8 when uploaded
                }
                return src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32;
            }
        case GGML_OP_SET_ROWS:
            // KV cache updates; accept on type contract only (works for the
            // pre-allocation probe where view buffers are NULL)
            GGML_LOG_DEBUG("%s: SET_ROWS probe: dst=%s src0=%s idx=%s src1=%s\n",
                           __func__, ggml_type_name(op->type),
                           op->src[0] ? ggml_type_name(op->src[0]->type) : "NULL",
                           op->src[1] ? ggml_type_name(op->src[1]->type) : "NULL",
                           op->src[2] ? ggml_type_name(op->src[2]->type) : "NULL");
            return op->src[0] && op->src[1] &&
                   op->src[0]->type == op->type &&
                   (op->src[1]->type == GGML_TYPE_I32 || op->src[1]->type == GGML_TYPE_I64);
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
