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

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
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
};

static std::vector<ggml_backend_device>      g_kunlun_devices;
static std::vector<ggml_backend_buffer_type> g_kunlun_buft;
static std::once_flag                        g_kunlun_init_flag;
static int                                   g_kunlun_n_dev = 0;

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
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            return ggml_backend_buffer_is_kunlun(op->src[0]) && ggml_backend_buffer_is_kunlun(op);
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
