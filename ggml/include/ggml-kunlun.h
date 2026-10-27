#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_kunlun_init(int device, const char * params);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_kunlun_reg(void);

#ifdef  __cplusplus
}
#endif
