#pragma once

#include <ggml.h>

#include "ggml-kunlun-convert.h"

// y[m,n] = x[m,k] * w[n,k]^T with ggml [k,n] weight layout. w_fmt describes how
// the weight bytes are stored on device; w_maxptr is the device fp32 scalar for
// int8 weights. Returns 0 on success.
int kunlun_ops_mul_mat(int device, const void * x, const void * w, void * y,
                       int64_t m, int64_t n, int64_t k,
                       enum kunlun_tensor_fmt w_fmt, const float * w_maxptr,
                       enum ggml_type x_type, enum ggml_type y_type);
