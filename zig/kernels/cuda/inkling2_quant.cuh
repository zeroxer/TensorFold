// Inkling2 quantized CUDA ABI. Details: zig/src/families/inkling2/ABI.txt.
#pragma once

#include <stdint.h>

#if defined(__CUDACC__)

// BF16 W[N,K] times BF16-valued float x[K], with FP32 accumulation and optional BF16 output.
extern "C" __global__ void tf_ink2_dense(const float* x, const uint16_t* w, float* y, int N, int K,
                                         int round_bf16);

// Quantize and reconstruct one full activation row in its checkpoint-defined NVFP4 or MXFP8 format.
extern "C" __global__ void tf_ink2_quant(const float* x, float* y, int K, int format);

// Stored-code matvec of one full expert, FP32 accumulation and optional BF16 output.
extern "C" __global__ void tf_ink2_qmv(const float* x, const unsigned char* w,
                                       const unsigned char* scales, const float* amax,
                                       const float* denom, float* y, int N, int K, int format,
                                       int round_bf16);

#endif  // __CUDACC__
