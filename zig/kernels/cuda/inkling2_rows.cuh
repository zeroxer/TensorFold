// Inkling2 row CUDA ABI. Details: zig/src/families/inkling2/ABI.txt.
#pragma once

#if defined(__CUDACC__)

#include <stdint.h>

extern "C" {

// Weighted RMSNorm of one row: FP32 reduction and BF16-rounded float output.
__global__ void tf_ink2_rms(const float* x, const uint16_t* w, float* y, int H, float eps);

// Depth attention over RMS-normalized sources, including normalized values.
__global__ void tf_ink2_attnres(const float* states, const float* partial, const float* query, float* y,
                                int H, int depth, int queries, float eps);

// Four-tap causal convolution, identity term and raw-input history update.
__global__ void tf_ink2_sconv(const float* x, const uint16_t* w, float* hist, const float* dyn, float* y,
                              int H, int head_dim, int shared_taps, float scale, int rounded);

// BF16 residual addition; replace drops the old residual at a block boundary.
__global__ void tf_ink2_add(const float* x, const float* y, float* z, int H, int replace, int rounded);

// Interleaved SiTU; routing gamma precedes BF16 rounding and activation quantization.
__global__ void tf_ink2_swiglu(const float* gu, float* y, int I, float gamma);

// Top-k by sigmoid+bias; unbiased normalized weights and stable lower-id ties.
__global__ void tf_ink2_route(const float* logits, const float* bias, int* ids, float* weights, int E,
                              int topk, float scale);

// Grid ceil(H/256), block256, no shared. Only the owning rank reads this token's BF16 table row.
__global__ void tf_ink2_embed(const uint16_t* table, float* out, int id, int first, int valid, int H);
// Grid ceil(H/256), block256, no shared. Nonalias input [count,H]; fixed ascending-order fp32 sum.
__global__ void tf_ink2_merge(const float* parts, float* out, int H, int count, int round_bf16);
// Grid ceil(count/256), block256, no shared. Invalid padded vocabulary rows become -Inf.
__global__ void tf_ink2_softcap(const float* logits, float* out, int count, int valid, float cap);
// Windowed variants preserve the scalar formulas and reduction order; grids span rows or flattened row elements.
__global__ void tf_ink2_rms_rows(const float* x, long long x_stride, const uint16_t* w, float* y,
                                 long long y_stride, int H, int rows, float eps);
__global__ void tf_ink2_attnres_rows(const float* states, const float* partial, const float* query, float* y,
                                     int H, int depth, int queries, int stack_stride, float eps, int rows);
__global__ void tf_ink2_sconv_window(const float* x, const uint16_t* w, float* hist, const float* dyn,
                                     float* y, float* snap, int H, int head_dim, int shared_taps,
                                     float scale, int rounded, int rows, long long snap_stride);
__global__ void tf_ink2_add_rows(const float* x, const float* y, float* z, int H, int rows,
                                 int replace, int rounded);
__global__ void tf_ink2_situ_rows(const float* gu, float* y, int I, const float* gammas,
                                  const unsigned int* row_map, int rows);
__global__ void tf_ink2_route_rows(const float* logits, const float* bias, int* ids, float* weights,
                                   int E, int topk, float scale, int rows);
__global__ void tf_ink2_merge_rows(const float* parts, float* out, int H, int count, int rows,
                                   int round_bf16);
__global__ void tf_ink2_softcap_rows(const float* logits, float* out, int count, int valid,
                                     float cap, int rows);

}

#endif
