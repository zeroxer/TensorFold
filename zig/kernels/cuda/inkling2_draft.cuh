// TensorFold Inkling2 DSpark draft CUDA ABI. Launch notes: zig/src/families/inkling2/ABI.txt.
#pragma once

#if defined(__CUDACC__)

#include <stdint.h>

extern "C" {

// Grid (rows), block 256, shared 18.1 KB; features/out BF16-valued F32, w BF16, alpha F32. Shapes in ABI.txt.
__global__ void tf_ink2_draft_mix(const float* features, const float* alpha, const uint16_t* w, float* out,
                                  int H, int taps, int rows, float eps);

// Grid (heads, rows), block 256, shared 32 B. In-place x[rows,heads,D]; F32 table has NeoX cos/sin halves.
__global__ void tf_ink2_draft_rope(float* x, const uint16_t* w, const float* table, int rows, int heads,
                                   int D, int first_pos, int normalize, float eps);

// Grid (ceil(I/256), rows), block 256, no shared. gate/up[rows,I]; y = BF16((g / (1 + expf(-g))) * u) in F32.
__global__ void tf_ink2_draft_silu(const float* gate, const float* up, float* out, int I, int rows);

// Grid (rows), block 256, shared 18.1 KB. Norm reads the unrounded F32 sum; pointers must be pairwise distinct.
__global__ void tf_ink2_draft_add_norm(const float* residual, const float* addend, const uint16_t* w,
                                       float* new_residual, float* out, int H, int rows, float eps);

// Grid (heads, block), block 256, shared 4.7 KB. Sliding-window context and the whole bidirectional block.
__global__ void tf_ink2_draft_attn(const float* q, const float* context_k, const float* context_v,
                                   const float* block_k, const float* block_v, float* out, int heads,
                                   int kv_heads, int D, int block, int anchor, int window);

// Grid (1), block 256, no shared. raw[3,rank] F32; each recurrent op BF16-rounds as in the reference.
__global__ void tf_ink2_markov_gate(const float* raw, const float* old_state, float* new_state,
                                    float* latent, int rank);

// Grid (ceil(N/256), 1), block 256, no shared. out = BF16(cap * tanhf(BF16(base + bias) / cap)); n >= valid is -Inf.
__global__ void tf_ink2_draft_softcap(const float* base, const float* bias, float* out, int N, int valid,
                                      float cap);

}

#endif
