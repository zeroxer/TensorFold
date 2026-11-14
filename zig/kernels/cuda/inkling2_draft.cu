// TensorFold Inkling2 DSpark draft kernels; fp32 math via explicit _rn ops, BF16 RNE rounds via bf().
#include "inkling2_draft.cuh"

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int NT = 256;     // threads per block, every kernel
constexpr int NW = NT / 32; // warps per block
constexpr int HMAX = 4608;  // served hidden width of the draft rows
constexpr int KMAX = 1032;  // attn keys: up to 1024 context rows plus the 8 block rows
constexpr int DMAXA = 128;  // attn head dim served
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bw(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ float qnan() { return __int_as_float(0x7fc00000); }

// FP32 block sum: strided local sums, warp xor tree, warp partials in ascending warp order.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float part = lane < NW ? red[lane] : 0.0f;
#pragma unroll
        for (int o = NW/2; o > 0; o >>= 1) part = __fadd_rn(part, __shfl_xor_sync(FULL, part, o));
        if (lane == 0) red[0] = part;
    }
    __syncthreads();
    float s = red[0];
    __syncthreads();
    return s;
}

// NaN-propagating FP32 block max over the same reduction shape as block_sum.
__device__ __forceinline__ float block_max(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(FULL, v, o);
        v = (isnan(v) || isnan(ov)) ? NAN : fmaxf(v, ov);
    }
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) s = (isnan(s) || isnan(red[w])) ? NAN : fmaxf(s, red[w]);
    __syncthreads();
    return s;
}

// One output row of quiet NaNs: bad geometry fails closed, never garbage.
__device__ __forceinline__ void nan_row(float* out, size_t base, int n, int t) {
    for (int c = t; c < n; c += NT) out[base + c] = qnan();
}

} // namespace

// (1) mix+RMS: mixed = sum_taps alpha[t]*features[row,t,h] fp32 in ascending tap order; out = BF16((mixed*inv)*w).
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_mix(const float* __restrict__ features, const float* __restrict__ alpha,
                  const uint16_t* __restrict__ w, float* __restrict__ out, int H, int taps, int rows,
                  float eps) {
    __shared__ float sm[HMAX];
    __shared__ float red[NW];
    const int row = blockIdx.x, t = threadIdx.x;
    if (row >= rows) return;
    const size_t base = static_cast<size_t>(row) * taps * H;
    if (H < 1 || H > HMAX || taps < 1) { nan_row(out, static_cast<size_t>(row) * H, H, t); return; }
    float sq = 0.0f;
    for (int c = t; c < H; c += NT) {
        float m = 0.0f;
        for (int tap = 0; tap < taps; ++tap)
            m = __fmaf_rn(features[base + static_cast<size_t>(tap) * H + c], alpha[tap], m);
        sm[c] = m;
        sq = __fadd_rn(sq, __fmul_rn(m, m));
    }
    const float variance = __fmaf_rn(block_sum(sq, red), 1.0f/static_cast<float>(H), eps);
    float inv; asm("rsqrt.approx.f32 %0, %1;" : "=f"(inv) : "f"(variance));
    for (int c = t; c < H; c += NT)
        out[static_cast<size_t>(row) * H + c] = bf(__fmul_rn(__fmul_rn(sm[c], inv), bw(w[c])));
}

// (2) per-head RMS(w,eps) rounded to BF16 BEFORE the neox rotation (normalize != 0), else raw halves; halves stay F32.
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_rope(float* __restrict__ x, const uint16_t* __restrict__ w, const float* __restrict__ table,
                   int rows, int heads, int D, int first_pos, int normalize, float eps) {
    __shared__ float red[NW];
    const int h = blockIdx.x, row = blockIdx.y, t = threadIdx.x;
    const size_t base = (static_cast<size_t>(row) * heads + h) * D;
    const int half = D / 2;
    if (D < 2 || (D & 1) != 0 || first_pos < 0) { nan_row(x, base, D, t); return; }
    const float* cs = table + static_cast<size_t>(first_pos + row) * D;
    float inv = 1.0f;
    if (normalize != 0) {
        float sq = 0.0f;
        for (int i = t; i < half; i += NT) {
            const float x1 = x[base + i], x2 = x[base + half + i];
            sq = __fadd_rn(sq, __fadd_rn(__fmul_rn(x1, x1), __fmul_rn(x2, x2)));
        }
        const float mean = __fdiv_rn(block_sum(sq, red), static_cast<float>(D));
        inv = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps))); // the reference norm uses precise 1/sqrt
    }
    for (int i = t; i < half; i += NT) { // both old halves of pair i are read before either is written
        float x1 = x[base + i], x2 = x[base + half + i];
        if (normalize != 0) {
            x1 = bf(__fmul_rn(__fmul_rn(x1, inv), bw(w[i])));
            x2 = bf(__fmul_rn(__fmul_rn(x2, inv), bw(w[half + i])));
        }
        const float c = cs[i], s = cs[half + i];
        x[base + i] = bf(__fsub_rn(__fmul_rn(x1, c), __fmul_rn(x2, s)));
        x[base + half + i] = bf(__fadd_rn(__fmul_rn(x2, c), __fmul_rn(x1, s)));
    }
}

// (3) y = BF16((g / (1 + expf(-g))) * u): precise div and exp, one BF16 round at the store; no cap on I.
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_silu(const float* __restrict__ gate, const float* __restrict__ up, float* __restrict__ out,
                   int I, int rows) {
    const int i = blockIdx.x * NT + threadIdx.x;
    if (i >= I) return;
    const size_t at = static_cast<size_t>(blockIdx.y) * I + i;
    const float g = gate[at];
    out[at] = bf(__fmul_rn(__fdiv_rn(g, __fadd_rn(1.0f, expf(-g))), up[at]));
}

// Norm uses the unrounded F32 sum; the saved residual and normalized output each round once to BF16.
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_add_norm(const float* __restrict__ residual, const float* __restrict__ addend,
                       const uint16_t* __restrict__ w, float* __restrict__ new_residual,
                       float* __restrict__ out, int H, int rows, float eps) {
    __shared__ float sm[HMAX];
    __shared__ float red[NW];
    const int row = blockIdx.x, t = threadIdx.x;
    if (row >= rows) return;
    const size_t base = static_cast<size_t>(row) * H;
    if (H < 1 || H > HMAX) {
        nan_row(new_residual, base, H, t);
        nan_row(out, base, H, t);
        return;
    }
    float sq = 0.0f;
    for (int c = t; c < H; c += NT) {
        const float raw = __fadd_rn(residual[base + c], addend[base + c]);
        sm[c] = raw;
        new_residual[base + c] = bf(raw);
        sq = __fadd_rn(sq, __fmul_rn(raw, raw));
    }
    const float mean = __fdiv_rn(block_sum(sq, red), static_cast<float>(H));
    const float inv = rsqrtf(__fadd_rn(mean, eps)); // flashinfer fallback norm: rsqrt of mean + eps
    for (int c = t; c < H; c += NT) out[base + c] = bf(__fmul_rn(__fmul_rn(sm[c], inv), bw(w[c])));
}

// Bidirectional block attention; window counts context keys before the block (reference WINDOW = sliding + block).
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_attn(const float* __restrict__ q, const float* __restrict__ context_k,
                   const float* __restrict__ context_v, const float* __restrict__ block_k,
                   const float* __restrict__ block_v, float* __restrict__ out, int heads, int kv_heads,
                   int D, int block, int anchor, int window) {
    __shared__ float sc[KMAX];
    __shared__ float qs[DMAXA];
    __shared__ float red[NW];
    const int h = blockIdx.x, b = blockIdx.y, t = threadIdx.x;
    const size_t obase = (static_cast<size_t>(b) * heads + h) * D;
    const int total = anchor + block;
    const bool ok = D >= 1 && D <= DMAXA && kv_heads >= 1 && heads >= kv_heads && heads % kv_heads == 0
        && block >= 1 && anchor >= 0 && total <= KMAX;
    if (!ok) { nan_row(out, obase, D, t); return; }
    const int kh = h / (heads / kv_heads);
    const int lo = window > 0 && anchor > window ? anchor - window : 0;
    const float root = sqrtf(static_cast<float>(D));
    if (t < D) qs[t] = q[obase + t];
    __syncthreads();
    float mx = -INFINITY;
    for (int j = lo + t; j < total; j += NT) { // fp32 dot ascending d; thread t owns sc[] slots j-lo == t mod 256
        const float* k = j < anchor
            ? context_k + (static_cast<size_t>(j) * kv_heads + kh) * D
            : block_k + (static_cast<size_t>(j - anchor) * kv_heads + kh) * D;
        float acc = 0.0f;
        for (int d = 0; d < D; ++d) acc = __fadd_rn(acc, __fmul_rn(qs[d], k[d]));
        const float s = __fdiv_rn(acc, root);
        sc[j - lo] = s;
        mx = (isnan(s) || isnan(mx)) ? NAN : fmaxf(mx, s);
    }
    mx = block_max(mx, red);
    float ps = 0.0f;
    for (int j = lo + t; j < total; j += NT) { // p back into the thread-owned sc[] slot
        const float p = expf(__fsub_rn(sc[j - lo], mx));
        sc[j - lo] = p;
        ps = __fadd_rn(ps, p);
    }
    const float den = block_sum(ps, red); // its barrier publishes every sc[] p before the V pass reads them
    if (t < D) {
        float acc = 0.0f;
        for (int j = lo; j < total; ++j) { // weighted V in ascending key order, fp32
            const float* v = j < anchor
                ? context_v + (static_cast<size_t>(j) * kv_heads + kh) * D
                : block_v + (static_cast<size_t>(j - anchor) * kv_heads + kh) * D;
            acc = __fadd_rn(acc, __fmul_rn(sc[j - lo], v[t]));
        }
        out[obase + t] = bf(__fdiv_rn(acc, den));
    }
}

// (6) one recurrent row: g,c,o = BF16(raw); g = BF16(1/(1+expf(-g))); c = BF16(tanhf(c)); latent = BF16(tanhf(o)).
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_markov_gate(const float* __restrict__ raw, const float* __restrict__ old_state,
                    float* __restrict__ new_state, float* __restrict__ latent, int rank) {
    for (int i = threadIdx.x; i < rank; i += NT) {
        const size_t at = i;
        float g = bf(raw[at]);
        const float cd = bf(raw[static_cast<size_t>(rank) + i]);
        const float o = bf(raw[static_cast<size_t>(2) * rank + i]);
        const float st = old_state[at]; // already BF16-valued F32, the reference adds no round here
        g = bf(__fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g))));
        const float cn = bf(tanhf(cd));
        const float keep = bf(__fmul_rn(g, st));
        const float upd = bf(__fmul_rn(bf(__fsub_rn(1.0f, g)), cn));
        new_state[at] = bf(__fadd_rn(keep, upd));
        latent[at] = bf(tanhf(o));
    }
}

// (7) BF16 add, then F32 tanh path: x = BF16(base+bias); out = BF16(cap * tanhf(x / cap)); valid-only tail is -Inf.
extern "C" __global__ void __launch_bounds__(NT)
tf_ink2_draft_softcap(const float* __restrict__ base, const float* __restrict__ bias,
                      float* __restrict__ out, int N, int valid, float cap) {
    const int i = blockIdx.x * NT + threadIdx.x;
    if (i >= N) return;
    if (i >= valid) { out[i] = -INFINITY; return; }
    const float x = bf(__fadd_rn(base[i], bias[i]));
    out[i] = bf(__fmul_rn(tanhf(__fdiv_rn(x, cap)), cap));
}
