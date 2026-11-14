// TensorFold Inkling2 row kernels. Equation and launch notes: zig/src/families/inkling2/ABI.txt.
#include "inkling2_rows.cuh"

#include <cuda_bf16.h>
#include <math.h>

namespace tf_ink2 {

constexpr int NT = 256;    // threads a block, every kernel
constexpr int NW = NT / 32;
constexpr int DMAX = 32;   // attn rows (finished blocks + partial) a call may hold
constexpr int EMAX = 2048; // router experts the single block can score
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float bw(uint16_t h) { return __uint_as_float(uint32_t(h) << 16); }
__device__ __forceinline__ float qnan() { return __int_as_float(0x7fc00000); }

// FP32 block sum: strided local sums, warp xor tree, then warp partials in ascending warp order.
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, o));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float s = red[0];
#pragma unroll
    for (int w = 1; w < NW; ++w) s = __fadd_rn(s, red[w]);
    __syncthreads();
    return s;
}

// Lower expert id wins an exact tie; a total order, so the reduce order cannot change the pick.
__device__ __forceinline__ bool better(float va, int ia, float vb, int ib) {
    return va > vb || (va == vb && ia < ib);
}

} // namespace tf_ink2

// (1) y = BF16(w * (x * rsqrt(mean(x^2) + eps))), w a BF16 row; grid (1), 256 threads, 32 B shared.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_rms(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ y, int H,
            float eps) {
    __shared__ float red[tf_ink2::NW];
    if (H < 1) return;
    float s = 0.0f;
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) s = __fadd_rn(s, __fmul_rn(x[c], x[c]));
    const float mean = __fdiv_rn(tf_ink2::block_sum(s, red), static_cast<float>(H));
    const float r = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT)
        y[c] = tf_ink2::bf(__fmul_rn(tf_ink2::bw(w[c]), __fmul_rn(x[c], r)));
}

// Depth attention over finished blocks and the partial residual, using normalized values.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_attnres(const float* __restrict__ states, const float* __restrict__ partial,
                const float* __restrict__ query, float* __restrict__ y, int H, int depth, int queries,
                float eps) {
    __shared__ float red[tf_ink2::NW];
    __shared__ float rr[tf_ink2::DMAX], lg[tf_ink2::DMAX], pw[tf_ink2::DMAX];
    const int q = blockIdx.x, D = depth + 1;
    if (q >= queries) return;
    if (H < 1 || depth < 0 || D > tf_ink2::DMAX) { // fail closed: a NaN row
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT) y[static_cast<size_t>(q) * H + c] = tf_ink2::qnan();
        return;
    }
    const auto row = [&](int d) { return d < depth ? states + static_cast<size_t>(d) * H : partial; };
    for (int d = 0; d < D; ++d) { // row scales: rsqrt(mean(s^2) + eps), no affine
        const float* s = row(d);
        float acc = 0.0f;
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT) acc = __fadd_rn(acc, __fmul_rn(s[c], s[c]));
        const float sum = tf_ink2::block_sum(acc, red);
        if (threadIdx.x == 0) {
            const float variance=__fmaf_rn(sum,1.0f/static_cast<float>(H),eps);
            float inv;asm("rsqrt.approx.f32 %0, %1;":"=f"(inv):"f"(variance));rr[d]=inv;
        }
        __syncthreads();
    }
    const float* qr = query + static_cast<size_t>(q) * H;
    const float query_scale = static_cast<float>(1.0/sqrt(static_cast<double>(H)));
    for (int d = 0; d < D; ++d) { // logits: (query . (s * r)) / sqrt(H), the dot in fp32
        const float* s = row(d);
        float acc = 0.0f;
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT)
            acc = __fmaf_rn(__fmul_rn(qr[c],query_scale),s[c],acc);
        const float dot = tf_ink2::block_sum(acc, red);
        if (threadIdx.x == 0) lg[d] = __fmul_rn(dot,rr[d]);
        __syncthreads();
    }
    if (threadIdx.x == 0) { // softmax over the depth + 1 rows, d ascending as the reference sums
        float mx = -INFINITY;
        for (int d = 0; d < D; ++d) mx = fmaxf(mx, lg[d]);
        for (int d = 0; d < D; ++d) {
            const float e = expf(__fsub_rn(lg[d], mx));
            pw[d] = __fmul_rn(e,rr[d]);
            lg[d] = e;
        }
        int width=1;while(width<D)width*=2;
        for(int d=D;d<width;++d)lg[d]=0;
        for(int half=width/2;half;half/=2)for(int d=0;d<half;++d)lg[d]=__fadd_rn(lg[d],lg[d+half]);
        rr[0]=__fdiv_rn(1.0f,lg[0]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) { // values are U itself, d ascending
        float acc = 0.0f;
        for (int d = 0; d < D; ++d)
            acc = __fmaf_rn(pw[d],row(d)[c],acc);
        y[static_cast<size_t>(q) * H + c] = __fmul_rn(acc,rr[0]);
    }
}

// Four-tap causal convolution; histories retain raw inputs.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_sconv(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ hist,
              const float* __restrict__ dyn, float* __restrict__ y, int H, int head_dim,
              int shared_taps, float scale, int rounded) {
    if (H < 1) return;
    const bool bad =
        shared_taps != 0 && dyn != nullptr && (head_dim < 1 || head_dim > H || H % head_dim != 0);
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) {
        const float xv[4] = {x[c], hist[c], hist[static_cast<size_t>(H) + c],
                             hist[static_cast<size_t>(2) * H + c]}; // lag 0 is the token itself
        if (bad) {
            y[c] = tf_ink2::qnan(); // invalid geometry: no output, no state change, never garbage
            continue;
        }
        float s = 0.0f;
        if (shared_taps == 0) { // static [H,4] channel major, tap reversed: lag L -> w[c*4 + 3 - L]
            for (int tap = 0; tap < 4; ++tap)
                s = __fmaf_rn(tf_ink2::bw(w[static_cast<size_t>(c) * 4 + tap]), xv[3-tap], s);
        } else { // dynamic: static_w [4,H] lag major plus per-head dyn [H/head_dim,4], both direct lag
            const int h = dyn != nullptr ? c / head_dim : 0;
            for (int lag = 0; lag < 4; ++lag) {
                float tap = tf_ink2::bw(w[static_cast<size_t>(lag) * H + c]);
                if (dyn != nullptr) tap = __fadd_rn(tap, dyn[static_cast<size_t>(h) * 4 + lag]);
                s = lag == 0 ? __fmul_rn(tap, xv[lag]) : __fmaf_rn(tap, xv[lag], s);
            }
        }
        const float value=__fmul_rn(__fadd_rn(s,x[c]),scale);
        y[c] = rounded ? tf_ink2::bf(value) : value;
        hist[static_cast<size_t>(2) * H + c] = xv[2];
        hist[static_cast<size_t>(H) + c] = xv[1];
        hist[c] = xv[0]; // the RAW input, not the rounded output
    }
}

// (4) z = BF16((replace ? 0 : x) + y); grid (ceil(H/256)), 256 threads, no shared.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_add(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ z, int H,
            int replace, int rounded) {
    const int i = blockIdx.x * tf_ink2::NT + threadIdx.x;
    if (i < H) {const float value=__fadd_rn(replace != 0 ? 0.0f : x[i], y[i]);z[i]=rounded?tf_ink2::bf(value):value;}
}

// Interleaved SiTU with routing gamma before BF16 rounding.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_swiglu(const float* __restrict__ gu, float* __restrict__ y, int I, float gamma) {
    const int c = blockIdx.x * tf_ink2::NT + threadIdx.x;
    if (c >= I) return;
    const float g = gu[2*c], u = gu[2*c+1];
    const float gc = __fmul_rn(4.0f, tanhf(__fmul_rn(g, 0.25f))); // / 4 is exact, * 0.25
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-g)));  // sigmoid of the raw gate
    const float uc = __fmul_rn(25.0f, tanhf(__fdiv_rn(u, 25.0f)));
    y[c] = tf_ink2::bf(__fmul_rn(__fmul_rn(__fmul_rn(gc, sg), uc), gamma));
}

namespace tf_ink2 {

// Fail closed for the router: -1 ids and quiet-NaN weights for the bounded prefix, nothing selected.
__device__ __forceinline__ void fail_closed(int* ids, float* weights, int topk, int t) {
    for (int k = t; k < topk && k < 32; k += NT) {
        ids[k] = -1;
        weights[k] = qnan();
    }
}

} // namespace tf_ink2

// Stable top-k by sigmoid+bias, with unbiased renormalized scores.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_route(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
              float* __restrict__ weights, int E, int topk, float scale) {
    __shared__ float sc[tf_ink2::EMAX];
    __shared__ uint8_t used[tf_ink2::EMAX];
    __shared__ float rv[tf_ink2::NW];
    __shared__ int ri[tf_ink2::NW];
    const int t = threadIdx.x, lane = t & 31, warp = t >> 5;
    const bool shape_ok = E >= 1 && E <= tf_ink2::EMAX && topk >= 1 && topk <= 32 && topk <= E;
    if (!shape_ok) {
        tf_ink2::fail_closed(ids, weights, topk, t);
        return;
    }
    float bad = isfinite(scale) ? 0.0f : 1.0f;
    for (int e = t; e < E; e += tf_ink2::NT) {
        const float lg = logits[e], b = bias[e];
        if (!isfinite(lg) || !isfinite(b)) bad = 1.0f;
        sc[e] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-lg))); // the unbiased selection score
        used[e] = 0;
    }
    if (tf_ink2::block_sum(bad, rv) > 0.0f) { // no silent mask: one non-finite input poisons all
        tf_ink2::fail_closed(ids, weights, topk, t);
        return;
    }
    float sum = 0.0f; // picked scores in pick order, as the reference accumulates them
    for (int k = 0; k < topk; ++k) {
        __syncthreads(); // the previous mark of used[] is visible to every scan below
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int e = t; e < E; e += tf_ink2::NT) {
            if (used[e] != 0) continue;
            const float v = __fadd_rn(sc[e], bias[e]);
            if (tf_ink2::better(v, e, bv, bi)) {
                bv = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(tf_ink2::FULL, bv, o);
            const int oi = __shfl_xor_sync(tf_ink2::FULL, bi, o);
            if (tf_ink2::better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            rv[warp] = bv;
            ri[warp] = bi;
        }
        __syncthreads();
        bv = rv[0];
        bi = ri[0];
#pragma unroll
        for (int w = 1; w < tf_ink2::NW; ++w)
            if (tf_ink2::better(rv[w], ri[w], bv, bi)) {
                bv = rv[w];
                bi = ri[w];
            }
        sum = __fadd_rn(sum, sc[bi]);
        if (t == 0) {
            ids[k] = bi;
            weights[k] = sc[bi];
        }
        if (t == 0) used[bi] = 1;
    }
    __syncthreads();
    for (int k = t; k < topk; k += tf_ink2::NT)
        weights[k] = __fdiv_rn(__fmul_rn(scale, weights[k]), __fadd_rn(sum, 1e-20f));
}

// A rank's embedding row, or exact zeros when the token belongs to another rank.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_embed(const uint16_t* table, float* out, int id, int first, int valid, int H) {
    int c = blockIdx.x*256+threadIdx.x;
    if (c < H) out[c] = id >= first && id < first+valid ?
        tf_ink2::bw(table[size_t(id-first)*H+c]) : 0.0f;
}

// Fixed rank/pick order. Partial sums are fp32; round only when the graph calls for a BF16 boundary.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_merge(const float* parts, float* out, int H, int count, int round_bf16) {
    int c = blockIdx.x*256+threadIdx.x;
    if (c >= H) return;
    float sum = 0;
    for (int p = 0; p < count; ++p) sum += parts[size_t(p)*H+c];
    out[c] = round_bf16 ? tf_ink2::bf(sum) : sum;
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_softcap(const float* logits, float* out, int count, int valid, float cap) {
    int i = blockIdx.x*256+threadIdx.x;
    if (i < count) out[i] = i < valid ? cap*tanhf(logits[i]/cap) : -INFINITY;
}

// (7) Windowed rows, one block a row: identical arithmetic to (1) per row; strides are in floats.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_rms_rows(const float* __restrict__ x, long long x_stride, const uint16_t* __restrict__ w,
                 float* __restrict__ y, long long y_stride, int H, int rows, float eps) {
    __shared__ float red[tf_ink2::NW];
    const int row = blockIdx.x;
    if (row >= rows || H < 1) return;
    const float* xr = x + static_cast<size_t>(row) * static_cast<size_t>(x_stride);
    float* yr = y + static_cast<size_t>(row) * static_cast<size_t>(y_stride);
    float s = 0.0f;
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) s = __fadd_rn(s, __fmul_rn(xr[c], xr[c]));
    const float mean = __fdiv_rn(tf_ink2::block_sum(s, red), static_cast<float>(H));
    const float r = __fdiv_rn(1.0f, sqrtf(__fadd_rn(mean, eps)));
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT)
        yr[c] = tf_ink2::bf(__fmul_rn(tf_ink2::bw(w[c]), __fmul_rn(xr[c], r)));
}

// (8) Windowed depth attention, grid (queries * rows): same sums per (row, query) as (2).
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_attnres_rows(const float* __restrict__ states, const float* __restrict__ partial,
                     const float* __restrict__ query, float* __restrict__ y, int H, int depth,
                     int queries, int stack_stride, float eps, int rows) {
    __shared__ float red[tf_ink2::NW];
    __shared__ float rr[tf_ink2::DMAX], lg[tf_ink2::DMAX], pw[tf_ink2::DMAX];
    const int q = blockIdx.x % queries, row = blockIdx.x / queries;
    if (row >= rows) return;
    const int D = depth + 1;
    if (H < 1 || depth < 0 || D > tf_ink2::DMAX) {
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT)
            y[static_cast<size_t>(row) * queries * H + static_cast<size_t>(q) * H + c] = tf_ink2::qnan();
        return;
    }
    const auto row_at = [&](int d) {
        return d < depth ? states + (static_cast<size_t>(row) * stack_stride + d) * H
                         : partial + static_cast<size_t>(row) * H;
    };
    for (int d = 0; d < D; ++d) {
        const float* s = row_at(d);
        float acc = 0.0f;
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT) acc = __fadd_rn(acc, __fmul_rn(s[c], s[c]));
        const float sum = tf_ink2::block_sum(acc, red);
        if (threadIdx.x == 0) {
            const float variance=__fmaf_rn(sum,1.0f/static_cast<float>(H),eps);
            float inv;asm("rsqrt.approx.f32 %0, %1;":"=f"(inv):"f"(variance));rr[d]=inv;
        }
        __syncthreads();
    }
    const float* qr = query + static_cast<size_t>(q) * H;
    const float query_scale = static_cast<float>(1.0/sqrt(static_cast<double>(H)));
    for (int d = 0; d < D; ++d) {
        const float* s = row_at(d);
        float acc = 0.0f;
        for (int c = threadIdx.x; c < H; c += tf_ink2::NT)
            acc = __fmaf_rn(__fmul_rn(qr[c],query_scale),s[c],acc);
        const float dot = tf_ink2::block_sum(acc, red);
        if (threadIdx.x == 0) lg[d] = __fmul_rn(dot,rr[d]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float mx = -INFINITY;
        for (int d = 0; d < D; ++d) mx = fmaxf(mx, lg[d]);
        for (int d = 0; d < D; ++d) {
            const float e = expf(__fsub_rn(lg[d], mx));
            pw[d] = __fmul_rn(e,rr[d]);
            lg[d] = e;
        }
        int width=1;while(width<D)width*=2;
        for(int d=D;d<width;++d)lg[d]=0;
        for(int half=width/2;half;half/=2)for(int d=0;d<half;++d)lg[d]=__fadd_rn(lg[d],lg[d+half]);
        rr[0]=__fdiv_rn(1.0f,lg[0]);
    }
    __syncthreads();
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) {
        float acc = 0.0f;
        for (int d = 0; d < D; ++d)
            acc = __fmaf_rn(pw[d],row_at(d)[c],acc);
        y[(static_cast<size_t>(row) * queries + q) * H + c] = __fmul_rn(acc,rr[0]);
    }
}

// Windowed causal convolution writes each prefix history into the 9-slot snapshot ring, starting at slot 1.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_sconv_window(const float* __restrict__ x, const uint16_t* __restrict__ w, float* __restrict__ hist,
                     const float* __restrict__ dyn, float* __restrict__ y, float* __restrict__ snap,
                     int H, int head_dim, int shared_taps, float scale, int rounded, int rows,
                     long long snap_stride) {
    if (H < 1) return;
    const bool bad =
        shared_taps != 0 && dyn != nullptr && (head_dim < 1 || head_dim > H || H % head_dim != 0);
    const int heads = head_dim > 0 ? H / head_dim : 1;
    for (int c = threadIdx.x; c < H; c += tf_ink2::NT) {
        float l1 = hist[c], l2 = hist[static_cast<size_t>(H) + c],
              l3 = hist[static_cast<size_t>(2) * H + c];
        for (int r = 0; r < rows; ++r) {
            const float xv[4] = {x[static_cast<size_t>(r) * H + c], l1, l2, l3}; // lag 0 is the row itself
            float out;
            if (bad) {
                out = tf_ink2::qnan(); // invalid geometry: no output, no state change
            } else {
                float s = 0.0f;
                if (shared_taps == 0) { // static [H,4] channel major, tap reversed
                    for (int tap = 0; tap < 4; ++tap)
                        s = __fmaf_rn(tf_ink2::bw(w[static_cast<size_t>(c) * 4 + tap]), xv[3-tap], s);
                } else { // dynamic: static_w [4,H] lag major plus per-head dyn rows, both direct lag
                    const int h = dyn != nullptr ? c / head_dim : 0;
                    for (int lag = 0; lag < 4; ++lag) {
                        float tap = tf_ink2::bw(w[static_cast<size_t>(lag) * H + c]);
                        if (dyn != nullptr)
                            tap = __fadd_rn(tap, dyn[(static_cast<size_t>(r) * heads + h) * 4 + lag]);
                        s = lag == 0 ? __fmul_rn(tap, xv[lag]) : __fmaf_rn(tap, xv[lag], s);
                    }
                }
                const float value=__fmul_rn(__fadd_rn(s,xv[0]),scale);
                out = rounded ? tf_ink2::bf(value) : value;
            }
            y[static_cast<size_t>(r) * H + c] = out;
            if (!bad) { // the history keeps the RAW inputs; snapshots r+1 hold the prefix state
                const float in = xv[0];
                l3 = l2; l2 = l1; l1 = in;
                float* sn = snap + static_cast<size_t>(r) * static_cast<size_t>(snap_stride);
                sn[c] = l1;
                sn[static_cast<size_t>(H) + c] = l2;
                sn[static_cast<size_t>(2) * H + c] = l3;
            }
        }
        if (!bad) {
            hist[c] = l1;
            hist[static_cast<size_t>(H) + c] = l2;
            hist[static_cast<size_t>(2) * H + c] = l3;
        }
    }
}

// (10) Windowed residual add over rows contiguous [rows, H]; identical element formula to (4).
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_add_rows(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ z,
                 int H, int rows, int replace, int rounded) {
    const int i = blockIdx.x * tf_ink2::NT + threadIdx.x;
    if (i >= rows * H) return;
    const float value=__fadd_rn(replace != 0 ? 0.0f : x[i], y[i]);
    z[i]=rounded?tf_ink2::bf(value):value;
}

// (11) Windowed SiTU over grouped rows: gamma per grouped row, output scattered to its original row.
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_situ_rows(const float* __restrict__ gu, float* __restrict__ y, int I,
                  const float* __restrict__ gammas, const unsigned int* __restrict__ row_map, int rows) {
    const int i = blockIdx.x * tf_ink2::NT + threadIdx.x;
    if (i >= rows * I) return;
    const int g = i / I, c = i % I;
    const float gate = gu[static_cast<size_t>(g) * 2 * I + 2 * c];
    const float u = gu[static_cast<size_t>(g) * 2 * I + 2 * c + 1];
    const float gc = __fmul_rn(4.0f, tanhf(__fmul_rn(gate, 0.25f))); // / 4 is exact, * 0.25
    const float sg = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-gate)));  // sigmoid of the raw gate
    const float uc = __fmul_rn(25.0f, tanhf(__fdiv_rn(u, 25.0f)));
    y[static_cast<size_t>(row_map[g]) * I + c] =
        tf_ink2::bf(__fmul_rn(__fmul_rn(__fmul_rn(gc, sg), uc), gammas[g]));
}

// (12) Windowed router, one block a row: identical selection and renormalization to (6).
extern "C" __global__ void __launch_bounds__(tf_ink2::NT)
tf_ink2_route_rows(const float* __restrict__ logits, const float* __restrict__ bias, int* __restrict__ ids,
                   float* __restrict__ weights, int E, int topk, float scale, int rows) {
    __shared__ float sc[tf_ink2::EMAX];
    __shared__ uint8_t used[tf_ink2::EMAX];
    __shared__ float rv[tf_ink2::NW];
    __shared__ int ri[tf_ink2::NW];
    const int row = blockIdx.x, t = threadIdx.x, lane = t & 31, warp = t >> 5;
    if (row >= rows) return;
    const float* lg = logits + static_cast<size_t>(row) * E;
    const float* bi = bias;
    int* idr = ids + static_cast<size_t>(row) * topk;
    float* wr = weights + static_cast<size_t>(row) * topk;
    const bool shape_ok = E >= 1 && E <= tf_ink2::EMAX && topk >= 1 && topk <= 32 && topk <= E;
    if (!shape_ok) {
        tf_ink2::fail_closed(idr, wr, topk, t);
        return;
    }
    float bad = isfinite(scale) ? 0.0f : 1.0f;
    for (int e = t; e < E; e += tf_ink2::NT) {
        const float v = lg[e], b = bi[e];
        if (!isfinite(v) || !isfinite(b)) bad = 1.0f;
        sc[e] = __fdiv_rn(1.0f, __fadd_rn(1.0f, expf(-v))); // the unbiased selection score
        used[e] = 0;
    }
    if (tf_ink2::block_sum(bad, rv) > 0.0f) { // no silent mask: one non-finite input poisons all
        tf_ink2::fail_closed(idr, wr, topk, t);
        return;
    }
    float sum = 0.0f; // picked scores in pick order, as the reference accumulates them
    for (int k = 0; k < topk; ++k) {
        __syncthreads(); // the previous mark of used[] is visible to every scan below
        float bv = -INFINITY;
        int bi_e = 0x7fffffff;
        for (int e = t; e < E; e += tf_ink2::NT) {
            if (used[e] != 0) continue;
            const float v = __fadd_rn(sc[e], bi[e]);
            if (tf_ink2::better(v, e, bv, bi_e)) {
                bv = v;
                bi_e = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(tf_ink2::FULL, bv, o);
            const int oi = __shfl_xor_sync(tf_ink2::FULL, bi_e, o);
            if (tf_ink2::better(ov, oi, bv, bi_e)) {
                bv = ov;
                bi_e = oi;
            }
        }
        if (lane == 0) {
            rv[warp] = bv;
            ri[warp] = bi_e;
        }
        __syncthreads();
        bv = rv[0];
        bi_e = ri[0];
#pragma unroll
        for (int w = 1; w < tf_ink2::NW; ++w)
            if (tf_ink2::better(rv[w], ri[w], bv, bi_e)) {
                bv = rv[w];
                bi_e = ri[w];
            }
        sum = __fadd_rn(sum, sc[bi_e]);
        if (t == 0) {
            idr[k] = bi_e;
            wr[k] = sc[bi_e];
        }
        if (t == 0) used[bi_e] = 1;
    }
    __syncthreads();
    for (int k = t; k < topk; k += tf_ink2::NT)
        wr[k] = __fdiv_rn(__fmul_rn(scale, wr[k]), __fadd_rn(sum, 1e-20f));
}

// (13) Windowed merge: parts [rows, count, H], picks summed ascending exactly like tf_ink2_merge.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_merge_rows(const float* parts, float* out, int H, int count, int rows, int round_bf16) {
    int i = blockIdx.x*256+threadIdx.x;
    if (i >= rows*H) return;
    int row = i/H, c = i%H;
    float sum = 0;
    for (int p = 0; p < count; ++p) sum += parts[size_t(row*count+p)*H+c];
    out[size_t(row)*H+c] = round_bf16 ? tf_ink2::bf(sum) : sum;
}

// (14) Windowed softcap over [rows, count]; padded vocabulary columns become -Inf per row.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_softcap_rows(const float* logits, float* out, int count, int valid, float cap, int rows) {
    int i = blockIdx.x*256+threadIdx.x;
    if (i >= rows*count) return;
    int col = i%count;
    out[i] = col < valid ? cap*tanhf(logits[i]/cap) : -INFINITY;
}
