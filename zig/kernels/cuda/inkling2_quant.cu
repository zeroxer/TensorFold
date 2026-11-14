// TensorFold Inkling2 quantized kernels. Equations: zig/src/families/inkling2/ABI.txt.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace {

// One BF16 round of an fp32 value, returned as float (the activations' designated rounding).
__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// Fixed 256-lane halving tree over shared memory, the inkling2 family's block reduction order.
__device__ __forceinline__ float sum256(float x, float* s) {
    const int t = threadIdx.x;
    s[t] = x;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) s[t] += s[t + k];
        __syncthreads();
    }
    return s[0];
}

__device__ __forceinline__ float max256(float x, float* s) {
    const int t = threadIdx.x;
    s[t] = x;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) s[t] = fmaxf(s[t], s[t + k]);
        __syncthreads();
    }
    return s[0];
}

// Nearest E2M1 magnitude, ties to even code, saturating at six; preserve the sign of zero.
__device__ __forceinline__ unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    unsigned c;
    if (a <= 0.25f) c = 0;
    else if (a < 0.75f) c = 1;
    else if (a <= 1.25f) c = 2;
    else if (a < 1.75f) c = 3;
    else if (a <= 2.5f) c = 4;
    else if (a < 3.5f) c = 5;
    else if (a <= 5.0f) c = 6;
    else c = 7;
    return signbit(v) ? (c | 8u) : c;
}

// An e2m1 code's value: magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6; code 8 is -0.0f.
__device__ __forceinline__ float e2m1_value(unsigned code) {
    float m;
    switch (code & 7u) {
        case 0: m = 0.0f; break;
        case 1: m = 0.5f; break;
        case 2: m = 1.0f; break;
        case 3: m = 1.5f; break;
        case 4: m = 2.0f; break;
        case 5: m = 3.0f; break;
        case 6: m = 4.0f; break;
        default: m = 6.0f; break;
    }
    return (code & 8u) ? -m : m;
}

// An e4m3 byte's fp32 value (e4m3fn: no infinities, 0x7f/0xff are NaN), via the hardware decode.
__device__ __forceinline__ float e4m3_value(unsigned char code) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3)));
}

// fp32 to an e4m3 byte, round to nearest even, saturating (the reference's scale round).
__device__ __forceinline__ unsigned char e4m3_code(float v) {
    return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// The four-over-six candidates' block scale: the capped product rounded through e4m3 and back.
__device__ __forceinline__ float e4m3_round(float v) { return e4m3_value(e4m3_code(v)); }

// UE8M0 scale: 2^(e-127), including subnormal exponent zero; exponent 255 represents NaN.
__device__ __forceinline__ float ue8m0_value(unsigned e) {
    return e == 255 ? NAN : (e == 0 ? 0x1p-127f : __uint_as_float((unsigned)e << 23));
}

// |(decoded * scale) * outer - x|, the reference candidate error's exact rounding order.
__device__ __forceinline__ float recon_err(float x, float d, float s, float o) {
    return fabsf(__fsub_rn(__fmul_rn(__fmul_rn(d, s), o), x));
}

constexpr int QKMAX = 4608;  // widest admitted quantization row (hidden size; w2 uses 2304)

}  // namespace

// Grid N, block256, shared1024B: BF16 dense matvec, FP32 sums, optional BF16 output.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_dense(const float* x, const uint16_t* w, float* y, int N, int K, int round_bf16) {
    __shared__ float s[256];
    const int n = blockIdx.x;
    if (n >= N) return;
    const uint16_t* row = w + (size_t)n * (size_t)K;
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += 256)
        acc = __fadd_rn(acc, __fmul_rn(x[k], __uint_as_float(uint32_t(row[k]) << 16)));
    const float v = sum256(acc, s);
    if (threadIdx.x == 0) y[n] = round_bf16 ? bf(v) : v;
}

// Grid1, block256, shared19456B: full-row activation quantization, K<=4608 and divisible by32.
__device__ void quant_impl(const float* x, float* y, uint16_t* normalized, float* row_amax, int K, int format) {
    __shared__ float row[QKMAX];
    __shared__ float s[256];
    const int t = threadIdx.x;
    bool bad = false;
    for (int k = t; k < K; k += 256) {
        const float v = x[k];
        row[k] = v;
        bad = bad || !isfinite(v);
    }
    if (__syncthreads_or(bad)) {  // every thread takes the same branch
        for (int k = t; k < K; k += 256) { if(y)y[k]=NAN; if(normalized)normalized[k]=__bfloat16_as_ushort(__float2bfloat16_rn(NAN)); }
        if(row_amax && t==0)*row_amax=NAN;
        return;
    }
    if (format == 0) {
        float m = 0.0f;
        for (int k = t; k < K; k += 256) m = fmaxf(m, fabsf(row[k]));
        const float amax = max256(m, s);
        if(row_amax && t==0)*row_amax=amax;
        __syncthreads();
        // per-token global scale: encode with 1536/amax, decode and reconstruct with amax/1536
        const float enc = fminf(__fdiv_rn(1536.0f, amax), FLT_MAX);
        const float dec = __fdiv_rn(1.0f, enc);
        const float outer = __fmul_rn(amax, 1.0f / 1536.0f);
        for (int b = t; b < K / 16; b += 256) {
            const float* xb = row + 16 * b;
            float bmax = fabsf(xb[0]);
#pragma unroll
            for (int j = 1; j < 16; ++j) bmax = fmaxf(bmax, fabsf(xb[j]));
            // the 6.0- and 4.0-based e4m3 block scales and their encode inverses
            const float s6 = e4m3_round(fminf(__fmul_rn(bmax, __fmul_rn(enc, 1.0f / 6.0f)), 448.0f));
            const float s4 = e4m3_round(fminf(__fmul_rn(bmax, __fmul_rn(enc, 0.25f)), 448.0f));
            const float i6 = fminf(__fdiv_rn(1.0f, __fmul_rn(s6, dec)), FLT_MAX);
            const float i4 = fminf(__fdiv_rn(1.0f, __fmul_rn(s4, dec)), FLT_MAX);
            float d6[16], d4[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                d6[j] = e2m1_value(e2m1_code(__fmul_rn(xb[j], i6)));
                d4[j] = e2m1_value(e2m1_code(__fmul_rn(xb[j], i4)));
            }
            // the 16 absolute errors, summed pairwise: 8 pairs, then 4, 2, 1
            float e6[8], e4[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                e6[j] = __fadd_rn(recon_err(xb[2 * j], d6[2 * j], s6, outer),
                                  recon_err(xb[2 * j + 1], d6[2 * j + 1], s6, outer));
                e4[j] = __fadd_rn(recon_err(xb[2 * j], d4[2 * j], s4, outer),
                                  recon_err(xb[2 * j + 1], d4[2 * j + 1], s4, outer));
            }
#pragma unroll
            for (int w = 4; w > 0; w >>= 1)
#pragma unroll
                for (int j = 0; j < w; ++j) {
                    e6[j] = __fadd_rn(e6[2 * j], e6[2 * j + 1]);
                    e4[j] = __fadd_rn(e4[2 * j], e4[2 * j + 1]);
                }
            const bool four = e4[0] < e6[0];  // ties keep the 6.0-based scale
            const float sc = four ? s4 : s6;
#pragma unroll
            for (int j = 0; j < 16; ++j) {
                const float value=__fmul_rn(four ? d4[j] : d6[j],sc);
                if(y)y[16*b+j]=__fmul_rn(value,outer);
                if(normalized)normalized[16*b+j]=__bfloat16_as_ushort(__float2bfloat16_rn(value));
            }
        }
    } else {
        if(row_amax && t==0)*row_amax=1.0f;
        for (int b = t; b < K / 32; b += 256) {
            const float* xb = row + 32 * b;
            float bmax = fabsf(xb[0]);
#pragma unroll
            for (int j = 1; j < 32; ++j) bmax = fmaxf(bmax, fabsf(xb[j]));
            // ceil(amax/448) to a power of two; raw <= 2^-127 keeps the 2^-127 subnormal scale
            const float raw = __fmul_rn(bmax, 1.0f / 448.0f);
            int e = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
            if (raw <= 0x1p-127f) e = 0;
            else if (e > 254) e = 254;
            const float scale = fmaxf(__uint_as_float((unsigned)e << 23), 0x1p-127f);
            const float inv = e == 0 ? 0x1p127f : __fdiv_rn(1.0f, scale);
#pragma unroll
            for (int j = 0; j < 32; ++j) {
                float q = __fmul_rn(xb[j], inv);
                q = fminf(fmaxf(q, -448.0f), 448.0f);  // row is finite, so q is; sat conversion too
                const float value=__fmul_rn(e4m3_value(e4m3_code(q)),scale);
                if(y)y[32*b+j]=value;
                if(normalized)normalized[32*b+j]=__bfloat16_as_ushort(__float2bfloat16_rn(value));
            }
        }
    }
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_quant(const float* x,float* y,int K,int format) { quant_impl(x,y,nullptr,nullptr,K,format); }

// FP4 normalized values (code*E4M3-scale) are exactly BF16; the row's outer scale remains separate.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_quant_norm(const float* x,float* y,uint16_t* normalized,float* amax,int K,int format,int rows) {
    int row=blockIdx.x;if(row>=rows)return;
    quant_impl(x+int64_t(row)*K,y?y+int64_t(row)*K:nullptr,normalized+int64_t(row)*K,amax+row,K,format);
}

// Grid N, block256, shared1024B: one stored expert matvec over reconstructed quantized activations.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_qmv(const float* x, const unsigned char* w, const unsigned char* scales,
                 const float* amax, const float* denom, float* y, int N, int K, int format,
                 int round_bf16) {
    __shared__ float s[256];
    const int n = blockIdx.x;
    if (n >= N) return;
    float acc = 0.0f;
    if (format == 0) {
        const float outer = __fdiv_rn(amax[0], denom[0]);
        const unsigned char* wr = w + (size_t)n * (size_t)(K / 2);
        const unsigned char* sr = scales + (size_t)n * (size_t)(K / 16);
        for (int k = threadIdx.x; k < K; k += 256) {
            const unsigned byte = wr[k >> 1];
            const unsigned code = (k & 1) ? (byte >> 4) : (byte & 0xFu);
            const float sc = e4m3_value(sr[k >> 4]);
            acc = __fadd_rn(acc, __fmul_rn(x[k], __fmul_rn(__fmul_rn(e2m1_value(code), sc), outer)));
        }
    } else {
        const unsigned char* wr = w + (size_t)n * (size_t)K;
        const unsigned char* sr = scales + (size_t)n * (size_t)(K / 32);
        for (int k = threadIdx.x; k < K; k += 256) {
            const float sc = ue8m0_value(sr[k >> 5]);
            acc = __fadd_rn(acc, __fmul_rn(x[k], __fmul_rn(e4m3_value(wr[k]), sc)));
        }
    }
    const float v = sum256(acc, s);
    if (threadIdx.x == 0) y[n] = round_bf16 ? bf(v) : v;
}
