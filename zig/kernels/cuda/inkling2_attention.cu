// TensorFold Inkling2 attention primitives. Details: zig/src/families/inkling2/ABI.txt.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float weight(const uint16_t* w, int i) {
    return __uint_as_float(uint32_t(w[i]) << 16);
}
__device__ float sum256(float x, float* s) {
    int t = threadIdx.x;
    s[t] = x;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) s[t] += s[t+k];
        __syncthreads();
    }
    return s[0];
}
__device__ float max256(float x, float* s) {
    int t = threadIdx.x;
    s[t] = x;
    __syncthreads();
    for (int k = 128; k; k >>= 1) {
        if (t < k) s[t] = (isnan(s[t]) || isnan(s[t+k])) ? NAN : fmaxf(s[t], s[t+k]);
        __syncthreads();
    }
    return s[0];
}
}

// Grid [heads], block 256, shared 1024B. Q/R are independently L2 clipped then globally log-scaled.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_qr(const float* q, const float* r, const uint16_t* qs, const uint16_t* rs,
                float* oq, float* orr, int D, int R, float threshold, float rel_threshold,
                float tau, float eps) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    float x = t < D ? q[h*D+t] * weight(qs, h*D+t) : 0;
    float y = t < R ? r[h*R+t] * weight(rs, h*R+t) : 0;
    float sq = sum256(x*x, scratch);
    float fq = isnan(sq) ? NAN : fminf(1.0f, threshold / (sqrtf(sq) + eps));
    __syncthreads();
    float sr = sum256(y*y, scratch);
    float fr = isnan(sr) ? NAN : fminf(1.0f, rel_threshold / (sqrtf(sr) + eps));
    if (t < D) oq[h*D+t] = bf(bf(x*fq)*tau);
    if (t < R) orr[h*R+t] = bf(bf(y*fr)*tau);
}

// Grid [kv_heads], block 256, shared 1024B. Cache [capacity,kv_heads,D] has absolute positions.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_kv(const float* k, const float* v, float* kc, float* vc,
                int D, int kv_heads, int pos, float threshold, float eps) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    float x = t < D ? k[h*D+t] : 0;
    float y = t < D ? v[h*D+t] : 0;
    float sk = sum256(x*x, scratch);
    float fk = isnan(sk) ? NAN : fminf(1.0f, threshold/(sqrtf(sk)+eps));
    __syncthreads();
    float sv = sum256(y*y, scratch);
    float fv = 1.0f/sqrtf(sv/D+eps);
    if (t < D) {
        int64_t at = (int64_t(pos)*kv_heads+h)*D+t;
        kc[at] = bf(x*fk);
        vc[at] = bf(y*fv);
    }
}

// Grid [ceil(extent/256),heads], block 256, no shared. Relative projection is stored BF16 [R,extent].
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_relative(const float* r, const uint16_t* w, float* profile, int R, int extent) {
    int j = blockIdx.x*256+threadIdx.x, h = blockIdx.y;
    if (j >= extent) return;
    float z = 0;
    for (int d = 0; d < R; ++d) z += r[h*R+d]*weight(w, d*extent+j);
    profile[int64_t(h)*extent+j] = z;
}

// Grid ceil((pos+1)/256) by heads, block256: scores over visible keys with bounded relative bias.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_scores(const float* q, const float* kc, const float* profile, float* scores,
                    int heads, int kv_heads, int D, int extent, int capacity, int pos, int window) {
    int kp = blockIdx.x*256+threadIdx.x, h = blockIdx.y;
    if (kp > pos) return;
    int lo = window > 0 ? max(0, pos-window+1) : 0;
    float z = -INFINITY;
    if (kp >= lo) {
        z = 0;
        int kh = h/(heads/kv_heads);
        for (int d = 0; d < D; ++d) z += q[h*D+d]*kc[(int64_t(kp)*kv_heads+kh)*D+d];
        z /= sqrtf(float(D));
        int dist = pos-kp;
        if (dist < extent) z += profile[int64_t(h)*extent+dist];
    }
    scores[int64_t(h)*capacity+kp] = z;
}

// Grid [heads], block 256, shared 1024B. Sink affects only denominator; output has two BF16 rounds.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_attend(const float* scores, const float* sink, const float* vc, const float* gate,
                    float* out, int heads, int kv_heads, int D, int capacity, int pos, int window) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    int lo = window > 0 ? max(0, pos-window+1) : 0;
    const float* sc = scores+int64_t(h)*capacity;
    float mx = t == 0 ? sink[h] : -INFINITY;
    for (int kp = lo+t; kp <= pos; kp += 256) {
        float z = sc[kp];
        mx = (isnan(mx) || isnan(z)) ? NAN : fmaxf(mx,z);
    }
    mx = max256(mx, scratch);
    __syncthreads();
    float den = t == 0 ? expf(sink[h]-mx) : 0;
    for (int kp = lo+t; kp <= pos; kp += 256) den += expf(sc[kp]-mx);
    den = sum256(den, scratch);
    if (t < D) {
        int kh = h/(heads/kv_heads);
        float z = 0;
        for (int kp = lo; kp <= pos; ++kp)
            z += (expf(sc[kp]-mx)/den)*vc[(int64_t(kp)*kv_heads+kh)*D+t];
        out[h*D+t] = bf(bf(z)*(1.0f/(1.0f+expf(-gate[h*D+t]))));
    }
}

// Windowed variants: row r sees the same keys and values through base+r as a scalar step.
extern "C" __global__ void __launch_bounds__(256)
tf_ink2_qr_rows(const float* q, const float* r, const uint16_t* qs, const uint16_t* rs,
                     float* oq, float* orr, int D, int R, int q_stride, int r_stride,
                     float threshold, float rel_threshold, const float* tau, float eps) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    const size_t row = blockIdx.y;
    const float* qrow = q + row * (size_t)q_stride;
    const float* rrow = r + row * (size_t)r_stride;
    float x = t < D ? qrow[h*D+t] * weight(qs, h*D+t) : 0;
    float y = t < R ? rrow[h*R+t] * weight(rs, h*R+t) : 0;
    float sq = sum256(x*x, scratch);
    float fq = isnan(sq) ? NAN : fminf(1.0f, threshold / (sqrtf(sq) + eps));
    __syncthreads();
    float sr = sum256(y*y, scratch);
    float fr = isnan(sr) ? NAN : fminf(1.0f, rel_threshold / (sqrtf(sr) + eps));
    const float tv = tau[row]; // host-computed per position, identical bits to the one-row caller
    if (t < D) oq[row*(size_t)q_stride + h*D+t] = bf(bf(x*fq)*tv);
    if (t < R) orr[row*(size_t)r_stride + h*R+t] = bf(bf(y*fr)*tv);
}

// Grid [kv_heads, rows]: appends rows k/v at cache positions base..base+rows-1.
extern "C" __global__ void __launch_bounds__(256)
tf_ink2_kv_rows(const float* k, const float* v, float* kc, float* vc,
                     int D, int kv_heads, int base, int rows, float threshold, float eps) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    const size_t row = blockIdx.y;
    const float* krow = k + row * (size_t)(kv_heads*D);
    const float* vrow = v + row * (size_t)(kv_heads*D);
    float x = t < D ? krow[h*D+t] : 0;
    float y = t < D ? vrow[h*D+t] : 0;
    float sk = sum256(x*x, scratch);
    float fk = isnan(sk) ? NAN : fminf(1.0f, threshold/(sqrtf(sk)+eps));
    __syncthreads();
    float sv = sum256(y*y, scratch);
    float fv = 1.0f/sqrtf(sv/D+eps);
    if (t < D) {
        int64_t at = (int64_t(base+row)*kv_heads+h)*D+t;
        kc[at] = bf(x*fk);
        vc[at] = bf(y*fv);
    }
}

// Grid [ceil(extent/256), heads, rows]: per-row relative profile [rows, heads, extent].
extern "C" __global__ void __launch_bounds__(256)
tf_ink2_relative_rows(const float* r, const uint16_t* w, float* profile, int R, int extent,
                           int heads) {
    int j = blockIdx.x*256+threadIdx.x, h = blockIdx.y;
    const size_t row = blockIdx.z;
    if (j >= extent) return;
    const float* rrow = r + row * (size_t)(heads*R);
    float z = 0;
    for (int d = 0; d < R; ++d) z += rrow[h*R+d]*weight(w, d*extent+j);
    profile[row*(size_t)(heads*extent) + int64_t(h)*extent+j] = z;
}

// Grid [ceil((base+rows)/256), heads, rows]: row r scores keys <= base + r, windowed when local.
extern "C" __global__ void __launch_bounds__(256)
tf_ink2_scores_rows(const float* q, const float* kc, const float* profile, float* scores,
                         int heads, int kv_heads, int D, int extent, int capacity, int base,
                         int rows, int window) {
    int kp = blockIdx.x*256+threadIdx.x, h = blockIdx.y;
    const int row = blockIdx.z;
    if (row >= rows) return;
    const int pos = base + row;
    if (kp > pos) return;
    int lo = window > 0 ? max(0, pos-window+1) : 0;
    float z = -INFINITY;
    if (kp >= lo) {
        z = 0;
        int kh = h/(heads/kv_heads);
        const float* qrow = q + (size_t)row*heads*D;
        for (int d = 0; d < D; ++d) z += qrow[h*D+d]*kc[(int64_t(kp)*kv_heads+kh)*D+d];
        z /= sqrtf(float(D));
        int dist = pos-kp;
        if (dist < extent) z += profile[(size_t)row*heads*extent + int64_t(h)*extent+dist];
    }
    scores[(size_t)row*heads*capacity + int64_t(h)*capacity+kp] = z;
}

// Grid [heads, rows]: per-row softmax over base + row + 1 scores with that row's sink and gate.
extern "C" __global__ void __launch_bounds__(256)
tf_ink2_attend_rows(const float* scores, const float* sink, const float* vc, const float* gate,
                         float* out, int heads, int kv_heads, int D, int capacity, int base,
                         int rows, int window) {
    __shared__ float scratch[256];
    int h = blockIdx.x, t = threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows) return;
    const int pos = base + row;
    int lo = window > 0 ? max(0, pos-window+1) : 0;
    const float* sc = scores + (size_t)row*heads*capacity + int64_t(h)*capacity;
    float mx = t == 0 ? sink[row*heads + h] : -INFINITY;
    for (int kp = lo+t; kp <= pos; kp += 256) {
        float z = sc[kp];
        mx = (isnan(mx) || isnan(z)) ? NAN : fmaxf(mx,z);
    }
    mx = max256(mx, scratch);
    __syncthreads();
    float den = t == 0 ? expf(sink[row*heads + h]-mx) : 0;
    for (int kp = lo+t; kp <= pos; kp += 256) den += expf(sc[kp]-mx);
    den = sum256(den, scratch);
    if (t < D) {
        int kh = h/(heads/kv_heads);
        float z = 0;
        for (int kp = lo; kp <= pos; ++kp)
            z += (expf(sc[kp]-mx)/den)*vc[(int64_t(kp)*kv_heads+kh)*D+t];
        const float* grow = gate + (size_t)row*heads*D;
        out[(size_t)row*heads*D + h*D+t] = bf(bf(z)*(1.0f/(1.0f+expf(-grow[h*D+t]))));
    }
}
