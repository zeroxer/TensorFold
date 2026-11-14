// Layout-neutral conversion and recurrent-head helpers for the shared TensorFold lane operators.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
namespace {
__device__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ float bw(uint16_t x) { return __uint_as_float(uint32_t(x)<<16); }
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_to_bf16(const float* x,uint16_t* y,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
    if(i<n) y[i]=__bfloat16_as_ushort(__float2bfloat16_rn(x[i]));
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_from_bf16(const uint16_t* x,float* y,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
    if(i<n) y[i]=bw(x[i]);
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_round_bf16(float* x,int64_t n) {
    int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;
    if(i<n) x[i]=bf(x[i]);
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_pack_mx_scales(const unsigned char* x,unsigned char* y,int n,int k,int64_t total) {
    int64_t i=int64_t(blockIdx.x)*256+threadIdx.x;if(i>=total)return;
    int kg=k/64,h=i%2,r=(i/2)%64,g=(i/128)%kg;
    int64_t row=(i/(128*kg))*64+r;
    y[i]=row<n?x[row*(k/32)+g*2+h]:127;
}

// Preserve the checkpoint's scaled-dot association: the two row amax values meet after the FP32 dot.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_quant_output(float* x,const float* amax,const float* weight_amax,const float* denominator,
                          int N,int rows,int nvfp4) {
    int i=blockIdx.x*256+threadIdx.x,row=blockIdx.y;
    if(i<N && row<rows){float scale=1;
        if(nvfp4)scale=(amax[row]*weight_amax[0])*(1.0f/(1536.0f*denominator[0]));
        x[int64_t(row)*N+i]=bf(x[int64_t(row)*N+i]*scale);
    }
}

// Each head row is folded in FP64, then stored in BF16 as the checkpoint's draft reference does.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_fold(const uint16_t* head,const uint16_t* projection,const uint16_t* bias,
                  uint16_t* folded,uint16_t* folded_bias,int N,int H,int R) {
    int row=blockIdx.x,col=threadIdx.x;
    if(row>=N || col>=R)return;
    double sum=0;
    for(int k=0;k<H;++k)sum+=double(bw(head[int64_t(row)*H+k]))*double(bw(projection[int64_t(k)*R+col]));
    folded[int64_t(row)*R+col]=__bfloat16_as_ushort(__double2bfloat16(sum));
    if(col==0) {
        double z=0;
        for(int k=0;k<H;++k)z+=double(bw(head[int64_t(row)*H+k]))*double(bw(bias[k]));
        folded_bias[row]=__bfloat16_as_ushort(__double2bfloat16(z));
    }
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_fuse_alpha(const uint16_t* w,float* y,int rows,int taps) {
    int row=blockIdx.x;
    if(row>=rows || threadIdx.x!=0)return;
    float mx=-INFINITY,den=0;
    for(int i=0;i<taps;++i)mx=fmaxf(mx,bw(w[row*taps+i]));
    for(int i=0;i<taps;++i){float v=expf(bw(w[row*taps+i])-mx);y[row*taps+i]=v;den+=v;}
    for(int i=0;i<taps;++i)y[row*taps+i]=bf(__fdiv_rn(y[row*taps+i],den));
}

// Hidden row j+1 and the last committed target-feature mixture form one RNN context row.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_rnn_context_input(const float* hidden,const float* feature,uint16_t* out,int H,int rows) {
    int i=blockIdx.x*256+threadIdx.x,row=blockIdx.y;
    if(i<2*H && row<rows){float v=i<H?hidden[(int64_t(row)+1)*H+i]:feature[i-H];out[int64_t(row)*2*H+i]=__bfloat16_as_ushort(__float2bfloat16_rn(v));}
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_rnn_state_input(const float* state,const uint16_t* embed,const uint32_t* token,uint16_t* out,int R,int vocab) {
    int i=blockIdx.x*256+threadIdx.x;uint32_t id=*token;
    if(i<2*R)out[i]=i<R?__bfloat16_as_ushort(__float2bfloat16_rn(state[i])):(id<uint32_t(vocab)?embed[int64_t(id)*R+i-R]:0);
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_add_bias(const float* x,const uint16_t* bias,float* out,int N,int rows,int rounded) {
    int i=blockIdx.x*256+threadIdx.x,row=blockIdx.y;
    if(i<N && row<rows){float v=x[int64_t(row)*N+i]+bw(bias[i]);out[int64_t(row)*N+i]=rounded?bf(v):v;}
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_add_float(const float* x,const float* y,float* out,int N) {
    int i=blockIdx.x*256+threadIdx.x;if(i<N)out[i]=x[i]+y[i];
}

// The rank's best score and global id; a deterministic lower-id tie break, excluding padding and NaNs.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_pick(const float* x,uint32_t* out,int valid,int first) {
    __shared__ float score[256];__shared__ uint32_t id[256];
    int t=threadIdx.x;float best=-INFINITY;uint32_t bi=0xffffffffu;
    for(int i=t;i<valid;i+=256)if(!isnan(x[i]) && (x[i]>best || (x[i]==best && uint32_t(i+first)<bi))){best=x[i];bi=i+first;}
    score[t]=best;id[t]=bi;__syncthreads();
    for(int k=128;k;k/=2){if(t<k && (score[t+k]>score[t] || (score[t+k]==score[t] && id[t+k]<id[t]))){score[t]=score[t+k];id[t]=id[t+k];}__syncthreads();}
    if(t==0){out[0]=__float_as_uint(score[0]);out[1]=id[0];}
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_pick_merge(const uint32_t* picks,uint32_t* out,int world,int index) {
    if(threadIdx.x || blockIdx.x)return;
    float best=-INFINITY;uint32_t bi=0xffffffffu;
    for(int r=0;r<world;++r){float v=__uint_as_float(picks[2*r]);uint32_t id=picks[2*r+1];if(v>best || (v==best && id<bi)){best=v;bi=id;}}
    out[index]=bi;
}

// Strided row-major source to packed BF16 rows: y[row*k + c] = bf16(x[row*src_stride + c]).
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_to_bf16_strided(const float* x, uint16_t* y, int64_t src_stride, int k, int64_t rows) {
    int64_t i = int64_t(blockIdx.x)*256+threadIdx.x;
    if (i >= rows*k) return;
    int64_t row = i/k;
    y[i] = __bfloat16_as_ushort(__float2bfloat16_rn(x[row*src_stride + (i%k)]));
}

// Row gather for grouped expert GEMMs: dst[i] = src[map[i/k]*k + i%k], 16- and 32-bit elements.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_gather_u16(const uint16_t* src, uint16_t* dst, const unsigned int* map, int k,
                        int64_t rows) {
    int64_t i = int64_t(blockIdx.x)*256+threadIdx.x;
    if (i >= rows*k) return;
    dst[i] = src[int64_t(map[i/k])*k + (i%k)];
}

extern "C" __global__ __launch_bounds__(256)
void tf_ink2_gather_f32(const float* src, float* dst, const unsigned int* map, int k,
                       int64_t rows) {
    int64_t i = int64_t(blockIdx.x)*256+threadIdx.x;
    if (i >= rows*k) return;
    dst[i] = src[int64_t(map[i/k])*k + (i%k)];
}

// Scatter grouped expert outputs back to (row, pick) slots: dst[(row*8 + pick)*k + c] = src[i].
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_scatter_f32(const float* src, float* dst, const unsigned int* row_map,
                              const unsigned int* pick_map, int picks, int k, int64_t rows) {
    int64_t i = int64_t(blockIdx.x)*256+threadIdx.x;
    if (i >= rows*k) return;
    int64_t g = i/k;
    dst[int64_t(row_map[g]*picks + pick_map[g])*k + (i%k)] = src[i];
}

// One merge over the gathered pick pairs laid out [rank][row][2]; thread t settles row t.
extern "C" __global__ __launch_bounds__(256)
void tf_ink2_pick_merge_rows(const uint32_t* picks, uint32_t* out, int world, int rows) {
    int t = threadIdx.x;
    if (blockIdx.x || t >= rows) return;
    float best = -INFINITY;
    uint32_t bi = 0xffffffffu;
    for (int r = 0; r < world; ++r) {
        float v = __uint_as_float(picks[2*rows*r + 2*t]);
        uint32_t id = picks[2*rows*r + 2*t + 1];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    out[t] = bi;
}
