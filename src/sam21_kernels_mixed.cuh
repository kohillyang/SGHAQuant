// sam21_kernels.cuh ¡ª custom CUDA kernels for the nano SAM 2.1 image encoder.
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// patch-embed im2col: writes rows already in stage-0 window-major order so the
// following GEMM lands every token where the trunk wants it.
__global__ void k_im2col(const float* __restrict__ img, half* __restrict__ col,
                         const int* __restrict__ w2n, int OH, int IH) {
  int j = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // window-major slot
  if (j >= OH * OH) return;
  int lane = threadIdx.x & 31;
  int n = w2n[j], oy = n / OH, ox = n % OH;
  half* dst = col + (size_t)j * 152;
  for (int t = lane; t < 152; t += 32) {
    float v = 0.f;
    if (t < 147) {
      int c = t / 49, r = t - c * 49, ky = r / 7, kx = r - ky * 7;
      int iy = oy * 4 - 3 + ky, ix = ox * 4 - 3 + kx;
      if ((unsigned)iy < (unsigned)IH && (unsigned)ix < (unsigned)IH)
        v = img[((size_t)c * IH + iy) * IH + ix];
    }
    dst[t] = __float2half(v);
  }
}

// LayerNorm over the channel axis, one warp per token, fp32 statistics.
//  * templated on the vector count so the register cache of the first pass is a
//    compile-time array -- with a runtime trip count it lands in local memory instead,
//    which cost more than the activation traffic itself;
//  * gamma/beta staged in shared once per block rather than read as global scalars.
//  * src (optional): output row t normalises input row src[t]; src[t] < 0 writes a zero row.
//    This is how the variant-generic hiera engine lands tokens in zero-padded window-major
//    order straight out of the LayerNorm (nullptr = identity, the hiera-L / ViT engines).
template<int NV>                       // NV = C/8 vectors of 8 halves per token
__global__ void k_ln_t(const half* __restrict__ x, half* __restrict__ y,
                       const float* __restrict__ g, const float* __restrict__ b, int T,
                       const int* __restrict__ src = nullptr, float eps = 1e-6f) {
  const int C = NV * 8;
  const int NR = (NV + 31) / 32;       // vectors handled per lane
  extern __shared__ float gb[];        // [C] gamma then [C] beta
  for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; }
  __syncthreads();
  int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  int lane = threadIdx.x & 31;
  const int ts = src ? src[t] : t;     // warp-uniform
  if (ts < 0) {
    float4* yz = (float4*)(y + (size_t)t * C);
    for (int v = lane; v < NV; v += 32) yz[v] = make_float4(0.f, 0.f, 0.f, 0.f);
    return;
  }
  const float4* xv = (const float4*)(x + (size_t)ts * C);
  float4 keep[NR];
  float s = 0.f, ss = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      float4 f = xv[v]; keep[m] = f;
      const half2* h = (const half2*)&f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        s += p.x + p.y; ss += p.x * p.x + p.y * p.y;
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
    ss += __shfl_xor_sync(0xffffffff, ss, o);
  }
  float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
  float4* yv = (float4*)(y + (size_t)t * C);
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      const half2* h = (const half2*)&keep[m];
      float4 o; half2* oh = (half2*)&o;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        int c = v * 8 + k * 2;
        oh[k] = __floats2half2_rn((p.x - mean) * rs * gb[c] + gb[C + c],
                                  (p.y - mean) * rs * gb[c + 1] + gb[C + c + 1]);
      }
      yv[v] = o;
    }
  }
}

#include "sam21_rot.cuh"      // k_ln_q8 (plain and rotated), k_rot_w

#include "sam21_attn.cuh"
#include "sam21_attn2.cuh"
#include "sam21_attn2m.cuh"   // MT = 2 variant of k_attn2, selected on sm_89
#include "sam21_gemm.cuh"
#include "sam21_gemm_s8_mixed.cuh"
#include "sam21_gemm70b.cuh"

// 2x2 max-pool inside each window (window-major in and out)
__global__ void k_maxpool(const half* __restrict__ src, half* __restrict__ dst,
                          int nW, int ws, int C) {
  int wsq = ws / 2, ws2 = ws * ws, ws2q = wsq * wsq;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nW * ws2q * (C / 2)) return;
  int c2 = idx % (C / 2), rest = idx / (C / 2);
  int tq = rest % ws2q, w = rest / ws2q;
  int oh = tq / wsq, ow = tq - oh * wsq;
  const half2* base = (const half2*)(src + (size_t)(w * ws2) * C);
  float2 mx = make_float2(-65504.f, -65504.f);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float2 p = __half22float2(base[(size_t)((2 * oh + a) * ws + (2 * ow + c)) * (C / 2) + c2]);
      mx.x = fmaxf(mx.x, p.x); mx.y = fmaxf(mx.y, p.y);
    }
  ((half2*)(dst + (size_t)(w * ws2q + tq) * C))[c2] = __floats2half2_rn(mx.x, mx.y);
}

// generic token gather: dst[i] = src[map[i]]
__global__ void k_gather(const half* __restrict__ src, half* __restrict__ dst,
                         const int* __restrict__ map, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * (C / 8)) return;
  int c = idx % (C / 8), t = idx / (C / 8);
  ((float4*)(dst + (size_t)t * C))[c] = ((const float4*)(src + (size_t)map[t] * C))[c];
}

// window-major [T,C] fp16 -> NCHW fp32
__global__ void k_scatter_nchw(const half* __restrict__ src, float* __restrict__ dst,
                               const int* __restrict__ n2w, int T, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * C) return;
  int c = idx % C, n = idx / C;
  dst[(size_t)c * T + n] = __half2float(src[(size_t)n2w[n] * C + c]);
}

// vision_features = neck.c1(x_s2) + nearest_2x(neck.c0(x_s3)), emitted as NCHW fp32
__global__ void k_vision_feat(const half* __restrict__ lat2, const half* __restrict__ lat3,
                              float* __restrict__ dst, const int* __restrict__ n2w2,
                              const int* __restrict__ n2w3, int H2, int C) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int T2 = H2 * H2;
  if (idx >= T2 * C) return;
  int c = idx % C, n = idx / C;
  int y = n / H2, x = n - y * H2;
  int n3 = (y >> 1) * (H2 >> 1) + (x >> 1);
  dst[(size_t)c * T2 + n] = __half2float(lat2[(size_t)n2w2[n] * C + c]) +
                            __half2float(lat3[(size_t)n2w3[n3] * C + c]);
}

// GELU applied in place to the fc1 output.  tanh form via __expf; the deviation from the
// erf form peaks near 1e-3 absolute, below the fp16 rounding of the tensor it writes.
// One float4 (8 halves) per thread: on V100 that is 20 % faster than two (2.85 vs 3.57 ms
// over the trunk, micro/gelubench.exe) because the shorter thread makes the memory
// pipeline deeper; on Ampere the two are within noise.  Only sm_70 still runs this pass --
// on sm_80+ the GELU lives in the fc1 GEMM epilogue.
__global__ void k_gelu(half* __restrict__ x, int n) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
  if (i >= n) return;
  float4 a = *(float4*)(x + i);
  half2* ha = (half2*)&a;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 p = __half22float2(ha[k]);
    p.x *= __frcp_rn(1.f + __expf(-1.5957691216057308f * (p.x + 0.044715f * p.x * p.x * p.x)));
    p.y *= __frcp_rn(1.f + __expf(-1.5957691216057308f * (p.y + 0.044715f * p.y * p.y * p.y)));
    ha[k] = __floats2half2_rn(p.x, p.y);
  }
  *(float4*)(x + i) = a;
}
