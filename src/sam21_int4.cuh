// sam21_int4.cuh -- W4A4 prototype (--int4): the int4 producers and weight packing for k_gemm_s8<..., I4 = true>.
//
// Packed int4 layout (both GEMM operands, every int4 buffer): row-major [rows, K/2] bytes, element 2j of a row
// in the LOW nibble of byte j and 2j+1 in the high nibble, two's complement codes -7..7 (sam21_gemm_s8.cuh,
// mma16864_s4, for why the s8 kernel's byte machinery is the m16n8k64 fragment).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "sam21_rot.cuh"     // rot_nb

// k_ln_q8<NV, 1> with a 4-bit quantiser: LayerNorm, I_{C/n} (x) H_n block Hadamard (n = rot_block(C)),
// per-token scale = absmax(rotated row) / 7, codes -7..7 packed two per byte into y [T, C/2].
// Same register layout and arithmetic as k_ln_q8<NV, 1> up to the quantiser.
template<int NV>
__global__ void k_ln_q4(const half* __restrict__ x, uint8_t* __restrict__ y,
                        float* __restrict__ ys, const float* __restrict__ g,
                        const float* __restrict__ b, int T, float eps = 1e-6f) {
  const int C = NV * 8;
  const int NR = (NV + 31) / 32;
  constexpr int NB = rot_nb(NV);
  static_assert(NB >= 2 && NB <= 16 && NV % NB == 0, "C = m * 2^k, k >= 4");
  extern __shared__ float gb[];
  for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; }
  __syncthreads();
  int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  int lane = threadIdx.x & 31;
  const float4* xv = (const float4*)(x + (size_t)t * C);
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
  float out[NR][8], mx = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      const half2* h = (const half2*)&keep[m];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        int c = v * 8 + k * 2;
        out[m][k * 2]     = (p.x - mean) * rs * gb[c] + gb[C + c];
        out[m][k * 2 + 1] = (p.y - mean) * rs * gb[c + 1] + gb[C + c + 1];
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) out[m][k] = 0.f;
    }
  }
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if ((k & st) == 0) {
          float a = out[m][k], c2 = out[m][k | st];
          out[m][k] = a + c2; out[m][k | st] = a - c2;
        }
#pragma unroll
  for (int st = 1; st < NB; st <<= 1)
#pragma unroll
    for (int m = 0; m < NR; ++m)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
        out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
      }
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float hn = rsqrtf((float)(NB * 8));        // the Hadamard's 1/sqrt(n), folded into the scale
  const float sc = mx * hn * (1.f / 7.f) + 1e-12f, inv = hn / sc;
  if (lane == 0) ys[t] = sc;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      uint32_t u = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int q = max(-7, min(7, __float2int_rn(out[m][k] * inv)));
        u |= ((uint32_t)q & 0xfu) << (4 * k);
      }
      *(uint32_t*)(y + (size_t)t * (C / 2) + v * 4) = u;
    }
  }
}

// RTN 4-bit weight quantiser (k_quant_w with 7 levels each side): w [K, N] fp16 -> wt [N, K] int8 codes -7..7,
// sw[n] = absmax / 7.  (--int4 without --gptq; timing runs use it, as the INT8 timing runs use RTN int8.)
__global__ void k_quant_w4(const half* __restrict__ w, int8_t* __restrict__ wt, float* __restrict__ sw, int K, int N) {
  const int n = blockIdx.x;
  __shared__ float red[32];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(__half2float(w[(size_t)k * N + n])));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float sc = red[0] * (1.f / 7.f) + 1e-12f, inv = 1.f / sc;
  if (threadIdx.x == 0) sw[n] = sc;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    const int q = __float2int_rn(__half2float(w[(size_t)k * N + n]) * inv);
    wt[(size_t)n * K + k] = (int8_t)max(-7, min(7, q));
  }
}

// int8 codes (-8..7 representable) [R, K] -> packed int4 [R, K/2]; counts codes outside -7..7 into *bad.
__global__ void k_pack_s4(const int8_t* __restrict__ src, uint8_t* __restrict__ dst, size_t npairs,
                          unsigned int* __restrict__ bad) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= npairs) return;
  const int a = src[2 * i], b = src[2 * i + 1];
  if (bad && (a < -7 || a > 7 || b < -7 || b > 7)) atomicAdd(bad, 1u);
  dst[i] = (uint8_t)(((uint32_t)a & 0xfu) | (((uint32_t)b & 0xfu) << 4));
}
