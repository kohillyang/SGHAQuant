// Copyright 2026 SGHAQuant contributors
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
// g/b contain offline D1-folded LN affine coefficients.
// SMEM = false (default): gamma / beta / d2 are read through L1 as float4 -- a lane owns 8 consecutive channels
// per 32-lane round, i.e. two aligned float4 per array -- with no per-CTA shared copy and no barrier (a CTA is
// 8 tokens, so the copy was 3C loads + 3C stores per 8 tokens).  SMEM = true is the previous kernel
// (NANO_DHD_LN_SMEM=1), kept for the A/B and the bit-identity gate; the arithmetic is the same expression.
// NOH (fc2static_engine_20261016, TIMING ONLY, env NANO_T_NOLNH=1): skip both butterfly stages (the in-register
// 8-point FWHT and the cross-lane stages), i.e. the LayerNorm producer without the Hadamard.  Wrong numerics (the
// weights still carry the rotation); used only to attribute the producer's rotation cost.
template<int NV, bool DIAGONALS=true, bool SMEM=false, bool NOH=false>
__global__ void k_ln_dhd4(const half* __restrict__ x, uint8_t* __restrict__ y, float* __restrict__ ys,
                         const float* __restrict__ g, const float* __restrict__ b, int T,
                         const int* __restrict__ src, const float* __restrict__ d2, float clip, float eps = 1e-6f) {
  constexpr int C = NV * 8, NR = (NV + 31) / 32, NB = rot_nb(NV), LDW = kpad4(C) / 8;
  static_assert(NB >= 2 && NB <= 16 && NV % NB == 0, "C = m * 2^k, k >= 4");
  extern __shared__ float gb[];
  float* hd2 = gb + 2*C;
  if constexpr (SMEM) {
    for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; if constexpr(DIAGONALS) hd2[i] = d2[i]; }
    __syncthreads();
  }
  const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  const int lane = threadIdx.x & 31;
  uint32_t* yr = (uint32_t*)(y + (size_t)t * (LDW * 4));
  const int ts = src ? src[t] : t;              // warp-uniform
  if (ts < 0) {
    for (int v = lane; v < LDW; v += 32) yr[v] = 0u;
    if (lane == 0) ys[t] = 1e-12f;
    return;
  }
  const float4* xv = (const float4*)(x + (size_t)ts * C);
  float4 keep[NR];
  float s = 0.f, ss = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      float4 f = xv[v]; keep[m] = f;
      const half2* h = (const half2*)&f;
#pragma unroll
      for (int k = 0; k < 4; ++k) { float2 p = __half22float2(h[k]); s += p.x + p.y; ss += p.x * p.x + p.y * p.y; }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); ss += __shfl_xor_sync(0xffffffff, ss, o); }
  const float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
  float out[NR][8], mx = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      const half2* h = (const half2*)&keep[m];
      float gg[8], bb[8];                       // gamma / beta of this lane's 8 channels
      if constexpr (SMEM) {
#pragma unroll
        for (int k = 0; k < 8; ++k) { gg[k] = gb[v * 8 + k]; bb[k] = gb[C + v * 8 + k]; }
      } else {
        const float4 g0 = ((const float4*)(g + v * 8))[0], g1 = ((const float4*)(g + v * 8))[1];
        const float4 b0 = ((const float4*)(b + v * 8))[0], b1 = ((const float4*)(b + v * 8))[1];
        gg[0] = g0.x; gg[1] = g0.y; gg[2] = g0.z; gg[3] = g0.w; gg[4] = g1.x; gg[5] = g1.y; gg[6] = g1.z; gg[7] = g1.w;
        bb[0] = b0.x; bb[1] = b0.y; bb[2] = b0.z; bb[3] = b0.w; bb[4] = b1.x; bb[5] = b1.y; bb[6] = b1.z; bb[7] = b1.w;
      }
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        out[m][k * 2]     = (p.x - mean) * rs * gg[k * 2] + bb[k * 2];
        out[m][k * 2 + 1] = (p.y - mean) * rs * gg[k * 2 + 1] + bb[k * 2 + 1];
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) out[m][k] = 0.f;
    }
  }
  if constexpr (!NOH) {
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if ((k & st) == 0) { const float a = out[m][k], c2 = out[m][k | st]; out[m][k] = a + c2; out[m][k | st] = a - c2; }
#pragma unroll
  for (int st = 1; st < NB; st <<= 1)
#pragma unroll
    for (int m = 0; m < NR; ++m)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
        out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
      }
  }
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if constexpr (DIAGONALS) {
      if (v < NV) {
        float dd[8];
        if constexpr (SMEM) {
#pragma unroll
          for (int k = 0; k < 8; ++k) dd[k] = hd2[v * 8 + k];
        } else {
          const float4 d0 = ((const float4*)(d2 + v * 8))[0], d1 = ((const float4*)(d2 + v * 8))[1];
          dd[0] = d0.x; dd[1] = d0.y; dd[2] = d0.z; dd[3] = d0.w; dd[4] = d1.x; dd[5] = d1.y; dd[6] = d1.z; dd[7] = d1.w;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) out[m][k] *= dd[k];
      }
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float hn = rsqrtf((float)(NB * 8));       // the Hadamard's 1/sqrt(n), folded into the scale
  const float sc = mx * hn * clip * (1.f / 7.f) + 1e-12f, inv = hn / sc;
  if (lane == 0) ys[t] = sc;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      uint32_t u = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int q = max(-7, min(7, __float2int_rn(out[m][k] * inv)));
        u |= ((uint32_t)q & 0xfu) << (4 * k);
      }
      yr[v] = u;
    } else if (v < LDW) {
      yr[v] = 0u;                                   // the K pad (C = 144: codes 144..159)
    }
  }
}
