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

// sam21_int4v1.cuh -- w4a4_samv1 (--int4 in nano_samv1): the rotated per-token int4 LayerNorm producer with
// nano_samv1's row map.
//
// k_ln_q4s<NV> is k_ln_q4h<NV> of w4a4_hiera/src/sam21_int4h.cuh (= k_ln_q8<NV, 1> of sam21_rot.cuh with the
// 4-bit quantiser of k_ln_q4, sam21_int4.cuh) with the K pad removed: SAM v1's C = 768 / 1024 / 1280 are
// multiples of 32, so the packed row is exactly C / 2 bytes.
//   * output row t normalises input row src[t] (src == nullptr: row t); src[t] < 0 -> zero codes and scale 1e-12.
//     nano_samv1 passes mNatSrcPad for the windowed blocks (rows >= Tv are SAM's zero padding AFTER norm1, whose
//     qkv is then the bias), exactly as the INT8 path's k_ln_q8<NV, 1>.
//   * LayerNorm in fp32, block Hadamard I_{C/n} (x) H_n with n = rot_block(C) = 128 for C = 768 / 1024 / 1280
//     (tools/int8_quant.py kind 'hbp'), per-token scale = absmax(rotated row) / 7 (+1e-12), codes -7..7,
//     element 2j in the low nibble of byte j (sam21_gemm_s8.cuh mma16864_s4).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include "sam21_rot.cuh"     // rot_nb

template<int NV>
__global__ void k_ln_q4s(const half* __restrict__ x, uint8_t* __restrict__ y, float* __restrict__ ys,
                         const float* __restrict__ g, const float* __restrict__ b, int T,
                         const int* __restrict__ src = nullptr, float eps = 1e-6f) {
  constexpr int C = NV * 8, NR = (NV + 31) / 32, NB = rot_nb(NV);
  static_assert(NB >= 2 && NB <= 16 && NV % NB == 0, "C = m * 2^k, k >= 4");
  static_assert(C % 32 == 0, "no K pad in nano_samv1");
  extern __shared__ float gb[];
  for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; }
  __syncthreads();
  const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  const int lane = threadIdx.x & 31;
  uint32_t* yr = (uint32_t*)(y + (size_t)t * (C / 2));
  const int ts = src ? src[t] : t;              // warp-uniform
  if (ts < 0) {
    for (int v = lane; v < NV; v += 32) yr[v] = 0u;
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
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        const int c = v * 8 + k * 2;
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
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float hn = rsqrtf((float)(NB * 8));       // the Hadamard's 1/sqrt(n), folded into the scale
  const float sc = mx * hn * (1.f / 7.f) + 1e-12f, inv = hn / sc;
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
    }
  }
}
