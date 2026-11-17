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
// sam21_attn2m.cuh — k_attn2 with MT m16 tiles (16*MT query rows) per warp, the
// FlashAttention-2 warp layout: every K/V fragment pulled from shared feeds MT mmas and
// every staged K/V tile serves NW*16*MT rows, at the price of MT x the accumulator and Q
// registers.  Same contract, same staging and softmax as k_attn2 (P.V in fp16 slices),
// same generic head dimension DH in {56, 64, 72, 96} and the same tail-tile key mask for
// a TK that is not a multiple of BK.  Requires sam21_attn2.cuh (helpers) first.

template<int DH, int NW, int MT, int BK, int STAGES, bool PF16, int MINB = 1>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn2m(
    const half* __restrict__ qkv, const half* __restrict__ Qpool, half* __restrict__ Xout,
    int ws2, int ws2q, int heads, int Cp, int Co, int TQ, int TK) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16 * MT;
  constexpr int DHP = attn2_dhp(DH);
  constexpr int LDS = attn2_lds(DH);
  constexpr int TILE = BK * LDS;
  constexpr int NT = BK / 8, NK = BK / 16, NO = DH / 8;
  constexpr int NKS = DH / 16;
  constexpr bool KTAIL = (DH % 16) == 8;
  constexpr int CHR = DH / 8;
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(DH % 8 == 0 && DH >= 16, "DH must be a multiple of 8");
  static_assert(LDS % 16 == 8, "staged row stride must be 8 mod 16 for conflict-free ldmatrix");

  extern __shared__ __align__(16) char smem_raw[];
  half* Ks = (half*)smem_raw;
  half* Vs = Ks + STAGES * TILE;

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ldk = 3 * Cp;
  const half* K = qkv + (size_t)(w * ws2) * ldk + Cp + h * DHP;
  const int row0 = q0 + warp * 16 * MT;

  uint32_t qa[MT][NKS][4], qt[MT][KTAIL ? 2 : 1];
  {
    const half* Qrow; int ldq;
    if (Qpool) { Qrow = Qpool + (size_t)b * ws2q * DHP; ldq = DHP; }
    else       { Qrow = qkv + (size_t)(w * ws2) * ldk + h * DHP; ldq = ldk; }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const half* p0 = Qrow + (size_t)min(row0 + mt * 16 + g, TQ - 1) * ldq + 2 * q4;
      const half* p1 = Qrow + (size_t)min(row0 + mt * 16 + g + 8, TQ - 1) * ldq + 2 * q4;
#pragma unroll
      for (int kk = 0; kk < NKS; ++kk) {
        qa[mt][kk][0] = ld32(p0 + kk * 16);     qa[mt][kk][1] = ld32(p1 + kk * 16);
        qa[mt][kk][2] = ld32(p0 + kk * 16 + 8); qa[mt][kk][3] = ld32(p1 + kk * 16 + 8);
      }
      if (KTAIL) { qt[mt][0] = ld32(p0 + NKS * 16); qt[mt][1] = ld32(p1 + NKS * 16); }
    }
  }

  constexpr int NCH = BK * CHR, NPT = (NCH + NW * 32 - 1) / (NW * 32);
  const half* gsrc[NPT]; uint32_t sdst[NPT];
#pragma unroll
  for (int j = 0; j < NPT; ++j) {
    int i = tid + j * NW * 32; if (i >= NCH) i = NCH - 1;
    int r = i / CHR, c = i - r * CHR;
    gsrc[j] = K + (size_t)r * ldk + c * 8;
    sdst[j] = smem_u32(Ks) + (uint32_t)(r * LDS + c * 8) * 2;
  }
  const uint32_t vsoff = (uint32_t)(STAGES * TILE) * 2;
  auto load_tile = [&](int stage, int k0) {
    const int koff = k0 * ldk;
    const uint32_t soff = (uint32_t)(stage * TILE) * 2;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
      if (NCH % (NW * 32) == 0 || tid + j * NW * 32 < NCH) {
        cp_async16(sdst[j] + soff, gsrc[j] + koff);
        cp_async16(sdst[j] + soff + vsoff, gsrc[j] + koff + Cp);
      }
    }
  };

  float o[MT][NO][4];
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int n = 0; n < NO; ++n) { o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f; }
    m[mt][0] = m[mt][1] = -1e30f; l[mt][0] = l[mt][1] = 0.f;
  }

  const int ntiles = (TK + BK - 1) / BK;
  const int tmask = (TK % BK) ? (ntiles - 1) : -1;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) load_tile(s, s * BK);
    cp_commit();
  }
  const int mi = lane >> 3, ri = lane & 7;

  for (int t = 0; t < ntiles; ++t) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (t + STAGES - 1 < ntiles) load_tile((t + STAGES - 1) % STAGES, (t + STAGES - 1) * BK);
    cp_commit();
    const half* Kt = Ks + (t % STAGES) * TILE;
    const half* Vt = Vs + (t % STAGES) * TILE;

    // ---- S = Q . K^T for all MT row tiles from one pass over the K fragments ----
    float s[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NT; ++n) { s[mt][n][0] = s[mt][n][1] = s[mt][n][2] = s[mt][n][3] = 0.f; }
#pragma unroll
    for (int n2 = 0; n2 < NT / 2; ++n2) {
      const half* kb = Kt + (16 * n2 + ri + ((mi & 2) ? 8 : 0)) * LDS + ((mi & 1) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int kk = 0; kk < NKS; ++kk) {
        ldsm_x4(bb, smem_u32(kb + kk * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816(s[mt][2 * n2], qa[mt][kk], bb[0], bb[1]);
          mma16816(s[mt][2 * n2 + 1], qa[mt][kk], bb[2], bb[3]);
        }
      }
      if (KTAIL) {
        ldsm_x2(bb, smem_u32(Kt + (16 * n2 + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma1688(s[mt][2 * n2], qt[mt], bb[0]);
          mma1688(s[mt][2 * n2 + 1], qt[mt], bb[1]);
        }
      }
    }

    // ---- tail-tile key mask (see sam21_attn2.cuh) ----
    if (t == tmask) {
      const int kb0 = t * BK + 2 * q4;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (kb0 + n * 8     >= TK) { s[mt][n][0] = -1e30f; s[mt][n][2] = -1e30f; }
          if (kb0 + n * 8 + 1 >= TK) { s[mt][n][1] = -1e30f; s[mt][n][3] = -1e30f; }
        }
    }

    // ---- online softmax per row tile ----
    uint32_t pa[MT][NK][4];
    bool moved = false;
    float c[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        tm0 = fmaxf(tm0, fmaxf(s[mt][n][0], s[mt][n][1]));
        tm1 = fmaxf(tm1, fmaxf(s[mt][n][2], s[mt][n][3]));
      }
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
      const float mn0 = fmaxf(m[mt][0], tm0), mn1 = fmaxf(m[mt][1], tm1);
      c[mt][0] = exp2f((m[mt][0] - mn0) * LOG2E);
      c[mt][1] = exp2f((m[mt][1] - mn1) * LOG2E);
      m[mt][0] = mn0; m[mt][1] = mn1;
      const float ms0 = mn0 * LOG2E, ms1 = mn1 * LOG2E;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        float p0 = exp2f(fmaf(s[mt][n][0], LOG2E, -ms0)), p1 = exp2f(fmaf(s[mt][n][1], LOG2E, -ms0));
        float p2 = exp2f(fmaf(s[mt][n][2], LOG2E, -ms1)), p3 = exp2f(fmaf(s[mt][n][3], LOG2E, -ms1));
        rs0 += p0 + p1; rs1 += p2 + p3;
        pa[mt][n >> 1][(n & 1) * 2]     = pack2(p0, p1);
        pa[mt][n >> 1][(n & 1) * 2 + 1] = pack2(p2, p3);
      }
      l[mt][0] = l[mt][0] * c[mt][0] + rs0; l[mt][1] = l[mt][1] * c[mt][1] + rs1;
      moved |= (c[mt][0] != 1.f) | (c[mt][1] != 1.f);
    }
    if (__any_sync(0xffffffff, moved)) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NO; ++n) {
          o[mt][n][0] *= c[mt][0]; o[mt][n][1] *= c[mt][0];
          o[mt][n][2] *= c[mt][1]; o[mt][n][3] *= c[mt][1];
        }
    }

    // ---- O += P . V, one pass over the V fragments for all MT row tiles ----
    uint32_t oh[PF16 ? MT : 1][PF16 ? NO : 1][2];
    if (PF16) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NO; ++n) { oh[mt][n][0] = oh[mt][n][1] = 0u; }
    }
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
      const half* vb = Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + ((mi & 2) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int n2 = 0; n2 < NKS; ++n2) {
        ldsm_x4_t(bb, smem_u32(vb + n2 * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          if (PF16) {
            mma16816_h(oh[mt][2 * n2], pa[mt][kk], bb[0], bb[1]);
            mma16816_h(oh[mt][2 * n2 + 1], pa[mt][kk], bb[2], bb[3]);
          } else {
            mma16816(o[mt][2 * n2], pa[mt][kk], bb[0], bb[1]);
            mma16816(o[mt][2 * n2 + 1], pa[mt][kk], bb[2], bb[3]);
          }
        }
      }
      if (KTAIL) {
        ldsm_x2_t(bb, smem_u32(Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          if (PF16) mma16816_h(oh[mt][NO - 1], pa[mt][kk], bb[0], bb[1]);
          else      mma16816(o[mt][NO - 1], pa[mt][kk], bb[0], bb[1]);
        }
      }
    }
    if (PF16) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NO; ++n) {
          float2 a = unpack2(oh[mt][n][0]), d = unpack2(oh[mt][n][1]);
          o[mt][n][0] += a.x; o[mt][n][1] += a.y; o[mt][n][2] += d.x; o[mt][n][3] += d.y;
        }
    }
  }

#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    float l0 = l[mt][0], l1 = l[mt][1];
    l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
    const int r0 = row0 + mt * 16 + g, r1 = r0 + 8;
    half* out0 = Xout + (size_t)(w * ws2q + r0) * Co + h * DH + 2 * q4;
    half* out1 = Xout + (size_t)(w * ws2q + r1) * Co + h * DH + 2 * q4;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      if (r0 < TQ) *(uint32_t*)(out0 + 8 * n) = pack2(o[mt][n][0] * inv0, o[mt][n][1] * inv0);
      if (r1 < TQ) *(uint32_t*)(out1 + 8 * n) = pack2(o[mt][n][2] * inv1, o[mt][n][3] * inv1);
    }
  }
#endif  // ATTN2_ENABLED
}
