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
// sam21_attn2q8.cuh — k_attn2m with QK^T on int8 tensor cores (mma.m16n8k32.s8).
//
// Q and K arrive as int8 with one scale per (token, head), which is the free axis of QK^T:
// the reduction runs over the head dimension, a (token, head) scale is constant along it,
// so S = sq[i] * sk[j] * (Q8 . K8^T) exactly, with no flush.  The producer is the qkv GEMM,
// whose column tile holds whole heads.  V stays fp16 and P.V is the k_attn2m code as is:
// P.V reduces over keys, the free axis of V is its channel, which no qkv tile sees whole.
//
// Layout (what the qkv producer writes):
//   q8:  [T, ld8] int8, head h of Q at column h*DH8, of K at kof8 + h*DH8;
//        DH8 = DH rounded up to 32, pad columns zero
//   s8:  [2*heads][sst] fp32, the Q scale of token t, head h at s8[h*sst + t],
//        the K scale at s8[(heads + h)*sst + t]
//   qkv: the fp16 buffer, of which only V (column 2*Cp + h*DHP) is read
// The softmax scale 1/sqrt(d) is already in Q, as on the fp16 path.
//
// Per score the dequant is one I2F and one FMUL by the key scale; the query scale folds into
// the exp2 argument (sq > 0, so the row max of sk*acc is the row max of the logits / sq).
// Requires sam21_attn2.cuh (helpers) first.

namespace attn2 {
#if ATTN2_ENABLED
__device__ __forceinline__ void mma16832s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
#endif
}  // namespace attn2

static __host__ __device__ constexpr int attn2q8_dh8(int dh) { return (dh + 31) / 32 * 32; }
// staged int8 K row stride in bytes: 16 mod 32 keeps the 8 rows of an ldmatrix in 8 distinct
// 4-bank groups (the int8 analogue of attn2_lds' 8 mod 16 halves)
static __host__ __device__ constexpr int attn2q8_lds8(int dh) { return attn2q8_dh8(dh) + 16; }

template<int DH, int NW, int MT, int BK, int STAGES, int MINB = 1>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn2q8(
    const int8_t* __restrict__ q8, const float* __restrict__ s8, const half* __restrict__ qkv,
    half* __restrict__ Xout, int ws2, int ws2q, int heads, int Cp, int Co, int TQ, int TK,
    int ld8, int kof8, int sst) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16 * MT;
  constexpr int DHP = attn2_dhp(DH);
  constexpr int LDS = attn2_lds(DH);          // staged V row stride, halves
  constexpr int DH8 = attn2q8_dh8(DH);
  constexpr int LDS8 = attn2q8_lds8(DH);      // staged K row stride, bytes
  constexpr int VTILE = BK * LDS;             // halves
  constexpr int KTILE = BK * LDS8;            // bytes
  constexpr int NT = BK / 8, NK = BK / 16, NO = DH / 8;
  constexpr int NKS = DH / 16;                // P.V full 16-column groups
  constexpr int NK8 = DH8 / 32;               // QK^T k32 steps
  constexpr bool KTAIL = (DH % 16) == 8;
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0 && DH % 8 == 0, "shape");
  static_assert(LDS8 % 32 == 16, "int8 row stride");

  extern __shared__ __align__(16) char smem_raw[];
  half* Vs = (half*)smem_raw;                              // [STAGES][BK][LDS]
  int8_t* Ks = (int8_t*)(smem_raw + STAGES * VTILE * 2);   // [STAGES][BK][LDS8]

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ldk = 3 * Cp;
  const int row0 = q0 + warp * 16 * MT;
  const int8_t* K8 = q8 + (size_t)(w * ws2) * ld8 + kof8 + h * DH8;
  const half* V = qkv + (size_t)(w * ws2) * ldk + 2 * Cp + h * DHP;
  const float* sK = s8 + (size_t)(heads + h) * sst + (size_t)w * ws2;

  // ---- Q fragments (mma.m16n8k32 A layout) and the per-row query scale, x LOG2E ----
  uint32_t qa[MT][NK8][4];
  float qs[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    const int ra = min(row0 + mt * 16 + g, TQ - 1), rb = min(row0 + mt * 16 + g + 8, TQ - 1);
    const int8_t* p0 = q8 + (size_t)(w * ws2 + ra) * ld8 + h * DH8 + 4 * q4;
    const int8_t* p1 = q8 + (size_t)(w * ws2 + rb) * ld8 + h * DH8 + 4 * q4;
#pragma unroll
    for (int kk = 0; kk < NK8; ++kk) {
      qa[mt][kk][0] = *(const uint32_t*)(p0 + kk * 32);      qa[mt][kk][1] = *(const uint32_t*)(p1 + kk * 32);
      qa[mt][kk][2] = *(const uint32_t*)(p0 + kk * 32 + 16); qa[mt][kk][3] = *(const uint32_t*)(p1 + kk * 32 + 16);
    }
    qs[mt][0] = s8[(size_t)h * sst + (size_t)w * ws2 + ra] * LOG2E;
    qs[mt][1] = s8[(size_t)h * sst + (size_t)w * ws2 + rb] * LOG2E;
  }

  // ---- K (int8) and V (fp16) tile staging ----
  constexpr int KCH = BK * (DH8 / 16), KPT = (KCH + NW * 32 - 1) / (NW * 32);
  constexpr int VCH = BK * (DH / 8),   VPT = (VCH + NW * 32 - 1) / (NW * 32);
  const int8_t* ksrc[KPT]; uint32_t kdst[KPT];
  const half* vsrc[VPT]; uint32_t vdst[VPT];
#pragma unroll
  for (int j = 0; j < KPT; ++j) {
    int i = tid + j * NW * 32; if (i >= KCH) i = KCH - 1;
    const int r = i / (DH8 / 16), c = i - r * (DH8 / 16);
    ksrc[j] = K8 + (size_t)r * ld8 + c * 16;
    kdst[j] = smem_u32(Ks) + (uint32_t)(r * LDS8 + c * 16);
  }
#pragma unroll
  for (int j = 0; j < VPT; ++j) {
    int i = tid + j * NW * 32; if (i >= VCH) i = VCH - 1;
    const int r = i / (DH / 8), c = i - r * (DH / 8);
    vsrc[j] = V + (size_t)r * ldk + c * 8;
    vdst[j] = smem_u32(Vs) + (uint32_t)(r * LDS + c * 8) * 2;
  }
  auto load_tile = [&](int stage, int k0) {
#pragma unroll
    for (int j = 0; j < KPT; ++j)
      if (KCH % (NW * 32) == 0 || tid + j * NW * 32 < KCH)
        cp_async16(kdst[j] + stage * KTILE, ksrc[j] + (size_t)k0 * ld8);
#pragma unroll
    for (int j = 0; j < VPT; ++j)
      if (VCH % (NW * 32) == 0 || tid + j * NW * 32 < VCH)
        cp_async16(vdst[j] + stage * VTILE * 2, vsrc[j] + (size_t)k0 * ldk);
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
    const int8_t* Kt = Ks + (t % STAGES) * KTILE;
    const half* Vt = Vs + (t % STAGES) * VTILE;

    // key scales of this lane's columns (n*8 + 2*q4, +1); the slack past TK is finite
    float ks[NT][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int kc = t * BK + n * 8 + 2 * q4;
      ks[n][0] = __ldg(sK + kc); ks[n][1] = __ldg(sK + kc + 1);
    }

    // ---- S = Q8 . K8^T (int32), for all MT row tiles from one pass over the K fragments ----
    int32_t si[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NT; ++n) { si[mt][n][0] = si[mt][n][1] = si[mt][n][2] = si[mt][n][3] = 0; }
#pragma unroll
    for (int n2 = 0; n2 < NT / 2; ++n2) {
      const int8_t* kb = Kt + (16 * n2 + ri + ((mi & 2) ? 8 : 0)) * LDS8 + ((mi & 1) ? 16 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int kk = 0; kk < NK8; ++kk) {
        ldsm_x4(bb, smem_u32(kb + kk * 32));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16832s8(si[mt][2 * n2], qa[mt][kk], bb[0], bb[1]);
          mma16832s8(si[mt][2 * n2 + 1], qa[mt][kk], bb[2], bb[3]);
        }
      }
    }
    // dequant by the key scale only: s = sk * acc = logit / sq
    float s[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        s[mt][n][0] = (float)si[mt][n][0] * ks[n][0]; s[mt][n][1] = (float)si[mt][n][1] * ks[n][1];
        s[mt][n][2] = (float)si[mt][n][2] * ks[n][0]; s[mt][n][3] = (float)si[mt][n][3] * ks[n][1];
      }

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

    // ---- online softmax per row tile, in units of logit / sq (m and ms are the true max x LOG2E) ----
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
      // m holds max(logit) * LOG2E; the masked -1e30 stays hugely negative after the scale
      const float mn0 = fmaxf(m[mt][0], tm0 * qs[mt][0]), mn1 = fmaxf(m[mt][1], tm1 * qs[mt][1]);
      c[mt][0] = exp2f(m[mt][0] - mn0);
      c[mt][1] = exp2f(m[mt][1] - mn1);
      m[mt][0] = mn0; m[mt][1] = mn1;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        float p0 = exp2f(fmaf(s[mt][n][0], qs[mt][0], -mn0)), p1 = exp2f(fmaf(s[mt][n][1], qs[mt][0], -mn0));
        float p2 = exp2f(fmaf(s[mt][n][2], qs[mt][1], -mn1)), p3 = exp2f(fmaf(s[mt][n][3], qs[mt][1], -mn1));
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

    // ---- O += P . V in per-tile fp16 slices (k_attn2m, PF16) ----
    uint32_t oh[MT][NO][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NO; ++n) { oh[mt][n][0] = oh[mt][n][1] = 0u; }
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
      const half* vb = Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + ((mi & 2) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int n2 = 0; n2 < NKS; ++n2) {
        ldsm_x4_t(bb, smem_u32(vb + n2 * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          mma16816_h(oh[mt][2 * n2], pa[mt][kk], bb[0], bb[1]);
          mma16816_h(oh[mt][2 * n2 + 1], pa[mt][kk], bb[2], bb[3]);
        }
      }
      if (KTAIL) {
        ldsm_x2_t(bb, smem_u32(Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) mma16816_h(oh[mt][NO - 1], pa[mt][kk], bb[0], bb[1]);
      }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        float2 a = unpack2(oh[mt][n][0]), d = unpack2(oh[mt][n][1]);
        o[mt][n][0] += a.x; o[mt][n][1] += a.y; o[mt][n][2] += d.x; o[mt][n][3] += d.y;
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

template<int DH, int BK, int STAGES>
static constexpr size_t attn2q8_smem() {
  return (size_t)STAGES * BK * (attn2_lds(DH) * 2 + attn2q8_lds8(DH));
}
