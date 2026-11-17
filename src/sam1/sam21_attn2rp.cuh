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
// sam21_attn2rp.cuh — k_attn2 (sam21_attn2.cuh) with SAM v1's decomposed relative-position bias.
//
// SAM's image encoder (ViTDet / MViTv2 style) adds, before the softmax, to the score of query
// q = (qy, qx) against key k = (ky, kx) of the same window (side x side tokens, row-major):
//     rel_h[q, ky] + rel_w[q, kx],   rel_h[q, ky] = q . Rh[qy - ky + side - 1],
//                                   rel_w[q, kx] = q . Rw[qx - kx + side - 1]
// (Rh, Rw = the learned rel_pos_h / rel_pos_w tables of 2*side-1 rows, q UNSCALED).  The engine
// precomputes, per (query row, head), the FULL products P = Q . [Rh | Rw]^T with one batched GEMM
// (the exporter scales the tables by sqrt(dh), so the pre-scaled Q gives SAM's numbers):
//     rel[(w*ws2 + r) * heads + h][j],  j < L = 2*side-1: Q . Rh[j],   L <= j < 2L: Q . Rw[j - L]
// and this kernel indexes it: bias = rel[qy - ky + side - 1] + rel[L + qx - kx + side - 1].
// Everything else (the qkv layout, the cp.async ring, the register-resident softmax, the tail
// key mask, the token-major output) is k_attn2's, reused through the attn2:: helpers.
//
// SIDE is a template parameter so that key -> (ky, kx) is a multiply-shift.  When BK divides SIDE
// (the 64 x 64 global blocks), ky is constant over a key tile and only kx varies: one rel_h load
// per row per tile plus one rel_w load per score element.  Otherwise (14 x 14 windows of 196 keys)
// every element does both lookups; those windows are 21x cheaper than a global block anyway.
// Warps whose 16 query rows all lie past TQ skip the math (they still stage K/V and hit the
// barriers): a 196-query window launched as 64-query CTAs wastes 60 rows of the last CTA otherwise.
#include "sam21_attn2.cuh"

// MAPPED: qkv and Xout are in NATURAL token order; rmap[w*ws2 + slot] gives the token row of every
// window slot (padding slots point at a zero-input row whose qkv is the bias, as SAM's zero padding
// after norm1 gives; rmap needs >= BK entries of slack past the last window).  Q, K and V are read
// through the map and the output is written straight to the token row (padding slots dropped,
// padrow = that zero row), so neither a slot-order qkv GEMM nor a gather pass is needed.
template<int DH, int SIDE, int NW, int BK, int STAGES, bool QF16, bool PF16, int MINB = 1, bool INK = false,
         bool MAPPED = false>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn2rp(
    const half* __restrict__ qkv, half* __restrict__ Xout, const half* __restrict__ rel, int relN,
    int ws2, int heads, int Cp, int Co, int TQ, int TK, const int* __restrict__ rmap = nullptr, int padrow = -1) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16;
  constexpr int DHP = attn2_dhp(DH);
  constexpr int LDS = attn2_lds(DH);
  constexpr int TILE = BK * LDS;
  constexpr int NT = BK / 8;
  constexpr int NK = BK / 16;
  constexpr int NO = DH / 8;
  constexpr int NKS = DH / 16;
  constexpr bool KTAIL = (DH % 16) == 8;
  constexpr int CHR = DH / 8;
  constexpr int L = 2 * SIDE - 1;
  // ROWT (window INK with 2*SIDE <= BK): every key tile is exactly two window rows (KPT = 2*SIDE keys,
  // the tile's last BK - KPT columns masked), so a lane's columns map to a FIXED (row-in-tile, kx):
  // rel_w and the mask are 8 registers per row, rel_h two shared loads per row per tile.
  constexpr bool ROWT = INK && (SIDE % BK != 0) && (2 * SIDE <= BK);
  constexpr int KPT = ROWT ? 2 * SIDE : BK;           // keys consumed per tile
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(LDS % 16 == 8, "staged row stride must be 8 mod 16 for conflict-free ldmatrix");

  extern __shared__ __align__(16) char smem_raw[];
  half* Ks = (half*)smem_raw;
  half* Vs = Ks + STAGES * TILE;

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ldk = 3 * Cp;
  const half* K = MAPPED ? qkv + Cp + h * DHP : qkv + (size_t)(w * ws2) * ldk + Cp + h * DHP;
  const int* wmap = MAPPED ? rmap + (size_t)w * ws2 : nullptr;
  const int row0 = q0 + warp * 16;
  const bool active = row0 < TQ;                 // warp-uniform

  uint32_t qa[NKS][4], qt[KTAIL ? 2 : 1];
  const int r0c = min(row0 + g, TQ - 1), r1c = min(row0 + g + 8, TQ - 1);
  {
    const half* Qrow = MAPPED ? qkv + h * DHP : qkv + (size_t)(w * ws2) * ldk + h * DHP;
    const half* p0 = Qrow + (size_t)(MAPPED ? wmap[r0c] : r0c) * ldk + 2 * q4;
    const half* p1 = Qrow + (size_t)(MAPPED ? wmap[r1c] : r1c) * ldk + 2 * q4;
#pragma unroll
    for (int kk = 0; kk < NKS; ++kk) {
      qa[kk][0] = ld32(p0 + kk * 16);     qa[kk][1] = ld32(p1 + kk * 16);
      qa[kk][2] = ld32(p0 + kk * 16 + 8); qa[kk][3] = ld32(p1 + kk * 16 + 8);
    }
    if (KTAIL) { qt[0] = ld32(p0 + NKS * 16); qt[1] = ld32(p1 + NKS * 16); }
  }
  // rel rows of this lane's two query rows, pre-offset so that relh(ky) = hb[-ky], relw(kx) = wb[-kx]
  const half* rp0 = rel + ((size_t)(w * ws2 + r0c) * heads + h) * relN;
  const half* rp1 = rel + ((size_t)(w * ws2 + r1c) * heads + h) * relN;
  // ---- INK: the products Q . table computed here on the tensor cores instead of read from a buffer.
  // `rel` is then the TRANSPOSED table [relN][DH] (row j = table column j), so that a B fragment of
  // mma.m16n8k16 (column n = g, rows 2*q4 (+1), 2*q4+8 (+1)) is two 32-bit loads of table row j(n).
  //   window (SIDE % BK != 0): P = Q[16 x DH] . T^T for all relN columns, parked per warp in shared
  //     memory (fp16, [16][relN]) behind the K/V ring and indexed exactly like the buffer path.
  //   global (SIDE % BK == 0, 16 | SIDE so a warp's 16 rows share qy and have qx = qx0 + r):
  //     rel_w: Pw = Q . Tw[L + qx0 + c]^T for c < 80 (10 n8 tiles), staged in the K/V area, and every
  //     lane gathers the 2 rows x 2 key-tile phases x 8 columns it will ever need into 16 half2
  //     registers (rel_w[r][kx] = Pw[r][r - kx + SIDE - 1]);
  //     rel_h: every 8 key rows (8*SIDE/BK tiles) one n8 mma of Q against Th[qy - ky + SIDE - 1] for
  //     those 8 ky, then a quad shuffle per tile hands each row its rel_h[ky].
  half2 rwreg[2][2][NT];                // [row g / g+8][key-tile phase][n] (global INK only)
  float hc[4] = {0.f, 0.f, 0.f, 0.f};  // rel_h of the current 8-ky group (global INK only)
  float rwin[2][NT][2];                 // ROWT: rel_w (or -1e30 = masked column) per [row][n][j]
  if constexpr (INK) {
    if constexpr (SIDE % BK == 0) {
      static_assert(SIDE % 16 == 0 && SIDE / BK == 2 && NT == 4, "global INK path: SIDE = 2*BK, BK = 32");
      half* Pw = (half*)smem_raw + warp * 16 * 80;
      const int qx0 = row0 % SIDE;
#pragma unroll
      for (int nt = 0; nt < 10; ++nt) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const half* tb = rel + (size_t)(L + qx0 + nt * 8 + g) * DH + 2 * q4;
#pragma unroll
        for (int kk = 0; kk < NKS; ++kk) mma16816(c, qa[kk], ld32(tb + kk * 16), ld32(tb + kk * 16 + 8));
        if (KTAIL) mma1688(c, qt, ld32(tb + NKS * 16));
        *(half2*)(Pw + g * 80 + nt * 8 + 2 * q4) = __floats2half2_rn(c[0], c[1]);
        *(half2*)(Pw + (g + 8) * 80 + nt * 8 + 2 * q4) = __floats2half2_rn(c[2], c[3]);
      }
      __syncwarp();
#pragma unroll
      for (int rr = 0; rr < 2; ++rr)
#pragma unroll
        for (int ph = 0; ph < 2; ++ph)
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            const int r = g + 8 * rr, kx = ph * BK + n * 8 + 2 * q4;
            rwreg[rr][ph][n] = __halves2half2(Pw[r * 80 + r - kx + SIDE - 1], Pw[r * 80 + r - kx - 1 + SIDE - 1]);
          }
      __syncthreads();                    // the K/V ring reuses this area
    } else {
      half* Ps = (half*)(smem_raw + (size_t)STAGES * TILE * 2 * sizeof(half)) + warp * 16 * relN;
      for (int nt = 0; nt < relN / 8; ++nt) {
        float c[4] = {0.f, 0.f, 0.f, 0.f};
        const half* tb = rel + (size_t)(nt * 8 + g) * DH + 2 * q4;
#pragma unroll
        for (int kk = 0; kk < NKS; ++kk) mma16816(c, qa[kk], ld32(tb + kk * 16), ld32(tb + kk * 16 + 8));
        if (KTAIL) mma1688(c, qt, ld32(tb + NKS * 16));
        *(half2*)(Ps + g * relN + nt * 8 + 2 * q4) = __floats2half2_rn(c[0], c[1]);
        *(half2*)(Ps + (g + 8) * relN + nt * 8 + 2 * q4) = __floats2half2_rn(c[2], c[3]);
      }
      __syncwarp();
      rp0 = Ps + g * relN; rp1 = Ps + (g + 8) * relN;
      if constexpr (ROWT) {
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
          const int rc = rr ? r1c : r0c, qxr = rc % SIDE;
          const half* pr = rr ? rp1 : rp0;
#pragma unroll
          for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int j = 0; j < 2; ++j) {
              const int c = n * 8 + 2 * q4 + j;
              const int kx = c < SIDE ? c : c - SIDE;
              rwin[rr][n][j] = c < KPT ? __half2float(pr[L + qxr - kx + SIDE - 1]) : -1e30f;
            }
        }
      }
    }
  }
  auto ldr = [](const half* p) -> float { return INK ? __half2float(*p) : __half2float(__ldg(p)); };
  const int qy0 = r0c / SIDE, qx0 = r0c - qy0 * SIDE, qy1 = r1c / SIDE, qx1 = r1c - qy1 * SIDE;
  const half* hb0 = rp0 + qy0 + SIDE - 1;
  const half* hb1 = rp1 + qy1 + SIDE - 1;
  const half* wb0 = rp0 + L + qx0 + SIDE - 1;
  const half* wb1 = rp1 + L + qx1 + SIDE - 1;

  constexpr int NCH = BK * CHR, NPT = (NCH + NW * 32 - 1) / (NW * 32);
  const half* gsrc[NPT]; uint32_t sdst[NPT]; int grow[NPT];
#pragma unroll
  for (int j = 0; j < NPT; ++j) {
    int i = tid + j * NW * 32; if (i >= NCH) i = NCH - 1;
    int r = i / CHR, c = i - r * CHR;
    grow[j] = r;
    gsrc[j] = MAPPED ? K + c * 8 : K + (size_t)r * ldk + c * 8;
    sdst[j] = smem_u32(Ks) + (uint32_t)(r * LDS + c * 8) * 2;
  }
  const uint32_t vsoff = (uint32_t)(STAGES * TILE) * 2;
  auto load_tile = [&](int stage, int k0) {
    const int koff = k0 * ldk;
    const uint32_t soff = (uint32_t)(stage * TILE) * 2;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
      if (NCH % (NW * 32) == 0 || tid + j * NW * 32 < NCH) {
        const half* src = MAPPED ? gsrc[j] + (size_t)__ldg(wmap + k0 + grow[j]) * ldk : gsrc[j] + koff;
        cp_async16(sdst[j] + soff, src);
        cp_async16(sdst[j] + soff + vsoff, src + Cp);
      }
    }
  };

  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  const int ntiles = (TK + KPT - 1) / KPT;
  const int tmask = ROWT ? -1 : ((TK % BK) ? (ntiles - 1) : -1);   // ROWT: the mask lives in rwin
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) load_tile(s, s * KPT);
    cp_commit();
  }
  const int mi = lane >> 3, ri = lane & 7;

  for (int t = 0; t < ntiles; ++t) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (t + STAGES - 1 < ntiles) load_tile((t + STAGES - 1) % STAGES, (t + STAGES - 1) * KPT);
    cp_commit();
    if (!active) continue;
    const half* Kt = Ks + (t % STAGES) * TILE;
    const half* Vt = Vs + (t % STAGES) * TILE;

    float s[NT][4];
    uint32_t sh[QF16 ? NT : 1][2];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (QF16) { sh[n][0] = sh[n][1] = 0u; }
      else { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
    }
#pragma unroll
    for (int n2 = 0; n2 < NT / 2; ++n2) {
      const half* kb = Kt + (16 * n2 + ri + ((mi & 2) ? 8 : 0)) * LDS + ((mi & 1) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int kk = 0; kk < NKS; ++kk) {
        ldsm_x4(bb, smem_u32(kb + kk * 16));
        if (QF16) {
          mma16816_h(sh[2 * n2], qa[kk], bb[0], bb[1]);
          mma16816_h(sh[2 * n2 + 1], qa[kk], bb[2], bb[3]);
        } else {
          mma16816(s[2 * n2], qa[kk], bb[0], bb[1]);
          mma16816(s[2 * n2 + 1], qa[kk], bb[2], bb[3]);
        }
      }
      if (KTAIL) {
        ldsm_x2(bb, smem_u32(Kt + (16 * n2 + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
        if (QF16) {
          mma1688_h(sh[2 * n2], qt, bb[0]);
          mma1688_h(sh[2 * n2 + 1], qt, bb[1]);
        } else {
          mma1688(s[2 * n2], qt, bb[0]);
          mma1688(s[2 * n2 + 1], qt, bb[1]);
        }
      }
    }
    if (QF16) {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        float2 a = unpack2(sh[n][0]), c = unpack2(sh[n][1]);
        s[n][0] = a.x; s[n][1] = a.y; s[n][2] = c.x; s[n][3] = c.y;
      }
    }

    // ---- relative-position bias ----
    if constexpr (ROWT) {
      // keys of tile t: window rows ky = 2t (columns c < SIDE) and 2t+1 (SIDE <= c < 2*SIDE)
      const float h00 = __half2float(hb0[-2 * t]), h01 = __half2float(hb0[-2 * t - 1]);
      const float h10 = __half2float(hb1[-2 * t]), h11 = __half2float(hb1[-2 * t - 1]);
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const bool hi = n * 8 + 2 * q4 + j >= SIDE;
          s[n][j]     += (hi ? h01 : h00) + rwin[0][n][j];
          s[n][2 + j] += (hi ? h11 : h10) + rwin[1][n][j];
        }
    } else if constexpr (INK && SIDE % BK == 0) {
      constexpr int TPG = 8 * SIDE / BK;                   // tiles per 8-ky group
      if (t % TPG == 0) {
        const int qy = row0 / SIDE, ky0 = (t / TPG) * 8;
        const half* tb = rel + (size_t)(qy - ky0 - g + SIDE - 1) * DH + 2 * q4;
        hc[0] = hc[1] = hc[2] = hc[3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < NKS; ++kk) mma16816(hc, qa[kk], ld32(tb + kk * 16), ld32(tb + kk * 16 + 8));
        if (KTAIL) mma1688(hc, qt, ld32(tb + NKS * 16));
      }
      const int kyi = (t / (SIDE / BK)) & 7, src = (lane & ~3) | (kyi >> 1);
      const float bh0 = __shfl_sync(0xffffffff, (kyi & 1) ? hc[1] : hc[0], src);
      const float bh1 = __shfl_sync(0xffffffff, (kyi & 1) ? hc[3] : hc[2], src);
      const bool ph = (t % (SIDE / BK)) != 0;              // key tile starts at kx = BK
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const float2 w0 = __half22float2(ph ? rwreg[0][1][n] : rwreg[0][0][n]);
        const float2 w1 = __half22float2(ph ? rwreg[1][1][n] : rwreg[1][0][n]);
        s[n][0] += bh0 + w0.x; s[n][1] += bh0 + w0.y;
        s[n][2] += bh1 + w1.x; s[n][3] += bh1 + w1.y;
      }
    } else if constexpr (SIDE % BK == 0) {
      const int kyt = (t * BK) / SIDE, kx0 = (t * BK) % SIDE + 2 * q4;
      const float bh0 = ldr(hb0 - kyt), bh1 = ldr(hb1 - kyt);
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int kx = kx0 + n * 8 + j;
          s[n][j]     += bh0 + ldr(wb0 - kx);
          s[n][2 + j] += bh1 + ldr(wb1 - kx);
        }
    } else {
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          const int k = t * BK + n * 8 + 2 * q4 + j;
          if (k < TK) {
            const int ky = k / SIDE, kx = k - ky * SIDE;
            s[n][j]     += ldr(hb0 - ky) + ldr(wb0 - kx);
            s[n][2 + j] += ldr(hb1 - ky) + ldr(wb1 - kx);
          }
        }
    }

    if (t == tmask) {
      const int kb0 = t * BK + 2 * q4;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (kb0 + n * 8     >= TK) { s[n][0] = -1e30f; s[n][2] = -1e30f; }
        if (kb0 + n * 8 + 1 >= TK) { s[n][1] = -1e30f; s[n][3] = -1e30f; }
      }
    }

    float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      tm0 = fmaxf(tm0, fmaxf(s[n][0], s[n][1]));
      tm1 = fmaxf(tm1, fmaxf(s[n][2], s[n][3]));
    }
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
    tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
    tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
    const float mn0 = fmaxf(m0, tm0), mn1 = fmaxf(m1, tm1);
    const float c0 = exp2f((m0 - mn0) * LOG2E), c1 = exp2f((m1 - mn1) * LOG2E);
    m0 = mn0; m1 = mn1;
    const float ms0 = mn0 * LOG2E, ms1 = mn1 * LOG2E;
    float rs0 = 0.f, rs1 = 0.f;
    uint32_t pa[NK][4];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      float p0 = exp2f(fmaf(s[n][0], LOG2E, -ms0)), p1 = exp2f(fmaf(s[n][1], LOG2E, -ms0));
      float p2 = exp2f(fmaf(s[n][2], LOG2E, -ms1)), p3 = exp2f(fmaf(s[n][3], LOG2E, -ms1));
      rs0 += p0 + p1; rs1 += p2 + p3;
      pa[n >> 1][(n & 1) * 2]     = pack2(p0, p1);
      pa[n >> 1][(n & 1) * 2 + 1] = pack2(p2, p3);
    }
    l0 = l0 * c0 + rs0; l1 = l1 * c1 + rs1;
    if (__any_sync(0xffffffff, (c0 != 1.f) | (c1 != 1.f))) {
#pragma unroll
      for (int n = 0; n < NO; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
    }

    uint32_t oh[PF16 ? NO : 1][2];
    if (PF16) {
#pragma unroll
      for (int n = 0; n < NO; ++n) { oh[n][0] = oh[n][1] = 0u; }
    }
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
      const half* vb = Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + ((mi & 2) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int n2 = 0; n2 < NKS; ++n2) {
        ldsm_x4_t(bb, smem_u32(vb + n2 * 16));
        if (PF16) {
          mma16816_h(oh[2 * n2], pa[kk], bb[0], bb[1]);
          mma16816_h(oh[2 * n2 + 1], pa[kk], bb[2], bb[3]);
        } else {
          mma16816(o[2 * n2], pa[kk], bb[0], bb[1]);
          mma16816(o[2 * n2 + 1], pa[kk], bb[2], bb[3]);
        }
      }
      if (KTAIL) {
        ldsm_x2_t(bb, smem_u32(Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
        if (PF16) mma16816_h(oh[NO - 1], pa[kk], bb[0], bb[1]);
        else      mma16816(o[NO - 1], pa[kk], bb[0], bb[1]);
      }
    }
    if (PF16) {
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        float2 a = unpack2(oh[n][0]), c = unpack2(oh[n][1]);
        o[n][0] += a.x; o[n][1] += a.y; o[n][2] += c.x; o[n][3] += c.y;
      }
    }
  }
  if (!active) return;

  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
  const int r0 = row0 + g, r1 = row0 + g + 8;
  const int t0 = MAPPED ? wmap[r0c] : w * ws2 + r0, t1 = MAPPED ? wmap[r1c] : w * ws2 + r1;
  const bool ok0 = r0 < TQ && (!MAPPED || t0 != padrow), ok1 = r1 < TQ && (!MAPPED || t1 != padrow);
  half* out0 = Xout + (size_t)t0 * Co + h * DH + 2 * q4;
  half* out1 = Xout + (size_t)t1 * Co + h * DH + 2 * q4;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    if (ok0) *(uint32_t*)(out0 + 8 * n) = pack2(o[n][0] * inv0, o[n][1] * inv0);
    if (ok1) *(uint32_t*)(out1 + 8 * n) = pack2(o[n][2] * inv1, o[n][3] * inv1);
  }
#endif  // ATTN2_ENABLED
}

// dynamic shared memory of k_attn2rp: the K/V ring, plus (window INK) the per-warp [16][relN] products
template<int DH, int BK, int STAGES>
static size_t attn2rp_smem(bool ink_window, int nw, int relN) {
  return (size_t)STAGES * BK * attn2_lds(DH) * 2 * 2 + (ink_window ? (size_t)nw * 16 * relN * 2 : 0);
}
