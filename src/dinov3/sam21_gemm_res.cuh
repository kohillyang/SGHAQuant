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
// sam21_gemm_res.cuh -- fp16 GEMM with a fused bias + residual epilogue for sm_80+:
//
//   D[M,N] = A[M,K] @ B[K,N] + bias[N] + R[M,N]      A, B, R, D row-major fp16, bias fp16 (R may alias D)
//
// This is k_gemm_gelu (sam21_gemm.cuh: cp.async ring, XOR-swizzled or padded shared tiles, ldmatrix,
// mma.m16n8k16 with fp16 or fp32 accumulation, smem-staged coalesced stores) with the GELU replaced by
// the residual add: the epilogue cuBLASLt runs for the attention-output projection (beta = 1 + BIAS).
// It exists for sm_120 (RTX 5090, 170 SMs), where cuBLASLt's fp16-accumulate kernels for the small
// proj shapes (4096 x 576 x 576: one wave of 128-row tiles) measured 25 % slower than on the 4090;
// the sm_89 / sm_86 paths keep cuBLASLt.  Same accumulation type as the cuBLASLt plan it replaces
// (CUBLAS_COMPUTE_16F under --f16acc), bias and residual added in fp32 before the single fp16 rounding.
// The residual tile is read in the fragment loop, before the barrier that precedes the stores, so an
// in-place D == R is safe (every element belongs to exactly one CTA).
// Requires M % BM == 0, K % 8 == 0, N % 8 == 0.
#include "sam21_gemm.cuh"

template<int BM, int BN, int BK, int WM, int WN, int STAGES, bool ACC16, bool SWZ = false, bool DB = true, int MINB = 1>
__global__ __launch_bounds__(WM * WN * 32, MINB) void k_gemm_res(
    const half* __restrict__ A, const half* __restrict__ B, const half* __restrict__ bias,
    const half* R, half* D, int M, int N, int K) {
#if GEMMK_ENABLED
  using namespace gemmk;
  constexpr int NT = WM * WN * 32;
  constexpr int LDA = SWZ ? BK : BK + 8, LDB = SWZ ? BN : BN + 8;
  constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MI = TM / 16, NI = TN / 8;
  constexpr int AKC = BK / 8, BNC = BN / 8;
  constexpr int ARPL = 64 / BK;
  static_assert(BK % 16 == 0 && TM % 16 == 0 && TN % 16 == 0, "tile shape");
  static_assert(!SWZ || (BNC >= 8 && BK <= 64), "swizzle needs >= 8 chunks per B row");
  auto swzA = [](int r, int c) { return SWZ ? (c ^ ((r / ARPL) & (AKC - 1))) : c; };
  auto swzB = [](int r, int c) { return SWZ ? (c ^ (r & 7)) : c; };

  extern __shared__ __align__(16) char smem_raw[];
  half* As = (half*)smem_raw;
  half* Bs = As + STAGES * A_STAGE;

  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp - wm * WN;
  const int g = lane >> 2, q4 = lane & 3;

  constexpr int ACH = BM * AKC, BCH = BK * BNC;
  constexpr int APT = (ACH + NT - 1) / NT, BPT = (BCH + NT - 1) / NT;
  const half* asrc[APT]; uint32_t adst[APT]; int akc[APT];
  const half* bsrc[BPT]; uint32_t bdst[BPT]; int bkr[BPT]; bool bok[BPT];
#pragma unroll
  for (int j = 0; j < APT; ++j) {
    int i = tid + j * NT; if (i >= ACH) i = ACH - 1;
    int r = i / AKC, c = i - r * AKC;
    asrc[j] = A + (size_t)(m0 + r) * K + c * 8;
    adst[j] = smem_u32(As + r * LDA + swzA(r, c) * 8);
    akc[j] = c * 8;
  }
#pragma unroll
  for (int j = 0; j < BPT; ++j) {
    int i = tid + j * NT; if (i >= BCH) i = BCH - 1;
    int r = i / BNC, c = i - r * BNC;
    int n = n0 + c * 8;
    bok[j] = n < N;
    bsrc[j] = B + (size_t)r * N + (bok[j] ? n : 0);
    bdst[j] = smem_u32(Bs + r * LDB + swzB(r, c) * 8);
    bkr[j] = r;
  }
  auto load_tile = [&](int stage, int kt) {
    const int k0 = kt * BK;
    const uint32_t ao = (uint32_t)(stage * A_STAGE) * 2, bo = (uint32_t)(stage * B_STAGE) * 2;
#pragma unroll
    for (int j = 0; j < APT; ++j)
      if (ACH % NT == 0 || tid + j * NT < ACH) {
        bool ok = k0 + akc[j] < K;
        cp_async16z(adst[j] + ao, asrc[j] + (ok ? k0 : 0), ok ? 16 : 0);
      }
#pragma unroll
    for (int j = 0; j < BPT; ++j)
      if (BCH % NT == 0 || tid + j * NT < BCH) {
        bool ok = bok[j] && (k0 + bkr[j] < K);
        cp_async16z(bdst[j] + bo, bsrc[j] + (ok ? (size_t)k0 * N : 0), ok ? 16 : 0);
      }
  };

  float acc[MI][NI][4];
  uint32_t acch[ACC16 ? MI : 1][ACC16 ? NI : 1][2];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      if (ACC16) { acch[mi][ni][0] = acch[mi][ni][1] = 0u; }
      else { acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0.f; }
    }

  const int ktiles = (K + BK - 1) / BK;
  constexpr int KS = BK / 16;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ktiles) load_tile(s, s);
    cp_commit();
  }
  const int lr = (lane & 7) + ((lane >> 3) & 1) * 8, lhi = lane >> 4;
  const uint32_t a_row = smem_u32(As) + (uint32_t)((wm * TM + lr) * LDA) * 2;
  const uint32_t b_row = smem_u32(Bs) + (uint32_t)(lr * LDB) * 2;
  uint32_t aoff[BK / 16], boff[NI / 2];
#pragma unroll
  for (int ks = 0; ks < BK / 16; ++ks) aoff[ks] = (uint32_t)swzA(wm * TM + lr, ks * 2 + lhi) * 16;
#pragma unroll
  for (int n2 = 0; n2 < NI / 2; ++n2) boff[n2] = (uint32_t)swzB(lr, wn * TN / 8 + n2 * 2 + lhi) * 16;

  constexpr int NB = DB ? 2 : 1;
  uint32_t af[NB][MI][4], bf[NB][NI][2];
  auto load_frags = [&](int buf, int stage, int ks) {
    const uint32_t at = a_row + (uint32_t)(stage * A_STAGE) * 2 + aoff[ks];
    const uint32_t bt = b_row + (uint32_t)(stage * B_STAGE + ks * 16 * LDB) * 2;
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) ldsm_x4(af[buf][mi], at + (uint32_t)(mi * 16 * LDA) * 2);
#pragma unroll
    for (int n2 = 0; n2 < NI / 2; ++n2) {
      uint32_t t[4];
      ldsm_x4_t(t, bt + boff[n2]);
      bf[buf][2 * n2][0] = t[0]; bf[buf][2 * n2][1] = t[1];
      bf[buf][2 * n2 + 1][0] = t[2]; bf[buf][2 * n2 + 1][1] = t[3];
    }
  };
  cp_wait<STAGES - 2>();
  __syncthreads();
  if (STAGES - 1 < ktiles) load_tile(STAGES - 1, STAGES - 1);
  cp_commit();
  if (DB) load_frags(0, 0, 0);
  for (int kt = 0; kt < ktiles; ++kt) {
    const int st = kt % STAGES;
#pragma unroll
    for (int ks = 0; ks < KS; ++ks) {
      const int cur = DB ? (ks & 1) : 0;
      if (!DB) load_frags(0, st, ks);
      if (ks == KS - 1) {
        if (kt + 1 < ktiles) {
          cp_wait<STAGES - 2>();
          __syncthreads();
          if (kt + STAGES < ktiles) load_tile(st, kt + STAGES);
          cp_commit();
          if (DB) load_frags(cur ^ 1, (kt + 1) % STAGES, 0);
        }
      } else if (DB) {
        load_frags(cur ^ 1, st, ks + 1);
      }
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          if (ACC16) mma16816_h(acch[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
          else       mma16816(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
        }
    }
  }

  // ---- epilogue: + bias + residual (fp32), stage the fp16 tile in shared, 16-byte coalesced stores ----
  constexpr int LDC = SWZ ? BN : BN + 8;
  static_assert((size_t)BM * LDC * 2 <= (size_t)STAGES * (A_STAGE + B_STAGE) * 2, "C tile must fit the ring");
  auto swzC = [](int r, int c) { return SWZ ? (c ^ (r & 7)) : c; };
  half* Cs = (half*)smem_raw;
  __syncthreads();
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) {
    const int cch = wn * TN / 8 + ni;
    const int coll = swzC(g, cch) * 8 + 2 * q4;
    const int col = n0 + cch * 8 + 2 * q4;
    const bool okc = col < N;
    const float2 bb = okc ? __half22float2(*(const half2*)(bias + col)) : make_float2(0.f, 0.f);
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
      float v0, v1, v2, v3;
      if (ACC16) {
        float2 a = unpack2(acch[mi][ni][0]), c = unpack2(acch[mi][ni][1]);
        v0 = a.x; v1 = a.y; v2 = c.x; v3 = c.y;
      } else {
        v0 = acc[mi][ni][0]; v1 = acc[mi][ni][1]; v2 = acc[mi][ni][2]; v3 = acc[mi][ni][3];
      }
      const int rowl = wm * TM + mi * 16 + g;
      v0 += bb.x; v1 += bb.y; v2 += bb.x; v3 += bb.y;
      if (R != nullptr && okc) {
        const float2 r0 = __half22float2(*(const half2*)(R + (size_t)(m0 + rowl) * N + col));
        const float2 r1 = __half22float2(*(const half2*)(R + (size_t)(m0 + rowl + 8) * N + col));
        v0 += r0.x; v1 += r0.y; v2 += r1.x; v3 += r1.y;
      }
      *(uint32_t*)(Cs + rowl * LDC + coll) = pack2(v0, v1);
      *(uint32_t*)(Cs + (rowl + 8) * LDC + coll) = pack2(v2, v3);
    }
  }
  __syncthreads();
  constexpr int CCH = BM * (BN / 8);
#pragma unroll 4
  for (int i = tid; i < CCH; i += NT) {
    const int r = i / (BN / 8), c = i - r * (BN / 8);
    const int col = n0 + c * 8;
    if (col < N)
      *(uint4*)(D + (size_t)(m0 + r) * N + col) = *(const uint4*)(Cs + r * LDC + swzC(r, c) * 8);
  }
#endif  // GEMMK_ENABLED
}

template<int BM, int BN, int BK, int STAGES, bool SWZ = false>
static constexpr size_t gemm_res_smem() {
  return SWZ ? (size_t)STAGES * (BM * BK + BK * BN) * 2
             : (size_t)STAGES * (BM * (BK + 8) + BK * (BN + 8)) * 2;
}
