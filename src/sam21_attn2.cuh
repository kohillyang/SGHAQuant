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
// sam21_attn2.cuh — flash attention for sm_80+ on raw mma.m16n8k16 / ldmatrix / cp.async.
//
// Same contract as k_attn (sam21_attn.cuh): reads Q, K, V straight out of the packed,
// head-padded qkv buffer ([T, 3*Cp], head h of Q at column h*DHP, of K at Cp + h*DHP, of
// V at 2*Cp + h*DHP with Cp = heads*DHP and ldk = 3*Cp, Q pre-scaled), writes the
// normalised output token-major into Xout[(w*ws2q + row)*Co + h*DH + col].
//
// The head dimension is a template parameter, DH in {56, 64, 72, 96}; DHP (the padded head
// stride inside qkv) is DH rounded up to a multiple of 16, i.e. 64, 64, 80, 96.  Everything
// that used to be hard-coded for 72 is derived from DH:
//   * QK^T runs NKS = DH/16 mma.m16n8k16 steps plus one m16n8k8 step when DH % 16 == 8
//     (DH = 56 and 72), so no padding lane is ever multiplied;
//   * P.V covers exactly DH output columns with NO = DH/8 n8 tiles (an x4 ldmatrix.trans
//     per full group of 16 columns, one x2 ldmatrix.trans for the trailing 8 columns when
//     DH % 16 == 8);
//   * the staged K/V row stride is LDS = DH when DH % 16 == 8, else DH + 8 halves.  The
//     ldmatrix conflict rule is that the 8 rows an 8x8 matrix load touches must start in 8
//     distinct 4-bank groups, i.e. (LDS/2)*r mod 32 for r = 0..7 must be 8 distinct
//     multiples of 4, which holds iff LDS % 16 == 8.  DH = 72 (144 B, stride 36 words == 4
//     mod 32) and DH = 56 (112 B, 28 words) already satisfy it and stay unpadded, exactly
//     as before; DH = 64 (128 B: every row starts on bank 0, an 8-way conflict) and DH = 96
//     (192 B: 48 words == 16 mod 32, a 4-way conflict) are padded by 8 halves, to 72 and
//     104.  Padding, not a swizzle: it keeps every cp.async destination and every ldmatrix
//     address a plain (row*LDS + col) and costs 12.5% / 8.3% of the K/V shared memory.
//
// What is different from k_attn:
//   * the [16 x BK] score tile never leaves registers: the mma accumulator layout is
//     repacked in place into the A-operand layout of the P.V mma, so there is no
//     accumulator -> shared -> registers -> shared -> fragment round trip and no smem
//     traffic for S or P at all (the FlashAttention-2 structure);
//   * K/V tiles are staged with cp.async into a STAGES-deep ring;
//   * Q is loaded once, directly from global, in fragment layout.
//
// TK (the key count) need NOT be a multiple of BK.  The last tile is loaded whole -- the
// caller must leave at least BK rows of readable, finite-valued slack after the last
// window -- and the keys at or past TK are masked out in the score registers before the
// softmax (score = -1e30, so they enter neither the row max nor the row sum, and their P
// is exactly 0, which is why the slack V rows only have to be finite).  The mask runs on
// the last tile only, behind one warp-uniform compare per tile, so a full tile pays
// nothing and a TK % BK == 0 launch is bit-identical to the unmasked kernel.
// Query rows past TQ are clamped to TQ-1 (computed, not written).
// sm_70 has none of ldmatrix / cp.async / m16n8k16, so the engine keeps k_attn there.

#include <cstdint>

// The body needs sm_80 PTX (ldmatrix, cp.async, mma.m16n8k16); in a sm_70 device pass the
// kernel compiles to an empty stub and the engine never launches it there.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define ATTN2_ENABLED 1
#else
#define ATTN2_ENABLED 0
#endif

// padded head stride inside the qkv buffer, and staged K/V row stride in shared memory
static __host__ __device__ constexpr int attn2_dhp(int dh) { return (dh + 15) / 16 * 16; }
static __host__ __device__ constexpr int attn2_lds(int dh) { return (dh % 16 == 8) ? dh : dh + 8; }

namespace attn2 {
#if ATTN2_ENABLED

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x2_t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
// D = A(16x16,row) . B(16x8,col) + D, fp16 in, fp32 accumulate
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// D = A(16x8,row) . B(8x8,col) + D
__device__ __forceinline__ void mma1688(float* c, const uint32_t* a, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(b0));
}
// fp16-accumulate variants (2x the tensor rate on GeForce Ampere): C/D are two packed half2
__device__ __forceinline__ void mma16816_h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma1688_h(uint32_t* c, const uint32_t* a, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
               "{%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(b0));
}
__device__ __forceinline__ float2 unpack2(uint32_t u) { return __half22float2(*(half2*)&u); }
__device__ __forceinline__ uint32_t ld32(const half* p) { return *(const uint32_t*)p; }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  half2 h = __floats2half2_rn(a, b);
  return *(uint32_t*)&h;
}
#endif  // ATTN2_ENABLED
}  // namespace attn2

// DH:   head dimension, one of {56, 64, 72, 96}.  DHP = attn2_dhp(DH) is the qkv head stride.
// QF16: accumulate QK^T in fp16 (scores rounded to fp16 once per k-step; ~1e-2 absolute
//       on a score of magnitude 10, i.e. the same trade the engine --f16acc GEMMs make).
// PF16: accumulate each BK-key slice of P.V in fp16 and add the slice into the fp32 O --
//       P <= 1 keeps the slice partials far from fp16 range, so this is accuracy-neutral.
// Q8OUT: write the attention result as int8 with a single calibrated scale (*oscale) instead
//        of fp16, so the proj GEMM can consume it directly.  One block owns one (window,
//        head), i.e. DH of the Co columns, so it cannot compute a per-token scale -- but with
//        a static one the quantisation is free here, and a separate pass is not needed.
template<int DH, int NW, int BK, int STAGES, bool QF16, bool PF16, int MINB = 1, bool Q8OUT = false>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn2(
    const half* __restrict__ qkv, const half* __restrict__ Qpool, void* __restrict__ Xoutv,
    int ws2, int ws2q, int heads, int Cp, int Co, int TQ, int TK,
    const float* __restrict__ oscale = nullptr) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16;
  constexpr int DHP = attn2_dhp(DH);      // head stride inside the qkv buffer
  constexpr int LDS = attn2_lds(DH);      // staged K/V row stride, halves (padded if needed)
  constexpr int TILE = BK * LDS;          // halves per K (or V) tile
  constexpr int NT = BK / 8;              // score n8 tiles per warp
  constexpr int NK = BK / 16;             // P.V k16 steps
  constexpr int NO = DH / 8;              // output n8 tiles
  constexpr int NKS = DH / 16;            // full k16 steps of QK^T / full 16-col groups of P.V
  constexpr bool KTAIL = (DH % 16) == 8;  // one extra k8 step / one extra 8-column group
  constexpr int CHR = DH / 8;             // 16 B chunks per staged K (or V) row
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(DH % 8 == 0 && DH >= 16, "DH must be a multiple of 8");
  static_assert(LDS % 16 == 8, "staged row stride must be 8 mod 16 for conflict-free ldmatrix");

  extern __shared__ __align__(16) char smem_raw[];
  half* Ks = (half*)smem_raw;           // [STAGES][BK][LDS]
  half* Vs = Ks + STAGES * TILE;        // [STAGES][BK][LDS]

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ldk = 3 * Cp;
  const half* K = qkv + (size_t)(w * ws2) * ldk + Cp + h * DHP;
  const int row0 = q0 + warp * 16;

  // ---- Q fragments, straight from global in mma A layout ----
  uint32_t qa[NKS][4], qt[KTAIL ? 2 : 1];
  {
    const half* Qrow; int ldq;
    if (Qpool) { Qrow = Qpool + (size_t)b * ws2q * DHP; ldq = DHP; }
    else       { Qrow = qkv + (size_t)(w * ws2) * ldk + h * DHP; ldq = ldk; }
    const half* p0 = Qrow + (size_t)min(row0 + g, TQ - 1) * ldq + 2 * q4;
    const half* p1 = Qrow + (size_t)min(row0 + g + 8, TQ - 1) * ldq + 2 * q4;
#pragma unroll
    for (int kk = 0; kk < NKS; ++kk) {
      qa[kk][0] = ld32(p0 + kk * 16);     qa[kk][1] = ld32(p1 + kk * 16);
      qa[kk][2] = ld32(p0 + kk * 16 + 8); qa[kk][3] = ld32(p1 + kk * 16 + 8);
    }
    if (KTAIL) { qt[0] = ld32(p0 + NKS * 16); qt[1] = ld32(p1 + NKS * 16); }
  }

  // ---- K/V tile staging: BK rows x CHR chunks of 16 B for each of K and V.  The chunk
  // -> (row, column) mapping is loop-invariant, so it is resolved once per thread. ----
  constexpr int NCH = BK * CHR, NPT = (NCH + NW * 32 - 1) / (NW * 32);
  const half* gsrc[NPT]; uint32_t sdst[NPT];
#pragma unroll
  for (int j = 0; j < NPT; ++j) {
    int i = tid + j * NW * 32; if (i >= NCH) i = NCH - 1;   // clamped: the tail is predicated
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

  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;    // rows g and g+8 of this warp

  const int ntiles = (TK + BK - 1) / BK;
  // the one tile that runs off the end of the window (-1 when TK % BK == 0, i.e. never)
  const int tmask = (TK % BK) ? (ntiles - 1) : -1;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < ntiles) load_tile(s, s * BK);
    cp_commit();
  }

  // ldmatrix lane -> row/col offsets (which of the four 8x8 matrices this lane addresses)
  const int mi = lane >> 3, ri = lane & 7;

  for (int t = 0; t < ntiles; ++t) {
    cp_wait<STAGES - 2>();
    __syncthreads();                                   // tile t visible; stage (t-1) free
    if (t + STAGES - 1 < ntiles) load_tile((t + STAGES - 1) % STAGES, (t + STAGES - 1) * BK);
    cp_commit();
    const half* Kt = Ks + (t % STAGES) * TILE;
    const half* Vt = Vs + (t % STAGES) * TILE;

    // ---- S = Q . K^T  (fp32 accumulators s[n][0..3]: rows g / g+8, cols 8n+2q4 (+1)) ----
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
      if (KTAIL) {   // k8 tail, channels NKS*16 .. DH-1
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

    // ---- tail mask.  This lane holds tile columns n*8 + 2*q4 (+1) for rows g and g+8, i.e.
    // keys t*BK + that column; everything at or past TK is pushed out of the softmax so it
    // enters neither the max nor the sum and gets P == 0 exactly.  Warp-uniform branch,
    // taken on at most one tile per block, so full tiles pay one compare. ----
    if (t == tmask) {
      const int kb0 = t * BK + 2 * q4;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (kb0 + n * 8     >= TK) { s[n][0] = -1e30f; s[n][2] = -1e30f; }
        if (kb0 + n * 8 + 1 >= TK) { s[n][1] = -1e30f; s[n][3] = -1e30f; }
      }
    }

    // ---- online softmax, entirely in registers ----
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
    // The running max settles after the first few tiles; skip the 4*NO multiplies (warp-
    // uniformly) whenever no row in the warp moved.
    if (__any_sync(0xffffffff, (c0 != 1.f) | (c1 != 1.f))) {
#pragma unroll
      for (int n = 0; n < NO; ++n) { o[n][0] *= c0; o[n][1] *= c0; o[n][2] *= c1; o[n][3] *= c1; }
    }

    // ---- O += P . V ----
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
      if (KTAIL) {   // trailing 8 output columns
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

  // ---- normalise and scatter into the token-major activation buffer ----
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
  const int r0 = row0 + g, r1 = row0 + g + 8;
  const size_t off0 = (size_t)(w * ws2q + r0) * Co + h * DH + 2 * q4;
  const size_t off1 = (size_t)(w * ws2q + r1) * Co + h * DH + 2 * q4;
  if (Q8OUT) {
    const float qinv = 1.f / *oscale;
    int8_t* q0 = (int8_t*)Xoutv + off0;
    int8_t* q1 = (int8_t*)Xoutv + off1;
    auto q2 = [&](float a, float b) {
      int ia = max(-127, min(127, __float2int_rn(a * qinv)));
      int ib = max(-127, min(127, __float2int_rn(b * qinv)));
      return (short)((ia & 0xff) | ((ib & 0xff) << 8));
    };
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      if (r0 < TQ) *(short*)(q0 + 8 * n) = q2(o[n][0] * inv0, o[n][1] * inv0);
      if (r1 < TQ) *(short*)(q1 + 8 * n) = q2(o[n][2] * inv1, o[n][3] * inv1);
    }
  } else {
    half* out0 = (half*)Xoutv + off0;
    half* out1 = (half*)Xoutv + off1;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      if (r0 < TQ) *(uint32_t*)(out0 + 8 * n) = pack2(o[n][0] * inv0, o[n][1] * inv0);
      if (r1 < TQ) *(uint32_t*)(out1 + 8 * n) = pack2(o[n][2] * inv1, o[n][3] * inv1);
    }
  }
#endif  // ATTN2_ENABLED
}

template<int DH, int BK, int STAGES>
static constexpr size_t attn2_smem() { return (size_t)STAGES * BK * attn2_lds(DH) * 2 * 2; }
