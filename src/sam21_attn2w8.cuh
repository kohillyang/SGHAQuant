#pragma once
// sam21_attn2w8.cuh — k_attn2 reading INT8 Q / K / V ("storage-only INT8", --win8).
//
// The windowed hiera blocks are memory-bound in the attention kernel (bytes ~ 8 C per token:
// Q, K, V read and O written in fp16), so this variant halves what it reads and leaves the
// math alone: the qkv GEMM (k_gemm_s8<..., W8G16>) writes Q, K and V as int8 with one fp16
// scale per (token, 16 channels), and the kernel dequantises (int8 -> fp16 x scale) while it
// stages K / V into the SAME fp16 shared tiles k_attn2 uses and while it builds the Q
// fragments.  QK^T, the softmax and P.V are k_attn2's code as is (fp16 mma.m16n8k16, the
// QF16 / PF16 accumulation modes unchanged), and O is written in fp16.
//
// Layout (what the producer writes):
//   q8: [rows, 3*Cp] int8, the fp16 qkv layout byte for byte (head h of Q at column h*DHP, of
//       K at Cp + h*DHP, of V at 2*Cp + h*DHP; pad columns DH..DHP-1 are zero)
//   sc: [3*Cp/16][sst] fp16, the scale of row r, columns 16c..16c+15 at sc[c*sst + r]
//       (head h's Q groups are c = h*DHP/16 + j, K groups Cp/16 + ..., V groups 2*Cp/16 + ...)
// Because the scale varies along the head dimension, it cannot be pulled out of QK^T or P.V;
// it is applied element-wise before the mma, which is why any 16-channel granularity is free.
//
// Staging: K / V tile t+1 is fetched into registers (16-byte int8 chunks = one scale group)
// right after the barrier that opens tile t, and dequantised into the other fp16 buffer after
// tile t's math -- the same one-tile look-ahead as k_attn2's two-stage cp.async ring, with one
// barrier per tile.  NBUF = 1 when the window is one tile (the 16-key windows): half the shared
// memory, no look-ahead needed.  int8 -> fp16 uses the 0x6400 magic (PRMT + HSUB2, exact) and
// one HMUL2 by the group scale, i.e. round_fp16(code * scale).
// Requires sam21_attn2.cuh (helpers, attn2_dhp / attn2_lds) first.

namespace attn2 {
#if ATTN2_ENABLED
// two int8 (the low 16 bits of u) -> half2, times s
__device__ __forceinline__ uint32_t w8_cvt2(uint32_t u, half2 s) {
  uint32_t h;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(h) : "r"(u ^ 0x8080u), "r"(0x64646464u), "r"(0x5150u));
  half2 v = __hsub2(*(half2*)&h, __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480)));
  v = __hmul2(v, s);
  return *(uint32_t*)&v;
}
// four int8 (one word) -> two half2, times s
__device__ __forceinline__ void w8_cvt4(uint32_t u, half2 s, uint32_t& lo, uint32_t& hi) {
  const uint32_t x = u ^ 0x80808080u;
  uint32_t a, b;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(a) : "r"(x), "r"(0x64646464u), "r"(0x5150u));
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(b) : "r"(x), "r"(0x64646464u), "r"(0x5352u));
  const half2 mg = __halves2half2(__ushort_as_half(0x6480), __ushort_as_half(0x6480));
  half2 va = __hmul2(__hsub2(*(half2*)&a, mg), s);
  half2 vb = __hmul2(__hsub2(*(half2*)&b, mg), s);
  lo = *(uint32_t*)&va; hi = *(uint32_t*)&vb;
}
#endif
}  // namespace attn2

// shared memory of k_attn2w8: NBUF fp16 K and V tiles, plus the int8 look-ahead chunks under CPA
template<int DH, int BK, int NBUF, bool CPA = false>
static __host__ __device__ constexpr size_t attn2w8_smem() {
  return (size_t)NBUF * BK * attn2_lds(DH) * 2 * 2 + (CPA ? (size_t)2 * BK * ((DH + 15) / 16) * 16 : 0);
}
// ... plus the pooled-Q tile of the fused-pooling mode (QP = 2), NW*16 rows
template<int DH, int BK, int NBUF, bool CPA, int NW>
static __host__ __device__ constexpr size_t attn2w8_smem_qp() { return attn2w8_smem<DH, BK, NBUF, CPA>() + (size_t)NW * 16 * attn2_lds(DH) * 2; }

// QP: the query-pooling blocks (--win8-pool), output rows w*ws2q + q.  QP = 1: Q comes pooled and in fp16 from Qpool
// ([(w*heads + h)*ws2q + q][DHP], k_pool_q_w8 below), exactly as k_attn2 takes it.  QP = 2: the kernel pools the
// query itself from the four int8 rows (window side wsx) with k_pool_q_w8's arithmetic, so the pooled Q is never
// written or re-read (same values bit for bit).
// CPA: the look-ahead tile goes through cp.async into a small int8 shared buffer instead of registers
// (each thread dequantises exactly the chunks it copied, so cp.async.wait_group is the only fence it
// needs): the 16-24 prefetch registers are not live across the tile's math.
template<int DH, int NW, int BK, bool QF16, bool PF16, int NBUF, int QP = 0, int MINB = 0, bool CPA = false>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn2w8(
    const int8_t* __restrict__ q8, const half* __restrict__ sc, half* __restrict__ Xout,
    int ws2, int heads, int Cp, int Co, int TQ, int TK, int sst,
    const half* __restrict__ Qpool = nullptr, int ws2q = 0, int wsx = 0) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16;
  constexpr int DHP = attn2_dhp(DH);
  constexpr int LDS = attn2_lds(DH);
  constexpr int TILE = BK * LDS;
  constexpr int NT = BK / 8, NK = BK / 16, NO = DH / 8, NKS = DH / 16;
  constexpr bool KTAIL = (DH % 16) == 8;
  constexpr int CHG = (DH + 15) / 16;      // 16-channel groups (16-byte int8 chunks) per staged row
  constexpr int GPH = DHP / 16;            // scale groups per head
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0 && DH % 8 == 0 && LDS % 16 == 8, "shape");
  static_assert(NBUF == 1 || NBUF == 2, "one or two fp16 buffers");

  extern __shared__ __align__(16) char smem_raw[];
  half* Ks = (half*)smem_raw;              // [NBUF][BK][LDS]
  half* Vs = Ks + NBUF * TILE;             // [NBUF][BK][LDS]

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ld8 = 3 * Cp;
  const size_t wrow = (size_t)w * ws2;
  const size_t G3 = (size_t)(Cp / 16) * sst;         // scale offset of one part (Q -> K -> V)
  const int row0 = q0 + warp * 16;

  // ---- K / V chunk -> (row, group) mapping, loop-invariant ----
  constexpr int NCH = BK * CHG, NPT = (NCH + NW * 32 - 1) / (NW * 32);
  int crow[NPT], cgrp[NPT];
#pragma unroll
  for (int j = 0; j < NPT; ++j) {
    int i = tid + j * NW * 32; if (i >= NCH) i = NCH - 1;
    crow[j] = i / CHG; cgrp[j] = i - crow[j] * CHG;
  }
  auto cok = [&](int j) { return NCH % (NW * 32) == 0 || tid + j * NW * 32 < NCH; };
  const int8_t* Kg = q8 + wrow * ld8 + Cp + h * DHP;          // K row 0 of the window
  const half* Ksc = sc + G3 + (size_t)(h * GPH) * sst + wrow;  // K scales, group 0, row 0
  uint4 kr[CPA ? 1 : NPT], vr[CPA ? 1 : NPT];
  half ksr[NPT], vsr[NPT];
  int8_t* K8s = (int8_t*)(Vs + NBUF * TILE);                   // CPA: [NCH][16] int8 K, then V
  auto fetch = [&](int k0) {
#pragma unroll
    for (int j = 0; j < NPT; ++j)
      if (cok(j)) {
        const int8_t* p = Kg + (size_t)(k0 + crow[j]) * ld8 + cgrp[j] * 16;
        if constexpr (CPA) {
          const uint32_t d = smem_u32(K8s + (tid + j * NW * 32) * 16);
          cp_async16(d, p);
          cp_async16(d + NCH * 16, p + Cp);
        } else {
          kr[j] = *(const uint4*)p;
          vr[j] = *(const uint4*)(p + Cp);
        }
        const half* s = Ksc + (size_t)cgrp[j] * sst + k0 + crow[j];
        ksr[j] = s[0];
        vsr[j] = s[G3];
      }
    if constexpr (CPA) cp_commit();
  };
  auto stash1 = [&](const uint4& q, half s, half* d, bool full) {
    const half2 s2 = __half2half2(s);
    uint32_t o[8];
    w8_cvt4(q.x, s2, o[0], o[1]); w8_cvt4(q.y, s2, o[2], o[3]);
    *(uint4*)d = make_uint4(o[0], o[1], o[2], o[3]);
    if (full) {
      w8_cvt4(q.z, s2, o[4], o[5]); w8_cvt4(q.w, s2, o[6], o[7]);
      *(uint4*)(d + 8) = make_uint4(o[4], o[5], o[6], o[7]);
    }
  };
  auto stash = [&](int buf) {
    if constexpr (CPA) cp_wait<0>();                   // this thread's own chunks have landed
#pragma unroll
    for (int j = 0; j < NPT; ++j)
      if (cok(j)) {
        const int off = buf * TILE + crow[j] * LDS + cgrp[j] * 16;
        const bool full = !(KTAIL && cgrp[j] == CHG - 1);    // the k8 tail group holds 8 channels
        if constexpr (CPA) {
          const int8_t* src = K8s + (tid + j * NW * 32) * 16;
          stash1(*(const uint4*)src, ksr[j], Ks + off, full);
          stash1(*(const uint4*)(src + NCH * 16), vsr[j], Vs + off, full);
        } else {
          stash1(kr[j], ksr[j], Ks + off, full);
          stash1(vr[j], vsr[j], Vs + off, full);
        }
      }
  };

  // tile 0 in flight before the Q loads, so the two latencies overlap
  fetch(0);

  // ---- Q fragments from the int8 rows, dequantised per 16-channel group ----
  uint32_t qa[NKS][4], qt[KTAIL ? 2 : 1];
  if constexpr (QP == 2) {
    // fused 2x2 pooling of the int8 Q rows with k_pool_q_w8's arithmetic, done cooperatively: one thread per
    // (pooled row, 16-channel group) reads the four source rows' 16-byte chunks and scales, dequantises, takes
    // the max and writes 16 fp16 into a shared [BQ][LDS] tile; the fragments are then read from there.
    half* Qs = (half*)(smem_raw + attn2w8_smem<DH, BK, NBUF, CPA>());
    const int wsq = wsx >> 1;
    constexpr int QCH = BQ * CHG;
    for (int i = tid; i < QCH; i += NW * 32) {
      const int pr = i / CHG, grp = i - pr * CHG;
      const int tq = min(q0 + pr, TQ - 1);
      const int oh = tq / wsq, ow = tq - oh * wsq;
      half2 mx[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) mx[k] = __floats2half2_rn(-65504.f, -65504.f);
#pragma unroll
      for (int sidx = 0; sidx < 4; ++sidx) {
        const int x = (2 * oh + (sidx >> 1)) * wsx + 2 * ow + (sidx & 1);
        const uint4 u = *(const uint4*)(q8 + (wrow + x) * ld8 + h * DHP + grp * 16);
        const half2 s2 = __half2half2(sc[(size_t)(h * GPH + grp) * sst + wrow + x]);
        uint32_t v[8];
        w8_cvt4(u.x, s2, v[0], v[1]); w8_cvt4(u.y, s2, v[2], v[3]);
        w8_cvt4(u.z, s2, v[4], v[5]); w8_cvt4(u.w, s2, v[6], v[7]);
#pragma unroll
        for (int k = 0; k < 8; ++k) mx[k] = __hmax2(mx[k], *(const half2*)&v[k]);
      }
      half* d = Qs + pr * LDS + grp * 16;
      const uint32_t* m32 = (const uint32_t*)mx;
      *(uint4*)d = make_uint4(m32[0], m32[1], m32[2], m32[3]);
      if (!(KTAIL && grp == CHG - 1)) *(uint4*)(d + 8) = make_uint4(m32[4], m32[5], m32[6], m32[7]);
    }
    __syncthreads();
    const half* p0 = Qs + (warp * 16 + g) * LDS + 2 * q4;
    const half* p1 = p0 + 8 * LDS;
#pragma unroll
    for (int kk = 0; kk < NKS; ++kk) {
      qa[kk][0] = ld32(p0 + kk * 16);     qa[kk][1] = ld32(p1 + kk * 16);
      qa[kk][2] = ld32(p0 + kk * 16 + 8); qa[kk][3] = ld32(p1 + kk * 16 + 8);
    }
    if (KTAIL) { qt[0] = ld32(p0 + NKS * 16); qt[1] = ld32(p1 + NKS * 16); }
  } else if constexpr (QP == 1) {          // pooled fp16 Q, k_attn2's fragment load
    const half* Qrow = Qpool + (size_t)b * ws2q * DHP;
    const half* p0 = Qrow + (size_t)min(row0 + g, TQ - 1) * DHP + 2 * q4;
    const half* p1 = Qrow + (size_t)min(row0 + g + 8, TQ - 1) * DHP + 2 * q4;
#pragma unroll
    for (int kk = 0; kk < NKS; ++kk) {
      qa[kk][0] = ld32(p0 + kk * 16);     qa[kk][1] = ld32(p1 + kk * 16);
      qa[kk][2] = ld32(p0 + kk * 16 + 8); qa[kk][3] = ld32(p1 + kk * 16 + 8);
    }
    if (KTAIL) { qt[0] = ld32(p0 + NKS * 16); qt[1] = ld32(p1 + NKS * 16); }
  } else {
    const int ra = min(row0 + g, TQ - 1), rb = min(row0 + g + 8, TQ - 1);
    const int8_t* p0 = q8 + (wrow + ra) * ld8 + h * DHP + 2 * q4;
    const int8_t* p1 = q8 + (wrow + rb) * ld8 + h * DHP + 2 * q4;
    const half* s0 = sc + (size_t)(h * GPH) * sst + wrow + ra;
    const half* s1 = sc + (size_t)(h * GPH) * sst + wrow + rb;
#pragma unroll
    for (int kk = 0; kk < NKS; ++kk) {
      const half2 sa = __half2half2(s0[(size_t)kk * sst]), sb = __half2half2(s1[(size_t)kk * sst]);
      qa[kk][0] = w8_cvt2(*(const uint16_t*)(p0 + kk * 16), sa);
      qa[kk][1] = w8_cvt2(*(const uint16_t*)(p1 + kk * 16), sb);
      qa[kk][2] = w8_cvt2(*(const uint16_t*)(p0 + kk * 16 + 8), sa);
      qa[kk][3] = w8_cvt2(*(const uint16_t*)(p1 + kk * 16 + 8), sb);
    }
    if (KTAIL) {
      qt[0] = w8_cvt2(*(const uint16_t*)(p0 + NKS * 16), __half2half2(s0[(size_t)NKS * sst]));
      qt[1] = w8_cvt2(*(const uint16_t*)(p1 + NKS * 16), __half2half2(s1[(size_t)NKS * sst]));
    }
  }
  stash(0);

  float o[NO][4];
#pragma unroll
  for (int n = 0; n < NO; ++n) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
  float m0 = -1e30f, m1 = -1e30f, l0 = 0.f, l1 = 0.f;

  const int ntiles = (TK + BK - 1) / BK;
  const int tmask = (TK % BK) ? (ntiles - 1) : -1;
  const int mi = lane >> 3, ri = lane & 7;

  for (int t = 0; t < ntiles; ++t) {
    __syncthreads();                                   // tile t staged; the other buffer free
    const bool more = t + 1 < ntiles;
    if (more) fetch((t + 1) * BK);
    const half* Kt = Ks + (t % NBUF) * TILE;
    const half* Vt = Vs + (t % NBUF) * TILE;

    // ---- S = Q . K^T (k_attn2 as is) ----
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
    if (t == tmask) {
      const int kb0 = t * BK + 2 * q4;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (kb0 + n * 8     >= TK) { s[n][0] = -1e30f; s[n][2] = -1e30f; }
        if (kb0 + n * 8 + 1 >= TK) { s[n][1] = -1e30f; s[n][3] = -1e30f; }
      }
    }

    // ---- online softmax ----
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

    if (more) {
      if (NBUF == 1) __syncthreads();                  // single buffer: every warp is done with it
      stash((t + 1) % NBUF);
    }
  }

  // ---- normalise and scatter (fp16, token-major) ----
  l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
  l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
  const float inv0 = 1.f / l0, inv1 = 1.f / l1;
  const int r0 = row0 + g, r1 = row0 + g + 8;
  const size_t orow = QP ? (size_t)w * ws2q : wrow;
  half* out0 = Xout + (orow + r0) * Co + h * DH + 2 * q4;
  half* out1 = Xout + (orow + r1) * Co + h * DH + 2 * q4;
#pragma unroll
  for (int n = 0; n < NO; ++n) {
    if (r0 < TQ) *(uint32_t*)(out0 + 8 * n) = pack2(o[n][0] * inv0, o[n][1] * inv0);
    if (r1 < TQ) *(uint32_t*)(out1 + 8 * n) = pack2(o[n][2] * inv1, o[n][3] * inv1);
  }
#endif  // ATTN2_ENABLED
}


// pooled Q for the transition blocks from the int8 qkv (--win8-pool): k_pool_q_g on the dequantised
// values round_fp16(code * scale), i.e. the same numbers k_pool_q_g would pool from the fp16 buffer
// the int8 codes stand for.  2x2 max over each window, padded head layout.
__global__ void k_pool_q_w8(const int8_t* __restrict__ q8, const half* __restrict__ sc, half* __restrict__ Qb,
                            int nW, int ws, int heads, int Cp, int dhp, int sst) {
#if ATTN2_ENABLED
  const int ws2 = ws * ws, wsq = ws / 2, ws2q = ws2 / 4;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nW * heads * ws2q * (dhp / 2)) return;
  const int d2 = idx % (dhp / 2), rest = idx / (dhp / 2);
  const int tq = rest % ws2q, b = rest / ws2q;
  const int h = b % heads, w = b / heads;
  const int oh = tq / wsq, ow = tq - oh * wsq;
  const int col = h * dhp + 2 * d2;
  const size_t r0 = (size_t)w * ws2;
  const half* s = sc + (size_t)(col >> 4) * sst + r0;
  half2 mx = __floats2half2_rn(-65504.f, -65504.f);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int r = (2 * oh + a) * ws + (2 * ow + c);
      const uint32_t u = *(const uint16_t*)(q8 + (r0 + r) * (3 * Cp) + col);
      const uint32_t v = attn2::w8_cvt2(u, __half2half2(s[r]));
      mx = __hmax2(mx, *(const half2*)&v);
    }
  ((half2*)(Qb + (size_t)(b * ws2q + tq) * dhp))[d2] = mx;
#endif
}
