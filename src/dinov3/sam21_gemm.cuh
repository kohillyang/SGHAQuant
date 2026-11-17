#pragma once
// sam21_gemm.cuh — fused fc1 GEMM + bias + GELU for sm_80+.
//
//   C[M,N] = gelu(A[M,K] @ B[K,N] + bias[N])      A, B, C row-major fp16, bias fp16
//
// cuBLASLt's GELU_BIAS epilogue costs as much as the standalone GELU pass it replaces
// (+3.1 ms on the 3090, measured with --lt-gelu), so the fusion has to be hand-written.
// This is a conventional Ampere multistage GEMM: BM x BN x BK block tiles staged with
// cp.async into a STAGES-deep ring, ldmatrix into mma.m16n8k16 fragments, WM x WN warps
// each owning a (BM/WM) x (BN/WN) sub-tile, and the bias + GELU applied on the accumulator
// fragments before the fp16 store.  Padded shared strides (BK+8, BN+8 halves) make every
// ldmatrix conflict-free.  K is handled at 8-halves granularity with zero-filled cp.async
// (the trunk has K = 144, i.e. 4.5 tiles of 32); N is predicated per 8-column chunk.
//
// Requires M % BM == 0, K % 8 == 0, N % 8 == 0.  Same GELU formula as k_gelu.

#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
#define GEMMK_ENABLED 1
#else
#define GEMMK_ENABLED 0
#endif

namespace gemmk {
#if GEMMK_ENABLED
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
// 16-byte async copy; src_bytes = 0 writes zeros instead of reading (used for K/N edges)
__device__ __forceinline__ void cp_async16z(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template<int N> __device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}
__device__ __forceinline__ void ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma16816_h(uint32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
               "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float2 unpack2(uint32_t u) { return __half22float2(*(half2*)&u); }
__device__ __forceinline__ uint32_t pack2(float a, float b) {
  half2 h = __floats2half2_rn(a, b);
  return *(uint32_t*)&h;
}
// tanh-form GELU: 0.5*x*(1 + tanh(0.79788456*(x + 0.044715*x^3))), one MUFU.TANH per
// element.  (k_gelu's sigmoid form uses __frcp_rn, which is an IEEE-exact subroutine with
// a CALL -- harmless in a bandwidth-bound pass, ruinous inside a GEMM epilogue.)
__device__ __forceinline__ float tanh_approx(float x) {
  float y; asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x)); return y;
}
#ifndef GEMMK_GELU_TANH
// sigmoid form with fast exp + fast reciprocal (two MUFU, ~1e-6 relative): same accuracy
// as k_gelu.  GEMMK_GELU_TANH selects the one-MUFU tanh.approx form (~5e-4 relative).
__device__ __forceinline__ float gelu(float x) {
  const float u = x * fmaf(x * x, 0.0713548162f, 1.5957691216f);
  return x * __fdividef(1.f, 1.f + __expf(-u));
}
#else
__device__ __forceinline__ float gelu(float x) {
  const float hx = 0.5f * x;
  const float u = x * fmaf(x * x, 0.0356774081f, 0.7978845608f);
  return fmaf(hx, tanh_approx(u), hx);
}
#endif
#endif
}  // namespace gemmk

// SWZ: XOR-swizzled shared layout with no padding (16 KB per 128x128x32 stage instead of
//      19 KB, which is what lets three stages fit two blocks per SM); otherwise padded rows.
// DB:  double-buffer the ldmatrix fragments (costs ~24-48 registers; off for the fp32-
//      accumulate variants so two 256-thread blocks still fit the register file).
template<int BM, int BN, int BK, int WM, int WN, int STAGES, bool ACC16, bool SWZ = false, bool DB = true>
__global__ __launch_bounds__(WM * WN * 32) void k_gemm_gelu(
    const half* __restrict__ A, const half* __restrict__ B, const half* __restrict__ bias,
    half* __restrict__ C, int M, int N, int K) {
#if GEMMK_ENABLED
  using namespace gemmk;
  constexpr int NT = WM * WN * 32;
  constexpr int LDA = SWZ ? BK : BK + 8, LDB = SWZ ? BN : BN + 8;   // halves
  constexpr int A_STAGE = BM * LDA, B_STAGE = BK * LDB;
  constexpr int TM = BM / WM, TN = BN / WN;          // warp tile
  constexpr int MI = TM / 16, NI = TN / 8;           // mma tiles per warp
  constexpr int AKC = BK / 8, BNC = BN / 8;          // 16-byte chunks per row
  constexpr int ARPL = 64 / BK;                      // A rows per 128-byte line (swizzle period)
  static_assert(BK % 16 == 0 && TM % 16 == 0 && TN % 16 == 0, "tile shape");
  static_assert(!SWZ || (BNC >= 8 && BK <= 64), "swizzle needs >= 8 chunks per B row");
  // logical chunk index -> physical chunk within the row
  auto swzA = [](int r, int c) { return SWZ ? (c ^ ((r / ARPL) & (AKC - 1))) : c; };
  auto swzB = [](int r, int c) { return SWZ ? (c ^ (r & 7)) : c; };

  extern __shared__ __align__(16) char smem_raw[];
  half* As = (half*)smem_raw;
  half* Bs = As + STAGES * A_STAGE;

  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp - wm * WN;
  const int g = lane >> 2, q4 = lane & 3;

  // ---- global -> shared chunk mapping, resolved once ----
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

  // ldmatrix lane -> (row, chunk) within a 16x16 fragment.  With SWZ the XOR term depends
  // only on the row of the lane (mi*16 and ks*16 are multiples of the swizzle period), so
  // the per-(ks, n2) chunk offsets are lane constants.
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
    for (int mi = 0; mi < MI; ++mi)
      ldsm_x4(af[buf][mi], at + (uint32_t)(mi * 16 * LDA) * 2);
#pragma unroll
    for (int n2 = 0; n2 < NI / 2; ++n2) {
      uint32_t t[4];
      ldsm_x4_t(t, bt + boff[n2]);
      bf[buf][2 * n2][0] = t[0]; bf[buf][2 * n2][1] = t[1];
      bf[buf][2 * n2 + 1][0] = t[2]; bf[buf][2 * n2 + 1][1] = t[3];
    }
  };

  // Software pipeline: S-1 tiles in flight in shared memory, and the fragments of the
  // next k16 step (or the next tile's first step) are loaded while the current step's
  // mmas run.  The barrier sits before the last step of each tile: by then every warp has
  // issued all its ldmatrix reads of this tile, so the stage can be refilled right after.
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

  // ---- epilogue: + bias, GELU, stage the fp16 tile in shared, 16-byte coalesced stores ----
  constexpr int LDC = SWZ ? BN : BN + 8;
  static_assert((size_t)BM * LDC * 2 <= (size_t)STAGES * (A_STAGE + B_STAGE) * 2, "C tile must fit the ring");
  auto swzC = [](int r, int c) { return SWZ ? (c ^ (r & 7)) : c; };
  half* Cs = (half*)smem_raw;
  __syncthreads();                                   // everyone is done reading the ring
#pragma unroll
  for (int ni = 0; ni < NI; ++ni) {
    const int cch = wn * TN / 8 + ni;                // logical 16-byte chunk of this n8 tile
    const int coll = swzC(g, cch) * 8 + 2 * q4;      // rows g and g+8 share (r & 7)
    const int col = n0 + cch * 8 + 2 * q4;
    const float2 bb = col < N ? __half22float2(*(const half2*)(bias + col)) : make_float2(0.f, 0.f);
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
      *(uint32_t*)(Cs + rowl * LDC + coll) = pack2(gelu(v0 + bb.x), gelu(v1 + bb.y));
      *(uint32_t*)(Cs + (rowl + 8) * LDC + coll) = pack2(gelu(v2 + bb.x), gelu(v3 + bb.y));
    }
  }
  __syncthreads();
  constexpr int CCH = BM * (BN / 8);
#pragma unroll 4
  for (int i = tid; i < CCH; i += NT) {
    const int r = i / (BN / 8), c = i - r * (BN / 8);
    const int col = n0 + c * 8;
    if (col < N)
      *(uint4*)(C + (size_t)(m0 + r) * N + col) = *(const uint4*)(Cs + r * LDC + swzC(r, c) * 8);
  }
#endif  // GEMMK_ENABLED
}

template<int BM, int BN, int BK, int STAGES, bool SWZ = false>
static constexpr size_t gemm_gelu_smem() {
  return SWZ ? (size_t)STAGES * (BM * BK + BK * BN) * 2
             : (size_t)STAGES * (BM * (BK + 8) + BK * (BN + 8)) * 2;
}
