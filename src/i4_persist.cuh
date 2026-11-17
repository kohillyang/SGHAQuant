#pragma once
// i4_persist.cuh -- persistent INT4 GEMMs for the W4A4 hiera encoder: the qkv GEMM under --win8 (W8G16 epilogue)
// and the DHD fc1 GEMM (GELU + H_64 + per-(token, 128) int4 producer).  2026-10-14 (int4_gemm_opt_20261014).
//
// Why: at hiera-L's shapes (K = 144..1152 int4, i.e. 2..9 k-tiles of 64 bytes) every 128x128 tile spends ~3500
// cycles waiting for its first k-tile (the L2 queue under load) and then runs an epilogue that touches no global
// operand at all, so the memory system idles during the epilogue and the ALUs idle during the prologue.  Here a
// CTA loops over tiles and, as soon as its main loop is done, issues the cp.async of the NEXT tile's first STAGES-1
// k-tiles before it starts the epilogue of the current one.  The ring is laid out stage-interleaved
// ([A_s | B_s] per stage, not all A stages then all B stages) so that stage STAGES-1 alone holds the epilogue's
// staging tile and stages 0..STAGES-2 are free for the prefetch.
//
// Operands: A rows have the pitch K (bytes, kpad4(K) / 2); B rows the pitch ldb >= K.  The engine passes row-padded
// copies of the qkv / fc1 weights with a 64-byte-multiple pitch: a 64-byte k-tile segment of a 288-byte row
// (K = 576) straddles two 128-byte lines for half of the rows, which costs ~15 % of the L2 -> SM throughput
// (scripts/l2pattern.cu: 3.64 TB/s at pitch 288 vs 4.30 at 320).  The A rows come from the LayerNorm producers and
// keep their pitch.
//
// Numerics: the main loop is the k_gemm_s8 <I4> loop (integer accumulation, order-free) and both epilogues are the
// shipped code token for token (k_gemm_s8 W8G16, k_gemm_s8_dhd ROT8/I4), so every output bit is the same as the
// non-persistent kernels'.  The one deliberate rewrite is the int4 code of the fc1 producer (I4_QMAGIC, below).
#include "sam21_gemm_s8_mixed.cuh"

// I4_QMAGIC = 1: the fc1 producer rounds its int4 codes with the 1.5 * 2^23 magic add (FMNMX, FMNMX, FADD on the
// FMA pipe) instead of F2I (cvt.rni, the quarter-rate XU pipe that GELU's two MUFU ops already load).  Same codes
// for every non-NaN input: clamping to [-7, 7] before rounding equals clamping the rounded integer (the bounds are
// integers), and for |p| <= 7 the FADD rounds p to the nearest integer, ties to even, exactly as cvt.rni does; the
// low nibble of the sum's bits is the two's-complement code.  (A NaN product would give -7 instead of 0.)
#ifndef I4_QMAGIC
#define I4_QMAGIC 1
#endif

namespace i4p {
__device__ __forceinline__ uint32_t q4code(float p) {
#if I4_QMAGIC
  const float c = fminf(fmaxf(p, -7.f), 7.f);
  return __float_as_uint(c + 12582912.f) & 0xfu;
#else
  return (uint32_t)max(-7, min(7, __float2int_rn(p))) & 0xfu;
#endif
}
}  // namespace i4p

// The shared main loop of one BM x BN tile: cp.async ring (stage-interleaved), ldmatrix, mma.m16n8k64 .s4.
// Expects stages 0..ST-2 of this tile already committed (by the first-tile prologue or the previous tile's
// prefetch); returns with every warp past its last ldmatrix (callers sync before reusing the ring).
#define I4P_COMMON(BM, BN, WM, WN, ST)                                                                         \
  using namespace s8k;                                                                                         \
  using namespace gemmk;                                                                                       \
  constexpr int BK = 64;                                                                                       \
  constexpr int NT = WM * WN * 32;                                                                             \
  constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK, STAGE = A_STAGE + B_STAGE;                               \
  constexpr int TM = BM / WM, TN = BN / WN;                                                                    \
  constexpr int MI = TM / 16, NI = TN / 8;                                                                     \
  constexpr int KC = BK / 16;                                                                                  \
  constexpr int RPL = 128 / BK;                                                                                \
  constexpr int KS = BK / 32;                                                                                  \
  static_assert(TM % 16 == 0 && TN % 16 == 0, "tile");                                                         \
  extern __shared__ __align__(32) char smem_raw[];                                                             \
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;                                               \
  const int wm = warp / WN, wn = warp - wm * WN;                                                               \
  const int g = lane >> 2, q4 = lane & 3;                                                                      \
  auto swz = [](int r, int c) { return c ^ ((r / RPL) & (KC - 1)); };                                          \
  constexpr int ACH = BM * KC, BCH = BN * KC;                                                                  \
  constexpr int APT = (ACH + NT - 1) / NT, BPT = (BCH + NT - 1) / NT;                                          \
  uint32_t adst[APT], bdst[BPT]; int arow[APT], brow[BPT], akc[APT], bkc[BPT];                                 \
  _Pragma("unroll") for (int j = 0; j < APT; ++j) {                                                            \
    int i = tid + j * NT; if (i >= ACH) i = ACH - 1;                                                           \
    const int r = i / KC, c = i - r * KC;                                                                      \
    arow[j] = r; akc[j] = c * 16; adst[j] = smem_u32(smem_raw + r * BK + swz(r, c) * 16);                     \
  }                                                                                                            \
  _Pragma("unroll") for (int j = 0; j < BPT; ++j) {                                                            \
    int i = tid + j * NT; if (i >= BCH) i = BCH - 1;                                                           \
    const int r = i / KC, c = i - r * KC;                                                                      \
    brow[j] = r; bkc[j] = c * 16; bdst[j] = smem_u32(smem_raw + A_STAGE + r * BK + swz(r, c) * 16);           \
  }                                                                                                            \
  const int8_t* asrc[APT]; const int8_t* bsrc[BPT]; bool bok[BPT];                                             \
  auto set_tile = [&](int m0_, int n0_) {                                                                      \
    _Pragma("unroll") for (int j = 0; j < APT; ++j) asrc[j] = A + (size_t)(m0_ + arow[j]) * K + akc[j];        \
    _Pragma("unroll") for (int j = 0; j < BPT; ++j) {                                                          \
      bok[j] = n0_ + brow[j] < N;                                                                              \
      bsrc[j] = Bt + (size_t)(bok[j] ? n0_ + brow[j] : 0) * ldb + bkc[j];                                      \
    }                                                                                                          \
  };                                                                                                           \
  auto load_tile = [&](int stage, int kt) {                                                                    \
    const int k0 = kt * BK;                                                                                    \
    const uint32_t so = stage * STAGE;                                                                         \
    _Pragma("unroll") for (int j = 0; j < APT; ++j)                                                            \
      if (ACH % NT == 0 || tid + j * NT < ACH) {                                                               \
        bool ok = k0 + akc[j] < K;                                                                             \
        cp_async16z(adst[j] + so, asrc[j] + (ok ? k0 : 0), ok ? 16 : 0);                                       \
      }                                                                                                        \
    _Pragma("unroll") for (int j = 0; j < BPT; ++j)                                                            \
      if (BCH % NT == 0 || tid + j * NT < BCH) {                                                               \
        bool ok = bok[j] && (k0 + bkc[j] < K);                                                                 \
        cp_async16z(bdst[j] + so, bsrc[j] + (ok ? k0 : 0), ok ? 16 : 0);                                       \
      }                                                                                                        \
  };                                                                                                           \
  const int alr = (lane & 7) + ((lane >> 3) & 1) * 8, alc = (lane >> 4);                                       \
  const int blr = (lane & 7) + ((lane >> 4) & 1) * 8, blc = (lane >> 3) & 1;                                   \
  const uint32_t a_row = smem_u32(smem_raw) + (uint32_t)((wm * TM + alr) * BK);                                \
  const uint32_t b_row = smem_u32(smem_raw + A_STAGE) + (uint32_t)((wn * TN + blr) * BK);                      \
  uint32_t aoff[KS], boff[KS];                                                                                 \
  _Pragma("unroll") for (int ks = 0; ks < KS; ++ks) {                                                          \
    aoff[ks] = swz(wm * TM + alr, ks * 2 + alc) * 16;                                                          \
    boff[ks] = swz(wn * TN + blr, ks * 2 + blc) * 16;                                                          \
  }                                                                                                            \
  constexpr int NB = DB ? 2 : 1;                                                                               \
  uint32_t af[NB][MI][4], bf[NB][NI][2];                                                                       \
  auto load_frags = [&](int buf, int stage, int ks) {                                                          \
    const uint32_t at = a_row + stage * STAGE + aoff[ks];                                                      \
    const uint32_t bt = b_row + stage * STAGE + boff[ks];                                                      \
    _Pragma("unroll") for (int mi = 0; mi < MI; ++mi) ldsm_x4(af[buf][mi], at + mi * 16 * BK);                 \
    _Pragma("unroll") for (int n2 = 0; n2 < NI / 2; ++n2) {                                                    \
      uint32_t t4[4];                                                                                          \
      ldsm_x4(t4, bt + n2 * 16 * BK);                                                                          \
      bf[buf][2 * n2][0] = t4[0]; bf[buf][2 * n2][1] = t4[1];                                                  \
      bf[buf][2 * n2 + 1][0] = t4[2]; bf[buf][2 * n2 + 1][1] = t4[3];                                          \
    }                                                                                                          \
  };                                                                                                           \
  const int ktiles = (K + BK - 1) / BK;

// the main loop body (acc zeroed by the caller); identical sequence to k_gemm_s8 <I4, GROUP = 0>
#define I4P_MAINLOOP()                                                                                         \
  cp_wait<ST - 2>(); __syncthreads();                                                                          \
  if (ST - 1 < ktiles) load_tile(ST - 1, ST - 1);                                                              \
  cp_commit();                                                                                                 \
  if (DB) load_frags(0, 0, 0);                                                                                 \
  for (int kt = 0; kt < ktiles; ++kt) {                                                                        \
    const int st = kt % ST;                                                                                    \
    _Pragma("unroll") for (int ks = 0; ks < KS; ++ks) {                                                        \
      const int cur = DB ? (ks & 1) : 0;                                                                       \
      if (!DB) load_frags(0, st, ks);                                                                          \
      if (ks == KS - 1) {                                                                                      \
        if (kt + 1 < ktiles) {                                                                                 \
          cp_wait<ST - 2>(); __syncthreads();                                                                  \
          if (kt + ST < ktiles) load_tile(st, kt + ST);                                                        \
          cp_commit();                                                                                         \
          if (DB) load_frags(cur ^ 1, (kt + 1) % ST, 0);                                                       \
        }                                                                                                      \
      } else if (DB) load_frags(cur ^ 1, st, ks + 1);                                                          \
      _Pragma("unroll") for (int mi = 0; mi < MI; ++mi)                                                        \
        _Pragma("unroll") for (int ni = 0; ni < NI; ++ni)                                                      \
          mma16864_s4(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);                               \
    }                                                                                                          \
  }

// ------------------------------------------------------------------------------------------------ qkv (W8G16)
// The qkv GEMM under --win8: int8 Q/K/V with one fp16 scale per (row, 16 columns), the epilogue of k_gemm_s8
// <W8G16> verbatim.  C8s = ring stage ST-1.
template<int BM, int BN, int WM, int WN, int ST, bool DB, int MINB>
__global__ __launch_bounds__(WM * WN * 32, MINB) void k_qkv4p(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bt, const float* __restrict__ sa,
    const float* __restrict__ sw, const half* __restrict__ bias, int8_t* __restrict__ C8, float* __restrict__ sa2,
    int M, int N, int K, int ldb, int sa_stride, int ntn, int ntiles) {
#if S8_ENABLED
  I4P_COMMON(BM, BN, WM, WN, ST)
  static_assert(ST >= 2 && (size_t)BM * BN <= (size_t)STAGE, "the int8 C tile must fit one ring stage");
  static_assert(TN % 16 == 0 && BN % 16 == 0, "16-column groups");
  int8_t* C8s = (int8_t*)(smem_raw + (ST - 1) * STAGE);
  half* S16 = (half*)sa2;
  int t = blockIdx.x;
  if (t >= ntiles) return;
  set_tile((t / ntn) * BM, (t % ntn) * BN);
  _Pragma("unroll") for (int s = 0; s < ST - 1; ++s) { if (s < ktiles) load_tile(s, s); cp_commit(); }
  for (; t < ntiles; t += gridDim.x) {
    const int by = t / ntn, bx = t - by * ntn;
    const int m0 = by * BM, n0 = bx * BN;
    int32_t acc[MI][NI][4];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
    I4P_MAINLOOP()
    __syncthreads();                                   // every warp is done with the ring
    {
      const int tn = t + gridDim.x;                    // prefetch the next tile's first ST-1 k-tiles
      if (tn < ntiles) {
        set_tile((tn / ntn) * BM, (tn % ntn) * BN);
#pragma unroll
        for (int s = 0; s < ST - 1; ++s) { if (s < ktiles) load_tile(s, s); cp_commit(); }
      }
    }
    // ---- epilogue: k_gemm_s8 <W8G16> verbatim (C8s is ring stage ST-1, untouched by the prefetch)
    constexpr int CPR = BN / 16;                       // 16-byte chunks per staged int8 row
    constexpr int SWM = (CPR % 8 == 0) ? 7 : 0;        // chunk XOR swizzle by row (conflict-free 2-byte stores)
#pragma unroll
    for (int j = 0; j < NI / 2; ++j) {
      const int cl = wn * TN + j * 16;                 // tile column of this 16-column group
      const int col = n0 + cl + 2 * q4;                // this lane: col, col+1 (ni = 2j), col+8, col+9 (ni = 2j+1)
      const bool ok = n0 + cl < N;                     // N % 16 == 0: a group is wholly in or out
      const float2 b0 = ok ? __half22float2(*(const half2*)(bias + col)) : make_float2(0.f, 0.f);
      const float2 b8 = ok ? __half22float2(*(const half2*)(bias + col + 8)) : make_float2(0.f, 0.f);
      const float w0 = ok ? sw[col] : 0.f, w1 = ok ? sw[col + 1] : 0.f;
      const float w8 = ok ? sw[col + 8] : 0.f, w9 = ok ? sw[col + 9] : 0.f;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int rowl = wm * TM + mi * 16 + g;
        const float s0 = sa[m0 + rowl], s1 = sa[m0 + rowl + 8];
        const int32_t* A0 = acc[mi][2 * j];
        const int32_t* A1 = acc[mi][2 * j + 1];
        const float v0 = fmaf((float)A0[0], s0 * w0, b0.x), v1 = fmaf((float)A0[1], s0 * w1, b0.y);
        const float v2 = fmaf((float)A1[0], s0 * w8, b8.x), v3 = fmaf((float)A1[1], s0 * w9, b8.y);
        const float v4 = fmaf((float)A0[2], s1 * w0, b0.x), v5 = fmaf((float)A0[3], s1 * w1, b0.y);
        const float v6 = fmaf((float)A1[2], s1 * w8, b8.x), v7 = fmaf((float)A1[3], s1 * w9, b8.y);
        float ma = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        float mb = fmaxf(fmaxf(fabsf(v4), fabsf(v5)), fmaxf(fabsf(v6), fabsf(v7)));
        ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 1)); ma = fmaxf(ma, __shfl_xor_sync(0xffffffff, ma, 2));
        mb = fmaxf(mb, __shfl_xor_sync(0xffffffff, mb, 1)); mb = fmaxf(mb, __shfl_xor_sync(0xffffffff, mb, 2));
        const half ha = __float2half_rn(fmaxf(ma * (1.f / 127.f), 6.103515625e-05f));
        const half hb = __float2half_rn(fmaxf(mb * (1.f / 127.f), 6.103515625e-05f));
        float ia, ib;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ia) : "f"(__half2float(ha)));
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ib) : "f"(__half2float(hb)));
        constexpr float MAG = 12582912.f;
        const uint32_t qa0 = __byte_perm(__float_as_uint(fmaf(v0, ia, MAG)), __float_as_uint(fmaf(v1, ia, MAG)), 0x0040);
        const uint32_t qa8 = __byte_perm(__float_as_uint(fmaf(v2, ia, MAG)), __float_as_uint(fmaf(v3, ia, MAG)), 0x0040);
        const uint32_t qb0 = __byte_perm(__float_as_uint(fmaf(v4, ib, MAG)), __float_as_uint(fmaf(v5, ib, MAG)), 0x0040);
        const uint32_t qb8 = __byte_perm(__float_as_uint(fmaf(v6, ib, MAG)), __float_as_uint(fmaf(v7, ib, MAG)), 0x0040);
        int8_t* ra = C8s + rowl * BN + (((cl >> 4) ^ (rowl & SWM)) << 4) + 2 * q4;
        int8_t* rb = C8s + (rowl + 8) * BN + (((cl >> 4) ^ ((rowl + 8) & SWM)) << 4) + 2 * q4;
        *(uint16_t*)ra = (uint16_t)qa0; *(uint16_t*)(ra + 8) = (uint16_t)qa8;
        *(uint16_t*)rb = (uint16_t)qb0; *(uint16_t*)(rb + 8) = (uint16_t)qb8;
        if (q4 == 0 && ok) {
          half* sp = S16 + (size_t)((n0 + cl) >> 4) * sa_stride + m0 + rowl;
          sp[0] = ha; sp[8] = hb;
        }
      }
    }
    __syncthreads();
    constexpr int CCH8 = BM * CPR;
#pragma unroll 4
    for (int i = tid; i < CCH8; i += NT) {
      const int r = i / CPR, c = i - r * CPR;
      if (n0 + c * 16 < N)
        *(uint4*)(C8 + (size_t)(m0 + r) * N + n0 + c * 16) = *(const uint4*)(C8s + r * BN + ((c ^ (r & SWM)) << 4));
    }
    __syncthreads();                                   // C8s read out before the next tile's stage ST-1 load
  }
#endif
}

// ------------------------------------------------------------------------------------------------ fc1 (DHD)
// GELU -> fp16 tile (Cs) -> per row H_64 (+ D1 / D2 when DIAGONALS) -> per-(row, 128) scale -> packed int4, the
// k_gemm_s8_dhd <GELU, Q8OUT, ROT8 = 64, I4> epilogue verbatim apart from i4p::q4code.  BN = 128 (one 128-column
// group per tile); Cs spans ring stages 1..ST-1, so ST = 3 (stage = 16 KB, Cs = 32 KB) and stage 0 takes the
// prefetch; the diagonals live in static shared memory.
// F2S (fc2static_engine_20261016), as in k_gemm_s8_dhd: bit 0 = static fc2-input scale per (layer, 128-column group)
// (inv = sinv[bx], no row maximum, no sa2 store); bit 1 = TIMING ONLY, no 64-point butterfly; bit 2 = TIMING ONLY, no D1
// multiply / staging; bit 3 = no D2 multiply / staging (exact when d2 = 1).  F2S = 0 is unchanged.
template<int BM, int WM, int WN, int ST, bool DB, int MINB, bool DIAGONALS, int F2S = 0>
__global__ __launch_bounds__(WM * WN * 32, MINB) void k_fc1dhd4p(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bt, const float* __restrict__ sa,
    const float* __restrict__ sw, const half* __restrict__ bias, int8_t* __restrict__ C8, float* __restrict__ sa2,
    int M, int N, int K, int ldb, const float* __restrict__ d1, const float* __restrict__ d2, float clip, int ntn, int ntiles,
    const float* __restrict__ sinv = nullptr) {
#if S8_ENABLED
  constexpr int BN = 128;
  I4P_COMMON(BM, BN, WM, WN, ST)
  constexpr int LDC = BN;
  static_assert(ST == 3 && (size_t)BM * LDC * 2 <= (size_t)(ST - 1) * STAGE, "Cs must fit ring stages 1..ST-1");
  static_assert(BM % 16 == 0 && NT % 32 == 0, "ROT8 tile");
  auto swzC = [](int r, int c) { return c ^ (r & 7); };
  half* Cs = (half*)(smem_raw + STAGE);
  __shared__ __align__(16) float Ds[DIAGONALS ? 2 * BN : 4];
  int t = blockIdx.x;
  if (t >= ntiles) return;
  set_tile((t / ntn) * BM, (t % ntn) * BN);
  _Pragma("unroll") for (int s = 0; s < ST - 1; ++s) { if (s < ktiles) load_tile(s, s); cp_commit(); }
  bool primed = true;                                  // stages 0..ST-2 of this tile are committed
  for (; t < ntiles; t += gridDim.x) {
    const int by = t / ntn, bx = t - by * ntn;
    const int m0 = by * BM, n0 = bx * BN;
    if (!primed) {                                     // the prefetch covered stage 0 only (stage 1 held Cs)
#pragma unroll
      for (int s = 1; s < ST - 1; ++s) { if (s < ktiles) load_tile(s, s); cp_commit(); }
    }
    int32_t acc[MI][NI][4];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
      for (int ni = 0; ni < NI; ++ni) acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
    I4P_MAINLOOP()
    __syncthreads();                                   // every warp is done with the ring
    primed = false;
    {
      const int tn = t + gridDim.x;                    // prefetch the next tile's k-tile 0 into stage 0
      if (tn < ntiles) {
        set_tile((tn / ntn) * BM, (tn % ntn) * BN);
        load_tile(0, 0); cp_commit();
      }
    }
    // ---- epilogue: k_gemm_s8_dhd <GELU, Q8OUT, ROT8 = 64, I4> verbatim
    if constexpr (DIAGONALS && !(F2S & 12)) {
      for (int i = tid; i < 2 * BN; i += NT) {
        const int col = n0 + (i < BN ? i : i - BN);
        Ds[i] = col < N ? (i < BN ? d1[col] : d2[col]) : 0.f;      // the 64-wide remainder of N = 576: unused
      }
    } else if constexpr (DIAGONALS && (F2S & 12) != 12) {         // F2S bit 2 / 3: stage only the diagonal still applied
      constexpr int lo = (F2S & 4) ? BN : 0, hi = (F2S & 8) ? BN : 2 * BN;
      for (int i = lo + tid; i < hi; i += NT) {
        const int col = n0 + (i < BN ? i : i - BN);
        Ds[i] = col < N ? (i < BN ? d1[col] : d2[col]) : 0.f;
      }
    }
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int cch = wn * TN / 8 + ni;
      const int coll = swzC(g, cch) * 8 + 2 * q4;
      const int col = n0 + cch * 8 + 2 * q4;
      const float2 bb = col < N ? __half22float2(*(const half2*)(bias + col)) : make_float2(0.f, 0.f);
      const float sw0 = col < N ? sw[col] : 0.f, sw1 = col + 1 < N ? sw[col + 1] : 0.f;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int rowl = wm * TM + mi * 16 + g;
        const float s0 = sa[m0 + rowl];
        const float s1 = sa[m0 + rowl + 8];
        const float a0 = (float)acc[mi][ni][0];
        const float a1 = (float)acc[mi][ni][1];
        const float a2 = (float)acc[mi][ni][2];
        const float a3 = (float)acc[mi][ni][3];
        float v0 = fmaf(a0, s0 * sw0, bb.x), v1 = fmaf(a1, s0 * sw1, bb.y);
        float v2 = fmaf(a2, s1 * sw0, bb.x), v3 = fmaf(a3, s1 * sw1, bb.y);
        v0 = gelu(v0); v1 = gelu(v1); v2 = gelu(v2); v3 = gelu(v3);
        *(uint32_t*)(Cs + rowl * LDC + coll) = pack2(v0, v1);
        *(uint32_t*)(Cs + (rowl + 8) * LDC + coll) = pack2(v2, v3);
      }
    }
    __syncthreads();
    {
      constexpr int RPP = NT / 2, PASSES = (BM + RPP - 1) / RPP;   // rows per pass, passes
      const int hh = tid & 1;
      const bool full = n0 + BN <= N;             // else this is the 64-wide remainder of N = 576
      for (int r = tid >> 1, ps = 0; PASSES > 1 ? r < BM : ps < 1; r += RPP, ++ps) {
      if (BM % RPP == 0 || r < BM) {              // warp-uniform: a pass is whole warps
      float v[64];
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int c = hh * 8 + i;                 // 8-column chunk of this row
        const uint4 u = *(const uint4*)(Cs + r * LDC + swzC(r, c) * 8);
        const half2* h2 = (const half2*)&u;
        float dd[8];                              // d1 of this chunk's 8 columns
        if constexpr (DIAGONALS && !(F2S & 4)) {
          const float4* dv = (const float4*)(Ds + hh * 64 + i * 8);
          const float4 da = dv[0], db = dv[1];
          dd[0] = da.x; dd[1] = da.y; dd[2] = da.z; dd[3] = da.w; dd[4] = db.x; dd[5] = db.y; dd[6] = db.z; dd[7] = db.w;
        }
#pragma unroll
        for (int k = 0; k < 4; ++k) {
          const float2 f = __half22float2(h2[k]);
          // __fmul_rn: the D1 product is rounded on its own, as in the frozen binary (see dhd_gemm.cuh)
          if constexpr(DIAGONALS && !(F2S & 4)){v[i*8+2*k] = __fmul_rn(f.x,dd[2*k]);v[i*8+2*k+1] = __fmul_rn(f.y,dd[2*k+1]);}else{v[i*8+2*k]=f.x;v[i*8+2*k+1]=f.y;}
        }
      }
      if constexpr (!(F2S & 2)) {
#pragma unroll
      for (int st = 1; st < 64; st <<= 1)
#pragma unroll
        for (int i = 0; i < 64; ++i)
          if ((i & st) == 0) { const float a = v[i], b2 = v[i | st]; v[i] = a + b2; v[i | st] = a - b2; }
      }
      // the Hadamard's 1/sqrt(n) is folded into the scale: the values stay un-normalised
      const float hn = rsqrtf(64.f);
      float mx = 0.f;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        if constexpr (DIAGONALS && !(F2S & 8)) {
          const float4 d = ((const float4*)(Ds + BN + hh * 64))[i];
          v[4 * i] *= d.x; v[4 * i + 1] *= d.y; v[4 * i + 2] *= d.z; v[4 * i + 3] *= d.w;
        }
        if constexpr (!(F2S & 1)) {
#pragma unroll
        for (int j = 0; j < 4; ++j) mx = fmaxf(mx, fabsf(v[4 * i + j]));
        }
      }
      float inv;
      if constexpr (F2S & 1) {
        inv = sinv[bx];                           // static: (1/8) / s_g of this 128-column group, no row maximum
      } else {
      // the per-row scale spans the whole 128-column group either way
      if (full) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
      const float sc = mx * hn * clip * (1.f / 7.f) + 1e-12f;
      inv = hn / sc;
      if (hh == 0) sa2[(size_t)(m0 + r) * ntn + bx] = sc;
      }
      if (full || hh == 0) {
        // 64 codes -7..7 -> 32 bytes, element 2j in the low nibble of byte j
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint32_t u = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j) u |= i4p::q4code(v[8 * i + j] * inv) << (4 * j);
          w[i] = u;
        }
        uint4* dst = (uint4*)(C8 + (size_t)(m0 + r) * (N / 2) + (n0 + hh * 64) / 2);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
      }
      }
      }
    }
    __syncthreads();                                   // Cs and Ds read out before the next tile reuses them
  }
#endif
}
#undef I4P_COMMON
#undef I4P_MAINLOOP
