// sam21_gemm_s8.cuh -- INT8 GEMM with a fused dequant + bias (+ optional GELU) epilogue.
//
//   C[M,N] = [gelu](sa[m] * sw[n] * (A_s8[M,K] @ Bt_s8[N,K]^T) + bias[N])   -> fp16
//
// Structure follows k_gemm_gelu (cp.async ring, XOR-swizzled unpadded shared tiles,
// ldmatrix, smem-staged coalesced stores) with mma.m16n8k32.s8 and the weight stored
// transposed to [N,K] so both operands are read with a non-transposed ldmatrix.
// sa is a per-row (per-token) activation scale, sw a per-column (per-output-channel)
// weight scale; both are folded into the epilogue, so nothing but fp16 leaves the kernel.
#pragma once
#include <cstdint>
#include "sam21_gemm.cuh"      // gemmk:: helpers (cp.async, ldmatrix, gelu, pack2)

// mma.m16n8k32.s8 and the cp.async/ldmatrix helpers all need sm_80; on an older device
// pass the kernel compiles to an empty stub and the engine refuses --int8 there.
#define S8_ENABLED GEMMK_ENABLED

namespace s8k {
#if S8_ENABLED
using namespace gemmk;
__device__ __forceinline__ void mma16832_s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// unsigned activation x signed weight.  A GELU output is bounded below by -0.169971, so
// shifting it up by that much makes it non-negative and lets the activation use all 256
// codes instead of the 255 symmetric ones -- the quantisation step halves.  The shift is
// undone by a per-output-channel constant folded into the bias, so it costs nothing here.
__device__ __forceinline__ void mma16832_u8s8(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// INT4 (--int4): mma.m16n8k64 .s4.s4.s32.  Per thread the A fragment is four .b32 registers of
// eight s4 each (a0..a31, row g for a0-7 / a16-23 and g+8 for a8-15 / a24-31, k = 8*tig + (i&7)
// [+32 for i >= 16]), B two .b32 of eight s4 (k = 8*tig + (i&7) [+32 for i >= 8], n = g), i.e.
// in BYTES of a K-packed row (two s4 per byte) exactly the fragment of m16n8k32 .s8 (four s8 per
// register, k = 4*tig + (i&3) [+16]).  So the s8 kernel's cp.async / swizzle / ldmatrix run
// unchanged on packed int4 rows with K counted in bytes (K/2), and only the mma changes.
// Packing: element 2j in the LOW nibble of byte j, 2j+1 in the high nibble (two's complement).
// (A consistent nibble order is all the GEMM needs: swapping the two nibbles of every byte in
// both operands is a permutation of K and leaves the dot product unchanged.)
__device__ __forceinline__ void mma16864_s4(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// zero-C forms (D = A.B, the accumulator input is RZ): the first k-step of a scale group under FF, which then
// needs no register zeroing after the flush.
__device__ __forceinline__ void mma16864_s4_z(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ void mma16832_s8_z(int32_t* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
               : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
// exact int32 -> fp32 for |x| < 2^22 without the quarter-rate I2F: 1.5 * 2^23 + x has a unit ulp
__device__ __forceinline__ float i2f_magic(int32_t x) { return __int_as_float(x + 0x4B400000) - 12582912.f; }
__device__ __forceinline__ constexpr float gelu_min() { return -0.169971f; }
__device__ __forceinline__ int cvt_s8_sat(float x) {
  int r; asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(x)); return r;
}
// --- block floating point (--fc2-bfp) -------------------------------------------------
// The fc2 activation scale of a (row, 128-column group) is constrained to a power of two,
// 2^p with p = ceil(log2(absmax/127)), so that the int8 GEMM can carry it by moving the
// binary point of its int32 accumulator instead of flushing it into a second fp32
// accumulator set (which costs 64 registers per thread, 2 -> 1 blocks/SM, +38 %).
// The exponent is stored as one byte per (row, group), biased by 128.
constexpr int BFP_PMIN = -100;      // exponent of an all-zero group (byte 28)
constexpr int BFP_BIAS = 128;
// smallest power of two >= x, as its exponent; exact, 3 integer ops, no MUFU.
__device__ __forceinline__ int bfp_exp_up(float x) {
  const uint32_t u = __float_as_uint(x);
  const int e = (int)((u >> 23) & 0xffu) - 127 + ((u & 0x7fffffu) ? 1 : 0);
  return (u == 0u || e < BFP_PMIN) ? BFP_PMIN : e;
}
__device__ __forceinline__ float bfp_pow2(int p) {      // 2^p for p in [-126, 127]
  return __int_as_float((uint32_t)(p + 127) << 23);
}
#endif
}

// BK in int8 elements (64 = two k32 steps; a 64-byte row, same swizzle as the fp16 BK=32).
// SASCALAR: the activation scale is one number for the whole tensor (sa[0]) instead of one
//           per token -- what proj gets, because its input is produced tile-wise by the
//           attention kernel and no producer there sees a whole row.
// RESID:    add a residual tensor in the epilogue (proj carries the block's shortcut, which
//           cuBLASLt used to do with beta=1).
// GROUP > 0: the activation scale varies along the reduction axis with that period, so the
//   int32 accumulator is flushed into an fp32 one once per group.  sa is then [M][K/GROUP].
//   This is what lets fc2 be quantised at all: its input is produced by the fc1 tile, which
//   sees 128 columns of a row -- exactly one group -- and so can emit a scale for it.
// Q8OUT: write the result as int8 with a per-(row, column-tile) scale instead of fp16, which
//   is the producer half of the same trick.  C8 gets the values, sa2 the [M][gridDim.x] scales.
// ROT8: before that quantisation, rotate the tile's columns by a block Hadamard of that width
//   (64 or 128; the 64-wide remainder tile of N = 576 always gets H_64), the block-diagonal
//   rotation fc2's weight was rotated by at start-up (k_rot_w<1>).  The rotated GELU output
//   has no sign bound, so the quantiser is the signed one, and the per-row scale is taken
//   over the rotated values.
// BFP: the K-varying scale is a power of two per (row, GROUP) carried as a biased byte in
//   `ex` instead of the fp32 array `sa`.  On the producer (Q8OUT) it is the scale the
//   epilogue quantises with; on the consumer (GROUP > 0) the group boundary becomes
//   acc = (acc >> v) << u with u/v the exponent difference of the two groups, so the int32
//   accumulator never leaves the register file and the fp32 shadow set disappears.
// QKV8 (the qkv GEMM under --qk8): the column tiles left of qkn (Q and K, whole heads of
//   BN/NGPT columns) are written as int8 into C8 [M, qkn] with one scale per (row, head) at
//   sa2[head * sa_stride + row], the free axis of the attention's QK^T; the tiles right of
//   qkn (V) are the usual fp16 into C.  Block-uniform, so a V tile runs the plain epilogue.
// W8G16 (the qkv GEMM under --win8): EVERY column is written as int8 into C8 [M, N] with one fp16
//   scale per (row, 16 columns) at ((half*)sa2)[(col / 16) * sa_stride + row]; nothing goes to C.
//   The attention dequantises before its fp16 tensor-core math, so the scale may vary along the
//   head dimension, and 16 divides every padded head (64 / 80 / 96) and every warp's column band:
//   the group max of 16 columns is the 4 values of two n8 fragments reduced over the lane quad
//   (two shuffles), with no shared-memory reduction and no constraint on the tile width.  The
//   scale is max(absmax / 127, 2^-14) rounded to fp16 (never subnormal, never zero) and the codes
//   are rounded (to nearest even) from the fp32 epilogue values against that rounded scale.
// I4 (--int4): both operands are packed int4 (see mma16864_s4), K is passed in BYTES (= K/2
//   elements), BK is in bytes (BK = 64 -> 128 int4 per k-tile), GROUP stays in ELEMENTS (128 =
//   64 bytes).  The group flush runs per mma k-step (32 bytes = 64 int4) so that BK = 128 bytes
//   (two groups per k-tile) works too.  With Q8OUT + ROT8 the producer emits packed int4 codes
//   -7..7 with scale = absmax / 7 into C8 [M, N/2]; every other epilogue is the int8 one.
// FF (fast flush, INT8 or INT4, GROUP > 0 only): the per-group flush runs inside the k-step loop, converts the
//   int32 group sum with i2f_magic (IADD + FADD, exact: |group sum| <= 128 * 127^2 < 2^22) instead of I2F, which
//   issues at 16/clk/SM on GA102 / AD102, and the next group starts with a zero-C mma instead of register
//   zeroing.  Bit-identical to the plain flush: (float)acc is exact either way and the FFMA is the same.
// F2S (samv1_static_fc2_20261026; producer Q8OUT + ROT8 only): STATIC fc2 input scale.  sa2 is then an INPUT, one float
//   s of this layer (sa2[0]); the codes are clamp(round(v * (hn / s)), -q, q), no row maximum, no shuffle, no scale
//   store.  The consumer is the plain GROUP = 0 GEMM with SASCALAR (sa[0] = the same s).  F2S = false: unchanged.
template<int BM, int BN, int BK, int WM, int WN, int STAGES, bool DB, bool GELU,
         bool SASCALAR = false, bool RESID = false, int GROUP = 0, bool Q8OUT = false,
         int NGPT = 1, bool U8IN = false, bool U8OUT = false, int ROT8 = 0, int MINB = 0,
         bool BFP = false, bool QKV8 = false, bool W8G16 = false, bool I4 = false, bool FF = false, bool F2S = false>
__global__ __launch_bounds__(WM * WN * 32, MINB) void k_gemm_s8(
    const int8_t* __restrict__ A, const int8_t* __restrict__ Bt, const float* __restrict__ sa,
    const float* __restrict__ sw, const half* __restrict__ bias, half* __restrict__ C,
    int M, int N, int K, const half* __restrict__ resid = nullptr,
    int8_t* __restrict__ C8 = nullptr, float* __restrict__ sa2 = nullptr, int sa_stride = 0,
    uint8_t* __restrict__ ex = nullptr, int qkn = 0) {
#if S8_ENABLED
  using namespace s8k;
  using namespace gemmk;
  constexpr int NT = WM * WN * 32;
  constexpr int A_STAGE = BM * BK, B_STAGE = BN * BK;      // bytes
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MI = TM / 16, NI = TN / 8;
  constexpr int KC = BK / 16;                              // 16-byte chunks per row
  constexpr int RPL = 128 / BK;                            // rows per 128-byte line
  static_assert(BK % 32 == 0 && TM % 16 == 0 && TN % 16 == 0, "tile");
  auto swz = [](int r, int c) { return c ^ ((r / RPL) & (KC - 1)); };

  extern __shared__ __align__(16) char smem_raw[];
  char* As = smem_raw;
  char* Bs = As + STAGES * A_STAGE;
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp - wm * WN;
  const int g = lane >> 2, q4 = lane & 3;

  constexpr int ACH = BM * KC, BCH = BN * KC;
  constexpr int APT = (ACH + NT - 1) / NT, BPT = (BCH + NT - 1) / NT;
  const int8_t* asrc[APT]; uint32_t adst[APT]; int akc[APT];
  const int8_t* bsrc[BPT]; uint32_t bdst[BPT]; int bkc[BPT]; bool bok[BPT];
#pragma unroll
  for (int j = 0; j < APT; ++j) {
    int i = tid + j * NT; if (i >= ACH) i = ACH - 1;
    int r = i / KC, c = i - r * KC;
    asrc[j] = A + (size_t)(m0 + r) * K + c * 16;
    adst[j] = smem_u32(As + r * BK + swz(r, c) * 16);
    akc[j] = c * 16;
  }
#pragma unroll
  for (int j = 0; j < BPT; ++j) {
    int i = tid + j * NT; if (i >= BCH) i = BCH - 1;
    int r = i / KC, c = i - r * KC;
    bok[j] = n0 + r < N;
    bsrc[j] = Bt + (size_t)(bok[j] ? n0 + r : 0) * K + c * 16;
    bdst[j] = smem_u32(Bs + r * BK + swz(r, c) * 16);
    bkc[j] = c * 16;
  }
  auto load_tile = [&](int stage, int kt) {
    const int k0 = kt * BK;
    const uint32_t ao = stage * A_STAGE, bo = stage * B_STAGE;
#pragma unroll
    for (int j = 0; j < APT; ++j)
      if (ACH % NT == 0 || tid + j * NT < ACH) {
        bool ok = k0 + akc[j] < K;
        cp_async16z(adst[j] + ao, asrc[j] + (ok ? k0 : 0), ok ? 16 : 0);
      }
#pragma unroll
    for (int j = 0; j < BPT; ++j)
      if (BCH % NT == 0 || tid + j * NT < BCH) {
        bool ok = bok[j] && (k0 + bkc[j] < K);
        cp_async16z(bdst[j] + bo, bsrc[j] + (ok ? k0 : 0), ok ? 16 : 0);
      }
  };

  constexpr bool FGRP = GROUP && !BFP;               // the fp32-flush group path
  int32_t acc[MI][NI][4];
  float facc[FGRP ? MI : 1][FGRP ? NI : 1][4];
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
      if (FGRP) facc[mi][ni][0] = facc[mi][ni][1] = facc[mi][ni][2] = facc[mi][ni][3] = 0.f;
    }

  const int ktiles = (K + BK - 1) / BK;
  constexpr int KTPG = GROUP ? (GROUP / BK) : 1;      // k-tiles per scale group
  // the row stride of sa is the producer's (tiles x groups-per-tile), which is not
  // ceil(K/GROUP) when the producer's last column tile is partial (N = 576 with BN = 128).
  const int NG = GROUP ? sa_stride : 1;
  // BFP: the accumulator's own binary point, i.e. the exponent of the group it is currently
  // accumulating, for the two rows (rowl, rowl + 8) this thread owns per mi.
  constexpr int EMI = (GROUP && BFP) ? MI : 1;
  int pcur[EMI][2];
  if constexpr (GROUP && BFP) {
#pragma unroll
    for (int mi = 0; mi < EMI; ++mi) {
      const int rowl = wm * TM + mi * 16 + g;
      pcur[mi][0] = (int)__ldg(ex + (size_t)(m0 + rowl) * NG) - s8k::BFP_BIAS;
      pcur[mi][1] = (int)__ldg(ex + (size_t)(m0 + rowl + 8) * NG) - s8k::BFP_BIAS;
    }
  }
  constexpr int KS = BK / 32;
  static_assert(!I4 || (!U8IN && !U8OUT && !BFP && !QKV8 && !W8G16), "I4: signed int4 operands, fp32 group flush only");
  static_assert(!I4 || !Q8OUT || ROT8, "I4 producer = the rotated fc1 epilogue");
  constexpr int KSPG = GROUP ? GROUP / (I4 ? 64 : 32) : 1;   // mma k-steps (64 int4 / 32 int8) per scale group
  static_assert(!(I4 && GROUP) || (GROUP % 64 == 0), "I4 group = whole k64 steps");
  static_assert(!FF || (GROUP && !BFP && !U8IN), "FF = the fp32 group flush of signed operands");
  constexpr bool INLOOP = FGRP && (I4 || FF);                // flush inside the k-step loop
  constexpr bool ZSTATIC = (KS % KSPG) == 0;                 // group starts at a compile-time ks
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < ktiles) load_tile(s, s); cp_commit(); }

  // A: lanes 0-7 rows 0-7 k0-15, 8-15 rows 8-15 k0-15, 16-23 rows 0-7 k16-31, 24-31 rows 8-15 k16-31
  const int alr = (lane & 7) + ((lane >> 3) & 1) * 8, alc = (lane >> 4);
  // B (n rows): lanes 0-7 rows 0-7 k0-15, 8-15 rows 0-7 k16-31, 16-23 rows 8-15 k0-15, 24-31 rows 8-15 k16-31
  const int blr = (lane & 7) + ((lane >> 4) & 1) * 8, blc = (lane >> 3) & 1;
  const uint32_t a_row = smem_u32(As) + (uint32_t)((wm * TM + alr) * BK);
  const uint32_t b_row = smem_u32(Bs) + (uint32_t)((wn * TN + blr) * BK);
  uint32_t aoff[KS], boff[KS];
#pragma unroll
  for (int ks = 0; ks < KS; ++ks) {
    aoff[ks] = swz(wm * TM + alr, ks * 2 + alc) * 16;
    boff[ks] = swz(wn * TN + blr, ks * 2 + blc) * 16;
  }
  constexpr int NB = DB ? 2 : 1;
  uint32_t af[NB][MI][4], bf[NB][NI][2];
  auto load_frags = [&](int buf, int stage, int ks) {
    const uint32_t at = a_row + stage * A_STAGE + aoff[ks];
    const uint32_t bt = b_row + stage * B_STAGE + boff[ks];
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) ldsm_x4(af[buf][mi], at + mi * 16 * BK);
#pragma unroll
    for (int n2 = 0; n2 < NI / 2; ++n2) {
      uint32_t t[4];
      ldsm_x4(t, bt + n2 * 16 * BK);
      bf[buf][2 * n2][0] = t[0]; bf[buf][2 * n2][1] = t[1];
      bf[buf][2 * n2 + 1][0] = t[2]; bf[buf][2 * n2 + 1][1] = t[3];
    }
  };
  cp_wait<STAGES - 2>(); __syncthreads();
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
          cp_wait<STAGES - 2>(); __syncthreads();
          if (kt + STAGES < ktiles) load_tile(st, kt + STAGES);
          cp_commit();
          if (DB) load_frags(cur ^ 1, (kt + 1) % STAGES, 0);
        }
      } else if (DB) load_frags(cur ^ 1, st, ks + 1);
      const bool gstart = FF && (ZSTATIC ? (ks % KSPG == 0) : ((kt * KS + ks) % KSPG == 0));
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          if (FF && gstart) {
            if (I4) mma16864_s4_z(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
            else    mma16832_s8_z(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
          }
          else if (I4)   mma16864_s4(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
          else if (U8IN) mma16832_u8s8(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
          else           mma16832_s8(acc[mi][ni], af[cur][mi], bf[cur][ni][0], bf[cur][ni][1]);
        }
      if constexpr (INLOOP) {
        // group boundary after this k-step (k-step index kt*KS + ks, KSPG steps per group)
        const int kstep = kt * KS + ks;
        if ((kstep + 1) % KSPG == 0 || (kt == ktiles - 1 && ks == KS - 1)) {
          const int gi = kstep / KSPG;
#pragma unroll
          for (int mi = 0; mi < MI; ++mi) {
            const int rowl = wm * TM + mi * 16 + g;
            const float s0 = sa[(size_t)(m0 + rowl) * NG + gi];
            const float s1 = sa[(size_t)(m0 + rowl + 8) * NG + gi];
#pragma unroll
            for (int ni = 0; ni < NI; ++ni) {
              if (FF) {
                facc[mi][ni][0] = fmaf(i2f_magic(acc[mi][ni][0]), s0, facc[mi][ni][0]);
                facc[mi][ni][1] = fmaf(i2f_magic(acc[mi][ni][1]), s0, facc[mi][ni][1]);
                facc[mi][ni][2] = fmaf(i2f_magic(acc[mi][ni][2]), s1, facc[mi][ni][2]);
                facc[mi][ni][3] = fmaf(i2f_magic(acc[mi][ni][3]), s1, facc[mi][ni][3]);
              } else {
                facc[mi][ni][0] += (float)acc[mi][ni][0] * s0;
                facc[mi][ni][1] += (float)acc[mi][ni][1] * s0;
                facc[mi][ni][2] += (float)acc[mi][ni][2] * s1;
                facc[mi][ni][3] += (float)acc[mi][ni][3] * s1;
                acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
              }
            }
          }
        }
      }
    }
    if constexpr (INLOOP) {
      // (the group flush ran inside the k-step loop above)
    } else if constexpr (GROUP && BFP) {
      // the accumulator stays int32 and keeps the current group's units; at a boundary it
      // is rescaled by the DIFFERENCE of the two exponents only (two integer ops per
      // register, truncating), and the last group's 2^p is applied by the epilogue.
      if ((kt + 1) % KTPG == 0 && kt != ktiles - 1) {
        const int gi = (kt + 1) / KTPG;
        int u[MI][2], v[MI][2];
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) {
          const int rowl = wm * TM + mi * 16 + g;
          const int p0 = (int)__ldg(ex + (size_t)(m0 + rowl) * NG + gi) - s8k::BFP_BIAS;
          const int p1 = (int)__ldg(ex + (size_t)(m0 + rowl + 8) * NG + gi) - s8k::BFP_BIAS;
          u[mi][0] = max(pcur[mi][0] - p0, 0); v[mi][0] = max(p0 - pcur[mi][0], 0); pcur[mi][0] = p0;
          u[mi][1] = max(pcur[mi][1] - p1, 0); v[mi][1] = max(p1 - pcur[mi][1], 0); pcur[mi][1] = p1;
        }
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int ni = 0; ni < NI; ++ni) {
            acc[mi][ni][0] = (acc[mi][ni][0] >> v[mi][0]) << u[mi][0];
            acc[mi][ni][1] = (acc[mi][ni][1] >> v[mi][0]) << u[mi][0];
            acc[mi][ni][2] = (acc[mi][ni][2] >> v[mi][1]) << u[mi][1];
            acc[mi][ni][3] = (acc[mi][ni][3] >> v[mi][1]) << u[mi][1];
          }
      }
    } else if (FGRP && ((kt + 1) % KTPG == 0 || kt == ktiles - 1)) {
      const int gi = kt / KTPG;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        const int rowl = wm * TM + mi * 16 + g;
        const float s0 = sa[(size_t)(m0 + rowl) * NG + gi];
        const float s1 = sa[(size_t)(m0 + rowl + 8) * NG + gi];
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) {
          facc[mi][ni][0] += (float)acc[mi][ni][0] * s0;
          facc[mi][ni][1] += (float)acc[mi][ni][1] * s0;
          facc[mi][ni][2] += (float)acc[mi][ni][2] * s1;
          facc[mi][ni][3] += (float)acc[mi][ni][3] * s1;
          acc[mi][ni][0] = acc[mi][ni][1] = acc[mi][ni][2] = acc[mi][ni][3] = 0;
        }
      }
    }
  }

  if constexpr (W8G16) {
    static_assert(!GELU && !SASCALAR && !RESID && !GROUP && !Q8OUT && !U8IN && !U8OUT && !ROT8 && !BFP && !QKV8,
                  "W8G16 is the plain qkv GEMM");
    static_assert(TN % 16 == 0 && BN % 16 == 0, "16-column groups");
    static_assert((size_t)BM * BN <= (size_t)STAGES * (A_STAGE + B_STAGE), "int8 C tile must fit the ring");
    constexpr int CPR = BN / 16;                       // 16-byte chunks per staged int8 row
    constexpr int SWM = (CPR % 8 == 0) ? 7 : 0;        // chunk XOR swizzle by row (conflict-free 2-byte stores)
    int8_t* C8s = (int8_t*)smem_raw;
    half* S16 = (half*)sa2;
    __syncthreads();                                   // every warp is done with the ring
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
        // one MUFU.RCP (the scale is a normal fp16 >= 2^-14): an IEEE division costs ~10 instructions per
        // row half here, and a 1-ulp reciprocal can at most move a code that sits on a .5 boundary
        float ia, ib;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ia) : "f"(__half2float(ha)));
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(ib) : "f"(__half2float(hb)));
        // codes = round-to-nearest-even(v / scale) through the 1.5 * 2^23 magic: the FFMA rounds at a unit ulp and
        // the low byte of the result's bits is the two's-complement code (one FFMA + half a PRMT per value instead
        // of an F2I on the conversion pipe).  No saturation is needed: |v| <= absmax and the fp16 scale is at
        // least absmax/127 * (1 - 2^-11), so |v / scale| <= 127.07 rounds to at most 127.
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
    return;
  }

  // epilogue: dequant (per-row * per-col), + bias, GELU, fp16, staged through shared
  constexpr int LDC = BN;
  constexpr bool Q8ANY = Q8OUT || QKV8;           // the int8 epilogue exists in this instantiation
  const bool q8 = Q8OUT || (QKV8 && n0 < qkn);    // ... and this tile takes it (block-uniform)
  static_assert(!QKV8 || (!Q8OUT && !ROT8 && !BFP && !U8OUT && !GROUP && !GELU && !RESID), "QKV8 is the plain qkv GEMM");
  static_assert((size_t)BM * LDC * 2 + (Q8ANY ? BM * 4 * (NGPT + WN) : 0)
                <= (size_t)STAGES * (A_STAGE + B_STAGE), "C tile must fit the ring");
  static_assert(!Q8ANY || (WN % NGPT == 0 && (BN / 16) % NGPT == 0), "NGPT must divide the tile");
  auto swzC = [](int r, int c) { return c ^ (r & 7); };
  half* Cs = (half*)smem_raw;
  float* Rs = (float*)(Cs + BM * LDC);            // per-row scale of this tile (Q8OUT only)
  float* Rp = Rs + BM * NGPT;                     // its per-warp-column partials
  __syncthreads();
  float rmax[Q8ANY ? MI : 1][2];
  if (Q8ANY)
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) { rmax[mi][0] = 0.f; rmax[mi][1] = 0.f; }
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
      // BFP: the row scale is the last group's power of two (the accumulator's own units)
      const int emi = EMI > 1 ? mi : 0;              // keeps pcur in range when it is a stub
      const float s0 = FGRP ? 1.f : (GROUP ? s8k::bfp_pow2(pcur[emi][0])
                                           : (SASCALAR ? sa[0] : sa[m0 + rowl]));
      const float s1 = FGRP ? 1.f : (GROUP ? s8k::bfp_pow2(pcur[emi][1])
                                           : (SASCALAR ? sa[0] : sa[m0 + rowl + 8]));
      const float a0 = FGRP ? facc[mi][ni][0] : (float)acc[mi][ni][0];
      const float a1 = FGRP ? facc[mi][ni][1] : (float)acc[mi][ni][1];
      const float a2 = FGRP ? facc[mi][ni][2] : (float)acc[mi][ni][2];
      const float a3 = FGRP ? facc[mi][ni][3] : (float)acc[mi][ni][3];
      float v0 = fmaf(a0, s0 * sw0, bb.x), v1 = fmaf(a1, s0 * sw1, bb.y);
      float v2 = fmaf(a2, s1 * sw0, bb.x), v3 = fmaf(a3, s1 * sw1, bb.y);
      if (GELU) { v0 = gelu(v0); v1 = gelu(v1); v2 = gelu(v2); v3 = gelu(v3); }
      if (RESID && col < N) {
        const float2 r0 = __half22float2(*(const half2*)(resid + (size_t)(m0 + rowl) * N + col));
        const float2 r1 = __half22float2(*(const half2*)(resid + (size_t)(m0 + rowl + 8) * N + col));
        v0 += r0.x; v1 += r0.y; v2 += r1.x; v3 += r1.y;
      }
      if (Q8ANY && !ROT8 && q8) {
        // the row maximum comes straight out of the registers that just produced the values:
        // the 128 columns of a row live in 4 lanes (q4) of 4 warps (wn), so it is two shuffles
        // and one small shared reduction, not a second pass over the staged tile.
        if (U8OUT) {   // values are shifted non-negative; the scale spans [0, max] over 255 codes
          rmax[mi][0] = fmaxf(rmax[mi][0], fmaxf(v0, v1) - gelu_min());
          rmax[mi][1] = fmaxf(rmax[mi][1], fmaxf(v2, v3) - gelu_min());
        } else {
          rmax[mi][0] = fmaxf(rmax[mi][0], fmaxf(fabsf(v0), fabsf(v1)));
          rmax[mi][1] = fmaxf(rmax[mi][1], fmaxf(fabsf(v2), fabsf(v3)));
        }
      }
      *(uint32_t*)(Cs + rowl * LDC + coll) = pack2(v0, v1);
      *(uint32_t*)(Cs + (rowl + 8) * LDC + coll) = pack2(v2, v3);
    }
  }
  if (Q8ANY && !ROT8 && q8) {
#pragma unroll
    for (int mi = 0; mi < MI; ++mi) {
      float a = rmax[mi][0], b = rmax[mi][1];
      a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 1)); a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, 2));
      b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 1)); b = fmaxf(b, __shfl_xor_sync(0xffffffff, b, 2));
      if (q4 == 0) {
        const int rowl = wm * TM + mi * 16 + g;
        Rp[rowl * WN + wn] = a;
        Rp[(rowl + 8) * WN + wn] = b;
      }
    }
  }
  __syncthreads();
  if (Q8OUT && ROT8) {
    // two threads per row, NT/2 rows per pass: one pass for the shipped 128-row 256-thread tile,
    // two for the sm_89 4-warp tile; a BM below NT/2 idles whole warps (the guard is warp-uniform).
    static_assert(!ROT8 || (BM % 16 == 0 && BN == 128 && NT % 32 == 0 && NGPT == 1 && !U8OUT), "ROT8 tile");
    static_assert(ROT8 == 0 || ROT8 == 64 || ROT8 == 128, "ROT8 = block width");
    // two threads per row, each holding 64 consecutive columns of the staged tile in fp32.
    // ROT8 = 64: H_64 on each thread's own columns, no cross-thread traffic (what ships);
    // ROT8 = 128: one more butterfly stage across the thread pair (64 shuffles per thread,
    // measured +0.4 ms on the 3090 for no accuracy the decoder can see).
    constexpr int RPP = NT / 2, PASSES = (BM + RPP - 1) / RPP;   // rows per pass, passes
    const int hh = tid & 1;
    const bool full = n0 + BN <= N;             // else this is the 64-wide remainder of N = 576
    // (a real loop for the multi-pass tile, the plain guarded body for the single-pass one:
    // the two forms compile differently, and the single-pass form is the 3090 kernel as is)
    for (int r = tid >> 1, ps = 0; PASSES > 1 ? r < BM : ps < 1; r += RPP, ++ps) {
    if (BM % RPP == 0 || r < BM) {              // warp-uniform: a pass is whole warps
    float v[64];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int c = hh * 8 + i;                 // 8-column chunk of this row
      const uint4 u = *(const uint4*)(Cs + r * LDC + swzC(r, c) * 8);
      const half2* h2 = (const half2*)&u;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __half22float2(h2[k]);
        v[i * 8 + 2 * k] = f.x; v[i * 8 + 2 * k + 1] = f.y;
      }
    }
#pragma unroll
    for (int st = 1; st < 64; st <<= 1)
#pragma unroll
      for (int i = 0; i < 64; ++i)
        if ((i & st) == 0) { const float a = v[i], b2 = v[i | st]; v[i] = a + b2; v[i | st] = a - b2; }
    const bool pair = ROT8 == 128 && full;      // block-uniform
    if (pair) {
#pragma unroll
      for (int i = 0; i < 64; ++i) {
        const float o = __shfl_xor_sync(0xffffffff, v[i], 1);
        v[i] = hh ? (o - v[i]) : (v[i] + o);
      }
    }
    // the Hadamard's 1/sqrt(n) is folded into the scale: the values stay un-normalised
    const float hn = rsqrtf(pair ? 128.f : 64.f);
    static_assert(!F2S || (Q8OUT && ROT8 && !BFP), "F2S = the static-scale fc1 producer");
    float mx = 0.f;
    if constexpr (!F2S) {
#pragma unroll
    for (int i = 0; i < 64; ++i) mx = fmaxf(mx, fabsf(v[i]));
    // the per-row scale spans the whole 128-column group either way
    if (full) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
    }
    // BFP (--fc2-bfp): round the scale UP to a power of two and emit its exponent as a byte
    // instead of the fp32 scale, so that fc2 can carry it with an accumulator shift.  hn is
    // itself a power of two (2^-3 for H_64), so inv stays exact and the codes are the fp32
    // path's codes rescaled by one exponent.
    const int bp = BFP ? s8k::bfp_exp_up(mx * hn * (1.f / 127.f)) : 0;
    const float sc = F2S ? sa2[0] : (BFP ? s8k::bfp_pow2(bp) : (mx * hn * (I4 ? (1.f / 7.f) : (1.f / 127.f)) + 1e-12f));
    // a group at the exponent floor is DEFINED to be zero, so the guard can retag it
    // without touching its codes (plain ViTs have 2-6 % of such groups; hiera-L has none)
    const float inv = (BFP && bp == s8k::BFP_PMIN) ? 0.f : hn / sc;
    if (hh == 0 && !F2S) {
      if (BFP) ex[(size_t)(m0 + r) * gridDim.x + blockIdx.x] = (uint8_t)(bp + s8k::BFP_BIAS);
      else     sa2[(size_t)(m0 + r) * gridDim.x + blockIdx.x] = sc;
    }
    if (I4) {
      if (full || hh == 0) {
        // 64 codes -7..7 -> 32 bytes, element 2j in the low nibble of byte j
        uint32_t w[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          uint32_t u = 0;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            const int q = max(-7, min(7, __float2int_rn(v[8 * i + j] * inv)));
            u |= ((uint32_t)q & 0xfu) << (4 * j);
          }
          w[i] = u;
        }
        uint4* dst = (uint4*)(C8 + (size_t)(m0 + r) * (N / 2) + (n0 + hh * 64) / 2);
        dst[0] = make_uint4(w[0], w[1], w[2], w[3]);
        dst[1] = make_uint4(w[4], w[5], w[6], w[7]);
      }
    } else if (full || hh == 0) {
      uint32_t w[16];
#pragma unroll
      for (int i = 0; i < 16; ++i)
        w[i] = (uint32_t)(cvt_s8_sat(v[4 * i] * inv) & 0xff) | ((uint32_t)(cvt_s8_sat(v[4 * i + 1] * inv) & 0xff) << 8) |
               ((uint32_t)(cvt_s8_sat(v[4 * i + 2] * inv) & 0xff) << 16) | ((uint32_t)cvt_s8_sat(v[4 * i + 3] * inv) << 24);
      uint4* dst = (uint4*)(C8 + (size_t)(m0 + r) * N + n0 + hh * 64);
#pragma unroll
      for (int i = 0; i < 4; ++i) dst[i] = make_uint4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
    }
    }
    }
  } else if (Q8ANY && q8) {
    // warp column wn owns a contiguous TN-wide band, so NGPT groups per tile is just a
    // partition of the WN partials -- one scale per (row, BN/NGPT columns).
    constexpr int WPG = WN / NGPT;
    for (int i = tid; i < BM * NGPT; i += NT) {
      const int r = i / NGPT, gq = i - r * NGPT;
      float m = 0.f;
#pragma unroll
      for (int w = 0; w < WPG; ++w) m = fmaxf(m, Rp[r * WN + gq * WPG + w]);
      const int bp = BFP ? s8k::bfp_exp_up(m * (1.f / 127.f)) : 0;
      const float sc = BFP ? s8k::bfp_pow2(bp)
                           : (m * (U8OUT ? (1.f / 255.f) : (1.f / 127.f)) + 1e-12f);
      Rs[r * NGPT + gq] = 1.f / sc;
      if (QKV8)     sa2[(size_t)(blockIdx.x * NGPT + gq) * sa_stride + m0 + r] = sc;   // [head][row]
      else if (BFP) ex[((size_t)(m0 + r) * gridDim.x + blockIdx.x) * NGPT + gq] = (uint8_t)(bp + s8k::BFP_BIAS);
      else          sa2[((size_t)(m0 + r) * gridDim.x + blockIdx.x) * NGPT + gq] = sc;
    }
    __syncthreads();
  }
  if (Q8ANY && !ROT8 && q8) {
    const int ldc8 = QKV8 ? qkn : N;
    constexpr int CCH8 = BM * (BN / 16);
#pragma unroll 2
    for (int i = tid; i < CCH8; i += NT) {
      const int r = i / (BN / 16), c = i - r * (BN / 16);
      const float inv = Rs[r * NGPT + c / ((BN / 16) / NGPT)];
      int8_t q[16];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int cc = c * 16 + j * 2;
        float2 v = __half22float2(*(const half2*)(Cs + r * LDC + swzC(r, cc >> 3) * 8 + (cc & 7)));
        if (U8OUT) {
          q[j * 2]     = (int8_t)(uint8_t)max(0, min(255, __float2int_rn((v.x - gelu_min()) * inv)));
          q[j * 2 + 1] = (int8_t)(uint8_t)max(0, min(255, __float2int_rn((v.y - gelu_min()) * inv)));
        } else {
          q[j * 2]     = (int8_t)max(-127, min(127, __float2int_rn(v.x * inv)));
          q[j * 2 + 1] = (int8_t)max(-127, min(127, __float2int_rn(v.y * inv)));
        }
      }
      if (n0 + c * 16 < N) *(uint4*)(C8 + (size_t)(m0 + r) * ldc8 + n0 + c * 16) = *(const uint4*)q;
    }
  } else if (!q8) {
    constexpr int CCH = BM * (BN / 8);
#pragma unroll 4
    for (int i = tid; i < CCH; i += NT) {
      const int r = i / (BN / 8), c = i - r * (BN / 8);
      const int col = n0 + c * 8;
      if (col < N) *(uint4*)(C + (size_t)(m0 + r) * N + col) = *(const uint4*)(Cs + r * LDC + swzC(r, c) * 8);
    }
  }
#endif  // S8_ENABLED
}

template<int BM, int BN, int BK, int STAGES>
static constexpr size_t s8_smem() { return (size_t)STAGES * (BM * BK + BN * BK); }

// --------------------------------------------------------------- BFP overflow guard
// The delta-shift accumulator is exact as long as a row's exponent spread stays inside the
// int32 headroom: |acc| <= sum_g (G * 127^2) * 2^(p_g - p_cur) <= NG * G * 127^2 * 2^SPREAD,
// so with G = 128 and NG = 36 (hiera-L stage 3, K = 4608) the bound is 2^26.15 * 2^SPREAD
// and SPREAD <= 4 is safe.  Nothing about the data guarantees that bound, so this kernel
// runs between fc1 and fc2 over the exponent array only (M x NG bytes, ~36 KB): it finds
// the rows whose spread exceeds EMAX and coarsens the offending groups -- their codes are
// shifted right by the clamp, which is exact arithmetic on a group already 2^EMAX below the
// row's largest, and re-tags them with the clamped exponent.  It also records what it saw.
__device__ unsigned long long g_bfp_viol = 0;      // (row, group) pairs clamped
__device__ int g_bfp_maxspread = 0;                // largest per-row exponent spread seen
__device__ int g_bfp_spread_ng[64];                // the same, per number of groups (per shape)
__device__ unsigned long long g_bfp_hist[64];      // rows by exponent spread (63 = 63 or more)
// One thread per row: NG is at most 36, so a warp per row wastes 31 lanes and 32x the
// blocks (stage 0 has 65536 rows).  The statistics are only collected when fix == 0
// (--fc2-bfp-check); on the working path a row that fits emax touches nothing.
__global__ void k_bfp_guard(uint8_t* __restrict__ ex, int8_t* __restrict__ C8, int M, int NG,
                            int K, int G, int emax, int fix) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  uint8_t* e = ex + (size_t)row * NG;
  int mx = 0, mn = 255;
  if (NG % 4 == 0) {   // four exponents per load, packed byte min/max (row*NG stays aligned)
    uint32_t m4 = 0, n4 = 0xffffffffu;
    const uint32_t* e4 = (const uint32_t*)e;
    for (int i = 0; i < NG / 4; ++i) { const uint32_t w = e4[i]; m4 = __vmaxu4(m4, w); n4 = __vminu4(n4, w); }
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      mx = max(mx, (int)((m4 >> (8 * b)) & 0xffu));
      mn = min(mn, (int)((n4 >> (8 * b)) & 0xffu));
    }
  } else {
    for (int g = 0; g < NG; ++g) { const int p = e[g]; mx = max(mx, p); mn = min(mn, p); }
  }
  if (!fix) {
    atomicMax(&g_bfp_maxspread, mx - mn);
    if (NG < 64) atomicMax(&g_bfp_spread_ng[NG], mx - mn);
    atomicAdd(&g_bfp_hist[min(mx - mn, 63)], 1ull);
  }
  if (mx - mn <= emax) return;
  const int floorp = mx - emax;
  for (int gi = 0; gi < NG; ++gi) {
    const int p = e[gi];
    if (p >= floorp) continue;
    if (!fix) { atomicAdd(&g_bfp_viol, 1ull); continue; }
    if (p == s8k::BFP_PMIN + s8k::BFP_BIAS) { e[gi] = (uint8_t)floorp; continue; }  // codes already 0
    const int s = floorp - p, n = min(gi * G + G, K) - gi * G;
    int8_t* q = C8 + (size_t)row * K + gi * G;
    if (s >= 8) {                     // below one code of the clamped scale: the group is 0
      for (int i = 0; i < n / 16; ++i) ((uint4*)q)[i] = make_uint4(0, 0, 0, 0);
    } else {
      for (int i = 0; i < n / 4; ++i) {
        const uint32_t w = ((uint32_t*)q)[i];
        uint32_t r = 0;
#pragma unroll
        for (int b = 0; b < 4; ++b)
          r |= ((uint32_t)((((int)(int8_t)((w >> (8 * b)) & 0xffu)) >> s) & 0xff)) << (8 * b);
        ((uint32_t*)q)[i] = r;
      }
    }
    e[gi] = (uint8_t)floorp;
  }
}
