#pragma once
// sam21_gemm70b.cuh — fused fc1 GEMM + bias + GELU for sm_70 (Volta) on raw mma.m8n8k4.
//
//   C[M,N] = gelu(A[M,K] @ B[K,N] + bias[N])      A, B, C row-major fp16, bias fp16, fp32 acc
//
// Why not wmma (sam21_gemm70.cuh): micro/fc1bench's ablation on the V100 shows that kernel
// spends most of its time moving fragments out of shared memory -- with the mma removed
// entirely it still takes 0.157 ms on the 4096x2304x576 shape, more than cuBLASLt's whole
// GEMM (0.136).  wmma::load_matrix_sync hides the fragment layout and issues narrow, bank-
// conflicting loads from a padded tile.  Volta's native instruction, mma.sync.m8n8k4, has a
// layout that micro/mma884probe.cu measured directly:
//   * the warp is four quad-pairs (lanes {0-3,16-19}, {4-7,20-23}, {8-11,24-27}, {12-15,28-31});
//     each computes an independent 8x8x4 product;
//   * inside a quad-pair a lane has rank r = (lane & 3) + 4 * (lane >> 4);
//     A (.row): the lane holds row r, k = 0..3  -- 4 consecutive halves of a row;
//     B (.row): the lane holds k = r & 3, cols 4*(r >> 2) .. +3  -- 4 consecutive halves of a row of B;
//     C (f32): slot j -> row (r & 1) + 2*((j >> 1) & 1) + 4*(r >> 2), col 2*((r >> 1) & 1) + (j & 1) + 4*(j >> 2);
//     C (f16): the lane holds row r, columns 0..7 -- a whole row, one 16-byte store.
// So both operands are read from row-major shared tiles as 16-byte loads (8 halves = two
// k-steps of A, or the lane's two column blocks of B), from XOR-swizzled unpadded tiles
// that are conflict-free for exactly these access patterns.  Each instruction covers a
// 16x16 output block: quad-pairs 0/1 take rows 0-7, 2/3 rows 8-15, 0/2 cols 0-7, 1/3 cols 8-15,
// so the two quad-pairs sharing an operand read the same shared address (a broadcast).
//
// Requires M % BM == 0, K % 8 == 0, N % 32 == 0 (N = 4C is), BK = 32.

#include <cstdint>

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
#define GEMM70B_ENABLED 1
#else
#define GEMM70B_ENABLED 0
#endif

namespace gemm70b {
#if GEMM70B_ENABLED
__device__ __forceinline__ float gelu70b(float x) {
  const float u = x * fmaf(x * x, 0.0713548162f, 1.5957691216f);
  return x * __fdividef(1.f, 1.f + __expf(-u));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void lds128(uint32_t* r, uint32_t a) {
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma884(float* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%0,%1,%2,%3,%4,%5,%6,%7};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
// fp16 accumulator: the same 8 elements as 4 f16x2 (slot pairs (0,1), (2,3), ...)
__device__ __forceinline__ void mma884_h(uint32_t* c, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m8n8k4.row.row.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};\n"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1));
}
#endif
}  // namespace gemm70b

// BM x BN x 32 tile, WM x WN warps, warp tile (BM/WM) x (BN/WN) as 16x16 mma blocks.
// ACC16: fp16 accumulators (mma.m8n8k4.f16), half the accumulator registers; what the engine
// runs on the V100 anyway (--f16acc).
template<int BM, int BN, int WM, int WN, int ABL = 0, bool ACC16 = false>
__global__ __launch_bounds__(WM * WN * 32) void k_gemm_gelu70b(
    const half* __restrict__ A, const half* __restrict__ B, const half* __restrict__ bias,
    half* __restrict__ C, int M, int N, int K) {
#if GEMM70B_ENABLED
  using namespace gemm70b;
  constexpr int BK = 32, NT = WM * WN * 32;
  constexpr int TM = BM / WM, TN = BN / WN;
  constexpr int MI = TM / 16, NI = TN / 16;          // 16x16 mma blocks per warp
  static_assert(TM % 16 == 0 && TN % 32 == 0, "warp tile: 16-row blocks, 32-column groups");
  constexpr int A_ST = BM * BK * 2, B_ST = BK * BN * 2;   // bytes per stage
  extern __shared__ __align__(16) char smem70b[];
  char* As = smem70b;                                // [2][BM][BK] halves, 16B chunks swizzled
  char* Bs = As + 2 * A_ST;                          // [2][BK][BN] halves, columns permuted + swizzled
  const int n0 = blockIdx.x * BN, m0 = blockIdx.y * BM;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int wm = warp / WN, wn = warp - wm * WN;
  const int qp = (lane >> 2) & 3, rk = (lane & 3) + 4 * (lane >> 4);   // quad-pair, rank inside it

  // ---- shared layout ----
  // A: row r, k-chunk c (8 halves) at r*64 + ((c ^ ((r >> 1) & 3)) * 16): the 4 distinct rows of
  //    a quarter-warp phase (r, r+1, r+2, r+3 of one 16-row block) land in 4 distinct bank groups.
  auto a_off = [](int r, int c) { return r * (BK * 2) + ((c ^ ((r >> 1) & 3)) << 4); };
  // B: row k; inside each 32-column group the 4-column pieces are reordered so that one lane's
  //    two column blocks (n and n+16) are adjacent: piece j (cols 4j..4j+3) -> position
  //    8*(j & 3) + 4*(j >> 2); then the 16-byte chunk index is XORed with s(k) = 4*(k&1) + (k>>1)
  //    so that the 8 lanes of a phase (k = 0..3 of two quad-pairs) hit 8 distinct bank groups.
  auto b_off = [](int k, int col) {                  // col: 0 .. BN-1 (logical)
    const int g = col >> 5, j = (col >> 2) & 7, e = col & 3;
    const int pos = g * 32 + 8 * (j & 3) + 4 * (j >> 2) + e;      // permuted column position
    const int chunk = pos >> 3, within = pos & 7;
    return k * (BN * 2) + (((chunk ^ ((4 * (k & 1) + ((k >> 1) & 3)) & 15))) << 4) + within * 2;
  };

  // ---- global -> register -> shared copy descriptors ----
  constexpr int ACH = BM * (BK / 8), BCH = BK * (BN / 8);       // 16-byte chunks per stage
  constexpr int APT = (ACH + NT - 1) / NT, BPT = (BCH + NT - 1) / NT;
  const half* asrc[APT]; int adst[APT]; bool aon[APT]; int akc[APT];
  const half* bsrc[BPT]; int bdst0[BPT], bdst1[BPT]; bool bon[BPT]; int bkr[BPT];
#pragma unroll
  for (int j = 0; j < APT; ++j) {
    int i = tid + j * NT; aon[j] = i < ACH; if (!aon[j]) i = ACH - 1;
    const int r = i / (BK / 8), c = i - r * (BK / 8);
    asrc[j] = A + (size_t)(m0 + r) * K + c * 8; adst[j] = a_off(r, c); akc[j] = c * 8;
  }
#pragma unroll
  for (int j = 0; j < BPT; ++j) {
    int i = tid + j * NT; bon[j] = i < BCH; if (!bon[j]) i = BCH - 1;
    const int k = i / (BN / 8), c = i - k * (BN / 8);           // 8-column chunk c of row k
    const int n = n0 + c * 8;
    bon[j] = bon[j] && n < N;
    bsrc[j] = B + (size_t)k * N + (n < N ? n : 0);
    bdst0[j] = b_off(k, c * 8); bdst1[j] = b_off(k, c * 8 + 4);  // the two 4-column pieces
    bkr[j] = k;
  }
  uint4 areg[APT], breg[BPT];
  auto fetch = [&](int kt) {
    const int k0 = kt * BK;
#pragma unroll
    for (int j = 0; j < APT; ++j) {
      const bool ok = aon[j] && (k0 + akc[j] < K);
      areg[j] = ok ? *(const uint4*)(asrc[j] + k0) : make_uint4(0, 0, 0, 0);
    }
#pragma unroll
    for (int j = 0; j < BPT; ++j) {
      const bool ok = bon[j] && (k0 + bkr[j] < K);
      breg[j] = ok ? *(const uint4*)(bsrc[j] + (size_t)k0 * N) : make_uint4(0, 0, 0, 0);
    }
  };
  auto store = [&](int buf) {
#pragma unroll
    for (int j = 0; j < APT; ++j)
      if (aon[j]) *(uint4*)(As + buf * A_ST + adst[j]) = areg[j];
#pragma unroll
    for (int j = 0; j < BPT; ++j)
      if (tid + j * NT < BCH) {
        *(uint2*)(Bs + buf * B_ST + bdst0[j]) = make_uint2(breg[j].x, breg[j].y);
        *(uint2*)(Bs + buf * B_ST + bdst1[j]) = make_uint2(breg[j].z, breg[j].w);
      }
  };

  float acc[ACC16 ? 1 : MI][ACC16 ? 1 : NI][8];
  uint32_t acch[ACC16 ? MI : 1][ACC16 ? NI : 1][4];
#pragma unroll
  for (int mi = 0; mi < (ACC16 ? MI : MI); ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      if (ACC16) { acch[mi][ni][0] = acch[mi][ni][1] = acch[mi][ni][2] = acch[mi][ni][3] = 0u; }
      else {
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[mi][ni][j] = 0.f;
      }
    }

  // per-lane fragment addresses (relative to a stage base)
  // A: row = wm*TM + mi*16 + 8*(qp >> 1) + rk, k-chunk kc (0..3): two k-steps per load.  The
  //    swizzle term depends on the row only, so one base per mi and the chunk XOR at use.
  uint32_t a_row[MI]; const int a_swz = ((wm * TM + 8 * (qp >> 1) + rk) >> 1) & 3;   // mi*16 does not change bits 1-2
#pragma unroll
  for (int mi = 0; mi < MI; ++mi) a_row[mi] = (wm * TM + mi * 16 + 8 * (qp >> 1) + rk) * (BK * 2);
  // B: row k = kq*4 + (rk & 3); the lane's 16-byte piece for 32-column group g of the warp tile
  //    holds cols 32g + 8*(qp&1) + 4*(rk>>2) + {0..3} and the same + 16: blocks (2g, 2g+1)
  constexpr int NG = TN / 32;
  uint32_t b_addr[NG][8];
#pragma unroll
  for (int g = 0; g < NG; ++g)
#pragma unroll
    for (int k = 0; k < 8; ++k)
      b_addr[g][k] = b_off((k * 4 + (rk & 3)), wn * TN + 32 * g + 8 * (qp & 1) + 4 * (rk >> 2));
  const uint32_t as_base = smem_u32(As), bs_base = smem_u32(Bs);

  const int ktiles = (K + BK - 1) / BK;
  fetch(0); store(0); __syncthreads();
  for (int kt = 0; kt < ktiles; ++kt) {
    const int buf = kt & 1;
    if (ABL != 2 && kt + 1 < ktiles) fetch(kt + 1);
    const uint32_t ab = as_base + buf * A_ST, bb = bs_base + buf * B_ST;
#pragma unroll
    for (int kc = 0; kc < 4; ++kc) {                 // 8 k per step: two m8n8k4 k-steps
      uint32_t af[MI][4];
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) lds128(af[mi], ab + a_row[mi] + ((kc ^ a_swz) << 4));
      uint32_t bf[NG][2][4];                         // [group][k-step][piece: cols +0..3 (regs 0,1), +16..19 (regs 2,3)]
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        lds128(bf[g][0], bb + b_addr[g][kc * 2]);
        lds128(bf[g][1], bb + b_addr[g][kc * 2 + 1]);
      }
      if (ABL != 3) {
#pragma unroll
        for (int mi = 0; mi < MI; ++mi)
#pragma unroll
          for (int g = 0; g < NG; ++g) {
            // k-step 0: A halves 0-3 (regs 0,1); k-step 1: halves 4-7 (regs 2,3)
            if (ACC16) {
              mma884_h(acch[mi][2 * g],     af[mi][0], af[mi][1], bf[g][0][0], bf[g][0][1]);
              mma884_h(acch[mi][2 * g + 1], af[mi][0], af[mi][1], bf[g][0][2], bf[g][0][3]);
              mma884_h(acch[mi][2 * g],     af[mi][2], af[mi][3], bf[g][1][0], bf[g][1][1]);
              mma884_h(acch[mi][2 * g + 1], af[mi][2], af[mi][3], bf[g][1][2], bf[g][1][3]);
            } else {
              mma884(acc[mi][2 * g],     af[mi][0], af[mi][1], bf[g][0][0], bf[g][0][1]);
              mma884(acc[mi][2 * g + 1], af[mi][0], af[mi][1], bf[g][0][2], bf[g][0][3]);
              mma884(acc[mi][2 * g],     af[mi][2], af[mi][3], bf[g][1][0], bf[g][1][1]);
              mma884(acc[mi][2 * g + 1], af[mi][2], af[mi][3], bf[g][1][2], bf[g][1][3]);
            }
          }
      } else {
#pragma unroll
        for (int mi = 0; mi < MI; ++mi) { if (ACC16) acch[mi][0][0] += af[mi][0]; else acc[mi][0][0] += __uint_as_float(af[mi][0]); }
#pragma unroll
        for (int g = 0; g < NG; ++g) { if (ACC16) acch[0][2 * g][1] += bf[g][0][0]; else acc[0][2 * g][1] += __uint_as_float(bf[g][0][0]) + __uint_as_float(bf[g][1][2]); }
      }
    }
    if (kt + 1 < ktiles) { __syncthreads(); store(buf ^ 1); __syncthreads(); }
  }

  // ---- epilogue: + bias, GELU, fp16, staged through shared, coalesced 16-byte stores ----
  __syncthreads();                                   // the ring is dead; the C tile aliases it
  constexpr int LDC = BN + 8;                        // halves; padded, the stores below are 4-byte
  half* Cs = (half*)smem70b;
  // (the dynamic shared size is max(ring, C tile), see gemm_gelu70b_smem)
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      const int rbase = wm * TM + mi * 16 + 8 * (qp >> 1), cbase = wn * TN + (ni >> 1) * 32 + 8 * (qp & 1) + ((ni & 1) ? 16 : 0);
      if (ACC16) {
        // fp16 accumulator layout (probed): lane rank r holds row r, columns 0..7 of the block
        const int row = rbase + rk;
        uint32_t w[4];
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const int gcol = n0 + cbase + 2 * h;
          const float2 bb = (ABL != 1 && gcol + 1 < N) ? __half22float2(*(const half2*)(bias + gcol)) : make_float2(0.f, 0.f);
          const float2 f = __half22float2(*(const half2*)&acch[mi][ni][h]);
          float v0 = f.x + bb.x, v1 = f.y + bb.y;
          if (ABL != 1) { v0 = gelu70b(v0); v1 = gelu70b(v1); }
          const half2 o = __floats2half2_rn(v0, v1); w[h] = *(const uint32_t*)&o;
        }
        *(uint4*)(Cs + row * LDC + cbase) = make_uint4(w[0], w[1], w[2], w[3]);
      } else {
#pragma unroll
      for (int j = 0; j < 8; j += 2) {
        const int row = rbase + (rk & 1) + 2 * ((j >> 1) & 1) + 4 * (rk >> 2);
        const int col = cbase + 2 * ((rk >> 1) & 1) + 4 * (j >> 2);
        const int gcol = n0 + col;
        const float2 bb = (ABL != 1 && gcol + 1 < N) ? __half22float2(*(const half2*)(bias + gcol)) : make_float2(0.f, 0.f);
        float v0 = acc[mi][ni][j] + bb.x, v1 = acc[mi][ni][j + 1] + bb.y;
        if (ABL != 1) { v0 = gelu70b(v0); v1 = gelu70b(v1); }
        *(half2*)(Cs + row * LDC + col) = __floats2half2_rn(v0, v1);
      }
      }
    }
  __syncthreads();
  constexpr int CCH = BM * (BN / 8);
#pragma unroll 4
  for (int i = tid; i < CCH; i += NT) {
    const int r = i / (BN / 8), c = i - r * (BN / 8);
    const int col = n0 + c * 8;
    if (col < N) *(uint4*)(C + (size_t)(m0 + r) * N + col) = *(const uint4*)(Cs + r * LDC + c * 8);
  }
#endif  // GEMM70B_ENABLED
}

template<int BM, int BN, int WM, int WN>
static constexpr size_t gemm_gelu70b_smem() {
  size_t ring = (size_t)2 * (BM * 32 + 32 * BN) * 2;
  size_t epi = (size_t)BM * (BN + 8) * 2;
  return ring > epi ? ring : epi;
}
