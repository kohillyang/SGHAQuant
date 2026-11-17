// sam21_int4h.cuh -- w4a4_hiera: the hiera-side producers of the W4A4 path (--int4, --mon) for
// k_gemm_s8<..., I4 = true> (sam21_gemm_s8.cuh; packed int4 layout and fragment notes in sam21_int4.cuh).
//
// * kpad4(K): the int4 row of a K-wide operand is padded to a multiple of 32 codes (16 bytes), because the
//   GEMM's cp.async loads are 16 bytes: K = 144 (hiera-L stage 0) becomes 160 codes / 80 bytes, the pad
//   codes are 0 in both operands, so the dot product is unchanged.  Every other hiera-L K is a multiple of 32.
// * k_ln_q4h<NV>: k_ln_q8<NV, 1> (rotated per-token LayerNorm producer, sam21_rot.cuh) with the 4-bit
//   quantiser, the slot map of nano_hiera (src[t] < 0 -> zero codes) and the padded row.
// * k_ln_mon4<C, BB>: the same producer with the structured transform T = B1 Pi B2 ("mon") between the
//   LayerNorm and the Hadamard, on the tensor cores (from micro/structT/lnTC.cu, which emits int8):
//     x_ln (fp16) -> B1: n = C/BB dense BB x BB blocks (mma.m16n8k16, f16 x f16 -> f32)
//                 -> Pi: the stride permutation, i.e. position q of block j goes to channel q*n + j
//                 -> B2: BB dense n x n blocks, one per position q (k16 steps + an m16n8k8 tail)
//                 -> R = I (x) H_nb (the shipped Hadamard, nb = rot_block(C)) -> per-token int4.
//   This is the simulation's x @ T @ R with T = (B1blk)[:, pi] @ B2blk / cnorm (w4a4_hiera/sim/tools/
//   struct_T_fact.py; cnorm and a scalar balance between B1 and B2 are folded into the coefficients by the
//   exporter).  Intermediates are fp16 (the simulation keeps fp32).
//   Shared memory per 16-row tile: X [16][C + 8] fp16 (LN output, later B2's output) and Z [16][ZS] fp16 (B1's
//   output, position-major with stride NPZ = n rounded up to even per position).  B2's padded k reads run into
//   the next position's slots (or a zeroed slack at the end of the row): their coefficients are exactly zero
//   and the values are finite, so they add nothing.  gamma / beta are read through L1 (no shared copy).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>
#include "sam21_rot.cuh"     // rot_nb, rot_block

__host__ __device__ constexpr int kpad4(int K) { return (K + 31) / 32 * 32; }

// ---------------------------------------------------------------- rotated per-token int4 LayerNorm (no T)
template<int NV>
__global__ void k_ln_q4h(const half* __restrict__ x, uint8_t* __restrict__ y, float* __restrict__ ys,
                         const float* __restrict__ g, const float* __restrict__ b, int T,
                         const int* __restrict__ src = nullptr, float eps = 1e-6f) {
  constexpr int C = NV * 8, NR = (NV + 31) / 32, NB = rot_nb(NV), LDW = kpad4(C) / 8;
  static_assert(NB >= 2 && NB <= 16 && NV % NB == 0, "C = m * 2^k, k >= 4");
  extern __shared__ float gb[];
  for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; }
  __syncthreads();
  const int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  const int lane = threadIdx.x & 31;
  uint32_t* yr = (uint32_t*)(y + (size_t)t * (LDW * 4));
  const int ts = src ? src[t] : t;              // warp-uniform
  if (ts < 0) {
    for (int v = lane; v < LDW; v += 32) yr[v] = 0u;
    if (lane == 0) ys[t] = 1e-12f;
    return;
  }
  const float4* xv = (const float4*)(x + (size_t)ts * C);
  float4 keep[NR];
  float s = 0.f, ss = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      float4 f = xv[v]; keep[m] = f;
      const half2* h = (const half2*)&f;
#pragma unroll
      for (int k = 0; k < 4; ++k) { float2 p = __half22float2(h[k]); s += p.x + p.y; ss += p.x * p.x + p.y * p.y; }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); ss += __shfl_xor_sync(0xffffffff, ss, o); }
  const float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
  float out[NR][8], mx = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      const half2* h = (const half2*)&keep[m];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        const int c = v * 8 + k * 2;
        out[m][k * 2]     = (p.x - mean) * rs * gb[c] + gb[C + c];
        out[m][k * 2 + 1] = (p.y - mean) * rs * gb[c + 1] + gb[C + c + 1];
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) out[m][k] = 0.f;
    }
  }
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int st = 1; st < 8; st <<= 1)
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if ((k & st) == 0) { const float a = out[m][k], c2 = out[m][k | st]; out[m][k] = a + c2; out[m][k | st] = a - c2; }
#pragma unroll
  for (int st = 1; st < NB; st <<= 1)
#pragma unroll
    for (int m = 0; m < NR; ++m)
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
        out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
      }
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float hn = rsqrtf((float)(NB * 8));       // the Hadamard's 1/sqrt(n), folded into the scale
  const float sc = mx * hn * (1.f / 7.f) + 1e-12f, inv = hn / sc;
  if (lane == 0) ys[t] = sc;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    const int v = lane + m * 32;
    if (v < NV) {
      uint32_t u = 0;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int q = max(-7, min(7, __float2int_rn(out[m][k] * inv)));
        u |= ((uint32_t)q & 0xfu) << (4 * k);
      }
      yr[v] = u;
    } else if (v < LDW) {
      yr[v] = 0u;                                   // the K pad (C = 144: codes 144..159)
    }
  }
}

// int8 codes [R, K] -> packed int4 [R, kpad4(K) / 2] (zero pad); counts codes outside -7..7 into *bad
__global__ void k_pack_s4p(const int8_t* __restrict__ src, uint8_t* __restrict__ dst, int R, int K,
                           unsigned int* __restrict__ bad) {
  const int Kh = kpad4(K) / 2;
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)R * Kh) return;
  const int r = (int)(i / Kh), j = (int)(i - (size_t)r * Kh), k0 = 2 * j;
  const int a = k0 < K ? src[(size_t)r * K + k0] : 0, b = k0 + 1 < K ? src[(size_t)r * K + k0 + 1] : 0;
  if (bad && (a < -7 || a > 7 || b < -7 || b > 7)) atomicAdd(bad, 1u);
  dst[i] = (uint8_t)(((uint32_t)a & 0xfu) | (((uint32_t)b & 0xfu) << 4));
}

// ---------------------------------------------------------------- the mon producer on the tensor cores
__device__ __forceinline__ void mon_mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// m16n8k8 f16: A two .f16x2 (a0 row g, a1 row g+8, k = 2*tig, 2*tig+1), B one .f16x2 (k = 2*tig, 2*tig+1, n = g)
__device__ __forceinline__ void mon_mma1688(float c[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(b0));
}
// FA (fp16 accumulation): the same shapes with an f16 accumulator, two .f16x2 registers (c0 row g, c1 row g + 8, cols
// 2 tig, 2 tig + 1)
__device__ __forceinline__ void mon_mma16816h(uint32_t c[2], const uint32_t a[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mon_mma1688h(uint32_t c[2], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4}, {%0,%1};\n"
               : "+r"(c[0]), "+r"(c[1]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mon_unpack_hacc(float acc[4], const uint32_t h[2]) {
  const float2 r0 = __half22float2(*(const half2*)&h[0]), r1 = __half22float2(*(const half2*)&h[1]);
  acc[0] = r0.x; acc[1] = r0.y; acc[2] = r1.x; acc[3] = r1.y;
}
__device__ __forceinline__ int lane_id_mon() { return threadIdx.x & 31; }
__device__ __forceinline__ uint32_t mon_smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mon_ldsm_x4(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mon_ldsm_x2(uint32_t* r, uint32_t a) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__host__ __device__ constexpr int mon_zstride(int base, int slack) {
  int s = base + (slack > 0 ? slack : 0);
  while (s % 16 != 8) ++s;                        // (ZS / 2) % 8 == 4: the 8 rows of an A load hit distinct banks
  return s;
}
// k steps of a mma over a k-extent L (padded): k16 steps, then one k8 step if the remainder is <= 8
__host__ __device__ constexpr int mon_ns16(int L) { return L / 16 + ((L % 16) > 8 ? 1 : 0); }
__host__ __device__ constexpr bool mon_t8(int L) { return (L % 16) > 0 && (L % 16) <= 8; }
template<int C, int BB, int LM = 0> struct MonCfg4 {
  static constexpr int N = C / BB;                // B1 blocks = B2 width
  // B1: k = the block's BB inputs (k16 steps + a k8 tail; the padded k reads the next block's / the row pad's columns,
  // whose coefficients are 0), n = its BB outputs in n-tiles of 8 (outputs >= BB dropped)
  static constexpr int NS1 = mon_ns16(BB);
  static constexpr bool T81 = mon_t8(BB);
  static constexpr int K1 = 16 * NS1 + (T81 ? 8 : 0);
  static constexpr int NT1 = (BB + 7) / 8;
  static constexpr int NPZ = LM ? (N + 7) / 8 * 8 : (N + 1) / 2 * 2;   // Z slots per position (LM: 16-byte slots; else even)
  static constexpr int NS16 = mon_ns16(N);        // B2 k16 steps
  static constexpr bool T8 = mon_t8(N);           // ... and one k8 step
  static constexpr int K2 = 16 * NS16 + (T8 ? 8 : 0);             // B2's padded k
  static constexpr int NT2 = (N + 7) / 8;         // B2 n-tiles
  static constexpr int XS = C + 8;                // X row stride (halves); [C, C + 8) zeroed once
  static constexpr int ZS = mon_zstride(BB * NPZ, LM ? 0 : K2 - NPZ);
  static_assert(!LM || K2 <= NPZ, "LM: B2's padded k stays inside the position's slots");
  static constexpr int NF1 = N * NS1 * NT1;       // B1 k16 fragments: uint2 per lane
  static constexpr int NF1T = T81 ? N * NT1 : 0;  // B1 k8 fragments: uint32 per lane
  static constexpr int NF2 = BB * NS16 * NT2;     // B2 k16 fragments: uint2 per lane
  static constexpr int NF2T = T8 ? BB * NT2 : 0;  // B2 k8 fragments: uint32 per lane
  static constexpr size_t SMEM = (size_t)16 * XS * 2 + (size_t)16 * ZS * 2;
  static_assert(C % BB == 0 && BB % 2 == 0 && C % 8 == 0 && K1 - BB <= 8, "mon shape");
};

// f1 [NF1][32] uint2 (block j, k16 step, n-tile), f1t [NF1T][32] uint32 (block j, n-tile): B1's k8 tail;
// f2 [NF2][32] uint2 (position q, k16 step, n-tile), f2t [NF2T][32] uint32 (position q, n-tile): B2's k8 tail.
// y [T][kpad4(C) / 2] packed int4, ys [T].
// RC (bit mask): bit 0 = B1's fragments, bit 1 = B2's fragments are loaded once per warp into registers (a warp always
// owns units warp + 8 i) and kept for all its tiles; a clear bit reads that set through L1 for every tile.
// HL = 1: phase 1 issues both rows' slot lookups and loads before the first row's reductions (more registers).
// LM = 1: A fragments through ldmatrix (B1 when BB % 8 == 0; B2 always, with 16-byte Z slots).
// FA = 1: the B1 / B2 products accumulate in fp16 (f16 mma) instead of fp32.
template<int C, int BB, int RC = 0, int HL = 0, int LM = 0, int FA = 0>
__global__ void __launch_bounds__(256) k_ln_mon4(const half* __restrict__ x, uint8_t* __restrict__ y, float* __restrict__ ys,
                                                 const float* __restrict__ g, const float* __restrict__ bta,
                                                 const uint2* __restrict__ f1, const uint32_t* __restrict__ f1t,
                                                 const uint2* __restrict__ f2, const uint32_t* __restrict__ f2t,
                                                 int T, const int* __restrict__ src,
                                                 float eps) {
  using M = MonCfg4<C, BB, LM>;
  constexpr int NV = C / 8, NR = (NV + 31) / 32, NB = rot_nb(NV), LDW = kpad4(C) / 8;
  static_assert(NB >= 2 && NB <= 16 && NV % NB == 0, "C = m * 2^k, k >= 4");
  constexpr bool LM1 = LM != 0 && BB % 8 == 0, LM2 = LM != 0;
  const int lrow = (lane_id_mon() & 7) + 8 * ((lane_id_mon() >> 3) & 1), lcol = 8 * (lane_id_mon() >> 4);
  extern __shared__ __align__(16) unsigned char mon_smem[];
  half* Xs = (half*)mon_smem;
  half* Zs = Xs + 16 * M::XS;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int gid = lane >> 2, tig = lane & 3;
  for (int i = threadIdx.x; i < 16 * M::ZS; i += blockDim.x) Zs[i] = __float2half(0.f);   // pads stay 0
  for (int i = threadIdx.x; i < 16 * 8; i += blockDim.x) Xs[(i >> 3) * M::XS + C + (i & 7)] = __float2half(0.f);
  constexpr int NU1 = M::N * M::NT1, UPW1 = (NU1 + 7) / 8;      // B1 units, per warp
  constexpr int NU2 = BB * M::NT2, UPW2 = (NU2 + 7) / 8;        // B2 units, per warp
  constexpr int S1 = M::NS1 > 0 ? M::NS1 : 1, S2 = M::NS16 > 0 ? M::NS16 : 1;
  constexpr bool R1 = (RC & 1) != 0, R2 = (RC & 2) != 0;
  uint2 c1[R1 ? UPW1 : 1][S1]; uint32_t c1t[R1 ? UPW1 : 1];
  uint2 c2[R2 ? UPW2 : 1][S2]; uint32_t c2t[R2 ? UPW2 : 1];
  if constexpr (R1) {
#pragma unroll
    for (int i = 0; i < UPW1; ++i) {
      const int u = warp + 8 * i;
      if (u < NU1) {
        const int j = u / M::NT1, nt = u - j * M::NT1;
#pragma unroll
        for (int ks = 0; ks < M::NS1; ++ks) c1[i][ks] = __ldg(f1 + ((size_t)(j * M::NS1 + ks) * M::NT1 + nt) * 32 + lane);
        if constexpr (M::T81) c1t[i] = __ldg(f1t + ((size_t)j * M::NT1 + nt) * 32 + lane);
      }
    }
  }
  if constexpr (R2) {
#pragma unroll
    for (int i = 0; i < UPW2; ++i) {
      const int u = warp + 8 * i;
      if (u < NU2) {
        const int q = u / M::NT2, nt = u - q * M::NT2;
#pragma unroll
        for (int ks = 0; ks < M::NS16; ++ks) c2[i][ks] = __ldg(f2 + ((size_t)(q * M::NS16 + ks) * M::NT2 + nt) * 32 + lane);
        if constexpr (M::T8) c2t[i] = __ldg(f2t + ((size_t)q * M::NT2 + nt) * 32 + lane);
      }
    }
  }
  __syncthreads();
  for (int tile = blockIdx.x; tile * 16 < T; tile += gridDim.x) {
    if constexpr (HL == 0) {
    // ---- 1. LayerNorm (the shipped arithmetic), rows warp and warp + 8, fp16 into X
    for (int r = warp; r < 16; r += 8) {
      const int t = tile * 16 + r;
      const int ts = src ? src[t] : t;
      half* xr = Xs + r * M::XS;
      if (ts < 0) {
        for (int v = lane; v < NV; v += 32) *(uint4*)(xr + v * 8) = make_uint4(0u, 0u, 0u, 0u);
        continue;
      }
      const float4* xv = (const float4*)(x + (size_t)ts * C);
      float4 keep[NR]; float s = 0.f, ss = 0.f;
#pragma unroll
      for (int m = 0; m < NR; ++m) {
        const int v = lane + m * 32;
        if (v < NV) {
          float4 f = xv[v]; keep[m] = f; const half2* h = (const half2*)&f;
#pragma unroll
          for (int k = 0; k < 4; ++k) { float2 p = __half22float2(h[k]); s += p.x + p.y; ss += p.x * p.x + p.y * p.y; }
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); ss += __shfl_xor_sync(0xffffffff, ss, o); }
      const float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
#pragma unroll
      for (int m = 0; m < NR; ++m) {
        const int v = lane + m * 32;
        if (v < NV) {
          const half2* h = (const half2*)&keep[m]; half2 o[4];
          const float4 g0 = __ldg((const float4*)(g + v * 8)), g1 = __ldg((const float4*)(g + v * 8 + 4));
          const float4 b0 = __ldg((const float4*)(bta + v * 8)), b1 = __ldg((const float4*)(bta + v * 8 + 4));
          const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
          const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
          for (int k = 0; k < 4; ++k) {
            float2 p = __half22float2(h[k]);
            o[k] = __floats2half2_rn((p.x - mean) * rs * gg[2 * k] + bb[2 * k], (p.y - mean) * rs * gg[2 * k + 1] + bb[2 * k + 1]);
          }
          *(uint4*)(xr + v * 8) = *(const uint4*)o;
        }
      }
    }
    } else {
    // ---- 1. LayerNorm (the shipped arithmetic), rows warp and warp + 8, fp16 into X; both rows' slot lookups and loads
    //      are issued before the first row's reductions
    {
      int tsr[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) { const int t = tile * 16 + warp + 8 * h; tsr[h] = src ? src[t] : t; }
      float4 keep[2][NR];
#pragma unroll
      for (int h = 0; h < 2; ++h)
        if (tsr[h] >= 0) {
          const float4* xv = (const float4*)(x + (size_t)tsr[h] * C);
#pragma unroll
          for (int m = 0; m < NR; ++m) { const int v = lane + m * 32; if (v < NV) keep[h][m] = xv[v]; }
        }
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        half* xr = Xs + (warp + 8 * h) * M::XS;
        if (tsr[h] < 0) {
          for (int v = lane; v < NV; v += 32) *(uint4*)(xr + v * 8) = make_uint4(0u, 0u, 0u, 0u);
          continue;
        }
        float s = 0.f, ss = 0.f;
#pragma unroll
        for (int m = 0; m < NR; ++m) {
          const int v = lane + m * 32;
          if (v < NV) {
            const half2* hh = (const half2*)&keep[h][m];
#pragma unroll
            for (int k = 0; k < 4; ++k) { float2 p = __half22float2(hh[k]); s += p.x + p.y; ss += p.x * p.x + p.y * p.y; }
          }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) { s += __shfl_xor_sync(0xffffffff, s, o); ss += __shfl_xor_sync(0xffffffff, ss, o); }
        const float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
#pragma unroll
        for (int m = 0; m < NR; ++m) {
          const int v = lane + m * 32;
          if (v < NV) {
            const half2* hh = (const half2*)&keep[h][m]; half2 o[4];
            const float4 g0 = __ldg((const float4*)(g + v * 8)), g1 = __ldg((const float4*)(g + v * 8 + 4));
            const float4 b0 = __ldg((const float4*)(bta + v * 8)), b1 = __ldg((const float4*)(bta + v * 8 + 4));
            const float gg[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
            const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
              float2 p = __half22float2(hh[k]);
              o[k] = __floats2half2_rn((p.x - mean) * rs * gg[2 * k] + bb[2 * k], (p.y - mean) * rs * gg[2 * k + 1] + bb[2 * k + 1]);
            }
            *(uint4*)(xr + v * 8) = *(const uint4*)o;
          }
        }
      }
    }
    }
    __syncthreads();
    // ---- 2. B1: unit = (block j, n-tile nt) over the 16 rows; output position q of block j -> Z[row][q*NPZ + j]
    auto b1unit = [&](int u, int i) {
      const int j = u / M::NT1, nt = u - j * M::NT1;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      uint32_t hacc[2] = {0u, 0u};
      const half* xr0 = Xs + gid * M::XS + j * BB + tig * 2;
      const half* xr1 = xr0 + 8 * M::XS;
      const uint32_t xl = LM1 ? mon_smem_u32(Xs + lrow * M::XS + j * BB + lcol) : 0u;
#pragma unroll
      for (int ks = 0; ks < M::NS1; ++ks) {
        uint32_t a[4];
        if constexpr (LM1) mon_ldsm_x4(a, xl + ks * 32);
        else {
          a[0] = *(const uint32_t*)(xr0 + ks * 16);
          a[1] = *(const uint32_t*)(xr1 + ks * 16);
          a[2] = *(const uint32_t*)(xr0 + ks * 16 + 8);
          a[3] = *(const uint32_t*)(xr1 + ks * 16 + 8);
        }
        uint2 bf;
        if constexpr (R1) bf = c1[i][ks];
        else bf = __ldg(f1 + ((size_t)(j * M::NS1 + ks) * M::NT1 + nt) * 32 + lane);
        if constexpr (FA != 0) mon_mma16816h(hacc, a, bf.x, bf.y); else mon_mma16816(acc, a, bf.x, bf.y);
      }
      if constexpr (M::T81) {
        uint32_t a0, a1;
        if constexpr (LM1) { uint32_t t2[2]; mon_ldsm_x2(t2, mon_smem_u32(Xs + (lane & 15) * M::XS + j * BB + M::NS1 * 16)); a0 = t2[0]; a1 = t2[1]; }
        else { a0 = *(const uint32_t*)(xr0 + M::NS1 * 16); a1 = *(const uint32_t*)(xr1 + M::NS1 * 16); }
        uint32_t bt;
        if constexpr (R1) bt = c1t[i];
        else bt = __ldg(f1t + ((size_t)j * M::NT1 + nt) * 32 + lane);
        if constexpr (FA != 0) mon_mma1688h(hacc, a0, a1, bt); else mon_mma1688(acc, a0, a1, bt);
      }
      if constexpr (FA != 0) mon_unpack_hacc(acc, hacc);
      const int q0 = nt * 8 + tig * 2;
      if (q0 < BB) {
        Zs[gid * M::ZS + q0 * M::NPZ + j]       = __float2half(acc[0]);
        Zs[(gid + 8) * M::ZS + q0 * M::NPZ + j] = __float2half(acc[2]);
      }
      if (q0 + 1 < BB) {
        Zs[gid * M::ZS + (q0 + 1) * M::NPZ + j]       = __float2half(acc[1]);
        Zs[(gid + 8) * M::ZS + (q0 + 1) * M::NPZ + j] = __float2half(acc[3]);
      }
    };
    if constexpr (R1) {
#pragma unroll
      for (int i = 0; i < UPW1; ++i) if (warp + 8 * i < NU1) b1unit(warp + 8 * i, i);
    } else {
      for (int u = warp; u < NU1; u += 8) b1unit(u, 0);
    }
    __syncthreads();
    // ---- 3. B2: unit = (position q, n-tile nt); output channel q * N + jp, back into X (natural order)
    auto b2unit = [&](int u, int i) {
      const int q = u / M::NT2, nt = u - q * M::NT2;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
      uint32_t hacc[2] = {0u, 0u};
      const half* zr0 = Zs + gid * M::ZS + q * M::NPZ + tig * 2;
      const half* zr1 = zr0 + 8 * M::ZS;
      const uint32_t zl = LM2 ? mon_smem_u32(Zs + lrow * M::ZS + q * M::NPZ + lcol) : 0u;
#pragma unroll
      for (int ks = 0; ks < M::NS16; ++ks) {
        uint32_t a[4];
        if constexpr (LM2) mon_ldsm_x4(a, zl + ks * 32);
        else {
          a[0] = *(const uint32_t*)(zr0 + ks * 16);
          a[1] = *(const uint32_t*)(zr1 + ks * 16);
          a[2] = *(const uint32_t*)(zr0 + ks * 16 + 8);
          a[3] = *(const uint32_t*)(zr1 + ks * 16 + 8);
        }
        uint2 bf;
        if constexpr (R2) bf = c2[i][ks];
        else bf = __ldg(f2 + ((size_t)(q * M::NS16 + ks) * M::NT2 + nt) * 32 + lane);
        if constexpr (FA != 0) mon_mma16816h(hacc, a, bf.x, bf.y); else mon_mma16816(acc, a, bf.x, bf.y);
      }
      if constexpr (M::T8) {
        uint32_t a0, a1;
        if constexpr (LM2) { uint32_t t2[2]; mon_ldsm_x2(t2, mon_smem_u32(Zs + (lane & 15) * M::ZS + q * M::NPZ + M::NS16 * 16)); a0 = t2[0]; a1 = t2[1]; }
        else { a0 = *(const uint32_t*)(zr0 + M::NS16 * 16); a1 = *(const uint32_t*)(zr1 + M::NS16 * 16); }
        uint32_t bt;
        if constexpr (R2) bt = c2t[i];
        else bt = __ldg(f2t + ((size_t)q * M::NT2 + nt) * 32 + lane);
        if constexpr (FA != 0) mon_mma1688h(hacc, a0, a1, bt); else mon_mma1688(acc, a0, a1, bt);
      }
      if constexpr (FA != 0) mon_unpack_hacc(acc, hacc);
      const int jp = nt * 8 + tig * 2;
      half* o0 = Xs + gid * M::XS + q * M::N + jp;
      half* o1 = o0 + 8 * M::XS;
      if (jp < M::N)     { o0[0] = __float2half(acc[0]); o1[0] = __float2half(acc[2]); }
      if (jp + 1 < M::N) { o0[1] = __float2half(acc[1]); o1[1] = __float2half(acc[3]); }
    };
    if constexpr (R2) {
#pragma unroll
      for (int i = 0; i < UPW2; ++i) if (warp + 8 * i < NU2) b2unit(warp + 8 * i, i);
    } else {
      for (int u = warp; u < NU2; u += 8) b2unit(u, 0);
    }
    __syncthreads();
    // ---- 4. the shipped Hadamard (contiguous NB-vector blocks), per-token absmax, int4, one row per warp
    for (int r = warp; r < 16; r += 8) {
      const int t = tile * 16 + r;
      float out[NR][8];
#pragma unroll
      for (int m = 0; m < NR; ++m) {
        const int v = lane + m * 32;
        if (v < NV) {
          const uint4 raw = *(const uint4*)(Xs + r * M::XS + v * 8); const half2* h = (const half2*)&raw;
#pragma unroll
          for (int k = 0; k < 4; ++k) { float2 p = __half22float2(h[k]); out[m][2 * k] = p.x; out[m][2 * k + 1] = p.y; }
        } else {
#pragma unroll
          for (int k = 0; k < 8; ++k) out[m][k] = 0.f;
        }
      }
#pragma unroll
      for (int m = 0; m < NR; ++m)
#pragma unroll
        for (int st = 1; st < 8; st <<= 1)
#pragma unroll
          for (int k = 0; k < 8; ++k)
            if ((k & st) == 0) { const float a = out[m][k], c2 = out[m][k | st]; out[m][k] = a + c2; out[m][k | st] = a - c2; }
#pragma unroll
      for (int st = 1; st < NB; st <<= 1)
#pragma unroll
        for (int m = 0; m < NR; ++m)
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
            out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
          }
      float mx = 0.f;
#pragma unroll
      for (int m = 0; m < NR; ++m)
#pragma unroll
        for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
#pragma unroll
      for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      const float hn = rsqrtf((float)(NB * 8));
      const float sc = mx * hn * (1.f / 7.f) + 1e-12f, inv = hn / sc;
      if (lane == 0) ys[t] = sc;
      uint32_t* yr = (uint32_t*)(y + (size_t)t * (LDW * 4));
#pragma unroll
      for (int m = 0; m < NR; ++m) {
        const int v = lane + m * 32;
        if (v < NV) {
          uint32_t uu = 0;
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int q = max(-7, min(7, __float2int_rn(out[m][k] * inv)));
            uu |= ((uint32_t)q & 0xfu) << (4 * k);
          }
          yr[v] = uu;
        } else if (v < LDW) {
          yr[v] = 0u;
        }
      }
    }
    __syncthreads();                                // X is rewritten by the next tile's LayerNorm
  }
}

// host: B1 [N][BB][BB] (block j, in i, out o) and B2 [BB][N][N] (position q, out jp, in j), fp32 -> fragment-packed
// halves in the k_ln_mon4<C, BB> order (padded k / n coefficients are 0).  Returns the largest |coefficient|.
static float mon4_pack(int C, int BB, const float* B1, const float* B2, std::vector<uint2>& f1, std::vector<uint32_t>& f1t,
                       std::vector<uint2>& f2, std::vector<uint32_t>& f2t) {
  const int N = C / BB;
  const int NS1 = mon_ns16(BB), NT1 = (BB + 7) / 8; const bool T81 = mon_t8(BB);
  const int NS16 = mon_ns16(N), NT2 = (N + 7) / 8; const bool T8 = mon_t8(N);
  float mx = 0.f;
  auto h2 = [&](float lo, float hi) {
    mx = std::max(mx, std::max(std::fabs(lo), std::fabs(hi)));
    half2 v = __floats2half2_rn(lo, hi); return *(uint32_t*)&v;
  };
  f1.assign((size_t)N * NS1 * NT1 * 32, make_uint2(0, 0));
  f1t.assign(T81 ? (size_t)N * NT1 * 32 : 0, 0u);
  f2.assign((size_t)BB * NS16 * NT2 * 32, make_uint2(0, 0));
  f2t.assign(T8 ? (size_t)BB * NT2 * 32 : 0, 0u);
  for (int j = 0; j < N; ++j)
    for (int nt = 0; nt < NT1; ++nt)
      for (int l = 0; l < 32; ++l) {
        const int n = nt * 8 + (l >> 2);
        auto B = [&](int kk) { return (kk < BB && n < BB) ? B1[((size_t)j * BB + kk) * BB + n] : 0.f; };
        for (int ks = 0; ks < NS1; ++ks) {
          const int k = ks * 16 + (l & 3) * 2;
          f1[((size_t)(j * NS1 + ks) * NT1 + nt) * 32 + l] = make_uint2(h2(B(k), B(k + 1)), h2(B(k + 8), B(k + 9)));
        }
        if (T81) {
          const int k = NS1 * 16 + (l & 3) * 2;
          f1t[((size_t)j * NT1 + nt) * 32 + l] = h2(B(k), B(k + 1));
        }
      }
  for (int q = 0; q < BB; ++q)
    for (int nt = 0; nt < NT2; ++nt)
      for (int l = 0; l < 32; ++l) {
        const int n = nt * 8 + (l >> 2);
        auto B = [&](int kk) { return (kk < N && n < N) ? B2[((size_t)q * N + n) * N + kk] : 0.f; };
        for (int ks = 0; ks < NS16; ++ks) {
          const int k = ks * 16 + (l & 3) * 2;
          f2[((size_t)(q * NS16 + ks) * NT2 + nt) * 32 + l] = make_uint2(h2(B(k), B(k + 1)), h2(B(k + 8), B(k + 9)));
        }
        if (T8) {
          const int k = NS16 * 16 + (l & 3) * 2;
          f2t[((size_t)q * NT2 + nt) * 32 + l] = h2(B(k), B(k + 1));
        }
      }
  return mx;
}
