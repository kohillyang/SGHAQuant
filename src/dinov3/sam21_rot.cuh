// sam21_rot.cuh -- the rotated int8 LayerNorm and the matching weight rotation.
//
// An orthogonal rotation R of a GEMM's input, folded into its weight (x R)(R^T W) = x W,
// leaves the GEMM exact and spreads every channel over a block of the row, so the per-token
// int8 absmax falls from 11-15x the rms to ~3.4x and the int8 cost of the LayerNorm-fed
// GEMMs drops 3x (tools/int8_rot.py).  For the LayerNorm-fed pair R = Q9 (x) H_n with
// C = 9n (n = 16/32/64/128): a Walsh-Hadamard transform on every contiguous block of n
// channels, then a 9-point orthonormal DCT across the nine blocks at each position.  For
// fc2 (its input produced tile-wise by fc1) R is a block-diagonal Hadamard on 128-column
// blocks, applied in the fc1 epilogue (sam21_gemm_s8.cuh, ROT8).
#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

__constant__ float c_dct9[81];         // Q[j][j'] = sqrt(2/9) c_j cos(pi (j'+0.5) j / 9)

// LayerNorm that emits INT8 with a per-token scale, for the int8 GEMM path.  The row is
// already cached in registers by the first pass, so the absmax needed for the scale costs
// one extra warp reduction and the quantisation is free: this is why the LayerNorm-fed
// GEMMs (qkv, fc1) are the cheap place to put int8 -- no separate quantise pass exists.
// ROT = 1 applies R = I_9 (x) H_n before the absmax (a Hadamard on each block of n = C/9
// channels), ROT = 2 additionally mixes the nine blocks with the 9-point DCT (R = Q9 (x) H_n).
// The two are within noise of each other in accuracy (tools/int8_rot.py: hb 0.999699,
// h9b 0.999726) and ROT = 1 is what ships: the DCT costs 72 shuffles and 9 table reads per
// lane, which on the 65536-token stage-0 LayerNorms added 1.4 ms to the forward.
// The row lives in registers as vectors of 8 halves, v = lane + 32m; block j is the
// NB = n/8 consecutive vectors [j*NB, j*NB + NB), which sit in one register slot
// m = (j*NB) >> 5 of lanes (j*NB & 31) + q.  So the Hadamard is 3 in-lane stages plus
// log2(NB) shuffle stages, and the DCT gathers one value per block with a shuffle whose
// source register slot is a compile-time constant.
// Width of the LN-side rotation block for a row of NV vectors: the largest power of two
// dividing NV, capped at 16 vectors (128 channels).  For hiera-L's C = 9 * 2^k this is NV / 9,
// i.e. exactly the shipped I9 (x) H_{C/9}; for a plain ViT (C = 768 = 6 * 128, 384 = 3 * 128,
// 1024 = 8 * 128) it is I_{C/128} (x) H_128, the `hbp` rotation of tools/int8_quant.py.
__host__ __device__ constexpr int rot_nb(int nv) {
  int nb = 1;
  while (nb < 16 && nv % (nb * 2) == 0) nb *= 2;
  return nb;
}
__host__ __device__ constexpr int rot_block(int K) { return rot_nb(K / 8) * 8; }   // in channels

// src (optional, as k_ln_t): output row t normalises input row src[t]; src[t] < 0 -> zero codes.
template<int NV, int ROT = 0>
__global__ void k_ln_q8(const half* __restrict__ x, int8_t* __restrict__ y,
                        float* __restrict__ ys, const float* __restrict__ g,
                        const float* __restrict__ b, int T, const int* __restrict__ src = nullptr,
                        float eps = 1e-6f) {
  const int C = NV * 8;
  const int NR = (NV + 31) / 32;
  constexpr int NB = rot_nb(NV);                // vectors per rotation block
  static_assert(!ROT || (NB >= 2 && NB <= 16 && NV % NB == 0), "C = m * 2^k, k >= 4");
  static_assert(ROT != 2 || NV == 9 * NB, "the DCT variant needs exactly nine blocks");
  extern __shared__ float gb[];
  float* qtab = gb + 2 * C;                     // the DCT table, staged for conflict-free reads
  for (int i = threadIdx.x; i < C; i += blockDim.x) { gb[i] = g[i]; gb[C + i] = b[i]; }
  if (ROT == 2) for (int i = threadIdx.x; i < 81; i += blockDim.x) qtab[i] = c_dct9[i];
  __syncthreads();
  int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= T) return;
  int lane = threadIdx.x & 31;
  const int ts = src ? src[t] : t;              // warp-uniform
  if (ts < 0) {
    uint2* yz = (uint2*)(y + (size_t)t * C);
    for (int v = lane; v < NV; v += 32) yz[v] = make_uint2(0u, 0u);
    if (lane == 0) ys[t] = 1e-12f;
    return;
  }
  const float4* xv = (const float4*)(x + (size_t)ts * C);
  float4 keep[NR];
  float s = 0.f, ss = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      float4 f = xv[v]; keep[m] = f;
      const half2* h = (const half2*)&f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        s += p.x + p.y; ss += p.x * p.x + p.y * p.y;
      }
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    s += __shfl_xor_sync(0xffffffff, s, o);
    ss += __shfl_xor_sync(0xffffffff, ss, o);
  }
  float mean = s / C, var = ss / C - mean * mean, rs = rsqrtf(var + eps);
  float out[NR][8], mx = 0.f;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      const half2* h = (const half2*)&keep[m];
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        float2 p = __half22float2(h[k]);
        int c = v * 8 + k * 2;
        out[m][k * 2]     = (p.x - mean) * rs * gb[c] + gb[C + c];
        out[m][k * 2 + 1] = (p.y - mean) * rs * gb[c + 1] + gb[C + c + 1];
      }
    } else {
#pragma unroll
      for (int k = 0; k < 8; ++k) out[m][k] = 0.f;
    }
  }
  if (ROT == 1) {
    // Hadamard inside each block, normalised: 3 in-lane stages, log2(NB) shuffle stages
#pragma unroll
    for (int m = 0; m < NR; ++m)
#pragma unroll
      for (int st = 1; st < 8; st <<= 1)
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if ((k & st) == 0) {
            float a = out[m][k], c2 = out[m][k | st];
            out[m][k] = a + c2; out[m][k | st] = a - c2;
          }
#pragma unroll
    for (int st = 1; st < NB; st <<= 1)
#pragma unroll
      for (int m = 0; m < NR; ++m)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
          out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
        }
  }
  if (ROT == 2) {
    // Hadamard inside each block: 3 stages over the 8 elements of a vector ...
#pragma unroll
    for (int m = 0; m < NR; ++m)
#pragma unroll
      for (int st = 1; st < 8; st <<= 1)
#pragma unroll
        for (int k = 0; k < 8; ++k)
          if ((k & st) == 0) {
            float a = out[m][k], c2 = out[m][k | st];
            out[m][k] = a + c2; out[m][k | st] = a - c2;
          }
    // ... then log2(NB) stages across the lanes of the block (every lane takes part)
#pragma unroll
    for (int st = 1; st < NB; st <<= 1)
#pragma unroll
      for (int m = 0; m < NR; ++m)
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          float o = __shfl_xor_sync(0xffffffff, out[m][k], st);
          out[m][k] = (lane & st) ? (o - out[m][k]) : (out[m][k] + o);
        }
    // DCT across the nine blocks: y_j' = sum_j Q[j][j'] h_j at every position of the block
    const float hn = rsqrtf((float)(NB * 8));
    float qc[NR][9];
#pragma unroll
    for (int m = 0; m < NR; ++m) {
      const int jp = (lane + m * 32) / NB;          // this vector's block (>= 9 if invalid)
#pragma unroll
      for (int j = 0; j < 9; ++j) qc[m][j] = jp < 9 ? qtab[j * 9 + jp] * hn : 0.f;
    }
    const int q = lane & (NB - 1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      float val[9];
#pragma unroll
      for (int j = 0; j < 9; ++j)
        val[j] = __shfl_sync(0xffffffff, out[(j * NB) >> 5][k], ((j * NB) & 31) + q);
#pragma unroll
      for (int m = 0; m < NR; ++m) {
        float a = 0.f;
#pragma unroll
        for (int j = 0; j < 9; ++j) a = fmaf(qc[m][j], val[j], a);
        out[m][k] = a;
      }
    }
  }
#pragma unroll
  for (int m = 0; m < NR; ++m)
#pragma unroll
    for (int k = 0; k < 8; ++k) mx = fmaxf(mx, fabsf(out[m][k]));
#pragma unroll
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  // ROT == 1 leaves the Hadamard un-normalised; its 1/sqrt(n) is folded into the scale
  const float hn = ROT == 1 ? rsqrtf((float)(NB * 8)) : 1.f;
  const float sc = mx * hn * (1.f / 127.f) + 1e-12f, inv = hn / sc;
  if (lane == 0) ys[t] = sc;
#pragma unroll
  for (int m = 0; m < NR; ++m) {
    int v = lane + m * 32;
    if (v < NV) {
      int8_t q8[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        int r = __float2int_rn(out[m][k] * inv);
        q8[k] = (int8_t)max(-127, min(127, r));
      }
      *(uint2*)(y + (size_t)t * C + v * 8) = *(const uint2*)q8;
    }
  }
}

// Rotate a [K, N] fp16 weight along K, one block per output channel n, so that
// (x R) . out[:, n] == x . w[:, n] for the R the activation producer applies.
// KIND 0: R = Q9 (x) H_{K/9}, the k_ln_q8<.., 2> rotation.
// KIND 2: R = I_{K/n} (x) H_n with n = rot_block(K) (= K/9 for hiera-L), the k_ln_q8<.., 1>
//         rotation (what ships).
// KIND 1: block-diagonal Hadamard on bs-wide blocks (bs = 64 or 128; with 128 the last block
//         of K = 576 is a 64-wide remainder), the rotation the fc1 epilogue applies to fc2's input.
// Dynamic shared: 2 * K floats.
template<int KIND>
__global__ void k_rot_w(const half* __restrict__ w, half* __restrict__ out, int K, int N, int bs = 128) {
  extern __shared__ float rw[];
  float* a = rw; float* bb = rw + K;
  const int n = blockIdx.x;
  for (int k = threadIdx.x; k < K; k += blockDim.x) a[k] = __half2float(w[(size_t)k * N + n]);
  __syncthreads();
  if (KIND == 0 || KIND == 2) {
    const int nh = KIND == 0 ? K / 9 : rot_block(K);     // KIND 2: the k_ln_q8<.., 1> block
    for (int st = 1; st < nh; st <<= 1) {
      for (int i = threadIdx.x; i < K / 2; i += blockDim.x) {
        const int blk = i / (nh / 2), r = i - blk * (nh / 2);
        const int lo = (r / st) * (2 * st) + (r % st), k0 = blk * nh + lo, k1 = k0 + st;
        const float p = a[k0], qv = a[k1];
        a[k0] = p + qv; a[k1] = p - qv;
      }
      __syncthreads();
    }
    const float hn = rsqrtf((float)nh);
    if (KIND == 2) {
      for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)k * N + n] = __float2half(a[k] * hn);
      return;
    }
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const int jp = k / nh, p = k - jp * nh;
      float acc = 0.f;
      for (int j = 0; j < 9; ++j) acc = fmaf(c_dct9[j * 9 + jp], a[j * nh + p], acc);
      bb[k] = acc * hn;
    }
    __syncthreads();
    for (int k = threadIdx.x; k < K; k += blockDim.x) out[(size_t)k * N + n] = __float2half(bb[k]);
  } else {
    const int hb = bs / 2;
    for (int st = 1; st < bs; st <<= 1) {
      for (int i = threadIdx.x; i < ((K + bs - 1) / bs) * hb; i += blockDim.x) {
        const int blk = i / hb, r = i - blk * hb;
        const int lo = (r / st) * (2 * st) + (r % st), k0 = blk * bs + lo, k1 = k0 + st;
        if (k1 < K) { const float p = a[k0], qv = a[k1]; a[k0] = p + qv; a[k1] = p - qv; }
      }
      __syncthreads();
    }
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
      const int nb = min(bs, K - (k / bs) * bs);
      out[(size_t)k * N + n] = __float2half(a[k] * rsqrtf((float)nb));
    }
  }
}
