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

// nano_hiera.cu — variant-generic SAM 2.1 hiera image-encoder engine (T / S / B+ / L).
//
// nano_sam21.cu is the hiera-L engine: it keeps the whole trunk in window-major token order,
// which works because every window of hiera-L divides its map.  T / S / B+ use 14x14 and 7x7
// windows on 64x64 and 32x32 maps, which the official code zero-pads to 70x70 / 35x35 per
// block, so this engine keeps the residual stream in NATURAL row-major order and
//   * the LayerNorm writes its rows through a slot -> token map into a zero-padded
//     window-major buffer (pad slots get zero rows), which the qkv GEMM and the attention
//     consume as whole windows (the attention masks nothing: pad tokens are real zero rows,
//     exactly as in the reference, where they attend as keys);
//   * the attention output is gathered back into natural order (pad slots dropped) before
//     the proj GEMM adds the residual;
//   * the transition blocks pool the shortcut in natural order and the queries per window.
// Global-attention blocks and the neck run in natural order with no maps.  Everything else --
// the INT8 producers, the rotated LayerNorm, the int8 GEMMs with the fc1 -> fc2 group scale,
// the flash attention (templated on the head dim: 96 / 56 / 72) -- is shared with nano_sam21.
//
// --batch N runs N images per launch.  The batch is ONE concatenated token stream, not a
// separate axis: image i owns natural tokens [i*T, (i+1)*T) of every stage buffer and slots
// [i*Tp, (i+1)*Tp) of every window-major buffer, so every GEMM is the batch-1 GEMM with M
// multiplied by N, the attention's blockIdx.y counts windows over the whole batch (window
// w = i*nW + w_local, which is what its w*ws2 row base already means), and the only padding
// left is the tail that rounds the concatenated stream up to the 128-row GEMM boundary.  The
// two token maps (slot -> natural, natural -> attention slot) carry the per-image offsets, so
// the LayerNorm / gather kernels are unchanged and N = 1 rebuilds the batch-1 maps exactly.
//
// build: nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 nano_hiera.cu -lcublasLt -o nano_hiera.exe
// run:   nano_hiera.exe --model hiera/tiny --input bench/input_img.bin --out-dir out --f16acc --int8-fc2 --rot
//        nano_hiera.exe --model hiera/tiny --batch 8 --f16acc

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CL(x) do{ cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  printf("cublasLt %s:%d status=%d\n",__FILE__,__LINE__,(int)s_); exit(1);} }while(0)

// The legacy (non-templated) attention headers take DH / DHP from the including file; the
// engine itself reads the head dim off the model and dispatches on it (see runAttnG).
static const int DH  = 72;
static const int DHP = 80;
#define ATTN2_DH_TEMPLATE 1        // sam21_attn2.cuh: k_attn2<DH, ...>, dispatched on cfg.dh below

#include "sam21_kernels_mixed.cuh"
#include "sam21_gemm_res.cuh"   // sm_120: fp16 GEMM + bias + residual (proj, fp16 fc2)
#include "sam21_attn2q8.cuh"    // --qk8: QK^T on int8 tensor cores, per-(token, head) scales
#include "sam21_attn2w8.cuh"    // --win8: int8 Q/K/V storage for the windowed attention, fp16 math
#include "sam21_int4.cuh"       // --int4 (W4A4, w4a4_engine prototype): k_quant_w4, k_pack_s4
#include "sam21_int4h.cuh"      // --int4 on hiera (w4a4_hiera): k_ln_q4h, k_pack_s4p, k_ln_mon4 (--mon)

#include "dhd_producer.cuh"
#include "dhd_gemm.cuh"
#include "i4_persist.cuh"       // int4_gemm_opt_20261014: persistent qkv (W8G16) / DHD fc1 INT4 GEMMs

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }
static inline int roundup(int a, int b) { return cdiv(a, b) * b; }

// ---------------------------------------------------------------- model config
static int gB = 1;              // --batch: images per launch
struct Blk {
  int din, dout, heads, ws, qpool, cp;
  // derived geometry (one image)
  int H, T, Hq, Tq;           // input map side / tokens, output (query) side / tokens
  int nWx, nW, Hp, Tp, Tpr;   // windows per side, windows, padded side, padded slots, GEMM rows
  int wo, Tpo;                // output window side (ws/2 if pooled), attention output slots
  int stage;
  // the same counts over the whole batch (applyBatch): the images are concatenated, so only
  // the GEMM row padding of the window-major stream (Tprb) is not gB times the batch-1 count
  int Tb, Tqb, Tpb, Tprb, Tpob, nWb;
};
struct Cfg { std::string name; int dims[4], heads[4], dh, dhp, nblk, stage_ends[4]; std::vector<Blk> b; };
static Cfg cfg;

static double jnum(const std::string& js, const char* key, double dflt) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) return dflt;
  p = js.find(':', p); return atof(js.c_str() + p + 1);
}
static void jarr(const std::string& js, const char* key, int* out, int n) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) { printf("config: no %s\n", key); exit(1); }
  p = js.find('[', p) + 1;
  for (int i = 0; i < n; ++i) { out[i] = atoi(js.c_str() + p); p = js.find(',', p) + 1; }
}
static void loadCfg(const std::string& prefix) {
  std::ifstream f(prefix + ".json");
  if (!f) { printf("cannot open %s.json\n", prefix.c_str()); exit(1); }
  std::stringstream ss; ss << f.rdbuf(); std::string js = ss.str();
  size_t p = js.find("\"name\""); p = js.find('"', js.find(':', p)) + 1;
  cfg.name = js.substr(p, js.find('"', p) - p);
  jarr(js, "dims", cfg.dims, 4); jarr(js, "heads", cfg.heads, 4); jarr(js, "stage_ends", cfg.stage_ends, 4);
  cfg.dh = (int)jnum(js, "dh", 72); cfg.dhp = (int)jnum(js, "dhp", 80); cfg.nblk = (int)jnum(js, "nblk", 48);
  size_t bp = js.find("\"blocks\"");
  int side = 256, stage = 0;
  for (int i = 0; i < cfg.nblk; ++i) {
    size_t o = js.find('{', bp), e = js.find('}', o); bp = e + 1;
    std::string ob = js.substr(o, e - o + 1);
    Blk b{};
    if ((int)jnum(ob, "i", -1) != i) { printf("config: block %d out of order\n", i); exit(1); }
    b.din = (int)jnum(ob, "din", 0); b.dout = (int)jnum(ob, "dout", 0); b.heads = (int)jnum(ob, "heads", 0);
    b.ws = (int)jnum(ob, "ws", 0); b.qpool = ob.find("\"qpool\": true") != std::string::npos; b.cp = (int)jnum(ob, "cp", 0);
    if (b.qpool) ++stage;
    b.stage = stage;
    b.H = side; b.T = side * side; b.Hq = b.qpool ? side / 2 : side; b.Tq = b.Hq * b.Hq;
    if (b.ws) {
      b.nWx = cdiv(b.H, b.ws); b.nW = b.nWx * b.nWx; b.Hp = b.nWx * b.ws;
      b.Tp = b.nW * b.ws * b.ws; b.Tpr = roundup(b.Tp, 128);
      b.wo = b.qpool ? b.ws / 2 : b.ws; b.Tpo = b.nW * b.wo * b.wo;
    } else {                                 // global attention: one window = the whole map
      b.nWx = 1; b.nW = 1; b.Hp = b.H; b.Tp = b.T; b.Tpr = b.T; b.wo = b.H; b.Tpo = b.T;
      if (b.qpool) { printf("config: pooled global block %d unsupported\n", i); exit(1); }
    }
    cfg.b.push_back(b);
    side = b.Hq;
  }
  printf("model %s: dims %d/%d/%d/%d heads %d.. dh %d (DHP %d), %d blocks\n", cfg.name.c_str(),
         cfg.dims[0], cfg.dims[1], cfg.dims[2], cfg.dims[3], cfg.heads[0], cfg.dh, cfg.dhp, cfg.nblk);
}
// batch-wide token counts.  gB = 1 reproduces the batch-1 geometry exactly: Tprb = roundup(Tp,
// 128) = Tpr, and a global block's Tp = T is already a multiple of 128.
static void applyBatch() {
  for (Blk& b : cfg.b) {
    b.Tb = gB * b.T; b.Tqb = gB * b.Tq; b.Tpb = gB * b.Tp;
    b.Tprb = roundup(b.Tpb, 128); b.Tpob = gB * b.Tpo; b.nWb = gB * b.nW;
    if ((long long)b.Tprb * 3 * b.cp > 2147483647ll) {   // the GEMM kernels index tiles with int
      printf("--batch %d: block qkv buffer %lld elements exceeds the int index range; use a smaller batch\n",
             gB, (long long)b.Tprb * 3 * b.cp);
      exit(1);
    }
  }
}

// ---------------------------------------------------------------- weight store
struct Tensor { void* d; size_t n; bool f16; };
static std::map<std::string, Tensor> W;
static void* gBlob = nullptr;
static void loadWeights(const std::string& prefix) {
  std::ifstream mf(prefix + ".manifest");
  std::ifstream bf(prefix + ".bin", std::ios::binary);
  if (!mf || !bf) { printf("cannot open weights %s.{manifest,bin}\n", prefix.c_str()); exit(1); }
  bf.seekg(0, std::ios::end); size_t bytes = (size_t)bf.tellg(); bf.seekg(0);
  std::vector<char> host(bytes); bf.read(host.data(), bytes);
  CK(cudaMalloc(&gBlob, bytes + 65536));
  CK(cudaMemset((char*)gBlob + bytes, 0, 65536));
  CK(cudaMemcpy(gBlob, host.data(), bytes, cudaMemcpyHostToDevice));
  std::string line;
  while (std::getline(mf, line)) {
    if (line.empty()) continue;
    std::istringstream ss(line);
    std::string name, dt, shape; size_t off, num;
    ss >> name >> dt >> off >> num >> shape;
    W[name] = {(char*)gBlob + off, num, dt == "f16"};
  }
  printf("weights: %zu tensors, %.1f MB\n", W.size(), bytes / 1e6);
}
static const void* WP(const std::string& k) {
  auto it = W.find(k);
  if (it == W.end()) { printf("missing weight %s\n", k.c_str()); exit(1); }
  return it->second.d;
}
static std::string bk(int i, const char* s) { char b[64]; snprintf(b, sizeof b, "b%d.%s", i, s); return b; }
static void writeBin(const char* path, const void* src, size_t bytes) {
  std::vector<char> h(bytes);
  CK(cudaMemcpy(h.data(), src, bytes, cudaMemcpyDeviceToHost));
  std::ofstream f(path, std::ios::binary); f.write(h.data(), bytes);
}
// reads up to `bytes`; returns how many bytes the file actually held (the rest stays zero)
static size_t readBin(const char* path, void* dst, size_t bytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot read %s\n", path); exit(1); }
  std::vector<char> h(bytes); f.read(h.data(), bytes);
  size_t got = (size_t)f.gcount();
  CK(cudaMemcpy(dst, h.data(), bytes, cudaMemcpyHostToDevice));
  return got;
}

// ---------------------------------------------------------------- kernels specific to this engine
__global__ void k_quant_w(const half* __restrict__ w, int8_t* __restrict__ wt, float* __restrict__ sw, int K, int N) {
  const int n = blockIdx.x;
  __shared__ float red[32];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) mx = fmaxf(mx, fabsf(__half2float(w[(size_t)k * N + n])));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = (threadIdx.x < (blockDim.x + 31) / 32) ? red[threadIdx.x] : 0.f;
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  const float sc = red[0] * (1.f / 127.f) + 1e-12f, inv = 1.f / sc;
  if (threadIdx.x == 0) sw[n] = sc;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = __float2int_rn(__half2float(w[(size_t)k * N + n]) * inv);
    wt[(size_t)n * K + k] = (int8_t)max(-127, min(127, q));
  }
}
__global__ void k_wsum(const int8_t* __restrict__ wt, int K, int N, float* __restrict__ out) {
  const int n = blockIdx.x;
  long long acc = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) acc += wt[(size_t)n * K + k];
  __shared__ long long red[32];
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) { long long t = 0; for (int i = 0; i < (blockDim.x + 31) / 32; ++i) t += red[i]; out[n] = (float)t; }
}
__global__ void k_bias_shift(const half* __restrict__ bias, const float* __restrict__ sw, const float* __restrict__ wsum,
                             float shift, int N, half* __restrict__ out) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) out[n] = __float2half(__half2float(bias[n]) - shift * sw[n] * wsum[n]);
}
// 2x2 max-pool of a natural-order [B*H*H, C] map into [B*H/2*H/2, C]
__global__ void k_maxpool_nat(const half* __restrict__ src, half* __restrict__ dst, int H, int C, int B) {
  const int Hq = H / 2, Tq = Hq * Hq, T = H * H;
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)B * Tq * (C / 2)) return;
  int c2 = (int)(idx % (C / 2)); size_t g = idx / (C / 2);
  int bi = (int)(g / Tq), t = (int)(g - (size_t)bi * Tq);
  int y = t / Hq, x = t - y * Hq;
  const half2* s = (const half2*)src + (size_t)bi * T * (C / 2);
  float2 mx = make_float2(-65504.f, -65504.f);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      float2 p = __half22float2(s[(size_t)((2 * y + a) * H + 2 * x + b) * (C / 2) + c2]);
      mx.x = fmaxf(mx.x, p.x); mx.y = fmaxf(mx.y, p.y);
    }
  ((half2*)dst)[g * (C / 2) + c2] = __floats2half2_rn(mx.x, mx.y);
}
// patch-embed im2col over a batch of CHW fp32 images, natural row-major slots
// (the batch-1 engine got the same rows through an identity slot -> token map).
__global__ void k_im2col_b(const float* __restrict__ img, half* __restrict__ col,
                           int OH, int IH, int B) {
  int j = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;   // token of the batch
  if (j >= B * OH * OH) return;
  int lane = threadIdx.x & 31;
  int bi = j / (OH * OH), n = j - bi * OH * OH, oy = n / OH, ox = n % OH;
  const float* im = img + (size_t)bi * 3 * IH * IH;
  half* dst = col + (size_t)j * 152;
  for (int t = lane; t < 152; t += 32) {
    float v = 0.f;
    if (t < 147) {
      int c = t / 49, r = t - c * 49, ky = r / 7, kx = r - ky * 7;
      int iy = oy * 4 - 3 + ky, ix = ox * 4 - 3 + kx;
      if ((unsigned)iy < (unsigned)IH && (unsigned)ix < (unsigned)IH)
        v = im[((size_t)c * IH + iy) * IH + ix];
    }
    dst[t] = __float2half(v);
  }
}
// the [T, C] patch-embed position table replicated over the batch (the patch GEMM adds it with beta = 1)
__global__ void k_pos_rep(const half* __restrict__ pos, half* __restrict__ out, int T, int C, int B) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)B * T * (C / 8)) return;
  size_t t = i / (C / 8); int c8 = (int)(i % (C / 8));
  ((float4*)(out + t * C))[c8] = ((const float4*)(pos + (size_t)(t % (size_t)T) * C))[c8];
}
// natural-order [B*T, C] fp16 -> [B, C, T] fp32 (the batch-1 engine used k_scatter_nchw with
// an identity map, which is this with B = 1)
__global__ void k_scatter_nchw_b(const half* __restrict__ src, float* __restrict__ dst,
                                 int T, int C, int B) {
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)B * T * C) return;
  int c = (int)(idx % C); size_t n = idx / C;
  int bi = (int)(n / T), nl = (int)(n - (size_t)bi * T);
  dst[((size_t)bi * C + c) * T + nl] = __half2float(src[n * C + c]);
}
// vision_features = neck.c1(x_s2) + nearest_2x(neck.c0(x_s3)), batched, emitted as NCHW fp32
__global__ void k_vision_feat_b(const half* __restrict__ lat2, const half* __restrict__ lat3,
                                float* __restrict__ dst, int H2, int C, int B) {
  const int T2 = H2 * H2, T3 = (H2 >> 1) * (H2 >> 1);
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (size_t)B * T2 * C) return;
  int c = (int)(idx % C); size_t n = idx / C;
  int bi = (int)(n / T2), nl = (int)(n - (size_t)bi * T2);
  int y = nl / H2, x = nl - y * H2;
  int n3 = (y >> 1) * (H2 >> 1) + (x >> 1);
  dst[((size_t)bi * C + c) * T2 + nl] = __half2float(lat2[n * C + c]) +
                                        __half2float(lat3[((size_t)bi * T3 + n3) * C + c]);
}
// pooled Q for the transition blocks: 2x2 max over each window, padded head layout (runtime dhp)
__global__ void k_pool_q_g(const half* __restrict__ qkv, half* __restrict__ Qb, int nW, int ws, int heads, int Cp, int dhp) {
  int ws2 = ws * ws, wsq = ws / 2, ws2q = ws2 / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nW * heads * ws2q * (dhp / 2)) return;
  int d2 = idx % (dhp / 2), rest = idx / (dhp / 2);
  int tq = rest % ws2q, b = rest / ws2q;
  int h = b % heads, w = b / heads;
  int oh = tq / wsq, ow = tq - oh * wsq;
  const half2* base = (const half2*)(qkv + (size_t)(w * ws2) * (3 * Cp) + h * dhp);
  int row3 = (3 * Cp) / 2;
  float2 mx = make_float2(-65504.f, -65504.f);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float2 p = __half22float2(base[(size_t)((2 * oh + a) * ws + (2 * ow + c)) * row3 + d2]);
      mx.x = fmaxf(mx.x, p.x); mx.y = fmaxf(mx.y, p.y);
    }
  ((half2*)(Qb + (size_t)(b * ws2q + tq) * dhp))[d2] = __floats2half2_rn(mx.x, mx.y);
}

// --qk8 stand-alone producer: int8 Q and K with one scale per (token, head), one warp per
// (token, group), group g < 2*heads (Q heads, then K heads).  Pad columns DH..DH8-1 are zero.
__global__ void k_quant_qk(const half* __restrict__ qkv, int8_t* __restrict__ q8, float* __restrict__ s8,
                           int T, int heads, int Cp, int dh, int dhp, int dh8, int sst) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wid >= T * 2 * heads) return;
  const int t = wid / (2 * heads), gq = wid - t * 2 * heads;
  const int part = gq / heads, h = gq - part * heads;
  const half* src = qkv + (size_t)t * 3 * Cp + part * Cp + h * dhp;
  float mx = 0.f;
  for (int d = lane; d < dh; d += 32) mx = fmaxf(mx, fabsf(__half2float(src[d])));
  for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
  const float sc = mx * (1.f / 127.f) + 1e-12f, inv = 1.f / sc;
  int8_t* dst = q8 + (size_t)t * (2 * heads * dh8) + gq * dh8;
  for (int d = lane; d < dh8; d += 32)
    dst[d] = d < dh ? (int8_t)max(-127, min(127, __float2int_rn(__half2float(src[d]) * inv))) : (int8_t)0;
  if (lane == 0) s8[(size_t)gq * sst + t] = sc;
}

// --fc2-pass stand-alone producer of fc2's int8 operand from the fp16 GELU output [M, N] of fc1:
// the block Hadamard (H_64 per 64 columns; ROT = 128 pairs the two halves of a 128-column group),
// the (row, 128-column group) absmax scale and the int8 codes, in the layout the GROUP = 128 fc2
// GEMM consumes (out [M, N] int8, sa2 [M][sastride]).  The arithmetic is that of the fused ROT8
// epilogue of k_gemm_s8 (which also rotates the fp16-rounded staged tile), so the two agree bit
// for bit.  Two threads per (row, group), 64 fp32 columns each on registers, 16-byte loads and
// stores; the 64-wide remainder group of N = 576 is one thread.  Grid (cdiv(N, 128), M / 128).
template<int ROT>
__global__ void __launch_bounds__(256) k_rotquant_fc2(const half* __restrict__ in, int8_t* __restrict__ out,
                                                      float* __restrict__ sa2, int N, int sastride) {
  const int hh = threadIdx.x & 1, r = blockIdx.y * 128 + (threadIdx.x >> 1), n0 = blockIdx.x * 128;
  const bool full = n0 + 128 <= N;            // block-uniform
  float v[64];
  if (full || hh == 0) {
    const uint4* src = (const uint4*)(in + (size_t)r * N + n0 + hh * 64);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const uint4 u = src[i];
      const half2* h2 = (const half2*)&u;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        const float2 f = __half22float2(h2[k]);
        v[i * 8 + 2 * k] = f.x; v[i * 8 + 2 * k + 1] = f.y;
      }
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; ++i) v[i] = 0.f;
  }
#pragma unroll
  for (int st = 1; st < 64; st <<= 1)
#pragma unroll
    for (int i = 0; i < 64; ++i)
      if ((i & st) == 0) { const float a = v[i], b2 = v[i | st]; v[i] = a + b2; v[i | st] = a - b2; }
  const bool pair = ROT == 128 && full;
  if (pair) {
#pragma unroll
    for (int i = 0; i < 64; ++i) {
      const float o = __shfl_xor_sync(0xffffffff, v[i], 1);
      v[i] = hh ? (o - v[i]) : (v[i] + o);
    }
  }
  const float hn = rsqrtf(pair ? 128.f : 64.f);
  float mx = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) mx = fmaxf(mx, fabsf(v[i]));
  if (full) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
  const float sc = mx * hn * (1.f / 127.f) + 1e-12f;
  const float inv = hn / sc;
  if (hh == 0) sa2[(size_t)r * sastride + blockIdx.x] = sc;
  if (full || hh == 0) {
    uint32_t w[16];
#pragma unroll
    for (int i = 0; i < 16; ++i)
      w[i] = (uint32_t)(s8k::cvt_s8_sat(v[4 * i] * inv) & 0xff) | ((uint32_t)(s8k::cvt_s8_sat(v[4 * i + 1] * inv) & 0xff) << 8) |
             ((uint32_t)(s8k::cvt_s8_sat(v[4 * i + 2] * inv) & 0xff) << 16) | ((uint32_t)s8k::cvt_s8_sat(v[4 * i + 3] * inv) << 24);
    uint4* dst = (uint4*)(out + (size_t)r * N + n0 + hh * 64);
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = make_uint4(w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3]);
  }
}

// ---------------------------------------------------------------- cuBLASLt plans (as nano_sam21.cu)
static cublasLtHandle_t LT;
static void* gWs = nullptr;
static size_t gWsBytes = 96ull << 20;
static int   gAccMode = 0;
static bool  gF16Acc = false;
static int   gAttnAcc = -1;
static bool  gArch80 = false, gArch89 = false;
static bool  gArch120 = false;    // sm_120 (RTX 5090, 170 SMs): per-kernel choices, see the gArch120 sites
static bool  gArch86 = false;     // sm_86 (RTX 3090, A10): the d = 96 global block prefers k_attn2m there
static bool  gInt8 = false, gInt8Fc2 = false, gRot = false, gFc2U8 = true, gLtGelu = false;
static int   gFc2Rot = 64;
// --fp16-blocks N: the first N transformer blocks keep the fp16 path (fp16 LayerNorm ->
// cuBLASLt / k_gemm_gelu with the untouched fp16 weights) even when --int8 / --int8-fc2 /
// --rot / --gptq are on.  The int8 weights of those blocks are still built (and rotated /
// GPTQ-loaded) but never read.  0 = every block on int8, i.e. the shipped behaviour.
static int   gFp16Blocks = 0;
// --qk8: the global-attention blocks run QK^T on int8 (sam21_attn2q8.cuh).  Q and K get one
// scale per (token, head), the free axis of QK^T; V and P.V stay fp16.  Off = the fp16
// attention of every block, byte for byte.
static bool  gQK8 = false;
// --qk8-win: the windowed blocks too (not the query-pooling ones).  Not worth it on hiera-T at
// batch 8 (3090): the 8x8 windows gain 0.10 ms and the 14x14 ones 0.01 ms each, while the
// 192-wide qkv tile wastes a quarter of stage 0's 288-column GEMM (+0.5 ms).  Short windows
// are softmax- and load-bound, not QK^T-bound.
static bool  gQK8Win = false;
static bool qk8Block(const Blk& b) { return gQK8 && (!b.ws || (gQK8Win && b.ws * b.ws >= 32)) && !b.qpool; }
static bool  gQK8Pass = false;    // --qk8-pass: quantise Q/K in a separate pass even where the qkv GEMM could
// --fc2-pass: fc1 writes its plain fp16 GELU output and a stand-alone kernel (k_rotquant_fc2)
// rotates and quantises it into fc2's int8 operand, instead of the fused ROT8 / Q8OUT epilogue.
// Same rotation, scales and layout; measures what a separate rotate-and-quantise pass costs.
static bool  gFc2Pass = false;
// --win8: "storage-only INT8" for the windowed blocks (not the query-pooling ones, not the global
// ones).  The int8 qkv GEMM writes Q, K and V as int8 with one fp16 scale per (token, 16 channels)
// (k_gemm_s8<..., W8G16>), and the attention (k_attn2w8, sam21_attn2w8.cuh) dequantises them into
// its fp16 shared tiles / fragments, so every byte of math is k_attn2's; only the bytes the qkv
// GEMM writes and the attention reads shrink (1 + 2/16 bytes per element instead of 2).
// --win8-glob: the global blocks that do not run --qk8 as well (hiera-L's three).
// --win8-pool: the query-pooling (transition) blocks as well: Q, K and V int8, the 2x2 max-pooled Q
// made from the int8 Q inside the attention kernel (the environment NANO_W8_POOLPASS=1 instead runs
// the stand-alone k_pool_q_w8 into the fp16 pooled-Q buffer, as the fp16 path does; same values).
// Off = the engine as it was, byte for byte.  Needs the int8 path (the qkv GEMM is the producer).
static bool  gWin8 = false, gWin8Glob = false, gWin8Pool = false, gW8PoolPass = false;
// the blocks it covers: windowed and not pooling (plus the global ones under --win8-glob and the
// pooling ones under --win8-pool), except where --qk8 already owns the block
static bool win8Block(const Blk& b) {
  return gWin8 && !qk8Block(b) && (b.qpool ? gWin8Pool : (b.ws || gWin8Glob));
}
// --int4 (W4A4, w4a4_hiera): qkv, fc1 and fc2 on mma.m16n8k64 .s4 with packed int4 operands (k_gemm_s8<..., I4>),
// on the INT8 path's structure (implies --int8-fc2 --rot): the LayerNorm emits rotated per-token int4 (k_ln_q4h),
// the fc1 epilogue H_64 + per-(token, 128) int4 for fc2, weights per output channel 4-bit (RTN at start-up, or
// the WBITS = 4 GPTQ codes of --gptq).  proj, attention (incl. --win8 / --qk8, whose producers stay in the qkv
// epilogue), sproj and the neck are the INT8 path's.  Rows of K = 144 are padded to 160 codes (kpad4).
// --mon <file>: the LayerNorm-fed producers apply T = B1 Pi B2 before the Hadamard (k_ln_mon4, tensor cores),
// one T per qkv / fc1 GEMM from the NANOMON1 file (tools/export_mon.py); --mon identity uses B1 = B2 = I
// (the same kernels and work, T = I: for timing and for checking against --int4).  The weights must carry
// W T^-T (the GPTQ file of the mon scheme); RTN start-up weights do not (timing only).
static bool  gInt4 = false;
static std::string gDhdFile;
static bool gDhdIdentity=false;
struct DhdGPU { int K=0,H=0;bool diagonals=true;float clip=1.f;float *d1=nullptr,*d2=nullptr,*gamma=nullptr,*beta=nullptr;bool d2unit=false; };
static std::vector<DhdGPU> gDhd1,gDhd2,gDhd3;
static void buildDhd();
// --fc2-static <file> (fc2static_engine_20261016, needs --dhd): STATIC fc2-input activation scales from a NANOF2S1 file
// (scripts/export_fc2static.py).  The fc1 epilogue quantises the rotated GELU output with a constant scale per
// (layer, 128-column group) instead of the per-(token, group) absmax, and fc2 runs the INT4 GEMM WITHOUT the per-group
// fp32 flush (GROUP = 0, one scalar row scale).  Two modes, both in the file:
//   1 sym    one scale s per layer (all groups equal); the fc2 GEMM's scalar is s
//   2 group  one scale s_g per 128-column group, folded into the fc2 weight columns by the exporter (weights.bin);
//            the fc2 GEMM's scalar is 1
// Off (no flag) = engine_new2 byte for byte.  Timing-only switches (wrong numerics, cost attribution only):
//   NANO_T_NOLNH=1   the LayerNorm producers of qkv / fc1 without the Hadamard butterflies
//   NANO_T_NOFC1H=1  the fc1 epilogue without the 64-point butterfly
//   NANO_T_NOFLUSH=1 (dynamic fc2 only) fc2 on the no-flush GEMM at the flush path's tile (row scale = HS[0])
struct F2SGPU { int K = 0, G = 0; float* sinv = nullptr; float* sgemm = nullptr; };
static std::string gFc2StaticFile;
static int gFc2StaticMode = 0;           // 0 off (dynamic), 1 sym, 2 group
static std::vector<F2SGPU> gF2S;
static bool gTNoLnH = false, gTNoFc1H = false, gTNoFlush = false;
// 16:50 (main-table campaign), the static path only:
//   NANO_T_NOFC1S=1  TIMING ONLY: the fc1 epilogue without the D1 (= S) multiply and its staging
//   NANO_T_NOD2=1    without the D2 multiplies (LayerNorm producers and fc1 epilogue): exact when d2 = 1 (SHAQuant exports)
// and a fixed-H export (no diagonals) now runs the static fc1 producer without any diagonal (as the dynamic path does).
static bool gTNoFc1S = false, gTNoD2 = false;
// fc2static_engine_20261016 fc2s4 (user decision 21:00): on the --fc2-static path the D2 multiply of a producer (LayerNorm
// of qkv / fc1, fc1 epilogue) is skipped by DEFAULT when every d2 of that layer is 1 (x * 1.0f == x: the outputs are the
// same bits; every SHAQuant / SGHAQuant export has d2 = 1).  NANO_D2_KEEP=1 restores the fc2s3 kernels (A/B).  The dynamic
// fc2 path (no --fc2-static) is left exactly as engine_new2.
static bool gD2Keep = false;
static bool d2Skip(const DhdGPU& t) { return gTNoD2 || (gFc2StaticMode && t.d2unit && !gD2Keep); }
static void buildFc2Static();
static std::string gMonFile;              // empty = no mon
static int8_t* gW4tmp = nullptr;          // int8 codes [N, K] before packing
struct MonLN { int C = 0, BB = 0; uint2* f1 = nullptr; uint32_t* f1t = nullptr; uint2* f2 = nullptr; uint32_t* f2t = nullptr; };
static std::vector<MonLN> gMon1, gMon2;   // per block: norm1 -> qkv (C = din), norm2 -> fc1 (C = dout)
static std::string gGptq, gDump,gFc2External;static int gFc2FixedBits=0;
static bool  gTune = true, gProf = false;
static bool  gServe = false;           // --serve: see main (w4a4_hiera accuracy harness)
static half *gTA = nullptr, *gTB = nullptr, *gTD = nullptr;

struct Plan { cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t La, Lb, Lc; cublasLtMatmulAlgo_t algo; int m, n, k; bool beta1, f16acc; };
static std::vector<Plan> gPlans;
static int makePlan(int m, int n, int k, cublasLtEpilogue_t epi, const void* bias, bool beta1, bool deepK = false) {
  Plan p{}; p.m = m; p.n = n; p.k = k; p.beta1 = beta1;
  gF16Acc = (gAccMode == 2) || (gAccMode == 1 && !deepK);
  p.f16acc = gF16Acc;
  cublasComputeType_t comp = gF16Acc ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
  cudaDataType_t st = gF16Acc ? CUDA_R_16F : CUDA_R_32F;
  CL(cublasLtMatmulDescCreate(&p.op, comp, st));
  cublasOperation_t tn = CUBLAS_OP_N;
  CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn)));
  CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  if (epi != CUBLASLT_EPILOGUE_DEFAULT) {
    CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi)));
    cudaDataType_t bt = CUDA_R_16F;
    CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
    CL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
  }
  CL(cublasLtMatrixLayoutCreate(&p.La, CUDA_R_16F, n, k, n));
  CL(cublasLtMatrixLayoutCreate(&p.Lb, CUDA_R_16F, k, m, k));
  CL(cublasLtMatrixLayoutCreate(&p.Lc, CUDA_R_16F, n, m, n));
  cublasLtMatmulPreference_t pref;
  CL(cublasLtMatmulPreferenceCreate(&pref));
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &gWsBytes, sizeof(gWsBytes)));
  const int NCAND = 12;
  cublasLtMatmulHeuristicResult_t hr[NCAND]; int found = 0;
  CL(cublasLtMatmulAlgoGetHeuristic(LT, p.op, p.La, p.Lb, p.Lc, p.Lc, pref, NCAND, hr, &found));
  if (!found) { printf("no algo for %dx%dx%d epi=%d\n", m, n, k, (int)epi); exit(1); }
  p.algo = hr[0].algo;
  if (gTune && found > 1) {
    static const float a32 = 1.f, b32 = 0.f;
    static const half a16 = __float2half(1.f), b16 = __float2half(0.f);
    const void* al = gF16Acc ? (const void*)&a16 : (const void*)&a32;
    const void* be = gF16Acc ? (const void*)&b16 : (const void*)&b32;
    float best = 1e30f;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    static const bool dbg = getenv("NANO_TUNE_DEBUG") != nullptr;
    for (int c = 0; c < found; ++c) {
      if (dbg) { printf("tune m=%d n=%d k=%d epi=%d cand %d/%d ws=%zu\n", m, n, k, (int)epi, c, found, hr[c].workspaceSize); fflush(stdout); }
      if (cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0) != CUBLAS_STATUS_SUCCESS) continue;
      for (int r = 0; r < 3; ++r) cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0));
      for (int r = 0; r < 12; ++r) cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
      CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
      float ms; CK(cudaEventElapsedTime(&ms, e0, e1));
      if (ms < best) { best = ms; p.algo = hr[c].algo; }
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
  }
  cublasLtMatmulPreferenceDestroy(pref);
  gPlans.push_back(p);
  return (int)gPlans.size() - 1;
}
static void runPlan(int id, const void* A, const void* B, const void* C, void* D, cudaStream_t s) {
  const Plan& p = gPlans[id];
  static const float a32 = 1.f, b32_0 = 0.f, b32_1 = 1.f;
  static const half a16 = __float2half(1.f), b16_0 = __float2half(0.f), b16_1 = __float2half(1.f);
  const void* alpha = p.f16acc ? (const void*)&a16 : (const void*)&a32;
  const void* beta = p.f16acc ? (p.beta1 ? (const void*)&b16_1 : (const void*)&b16_0) : (p.beta1 ? (const void*)&b32_1 : (const void*)&b32_0);
  CL(cublasLtMatmul(LT, p.op, alpha, B, p.La, A, p.Lb, beta, p.beta1 ? C : D, p.Lc, D, p.Lc, &p.algo, gWs, gWsBytes, s));
}

// ---------------------------------------------------------------- buffers and maps
static half *Xs[4];                      // residual stream per stage, natural order
static half *Xn, *QKV, *HID, *AO, *SC, *Qb, *COL, *POS, *L2, *L3, *S4b, *S8b;
static float *dIn, *dVF, *dS4, *dS8;
static int8_t *XQ = nullptr, *HQ = nullptr;
static float *XS = nullptr, *HS = nullptr;
static int8_t *QK8 = nullptr;             // --qk8: int8 Q and K, [rows, 2*heads*DH8]
static float  *QKS = nullptr;             // their scales, [2*heads][gQKSst]
static int     gQKSst = 0;
static int8_t *W8Q = nullptr;             // --win8: int8 Q/K/V, [rows, 3*cp] (the fp16 qkv layout)
static half   *W8S = nullptr;             // their fp16 scales, [3*cp/16][gW8Sst], gW8Sst = the block's Tprb + 128
static int     gW8Sst = 0;
struct Q8W { int8_t* w; float* s; };
static std::vector<Q8W> QWqkv, QWfc1, QWfc2,MixedFc2;
static void buildInt4Pitched();   // int4_gemm_opt_20261014 (row-padded qkv / fc1 weights for the persistent GEMMs)
static std::vector<half*> Bfc2;
static half* gRotTmp = nullptr;
// per block: slot -> natural token (LN write, -1 = pad) and natural query -> output slot (gather)
static std::vector<int*> mSlot2Nat, mNat2Slot;

// every activation buffer carries 64 KB of zeroed slack: some cuBLASLt heuristic kernels over-read
// their last tile (seen as an illegal address on the 3090 with K = 96, tuning on), and the attention
// reads whole key tiles past a window
static half* dAlloc(size_t n) { void* p; CK(cudaMalloc(&p, n * sizeof(half) + 65536)); CK(cudaMemset(p, 0, n * sizeof(half) + 65536)); return (half*)p; }
static int* upload(const std::vector<int>& v) {
  void* p; CK(cudaMalloc(&p, v.size() * 4)); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return (int*)p;
}

static void buildMaps() {
  std::map<std::string, int*> cache;
  mSlot2Nat.assign(cfg.nblk, nullptr); mNat2Slot.assign(cfg.nblk, nullptr);
  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& b = cfg.b[i];
    if (!b.ws) continue;
    char key[64];
    snprintf(key, sizeof key, "s%d_%d", b.H, b.ws);
    if (!cache.count(key)) {
      // image bi holds window slots [bi*Tp, (bi+1)*Tp) and natural tokens [bi*T, (bi+1)*T);
      // the tail up to the 128-row GEMM boundary is pad, exactly as at batch 1
      std::vector<int> v(b.Tprb, -1);
      for (int bi = 0; bi < gB; ++bi)
        for (int s = 0; s < b.Tp; ++s) {
          int w = s / (b.ws * b.ws), r = s % (b.ws * b.ws);
          int a = w / b.nWx, c = w % b.nWx, y = a * b.ws + r / b.ws, x = c * b.ws + r % b.ws;
          v[bi * b.Tp + s] = (y < b.H && x < b.H) ? bi * b.T + y * b.H + x : -1;
        }
      cache[key] = upload(v);
    }
    mSlot2Nat[i] = cache[key];
    snprintf(key, sizeof key, "o%d_%d_%d", b.Hq, b.wo, b.nWx);
    if (!cache.count(key)) {
      std::vector<int> v(b.Tqb);
      for (int bi = 0; bi < gB; ++bi)
        for (int n = 0; n < b.Tq; ++n) {
          int y = n / b.Hq, x = n % b.Hq;
          int a = y / b.wo, c = x / b.wo;
          v[bi * b.Tq + n] = bi * b.Tpo + (a * b.nWx + c) * b.wo * b.wo + (y % b.wo) * b.wo + (x % b.wo);
        }
      cache[key] = upload(v);
    }
    mNat2Slot[i] = cache[key];
  }
}

static void allocAll() {
  size_t mxXn = 0, mxQKV = 0, mxHID = 0, mxAO = 0, mxSC = 0, mxQb = 0, mxXQ = 0, mxXS = 0, mxHQ = 0, mxHS = 0;
  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& b = cfg.b[i];
    mxXn = std::max(mxXn, std::max((size_t)b.Tprb * b.din, (size_t)b.Tqb * b.dout));
    // the fp16 qkv buffer serves the blocks that still produce fp16 Q/K/V: under --win8 the int8 ones
    // do not (the debug build compares against the fp16 path on every block)
#ifndef W8DEBUG
    if (!(gInt8 && i >= gFp16Blocks && win8Block(b)))
#endif
    mxQKV = std::max(mxQKV, (size_t)b.Tprb * 3 * b.cp);
    mxHID = std::max(mxHID, std::max((size_t)b.Tqb * 4 * b.dout, (size_t)b.Tb * b.dout));
    mxAO = std::max(mxAO, (size_t)b.Tpob * b.dout);
    if (b.qpool) { mxSC = std::max(mxSC, (size_t)b.Tqb * b.dout); mxQb = std::max(mxQb, (size_t)b.nWb * b.heads * (b.wo * b.wo) * cfg.dhp); }
    mxXQ = std::max(mxXQ, (size_t)b.Tprb * b.din); mxXS = std::max(mxXS, (size_t)b.Tprb);
    mxHQ = std::max(mxHQ, (size_t)b.Tqb * 4 * b.dout); mxHS = std::max(mxHS, (size_t)b.Tqb * cdiv(4 * b.dout, 128));
  }
  for (int st = 0; st < 4; ++st) Xs[st] = dAlloc((size_t)gB * (65536 >> (2 * st)) * cfg.dims[st]);
  Xn = dAlloc(mxXn); QKV = dAlloc(mxQKV + 64 * cfg.dhp); HID = dAlloc(mxHID); AO = dAlloc(mxAO);
  SC = dAlloc(std::max(mxSC, (size_t)1)); Qb = dAlloc(std::max(mxQb, (size_t)1) + 64 * cfg.dhp);
  COL = dAlloc((size_t)gB * 65536 * 152);
  POS = dAlloc((size_t)gB * 65536 * cfg.dims[0]);
  L2 = dAlloc((size_t)gB * 4096 * 256); L3 = dAlloc((size_t)gB * 1024 * 256);
  S4b = dAlloc((size_t)gB * 65536 * 32); S8b = dAlloc((size_t)gB * 16384 * 64);
  CK(cudaMalloc(&dIn, (size_t)gB * 3 * 1024 * 1024 * sizeof(float)));
  CK(cudaMalloc(&dVF, (size_t)gB * 256 * 64 * 64 * sizeof(float)));
  CK(cudaMalloc(&dS4, (size_t)gB * 32 * 256 * 256 * sizeof(float)));
  CK(cudaMalloc(&dS8, (size_t)gB * 64 * 128 * 128 * sizeof(float)));
  // cuBLASLt autotune scratch: the largest A (M x K), B (K x N) and D (M x N) of ANY plan --
  // the transition blocks' qkv (N = 3 * cp of the next stage) is wider than 4 * dims[0].
  // Only the tuner reads these, so --no-tune skips the allocation (at batch 8 they are GBs).
  if (gTune) {
    size_t mxA = (size_t)gB * 65536 * 152, mxB = (size_t)152 * cfg.dims[0], mxD = (size_t)gB * 65536 * cfg.dims[0];
    for (const Blk& b : cfg.b) {
      const size_t A[5] = {(size_t)b.Tprb * b.din, (size_t)b.Tb * b.din, (size_t)b.Tqb * b.dout, (size_t)b.Tqb * b.dout, (size_t)b.Tqb * 4 * b.dout};
      const size_t Bm[5] = {(size_t)b.din * 3 * b.cp, (size_t)b.din * b.dout, (size_t)b.dout * b.dout, (size_t)b.dout * 4 * b.dout, (size_t)4 * b.dout * b.dout};
      const size_t D[5] = {(size_t)b.Tprb * 3 * b.cp, (size_t)b.Tb * b.dout, (size_t)b.Tqb * b.dout, (size_t)b.Tqb * 4 * b.dout, (size_t)b.Tqb * b.dout};
      for (int j = 0; j < 5; ++j) { mxA = std::max(mxA, A[j]); mxB = std::max(mxB, Bm[j]); mxD = std::max(mxD, D[j]); }
    }
    mxA = std::max(mxA, (size_t)gB * 4096 * cfg.dims[2]); mxB = std::max(mxB, (size_t)cfg.dims[3] * 256);
    mxD = std::max(mxD, (size_t)gB * 4096 * 256);
    gTA = dAlloc(mxA); gTB = dAlloc(mxB); gTD = dAlloc(mxD);
  }
  if (gInt8) {
    CK(cudaMalloc(&XQ, mxXQ)); CK(cudaMemset(XQ, 0, mxXQ));
    CK(cudaMalloc(&XS, mxXS * sizeof(float))); CK(cudaMemset(XS, 0, mxXS * sizeof(float)));
  }
  if (gInt8Fc2) { CK(cudaMalloc(&HQ, mxHQ)); CK(cudaMemset(HQ, 0, mxHQ)); CK(cudaMalloc(&HS, mxHS * sizeof(float))); }
  if (gQK8) {
    // zeroed slack past the last row: the kernel reads whole key tiles and their scales
    size_t q8 = 0; int hmax = 0;
    for (const Blk& b : cfg.b)
      if (qk8Block(b)) { q8 = std::max(q8, (size_t)b.Tprb * 2 * b.heads * attn2q8_dh8(cfg.dh)); gQKSst = std::max(gQKSst, b.Tprb + 128); hmax = std::max(hmax, b.heads); }
    if (hmax) {
      CK(cudaMalloc(&QK8, q8 + 65536)); CK(cudaMemset(QK8, 0, q8 + 65536));
      const size_t ns = (size_t)2 * hmax * gQKSst * sizeof(float);
      CK(cudaMalloc(&QKS, ns)); CK(cudaMemset(QKS, 0, ns));
    }
  }
  if (gWin8) {
    // zeroed slack past the last row of the largest block: the attention reads whole key tiles
    // (up to BK = 32 rows past a window) and their scales; rows past a block's own Tprb hold
    // zeros or an earlier block's finite codes / scales, and the masked keys get P = 0
    // (the scale stride is the block's own Tprb + 128, set per block in forward)
    size_t q8 = 0, ns = 0;
    for (const Blk& b : cfg.b)
      if (win8Block(b)) { q8 = std::max(q8, (size_t)(b.Tprb + 128) * 3 * b.cp); ns = std::max(ns, (size_t)(b.Tprb + 128) * (3 * b.cp / 16) * sizeof(half)); }
    if (q8) {
      CK(cudaMalloc(&W8Q, q8 + 65536)); CK(cudaMemset(W8Q, 0, q8 + 65536));
      CK(cudaMalloc(&W8S, ns)); CK(cudaMemset(W8S, 0, ns));
      printf("win8: int8 qkv %.1f MB + fp16 scales %.1f MB (one per token and 16 channels)\n", (q8 + 65536) / 1e6, ns / 1e6);
    }
  }
  buildMaps();
  k_pos_rep<<<cdiv((int)((size_t)gB * 65536 * (cfg.dims[0] / 8)), 256), 256>>>(
      (const half*)WP("pos"), POS, 65536, cfg.dims[0], gB);
  CK(cudaDeviceSynchronize());
  size_t fr = 0, tt = 0; CK(cudaMemGetInfo(&fr, &tt));
  printf("batch %d: %d tokens at stage 0, %.0f MB free of %.0f MB\n", gB, gB * 65536, fr / 1e6, tt / 1e6);
}

// ---------------------------------------------------------------- int8 weights
static void quantRotated(const half* w, int8_t* wt, float* sw, int K, int N, int kind) {
  if (!gRotTmp) CK(cudaMalloc(&gRotTmp, (size_t)4 * cfg.dims[3] * std::max(4 * cfg.dims[3], 3 * cfg.heads[3] * cfg.dhp) * sizeof(half)));
  const size_t sh = 2ull * K * sizeof(float);
  if (kind == 2) k_rot_w<2><<<N, 256, sh>>>(w, gRotTmp, K, N);
  else           k_rot_w<1><<<N, 256, sh>>>(w, gRotTmp, K, N, gFc2Rot);
  k_quant_w<<<N, 256>>>(gRotTmp, wt, sw, K, N);
  CK(cudaDeviceSynchronize());
}
// --int4: int8 codes [R, K] (device) -> packed int4 [R, kpad4(K) / 2] at dst; any code outside -7..7 is fatal
static void packS4p(const int8_t* codes, int8_t* dst, int R, int K) {
  static unsigned int* dbad = nullptr;
  if (!dbad) CK(cudaMalloc(&dbad, sizeof(unsigned int)));
  CK(cudaMemset(dbad, 0, sizeof(unsigned int)));
  const size_t np = (size_t)R * (kpad4(K) / 2);
  k_pack_s4p<<<(unsigned)((np + 255) / 256), 256>>>(codes, (uint8_t*)dst, R, K, dbad);
  unsigned int h = 0; CK(cudaMemcpy(&h, dbad, sizeof h, cudaMemcpyDeviceToHost));
  if (h) { printf("int4 pack: %u code pairs outside -7..7 (not a WBITS=4 file?)\n", h); exit(1); }
}
static void loadGptq(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot open %s\n", path.c_str()); exit(1); }
  char magic[8]; int32_t nblk, flags;
  f.read(magic, 8); f.read((char*)&nblk, 4); f.read((char*)&flags, 4);
  if (memcmp(magic, "NANOGPTQ", 8) || nblk != cfg.nblk) { printf("%s: bad header (nblk %d)\n", path.c_str(), nblk); exit(1); }
  std::vector<char> buf;
  auto take = [&](void* dst, size_t bytes) {
    buf.resize(bytes); f.read(buf.data(), bytes);
    if (!f) { printf("%s: truncated\n", path.c_str()); exit(1); }
    CK(cudaMemcpy(dst, buf.data(), bytes, cudaMemcpyHostToDevice));
  };
  // --int4: the file's int8 codes [R, K] are staged in gW4tmp and packed two per byte (rows padded to kpad4(K))
  auto takeW = [&](int8_t* dst, int R, int K) {
    if (!gInt4) { take(dst, (size_t)R * K); return; }
    take(gW4tmp, (size_t)R * K); packS4p(gW4tmp, dst, R, K);
  };
  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& c = cfg.b[i];
    takeW(QWqkv[i].w, 3 * c.cp, c.din); take(QWqkv[i].s, (size_t)(3 * c.cp) * 4);
    takeW(QWfc1[i].w, 4 * c.dout, c.dout); take(QWfc1[i].s, (size_t)(4 * c.dout) * 4);
    if (gInt8Fc2) { takeW(QWfc2[i].w, c.dout, 4 * c.dout); take(QWfc2[i].s, (size_t)c.dout * 4); }
    else f.seekg((std::streamoff)((size_t)c.dout * (4 * c.dout) + (size_t)c.dout * 4), std::ios::cur);
  }
  CK(cudaDeviceSynchronize());
  printf("gptq weights: %s (flags %d)%s\n", path.c_str(), flags, gInt4 ? ", packed int4" : "");
}
// --int4: rotate exactly as quantRotated, quantise to 4-bit codes (k_quant_w4), pack (padded) into dst.w
static void quantRotated4(const half* w, const Q8W& dst, int K, int N, int kind) {
  if (!gRotTmp) CK(cudaMalloc(&gRotTmp, (size_t)4 * cfg.dims[3] * std::max(4 * cfg.dims[3], 3 * cfg.heads[3] * cfg.dhp) * sizeof(half)));
  const size_t sh = 2ull * K * sizeof(float);
  if (kind == 2) k_rot_w<2><<<N, 256, sh>>>(w, gRotTmp, K, N);
  else           k_rot_w<1><<<N, 256, sh>>>(w, gRotTmp, K, N, gFc2Rot);
  k_quant_w4<<<N, 256>>>(gRotTmp, gW4tmp, dst.s, K, N);
  CK(cudaDeviceSynchronize());
  packS4p(gW4tmp, dst.w, N, K);
}
// --mon identity: B1 = B2 = I still leaves the stride permutation, T = Pi (x T = x[pi], pi[q*n + j] = j*b + q), so the
// RTN weights of qkv / fc1 take W T^-T = W Pi: rows of the [K, N] fp16 weight permuted, out[c] = w[pi[c]]
// the block size b of the identity mode: 32, and at C = 144 NANO_MON_IDB144 (default 36; 16 is the other kernel)
static int monBB(int C) {
  static int b144 = 0;
  if (!b144) { const char* e = getenv("NANO_MON_IDB144"); b144 = e ? atoi(e) : 36; }
  return C == 144 ? b144 : C == 96 ? 24 : C == 112 ? 16 : 32;
}
// the (C, b) pairs k_ln_mon4 is instantiated for (launchLNMon4)
static bool monKnown(int C, int b) {
  switch (C) {
    case 144: return b == 16 || b == 36;
    case 96: return b == 24;
    case 112: return b == 16;
    case 192: case 224: case 288: case 384: case 448: case 576: case 768: case 896: case 1152: return b == 32;
    default: return false;
  }
}
__global__ void k_perm_rows(const half* __restrict__ w, half* __restrict__ out, int K, int N, int BB) {
  const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)K * N) return;
  const int c = (int)(i / N), col = (int)(i - (size_t)c * N), n = K / BB;
  const int q = c / n, j = c - q * n;
  out[i] = w[(size_t)(j * BB + q) * N + col];
}
static half* gPermTmp = nullptr;
static const half* monIdentW(const half* w, int K, int N) {
  if (gMonFile != "identity") return w;
  if (!gPermTmp) CK(cudaMalloc(&gPermTmp, (size_t)4 * cfg.dims[3] * std::max(4 * cfg.dims[3], 3 * cfg.heads[3] * cfg.dhp) * sizeof(half)));
  k_perm_rows<<<(unsigned)(((size_t)K * N + 255) / 256), 256>>>(w, gPermTmp, K, N, monBB(K));
  return gPermTmp;
}
static void buildInt4Weights() {
  const int NB = cfg.nblk;
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB); Bfc2.assign(NB, nullptr);
  size_t bytes = 0, sn = 0, tmp = 0;
  for (const Blk& c : cfg.b) {
    bytes += (size_t)(3 * c.cp) * kpad4(c.din) / 2 + (size_t)(4 * c.dout) * kpad4(c.dout) / 2 + (size_t)c.dout * kpad4(4 * c.dout) / 2;
    sn += 3 * c.cp + 4 * c.dout + c.dout;
    tmp = std::max(tmp, std::max((size_t)(3 * c.cp) * c.din, (size_t)(4 * c.dout) * c.dout));
  }
  int8_t* blob; CK(cudaMalloc(&blob, bytes));
  float* sblob; CK(cudaMalloc(&sblob, sn * sizeof(float)));
  CK(cudaMalloc(&gW4tmp, tmp));
  size_t off = 0, soff = 0;
  for (int i = 0; i < NB; ++i) {
    const Blk& c = cfg.b[i];
    const int nq = 3 * c.cp, nf = 4 * c.dout;
    QWqkv[i] = {blob + off, sblob + soff}; off += (size_t)nq * kpad4(c.din) / 2; soff += nq;
    QWfc1[i] = {blob + off, sblob + soff}; off += (size_t)nf * kpad4(c.dout) / 2; soff += nf;
    QWfc2[i] = {blob + off, sblob + soff}; off += (size_t)c.dout * kpad4(nf) / 2; soff += c.dout;
    quantRotated4(monIdentW((const half*)WP(bk(i, "qkv.w")), c.din, nq), QWqkv[i], c.din, nq, 2);
    quantRotated4(monIdentW((const half*)WP(bk(i, "fc1.w")), c.dout, nf), QWfc1[i], c.dout, nf, 2);
    quantRotated4((const half*)WP(bk(i, "fc2.w")), QWfc2[i], nf, c.dout, 1);
  }
  CK(cudaDeviceSynchronize());
  if (!gGptq.empty()) loadGptq(gGptq);
  printf("int4 weights: %.1f MB packed (qkv + fc1 + fc2 of %d blocks), rotated%s\n", bytes / 1e6, NB,
         gGptq.empty() ? ", RTN 4-bit" : ", GPTQ codes");
}
// --mon: the T = B1 Pi B2 factors of every LayerNorm-fed GEMM, packed into k_ln_mon4's fragment order.
// NANOMON1 file (tools/export_mon.py): "NANOMON1", int32 nblk, then per block, for qkv (C = din) and fc1 (C = dout):
// int32 C, b, n, float B1[n][b][b] (block j, in, out), float B2[b][n][n] (position q, out, in) -- cnorm and the
// B1 / B2 balance already folded in.  "identity": B1 = B2 = I with the engine's own b (16 at C = 144, else 32).
static void buildMon() {
  const bool ident = gMonFile == "identity";
  std::ifstream f;
  if (!ident) {
    f.open(gMonFile, std::ios::binary);
    if (!f) { printf("cannot open %s\n", gMonFile.c_str()); exit(1); }
    char magic[8]; int32_t nblk = 0;
    f.read(magic, 8); f.read((char*)&nblk, 4);
    if (memcmp(magic, "NANOMON1", 8) || nblk != cfg.nblk) { printf("%s: bad header (nblk %d)\n", gMonFile.c_str(), nblk); exit(1); }
  }
  gMon1.assign(cfg.nblk, MonLN()); gMon2.assign(cfg.nblk, MonLN());
  float cmax = 0.f; size_t bytes = 0;
  for (int i = 0; i < cfg.nblk; ++i)
    for (int w = 0; w < 2; ++w) {
      const int C = w == 0 ? cfg.b[i].din : cfg.b[i].dout;
      int BB = monBB(C), N = C / BB;
      std::vector<float> B1, B2;
      if (ident) {
        B1.assign((size_t)N * BB * BB, 0.f); B2.assign((size_t)BB * N * N, 0.f);
        for (int j = 0; j < N; ++j) for (int k = 0; k < BB; ++k) B1[((size_t)j * BB + k) * BB + k] = 1.f;
        for (int q = 0; q < BB; ++q) for (int k = 0; k < N; ++k) B2[((size_t)q * N + k) * N + k] = 1.f;
      } else {
        int32_t h[3]; f.read((char*)h, 12);
        // the file's b selects the kernel: C = 144 has b = 16 (n 9) and b = 36 (n 4), every other width b = 32
        const bool known = monKnown(C, h[1]);
        if (!f || h[0] != C || !known || h[1] * h[2] != C) {
          printf("%s: block %d %s: C %d b %d n %d; no k_ln_mon4 for this pair (C %d: b %d%s)\n", gMonFile.c_str(), i,
                 w ? "fc1" : "qkv", h[0], h[1], h[2], C, monBB(C), C == 144 ? " or 16" : "");
          exit(1);
        }
        BB = h[1]; N = h[2];
        B1.assign((size_t)N * BB * BB, 0.f); B2.assign((size_t)BB * N * N, 0.f);
        f.read((char*)B1.data(), B1.size() * 4); f.read((char*)B2.data(), B2.size() * 4);
        if (!f) { printf("%s: truncated\n", gMonFile.c_str()); exit(1); }
      }
      std::vector<uint2> f1, f2; std::vector<uint32_t> f1t, f2t;
      cmax = std::max(cmax, mon4_pack(C, BB, B1.data(), B2.data(), f1, f1t, f2, f2t));
      MonLN& m = w == 0 ? gMon1[i] : gMon2[i];
      m.C = C; m.BB = BB;
      CK(cudaMalloc(&m.f1, f1.size() * 8)); CK(cudaMemcpy(m.f1, f1.data(), f1.size() * 8, cudaMemcpyHostToDevice));
      CK(cudaMalloc(&m.f2, f2.size() * 8)); CK(cudaMemcpy(m.f2, f2.data(), f2.size() * 8, cudaMemcpyHostToDevice));
      if (!f1t.empty()) { CK(cudaMalloc(&m.f1t, f1t.size() * 4)); CK(cudaMemcpy(m.f1t, f1t.data(), f1t.size() * 4, cudaMemcpyHostToDevice)); }
      if (!f2t.empty()) { CK(cudaMalloc(&m.f2t, f2t.size() * 4)); CK(cudaMemcpy(m.f2t, f2t.data(), f2t.size() * 4, cudaMemcpyHostToDevice)); }
      bytes += f1.size() * 8 + f1t.size() * 4 + f2.size() * 8 + f2t.size() * 4;
    }
  if (cmax > 60000.f) { printf("mon: coefficient %.3g overflows fp16\n", cmax); exit(1); }
  printf("mon: %s, %d LayerNorms (b at C = 144: %d), fragments %.2f MB, max |coef| %.4g\n",
         ident ? "identity (B1 = B2 = I)" : gMonFile.c_str(), 2 * cfg.nblk, gMon1[0].BB, bytes / 1e6, cmax);
}
#include "structure_ln16.cuh"
static void loadMixedFc2(){
 std::ifstream f(gFc2External,std::ios::binary);char magic[8];int n,bits;f.read(magic,8);f.read((char*)&n,4);f.read((char*)&bits,4);if(!f||memcmp(magic,"NANOF C2",8)||n!=cfg.nblk||bits!=gFc2FixedBits){printf("bad externalfc2%c",10);exit(1);}MixedFc2.resize(n);
 for(int i=0;i<n;i++){int K=4*cfg.b[i].dout,N=cfg.b[i].dout;std::vector<int8_t>q(N*K);std::vector<float>sc(N);f.read((char*)q.data(),q.size());f.read((char*)sc.data(),N*4);if(!f){printf("truncatedfc2%c",10);exit(1);}int8_t*dw;float*ds;CK(cudaMalloc(&dw,(size_t)N*K));CK(cudaMalloc(&ds,N*4));CK(cudaMemcpy(ds,sc.data(),N*4,cudaMemcpyHostToDevice));if(bits==8)CK(cudaMemcpy(dw,q.data(),q.size(),cudaMemcpyHostToDevice));else{CK(cudaMemcpy(gW4tmp,q.data(),q.size(),cudaMemcpyHostToDevice));packS4p(gW4tmp,dw,N,K);}MixedFc2[i]={dw,ds};}
}
static void buildInt8Weights() {
  if (gInt4) { buildInt4Weights(); if (!gMonFile.empty()) buildMon(); if (!gKronFile.empty()) buildKron();if(gFc2FixedBits)loadMixedFc2(); if(!gDhdFile.empty())buildDhd(); if(!gFc2StaticFile.empty())buildFc2Static(); buildInt4Pitched(); return; }
  const int NB = cfg.nblk;
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB); Bfc2.assign(NB, nullptr);
  size_t bytes = 0, sn = 0;
  for (const Blk& c : cfg.b) {
    bytes += (size_t)(3 * c.cp) * c.din + (size_t)(4 * c.dout) * c.dout + (gInt8Fc2 ? (size_t)c.dout * (4 * c.dout) : 0);
    sn += 3 * c.cp + 4 * c.dout + c.dout;
  }
  int8_t* blob; CK(cudaMalloc(&blob, bytes));
  float* sblob; CK(cudaMalloc(&sblob, sn * sizeof(float)));
  size_t off = 0, soff = 0;
  for (int i = 0; i < NB; ++i) {
    const Blk& c = cfg.b[i];
    int nq = 3 * c.cp, kq = c.din, nf = 4 * c.dout, kf = c.dout;
    QWqkv[i] = {blob + off, sblob + soff}; off += (size_t)nq * kq; soff += nq;
    QWfc1[i] = {blob + off, sblob + soff}; off += (size_t)nf * kf; soff += nf;
    const half* wq = (const half*)WP(bk(i, "qkv.w")); const half* wf = (const half*)WP(bk(i, "fc1.w"));
    if (gRot) quantRotated(wq, QWqkv[i].w, QWqkv[i].s, kq, nq, 2); else k_quant_w<<<nq, 256>>>(wq, QWqkv[i].w, QWqkv[i].s, kq, nq);
    if (gRot) quantRotated(wf, QWfc1[i].w, QWfc1[i].s, kf, nf, 2); else k_quant_w<<<nf, 256>>>(wf, QWfc1[i].w, QWfc1[i].s, kf, nf);
    if (gInt8Fc2) {
      QWfc2[i] = {blob + off, sblob + soff}; off += (size_t)c.dout * (4 * c.dout); soff += c.dout;
      const half* w2 = (const half*)WP(bk(i, "fc2.w"));
      if (gRot) quantRotated(w2, QWfc2[i].w, QWfc2[i].s, 4 * c.dout, c.dout, 1);
      else      k_quant_w<<<c.dout, 256>>>(w2, QWfc2[i].w, QWfc2[i].s, 4 * c.dout, c.dout);
      if (gFc2U8) {
        float* ws; CK(cudaMalloc(&ws, c.dout * sizeof(float)));
        CK(cudaMalloc(&Bfc2[i], c.dout * sizeof(half)));
        k_wsum<<<c.dout, 256>>>(QWfc2[i].w, 4 * c.dout, c.dout, ws);
        k_bias_shift<<<cdiv(c.dout, 128), 128>>>((const half*)WP(bk(i, "fc2.b")), QWfc2[i].s, ws, 0.169971f, c.dout, Bfc2[i]);
        CK(cudaDeviceSynchronize()); CK(cudaFree(ws));
      }
    }
  }
  CK(cudaDeviceSynchronize());
  if (!gGptq.empty()) loadGptq(gGptq);
  printf("int8 weights: %.1f MB (%s of %d blocks)%s%s\n", bytes / 1e6, gInt8Fc2 ? "qkv + fc1 + fc2" : "qkv + fc1", NB,
         gRot ? ", rotated" : "", gGptq.empty() ? "" : ", GPTQ");
}

// ---------------------------------------------------------------- launchers
#define LN_SWITCH(C, CALL) switch (C) { \
  case 96: CALL(12); break; case 112: CALL(14); break; case 144: CALL(18); break; \
  case 192: CALL(24); break; case 224: CALL(28); break; case 288: CALL(36); break; \
  case 384: CALL(48); break; case 448: CALL(56); break; case 576: CALL(72); break; \
  case 768: CALL(96); break; case 896: CALL(112); break; case 1152: CALL(144); break; \
  default: printf("unexpected LN width %d\n", C); exit(1); }
static void launchLN(const half* x, half* y, const float* g, const float* b, int T, int C, cudaStream_t s, const int* src = nullptr) {
  int grid = cdiv(T, 8), thr = 256; size_t sh = 2 * C * sizeof(float);
#define LNT(NV) k_ln_t<NV><<<grid, thr, sh, s>>>(x, y, g, b, T, src)
  LN_SWITCH(C, LNT)
#undef LNT
}
static void launchLNQ8(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C, cudaStream_t s, const int* src = nullptr) {
  int grid = cdiv(T, 8), thr = 256; size_t sh = 2 * C * sizeof(float);
  if (gRot) {
    // NANO_LN_L1=1 (dhd_speed): gamma / beta through L1 as float4, the DHD producer's read path, so that the W8
    // baseline is timed with the same LayerNorm engineering.  Off by default: the shipped kernel is unchanged.
    static const bool l1env = getenv("NANO_LN_L1") != nullptr;
    if (l1env && (((uintptr_t)g | (uintptr_t)b) & 15) == 0) {
#define LNQR1(NV) k_ln_q8<NV, 1, true><<<grid, thr, 0, s>>>(x, y, ys, g, b, T, src)
      LN_SWITCH(C, LNQR1)
#undef LNQR1
      return;
    }
    if (l1env) { printf("NANO_LN_L1: unaligned gamma / beta at C = %d\n", C); exit(1); }
#define LNQR(NV) k_ln_q8<NV, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src)
    LN_SWITCH(C, LNQR)
#undef LNQR
    return;
  }
#define LNQ(NV) k_ln_q8<NV><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src)
  LN_SWITCH(C, LNQ)
#undef LNQ
}

template<bool GELU, bool SASCALAR = false, bool RESID = false, int GROUP = 0, bool Q8OUT = false,
         int NGPT = 1, int WM = 2, int WN = 4, bool U8IN = false, bool U8OUT = false, int ROT8 = 0,
         bool DB = false, int MINB = 0, int BM = 128, int BN = 128, int ST = 3>
static void runS8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C,
                  int M, int N, int K, cudaStream_t s, const half* resid = nullptr,
                  int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0) {
  constexpr int BK = 64;
  if constexpr (!GELU && !SASCALAR && !RESID && GROUP == 0 && !Q8OUT && !U8IN && !U8OUT && ROT8 == 0 && BM == 128 && BN == 128 && ST == 3 && !DB) {
    if (gArch120) {
      // sm_120: the 1024-row stage-3 shapes (N >= 2304) fill the 170 SMs in one wave with a 192-wide tile
      // (160 CTAs instead of 240 = 1.4 waves; build5090/retune/s8sweep_qkv.log: 1.11-1.16x); the rest keep the
      // generic tile
      if (M <= 1024 && N >= 2304) {
        constexpr int BN2 = 192;
        constexpr size_t sh2 = s8_smem<BM, BN2, BK, ST>();
        static bool attr2 = false;
        if (!attr2) {
          CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN2, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
          attr2 = true;
        }
        k_gemm_s8<BM, BN2, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>
            <<<dim3(cdiv(N, BN2), M / BM), WM * WN * 32, sh2, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride);
        return;
      }
    } else if (gArch89 && K >= 576) {
      constexpr int ST2 = 2;
      constexpr size_t sh2 = s8_smem<BM, BN, BK, ST2>();
      static bool attr2 = false;
      if (!attr2) {
        CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, 2, 2, ST2, false, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
        attr2 = true;
      }
      k_gemm_s8<BM, BN, BK, 2, 2, ST2, false, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8>
          <<<dim3(cdiv(N, BN), M / BM), 128, sh2, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride);
      return;
    }
  }
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr = true;
  }
  k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>
      <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride);
}
// --qk8 on the int8 path: the qkv GEMM itself is the Q/K producer.  A column tile of BN =
// 2 * DHP holds two whole heads, so the per-(token, head) scale of Q and K is the epilogue's
// per-(row, BN/2) row max (QKV8 in sam21_gemm_s8.cuh); the V tiles stay fp16.  Needs DHP ==
// DH8 (the int8 head stride), i.e. head dims 96 and 64 (hiera-T/S and B+), not 72 (L).
static bool qkvFused(const Blk& b) { return qk8Block(b) && gInt8 && cfg.dhp == attn2q8_dh8(cfg.dh) && (cfg.dhp == 96 || cfg.dhp == 64); }
static void runQkv8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K, int qkn, cudaStream_t s) {
#define QKV8K(BN) do { \
    constexpr size_t sh = s8_smem<128, BN, 64, 3>(); \
    static bool attr = false; \
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_s8<128, BN, 64, 2, 4, 3, false, false, false, false, 0, false, 2, false, false, 0, 0, false, true>, \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; } \
    k_gemm_s8<128, BN, 64, 2, 4, 3, false, false, false, false, 0, false, 2, false, false, 0, 0, false, true> \
        <<<dim3(cdiv(N, BN), M / 128), 256, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, nullptr, QK8, QKS, gQKSst, nullptr, qkn); } while (0)
  if (cfg.dhp == 96) QKV8K(192); else QKV8K(128);
#undef QKV8K
}
// --win8 producer: the plain int8 qkv GEMM with the W8G16 epilogue (int8 Q/K/V into W8Q, one fp16
// scale per (row, 16 columns) into W8S), on the tile runS8 picks for the plain qkv GEMM on each arch
// (a two-stage ring below K = 576 and 64-row tiles were both slower on the 4090: win8/logs/explore_v6)
static void runQkvW8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, int M, int N, int K, cudaStream_t s) {
#define QKVW8K(BM, BN, WM, WN, ST) do { \
    constexpr size_t sh = s8_smem<BM, BN, 64, ST>(); \
    static bool attr = false; \
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, 64, WM, WN, ST, false, false, false, false, 0, false, 1, false, false, 0, 0, false, false, true>, \
                                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; } \
    k_gemm_s8<BM, BN, 64, WM, WN, ST, false, false, false, false, 0, false, 1, false, false, 0, 0, false, false, true> \
        <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, nullptr, M, N, K, nullptr, W8Q, (float*)W8S, gW8Sst); } while (0)
  if (gArch120 && M <= 1024 && N >= 2304) QKVW8K(128, 192, 2, 4, 3);
  else if (!gArch120 && gArch89 && K >= 576) QKVW8K(128, 128, 2, 2, 2);
  else QKVW8K(128, 128, 2, 4, 3);
#undef QKVW8K
}
template<bool U8IN>
static void runFc2S8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K, cudaStream_t s, int sastride) {
  if (gArch120) {
    // sm_120 (170 SMs), from build5090/retune/s8sweep_fc2.log: the batch-1 ViT shapes want 32-row tiles (3x),
    // the 1024-row stage-3 shapes 64x128 (1.6-1.8x), the narrow stage-0/1 widths 128x64 with 4x1 warps
    // (1.2-1.4x), everything else 128x128 with 16 warps and a 2-stage ring (1.1x); the Ada 192-wide tile
    // leaves 44 % of the SMs idle here.
    const int ctas = cdiv(M, 128) * cdiv(N, 128);
    if (M <= 512)
      runS8<false, false, true, 128, false, 1, 2, 2, U8IN, false, 0, false, 0, 32, 64>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
    else if (ctas < 100)
      runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 64, 128>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
    else if (N < 384 && N % 128 != 0)
      runS8<false, false, true, 128, false, 1, 4, 1, U8IN, false, 0, false, 0, 128, 64>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
    else
      runS8<false, false, true, 128, false, 1, 4, 4, U8IN, false, 0, false, 0, 128, 128, 2>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  } else if (gArch89) {
    if (N == 288)       runS8<false, false, true, 128, false, 1, 4, 2, U8IN, false, 0, false, 0, 128, 64>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
    else if (N == 1152) runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 64, 192>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
    else                runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 128, 192>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  } else {
    runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  }
}

// ---------------------------------------------------------------- --int4 launchers (w4a4_hiera)
static float* dhdCopy(const std::vector<float>& v){float*p;CK(cudaMalloc(&p,v.size()*sizeof(float)));CK(cudaMemcpy(p,v.data(),v.size()*sizeof(float),cudaMemcpyHostToDevice));return p;}
static void buildDhd(){
 std::ifstream f(gDhdFile,std::ios::binary);char magic[8];int n,learned;f.read(magic,8);f.read((char*)&n,4);f.read((char*)&learned,4);
 if(!f||memcmp(magic,"NANODHD1",8)||n!=cfg.nblk){printf("Invalid DHD header\n");exit(1);}
 gDhd1.resize(n);gDhd2.resize(n);gDhd3.resize(n);
 for(int bi=0;bi<n;bi++)for(int fam=0;fam<3;fam++){
  DhdGPU t;t.diagonals=learned!=0;f.read((char*)&t.K,4);f.read((char*)&t.H,4);f.read((char*)&t.clip,4);
  const int expected=fam==0?cfg.b[bi].din:fam==1?cfg.b[bi].dout:4*cfg.b[bi].dout;
  if(t.K!=expected||t.H!=(fam==2?64:rot_block(expected))||!std::isfinite(t.clip)||t.clip<=0||t.clip>1){printf("Invalid DHD dimensions/clip\n");exit(1);}
  std::vector<float>a(t.K),b(t.K);f.read((char*)a.data(),t.K*4);f.read((char*)b.data(),t.K*4);
  if(!f){printf("Truncated DHD factors\n");exit(1);}
  for(int j=0;j<t.K;j++){if(!std::isfinite(a[j])||!std::isfinite(b[j])||a[j]<=0||b[j]<=0){printf("Invalid DHD diagonal\n");exit(1);}if(gDhdIdentity)a[j]=b[j]=1.f;}
  if(gDhdIdentity)t.clip=1.f;
  t.d2unit=std::all_of(b.begin(),b.end(),[](float v){return v==1.f;});   // fc2s4: every d2 = 1 -> its multiply can be skipped exactly
  t.d1=dhdCopy(a);t.d2=dhdCopy(b);
  if(fam<2){
   std::vector<float>g(t.K),beta(t.K);const char*gw=fam==0?"ln1.w":"ln2.w";const char*gb=fam==0?"ln1.b":"ln2.b";
   CK(cudaMemcpy(g.data(),WP(bk(bi,gw)),t.K*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(beta.data(),WP(bk(bi,gb)),t.K*4,cudaMemcpyDeviceToHost));
   for(int j=0;j<t.K;j++){g[j]*=a[j];beta[j]*=a[j];}t.gamma=dhdCopy(g);t.beta=dhdCopy(beta);
  }
  (fam==0?gDhd1:fam==1?gDhd2:gDhd3)[bi]=t;
 }
 char extra;if(f.read(&extra,1)){printf("Trailing DHD factor bytes\n");exit(1);}
 printf("DHD all144: offline D1-folded LN, FWHT, D2, learned clip; fc2 fused post-GELU; identity=%d learned=%d\n",(int)gDhdIdentity,learned);
}
// NANOF2S1: char[8] magic, int32 nblk, int32 mode (1 sym / 2 group); per block: int32 K (= 4 * dout), int32 G
// (= ceil(K / 128)), float32 s[G] (the activation scale of each 128-column group of the rotated, normalised fc2 input:
// code = clamp(round(x / s_g), -7, 7)), float32 sgemm (the fc2 GEMM's scalar row scale: s for sym, 1 for group).
static void buildFc2Static() {
  std::ifstream f(gFc2StaticFile, std::ios::binary); char magic[8]; int n = 0, mode = 0;
  f.read(magic, 8); f.read((char*)&n, 4); f.read((char*)&mode, 4);
  if (!f || memcmp(magic, "NANOF2S1", 8) || n != cfg.nblk || (mode != 1 && mode != 2)) { printf("Invalid fc2-static header\n"); exit(1); }
  gF2S.resize(n);
  double smin = 1e30, smax = 0;
  for (int bi = 0; bi < n; ++bi) {
    int K = 0, G = 0; f.read((char*)&K, 4); f.read((char*)&G, 4);
    if (!f || K != 4 * cfg.b[bi].dout || G != cdiv(K, 128)) { printf("Invalid fc2-static dimensions (block %d)\n", bi); exit(1); }
    std::vector<float> s(G), inv(G); float sg = 0.f;
    f.read((char*)s.data(), G * 4); f.read((char*)&sg, 4);
    if (!f) { printf("Truncated fc2-static file\n"); exit(1); }
    for (int g = 0; g < G; ++g) {
      if (!std::isfinite(s[g]) || s[g] <= 0.f || (mode == 1 && s[g] != s[0])) { printf("Invalid fc2-static scale (block %d group %d)\n", bi, g); exit(1); }
      inv[g] = 0.125f / s[g];                          // (1/sqrt(64)) / s_g: the epilogue's values are un-normalised H_64 outputs
      smin = std::min(smin, (double)s[g]); smax = std::max(smax, (double)s[g]);
    }
    if (!std::isfinite(sg) || sg <= 0.f || (mode == 1 && sg != s[0]) || (mode == 2 && sg != 1.f)) { printf("Invalid fc2-static GEMM scale (block %d)\n", bi); exit(1); }
    gF2S[bi].K = K; gF2S[bi].G = G; gF2S[bi].sinv = dhdCopy(inv); gF2S[bi].sgemm = dhdCopy(std::vector<float>(1, sg));
  }
  char extra; if (f.read(&extra, 1)) { printf("Trailing fc2-static bytes\n"); exit(1); }
  gFc2StaticMode = mode;
  {
    int unit = 0, diag = 0;
    for (const auto* v : {&gDhd1, &gDhd2, &gDhd3}) for (const DhdGPU& t : *v) { diag += t.diagonals; unit += t.diagonals && t.d2unit; }
    printf("fc2-static: d2 = 1 on %d of %d diagonal producers -> D2 multiply skipped there%s\n", unit, diag,
           getenv("NANO_D2_KEEP") && atoi(getenv("NANO_D2_KEEP")) ? " (NOT: NANO_D2_KEEP)" : "");
  }
  printf("fc2-static: %s fc2-input scales (%d blocks, s in [%.4g, %.4g]); fc1 epilogue without the row maximum, fc2 INT4 GEMM without the group flush\n",
         mode == 1 ? "per-layer (sym)" : "per-128-group, folded into the fc2 weights (group)", n, smin, smax);
}
static void launchDhdLN(const half*x,int8_t*y,float*scale,int M,const DhdGPU&t,cudaStream_t stream,const int*map){
 // gamma / beta / d2 through L1 (no shared copy); NANO_DHD_LN_SMEM=1 is the previous per-CTA shared copy
 static const bool smem=getenv("NANO_DHD_LN_SMEM")!=nullptr;
 int grid=cdiv(M,8),thr=256;size_t sh=smem?(t.diagonals?3:2)*t.K*sizeof(float):0;
 if (d2Skip(t) && !smem) {                        // fc2s4 default on the static path when every d2 = 1 (or NANO_T_NOD2): exact
#define DHDLND(NV) do{if(gTNoLnH)k_ln_dhd4<NV,false,false,true><<<grid,thr,0,stream>>>(x,(uint8_t*)y,scale,t.gamma,t.beta,M,map,t.d2,t.clip);else k_ln_dhd4<NV,false,false,false><<<grid,thr,0,stream>>>(x,(uint8_t*)y,scale,t.gamma,t.beta,M,map,t.d2,t.clip);}while(0)
  LN_SWITCH(t.K,DHDLND)
#undef DHDLND
  return;
 }
 if (gTNoLnH) {                                   // TIMING ONLY: no Hadamard butterflies in the producer
#define DHDLNH(NV) do{if(t.diagonals)k_ln_dhd4<NV,true,false,true><<<grid,thr,0,stream>>>(x,(uint8_t*)y,scale,t.gamma,t.beta,M,map,t.d2,t.clip);else k_ln_dhd4<NV,false,false,true><<<grid,thr,0,stream>>>(x,(uint8_t*)y,scale,t.gamma,t.beta,M,map,t.d2,t.clip);}while(0)
  LN_SWITCH(t.K,DHDLNH)
#undef DHDLNH
  return;
 }
#define DHDLNK(NV,DG,SM) k_ln_dhd4<NV,DG,SM><<<grid,thr,sh,stream>>>(x,(uint8_t*)y,scale,t.gamma,t.beta,M,map,t.d2,t.clip)
#define DHDLN(NV) do{if(smem){if(t.diagonals)DHDLNK(NV,true,true);else DHDLNK(NV,false,true);}else{if(t.diagonals)DHDLNK(NV,true,false);else DHDLNK(NV,false,false);}}while(0)
 LN_SWITCH(t.K,DHDLN)
#undef DHDLN
#undef DHDLNK
}
template<int BM,int WM,int WN,int ST,bool DB=false,int MINB=0,bool DIAGONALS=true,bool CAPTURE=false,int F2S=0>
static void runDhdFc1Tile(const int8_t*A,const Q8W&w,const float*sa,const half*bias,half*H,int M,int N,int K,cudaStream_t stream,int8_t*HQ,float*HS,const DhdGPU&t,const float*sinv=nullptr){
 constexpr size_t sh=s8_smem<BM,128,64,ST>();
 auto kernel=k_gemm_s8_dhd<BM,128,64,WM,WN,ST,DB,true,false,false,0,true,1,false,false,64,MINB,false,false,false,true,false,DIAGONALS,CAPTURE,F2S>;
 static bool attr=false;if(!attr){CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));attr=true;}
 kernel<<<dim3(cdiv(N,128),cdiv(M,BM)),WM*WN*32,sh,stream>>>(A,w.w,sa,w.s,bias,H,M,N,kpad4(K)/2,nullptr,HQ,HS,0,nullptr,0,t.d1,t.d2,t.clip,sinv);
}
// int4_gemm_opt_20261014: the persistent kernels of i4_persist.cuh run (resident CTAs per SM) x SMs CTAs, each
// looping over tiles in raster order (tile t = row tile t / ntn, column tile t % ntn, as blockIdx.y / x before).
static int i4pGrid(const void* kern, int threads, size_t smem, int ntiles) {
  int dev = 0, sms = 0, occ = 0;
  CK(cudaGetDevice(&dev)); CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kern, threads, smem));
  return std::min(ntiles, std::max(1, occ) * sms);
}
// int4_gemm_opt_20261014: row-padded copies of the qkv / fc1 int4 weights for the persistent kernels (pitch rounded
// up to 64 bytes: K = 144 / 288 / 576 rows are 80 / 144 / 288 bytes, whose 64-byte k-tile segments straddle 128-byte
// lines).  Same codes, zero padding that the kernels never read (their k predicate stops at kpad4(K) / 2).
// NANO_I4_NOPITCH=1 keeps the packed pitch (for attribution).
static std::map<const int8_t*, std::pair<const int8_t*, int>> gI4Pitched;
static void buildInt4Pitched() {
  if (getenv("NANO_I4_NOPITCH")) { printf("NANO_I4_NOPITCH: persistent int4 GEMMs read the packed weight pitch%c", 10); return; }
  size_t bytes = 0;
  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& c = cfg.b[i];
    const Q8W* ws[2] = {&QWqkv[i], &QWfc1[i]};
    const int rows[2] = {3 * c.cp, 4 * c.dout}, ks[2] = {c.din, c.dout};
    for (int f = 0; f < 2; ++f) {
      const int src = kpad4(ks[f]) / 2, dst = roundup(src, 64);
      if (dst == src || gI4Pitched.count(ws[f]->w)) continue;
      int8_t* p; CK(cudaMalloc(&p, (size_t)rows[f] * dst)); CK(cudaMemset(p, 0, (size_t)rows[f] * dst));
      CK(cudaMemcpy2D(p, dst, ws[f]->w, src, src, rows[f], cudaMemcpyDeviceToDevice));
      gI4Pitched[ws[f]->w] = {p, dst}; bytes += (size_t)rows[f] * dst;
    }
  }
  CK(cudaDeviceSynchronize());
  printf("int4 qkv / fc1 weights: %.1f MB row-padded copies (64-byte pitch) for the persistent GEMMs%c", bytes / 1e6, 10);
}
static std::pair<const int8_t*, int> i4B(const Q8W& w, int K) {
  auto it = gI4Pitched.find(w.w);
  return it == gI4Pitched.end() ? std::make_pair((const int8_t*)w.w, kpad4(K) / 2) : it->second;
}
template<int BM,int WM,int WN,int ST,bool DB,int MINB,bool DIAGONALS,int F2S=0>
static void runDhdFc1P(const int8_t*A,const Q8W&w,const float*sa,const half*bias,int M,int N,int K,cudaStream_t stream,int8_t*HQ,float*HS,const DhdGPU&t,const float*sinv=nullptr){
 constexpr size_t sh=s8_smem<BM,128,64,ST>();
 auto kernel=k_fc1dhd4p<BM,WM,WN,ST,DB,MINB,DIAGONALS,F2S>;
 const int ntn=cdiv(N,128),ntiles=ntn*cdiv(M,BM);
 static int grid=0;if(!grid){CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));grid=i4pGrid((const void*)kernel,WM*WN*32,sh,1<<30);}
 const auto b=i4B(w,K);
 kernel<<<std::min(grid,ntiles),WM*WN*32,sh,stream>>>(A,b.first,sa,w.s,bias,HQ,HS,M,N,kpad4(K)/2,b.second,t.d1,t.d2,t.clip,ntn,ntiles,sinv);
}
// fc2static_engine_20261016: the static-scale / no-butterfly fc1 producers (F2S = 1 static, 2 dynamic without the
// butterfly [timing only], 3 static without the butterfly [timing only]); diagonals (SHAQuant exports) only, every
// candidate tile of runDhdFc1 (no capture variant).
template<int F2S,bool DG=true>
static void runDhdFc1F(int cand,const int8_t*A,const Q8W&w,const float*sa,const half*bias,half*H,int M,int N,int K,cudaStream_t s,int8_t*HQ,float*HS,const DhdGPU&t,const float*sinv){
 switch(cand){
 case 1:runDhdFc1Tile<128,2,4,3,false,0,DG,false,F2S>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
 case 2:runDhdFc1Tile<64,2,4,3,false,0,DG,false,F2S>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
 case 3:runDhdFc1Tile<256,4,2,3,false,0,DG,false,F2S>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
 case 4:runDhdFc1P<128,2,2,3,true,1,DG,F2S>(A,w,sa,bias,M,N,K,s,HQ,HS,t,sinv);break;
 case 5:runDhdFc1P<128,2,4,3,false,2,DG,F2S>(A,w,sa,bias,M,N,K,s,HQ,HS,t,sinv);break;
 default:runDhdFc1Tile<128,2,2,3,true,1,DG,false,F2S>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
 }
}
// the capture variant (NANO_DHD_CAPTURE: also writes the pre-transform fp16 row) is a separate instantiation
#define DHDFC1(BM,WM,WN,ST,DB,MINB,DG) do{if(cap)runDhdFc1Tile<BM,WM,WN,ST,DB,MINB,DG,true>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t);else runDhdFc1Tile<BM,WM,WN,ST,DB,MINB,DG,false>(A,w,sa,bias,H,M,N,K,s,HQ,HS,t);}while(0)
// candidates 4 / 5: the persistent kernel (same tile as 0 / 1); under NANO_DHD_CAPTURE they fall back to 0 / 1
#define DHDFC1P(BM,WM,WN,ST,DB,MINB,DG) do{if(cap)DHDFC1(BM,WM,WN,ST,DB,MINB,DG);else runDhdFc1P<BM,WM,WN,ST,DB,MINB,DG>(A,w,sa,bias,M,N,K,s,HQ,HS,t);}while(0)
static void runDhdFc1(int cand,const int8_t*A,const Q8W&w,const float*sa,const half*bias,half*H,int M,int N,int K,cudaStream_t s,int8_t*HQ,float*HS,const DhdGPU&t,const float*sinv=nullptr){
 static const bool cap=getenv("NANO_DHD_CAPTURE")!=nullptr;
 // fc2static_engine_20261016: static fc2 scales and / or the timing-only no-butterfly epilogue (DIAGONALS kernels; the
 // SHAQuant exports these modes are for have diagonals; a fixed-H export carries d1 = d2 = 1 and runs them unchanged)
 // fc2s4: bit 3 (no D2) by default on the static path when every d2 of this layer is 1 (d2Skip)
 const int f2s=(sinv?1:0)|(gTNoFc1H?2:0)|(gTNoFc1S?4:0)|(((sinv&&d2Skip(t))||gTNoD2)?8:0);
 if(f2s){
  if(!t.diagonals){                                // fixed-H export: no diagonal work, as the dynamic DIAGONALS=false path
   const int f=f2s&~8;                             // (no diagonal at all: the D2 bit is moot)
   if(f==1)runDhdFc1F<1,false>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);
   else if(f==3)runDhdFc1F<3,false>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);
   else{printf("fc2-static: switch combination %d not built for a fixed-H export\n",f2s);exit(1);}
   return;
  }
  switch(f2s){
  case 1:runDhdFc1F<1>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
  case 2:runDhdFc1F<2>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
  case 3:runDhdFc1F<3>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
  case 5:runDhdFc1F<5>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
  case 9:runDhdFc1F<9>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;
  case 11:runDhdFc1F<11>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;   // fc2s4: no-D2 + timing-only no-H64
  case 13:runDhdFc1F<13>(cand,A,w,sa,bias,H,M,N,K,s,HQ,HS,t,sinv);break;   // fc2s4: no-D2 + timing-only no-S
  default:printf("fc2-static: fc1 switch combination %d not built\n",f2s);exit(1);
  }
  return;}
 if(!t.diagonals){switch(cand){
 case 1:DHDFC1(128,2,4,3,false,0,false);break;
 case 2:DHDFC1(64,2,4,3,false,0,false);break;
 case 3:DHDFC1(256,4,2,3,false,0,false);break;
 case 4:DHDFC1P(128,2,2,3,true,1,false);break;
 case 5:DHDFC1P(128,2,4,3,false,2,false);break;
 default:DHDFC1(128,2,2,3,true,1,false);break;}return;}
 switch(cand){
 case 1:DHDFC1(128,2,4,3,false,0,true);break;
 case 2:DHDFC1(64,2,4,3,false,0,true);break;
 case 3:DHDFC1(256,4,2,3,false,0,true);break;
 case 4:DHDFC1P(128,2,2,3,true,1,true);break;
 case 5:DHDFC1P(128,2,4,3,false,2,true);break;
 default:DHDFC1(128,2,2,3,true,1,true);break;
 }
}
#undef DHDFC1P
#undef DHDFC1
// Bounded capture of64 actual rows per producer, enabled only for numerical gates.
static void dhdHostBin(const char*path,const void*p,size_t n){std::ofstream f(path,std::ios::binary);f.write((const char*)p,n);if(!f){printf("DHD capture write failed\n");exit(1);}}
static void captureDhd(int block,const char*family,const half*x,const int8_t*q,const float*scale,int M,const DhdGPU&t,const int*map,cudaStream_t stream){
 const char*dir=getenv("NANO_DHD_CAPTURE");if(!dir||!*dir)return;CK(cudaStreamSynchronize(stream));
 const bool fc2=strcmp(family,"fc2")==0;const int S=fc2?cdiv(t.K,128):1;
 std::vector<int>indices;for(int j=0;j<std::min(32,M);j++)indices.push_back(j);for(int j=std::max(32,M-32);j<M;j++)indices.push_back(j);
 std::vector<half>vx(indices.size()*t.K, __float2half(0.f));std::vector<int8_t>vq(indices.size()*kpad4(t.K)/2);std::vector<float>vs(indices.size()*S);std::vector<int>valid(indices.size(),1);
 for(size_t j=0;j<indices.size();j++){
  int row=indices[j],src=row;if(map)CK(cudaMemcpy(&src,map+row,4,cudaMemcpyDeviceToHost));valid[j]=src>=0;
  if(src>=0)CK(cudaMemcpy(vx.data()+j*t.K,x+(size_t)src*t.K,t.K*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vq.data()+j*kpad4(t.K)/2,q+(size_t)row*kpad4(t.K)/2,kpad4(t.K)/2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(vs.data()+j*S,scale+(size_t)row*S,S*4,cudaMemcpyDeviceToHost));
 }
 std::string prefix=std::string(dir)+"/b"+std::to_string(block)+"_"+family;
 dhdHostBin((prefix+"_x.bin").c_str(),vx.data(),vx.size()*2);dhdHostBin((prefix+"_q.bin").c_str(),vq.data(),vq.size());dhdHostBin((prefix+"_scale.bin").c_str(),vs.data(),vs.size()*4);dhdHostBin((prefix+"_valid.bin").c_str(),valid.data(),valid.size()*4);
 if(!fc2){std::vector<float>g(t.K),b(t.K);CK(cudaMemcpy(g.data(),t.gamma,t.K*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(b.data(),t.beta,t.K*4,cudaMemcpyDeviceToHost));dhdHostBin((prefix+"_gamma.bin").c_str(),g.data(),t.K*4);dhdHostBin((prefix+"_beta.bin").c_str(),b.data(),t.K*4);}
}

static void launchLNQ4(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C, cudaStream_t s, const int* src = nullptr) {
  int grid = cdiv(T, 8), thr = 256; size_t sh = 2 * C * sizeof(float);
  uint8_t* yu = (uint8_t*)y;
#define LNQ4(NV) k_ln_q4h<NV><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, src)
  LN_SWITCH(C, LNQ4)
#undef LNQ4
}
// --mon: k_ln_mon4<C, BB, RC, HL, LM, FA> as a persistent grid (SMs x resident CTAs, capped at the 16-row tiles).  RC is a bit mask:
// bit 0 keeps B1's coefficient fragments in registers, bit 1 B2's (the clear sets are read through L1 per tile).
// Defaults: 3 at C = 144 / 288, 1 at C = 576 (both sets = 171 registers, one CTA per SM), 0 at C = 1152;
// NANO_MON_RC = "a,b,c,d" overrides per C = 144 / 288 / 576 / 1152.
static int gSMs = 0;
template<int C, int BB, int RC, int HL, int LM, int FA>
static void runMon4(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, const MonLN& m, cudaStream_t s, const int* src) {
  using MC = MonCfg4<C, BB, LM>;
  static int occ = 0;
  if (!occ) {
    CK(cudaFuncSetAttribute(k_ln_mon4<C, BB, RC, HL, LM, FA>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)MC::SMEM));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_ln_mon4<C, BB, RC, HL, LM, FA>, 256, MC::SMEM));
    cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, k_ln_mon4<C, BB, RC, HL, LM, FA>));
    if (occ < 1) occ = 1;
    printf("mon kernel C %d b %d n %d RC %d HL %d LM %d FA %d: smem %zu B, %d registers, %zu B local, %d CTA/SM\n", C, BB, MC::N, RC, HL, LM, FA, MC::SMEM,
           fa.numRegs, (size_t)fa.localSizeBytes, occ);
  }
  if (m.C != C || m.BB != BB || T % 16) { printf("mon: LN C %d b %d (factors C %d b %d), T %d\n", C, BB, m.C, m.BB, T); exit(1); }
  const int grid = std::min(T / 16, gSMs * occ);
  k_ln_mon4<C, BB, RC, HL, LM, FA><<<grid, 256, MC::SMEM, s>>>(x, (uint8_t*)y, ys, g, b, m.f1, m.f1t, m.f2, m.f2t, T, src, 1e-6f);
}
// the NANO_MON_* env index of a LayerNorm width: the model's stage with that width (hiera-L: 144 / 288 / 576 / 1152 -> 0..3)
static int monStage(int C) {
  for (int s = 0; s < 4; ++s) if (cfg.dims[s] == C) return s;
  return 3;
}
static int monFA(int C) {
  static int v[4] = {-2, -2, -2, -2};
  if (v[0] == -2) {
    v[0] = v[1] = v[2] = v[3] = -1;
    if (const char* e = getenv("NANO_MON_FA")) { sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]); printf("NANO_MON_FA=%s (fp16 accumulation in the mon products)\n", e); }
  }
  const int st = monStage(C);
  return v[st] > 0 ? 1 : 0;
}
static int monLM(int C) {
  static int v[4] = {-2, -2, -2, -2};
  if (v[0] == -2) {
    v[0] = v[1] = v[2] = v[3] = -1;
    if (const char* e = getenv("NANO_MON_LM")) { sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]); printf("NANO_MON_LM=%s\n", e); }
  }
  const int st = monStage(C);
  return v[st] > 0 ? 1 : 0;
}
static int monHL(int C) {
  static int v[4] = {-2, -2, -2, -2};
  if (v[0] == -2) {
    v[0] = v[1] = v[2] = v[3] = -1;
    if (const char* e = getenv("NANO_MON_HL")) { sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]); printf("NANO_MON_HL=%s\n", e); }
  }
  const int st = monStage(C);
  return v[st] > 0 ? 1 : 0;
}
static int monRC(int C) {
  static int v[4] = {-2, -2, -2, -2};
  if (v[0] == -2) {
    v[0] = v[1] = v[2] = v[3] = -1;
    if (const char* e = getenv("NANO_MON_RC")) { sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]); printf("NANO_MON_RC=%s\n", e); }
  }
  const int st = monStage(C);
  return v[st] >= 0 ? v[st] : (C <= 448 ? 3 : C <= 896 ? 1 : 0);
}
static void launchLNMon4(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C, const MonLN& m,
                         cudaStream_t s, const int* src = nullptr) {
  const int rc = monFA(C) ? 16 + (monRC(C) & 3) : (monRC(C) & 3) + 4 * monHL(C) + 8 * monLM(C);
#define MONV(CC, BBB, V) runMon4<CC, BBB, ((V) & 3), (((V) >> 2) & 1), (((V) >> 3) & 1), ((V) >> 4)>(x, y, ys, g, b, T, m, s, src)
#define MON(CC, BBB) do { switch (rc) { \
    case 1: MONV(CC, BBB, 1); break;   case 2: MONV(CC, BBB, 2); break;   case 3: MONV(CC, BBB, 3); break; \
    case 4: MONV(CC, BBB, 4); break;   case 5: MONV(CC, BBB, 5); break;   case 6: MONV(CC, BBB, 6); break; \
    case 7: MONV(CC, BBB, 7); break;   case 8: MONV(CC, BBB, 8); break;   case 9: MONV(CC, BBB, 9); break; \
    case 10: MONV(CC, BBB, 10); break; case 11: MONV(CC, BBB, 11); break; case 12: MONV(CC, BBB, 12); break; \
    case 13: MONV(CC, BBB, 13); break; case 14: MONV(CC, BBB, 14); break; case 15: MONV(CC, BBB, 15); break; \
    case 16: MONV(CC, BBB, 16); break; case 17: MONV(CC, BBB, 17); break; case 18: MONV(CC, BBB, 18); break; \
    case 19: MONV(CC, BBB, 19); break; \
    default: MONV(CC, BBB, 0); } } while (0)
#define MONR(CC, BBB) do { if (rc >= 4) { static bool w = false; if (!w) { printf("mon C %d: HL / LM / FA not built for this width, RC %d only\n", CC, rc & 3); w = true; } } \
    switch (rc & 3) { case 1: MONV(CC, BBB, 1); break; case 2: MONV(CC, BBB, 2); break; case 3: MONV(CC, BBB, 3); break; \
    default: MONV(CC, BBB, 0); } } while (0)
  switch (C) {
    case 144:  if (m.BB == 36) MON(144, 36); else MON(144, 16); break;
    case 288:  MON(288, 32); break;
    case 576:  MON(576, 32); break;
    case 1152: MON(1152, 32); break;
    case 96:   MONR(96, 24); break;
    case 112:  MONR(112, 16); break;
    case 192:  MONR(192, 32); break;
    case 224:  MONR(224, 32); break;
    case 384:  MONR(384, 32); break;
    case 448:  MONR(448, 32); break;
    case 768:  MONR(768, 32); break;
    case 896:  MONR(896, 32); break;
    default: printf("--mon: no k_ln_mon4 for LN width %d\n", C); exit(1);
  }
#undef MONR
#undef MON
#undef MONV
}
// k_gemm_s8<..., I4 = true>: K is given in ELEMENTS here (the row is padded to kpad4(K) codes) and passed to the
// kernel in bytes (kpad4(K) / 2); BK in bytes.  EPI: 0 plain (qkv), 1 W8G16 (qkv under --win8), 2 QKV8 (--qk8 fused)
template<int EPI, bool GELU, bool RESID, int GROUP, bool Q8OUT, int ROT8, int BM, int BN, int WM, int WN, int ST,
         bool DB = false, int MINB = 0, bool FF = false>
static void runS4(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K,
                  cudaStream_t s, const half* resid = nullptr, int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0,
                  int qkn = 0) {
  constexpr int BK = 64;
  constexpr int NGPT = EPI == 2 ? 2 : 1;
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  auto kern = k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, false, RESID, GROUP, Q8OUT, NGPT, false, false, ROT8, MINB, false,
                        EPI == 2, EPI == 1, true, FF>;
  static bool attr = false;
  if (!attr) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
  if (M % BM) { printf("int4 GEMM: M %d not a multiple of BM %d\n", M, BM); exit(1); }
  kern<<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, kpad4(K) / 2, resid, C8, sa2,
                                                          sa_stride, nullptr, qkn, nullptr, 1.f);
}

static int bd32cand(){const char*e=getenv("NANO_BD32_CANDIDATE");return e?atoi(e):0;}
template<int BM,int WM,int WN,int ST,bool DB,int MINB=0>
static void runBD32(const int8_t*A,const Q8W&w,const float*sa,const half*bias,half*H,int M,int N,int K,cudaStream_t stream,int8_t*HQ,float*HS,const StructureGPU&t){
 constexpr size_t ring=s8_smem<BM,128,64,ST>();constexpr size_t scratch=BM*128*4+WM*WN*2048;
 constexpr size_t sh=ring>scratch?ring:scratch;
 auto kern=k_gemm_s8<BM,128,64,WM,WN,ST,DB,true,false,false,0,false,1,false,false,0,MINB,false,false,false,true,false,false,true>;
 static bool attr=false;if(!attr){CK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh));cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,kern));printf("BD32_RESOURCE BM%d WM%d WN%d ST%d DB%d regs%d local%zu static%zu dynamic%zu\n",BM,WM,WN,ST,DB,at.numRegs,at.localSizeBytes,at.sharedSizeBytes,sh);attr=true;}
 if(M%BM||N%32||t.b!=32){printf("BAD_BD32_TILE\n");exit(2);}kern<<<dim3(cdiv(N,128),M/BM),WM*WN*32,sh,stream>>>(A,w.w,sa,w.s,bias,H,M,N,kpad4(K)/2,nullptr,HQ,HS,0,nullptr,0,(const half*)t.first,t.clip);
}
static void runBD32Choice(int cand,const int8_t*A,const Q8W&w,const float*sa,const half*bias,half*H,int M,int N,int K,cudaStream_t stream,int8_t*HQ,float*HS,const StructureGPU&t){
 switch(cand){
  case 10:runBD32<32,2,2,3,false>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  case 11:runBD32<64,2,2,3,false>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  case 12:runBD32<128,2,2,3,false>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  case 13:runBD32<64,2,2,3,true,1>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  case 14:runBD32<128,2,2,3,true,1>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  case 15:runBD32<64,2,4,3,false>(A,w,sa,bias,H,M,N,K,stream,HQ,HS,t);break;
  default:printf("BAD_BD32_CANDIDATE %d\n",cand);exit(2);
 }
}
// Tile choice per GEMM and stage (default = the INT8 path's sm_89 / generic rule, see i4tile).  NANO_I4_QKV /
// NANO_I4_FC1 / NANO_I4_FC2 = "a,b,c,d" override the candidate index per stage (0..3) for tuning.
static int i4env(const char* name, int stage, int dflt) {
  const char* e = getenv(name);
  if (!e) return dflt;
  int v[4] = {-1, -1, -1, -1};
  sscanf(e, "%d,%d,%d,%d", &v[0], &v[1], &v[2], &v[3]);
  return (stage >= 0 && stage < 4 && v[stage] >= 0) ? v[stage] : dflt;
}
// qkv candidates (EPI 0 plain / 1 W8G16 / 2 QKV8 share them):
//   0 128x128 2x4 ST3   1 128x128 2x2 ST2   2 128x64 2x2 ST3   3 64x128 2x4 ST3   4 128x256 2x4 ST3   5 256x128 4x2 ST3
//   int4_gemm_opt_20261014, W8G16 only (other epilogues: 6 -> 1, 7 -> 0): the persistent kernel k_qkv4p with the
//   next tile's first k-tiles prefetched during the epilogue:   6 128x128 2x2 ST2   7 128x128 2x4 ST3
template<int BM, int BN, int WM, int WN, int ST, bool DB = false, int MINB = 0>
static void runQkv4P(const int8_t* A, const Q8W& w, const float* sa, const half* bias, int M, int N, int K,
                     cudaStream_t s, int8_t* C8, float* sa2, int sa_stride) {
  constexpr size_t sh = s8_smem<BM, BN, 64, ST>();
  auto kern = k_qkv4p<BM, BN, WM, WN, ST, DB, MINB>;
  static int grid = 0;
  if (!grid) { CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); grid = i4pGrid((const void*)kern, WM * WN * 32, sh, 1 << 30); }
  if (M % BM) { printf("int4 GEMM: M %d not a multiple of BM %d\n", M, BM); exit(1); }
  const int ntn = cdiv(N, BN), ntiles = ntn * (M / BM);
  const auto b = i4B(w, K);
  kern<<<std::min(grid, ntiles), WM * WN * 32, sh, s>>>(A, b.first, sa, w.s, bias, C8, sa2, M, N, kpad4(K) / 2, b.second, sa_stride, ntn, ntiles);
}
template<int EPI>
static void runQkv4(int cand, const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K,
                    cudaStream_t s, int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0) {
#define QKV4(BM, BN, WM, WN, ST) runS4<EPI, false, false, 0, false, 0, BM, BN, WM, WN, ST>(A, w, sa, bias, C, M, N, K, s, nullptr, C8, sa2, sa_stride)
  if (EPI != 1 && cand >= 6) cand = cand == 6 ? 1 : 0;
  switch (cand) {
    case 1: QKV4(128, 128, 2, 2, 2); break;
    case 2: QKV4(128, 64, 2, 2, 3); break;
    case 3: QKV4(64, 128, 2, 4, 3); break;
    case 4: QKV4(128, 256, 2, 4, 3); break;
    case 5: QKV4(256, 128, 4, 2, 3); break;
    case 6: if constexpr (EPI == 1) runQkv4P<128, 128, 2, 2, 2>(A, w, sa, bias, M, N, K, s, C8, sa2, sa_stride); break;
    case 7: if constexpr (EPI == 1) runQkv4P<128, 128, 2, 4, 3, false, 2>(A, w, sa, bias, M, N, K, s, C8, sa2, sa_stride); break;
    default: QKV4(128, 128, 2, 4, 3); break;
  }
#undef QKV4
}
// --qk8 fused on the int4 path (head dims 96 / 64, as runQkv8): BN = two whole padded heads
static void runQkv84(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K, int qkn, cudaStream_t s) {
  if (cfg.dhp == 96) runS4<2, false, false, 0, false, 0, 128, 192, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s, nullptr, QK8, QKS, gQKSst, qkn);
  else               runS4<2, false, false, 0, false, 0, 128, 128, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s, nullptr, QK8, QKS, gQKSst, qkn);
}
// fc1 candidates (GELU + H_64 + per-(token, 128) int4 producer; ROT8 needs BN = 128):
//   0 128x128 2x2 ST3 DB MINB1 (the INT8 sm_89 tile)   1 128x128 2x4 ST3   2 64x128 2x4 ST3   3 256x128 4x2 ST3
static void runFc14(int cand, const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K,
                    cudaStream_t s, int8_t* C8, float* sa2) {
  switch (cand) {
    case 1: runS4<0, true, false, 0, true, 64, 128, 128, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    case 2: runS4<0, true, false, 0, true, 64, 64, 128, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    case 3: runS4<0, true, false, 0, true, 64, 256, 128, 4, 2, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    default: runS4<0, true, false, 0, true, 64, 128, 128, 2, 2, 3, true, 1>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
  }
}
// fc2 candidates (bias + residual, per-(token, 128) group flush):
//   0 128x192 2x4 ST3   1 64x192 2x4 ST3   2 128x64 4x2 ST3   3 128x128 2x4 ST3   4 64x128 2x4 ST3   5 64x64 2x2 ST3
//   6 128x128 4x4 ST2   7 128x192 2x4 ST3 FF   8 128x64 4x2 ST3 FF
//   int4_gemm_opt_20261014, column tiles that divide hiera-L's N = 144 * 2^s (padded-row staging, CPAD):
//   9 128x144 8x1 ST3   10 128x144 2x3 ST3   11 64x144 4x1 ST3   12 64x144 2x3 ST3   13 128x96 4x2 ST3
//   14 128x144 8x1 ST4   15 64x144 4x1 ST4   16 128x192 2x4 ST4
static void runFc24(int cand, const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* X_, int M, int N, int K,
                    cudaStream_t s, int sastride) {
#define FC24(BM, BN, WM, WN, ST, FFX) runS4<0, false, true, 128, false, 0, BM, BN, WM, WN, ST, false, 0, FFX>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride)
  switch (cand) {
    case 1: FC24(64, 192, 2, 4, 3, false); break;
    case 2: FC24(128, 64, 4, 2, 3, false); break;
    case 3: FC24(128, 128, 2, 4, 3, false); break;
    case 4: FC24(64, 128, 2, 4, 3, false); break;
    case 5: FC24(64, 64, 2, 2, 3, false); break;
    case 6: FC24(128, 128, 4, 4, 2, false); break;
    case 7: FC24(128, 192, 2, 4, 3, true); break;
    case 8: FC24(128, 64, 4, 2, 3, true); break;
    case 9: FC24(128, 144, 8, 1, 3, false); break;
    case 10: FC24(128, 144, 2, 3, 3, false); break;
    case 11: FC24(64, 144, 4, 1, 3, false); break;
    case 12: FC24(64, 144, 2, 3, 3, false); break;
    case 13: FC24(128, 96, 4, 2, 3, false); break;
    case 14: FC24(128, 144, 8, 1, 4, false); break;
    case 15: FC24(64, 144, 4, 1, 4, false); break;
    case 16: FC24(128, 192, 2, 4, 4, false); break;
    default: FC24(128, 192, 2, 4, 3, false); break;
  }
#undef FC24
}
// fc2static_engine_20261016: fc2 WITHOUT the per-group flush (k_gemm_s8 <I4, GROUP = 0, SASCALAR, RESID>): one int32
// accumulator over the whole K (|sum| <= 49 K < 2^31 for every hiera K), epilogue acc * (sa[0] * sw[col]) + bias + resid.
// sa[0] is the static scale s (sym), 1 (group: the scales are in the weights), or, under the timing-only
// NANO_T_NOFLUSH, HS[0] (dynamic codes, wrong numerics).
template<int BM, int BN, int WM, int WN, int ST, bool DB = false, int MINB = 0>
static void runS4S(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K,
                   cudaStream_t s, const half* resid) {
  constexpr int BK = 64;
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  auto kern = k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, false, true, true, 0, false, 1, false, false, 0, MINB, false,
                        false, false, true, false>;
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    if (getenv("NANO_F2S_VERBOSE")) {
      cudaFuncAttributes fa; CK(cudaFuncGetAttributes(&fa, kern));
      printf("fc2 no-flush tile %dx%d %dx%d ST%d: %d registers, %zu B local, smem %zu B%c", BM, BN, WM, WN, ST, fa.numRegs, (size_t)fa.localSizeBytes, sh, 10);
    }
    attr = true;
  }
  if (M % BM) { printf("int4 GEMM: M %d not a multiple of BM %d\n", M, BM); exit(1); }
  kern<<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, kpad4(K) / 2, resid, nullptr, nullptr,
                                                          0, nullptr, 0, nullptr, 1.f);
}
// candidates: 0..16 = the tiles of runFc24 (7 / 8 = 0 / 2: FF is a flush option), plus wider warp tiles that the
// flush path's register budget excluded (its fp32 shadow accumulator is gone here):
//   17 128x144 4x1 ST3   18 128x288 4x2 ST3   19 128x128 2x2 ST3   20 128x192 4x2 ST3   21 128x96 2x2 ST3
//   22 256x96 4x2 ST3    23 64x288 2x2 ST3    24 128x96 4x1 ST3    25 64x144 2x1 ST3    26 256x128 4x2 ST3
static void runFc24S(int cand, const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* X_, int M, int N, int K,
                     cudaStream_t s) {
#define FC24S(BM, BN, WM, WN, ST) runS4S<BM, BN, WM, WN, ST>(A, w, sa, bias, X_, M, N, K, s, X_)
  switch (cand) {
    case 1: FC24S(64, 192, 2, 4, 3); break;
    case 2: case 8: FC24S(128, 64, 4, 2, 3); break;
    case 3: FC24S(128, 128, 2, 4, 3); break;
    case 4: FC24S(64, 128, 2, 4, 3); break;
    case 5: FC24S(64, 64, 2, 2, 3); break;
    case 6: FC24S(128, 128, 4, 4, 2); break;
    case 9: FC24S(128, 144, 8, 1, 3); break;
    case 10: FC24S(128, 144, 2, 3, 3); break;
    case 11: FC24S(64, 144, 4, 1, 3); break;
    case 12: FC24S(64, 144, 2, 3, 3); break;
    case 13: FC24S(128, 96, 4, 2, 3); break;
    case 14: FC24S(128, 144, 8, 1, 4); break;
    case 15: FC24S(64, 144, 4, 1, 4); break;
    case 16: FC24S(128, 192, 2, 4, 4); break;
    case 17: FC24S(128, 144, 4, 1, 3); break;
    case 18: FC24S(128, 288, 4, 2, 3); break;
    case 19: FC24S(128, 128, 2, 2, 3); break;
    case 20: FC24S(128, 192, 4, 2, 3); break;
    case 21: FC24S(128, 96, 2, 2, 3); break;
    case 22: FC24S(256, 96, 4, 2, 3); break;
    case 23: FC24S(64, 288, 2, 2, 3); break;
    case 24: FC24S(128, 96, 4, 1, 3); break;
    case 25: FC24S(64, 144, 2, 1, 3); break;
    case 26: FC24S(256, 128, 4, 2, 3); break;
    default: FC24S(128, 192, 2, 4, 3); break;
  }
#undef FC24S
}
// defaults: the INT8 path's rules with K counted in elements (sm_89: qkv 2x2 / 2 stages from K = 576, fc1 2x2 DB,
// fc2 by N; elsewhere the generic tiles)
static int i4QkvDefault(int K) { return (gArch89 && K >= 576) ? 1 : 0; }
static int i4Fc1Default() { return gArch89 ? 0 : 1; }
static int i4Fc2Default(int N) { return !gArch89 ? 3 : N == 288 ? 2 : N == 1152 ? 1 : 0; }
// int4_gemm_opt_20261014: sm_89 rules for hiera-L's widths (K or N a multiple of 144), chosen from interleaved
// per-GEMM timings at batch 1 and 8 (bench/paper/int4_gemm_opt_20261014/RESULTS.txt).  Every other width, GPU and
// epilogue keeps the rules above.  All candidates of one GEMM give the same output bits (the int32 accumulation is
// order-free and the epilogues are per element / per row group), so these are speed choices only.
//   qkv under --win8 (W8G16): K = 144 / 576 -> 0 (128x128 2x4 ST3), K = 288 -> 7 (persistent 2x4), K = 1152 -> 6
//   (persistent 2x2).
//   DHD fc1: K <= 288 -> 4 (persistent 2x2 DB); larger K: up to 8 tiles per SM -> 5 (persistent 2x4), else the
//   non-persistent 2x2 DB tile (0) without diagonals, 2x4 (1) with them.
//   fc2: N = 144 / 288 -> 9 (128x144); otherwise the 128x192 tile (0) once its grid fills the SMs, else 144-wide
//   tiles: 128x144 (9) or, when that is still short of the SMs, 64x144 (12).
static int i4QkvDefaultW8(int K) {
  if (gArch89 && K % 144 == 0) return K == 288 ? 7 : K >= 1152 ? 6 : 0;
  return i4QkvDefault(K);
}
static int i4Fc1DhdDefault(int M, int K, bool diagonals) {
  if (!gArch89 || K % 144) return i4Fc1Default();
  if (K <= 288) return 4;
  const long long tiles = (long long)cdiv(M, 128) * cdiv(4 * K, 128);
  return tiles <= 8LL * gSMs ? 5 : (diagonals ? 1 : 0);
}
static int i4Fc2DefaultM(int N, int M) {
  if (!gArch89 || N % 144) return i4Fc2Default(N);
  if (N <= 288) return 9;
  if ((long long)(M / 128) * cdiv(N, 192) >= gSMs) return 0;
  return (long long)(M / 128) * (N / 144) >= gSMs ? 9 : 12;
}
// fc2static_engine_20261016: the default tile of the no-flush fc2 (runFc24S), sm_89 rule from the brief tune
// (runs/tune: per-stage fc2_gemm of one eager pass, two interleaved repeats, Hiera-L / Hiera-T at B1 / B8; RESULTS.txt):
//   * many waves of 128x128 tiles (>= 3 per SM) and N >= 384: 3 (128x128 2x4 ST3); Hiera-L B8 stage 2 4.37 vs 4.68 ms
//     for the flush path's 128x192 tile, Hiera-T B8 stages 2 / 3
//   * otherwise Hiera-L's widths (N a multiple of 144): the flush path's rule (128x144 / 64x144 / 128x192)
//   * otherwise (Hiera-T's widths): N <= 96 -> 13 (128x96 4x2); 128x192 (0) once its grid fills the SMs; 128x96 (13) when
//     that fills them; else 4 (64x128 2x4)
// The wider-warp candidates 17..26 never won by more than the noise.  Every candidate gives the same output bits (one
// int32 accumulator, per-element epilogue).  NANO_I4_FC2 overrides per stage as for runFc24; NANO_F2S_FLUSHTILE=1 uses the
// flush path's rule (A/B of the tile choice).  Non-sm_89 GPUs keep the flush path's rule.
static int i4Fc2StaticDefault(int N, int M) {
  static const bool flushTile = getenv("NANO_F2S_FLUSHTILE") && atoi(getenv("NANO_F2S_FLUSHTILE"));
  if (flushTile || !gArch89) return i4Fc2DefaultM(N, M);
  const long long mt = M / 128;
  if (N >= 384 && mt * cdiv(N, 128) >= 3LL * gSMs) return 3;
  if (N % 144 == 0) return i4Fc2DefaultM(N, M);
  if (N <= 96) return 13;
  if (mt * cdiv(N, 192) >= gSMs) return 0;
  if (N % 96 == 0 && mt * (N / 96) >= gSMs) return 13;
  return 4;
}

// sm_120 (RTX 5090): fp16 GEMM + bias + residual on the engine's own mma.sync core (sam21_gemm_res.cuh, the
// fc1+GELU kernel with the epilogue swapped).  cuBLASLt's kernels for the BIAS epilogue with beta = 1 run at
// 1.7-2.5x the plain GEMM's time there (build5090/retune/projbench.log: 4096x576x576 0.0223 vs 0.0123 ms,
// an exhaustive algo/tile/split-K search included; bias alone or beta = 1 alone are fine), and this kernel
// with the same fp16 accumulation is 1.3-1.9x faster than that plan.  128x64 tiles (4x1 warps) when N is not
// a multiple of 128 or the 128x128 grid would leave SMs idle, else 128x128 (2x4 warps).  M must be a
// multiple of 128 (every engine pads its token count to that).
static void runResF16(const half* A, const half* W, const half* bias, const half* R, half* D,
                      int M, int N, int K, cudaStream_t s) {
  const int ctas = cdiv(M, 128) * cdiv(N, 128);
  if (N % 128 != 0 || ctas < 170) {
    constexpr size_t sh = gemm_res_smem<128, 64, 32, 3, true>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_res<128, 64, 32, 4, 1, 3, true, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k_gemm_res<128, 64, 32, 4, 1, 3, true, true, true><<<dim3(cdiv(N, 64), M / 128), 128, sh, s>>>(A, W, bias, R, D, M, N, K);
  } else {
    constexpr size_t sh = gemm_res_smem<128, 128, 32, 3, true>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_res<128, 128, 32, 2, 4, 3, true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k_gemm_res<128, 128, 32, 2, 4, 3, true, true, false><<<dim3(cdiv(N, 128), M / 128), 256, sh, s>>>(A, W, bias, R, D, M, N, K);
  }
}
// the two cuBLASLt plans with a residual (proj, and fc2 on the fp16 path) take this kernel on sm_120 under --f16acc
static bool useResF16(int M) { return gArch120 && gAccMode == 2 && M > 512; }
static void runFc1Gelu(const half* A, const half* Wt, const half* bias, half* C, int M, int N, int K, bool f16, cudaStream_t s) {
  if (f16) {
    constexpr size_t sh = gemm_gelu_smem<128, 128, 32, 3, true>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_gelu<128, 128, 32, 2, 4, 3, true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k_gemm_gelu<128, 128, 32, 2, 4, 3, true, true, false><<<dim3(cdiv(N, 128), M / 128), 256, sh, s>>>(A, Wt, bias, C, M, N, K);
  } else {
    constexpr size_t sh = gemm_gelu_smem<128, 64, 32, 4, true>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k_gemm_gelu<128, 64, 32, 2, 2, 4, false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k_gemm_gelu<128, 64, 32, 2, 2, 4, false, true, true><<<dim3(cdiv(N, 64), M / 128), 128, sh, s>>>(A, Wt, bias, C, M, N, K);
  }
}

// Attention dispatch.  With the head-dim-templated kernels (ATTN2_DH_TEMPLATE, sam21_attn2.cuh)
// every variant runs; the legacy headers only serve DH = 72 (hiera-L).
//
// The kernels put (window, head) on blockIdx.y, which CUDA caps at 65535, and a batch of N
// images has N times as many windows (hiera-L stage 1 is already 1024 windows x 8 heads, so
// N = 8 would ask for 65536).  Every base address the kernel forms is a whole number of
// windows from its argument -- K/Q at w*ws2 rows of qkv, the pooled Q at (w*heads + h)*ws2q
// rows, the output at w*ws2q rows -- so the windows are launched in chunks with the three
// pointers advanced by the chunk's first window.  One chunk (w0 = 0) covers every batch-1
// shape, which is why batch 1 is unchanged.
static void runAttnG(int nWtot, const half* qkv0, const half* qpool0, half* xout0, int ws2, int ws2q, int heads, int Cp, int Co, cudaStream_t s) {
  int acc = gAttnAcc >= 0 ? gAttnAcc : 1;
  const int maxW = 65535 / heads;
  for (int w0 = 0; w0 < nWtot; w0 += maxW) {
  const int B = std::min(maxW, nWtot - w0) * heads;
  const half* qkv = qkv0 + (size_t)w0 * ws2 * (3 * Cp);
  const half* qpool = qpool0 ? qpool0 + (size_t)w0 * heads * ws2q * cfg.dhp : nullptr;
  half* xout = xout0 + (size_t)w0 * ws2q * Co;
#ifdef ATTN2_DH_TEMPLATE
#define ATTN_DH(DHX, NW, BK, QF, PF) \
  k_attn2<DHX, NW, BK, 2, QF, PF><<<dim3(cdiv(ws2q, NW * 16), B), NW * 32, attn2_smem<DHX, BK, 2>(), s>>>(qkv, qpool, xout, ws2, ws2q, heads, Cp, Co, ws2q, ws2)
#define ATTN_CFG(DHX, NW, BK) do { if (acc == 2) ATTN_DH(DHX, NW, BK, true, true); else if (acc == 1) ATTN_DH(DHX, NW, BK, false, true); else ATTN_DH(DHX, NW, BK, false, false); } while (0)
// sm_120 (build5090/retune/attnsweep_{72,96,56}.log): one 4-warp CTA per 8x8 window (1.24-1.34x), and the MT=2 kernel
// (32 query rows per warp, 128-query CTAs) on the >= 256-query blocks except for head dim 96, where it does not win
#define ATTN_M2(DHX) k_attn2m<DHX, 4, 2, 32, 2, true, 2><<<dim3(cdiv(ws2q, 128), B), 128, attn2_smem<DHX, 32, 2>(), s>>>(qkv, qpool, xout, ws2, ws2q, heads, Cp, Co, ws2q, ws2)
// The same tile without the MINB = 2 cap: on sm_86 the d = 96 global block (hiera-T/S stage 3,
// 4096 queries x 4096 keys) is 16-20 % faster with it than with NW4_BK32 (work_attn_sam3/
// d96_3090.csv and bench/paper/attn_compare/raw/{3090,3090b,a10}_attn_bench.csv).  sm_89 and
// sm_120 keep NW4_BK32 -- there MT = 2 loses at d = 96 -- and so does every other d = 96 shape.
#define ATTN_M2B(DHX) k_attn2m<DHX, 4, 2, 32, 2, true, 1><<<dim3(cdiv(ws2q, 128), B), 128, attn2_smem<DHX, 32, 2>(), s>>>(qkv, qpool, xout, ws2, ws2q, heads, Cp, Co, ws2q, ws2)
// sm_89 takes it too once the batch gives the global block >= 16 (image, head) rows: at batch 8
// on hiera-T 1.04 against 1.22 ms for NW4_BK32 (micro/attnq8.cu on the 4090), while at batch 1
// (4 rows, 128 CTAs of 128 queries) NW4_BK32 stays ahead, 0.154 against 0.166 ms.
#define ATTN_ANY(DHX) do { if (ws2q <= 16) ATTN_CFG(DHX, 1, 16); \
  else if (ws2 == 64) { if (gArch120) ATTN_CFG(DHX, 4, 32); else ATTN_CFG(DHX, 2, 32); } \
  else if (gArch120 && acc == 1 && ws2q >= 256 && DHX != 96) ATTN_M2(DHX); \
  else if ((gArch86 || (gArch89 && B >= 16)) && acc == 1 && ws2q >= 4096 && DHX == 96) ATTN_M2B(DHX); \
  else ATTN_CFG(DHX, 4, 32); } while (0)
  switch (cfg.dh) {
    case 56: ATTN_ANY(56); break;
    case 72: ATTN_ANY(72); break;
    case 96: ATTN_ANY(96); break;
    default: printf("no attention kernel for head dim %d\n", cfg.dh); exit(1);
  }
#undef ATTN_ANY
#undef ATTN_M2B
#undef ATTN_M2
#undef ATTN_CFG
#undef ATTN_DH
#else
  if (cfg.dh != DH) { printf("this build's attention serves head dim %d only (model: %d)\n", DH, cfg.dh); exit(1); }
#define ATTN2(NW, BK, QF, PF) \
  k_attn2<NW, BK, 2, QF, PF><<<dim3(cdiv(ws2q, NW * 16), B), NW * 32, attn2_smem<BK, 2>(), s>>>(qkv, qpool, xout, ws2, ws2q, heads, Cp, Co, ws2q, ws2)
#define ATTN2_ACC(NW, BK) do { if (acc == 2) ATTN2(NW, BK, true, true); else if (acc == 1) ATTN2(NW, BK, false, true); else ATTN2(NW, BK, false, false); } while (0)
  if (ws2q <= 16) ATTN2_ACC(1, 16);
  else if (ws2 == 64) ATTN2_ACC(2, 32);
  else ATTN2_ACC(4, 32);
#undef ATTN2_ACC
#undef ATTN2
#endif
  }
}

// --qk8 attention of a global block: one window per image, (image, head) on blockIdx.y.
// NW4 MT2 BK64 (micro/attnq8.cu on the 3090: 1.52x the shipped fp16 kernel on hiera-T's
// 4096-token block at batch 8).
// Windowed blocks (--qk8-win) take one 4-warp CTA of 16 or 32 query rows per warp by window
// size, in window chunks as runAttnG (blockIdx.y <= 65535); the chunk offset moves the three
// row-indexed bases (q8, its scales, V) and the output, as there.
static void runAttnQ8(int nWtot, const half* qkv0, half* xout0, int ws2, int heads, int Cp, int Co, cudaStream_t s) {
  const int dh8 = attn2q8_dh8(cfg.dh), ld8 = 2 * heads * dh8;
  const int maxW = 65535 / heads;
  for (int w0 = 0; w0 < nWtot; w0 += maxW) {
  const int B = std::min(maxW, nWtot - w0) * heads;
  const size_t r0 = (size_t)w0 * ws2;
  const int8_t* q8 = QK8 + r0 * ld8;
  const float* s8 = QKS + r0;
  const half* qkv = qkv0 + r0 * (3 * Cp);
  half* xout = xout0 + r0 * Co;
#define ATTNQ8K(DHX, MT, BK) do { \
    constexpr size_t sh = attn2q8_smem<DHX, BK, 2>(); \
    static bool attr = false; \
    if (!attr && sh > 48 * 1024) { CK(cudaFuncSetAttribute(k_attn2q8<DHX, 4, MT, BK, 2, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; } \
    k_attn2q8<DHX, 4, MT, BK, 2, 1><<<dim3(cdiv(ws2, 64 * MT), B), 128, sh, s>>>(q8, s8, qkv, xout, ws2, ws2, heads, Cp, Co, ws2, ws2, \
                                                                             ld8, heads * attn2q8_dh8(DHX), gQKSst); } while (0)
#define ATTNQ8(DHX) do { if (ws2 >= 1024) ATTNQ8K(DHX, 2, 64); else if (ws2 > 64) ATTNQ8K(DHX, 1, 64); else ATTNQ8K(DHX, 1, 32); } while (0)
  switch (cfg.dh) {
    case 56: ATTNQ8(56); break;
    case 72: ATTNQ8(72); break;
    case 96: ATTNQ8(96); break;
    default: printf("no int8 attention kernel for head dim %d\n", cfg.dh); exit(1);
  }
#undef ATTNQ8
#undef ATTNQ8K
  }
}

// --win8 attention: k_attn2w8 on k_attn2's CTA shapes (runAttnG's sm_86 / sm_89 choice for the
// windowed blocks: 1 warp x BK 16 for <= 16 queries, 2 warps x BK 32 for 64 keys (4 on sm_120),
// else 4 warps x BK 32), the same accumulation modes, and the same window chunking.  A window of
// one key tile (the 16-key ones) gets a single fp16 buffer.  The query-pooling blocks
// (--win8-pool) pool the int8 Q inside the kernel (QP = 2) unless a pooled fp16 Q is passed (QP = 1).
// The 4-warp shapes with head dim 72 / 96 stage the look-ahead tile through cp.async into shared
// memory (CPA) under a register cap of 4 / 3 CTAs per SM: register staging would cost them a CTA
// per SM against k_attn2 (137 vs 126 and 176 vs 157 registers), and the cap without CPA spills
// (win8/logs/explore_v8, explore_v6: hiera-L 256-key windows 5.88 -> 5.62 ms, hiera-T 196-key
// 0.43 -> 0.39 ms per forward); the 1- and 2-warp shapes are faster with register staging.
static constexpr bool w8_cpa(int dh, int nw) { return nw == 4 && (dh == 72 || dh == 96); }
// (MINB = 0 = no launch-bound minimum: with an explicit minimum of 1 ptxas lets the uncapped shapes grow,
// e.g. DH 56 / 4 warps from 96 to 135 registers, 5 -> 3 CTAs per SM)
static constexpr int w8_minb(int dh, int nw) { return !w8_cpa(dh, nw) ? 0 : dh == 96 ? 3 : 4; }
static void runAttnW8(int nWtot, const half* qpool0, half* xout0, int ws2, int ws2q, int heads, int Cp, int Co, cudaStream_t s, int wsx = 0) {
  const int acc = gAttnAcc >= 0 ? gAttnAcc : 1;
  const int maxW = 65535 / heads;
  for (int w0 = 0; w0 < nWtot; w0 += maxW) {
  const int B = std::min(maxW, nWtot - w0) * heads;
  const size_t r0 = (size_t)w0 * ws2;
  const int8_t* q8 = W8Q + r0 * 3 * Cp;
  const half* sc = W8S + r0;
  const half* qpool = qpool0 ? qpool0 + (size_t)w0 * heads * ws2q * cfg.dhp : nullptr;
  half* xout = xout0 + (size_t)w0 * ws2q * Co;
#define W8KC(DHX, NW, BK, NB, QF, PF, MB, CP) do { if (qpool) \
  k_attn2w8<DHX, NW, BK, QF, PF, NB, 1, MB, CP><<<dim3(cdiv(ws2q, NW * 16), B), NW * 32, attn2w8_smem<DHX, BK, NB, CP>(), s>>>(q8, sc, xout, ws2, heads, Cp, Co, ws2q, ws2, gW8Sst, qpool, ws2q); \
  else if (ws2q != ws2) \
  k_attn2w8<DHX, NW, BK, QF, PF, NB, 2, MB, CP><<<dim3(cdiv(ws2q, NW * 16), B), NW * 32, attn2w8_smem_qp<DHX, BK, NB, CP, NW>(), s>>>(q8, sc, xout, ws2, heads, Cp, Co, ws2q, ws2, gW8Sst, nullptr, ws2q, wsx); \
  else \
  k_attn2w8<DHX, NW, BK, QF, PF, NB, 0, MB, CP><<<dim3(cdiv(ws2, NW * 16), B), NW * 32, attn2w8_smem<DHX, BK, NB, CP>(), s>>>(q8, sc, xout, ws2, heads, Cp, Co, ws2, ws2, gW8Sst); } while (0)
#define W8K(DHX, NW, BK, NB, QF, PF) W8KC(DHX, NW, BK, NB, QF, PF, w8_minb(DHX, NW), w8_cpa(DHX, NW))
#define W8CFG(DHX, NW, BK, NB) do { if (acc == 2) W8K(DHX, NW, BK, NB, true, true); else if (acc == 1) W8K(DHX, NW, BK, NB, false, true); else W8K(DHX, NW, BK, NB, false, false); } while (0)
#define W8ANY(DHX) do { if (ws2q <= 16) { if (ws2 <= 16) W8CFG(DHX, 1, 16, 1); else W8CFG(DHX, 1, 16, 2); } \
  else if (ws2 == 64) { if (gArch120) W8CFG(DHX, 4, 32, 2); else W8CFG(DHX, 2, 32, 2); } \
  else W8CFG(DHX, 4, 32, 2); } while (0)
  switch (cfg.dh) {
    case 56: W8ANY(56); break;
    case 72: W8ANY(72); break;
    case 96: W8ANY(96); break;
    default: printf("no win8 attention kernel for head dim %d\n", cfg.dh); exit(1);
  }
#undef W8ANY
#undef W8CFG
#undef W8K
#undef W8KC
  }
}

#ifdef W8DEBUG
// debug build only (-DW8DEBUG): per-block error of the --win8 codes and attention output against the fp16 path
__device__ double g_w8acc[16];
__device__ __forceinline__ void w8_red(double v, int k) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
  if ((threadIdx.x & 31) == 0) atomicAdd(&g_w8acc[k], v);
}
// grid-stride; acc[2p] = sum ref^2, acc[2p+1] = sum err^2 of part p (Q, K, V), real channels only
__global__ void k_w8_cmp_qkv(const half* __restrict__ ref, const int8_t* __restrict__ q8, const half* __restrict__ sc,
                             int rows, int N, int Cp, int dh, int dhp, int sst) {
  double acc[6] = {0, 0, 0, 0, 0, 0};
  const size_t tot = (size_t)rows * N;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x) {
    const int col = (int)(i % N); const size_t r = i / N;
    if (col % dhp >= dh) continue;
    const float a = __half2float(ref[i]);
    const float s = __half2float(sc[(size_t)(col / 16) * sst + r]);
    const float d = __half2float(__float2half_rn((float)q8[i] * s));
    const int part = col / Cp;
    acc[part * 2] += (double)a * a; acc[part * 2 + 1] += (double)(d - a) * (d - a);
  }
  for (int k = 0; k < 6; ++k) w8_red(acc[k], k);
}
__global__ void k_w8_cmp_out(const half* __restrict__ ref, const half* __restrict__ x, size_t n) {
  double acc[4] = {0, 0, 0, 0};
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    const float a = __half2float(ref[i]), b = __half2float(x[i]);
    acc[0] += (double)a * a; acc[1] += (double)(b - a) * (b - a); acc[2] += (double)a * b; acc[3] += (double)b * b;
  }
  for (int k = 0; k < 4; ++k) w8_red(acc[k], 6 + k);
}
// overwrite the fp16 qkv columns of the parts in `mask` (1 = Q, 2 = K, 4 = V) by the dequantised int8 codes
__global__ void k_w8_deq(half* __restrict__ qkv, const int8_t* __restrict__ q8, const half* __restrict__ sc,
                         int rows, int N, int Cp, int sst, int mask) {
  const size_t tot = (size_t)rows * N;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < tot; i += (size_t)gridDim.x * blockDim.x) {
    const int col = (int)(i % N); const size_t r = i / N;
    if (!((mask >> (col / Cp)) & 1)) continue;
    qkv[i] = __float2half_rn((float)q8[i] * __half2float(sc[(size_t)(col / 16) * sst + r]));
  }
}
static half *gW8Scr = nullptr, *gW8Scr2 = nullptr, *gW8Qb = nullptr, *gW8Qb2 = nullptr;
static bool gW8Dbg = true;
static void w8cmp(const half* ref, const half* x, size_t n, cudaStream_t s, double* z) {
  double zz[16] = {0};
  CK(cudaMemcpyToSymbol(g_w8acc, zz, sizeof zz));
  k_w8_cmp_out<<<2048, 256, 0, s>>>(ref, x, n);
  CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
  CK(cudaMemcpyFromSymbol(z, g_w8acc, sizeof zz));
}
static bool w8same(const half* a, const half* b, size_t n) {
  std::vector<uint16_t> ha(n), hb(n);
  CK(cudaMemcpy(ha.data(), a, n * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb.data(), b, n * 2, cudaMemcpyDeviceToHost));
  return memcmp(ha.data(), hb.data(), n * 2) == 0;
}
#endif

// ---------------------------------------------------------------- profiling
enum PTag { P_IM2COL, P_PATCH, P_LN, P_QKV, P_SPROJ, P_POOL, P_ATTN, P_GATHER, P_PROJ, P_FC1, P_FC2Q, P_FC2, P_NECK, P_NTAG };
static const char* PNAME[P_NTAG] = {"im2col", "patch_gemm", "LN_inclusive_producer", "qkv_gemm", "sproj_gemm", "maxpool", "attention", "gather", "proj_gemm", "fc1_inclusive_GELU", "fc2_transform_quant", "fc2_gemm", "neck"};
static std::vector<cudaEvent_t> gEv; static std::vector<int> gTag; static int gEvN = 0;
static std::vector<int> gStg; static int gProfStage = -1;     // w4a4_hiera: the stage of each interval (-1 patch, 4 neck)
static void mark(int tag, cudaStream_t s) {
  if (!gProf) return;
  if (gEvN >= (int)gEv.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); gEv.push_back(e); }
  CK(cudaEventRecord(gEv[gEvN], s));
  if ((int)gTag.size() <= gEvN) gTag.push_back(tag); else gTag[gEvN] = tag;
  if ((int)gStg.size() <= gEvN) gStg.push_back(gProfStage); else gStg[gEvN] = gProfStage;
  ++gEvN;
}
static void profReport() {
  double acc[P_NTAG] = {0}; double tot = 0;
  std::vector<double> blk(cfg.nblk, 0.0);
  for (int i = 0; i + 1 < gEvN; ++i) {
    float ms; CK(cudaEventElapsedTime(&ms, gEv[i], gEv[i + 1]));
    if (gTag[i] >= 0 && gTag[i] < P_NTAG) acc[gTag[i]] += ms;
    if (gTag[i] >= 1000) { acc[P_ATTN] += ms; blk[gTag[i] - 1000] += ms; }   // attention of block tag - 1000
    tot += ms;
  }
  printf("---- per-category GPU time (single un-captured pass) ----\n");
  for (int t = 0; t < P_NTAG; ++t) if (acc[t] > 0) printf("  %-12s %8.3f ms  %5.1f%%\n", PNAME[t], acc[t], 100 * acc[t] / tot);
  printf("  %-12s %8.3f ms\n", "TOTAL", tot);
  for (int i = 0; i < cfg.nblk; ++i)
    printf("  attn b%-2d ws=%-2d%s %8.3f ms\n", i, cfg.b[i].ws, cfg.b[i].qpool ? " qpool" : "      ", blk[i]);
  // w4a4_hiera: the same intervals by stage (column = category, ms), the LayerNorm column is the two LN producers
  double st[6][P_NTAG] = {{0}};
  for (int i = 0; i + 1 < gEvN; ++i) {
    float ms; CK(cudaEventElapsedTime(&ms, gEv[i], gEv[i + 1]));
    const int tg = gTag[i] >= 1000 ? (int)P_ATTN : gTag[i];
    const int sg = gTag[i] >= 1000 ? cfg.b[gTag[i] - 1000].stage : gStg[i];
    if (tg >= 0 && tg < P_NTAG) st[std::min(5, std::max(0, sg + 1))][tg] += ms;
  }
  printf("---- per-stage x per-category GPU time (same pass, ms) ----\n  %-8s", "stage");
  for (int t = 0; t < P_NTAG; ++t) printf(" %11s", PNAME[t]);
  printf(" %9s\n", "sum");
  const char* SN[6] = {"patch", "s0", "s1", "s2", "s3", "neck"};
  for (int r = 0; r < 6; ++r) {
    double sum = 0; for (int t = 0; t < P_NTAG; ++t) sum += st[r][t];
    if (sum == 0) continue;
    printf("  %-8s", SN[r]);
    for (int t = 0; t < P_NTAG; ++t) printf(" %11.3f", st[r][t]);
    printf(" %9.3f\n", sum);
  }
}


static bool gCaptureProducers = getenv("NANO_CAPTURE_PRODUCERS") != nullptr;
static void captureProducer(int bi,const char*family,const half*x,const int8_t*q,const float*sc,
 const float*g,const float*be,int K,int rows,int group,const int*map,cudaStream_t stream){
 if(!gCaptureProducers)return;CK(cudaStreamSynchronize(stream));int M=std::min(128,rows),kp=kpad4(K),ng=group?cdiv(K,group):1;
 std::string dir=getenv("NANO_CAPTURE_PRODUCERS");char nm[80];snprintf(nm,sizeof nm,"/b%02d_%s",bi,family);std::string base=dir+nm;
 std::vector<int>mp(M);if(map)CK(cudaMemcpy(mp.data(),map,M*4,cudaMemcpyDeviceToHost));else for(int i=0;i<M;i++)mp[i]=i;
 if(!strcmp(family,"fc2") && bd32cand()>=10){
   // Fused path deliberately has no globalHID. Only Q/sc are valid capture data.
   writeBin((base+"_q.bin").c_str(),q,(size_t)M*kp/2);writeBin((base+"_sc.bin").c_str(),sc,(size_t)M*ng*4);return;
 }
 std::vector<half>raw((size_t)M*K);for(int r=0;r<M;r++)if(mp[r]>=0)CK(cudaMemcpy(raw.data()+(size_t)r*K,x+(size_t)mp[r]*K,K*2,cudaMemcpyDeviceToHost));
 std::ofstream f(base+"_input.bin",std::ios::binary);f.write((char*)raw.data(),raw.size()*2);f.close();
 // Input file is already gathered in actual producer output-row order. Save source map separately.
 std::ofstream mf(base+"_source_map.bin",std::ios::binary);mf.write((char*)mp.data(),M*4);mf.close();
 writeBin((base+"_q.bin").c_str(),q,(size_t)M*kp/2);writeBin((base+"_sc.bin").c_str(),sc,(size_t)M*ng*4);
 if(g){writeBin((base+"_gamma.bin").c_str(),g,K*4);writeBin((base+"_beta.bin").c_str(),be,K*4);}
 printf("CAPTURE_PRODUCER block=%d family=%s K=%d rows=%d sampled=%d group=%d\n",bi,family,K,rows,M,group);
}
// ---------------------------------------------------------------- forward
struct BlkPlan { int qkv, sproj, proj, fc1, fc2; };
static std::vector<BlkPlan> BP;
static int gpPatch, gpC0, gpC1, gpS4, gpS8;
static void buildPlans() {
  gpPatch = makePlan(gB * 65536, cfg.dims[0], 152, CUBLASLT_EPILOGUE_DEFAULT, nullptr, true);
  BP.resize(cfg.nblk);
  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& c = cfg.b[i];
    BP[i].qkv = makePlan(c.Tprb, 3 * c.cp, c.din, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "qkv.b")), false);
    BP[i].sproj = c.din != c.dout ? makePlan(c.Tb, c.dout, c.din, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "sproj.b")), false) : -1;
    BP[i].proj = makePlan(c.Tqb, c.dout, c.dout, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "proj.b")), true);
    BP[i].fc1 = makePlan(c.Tqb, 4 * c.dout, c.dout, gLtGelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "fc1.b")), false);
    BP[i].fc2 = makePlan(c.Tqb, c.dout, 4 * c.dout, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "fc2.b")), true, true);
  }
  gpC0 = makePlan(gB * 1024, 256, cfg.dims[3], CUBLASLT_EPILOGUE_BIAS, WP("neck.c0.b"), false);
  gpC1 = makePlan(gB * 4096, 256, cfg.dims[2], CUBLASLT_EPILOGUE_BIAS, WP("neck.c1.b"), false);
  gpS4 = makePlan(gB * 65536, 32, cfg.dims[0], CUBLASLT_EPILOGUE_BIAS, WP("neck.s4.b"), false);
  gpS8 = makePlan(gB * 16384, 64, cfg.dims[1], CUBLASLT_EPILOGUE_BIAS, WP("neck.s8.b"), false);
}

static void forward(cudaStream_t s) {
  const int TH = 256;
  gProfStage = -1;
  mark(P_IM2COL, s);
  k_im2col_b<<<cdiv(gB * 65536, 8), 256, 0, s>>>(dIn, COL, 256, 1024, gB);
  mark(P_PATCH, s);
  runPlan(gpPatch, COL, WP("patch.w"), POS, Xs[0], s);

  for (int i = 0; i < cfg.nblk; ++i) {
    const Blk& c = cfg.b[i];
    gProfStage = c.stage;
    half* X = Xs[c.qpool ? c.stage - 1 : c.stage];
    half* Y = Xs[c.stage];
    const float* g1 = (const float*)WP(bk(i, "ln1.w")); const float* b1 = (const float*)WP(bk(i, "ln1.b"));
    const float* g2 = (const float*)WP(bk(i, "ln2.w")); const float* b2 = (const float*)WP(bk(i, "ln2.b"));
    const int* map = c.ws ? mSlot2Nat[i] : nullptr;
    const int Mq = c.Tprb;                           // qkv rows: the batch's padded window slots (or B*T)
    // --fp16-blocks N keeps blocks 0..N-1 on the fp16 path; N = 0 (default) is every block int8.
    const bool i8 = gInt8 && i >= gFp16Blocks;
    const bool i4 = gInt4 && i8;                     // --int4: qkv / fc1 / fc2 on int4 (the rest is the INT8 path)
    const bool mon = i4 && !gMonFile.empty();        // --mon: T = B1 Pi B2 in the LayerNorm producers
    // block-0 operands for the correctness checks (NANO_DUMP_BLOCK=i: block i instead, files keep the _b0 names; aid only)
    static const int dumpBlk = getenv("NANO_DUMP_BLOCK") ? atoi(getenv("NANO_DUMP_BLOCK")) : 0;
    const bool dmp = !gDump.empty() && i == dumpBlk;
    if (dmp && gInt4) {
      writeBin((gDump + "/x_in_b0.bin").c_str(), X, (size_t)c.Tb * c.din * 2);
      if (map) writeBin((gDump + "/map_b0.bin").c_str(), map, (size_t)Mq * 4);
      writeBin((gDump + "/w_qkv_b0.bin").c_str(), QWqkv[i].w, (size_t)3 * c.cp * kpad4(c.din) / 2);
      writeBin((gDump + "/s_qkv_b0.bin").c_str(), QWqkv[i].s, (size_t)3 * c.cp * 4);
      writeBin((gDump + "/w_fc1_b0.bin").c_str(), QWfc1[i].w, (size_t)4 * c.dout * kpad4(c.dout) / 2);
      writeBin((gDump + "/s_fc1_b0.bin").c_str(), QWfc1[i].s, (size_t)4 * c.dout * 4);
      writeBin((gDump + "/w_fc2_b0.bin").c_str(), QWfc2[i].w, (size_t)c.dout * kpad4(4 * c.dout) / 2);
      writeBin((gDump + "/s_fc2_b0.bin").c_str(), QWfc2[i].s, (size_t)c.dout * 4);
    }
    mark(P_LN, s);
    if(i4 && !gDhdFile.empty()) launchDhdLN(X,XQ,XS,Mq,gDhd1[i],s,map);
    else if (i4 && !gKronFile.empty()) launchKron(X, XQ, XS, g1, b1, Mq, gKron1[i], s, map);
    else if (mon)     launchLNMon4(X, XQ, XS, g1, b1, Mq, c.din, gMon1[i], s, map);
    else if (i4) launchLNQ4(X, XQ, XS, g1, b1, Mq, c.din, s, map);
    else if (i8) launchLNQ8(X, XQ, XS, g1, b1, Mq, c.din, s, map);
    else       launchLN(X, Xn, g1, b1, Mq, c.din, s, map);
    if (dmp && gInt4) {
      writeBin((gDump + "/xq_ln1_b0.bin").c_str(), XQ, (size_t)Mq * kpad4(c.din) / 2);
      writeBin((gDump + "/xs_ln1_b0.bin").c_str(), XS, (size_t)Mq * 4);
    }
    if(i4&&!gDhdFile.empty())captureDhd(i,"qkv",X,XQ,XS,Mq,gDhd1[i],map,s);
    if(i4&&!gKronFile.empty())captureProducer(i,"qkv",X,XQ,XS,g1,b1,c.din,Mq,0,map,s);
    mark(P_QKV, s);
    const bool fusedQK = i8 && qkvFused(c) && !gQK8Pass;
    const bool w8 = i8 && win8Block(c);              // --win8: int8 Q/K/V + fp16 scales, no fp16 qkv
    if (w8) gW8Sst = c.Tprb + 128;                   // this block's scale stride (rows + key-tile slack)
    const half* qb = (const half*)WP(bk(i, "qkv.b"));
    if (i4 && w8) runQkv4<1>(i4env("NANO_I4_QKV", c.stage, i4QkvDefaultW8(c.din)), XQ, QWqkv[i], XS, qb, nullptr, Mq, 3 * c.cp, c.din, s,
                             W8Q, (float*)W8S, gW8Sst);
    else if (i4 && fusedQK) runQkv84(XQ, QWqkv[i], XS, qb, QKV, Mq, 3 * c.cp, c.din, 2 * c.cp, s);
    else if (i4) runQkv4<0>(i4env("NANO_I4_QKV", c.stage, i4QkvDefault(c.din)), XQ, QWqkv[i], XS, qb, QKV, Mq, 3 * c.cp, c.din, s);
    else if (w8) runQkvW8(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), Mq, 3 * c.cp, c.din, s);
    else if (fusedQK) runQkv8(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, Mq, 3 * c.cp, c.din, 2 * c.cp, s);
    else if (i8) runS8<false>(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, Mq, 3 * c.cp, c.din, s);
    else       runPlan(BP[i].qkv, Xn, WP(bk(i, "qkv.w")), nullptr, QKV, s);
    if (dmp && gInt4 && !w8) writeBin((gDump + "/qkv_b0.bin").c_str(), QKV, (size_t)Mq * 3 * c.cp * 2);
    if (qk8Block(c) && !fusedQK)                     // stand-alone Q/K quantisation (fp16 path, head dim 72, --qk8-pass)
      k_quant_qk<<<cdiv(Mq * 2 * c.heads * 32, TH), TH, 0, s>>>(QKV, QK8, QKS, Mq, c.heads, c.cp, cfg.dh, cfg.dhp,
                                                                attn2q8_dh8(cfg.dh), gQKSst);

    half* shortcut = X;
    if (c.din != c.dout) {                           // sproj on the natural-order LayerNorm output, then 2x2 max-pool
      if (i8 || map) launchLN(X, Xn, g1, b1, c.Tb, c.din, s);
      mark(P_SPROJ, s);
      runPlan(BP[i].sproj, Xn, WP(bk(i, "sproj.w")), nullptr, HID, s);
      mark(P_POOL, s);
      k_maxpool_nat<<<cdiv(c.Tqb * c.dout / 2, TH), TH, 0, s>>>(HID, SC, c.H, c.dout, gB);
      shortcut = SC;
    }
    const int ws2 = c.ws ? c.ws * c.ws : c.T, ws2q = c.qpool ? ws2 / 4 : ws2;
    const half* qpool = nullptr;
    if (c.qpool && !(w8 && !gW8PoolPass)) {          // --win8-pool: the attention pools Q itself (QP = 2)
      mark(P_POOL, s);
      if (w8) k_pool_q_w8<<<cdiv(c.nWb * c.heads * ws2q * (cfg.dhp / 2), TH), TH, 0, s>>>(W8Q, W8S, Qb, c.nWb, c.ws, c.heads, c.cp, cfg.dhp, gW8Sst);
      else    k_pool_q_g<<<cdiv(c.nWb * c.heads * ws2q * (cfg.dhp / 2), TH), TH, 0, s>>>(QKV, Qb, c.nWb, c.ws, c.heads, c.cp, cfg.dhp);
      qpool = Qb;
    }
    mark(1000 + i, s);                               // P_ATTN, split by block in profReport
    half* aout = c.ws ? AO : Xn;                     // windowed: window-major slots; global: natural order directly
#ifdef W8DEBUG
    if (w8 && gW8Dbg) {
      double z[16] = {0}, a[16], v[16], q[16];
      CK(cudaMemcpyToSymbol(g_w8acc, z, sizeof z));
      const half* qb = (const half*)WP(bk(i, "qkv.b"));
      runS8<false>(XQ, QWqkv[i], XS, qb, QKV, Mq, 3 * c.cp, c.din, s);
      k_w8_cmp_qkv<<<2048, 256, 0, s>>>(QKV, W8Q, W8S, c.Tpb, 3 * c.cp, c.cp, cfg.dh, cfg.dhp, gW8Sst);
      CK(cudaStreamSynchronize(s)); CK(cudaMemcpyFromSymbol(z, g_w8acc, sizeof z));
      if (!gW8Scr) {
        size_t mx = 0, mq = 1;
        for (const Blk& b : cfg.b) { mx = std::max(mx, (size_t)std::max(b.Tpob, b.Tqb) * b.dout); if (b.qpool) mq = std::max(mq, (size_t)b.nWb * b.heads * (b.wo * b.wo) * cfg.dhp); }
        gW8Scr = dAlloc(mx); gW8Scr2 = dAlloc(mx); gW8Qb = dAlloc(mq + 64 * cfg.dhp); gW8Qb2 = dAlloc(mq + 64 * cfg.dhp);
      }
      const size_t n = (size_t)c.Tpob * c.dout, nq = (size_t)c.nWb * c.heads * ws2q * cfg.dhp;
      const int pg = cdiv(c.nWb * c.heads * ws2q * (cfg.dhp / 2), TH);
      if (c.qpool) k_pool_q_g<<<pg, TH, 0, s>>>(QKV, gW8Qb, c.nWb, c.ws, c.heads, c.cp, cfg.dhp);        // exact pooled Q
      runAttnG(c.nWb, QKV, c.qpool ? gW8Qb : nullptr, gW8Scr, ws2, ws2q, c.heads, c.cp, c.dout, s);      // exact fp16 path
      runAttnW8(c.nWb, qpool, aout, ws2, ws2q, c.heads, c.cp, c.dout, s, c.ws);                              // the int8 kernel
      w8cmp(gW8Scr, aout, n, s, a);
      k_w8_deq<<<2048, 256, 0, s>>>(QKV, W8Q, W8S, Mq, 3 * c.cp, c.cp, gW8Sst, 4);                     // V only
      runAttnG(c.nWb, QKV, c.qpool ? gW8Qb : nullptr, gW8Scr2, ws2, ws2q, c.heads, c.cp, c.dout, s);
      w8cmp(gW8Scr, gW8Scr2, n, s, v);
      runS8<false>(XQ, QWqkv[i], XS, qb, QKV, Mq, 3 * c.cp, c.din, s);
      k_w8_deq<<<2048, 256, 0, s>>>(QKV, W8Q, W8S, Mq, 3 * c.cp, c.cp, gW8Sst, 3);                     // Q and K only
      if (c.qpool) k_pool_q_g<<<pg, TH, 0, s>>>(QKV, gW8Qb2, c.nWb, c.ws, c.heads, c.cp, cfg.dhp);       // pooled from the codes
      runAttnG(c.nWb, QKV, c.qpool ? gW8Qb2 : nullptr, gW8Scr2, ws2, ws2q, c.heads, c.cp, c.dout, s);
      w8cmp(gW8Scr, gW8Scr2, n, s, q);
      k_w8_deq<<<2048, 256, 0, s>>>(QKV, W8Q, W8S, Mq, 3 * c.cp, c.cp, gW8Sst, 4);                     // all three: k_attn2 on the
      runAttnG(c.nWb, QKV, c.qpool ? gW8Qb2 : nullptr, gW8Scr2, ws2, ws2q, c.heads, c.cp, c.dout, s);    // dequantised fp16 == k_attn2w8?
      CK(cudaStreamSynchronize(s));
      const bool same = w8same(gW8Scr2, aout, n), sameq = !qpool || w8same(gW8Qb2, qpool, nq);
      printf("W8DBG b%-2d ws=%-2d%s heads=%-2d  rel-rms err  Q %.5f  K %.5f  V %.5f | attn out: all %.5f (cos %.7f)  V-only %.5f  QK-only %.5f | kernel == k_attn2(dequant): %s%s\n",
             i, c.ws, c.qpool ? " qpool" : "", c.heads, sqrt(z[1] / z[0]), sqrt(z[3] / z[2]), sqrt(z[5] / z[4]), sqrt(a[7] / a[6]), a[8] / sqrt(a[6] * a[9]),
             sqrt(v[7] / v[6]), sqrt(q[7] / q[6]), same ? "IDENTICAL" : "DIFFER", qpool ? (sameq ? ", pooled Q IDENTICAL" : ", pooled Q DIFFER") : (c.qpool ? ", fused pooling" : ""));
    } else
#endif
    if (w8)               runAttnW8(c.nWb, qpool, aout, ws2, ws2q, c.heads, c.cp, c.dout, s, c.ws);
    else if (qk8Block(c)) runAttnQ8(c.nWb, QKV, aout, ws2, c.heads, c.cp, c.dout, s);
    else                  runAttnG(c.nWb, QKV, qpool, aout, ws2, ws2q, c.heads, c.cp, c.dout, s);
    if (c.ws) {
      mark(P_GATHER, s);
      k_gather<<<cdiv(c.Tqb * (c.dout / 8), TH), TH, 0, s>>>(AO, Xn, mNat2Slot[i], c.Tqb, c.dout);
    }
    mark(P_PROJ, s);
    if (useResF16(c.Tqb)) runResF16(Xn, (const half*)WP(bk(i, "proj.w")), (const half*)WP(bk(i, "proj.b")), shortcut, Y, c.Tqb, c.dout, c.dout, s);
    else runPlan(BP[i].proj, Xn, WP(bk(i, "proj.w")), shortcut, Y, s);
    mark(P_LN, s);
    if(i4 && !gDhdFile.empty()) launchDhdLN(Y,XQ,XS,c.Tqb,gDhd2[i],s,nullptr);
    else if (i4 && !gKronFile.empty()) launchKron(Y, XQ, XS, g2, b2, c.Tqb, gKron2[i], s);
    else if (mon)     launchLNMon4(Y, XQ, XS, g2, b2, c.Tqb, c.dout, gMon2[i], s);
    else if (i4) launchLNQ4(Y, XQ, XS, g2, b2, c.Tqb, c.dout, s);
    else if (i8) launchLNQ8(Y, XQ, XS, g2, b2, c.Tqb, c.dout, s);
    else       launchLN(Y, Xn, g2, b2, c.Tqb, c.dout, s);
    if (dmp && gInt4) {
      writeBin((gDump + "/x_mid_b0.bin").c_str(), Y, (size_t)c.Tqb * c.dout * 2);
      writeBin((gDump + "/xq_ln2_b0.bin").c_str(), XQ, (size_t)c.Tqb * kpad4(c.dout) / 2);
      writeBin((gDump + "/xs_ln2_b0.bin").c_str(), XS, (size_t)c.Tqb * 4);
    }
    if(i4&&!gDhdFile.empty())captureDhd(i,"fc1",Y,XQ,XS,c.Tqb,gDhd2[i],nullptr,s);
    if(i4&&!gKronFile.empty())captureProducer(i,"fc1",Y,XQ,XS,g2,b2,c.dout,c.Tqb,0,nullptr,s);
    mark(P_FC1, s);
    const half* fb1 = (const half*)WP(bk(i, "fc1.b"));
    const int H4 = 4 * c.dout;
    if(i4 && !gDhdFile.empty()) {
      runDhdFc1(i4env("NANO_I4_FC1",c.stage,i4Fc1DhdDefault(c.Tqb,c.dout,gDhd3[i].diagonals)),XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s,HQ,HS,gDhd3[i],
                gFc2StaticMode?gF2S[i].sinv:nullptr);
    } else if (i4 && !gKronFile.empty()) {
      if(gFc2FixedBits==8){constexpr size_t sh=s8_smem<128,128,64,3>();auto k=k_gemm_s8<128,128,64,2,2,3,true,true,false,false,0,true,1,false,false,64,1,false,false,false,true,false,true>;static bool attr=false;if(!attr){CK(cudaFuncSetAttribute(k,cudaFuncAttributeMaxDynamicSharedMemorySize,sh));attr=true;}k<<<dim3(cdiv(H4,128),c.Tqb/128),128,sh,s>>>(XQ,QWfc1[i].w,XS,QWfc1[i].s,fb1,HID,c.Tqb,H4,kpad4(c.dout)/2,nullptr,HQ,HS,0,nullptr,0,nullptr,1.f);}
      else if(gFc2FixedBits==4)runFc14(i4env("NANO_I4_FC1",c.stage,i4Fc1Default()),XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s,HQ,HS);
      else if(bd32cand()>=10 && gKron3[i].b==32)runBD32Choice(bd32cand(),XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s,HQ,HS,gKron3[i]);
      else if(bd32cand()==1)runS4<0,true,false,0,false,0,128,128,2,2,3,true,1>(XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s);
      else if(bd32cand()==2)runS4<0,true,false,0,false,0,64,128,2,2,3>(XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s);
      else if(bd32cand()==3)runS4<0,true,false,0,false,0,32,128,2,2,3>(XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s);
      else if(bd32cand()==4)runS4<0,true,false,0,false,0,128,128,2,4,3>(XQ,QWfc1[i],XS,fb1,HID,c.Tqb,H4,c.dout,s);
      else runS4<0, true, false, 0, false, 0, 128, 128, 2, 2, 3>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s);
      mark(P_FC2Q, s);
      if(!gFc2FixedBits && !(bd32cand()>=10 && gKron3[i].b==32))launchKron(HID, HQ, HS, nullptr, nullptr, c.Tqb, gKron3[i], s, nullptr, nullptr, 128);
    } else if (i4) {
      runFc14(i4env("NANO_I4_FC1", c.stage, i4Fc1Default()), XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, HQ, HS);
    } else if (i8 && gInt8Fc2 && gFc2Pass) {
      // --fc2-pass: the fused kernel's tile with the plain fp16 GELU epilogue, then the pass
      if (gArch89 && gFc2Rot == 64) runS8<true, false, false, 0, false, 1, 2, 2, false, false, 0, true, 1>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s);
      else                          runS8<true>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s);
      mark(P_FC2Q, s);
      const dim3 grid(cdiv(H4, 128), c.Tqb / 128);
      if (gFc2Rot == 128) k_rotquant_fc2<128><<<grid, 256, 0, s>>>(HID, HQ, HS, H4, cdiv(H4, 128));
      else                k_rotquant_fc2<64><<<grid, 256, 0, s>>>(HID, HQ, HS, H4, cdiv(H4, 128));
    } else if (i8 && gInt8Fc2) {
      if (gRot && gFc2Rot == 128) runS8<true, false, false, 0, true, 1, 2, 4, false, false, 128>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);
      else if (gRot && gArch120)  runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);   // sm_120 placeholder
      else if (gRot && gArch89)   runS8<true, false, false, 0, true, 1, 2, 2, false, false, 64, true, 1>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);
      else if (gRot)              runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);
      else if (gFc2U8)            runS8<true, false, false, 0, true, 1, 2, 4, false, true>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);
      else                        runS8<true, false, false, 0, true, 1>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s, nullptr, HQ, HS);
    } else if (i8) {
      runS8<true>(XQ, QWfc1[i], XS, fb1, HID, c.Tqb, H4, c.dout, s);
    } else if (!gLtGelu) {
      runFc1Gelu(Xn, (const half*)WP(bk(i, "fc1.w")), fb1, HID, c.Tqb, H4, c.dout, gAccMode >= 1, s);
    } else {
      runPlan(BP[i].fc1, Xn, WP(bk(i, "fc1.w")), nullptr, HID, s);
    }
    if (dmp && gInt4) {
      writeBin((gDump + "/hq_b0.bin").c_str(), HQ, (size_t)c.Tqb * H4 / 2);
      writeBin((gDump + "/hs_b0.bin").c_str(), HS, (size_t)c.Tqb * cdiv(H4, 128) * 4);
    }
    if(i4&&!gDhdFile.empty())captureDhd(i,"fc2",HID,HQ,HS,c.Tqb,gDhd3[i],nullptr,s);
    if(i4&&!gKronFile.empty()&&!gFc2FixedBits)captureProducer(i,"fc2",HID,HQ,HS,nullptr,nullptr,H4,c.Tqb,128,nullptr,s);
    mark(P_FC2, s);
    if(i4&&gFc2FixedBits==8)runFc2S8<false>(HQ,MixedFc2[i],HS,(const half*)WP(bk(i,"fc2.b")),Y,c.Tqb,c.dout,H4,s,cdiv(H4,128));
    else if (i4 && gFc2StaticMode) {                 // --fc2-static: no group flush, the static scalar (s, or 1 with folded weights)
      runFc24S(i4env("NANO_I4_FC2", c.stage, i4Fc2StaticDefault(c.dout, c.Tqb)), HQ, QWfc2[i], gF2S[i].sgemm, (const half*)WP(bk(i, "fc2.b")), Y,
               c.Tqb, c.dout, H4, s);
    } else if (i4 && gTNoFlush) {                    // TIMING ONLY: dynamic codes, the no-flush GEMM at the flush path's tile
      runFc24S(i4env("NANO_I4_FC2", c.stage, i4Fc2DefaultM(c.dout, c.Tqb)), HQ, QWfc2[i], HS, (const half*)WP(bk(i, "fc2.b")), Y,
               c.Tqb, c.dout, H4, s);
    } else if (i4) {
      runFc24(i4env("NANO_I4_FC2", c.stage, i4Fc2DefaultM(c.dout, c.Tqb)), HQ, gFc2FixedBits==4?MixedFc2[i]:QWfc2[i], HS, (const half*)WP(bk(i, "fc2.b")), Y, c.Tqb,
              c.dout, H4, s, cdiv(H4, 128));
    } else if (i8 && gInt8Fc2) {
      const half* fb = gFc2U8 ? Bfc2[i] : (const half*)WP(bk(i, "fc2.b"));
      const int sastride = cdiv(H4, 128);
      if (gFc2U8) runFc2S8<true>(HQ, QWfc2[i], HS, fb, Y, c.Tqb, c.dout, H4, s, sastride);
      else        runFc2S8<false>(HQ, QWfc2[i], HS, fb, Y, c.Tqb, c.dout, H4, s, sastride);
    } else if (useResF16(c.Tqb)) {
      runResF16(HID, (const half*)WP(bk(i, "fc2.w")), (const half*)WP(bk(i, "fc2.b")), Y, Y, c.Tqb, c.dout, H4, s);
    } else {
      runPlan(BP[i].fc2, HID, WP(bk(i, "fc2.w")), Y, Y, s);
    }
    if (!gDump.empty() && (i == 0 || i == cfg.stage_ends[0] || i == cfg.stage_ends[1] || i == cfg.stage_ends[2] || i == cfg.stage_ends[3])) {
      char nm[64]; snprintf(nm, sizeof nm, "/blk%d.bin", i);
      writeBin((gDump + nm).c_str(), Y, (size_t)c.Tqb * c.dout * 2);
    }
  }
  // ---- neck (natural order: identity maps) ----
  gProfStage = 4;
  mark(P_NECK, s);
  runPlan(gpC0, Xs[3], WP("neck.c0.w"), nullptr, L3, s);
  runPlan(gpC1, Xs[2], WP("neck.c1.w"), nullptr, L2, s);
  k_vision_feat_b<<<cdiv(gB * 4096 * 256, TH), TH, 0, s>>>(L2, L3, dVF, 64, 256, gB);
  runPlan(gpS4, Xs[0], WP("neck.s4.w"), nullptr, S4b, s);
  k_scatter_nchw_b<<<cdiv(gB * 65536 * 32, TH), TH, 0, s>>>(S4b, dS4, 65536, 32, gB);
  runPlan(gpS8, Xs[1], WP("neck.s8.w"), nullptr, S8b, s);
  k_scatter_nchw_b<<<cdiv(gB * 16384 * 64, TH), TH, 0, s>>>(S8b, dS8, 16384, 64, gB);
  if (!gDump.empty()) { writeBin((gDump + "/first_vf.bin").c_str(), dVF, (size_t)gB*256*4096*4); writeBin((gDump + "/first_s4.bin").c_str(), dS4, (size_t)gB*32*65536*4); }
  mark(-1, s);
}

// ---------------------------------------------------------------- driver
int main(int argc, char** argv) {
  std::string model = "hiera/large"; const char* input = nullptr; const char* outDir = nullptr;
  int warmup = 30, iters = 200, useGraph = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--input" && i + 1 < argc) input = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) outDir = argv[++i];
    else if (a == "--dump" && i + 1 < argc) gDump = argv[++i];
    else if (a == "--batch" && i + 1 < argc) gB = atoi(argv[++i]);
    else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
    else if (a == "--no-graph") useGraph = 0;
    else if (a == "--f16acc") gAccMode = 2;
    else if (a == "--f16acc-shallow") gAccMode = 1;
    else if (a == "--attn-acc" && i + 1 < argc) gAttnAcc = atoi(argv[++i]);
    else if (a == "--int8") gInt8 = true;
    else if (a == "--int8-fc2") { gInt8 = true; gInt8Fc2 = true; }
    else if (a == "--int4") { gInt4 = true; gInt8 = true; gInt8Fc2 = true; gRot = true; }
    else if(a=="--fc2-fixed-bits"&&i+1<argc)gFc2FixedBits=atoi(argv[++i]);
    else if(a=="--fc2-weights"&&i+1<argc)gFc2External=argv[++i];
    else if(a=="--dhd" && i+1<argc)gDhdFile=argv[++i];
    else if(a=="--dhd-identity")gDhdIdentity=true;
    else if (a == "--fc2-static" && i + 1 < argc) gFc2StaticFile = argv[++i];
    else if (a == "--kron" && i + 1 < argc) gKronFile = argv[++i];
    else if (a == "--mon" && i + 1 < argc) gMonFile = argv[++i];
    else if (a == "--fc2-s8") gFc2U8 = false;
    else if (a == "--rot") { gInt8 = true; gRot = true; }
    else if (a == "--fc2-rot" && i + 1 < argc) gFc2Rot = atoi(argv[++i]);
    else if (a == "--gptq" && i + 1 < argc) { gInt8 = true; gRot = true; gGptq = argv[++i]; }
    else if (a == "--fp16-blocks" && i + 1 < argc) gFp16Blocks = atoi(argv[++i]);
    else if (a == "--lt-gelu") gLtGelu = true;
    else if (a == "--qk8") gQK8 = true;
    else if (a == "--qk8-pass") { gQK8 = true; gQK8Pass = true; }
    else if (a == "--fc2-pass") gFc2Pass = true;
    else if (a == "--qk8-win") { gQK8 = true; gQK8Win = true; }
    else if (a == "--win8") gWin8 = true;
    else if (a == "--win8-glob") { gWin8 = true; gWin8Glob = true; }
    else if (a == "--win8-pool") { gWin8 = true; gWin8Pool = true; }
    else if (a == "--profile") gProf = true;
    else if (a == "--no-tune") gTune = false;
    else if (a == "--serve") gServe = true;
    else { printf("unknown arg %s\n", argv[i]); return 1; }
  }
  if (gB < 1) { printf("--batch must be >= 1\n"); return 1; }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  gArch80 = prop.major >= 8; gArch89 = prop.major == 8 && prop.minor == 9;
  if (const char* e = getenv("NANO_ARCH89")) { gArch89 = atoi(e) != 0; printf("NANO_ARCH89 override: gArch89=%d%c", (int)gArch89, 10); }
  gArch120 = prop.major == 12;
  if (const char* e = getenv("NANO_ARCH120")) { gArch120 = atoi(e) != 0; printf("NANO_ARCH120 override: gArch120=%d%c", (int)gArch120, 10); }
  gArch86 = prop.major == 8 && prop.minor == 6;
  if (const char* e = getenv("NANO_W8_POOLPASS")) { gW8PoolPass = atoi(e) != 0; printf("NANO_W8_POOLPASS=%d%c", (int)gW8PoolPass, 10); }
  if (const char* e = getenv("NANO_ARCH86")) { gArch86 = atoi(e) != 0; printf("NANO_ARCH86 override: gArch86=%d%c", (int)gArch86, 10); }
  if (!gArch80) { printf("nano_hiera needs sm_80+\n"); return 1; }
  gSMs = prop.multiProcessorCount;
  if(!gDhdFile.empty() && (!gInt4 || gGptq.empty() || !gMonFile.empty() || !gKronFile.empty() || gFc2FixedBits || gQK8 || gFp16Blocks)) { printf("DHD requires all144 INT4 GEMMs and external weights; incompatible extra quantization flags\n");return 1; }
  if(gDhdIdentity && gDhdFile.empty()){printf("--dhd-identity requires --dhd\n");return 1;}
  if (!gFc2StaticFile.empty() && gDhdFile.empty()) { printf("--fc2-static requires --dhd (the SHAQuant / DHD INT4 path)\n"); return 1; }
  gTNoLnH = getenv("NANO_T_NOLNH") && atoi(getenv("NANO_T_NOLNH"));
  gTNoFc1H = getenv("NANO_T_NOFC1H") && atoi(getenv("NANO_T_NOFC1H"));
  gTNoFlush = getenv("NANO_T_NOFLUSH") && atoi(getenv("NANO_T_NOFLUSH"));
  gTNoFc1S = getenv("NANO_T_NOFC1S") && atoi(getenv("NANO_T_NOFC1S"));
  gTNoD2 = getenv("NANO_T_NOD2") && atoi(getenv("NANO_T_NOD2"));
  if ((gTNoFc1S || gTNoD2) && gFc2StaticFile.empty()) { printf("NANO_T_NOFC1S / NANO_T_NOD2 are switches of the --fc2-static path\n"); return 1; }
  if (gTNoFc1S) printf("TIMING-ONLY SWITCH (outputs are NOT valid): fc1 epilogue without the D1 (S) multiply\n");
  if (gTNoD2) printf("NANO_T_NOD2: no D2 multiplies (LayerNorm producers, fc1 epilogue); exact when every d2 = 1\n");
  gD2Keep = getenv("NANO_D2_KEEP") && atoi(getenv("NANO_D2_KEEP"));
  if (gD2Keep) printf("NANO_D2_KEEP: the D2 multiplies stay on every layer (the fc2s3 kernels)\n");
  if ((gTNoLnH || gTNoFc1H || gTNoFlush) && gDhdFile.empty()) { printf("NANO_T_* timing switches need --dhd\n"); return 1; }
  if (gTNoFlush && !gFc2StaticFile.empty()) { printf("NANO_T_NOFLUSH is the dynamic path's switch (--fc2-static has no flush)\n"); return 1; }
  if (gTNoLnH || gTNoFc1H || gTNoFlush)
    printf("TIMING-ONLY SWITCHES (outputs are NOT valid):%s%s%s\n", gTNoLnH ? " LN producers without the Hadamard," : "",
           gTNoFc1H ? " fc1 epilogue without the 64-point butterfly," : "", gTNoFlush ? " fc2 without the group flush (row scale HS[0])," : "");
  if (!gKronFile.empty() && (!gInt4 || gGptq.empty() || !gMonFile.empty())) { printf("--kron requires --int4 --gptq and no --mon%c", 10); return 1; }
  if (!gMonFile.empty() && !gInt4) { printf("--mon needs --int4\n"); return 1; }
  if (gInt4) {
    if (prop.major != 8) { printf("--int4 needs the sm_80/86/89 INT4 mma (mma.m16n8k64.s4); this is sm_%d%d\n", prop.major, prop.minor); return 1; }
    if (gFc2Pass || gFc2Rot != 64) { printf("--int4 runs the fused H_64 fc1 producer only (no --fc2-pass / --fc2-rot 128)\n"); return 1; }
    printf("int4: W4A4 qkv / fc1 / fc2%s\n", gMonFile.empty() ? "" : (", mon T in the LayerNorm producers: " + gMonFile).c_str());
  }
  if (gRot && gInt8Fc2) {
    gFc2U8 = false;
    if (gFc2Rot != 64 && gFc2Rot != 128) { printf("--fc2-rot must be 64 or 128\n"); return 1; }
  }
  if (gFc2Pass && !(gRot && gInt8Fc2)) { printf("--fc2-pass needs --int8-fc2 with --rot / --gptq\n"); return 1; }
  if (gFc2Pass) printf("fc2-pass: fc1 writes fp16, k_rotquant_fc2<%d> makes the int8 fc2 operand\n", gFc2Rot);
  if (gWin8 && !gInt8) { printf("--win8 needs the int8 path (--int8 / --int8-fc2 / --rot / --gptq)\n"); return 1; }
  if (gWin8) printf("win8: int8 Q/K/V storage (fp16 scale per token and 16 channels) on the windowed blocks%s%s\n",
                    gWin8Pool ? ", the query-pooling blocks" : "", gWin8Glob ? ", the global blocks without --qk8" : "");
  printf("GPU: %s sm_%d%d, SMs=%d, accmode=%d gemms=%s%s%s\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount, gAccMode,
         gInt4 ? "int4(qkv+fc1+fc2)" : gInt8Fc2 ? "int8(qkv+fc1+fc2)" : gInt8 ? "int8(qkv+fc1)" : "fp16", gRot ? " rot" : "", gGptq.empty() ? "" : " gptq");
  loadCfg(model);
  applyBatch();
  if (gFp16Blocks < 0) gFp16Blocks = 0;
  if (gFp16Blocks > cfg.nblk) gFp16Blocks = cfg.nblk;
  if (gFp16Blocks && gInt8) printf("fp16-blocks: blocks 0..%d stay fp16 (%d of %d int8)\n",
                                   gFp16Blocks - 1, cfg.nblk - gFp16Blocks, cfg.nblk);
  loadWeights(model);
  allocAll();
  CL(cublasLtCreate(&LT));
  CK(cudaMalloc(&gWs, gWsBytes));
  buildPlans();
  if (gInt8) buildInt8Weights();
  const size_t imgBytes = 3ull * 1024 * 1024 * sizeof(float);
  if (input) {
    size_t got = readBin(input, dIn, (size_t)gB * imgBytes);
    if (got == imgBytes && gB > 1) {        // one image in the file: fill the batch with it
      for (int b = 1; b < gB; ++b) CK(cudaMemcpy((char*)dIn + b * imgBytes, dIn, imgBytes, cudaMemcpyDeviceToDevice));
      printf("input %s: 1 image replicated over the batch of %d\n", input, gB);
    } else if (got < (size_t)gB * imgBytes) {
      printf("input %s: %zu of %zu bytes (%.2f of %d images); the rest is zero\n",
             input, got, (size_t)gB * imgBytes, (double)got / imgBytes, gB);
    }
  } else CK(cudaMemset(dIn, 0, (size_t)gB * imgBytes));

  cudaStream_t s; CK(cudaStreamCreate(&s));
  forward(s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
#ifdef W8DEBUG
  gW8Dbg = false;
#endif
  gDump.clear();gCaptureProducers=false;
  if (gServe) {
    // --serve (w4a4_hiera accuracy harness): one request per stdin line "<input.bin> <out_dir>", the forward of that
    // input (the same kernels as one launch of the timed graph), the three outputs written as with --out-dir, then
    // "SERVED <n>" on stdout.  An empty line, EOF or "quit" ends the process.
    printf("SERVE_READY\n"); fflush(stdout);
    std::string line; int n = 0;
    while (std::getline(std::cin, line)) {
      std::istringstream ls(line); std::string in, od; ls >> in >> od;
      if (in.empty() || in == "quit" || od.empty()) break;
      readBin(in.c_str(), dIn, (size_t)gB * imgBytes);
      forward(s); CK(cudaStreamSynchronize(s)); CK(cudaGetLastError());
      writeBin((od + "/ours_vf.bin").c_str(), dVF, (size_t)gB * 256 * 64 * 64 * 4);
      writeBin((od + "/ours_s4.bin").c_str(), dS4, (size_t)gB * 32 * 256 * 256 * 4);
      writeBin((od + "/ours_s8.bin").c_str(), dS8, (size_t)gB * 64 * 128 * 128 * 4);
      printf("SERVED %d\n", ++n); fflush(stdout);
    }
    return 0;
  }
  if (gProf) {
    for (int r = 0; r < 3; ++r) { gEvN = 0; forward(s); CK(cudaStreamSynchronize(s)); }
    profReport(); gProf = false;
  }
  cudaGraphExec_t exec = nullptr;
  if (useGraph) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    forward(s);
    CK(cudaStreamEndCapture(s, &g));
    size_t nn=0; CK(cudaGraphGetNodes(g,nullptr,&nn)); printf("graph_nodes=%zu%c",nn,10); std::vector<cudaGraphNode_t> ns(nn); CK(cudaGraphGetNodes(g,ns.data(),&nn));
    CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    CK(cudaGraphDestroy(g));
  }
  auto once = [&] { if (exec) CK(cudaGraphLaunch(exec, s)); else forward(s); };
  for (int i = 0; i < warmup; ++i) once();
  CK(cudaStreamSynchronize(s));
  std::vector<float> ts;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < iters; ++i) {
    CK(cudaEventRecord(e0, s)); once(); CK(cudaEventRecord(e1, s));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  double sum = 0; for (float t : ts) sum += t;
  printf("latency_ms mean=%.4f p50=%.4f min=%.4f iters=%d graph=%d batch=%d per_image=%.4f %.1f_img_s\n",
         sum / ts.size(), ts[ts.size() / 2], ts[0], iters, useGraph, gB,
         ts[ts.size() / 2] / gB, 1000.0 * gB / ts[ts.size() / 2]);
  if (outDir) {
    std::string d = outDir;
    writeBin((d + "/ours_vf.bin").c_str(), dVF, (size_t)gB * 256 * 64 * 64 * 4);
    writeBin((d + "/ours_s4.bin").c_str(), dS4, (size_t)gB * 32 * 256 * 256 * 4);
    writeBin((d + "/ours_s8.bin").c_str(), dS8, (size_t)gB * 64 * 128 * 128 * 4);
    printf("dumped outputs to %s\n", outDir);
  }
  return 0;
}
