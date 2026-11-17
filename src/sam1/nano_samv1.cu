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

// nano_samv1.cu — the original SAM (v1) image encoder (ImageEncoderViT, ViT-B / L / H) on the nano kernels.
//
// Same structure as nano_sam3.cu (a natural-order residual stream; the LayerNorm writes its rows
// straight into window-major slot order through a slot->token map; qkv GEMM + attention on whole
// windows; k_gather back to natural order before the proj GEMM adds the residual), batched like
// nano_vit.cu (B images in one token buffer), plus what SAM v1 needs on top:
//   * 1024 px / patch 16 -> 64 x 64 tokens, absolute position table (+ conv bias) as the patch
//     GEMM's beta = 1 operand;
//   * 14 x 14 windows on a 64 x 64 map: 64 % 14 != 0, so the map is zero-padded to 70 x 70 = 25
//     windows of 196 slots per image, as SAM's window_partition does AFTER norm1: the padded slots
//     are zero LayerNorm rows (qkv = bias) and they DO take part in the attention as keys (SAM
//     does not mask them); their outputs are dropped by the gather.  Global blocks are one window
//     of 4096 keys per image;
//   * decomposed relative-position bias in every block (sam21_attn2rp.cuh): one batched GEMM
//     P = Q . [Rh | Rw]^T per block (cuBLASLt, batch = heads), indexed inside the attention;
//   * head dim 80 (ViT-H) or 64 (ViT-B/L): compile with -DSAMV1_DH=80 or 64;
//   * neck: 1x1 conv C -> 256, LayerNorm2d, 3x3 conv 256 -> 256, LayerNorm2d; output fp32 NCHW
//     [B, 256, 64, 64] (fp32 accumulation for the two neck GEMMs in every mode).
// INT8 (--int8-fc2 --rot / --gptq) exactly as nano_vit / nano_sam3: rotated LayerNorm emitting
// int8 + per-token scale for qkv and fc1, fc1's epilogue emitting fc2's int8 operand with one scale
// per (token, 128-group), proj and attention in fp16.
// W4A4 (--int4, w4a4_samv1 copy; the ViT prototype's machinery, remeasure_2026-10/w4a4_engine): qkv, fc1, fc2 on
// mma.m16n8k64.s4 with packed int4 operands (k_gemm_s8<..., I4 = true>, sam21_gemm_s8.cuh); the LayerNorm emits
// rotated per-token int4 (k_ln_q4s, sam21_int4v1.cuh, with the windowed blocks' row map); the fc1 epilogue emits
// fc2's H_64-rotated per-(token, 128) int4 operand; weights per output channel 4-bit (RTN at start-up, or the
// WBITS = 4 GPTQ codes of --gptq).  proj, attention, rel-pos and the neck are the INT8 path's (fp16).
// --serve (accuracy harness): one stdin request per line "<in.bin> <out dir>" -> <out dir>/ours_emb.bin, "SERVED n".
//
// build: nvcc -O3 -std=c++17 -DSAMV1_DH=80 -gencode arch=compute_89,code=sm_89 nano_samv1.cu -lcublasLt -o nano_samv1_h
// run:   ./nano_samv1_h --model models/sam_vit_h --batch 8 --f16acc [--int8-fc2 --gptq gptq_h.bin]
//        ./nano_samv1_h ... --input-list list.txt --out-dir emb/   (fp32 [B,3,1024,1024] -> fp32 [B,256,64,64])
// lib:   add -DSAMV1_LIB -shared -Xcompiler -fPIC: samv1_init("<args>"), samv1_run(dIn, dOut) (device pointers)

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

#ifndef SAMV1_DH
#define SAMV1_DH 80
#endif
static const int DH = SAMV1_DH, DHP = SAMV1_DH;     // 64 and 80 are multiples of 16: no head padding

#include "sam21_kernels.cuh"
#include "sam21_attn2rp.cuh"
#include "sam21_int4.cuh"       // --int4: k_quant_w4, k_pack_s4 (the ViT prototype's, unchanged)
#include "sam21_int4v1.cuh"     // --int4: k_ln_q4s (rotated per-token int4 LayerNorm with the row map)

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }
static inline int roundup(int a, int b) { return cdiv(a, b) * b; }

// ---------------------------------------------------------------- config
struct Cfg {
  std::string name;
  int C = 1280, heads = 16, dh = 80, depth = 32, window = 14, img = 1024, patch = 16, G = 64,
      outc = 256, hidden = 5120, relNw = 56, relNg = 256;
  float eps = 1e-6f;
  std::vector<int> global;
  int Cp() const { return heads * DHP; }
  int Kp() const { return 3 * patch * patch; }
  int HW() const { return G * G; }
  int nWs() const { return cdiv(G, window); }            // windows per side (5)
  int ws2() const { return window * window; }             // 196
  int slotsImg() const { return nWs() * nWs() * ws2(); }  // 4900
  bool isGlobal(int i) const { for (int g : global) if (g == i) return true; return false; }
};
static Cfg cfg;

static double jnum(const std::string& js, const char* key, double dflt) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) return dflt;
  p = js.find(':', p); return atof(js.c_str() + p + 1);
}
static std::vector<int> jarr(const std::string& js, const char* key) {
  std::vector<int> v;
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) return v;
  p = js.find('[', p) + 1;
  size_t e = js.find(']', p);
  while (p < e) {
    while (p < e && (js[p] == ' ' || js[p] == ',' || js[p] == '\n')) ++p;
    if (p >= e) break;
    v.push_back(atoi(js.c_str() + p));
    while (p < e && js[p] != ',') ++p;
  }
  return v;
}
static void loadCfg(const std::string& prefix) {
  std::ifstream f(prefix + ".json");
  if (!f) { printf("cannot open %s.json\n", prefix.c_str()); exit(1); }
  std::stringstream ss; ss << f.rdbuf(); std::string js = ss.str();
  size_t p = js.find("\"name\""); p = js.find('"', js.find(':', p)) + 1;
  cfg.name = js.substr(p, js.find('"', p) - p);
  cfg.C = (int)jnum(js, "C", 1280); cfg.heads = (int)jnum(js, "heads", 16); cfg.dh = (int)jnum(js, "dh", 80);
  cfg.depth = (int)jnum(js, "depth", 32); cfg.window = (int)jnum(js, "window", 14);
  cfg.img = (int)jnum(js, "img", 1024); cfg.patch = (int)jnum(js, "patch", 16); cfg.G = (int)jnum(js, "grid", 64);
  cfg.outc = (int)jnum(js, "out_chans", 256); cfg.hidden = (int)jnum(js, "hidden", 4 * cfg.C);
  cfg.eps = (float)jnum(js, "eps", 1e-6); cfg.relNw = (int)jnum(js, "relN_win", 56); cfg.relNg = (int)jnum(js, "relN_glb", 256);
  cfg.global = jarr(js, "global_layers");
  if (cfg.dh != DH) { printf("this build is for head dim %d (model has %d): rebuild with -DSAMV1_DH=%d\n", DH, cfg.dh, cfg.dh); exit(1); }
  if (cfg.window != 14 || cfg.G != 64 || cfg.patch != 16 || cfg.img != 1024 || cfg.outc != 256) { printf("unexpected geometry\n"); exit(1); }
  if (cfg.hidden != 4 * cfg.C) { printf("mlp ratio must be 4\n"); exit(1); }
  printf("model %s: C=%d heads=%d dh=%d depth=%d window=%d global={", cfg.name.c_str(), cfg.C, cfg.heads, cfg.dh,
         cfg.depth, cfg.window);
  for (size_t i = 0; i < cfg.global.size(); ++i) printf("%s%d", i ? "," : "", cfg.global[i]);
  printf("} relN win/glb %d/%d eps=%g\n", cfg.relNw, cfg.relNg, cfg.eps);
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

// ---------------------------------------------------------------- int8 weights (as nano_vit.cu)
__global__ void k_quant_w(const half* __restrict__ w, int8_t* __restrict__ wt,
                          float* __restrict__ sw, int K, int N) {
  const int n = blockIdx.x;
  __shared__ float red[32];
  float mx = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    mx = fmaxf(mx, fabsf(__half2float(w[(size_t)k * N + n])));
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
    float v = __half2float(w[(size_t)k * N + n]);
    int q = __float2int_rn(v * inv);
    wt[(size_t)n * K + k] = (int8_t)max(-127, min(127, q));
  }
}

// ---------------------------------------------------------------- SAM v1 kernels
// patch-embed im2col from the fp32 NCHW batch (already normalised and zero-padded to 1024^2 by
// SAM's preprocess), K = c*256 + ky*16 + kx; one thread per 4 consecutive kx.
__global__ void k_im2col_samv1(const float* __restrict__ img, half* __restrict__ col, int Mp, int G, int P, int IMG) {
  const int K = 3 * P * P, KQ = K / 4;
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)Mp * KQ) return;
  const int t = (int)(i / KQ), k = (int)(i % KQ) * 4;
  const int c = k / (P * P), r = k - c * P * P, ky = r / P, kx = r - ky * P;
  const int b = t / (G * G), p = t - b * G * G, py = p / G, px = p - py * G;
  const float4 v = *(const float4*)(img + ((size_t)(b * 3 + c) * IMG + py * P + ky) * IMG + px * P + kx);
  half2 h[2] = {__floats2half2_rn(v.x, v.y), __floats2half2_rn(v.z, v.w)};
  *(uint2*)(col + (size_t)t * K + k) = *(const uint2*)h;
}
// the [HW, C] position table replicated over the batch
__global__ void k_pos_rep(const half* __restrict__ pos, half* __restrict__ out, int T, int S, int C) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)T * (C / 8)) return;
  int t = (int)(i / (C / 8)), c8 = (int)(i % (C / 8));
  ((float4*)(out + (size_t)t * C))[c8] = ((const float4*)(pos + (size_t)(t % S) * C))[c8];
}
// zero-padded 3x3 im2col over B stacked H x W maps, K = (ky*3+kx)*C + ic; rows >= B*H*W zeroed.
__global__ void k_im2col3x3_b(const half* __restrict__ src, half* __restrict__ dst, int H, int W, int C, int Tv, int Tp) {
  const int NV = C / 8, HW = H * W;
  int n = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (n >= Tp) return;
  int lane = threadIdx.x & 31;
  half* d = dst + (size_t)n * 9 * C;
  const int b = n / HW, p = n - b * HW, y = p / W, x = p - y * W;
  const bool valid = n < Tv;
  for (int i = lane; i < 9 * NV; i += 32) {
    int g = i / NV, v = i - g * NV;
    int ky = g / 3, kx = g - ky * 3;
    int sy = y + ky - 1, sx = x + kx - 1;
    float4 val = make_float4(0.f, 0.f, 0.f, 0.f);
    if (valid && (unsigned)sy < (unsigned)H && (unsigned)sx < (unsigned)W)
      val = ((const float4*)(src + (size_t)(b * HW + sy * W + sx) * C))[v];
    ((float4*)(d + (size_t)g * C))[v] = val;
  }
}
// [B*HW, C] fp16 -> [B, C, HW] fp32
__global__ void k_to_nchw_b(const half* __restrict__ src, float* __restrict__ dst, int HW, int C) {
  __shared__ half tile[32][33];
  const half* s = src + (size_t)blockIdx.z * HW * C;
  float* d = dst + (size_t)blockIdx.z * HW * C;
  int t0 = blockIdx.x * 32, c0 = blockIdx.y * 32, tx = threadIdx.x, ty = threadIdx.y;
  for (int r = 0; r < 32; r += 8) {
    int t = t0 + ty + r, c = c0 + tx;
    tile[ty + r][tx] = (t < HW && c < C) ? s[(size_t)t * C + c] : __float2half(0.f);
  }
  __syncthreads();
  for (int r = 0; r < 32; r += 8) {
    int c = c0 + ty + r, t = t0 + tx;
    if (c < C && t < HW) d[(size_t)c * HW + t] = __half2float(tile[tx][ty + r]);
  }
}

// ---------------------------------------------------------------- cuBLASLt plans
static cublasLtHandle_t LT;
static void* gWs = nullptr;
static size_t gWsBytes = 96ull << 20;
static int   gAccMode = 0;        // 0 = fp32 acc, 1 = fp16 acc except deep-K, 2 = fp16 acc
static int   gAttnAcc = -1;
static bool  gArch80 = false, gArch89 = false;
static bool  gInt8 = false, gInt8Fc2 = false, gRot = false, gLtGelu = false;
static int   gFc2Rot = 64;
// --int4 (W4A4, w4a4_samv1): implies --int8-fc2 --rot; qkv / fc1 / fc2 on the INT4 mma, see the header comment
static bool  gInt4 = false;
static int8_t* gW4tmp = nullptr;    // --int4: int8 codes [N, K] before packing
static std::string gGptq, gDump;
// samv1_static_fc2_20261026: --fc2-static <file> (needs --int4): one static symmetric fc2-input scale per block.
// File "NANOF2S1" + int32 depth + float32[depth] s (= a_ratio * s_amax / 7 of a protocol-P fit, scripts/export_f2s.py).
// fc1 runs the F2S producer (codes clamp(round(v * hn / s)), no row max, no scale store), fc2 the GROUP = 0 SASCALAR
// int4 GEMM (one int32 accumulator over K, epilogue acc * s * sw[col] + bias + residual).  Without it: unchanged.
static std::string gF2SFile;
static float* dF2S = nullptr;
// w8a8_static_fc2_20261026: --fc2-static also on the sm_89 W8A8 path (--int8-fc2 --rot --gptq, H_64): the existing F2S
// producer instantiated with I4 = false (codes cvt.rni.sat.s8(v * hn / s), i.e. -128..127, one s per block), fc2 the
// existing GROUP = 0 SASCALAR INT8 GEMM on runFc2S8's sm_89 tile (128x192, 2x4 warps); no kernel source change.
// --w8-f2s-dump <dir> (DYNAMIC W8A8 path, first forward only): per block every NANO_W8_DUMP_STRIDE-th (default 8) of the
// Tv rows of fc2's codes / scales, max HS over all Tv rows, fc2 int8 weight and column scales (calibration).
// --w8-f2s-gate <dir> (first forward only, block NANO_W8_GATE_BLOCK, default 0): fp16 GELU output (an extra plain-epilogue
// fc1 launch of the same tile), codes, (dynamic) scales, residual before / after fc2, fc2 weight / scales / bias.
static std::string gW8F2SDump, gW8F2SGate;
static bool gW8First = true;
static bool  gTune = true, gProf = false;
// --rel-gemm: the rel-pos products come from a batched cuBLASLt GEMM into a [rows][heads][relN] buffer
// (the first implementation); default: computed inside the attention kernel from the transposed tables.
static bool gRelGemm = false;
// natural-order qkv for the windowed blocks (default; --slot-qkv = the first implementation): the LayerNorm
// and the qkv GEMM run over the MQ = Mp + 128 token rows (rows >= Tv are zero inputs, so their qkv is the
// bias = SAM's zero padding), the attention reads the window slots through mSlotRow and writes the token
// rows directly: 16 % fewer qkv rows than the 25 x 196-slot layout and no gather pass.
static bool gNatQkv = true;
static std::vector<half*> gRelT;          // per block: the transposed table [relN][DH]
static half *gTA = nullptr, *gTB = nullptr, *gTD = nullptr;

struct Plan {
  cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t La, Lb, Lc; cublasLtMatmulAlgo_t algo;
  int m, n, k; bool beta1, f16acc;
};
static std::vector<Plan> gPlans;

// Generic: D[m, n] (+)= A[m, k] . B[k, n] (+ bias), all row-major fp16.  lda/ldc default to k/n;
// batch > 1 runs `batch` GEMMs with element strides sA / sB / sC (sB = 0: shared weight).
// accOverride: -1 = follow gAccMode, 0 = force fp32 accumulation.
static int makePlan(int m, int n, int k, cublasLtEpilogue_t epi, const void* bias, bool beta1,
                    bool deepK = false, int lda = 0, int ldc = 0, int batch = 1, long long sA = 0,
                    long long sB = 0, long long sC = 0, int accOverride = -1) {
  Plan p{}; p.m = m; p.n = n; p.k = k; p.beta1 = beta1;
  bool f16 = (gAccMode == 2) || (gAccMode == 1 && !deepK);
  if (accOverride == 0) f16 = false;
  p.f16acc = f16;
  cublasComputeType_t comp = f16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
  cudaDataType_t st = f16 ? CUDA_R_16F : CUDA_R_32F;
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
  if (!lda) lda = k;
  if (!ldc) ldc = n;
  CL(cublasLtMatrixLayoutCreate(&p.La, CUDA_R_16F, n, k, n));
  CL(cublasLtMatrixLayoutCreate(&p.Lb, CUDA_R_16F, k, m, lda));
  CL(cublasLtMatrixLayoutCreate(&p.Lc, CUDA_R_16F, n, m, ldc));
  if (batch > 1) {
    int32_t bc = batch;
    CL(cublasLtMatrixLayoutSetAttribute(p.La, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    CL(cublasLtMatrixLayoutSetAttribute(p.Lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    CL(cublasLtMatrixLayoutSetAttribute(p.Lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    int64_t oa = sB, ob = sA, oc = sC;     // La is the weight ("B" here), Lb the activation
    CL(cublasLtMatrixLayoutSetAttribute(p.La, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &oa, sizeof(oa)));
    CL(cublasLtMatrixLayoutSetAttribute(p.Lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &ob, sizeof(ob)));
    CL(cublasLtMatrixLayoutSetAttribute(p.Lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &oc, sizeof(oc)));
  }
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
    const void* al = f16 ? (const void*)&a16 : (const void*)&a32;
    const void* be = f16 ? (const void*)&b16 : (const void*)&b32;
    float best = 1e30f;
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int c = 0; c < found; ++c) {
      cublasStatus_t st2 = cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc,
                                          gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
      if (st2 != CUBLAS_STATUS_SUCCESS) continue;
      for (int r = 0; r < 2; ++r)
        cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0));
      for (int r = 0; r < 6; ++r)
        cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
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
  static const half  a16 = __float2half(1.f), b16_0 = __float2half(0.f), b16_1 = __float2half(1.f);
  const void* alpha = p.f16acc ? (const void*)&a16 : (const void*)&a32;
  const void* beta  = p.f16acc ? (p.beta1 ? (const void*)&b16_1 : (const void*)&b16_0)
                               : (p.beta1 ? (const void*)&b32_1 : (const void*)&b32_0);
  CL(cublasLtMatmul(LT, p.op, alpha, B, p.La, A, p.Lb, beta, p.beta1 ? C : D, p.Lc, D, p.Lc,
                    &p.algo, gWs, gWsBytes, s));
}

// ---------------------------------------------------------------- buffers
static int gB = 8, Mp = 0, Tv = 0, Sw = 0, Mw = 0, MR = 0;
static half *X, *Xn, *QKV, *HID, *AO, *COL, *POS, *REL = nullptr, *N1, *N1n, *NCOL, *N2, *N2n;
static float *dIn = nullptr, *dOut = nullptr;
static int8_t *XQ = nullptr, *HQ = nullptr;
static float *XS = nullptr, *HS = nullptr;
static int *mSlot2Nat = nullptr, *mNat2Slot = nullptr, *mSlotRow = nullptr, *mNatSrcPad = nullptr;
static int MQ = 0;
struct Q8W { int8_t* w; float* s; };
static std::vector<Q8W> QWqkv, QWfc1, QWfc2;
static half* gRotTmp = nullptr;

static half* dAlloc(size_t n) { void* p; CK(cudaMalloc(&p, n * sizeof(half) + 65536)); CK(cudaMemset(p, 0, n * sizeof(half) + 65536)); return (half*)p; }
static int* upload(const std::vector<int>& v) {
  void* p; CK(cudaMalloc(&p, v.size() * 4)); CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice)); return (int*)p;
}

static void buildMaps() {
  const int G = cfg.G, ws = cfg.window, nW = cfg.nWs(), ws2 = cfg.ws2(), SI = cfg.slotsImg(), HW = cfg.HW();
  std::vector<int> s2n(Mw, -1), n2s(Mp);
  for (int s = 0; s < Sw; ++s) {
    int b = s / SI, r = s - b * SI, w = r / ws2, i = r - w * ws2;
    int y = (w / nW) * ws + i / ws, x = (w % nW) * ws + i % ws;
    if (y < G && x < G) { s2n[s] = b * HW + y * G + x; n2s[b * HW + y * G + x] = s; }
  }
  mSlot2Nat = upload(s2n); mNat2Slot = upload(n2s);
  std::vector<int> sr(Sw + 64, Tv), src(MQ);           // padding slots (and the slack) -> zero row Tv
  for (int s = 0; s < Sw; ++s) if (s2n[s] >= 0) sr[s] = s2n[s];
  for (int t = 0; t < MQ; ++t) src[t] = t < Tv ? t : -1;
  mSlotRow = upload(sr); mNatSrcPad = upload(src);
}

static void allocAll() {
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, F = cfg.outc;
  Tv = gB * cfg.HW(); Mp = roundup(Tv, 128);
  Sw = gB * cfg.slotsImg(); Mw = roundup(Sw, 128); MQ = Mp + 128; MR = std::max(std::max(Mp, Mw), MQ);
  X = dAlloc((size_t)Mp * C); Xn = dAlloc((size_t)MR * C);
  QKV = dAlloc((size_t)MR * 3 * Cp + 64 * 3 * Cp);
  HID = dAlloc((size_t)Mp * H4);
  AO = dAlloc((size_t)MR * C);
  COL = dAlloc((size_t)Mp * cfg.Kp());
  POS = dAlloc((size_t)Mp * C);
  if (gRelGemm) REL = dAlloc((size_t)MR * cfg.heads * std::max(cfg.relNg, cfg.relNw));
  // transposed rel tables for the in-kernel products: relt [DH][relN] -> [relN][DH]
  gRelT.resize(cfg.depth);
  for (int i = 0; i < cfg.depth; ++i) {
    const int rn = cfg.isGlobal(i) ? cfg.relNg : cfg.relNw;
    std::vector<half> a((size_t)DH * rn), t((size_t)DH * rn);
    CK(cudaMemcpy(a.data(), WP(bk(i, "relt")), a.size() * 2, cudaMemcpyDeviceToHost));
    for (int k = 0; k < DH; ++k) for (int j = 0; j < rn; ++j) t[(size_t)j * DH + k] = a[(size_t)k * rn + j];
    CK(cudaMalloc(&gRelT[i], t.size() * 2)); CK(cudaMemcpy(gRelT[i], t.data(), t.size() * 2, cudaMemcpyHostToDevice));
  }
  N1 = dAlloc((size_t)Mp * F); N1n = dAlloc((size_t)Mp * F); NCOL = dAlloc((size_t)Mp * 9 * F);
  N2 = dAlloc((size_t)Mp * F); N2n = dAlloc((size_t)Mp * F);
  CK(cudaMalloc(&dIn, (size_t)gB * 3 * cfg.img * cfg.img * sizeof(float)));
  CK(cudaMemset(dIn, 0, (size_t)gB * 3 * cfg.img * cfg.img * sizeof(float)));
  CK(cudaMalloc(&dOut, (size_t)gB * F * cfg.HW() * sizeof(float)));
  if (gTune) {          // autotune scratch: the largest A / B / D of any plan
    size_t a = std::max({(size_t)MR * 3 * Cp, (size_t)Mp * H4, (size_t)Mp * 9 * F});
    size_t d = std::max({(size_t)MR * 3 * Cp, (size_t)Mp * H4, gRelGemm ? (size_t)MR * cfg.heads * cfg.relNg : (size_t)0});
    gTA = dAlloc(a); gTB = dAlloc((size_t)H4 * std::max(H4, 3 * Cp)); gTD = dAlloc(d);
  }
  if (gInt8) {
    CK(cudaMalloc(&XQ, (size_t)MR * C)); CK(cudaMemset(XQ, 0, (size_t)MR * C));
    CK(cudaMalloc(&XS, (size_t)MR * sizeof(float))); CK(cudaMemset(XS, 0, (size_t)MR * sizeof(float)));
  }
  if (gInt8Fc2) {
    CK(cudaMalloc(&HQ, (size_t)Mp * H4)); CK(cudaMemset(HQ, 0, (size_t)Mp * H4));
    CK(cudaMalloc(&HS, (size_t)Mp * cdiv(H4, 128) * sizeof(float)));
  }
  buildMaps();
  k_pos_rep<<<cdiv(Mp * (C / 8), 256), 256>>>((const half*)WP("pos"), POS, Mp, cfg.HW(), C);
  CK(cudaDeviceSynchronize());
  printf("batch %d: %d tokens (padded %d), %d window slots (padded %d)\n", gB, Tv, Mp, Sw, Mw);
}
static void freeTuneScratch() {
  if (gTA) { cudaFree(gTA); cudaFree(gTB); cudaFree(gTD); gTA = gTB = gTD = nullptr; }
}

// ---------------------------------------------------------------- int8 weights (as nano_sam3.cu)
static void quantRotated(const half* w, int8_t* wt, float* sw, int K, int N, int kind) {
  if (!gRotTmp) CK(cudaMalloc(&gRotTmp, (size_t)cfg.hidden * std::max(cfg.hidden, 3 * cfg.Cp()) * sizeof(half)));
  const size_t sh = 2ull * K * sizeof(float);
  if (kind == 2) k_rot_w<2><<<N, 256, sh>>>(w, gRotTmp, K, N);
  else           k_rot_w<1><<<N, 256, sh>>>(w, gRotTmp, K, N, gFc2Rot);
  k_quant_w<<<N, 256>>>(gRotTmp, wt, sw, K, N);
  CK(cudaDeviceSynchronize());
}
// --int4: int8 codes [rows, K] (device) -> packed int4 [rows, K/2] at dst; any code outside -7..7 is fatal
static void packS4(const int8_t* codes, int8_t* dst, size_t n) {
  static unsigned int* dbad = nullptr;
  if (!dbad) CK(cudaMalloc(&dbad, sizeof(unsigned int)));
  CK(cudaMemset(dbad, 0, sizeof(unsigned int)));
  const size_t np = n / 2;
  k_pack_s4<<<(unsigned)((np + 255) / 256), 256>>>(codes, (uint8_t*)dst, np, dbad);
  unsigned int h = 0; CK(cudaMemcpy(&h, dbad, sizeof h, cudaMemcpyDeviceToHost));
  if (h) { printf("int4 pack: %u code pairs outside -7..7 (not a WBITS=4 file?)\n", h); exit(1); }
}
static void loadGptq(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot open %s\n", path.c_str()); exit(1); }
  char magic[8]; int32_t nblk, flags;
  f.read(magic, 8); f.read((char*)&nblk, 4); f.read((char*)&flags, 4);
  if (memcmp(magic, "NANOGPTQ", 8) || nblk != cfg.depth) { printf("%s: bad header (nblk %d)\n", path.c_str(), nblk); exit(1); }
  std::vector<char> buf;
  auto take = [&](void* dst, size_t bytes) {
    buf.resize(bytes); f.read(buf.data(), bytes);
    if (!f) { printf("%s: truncated\n", path.c_str()); exit(1); }
    CK(cudaMemcpy(dst, buf.data(), bytes, cudaMemcpyHostToDevice));
  };
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden;
  // --int4: the file's int8 codes (a WBITS = 4 export, -7..7) are staged in gW4tmp and packed two per byte
  auto takeW = [&](int8_t* dst, size_t n) {
    if (!gInt4) { take(dst, n); return; }
    take(gW4tmp, n); packS4(gW4tmp, dst, n);
  };
  for (int i = 0; i < cfg.depth; ++i) {
    takeW(QWqkv[i].w, (size_t)(3 * Cp) * C); take(QWqkv[i].s, (size_t)(3 * Cp) * 4);
    takeW(QWfc1[i].w, (size_t)H4 * C);       take(QWfc1[i].s, (size_t)H4 * 4);
    if (gInt8Fc2) { takeW(QWfc2[i].w, (size_t)C * H4); take(QWfc2[i].s, (size_t)C * 4); }
    else f.seekg((std::streamoff)((size_t)C * H4 + (size_t)C * 4), std::ios::cur);
  }
  printf("gptq weights: %s (flags %d)\n", path.c_str(), flags);
}
// --int4: rotate exactly as quantRotated, quantise to 4-bit codes (k_quant_w4), pack into dst.w
static void quantRotated4(const half* w, const Q8W& dst, int K, int N, int kind) {
  if (!gRotTmp) CK(cudaMalloc(&gRotTmp, (size_t)cfg.hidden * std::max(cfg.hidden, 3 * cfg.Cp()) * sizeof(half)));
  const size_t sh = 2ull * K * sizeof(float);
  if (kind == 2) k_rot_w<2><<<N, 256, sh>>>(w, gRotTmp, K, N);
  else           k_rot_w<1><<<N, 256, sh>>>(w, gRotTmp, K, N, gFc2Rot);
  k_quant_w4<<<N, 256>>>(gRotTmp, gW4tmp, dst.s, K, N);
  CK(cudaDeviceSynchronize());
  packS4(gW4tmp, dst.w, (size_t)K * N);
}
static void buildInt4Weights() {
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, NB = cfg.depth;
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB);
  const size_t per = ((size_t)(3 * Cp) * C + (size_t)H4 * C + (size_t)C * H4) / 2;   // packed bytes
  const size_t sper = 3 * Cp + H4 + C;
  int8_t* blob; CK(cudaMalloc(&blob, per * NB));
  float* sblob; CK(cudaMalloc(&sblob, sper * NB * sizeof(float)));
  CK(cudaMalloc(&gW4tmp, (size_t)std::max(H4, 3 * Cp) * C));      // the largest [N, K] code matrix
  size_t off = 0, soff = 0;
  for (int i = 0; i < NB; ++i) {
    QWqkv[i] = {blob + off, sblob + soff}; off += (size_t)(3 * Cp) * C / 2; soff += 3 * Cp;
    QWfc1[i] = {blob + off, sblob + soff}; off += (size_t)H4 * C / 2;       soff += H4;
    QWfc2[i] = {blob + off, sblob + soff}; off += (size_t)C * H4 / 2;       soff += C;
    quantRotated4((const half*)WP(bk(i, "qkv.w")), QWqkv[i], C, 3 * Cp, 2);
    quantRotated4((const half*)WP(bk(i, "fc1.w")), QWfc1[i], C, H4, 2);
    quantRotated4((const half*)WP(bk(i, "fc2.w")), QWfc2[i], H4, C, 1);
  }
  CK(cudaDeviceSynchronize());
  if (gRotTmp) { cudaFree(gRotTmp); gRotTmp = nullptr; }
  if (!gGptq.empty()) loadGptq(gGptq);
  cudaFree(gW4tmp); gW4tmp = nullptr;
  printf("int4 weights: %.1f MB packed (qkv + fc1 + fc2 of %d blocks), rotated%s\n", per * NB / 1e6, NB,
         gGptq.empty() ? ", RTN 4-bit" : ", GPTQ codes");
}
static void buildInt8Weights() {
  if (gInt4) { buildInt4Weights(); return; }
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, NB = cfg.depth;
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB);
  size_t per = (size_t)(3 * Cp) * C + (size_t)H4 * C + (gInt8Fc2 ? (size_t)C * H4 : 0);
  size_t sper = 3 * Cp + H4 + C;
  int8_t* blob; CK(cudaMalloc(&blob, per * NB));
  float* sblob; CK(cudaMalloc(&sblob, sper * NB * sizeof(float)));
  size_t off = 0, soff = 0;
  for (int i = 0; i < NB; ++i) {
    QWqkv[i] = {blob + off, sblob + soff}; off += (size_t)(3 * Cp) * C; soff += 3 * Cp;
    QWfc1[i] = {blob + off, sblob + soff}; off += (size_t)H4 * C;       soff += H4;
    const half* wq = (const half*)WP(bk(i, "qkv.w")); const half* wf = (const half*)WP(bk(i, "fc1.w"));
    if (gRot) quantRotated(wq, QWqkv[i].w, QWqkv[i].s, C, 3 * Cp, 2);
    else      k_quant_w<<<3 * Cp, 256>>>(wq, QWqkv[i].w, QWqkv[i].s, C, 3 * Cp);
    if (gRot) quantRotated(wf, QWfc1[i].w, QWfc1[i].s, C, H4, 2);
    else      k_quant_w<<<H4, 256>>>(wf, QWfc1[i].w, QWfc1[i].s, C, H4);
    if (gInt8Fc2) {
      QWfc2[i] = {blob + off, sblob + soff}; off += (size_t)C * H4; soff += C;
      const half* w2 = (const half*)WP(bk(i, "fc2.w"));
      if (gRot) quantRotated(w2, QWfc2[i].w, QWfc2[i].s, H4, C, 1);
      else      k_quant_w<<<C, 256>>>(w2, QWfc2[i].w, QWfc2[i].s, H4, C);
    }
  }
  CK(cudaDeviceSynchronize());
  if (gRotTmp) { cudaFree(gRotTmp); gRotTmp = nullptr; }
  if (!gGptq.empty()) loadGptq(gGptq);
  printf("int8 weights: %.1f MB (%s of %d blocks)%s%s\n", per * NB / 1e6,
         gInt8Fc2 ? "qkv + fc1 + fc2" : "qkv + fc1", NB, gRot ? ", rotated" : "", gGptq.empty() ? "" : ", GPTQ");
}

// ---------------------------------------------------------------- launchers
static void launchLN(const half* x, half* y, const float* g, const float* b, int T, int C,
                     cudaStream_t s, const int* src = nullptr) {
  int grid = cdiv(T, 8), thr = 256; size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  switch (C) {
    case 256:  k_ln_t<32><<<grid, thr, sh, s>>>(x, y, g, b, T, src, e); break;
    case 768:  k_ln_t<96><<<grid, thr, sh, s>>>(x, y, g, b, T, src, e); break;
    case 1024: k_ln_t<128><<<grid, thr, sh, s>>>(x, y, g, b, T, src, e); break;
    case 1280: k_ln_t<160><<<grid, thr, sh, s>>>(x, y, g, b, T, src, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}
static void launchLNQ8(const half* x, int8_t* y, float* ys, const float* g, const float* b,
                       int T, int C, cudaStream_t s, const int* src = nullptr) {
  dim3 grid(cdiv(T, 8)), thr(256); size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  if (gRot) {
    switch (C) {
      case 768:  k_ln_q8<96, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
      case 1024: k_ln_q8<128, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
      case 1280: k_ln_q8<160, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
      default: printf("unexpected LN width %d\n", C); exit(1);
    }
    return;
  }
  switch (C) {
    case 768:  k_ln_q8<96><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
    case 1024: k_ln_q8<128><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
    case 1280: k_ln_q8<160><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, src, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}

template<bool GELU, bool SASCALAR = false, bool RESID = false, int GROUP = 0, bool Q8OUT = false,
         int NGPT = 1, int WM = 2, int WN = 4, bool U8IN = false, bool U8OUT = false, int ROT8 = 0,
         bool DB = false, int MINB = 0, int BM = 128, int BN = 128, int ST = 3>
static void runS8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C,
                  int M, int N, int K, cudaStream_t s, const half* resid = nullptr,
                  int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0) {
  constexpr int BK = 64;
  if constexpr (!GELU && !SASCALAR && !RESID && GROUP == 0 && !Q8OUT && !U8IN && !U8OUT && ROT8 == 0 &&
                BM == 128 && BN == 128 && ST == 3 && !DB) {
    if (gArch89 && K >= 576) {
      constexpr int ST2 = 2;
      constexpr size_t sh2 = s8_smem<BM, BN, BK, ST2>();
      static bool attr2 = false;
      if (!attr2) {
        CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, 2, 2, ST2, false, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
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
    CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr = true;
  }
  k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB>
      <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride);
}
static int gFc2Tile = 0;
static void runFc2S8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C,
                     int M, int N, int K, cudaStream_t s, int sastride) {
  // --fc2-tile: 1 = 128x128 / 4x4 warps / 2 stages, 2 = 128x128 / 2x4 / 3 stages, 3 = 64x192 / 2x4 (sweep options)
  if (gFc2Tile == 1)
    runS8<false, false, true, 128, false, 1, 4, 4, false, false, 0, false, 0, 128, 128, 2>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  else if (gFc2Tile == 2)
    runS8<false, false, true, 128, false, 1, 2, 4, false, false, 0, false, 0, 128, 128, 3>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  else if (gFc2Tile == 3)
    runS8<false, false, true, 128, false, 1, 2, 4, false, false, 0, false, 0, 64, 192, 3>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  else if (gArch89)
    runS8<false, false, true, 128, false, 1, 2, 4, false, false, 0, false, 0, 128, 192>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
  else
    runS8<false, false, true, 128, false, 1, 2, 4, false, false, 0>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride);
}
// ---------------------------------------------------------------- --int4 launchers (w4a4_samv1)
static void launchLNQ4(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C,
                       cudaStream_t s, const int* src = nullptr) {
  dim3 grid(cdiv(T, 8)), thr(256); size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  uint8_t* yu = (uint8_t*)y;
  switch (C) {
    case 768:  k_ln_q4s<96><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, src, e); break;
    case 1024: k_ln_q4s<128><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, src, e); break;
    case 1280: k_ln_q4s<160><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, src, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}
// k_gemm_s8<..., I4 = true>: K is given in ELEMENTS here and passed to the kernel in bytes (K / 2); BK in bytes
// (64 = 128 int4); GROUP in elements.  Same template as the ViT prototype's runS4.
template<bool GELU, bool RESID, int GROUP, bool Q8OUT, int ROT8, int BM, int BN, int BK, int WM, int WN, int ST,
         bool DB = false, int MINB = 0, bool SASC = false, bool F2S = false>
static void runS4(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K,
                  cudaStream_t s, const half* resid = nullptr, int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0) {
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASC, RESID, GROUP, Q8OUT, 1, false, false, ROT8, MINB, false, false, false, true, false, F2S>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr = true;
  }
  if (M % BM) { printf("runS4: M %d not a multiple of BM %d\n", M, BM); exit(1); }
  k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASC, RESID, GROUP, Q8OUT, 1, false, false, ROT8, MINB, false, false, false, true, false, F2S>
      <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K / 2, resid, C8, sa2, sa_stride);
}
// Tile candidates of the three INT4 GEMMs (micro/gemm4tile_v1.cu times the same list at the SAM v1 shapes).
// Default (index -1 = the rule below), from the RTX 4090 micro-benchmark (w4a4_samv1 timing/m1, gemm4tile_b{1,8}.log):
//   sm_89: qkv c1 = 128x128 2x4 warps 3 stages (the INT8 path's 2x2 / 2-stage tile, c0, is 9-16 % slower at batch 8);
//          fc1 c0 = 128x128 2x2 3 stages double-buffered fragments (the INT8 path's);
//          fc2 c0 = 128x192 2x4 (the INT8 path's), except ViT-L (N = 1024) at one image per batch (M <= 4096 rows):
//          c1 = 128x128 4x4 2 stages (0.0499 vs 0.0684 ms; end to end 7.77-7.85 vs 8.16-8.17 ms, timing/tc1).
//          (ViT-H's c5 at batch 1 was 3 % faster in the micro-benchmark but 0.4-0.6 % slower end to end, timing/tc2.)
//   sm_86: the ViT prototype's sm_86 rules at M > 2048 (qkv 128x128 2x4, fc1 as sm_89, fc2 64x128 for N < 1024 else
//          128x128 4x4 2 stages); not tuned here.
// NANO_I4_QKV / NANO_I4_FC1 / NANO_I4_FC2 = <index> force one candidate (tuning only; every candidate gives
// bit-identical outputs, the int32 sums are exact and the fp32 epilogue / group flush order does not depend on the tile).
static int gI4Q = -2, gI4F1 = -2, gI4F2 = -2;
static int envIdx(const char* n) { const char* e = getenv(n); return e && *e ? atoi(e) : -1; }
static void i4Env() {
  if (gI4Q != -2) return;
  gI4Q = envIdx("NANO_I4_QKV"); gI4F1 = envIdx("NANO_I4_FC1"); gI4F2 = envIdx("NANO_I4_FC2");
  if (gI4Q >= 0 || gI4F1 >= 0 || gI4F2 >= 0) printf("NANO_I4 tile override: qkv %d fc1 %d fc2 %d%c", gI4Q, gI4F1, gI4F2, 10);
}
static void runQkv4(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K, cudaStream_t s) {
  i4Env();
  int c = gI4Q >= 0 ? gI4Q : 1;
  switch (c) {
    case 0: runS4<false, false, 0, false, 0, 128, 128, 64, 2, 2, 2>(A, w, sa, bias, C, M, N, K, s); break;
    case 1: runS4<false, false, 0, false, 0, 128, 128, 64, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s); break;
    case 2: runS4<false, false, 0, false, 0, 128, 256, 64, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s); break;
    case 3: runS4<false, false, 0, false, 0, 256, 128, 64, 4, 2, 3>(A, w, sa, bias, C, M, N, K, s); break;
    case 4: runS4<false, false, 0, false, 0, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s); break;
    case 5: runS4<false, false, 0, false, 0, 128, 128, 64, 2, 2, 3>(A, w, sa, bias, C, M, N, K, s); break;
    default: printf("NANO_I4_QKV %d: no such candidate\n", c); exit(1);
  }
}
static void runFc14(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K,
                    cudaStream_t s, int8_t* C8, float* sa2) {
  i4Env();
  int c = gI4F1 >= 0 ? gI4F1 : 0;
  switch (c) {
    case 0: runS4<true, false, 0, true, 64, 128, 128, 64, 2, 2, 3, true, 1>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    case 1: runS4<true, false, 0, true, 64, 128, 128, 64, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    case 2: runS4<true, false, 0, true, 64, 256, 128, 64, 4, 2, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    case 3: runS4<true, false, 0, true, 64, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2); break;
    default: printf("NANO_I4_FC1 %d: no such candidate\n", c); exit(1);
  }
}
static void runFc24(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* X_, int M, int N, int K,
                    cudaStream_t s, int sastride) {
  i4Env();
  int c = gI4F2 >= 0 ? gI4F2
        : gArch89 ? ((M <= 4096 && N == 1024) ? 1 : 0)
        : (N < 1024 ? 5 : 1);
  switch (c) {
    case 0: runS4<false, true, 128, false, 0, 128, 192, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 1: runS4<false, true, 128, false, 0, 128, 128, 64, 4, 4, 2>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 2: runS4<false, true, 128, false, 0, 128, 128, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 3: runS4<false, true, 128, false, 0, 64, 192, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 4: runS4<false, true, 128, false, 0, 128, 64, 64, 4, 2, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 5: runS4<false, true, 128, false, 0, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    case 6: runS4<false, true, 128, false, 0, 64, 64, 64, 2, 2, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride); break;
    default: printf("NANO_I4_FC2 %d: no such candidate\n", c); exit(1);
  }
}
// samv1_static_fc2: the static-scale fc1 producer / fc2 consumer (same tile candidates and default rule as runFc14 /
// runFc24; sf = &dF2S[block]).
static void runFc14S(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K,
                     cudaStream_t s, int8_t* C8, float* sf) {
  i4Env();
  int c = gI4F1 >= 0 ? gI4F1 : 0;
  switch (c) {
    case 0: runS4<true, false, 0, true, 64, 128, 128, 64, 2, 2, 3, true, 1, false, true>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sf); break;
    case 1: runS4<true, false, 0, true, 64, 128, 128, 64, 2, 4, 3, false, 0, false, true>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sf); break;
    case 2: runS4<true, false, 0, true, 64, 256, 128, 64, 4, 2, 3, false, 0, false, true>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sf); break;
    case 3: runS4<true, false, 0, true, 64, 64, 128, 64, 2, 4, 3, false, 0, false, true>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sf); break;
    default: printf("NANO_I4_FC1 %d: no such candidate\n", c); exit(1);
  }
}
static void runFc24S(const int8_t* A, const Q8W& w, const float* sf, const half* bias, half* X_, int M, int N, int K,
                     cudaStream_t s) {
  i4Env();
  int c = gI4F2 >= 0 ? gI4F2
        : gArch89 ? ((M <= 4096 && N == 1024) ? 1 : 0)
        : (N < 1024 ? 5 : 1);
  switch (c) {
    case 0: runS4<false, true, 0, false, 0, 128, 192, 64, 2, 4, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 1: runS4<false, true, 0, false, 0, 128, 128, 64, 4, 4, 2, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 2: runS4<false, true, 0, false, 0, 128, 128, 64, 2, 4, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 3: runS4<false, true, 0, false, 0, 64, 192, 64, 2, 4, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 4: runS4<false, true, 0, false, 0, 128, 64, 64, 4, 2, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 5: runS4<false, true, 0, false, 0, 64, 128, 64, 2, 4, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    case 6: runS4<false, true, 0, false, 0, 64, 64, 64, 2, 2, 3, false, 0, true>(A, w, sf, bias, X_, M, N, K, s, X_); break;
    default: printf("NANO_I4_FC2 %d: no such candidate\n", c); exit(1);
  }
}
static void runFc1Gelu(const half* A, const half* Wt, const half* bias, half* C, int M, int N,
                       int K, bool f16, cudaStream_t s) {
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

// Attention with the rel-pos bias.  Windowed: nWin = B*25 windows of 196 slots (keys include the
// padded slots, as in SAM).  Global: B windows of 4096 tokens.
static int gAttnNW = 4, gAttnMinB = 1;
static void runAttnRP(bool glb, const half* qkv, half* out, const half* rel, cudaStream_t s) {
  const int Cp = cfg.Cp(), Co = cfg.C;
  const int ws2 = glb ? cfg.HW() : cfg.ws2(), nWin = glb ? gB : gB * cfg.nWs() * cfg.nWs();
  const int relN = glb ? cfg.relNg : cfg.relNw;
  const int acc = gAttnAcc >= 0 ? gAttnAcc : 1;
  const int Bw = nWin * cfg.heads;
#define ATT(SIDE, NW, QF, PF, MB) do { if (gRelGemm) \
  k_attn2rp<DH, SIDE, NW, 32, 2, QF, PF, MB, false><<<dim3(cdiv(ws2, NW * 16), Bw), NW * 32, attn2_smem<DH, 32, 2>(), s>>>( \
      qkv, out, rel, relN, ws2, cfg.heads, Cp, Co, ws2, ws2); \
  else if (mapped) k_attn2rp<DH, SIDE, NW, 32, 2, QF, PF, MB, true, true><<<dim3(cdiv(ws2, NW * 16), Bw), NW * 32, \
      attn2rp_smem<DH, 32, 2>(SIDE % 32 != 0, NW, relN), s>>>(qkv, out, rel, relN, ws2, cfg.heads, Cp, Co, ws2, ws2, mSlotRow, Tv); \
  else k_attn2rp<DH, SIDE, NW, 32, 2, QF, PF, MB, true><<<dim3(cdiv(ws2, NW * 16), Bw), NW * 32, \
      attn2rp_smem<DH, 32, 2>(SIDE % 32 != 0, NW, relN), s>>>(qkv, out, rel, relN, ws2, cfg.heads, Cp, Co, ws2, ws2); } while (0)
#define ATT_ACC(SIDE, NW, MB) do { if (acc == 2) ATT(SIDE, NW, true, true, MB); else if (acc == 1) ATT(SIDE, NW, false, true, MB); \
                                   else ATT(SIDE, NW, false, false, MB); } while (0)
  // --attn-minb 3 caps the kernels at 170 registers so that three 128-thread CTAs fit an SM (13 = window only)
  const bool mapped = !glb && gNatQkv && !gRelGemm;
  if (glb && gAttnMinB == 3) ATT_ACC(64, 4, 3);
  else if (glb) ATT_ACC(64, 4, 1);
  else if (gAttnMinB == 3 || gAttnMinB == 13) ATT_ACC(14, 4, 3);
  else if (gAttnNW == 8) ATT_ACC(14, 8, 1);
  else ATT_ACC(14, 4, 1);
#undef ATT_ACC
#undef ATT
}

// ---------------------------------------------------------------- profiling
enum PTag { P_IM2COL, P_PATCH, P_LN, P_QKV, P_REL, P_ATTN, P_ATTNG, P_GATHER, P_PROJ, P_FC1, P_FC2, P_NECK, P_NTAG };
static const char* PNAME[P_NTAG] = {"im2col", "patch_gemm", "layernorm", "qkv_gemm", "relpos_gemm", "attn_window", "attn_global",
                                    "gather", "proj_gemm", "fc1_gemm", "fc2_gemm", "neck"};
static std::vector<cudaEvent_t> gEv; static std::vector<int> gTag; static int gEvN = 0;
static void mark(int tag, cudaStream_t s) {
  if (!gProf) return;
  if (gEvN >= (int)gEv.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); gEv.push_back(e); }
  CK(cudaEventRecord(gEv[gEvN], s));
  if ((int)gTag.size() <= gEvN) gTag.push_back(tag); else gTag[gEvN] = tag;
  ++gEvN;
}
static void profReport() {
  double acc[P_NTAG] = {0}, accg[P_NTAG] = {0}; double tot = 0;
  for (int i = 0; i + 1 < gEvN; ++i) {
    float ms; CK(cudaEventElapsedTime(&ms, gEv[i], gEv[i + 1]));
    if (gTag[i] >= 0 && gTag[i] < P_NTAG) acc[gTag[i]] += ms;
    tot += ms;
  }
  (void)accg;
  printf("---- per-category GPU time (single un-captured pass, batch %d) ----\n", gB);
  for (int t = 0; t < P_NTAG; ++t)
    if (acc[t] > 0) printf("  %-12s %8.3f ms  %5.1f%%\n", PNAME[t], acc[t], 100 * acc[t] / tot);
  printf("  %-12s %8.3f ms\n", "TOTAL", tot);
}

// ---------------------------------------------------------------- plans
struct BlkPlan { int qkv, proj, fc1, fc2; };
static std::vector<BlkPlan> BP;
static int gpPatch, gpRelW, gpRelG, gpN1, gpN2;
static void buildPlans() {
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, F = cfg.outc;
  gpPatch = makePlan(Mp, C, cfg.Kp(), CUBLASLT_EPILOGUE_DEFAULT, nullptr, true);
  BP.resize(cfg.depth);
  int qkvW = -1, qkvG = -1;   (void)qkvW; (void)qkvG;
  for (int i = 0; i < cfg.depth; ++i) {
    const int M = cfg.isGlobal(i) ? Mp : (gNatQkv ? MQ : Mw);
    BP[i].qkv  = gInt8 ? -1 : makePlan(M, 3 * Cp, C, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "qkv.b")), false);
    BP[i].proj = makePlan(Mp, C, C, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "proj.b")), true);
    BP[i].fc1  = gLtGelu ? makePlan(Mp, H4, C, CUBLASLT_EPILOGUE_GELU_BIAS, WP(bk(i, "fc1.b")), false) : -1;
    BP[i].fc2  = gInt8Fc2 ? -1 : makePlan(Mp, C, H4, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "fc2.b")), true, true);
  }
  // P[row][h][0..relN) = Q[row, h] . relt[:, 0..relN): batch over heads, shared table, fp32 accumulation
  if (gRelGemm) gpRelW = makePlan(Mw, cfg.relNw, DH, CUBLASLT_EPILOGUE_DEFAULT, nullptr, false, false, 3 * Cp, cfg.heads * cfg.relNw,
                    cfg.heads, DHP, 0, cfg.relNw, 0);
  if (gRelGemm) gpRelG = makePlan(Mp, cfg.relNg, DH, CUBLASLT_EPILOGUE_DEFAULT, nullptr, false, false, 3 * Cp, cfg.heads * cfg.relNg,
                    cfg.heads, DHP, 0, cfg.relNg, 0);
  gpN1 = makePlan(Mp, F, C, CUBLASLT_EPILOGUE_DEFAULT, nullptr, false, true, 0, 0, 1, 0, 0, 0, 0);
  gpN2 = makePlan(Mp, F, 9 * F, CUBLASLT_EPILOGUE_DEFAULT, nullptr, false, true, 0, 0, 1, 0, 0, 0, 0);
  printf("plans: %zu cuBLASLt plans%s\n", gPlans.size(), gTune ? " (tuned)" : "");
}

// ---------------------------------------------------------------- forward
// ---- w8a8_static_fc2_20261026 dumps (first eager forward only) ----
static int w8GateBlock() { const char* e = getenv("NANO_W8_GATE_BLOCK"); return e ? atoi(e) : 0; }
static void w8HostFile(const std::string& p, const void* h, size_t bytes) { std::ofstream f(p, std::ios::binary); f.write((const char*)h, bytes); }
static void w8f2sDumpPre(int i, half* X, cudaStream_t s) {
  const int C = cfg.C, H4 = 4 * C, M = Tv, G = cdiv(H4, 128);
  CK(cudaStreamSynchronize(s));
  char nm[256];
  if (!gW8F2SDump.empty()) {
    const int R = getenv("NANO_W8_DUMP_STRIDE") ? atoi(getenv("NANO_W8_DUMP_STRIDE")) : 8;
    std::vector<int8_t> hq((size_t)M * H4); std::vector<float> hs((size_t)M * G);
    CK(cudaMemcpy(hq.data(), HQ, hq.size(), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hs.data(), HS, hs.size() * 4, cudaMemcpyDeviceToHost));
    float mx = 0.f; for (float v : hs) mx = std::max(mx, v);
    const int Mr = (M + R - 1) / R;
    std::vector<int8_t> sq((size_t)Mr * H4); std::vector<float> ss((size_t)Mr * G);
    for (int r = 0; r < Mr; ++r) {
      memcpy(sq.data() + (size_t)r * H4, hq.data() + (size_t)r * R * H4, H4);
      memcpy(ss.data() + (size_t)r * G, hs.data() + (size_t)r * R * G, G * 4);
    }
    snprintf(nm, sizeof nm, "/b%02d_hq.bin", i); w8HostFile(gW8F2SDump + nm, sq.data(), sq.size());
    snprintf(nm, sizeof nm, "/b%02d_hs.bin", i); w8HostFile(gW8F2SDump + nm, ss.data(), ss.size() * 4);
    char meta[256]; snprintf(meta, sizeof meta, "{\"block\": %d, \"M\": %d, \"rows\": %d, \"stride\": %d, \"K\": %d, \"N\": %d, \"G\": %d, \"hs_max_all\": %.9g}\n",
                             i, M, Mr, R, H4, C, G, mx);
    snprintf(nm, sizeof nm, "/b%02d_meta.json", i); w8HostFile(gW8F2SDump + nm, meta, strlen(meta));
    snprintf(nm, sizeof nm, "/b%02d_w.bin", i);  writeBin((gW8F2SDump + nm).c_str(), QWfc2[i].w, (size_t)C * H4);
    snprintf(nm, sizeof nm, "/b%02d_sw.bin", i); writeBin((gW8F2SDump + nm).c_str(), QWfc2[i].s, (size_t)C * 4);
  }
  if (!gW8F2SGate.empty() && i == w8GateBlock()) {
    const std::string d = gW8F2SGate;
    writeBin((d + "/hq.bin").c_str(), HQ, (size_t)M * H4);
    if (!dF2S) writeBin((d + "/hs.bin").c_str(), HS, (size_t)M * G * 4);
    writeBin((d + "/y_res.bin").c_str(), X, (size_t)M * C * 2);
    writeBin((d + "/w.bin").c_str(), QWfc2[i].w, (size_t)C * H4);
    writeBin((d + "/sw.bin").c_str(), QWfc2[i].s, (size_t)C * 4);
    writeBin((d + "/bias.bin").c_str(), WP(bk(i, "fc2.b")), (size_t)C * 2);
    runS8<true, false, false, 0, false, 1, 2, 2, false, false, 0, true, 1>(XQ, QWfc1[i], XS, (const half*)WP(bk(i, "fc1.b")), HID, Mp, H4, C, s);
    CK(cudaStreamSynchronize(s));
    writeBin((d + "/hid.bin").c_str(), HID, (size_t)M * H4 * 2);
    float sg = 0.f, inv = 0.f;
    if (dF2S) { CK(cudaMemcpy(&sg, dF2S + i, 4, cudaMemcpyDeviceToHost)); inv = 0.125f / sg; }
    w8HostFile(d + "/sinv.bin", &inv, 4); w8HostFile(d + "/sg.bin", &sg, 4);
    char meta[256]; snprintf(meta, sizeof meta, "{\"block\": %d, \"M\": %d, \"K\": %d, \"N\": %d, \"G\": %d, \"static\": %d, \"samv1\": 1}\n", i, M, H4, C, G, (int)(dF2S != nullptr));
    w8HostFile(d + "/meta.json", meta, strlen(meta));
  }
}
static void forward(cudaStream_t s) {
  const int TH = 256, C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, F = cfg.outc, HW = cfg.HW();
  mark(P_IM2COL, s);
  k_im2col_samv1<<<cdiv(Mp * (cfg.Kp() / 4), TH), TH, 0, s>>>(dIn, COL, Mp, cfg.G, cfg.patch, cfg.img);
  mark(P_PATCH, s);
  runPlan(gpPatch, COL, WP("patch.w"), POS, X, s);
  if (!gDump.empty()) writeBin((gDump + "/x_embed.bin").c_str(), X, (size_t)HW * C * 2);
  // --dump with a quantised path (w4a4_samv1 check aid): block 0's weights as the GEMMs read them (int4: packed
  // [N, K/2]), its quantised LN1 / LN2 outputs and scales, the qkv output, the residual stream after proj, and fc2's
  // quantised operand with its (token, group) scales; Tv rows (all images; the checks run batch 1).
  const bool dump0 = !gDump.empty() && gInt8 && gInt8Fc2;
  const int dv = gInt4 ? 2 : 1;
  if (dump0) {
    writeBin((gDump + "/w_qkv_b0.bin").c_str(), QWqkv[0].w, (size_t)3 * Cp * C / dv);
    writeBin((gDump + "/s_qkv_b0.bin").c_str(), QWqkv[0].s, (size_t)3 * Cp * 4);
    writeBin((gDump + "/w_fc1_b0.bin").c_str(), QWfc1[0].w, (size_t)H4 * C / dv);
    writeBin((gDump + "/s_fc1_b0.bin").c_str(), QWfc1[0].s, (size_t)H4 * 4);
    writeBin((gDump + "/w_fc2_b0.bin").c_str(), QWfc2[0].w, (size_t)C * H4 / dv);
    writeBin((gDump + "/s_fc2_b0.bin").c_str(), QWfc2[0].s, (size_t)C * 4);
  }

  for (int i = 0; i < cfg.depth; ++i) {
    const bool glb = cfg.isGlobal(i);
    const int M = glb ? Mp : (gNatQkv ? MQ : Mw);
    const int* map = glb ? nullptr : (gNatQkv ? mNatSrcPad : mSlot2Nat);
    const float* g1 = (const float*)WP(bk(i, "ln1.w")); const float* b1 = (const float*)WP(bk(i, "ln1.b"));
    const float* g2 = (const float*)WP(bk(i, "ln2.w")); const float* b2 = (const float*)WP(bk(i, "ln2.b"));
    mark(P_LN, s);
    if (gInt4)      launchLNQ4(X, XQ, XS, g1, b1, M, C, s, map);
    else if (gInt8) launchLNQ8(X, XQ, XS, g1, b1, M, C, s, map);
    else       launchLN(X, Xn, g1, b1, M, C, s, map);
    if (dump0 && i == 0) {
      writeBin((gDump + "/xq_ln1_b0.bin").c_str(), XQ, (size_t)Tv * C / dv);
      writeBin((gDump + "/xs_ln1_b0.bin").c_str(), XS, (size_t)Tv * 4);
    }
    mark(P_QKV, s);
    if (gInt4)      runQkv4(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, M, 3 * Cp, C, s);
    else if (gInt8) runS8<false>(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, M, 3 * Cp, C, s);
    else       runPlan(BP[i].qkv, Xn, WP(bk(i, "qkv.w")), nullptr, QKV, s);
    if (dump0 && i == 0 && (glb || gNatQkv)) writeBin((gDump + "/qkv_b0.bin").c_str(), QKV, (size_t)Tv * 3 * Cp * 2);
    if (gRelGemm) {
      mark(P_REL, s);
      runPlan(glb ? gpRelG : gpRelW, QKV, WP(bk(i, "relt")), nullptr, REL, s);
    }
    mark(glb ? P_ATTNG : P_ATTN, s);
    runAttnRP(glb, QKV, (glb || gNatQkv) ? Xn : AO, gRelGemm ? REL : gRelT[i], s);
    if (!glb && !gNatQkv) {
      mark(P_GATHER, s);
      k_gather<<<cdiv(Mp * (C / 8), TH), TH, 0, s>>>(AO, Xn, mNat2Slot, Mp, C);
    }
    mark(P_PROJ, s);
    runPlan(BP[i].proj, Xn, WP(bk(i, "proj.w")), X, X, s);
    if (dump0 && i == 0) writeBin((gDump + "/x_mid_b0.bin").c_str(), X, (size_t)Tv * C * 2);
    mark(P_LN, s);
    if (gInt4)      launchLNQ4(X, XQ, XS, g2, b2, Mp, C, s);
    else if (gInt8) launchLNQ8(X, XQ, XS, g2, b2, Mp, C, s);
    else       launchLN(X, Xn, g2, b2, Mp, C, s);
    if (dump0 && i == 0) {
      writeBin((gDump + "/xq_ln2_b0.bin").c_str(), XQ, (size_t)Tv * C / dv);
      writeBin((gDump + "/xs_ln2_b0.bin").c_str(), XS, (size_t)Tv * 4);
    }
    mark(P_FC1, s);
    const half* fb1 = (const half*)WP(bk(i, "fc1.b"));
    if (gInt4 && dF2S) {
      runFc14S(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, HQ, dF2S + i);
    } else if (gInt4) {
      runFc14(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, HQ, HS);
    } else if (gInt8 && gInt8Fc2 && dF2S) {
      constexpr size_t sh = s8_smem<128, 128, 64, 3>();
      auto k = k_gemm_s8<128, 128, 64, 2, 2, 3, true, true, false, false, 0, true, 1, false, false, 64, 1, false, false, false, false, false, true>;
      static bool attr = false;
      if (!attr) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
      k<<<dim3(cdiv(H4, 128), Mp / 128), 128, sh, s>>>(XQ, QWfc1[i].w, XS, QWfc1[i].s, fb1, HID, Mp, H4, C, nullptr, HQ, dF2S + i, 0, nullptr, 0);
    } else if (gInt8 && gInt8Fc2) {
      if (gRot && gFc2Rot == 128) runS8<true, false, false, 0, true, 1, 2, 4, false, false, 128>(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, nullptr, HQ, HS);
      else if (gRot && gArch89)   runS8<true, false, false, 0, true, 1, 2, 2, false, false, 64, true, 1>(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, nullptr, HQ, HS);
      else if (gRot)              runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64>(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, nullptr, HQ, HS);
      else                        runS8<true, false, false, 0, true, 1>(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s, nullptr, HQ, HS);
    } else if (gInt8) {
      runS8<true>(XQ, QWfc1[i], XS, fb1, HID, Mp, H4, C, s);
    } else if (!gLtGelu) {
      runFc1Gelu(Xn, (const half*)WP(bk(i, "fc1.w")), fb1, HID, Mp, H4, C, gAccMode >= 1, s);
    } else {
      runPlan(BP[i].fc1, Xn, WP(bk(i, "fc1.w")), nullptr, HID, s);
    }
    if (dump0 && i == 0) {
      writeBin((gDump + "/hq_b0.bin").c_str(), HQ, (size_t)Tv * H4 / dv);
      writeBin((gDump + "/hs_b0.bin").c_str(), HS, (size_t)Tv * cdiv(H4, 128) * 4);
    }
    if (gW8First && gInt8 && gInt8Fc2 && !gInt4 && (!gW8F2SDump.empty() || !gW8F2SGate.empty())) w8f2sDumpPre(i, X, s);
    mark(P_FC2, s);
    if (!gInt4 && gInt8 && gInt8Fc2 && dF2S)
      runS8<false, true, true, 0, false, 1, 2, 4, false, false, 0, false, 0, 128, 192>(HQ, QWfc2[i], dF2S + i, (const half*)WP(bk(i, "fc2.b")), X, Mp, C, H4, s, X);
    else if (gInt4 && dF2S)     runFc24S(HQ, QWfc2[i], dF2S + i, (const half*)WP(bk(i, "fc2.b")), X, Mp, C, H4, s);
    else if (gInt4)             runFc24(HQ, QWfc2[i], HS, (const half*)WP(bk(i, "fc2.b")), X, Mp, C, H4, s, cdiv(H4, 128));
    else if (gInt8 && gInt8Fc2) runFc2S8(HQ, QWfc2[i], HS, (const half*)WP(bk(i, "fc2.b")), X, Mp, C, H4, s, cdiv(H4, 128));
    else                   runPlan(BP[i].fc2, HID, WP(bk(i, "fc2.w")), X, X, s);
    if (gW8First && gInt8 && gInt8Fc2 && !gInt4 && !gW8F2SGate.empty() && i == w8GateBlock()) {
      CK(cudaStreamSynchronize(s)); writeBin((gW8F2SGate + "/y_out.bin").c_str(), X, (size_t)Tv * C * 2);
    }
    if (!gDump.empty()) {
      char nm[64]; snprintf(nm, sizeof nm, "/blk%02d.bin", i);
      writeBin((gDump + nm).c_str(), X, (size_t)HW * C * 2);
    }
  }
  mark(P_NECK, s);
  runPlan(gpN1, X, WP("neck.c1.w"), nullptr, N1, s);
  launchLN(N1, N1n, (const float*)WP("neck.ln1.w"), (const float*)WP("neck.ln1.b"), Mp, F, s);
  k_im2col3x3_b<<<cdiv(Mp, 8), 256, 0, s>>>(N1n, NCOL, cfg.G, cfg.G, F, Tv, Mp);
  runPlan(gpN2, NCOL, WP("neck.c2.w"), nullptr, N2, s);
  launchLN(N2, N2n, (const float*)WP("neck.ln2.w"), (const float*)WP("neck.ln2.b"), Mp, F, s);
  k_to_nchw_b<<<dim3(cdiv(HW, 32), cdiv(F, 32), gB), dim3(32, 8), 0, s>>>(N2n, dOut, HW, F);
  mark(-1, s);
}

// ---------------------------------------------------------------- driver
static size_t readFile(const char* path, std::vector<char>& h) {
  std::ifstream f(path, std::ios::binary);
  if (!f) { printf("cannot read %s\n", path); exit(1); }
  f.seekg(0, std::ios::end); size_t n = (size_t)f.tellg(); f.seekg(0);
  h.resize(n); f.read(h.data(), n); return n;
}
static std::string baseName(const std::string& p) {
  size_t a = p.find_last_of("/\\"); std::string b = a == std::string::npos ? p : p.substr(a + 1);
  size_t d = b.find_last_of('.'); return d == std::string::npos ? b : b.substr(0, d);
}

static std::string gModel = "models/sam_vit_h", gInput, gInputList, gOutDir;
static int gWarmup = 10, gIters = 50, gUseGraph = 1;
static bool gServe = false;
static cudaStream_t gS;
static cudaGraphExec_t gExec = nullptr;

static int setup(int argc, char** argv) {
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) gModel = argv[++i];
    else if (a == "--batch" && i + 1 < argc) gB = atoi(argv[++i]);
    else if (a == "--input" && i + 1 < argc) gInput = argv[++i];
    else if (a == "--input-list" && i + 1 < argc) gInputList = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) gOutDir = argv[++i];
    else if (a == "--dump" && i + 1 < argc) gDump = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) gWarmup = atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) gIters = atoi(argv[++i]);
    else if (a == "--no-graph") gUseGraph = 0;
    else if (a == "--f16acc") gAccMode = 2;
    else if (a == "--f16acc-shallow") gAccMode = 1;
    else if (a == "--attn-acc" && i + 1 < argc) gAttnAcc = atoi(argv[++i]);
    else if (a == "--attn-nw" && i + 1 < argc) gAttnNW = atoi(argv[++i]);
    else if (a == "--rel-gemm") { gRelGemm = true; gNatQkv = false; }
    else if (a == "--slot-qkv") gNatQkv = false;
    else if (a == "--fc2-tile" && i + 1 < argc) gFc2Tile = atoi(argv[++i]);
    else if (a == "--attn-minb" && i + 1 < argc) gAttnMinB = atoi(argv[++i]);
    else if (a == "--int8") gInt8 = true;
    else if (a == "--int8-fc2") { gInt8 = true; gInt8Fc2 = true; }
    else if (a == "--int4") { gInt4 = true; gInt8 = true; gInt8Fc2 = true; gRot = true; }
    else if (a == "--serve") gServe = true;
    else if (a == "--rot") { gInt8 = true; gRot = true; }
    else if (a == "--fc2-rot" && i + 1 < argc) gFc2Rot = atoi(argv[++i]);
    else if (a == "--gptq" && i + 1 < argc) { gInt8 = true; gRot = true; gGptq = argv[++i]; }
    else if (a == "--fc2-static" && i + 1 < argc) gF2SFile = argv[++i];
    else if (a == "--w8-f2s-dump" && i + 1 < argc) gW8F2SDump = argv[++i];
    else if (a == "--w8-f2s-gate" && i + 1 < argc) gW8F2SGate = argv[++i];
    else if (a == "--lt-gelu") gLtGelu = true;
    else if (a == "--profile") gProf = true;
    else if (a == "--no-tune") gTune = false;
    else { printf("unknown arg %s\n", argv[i]); return 1; }
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  gArch80 = prop.major >= 8; gArch89 = prop.major == 8 && prop.minor == 9;
  if (const char* e = getenv("NANO_ARCH89")) { gArch89 = atoi(e) != 0; printf("NANO_ARCH89 override: gArch89=%d\n", (int)gArch89); }
  if (!gArch80) { printf("nano_samv1 needs sm_80+\n"); return 1; }
  if (gRot && gInt8Fc2 && gFc2Rot != 64 && gFc2Rot != 128) { printf("--fc2-rot must be 64 or 128\n"); return 1; }
  if (gInt4) {
    if (prop.major != 8) { printf("--int4 needs the sm_80/86/89 INT4 mma (mma.m16n8k64.s4); this is sm_%d%d\n", prop.major, prop.minor); return 1; }
    if (gFc2Rot != 64) { printf("--int4 runs the fused H_64 fc1 producer only (no --fc2-rot 128)\n"); return 1; }
    if (gRelGemm || !gNatQkv) { printf("--int4 runs the default natural-order qkv layout only\n"); return 1; }
  }
  printf("GPU: %s sm_%d%d, SMs=%d, accmode=%d attn-acc=%d rel=%s gemms=%s%s%s\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount, gAccMode, gAttnAcc >= 0 ? gAttnAcc : 1, gRelGemm ? "gemm" : gNatQkv ? "in-kernel, natural-order qkv" : "in-kernel, slot-order qkv",
         gInt4 ? "int4(qkv+fc1+fc2)" : gInt8Fc2 ? "int8(qkv+fc1+fc2)" : gInt8 ? "int8(qkv+fc1)" : "fp16",
         gRot ? " rot" : "", gGptq.empty() ? "" : " gptq");
  loadCfg(gModel);
  loadWeights(gModel);
  CL(cublasLtCreate(&LT));
  CK(cudaMalloc(&gWs, gWsBytes));
  allocAll();
  buildPlans();
  freeTuneScratch();
  if (gInt8) buildInt8Weights();
  if (!gF2SFile.empty()) {     // samv1_static_fc2
    if (!gInt4 && !(gInt8Fc2 && gRot && gFc2Rot == 64 && gArch89)) { printf("--fc2-static needs --int4, or the sm_89 W8A8 path (--int8-fc2 --rot/--gptq, H_64)\n"); return 1; }
    if (!gW8F2SDump.empty()) { printf("--w8-f2s-dump is a dump of the DYNAMIC path\n"); return 1; }
    std::vector<char> h; size_t n = readFile(gF2SFile.c_str(), h);
    int dep = 0; if (n >= 12) memcpy(&dep, h.data() + 8, 4);
    if (n < 12 || memcmp(h.data(), "NANOF2S1", 8) || dep != cfg.depth || n != 12 + 4 * (size_t)dep) {
      printf("--fc2-static %s: bad file (size %zu, depth %d, model depth %d)\n", gF2SFile.c_str(), n, dep, cfg.depth); return 1; }
    std::vector<float> sv(dep); memcpy(sv.data(), h.data() + 12, 4 * (size_t)dep);
    float mn = 1e30f, mx = 0.f;
    for (float v : sv) { if (!(v > 0.f) || !std::isfinite(v)) { printf("--fc2-static: scale %g not positive\n", v); return 1; } mn = std::min(mn, v); mx = std::max(mx, v); }
    CK(cudaMalloc(&dF2S, 4 * (size_t)dep)); CK(cudaMemcpy(dF2S, sv.data(), 4 * (size_t)dep, cudaMemcpyHostToDevice));
    printf("fc2-static: %d static fc2 scales from %s (min %.6g max %.6g)\n", dep, gF2SFile.c_str(), mn, mx);
  }
  if (!gInput.empty()) {
    std::vector<char> h; size_t n = readFile(gInput.c_str(), h);
    const size_t inBytes = (size_t)gB * 3 * cfg.img * cfg.img * sizeof(float);
    size_t off = n > inBytes ? n - inBytes : 0, cp = std::min(n - off, inBytes);
    CK(cudaMemcpy(dIn, h.data() + off, cp, cudaMemcpyHostToDevice));
  }
  CK(cudaStreamCreate(&gS));
  forward(gS); CK(cudaStreamSynchronize(gS));
  CK(cudaGetLastError());
  gDump.clear(); gW8First = false;
  if (gProf) {
    for (int r = 0; r < 3; ++r) { gEvN = 0; forward(gS); CK(cudaStreamSynchronize(gS)); }
    profReport();
    gProf = false;
  }
  if (gUseGraph) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(gS, cudaStreamCaptureModeThreadLocal));
    forward(gS);
    CK(cudaStreamEndCapture(gS, &g));
    CK(cudaGraphInstantiate(&gExec, g, nullptr, nullptr, 0));
    CK(cudaGraphDestroy(g));
  }
  fflush(stdout);       // the ctypes library shares stdout with python
  return 0;
}
static void once() { if (gExec) CK(cudaGraphLaunch(gExec, gS)); else forward(gS); }

#ifdef SAMV1_LIB
#ifdef _WIN32
#define SAMV1_API extern "C" __declspec(dllexport)
#else
#define SAMV1_API extern "C"
#endif
static std::vector<std::string> gArgStore;
SAMV1_API int samv1_init(const char* args) {
  std::istringstream ss(args); std::string t;
  while (ss >> t) gArgStore.push_back(t);
  std::vector<char*> av; for (auto& a : gArgStore) av.push_back((char*)a.c_str());
  return setup((int)av.size(), av.data());
}
SAMV1_API int samv1_batch() { return gB; }
// in: device fp32 [B,3,1024,1024]; out: device fp32 [B,256,64,64]; synchronous
SAMV1_API int samv1_run(const float* in, float* out) {
  CK(cudaMemcpyAsync(dIn, in, (size_t)gB * 3 * cfg.img * cfg.img * sizeof(float), cudaMemcpyDeviceToDevice, gS));
  once();
  CK(cudaMemcpyAsync(out, dOut, (size_t)gB * cfg.outc * cfg.HW() * sizeof(float), cudaMemcpyDeviceToDevice, gS));
  CK(cudaStreamSynchronize(gS));
  return 0;
}
#else
int main(int argc, char** argv) {
  if (setup(argc - 1, argv + 1)) return 1;
  const size_t inBytes = (size_t)gB * 3 * cfg.img * cfg.img * sizeof(float);
  const size_t outBytes = (size_t)gB * cfg.outc * cfg.HW() * sizeof(float);
  std::vector<char> hin, hout(outBytes);
  if (gServe) {
    // w4a4_samv1 accuracy harness: one request per stdin line "<in.bin> <out dir>": the file (exactly B x 3 x 1024 x
    // 1024 fp32) -> dIn, one forward (the CUDA graph, as the timing runs), <out dir>/ours_emb.bin = fp32 [B,256,64,64]
    printf("SERVE_READY batch %d\n", gB); fflush(stdout);
    std::string line; int nreq = 0;
    while (std::getline(std::cin, line)) {
      std::istringstream ls(line); std::string in, od;
      if (!(ls >> in >> od)) continue;
      const size_t n = readFile(in.c_str(), hin);
      if (n != inBytes) { printf("SERVE_ERROR %s: %zu bytes, expected %zu\n", in.c_str(), n, inBytes); fflush(stdout); return 1; }
      CK(cudaMemcpy(dIn, hin.data(), inBytes, cudaMemcpyHostToDevice));
      once(); CK(cudaStreamSynchronize(gS)); CK(cudaGetLastError());
      CK(cudaMemcpy(hout.data(), dOut, outBytes, cudaMemcpyDeviceToHost));
      std::ofstream of(od + "/ours_emb.bin", std::ios::binary); of.write(hout.data(), outBytes); of.close();
      if (!of) { printf("SERVE_ERROR cannot write %s/ours_emb.bin\n", od.c_str()); fflush(stdout); return 1; }
      printf("SERVED %d\n", ++nreq); fflush(stdout);
    }
    return 0;
  }
  if (!gInputList.empty()) {
    if (gOutDir.empty()) { printf("--input-list needs --out-dir\n"); return 1; }
    std::ifstream lf(gInputList); std::string line; int nfile = 0;
    while (std::getline(lf, line)) {
      if (line.empty()) continue;
      size_t n = readFile(line.c_str(), hin);
      size_t off = n > inBytes ? n - inBytes : 0, cp = std::min(n - off, inBytes);
      CK(cudaMemcpy(dIn, hin.data() + off, cp, cudaMemcpyHostToDevice));
      if (cp < inBytes) CK(cudaMemset((char*)dIn + cp, 0, inBytes - cp));
      once(); CK(cudaStreamSynchronize(gS));
      CK(cudaMemcpy(hout.data(), dOut, outBytes, cudaMemcpyDeviceToHost));
      std::ofstream of(gOutDir + "/" + baseName(line) + ".emb.bin", std::ios::binary);
      of.write(hout.data(), outBytes);
      ++nfile;
    }
    printf("wrote embeddings for %d batch files to %s\n", nfile, gOutDir.c_str());
    return 0;
  }
  for (int i = 0; i < gWarmup; ++i) once();
  CK(cudaStreamSynchronize(gS));
  std::vector<float> ts;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for (int i = 0; i < gIters; ++i) {
    CK(cudaEventRecord(e0, gS)); once(); CK(cudaEventRecord(e1, gS));
    CK(cudaEventSynchronize(e1));
    float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  double sum = 0; for (float t : ts) sum += t;
  printf("latency_ms/batch mean=%.4f p50=%.4f min=%.4f  per_image=%.4f ms  %.2f img/s  (batch %d, iters %d, graph %d)\n",
         sum / ts.size(), ts[ts.size() / 2], ts[0], ts[ts.size() / 2] / gB, 1000.0 * gB / ts[ts.size() / 2], gB, gIters, gUseGraph);
  if (!gOutDir.empty()) {
    CK(cudaMemcpy(hout.data(), dOut, outBytes, cudaMemcpyDeviceToHost));
    std::ofstream of(gOutDir + "/out.emb.bin", std::ios::binary); of.write(hout.data(), outBytes);
    printf("dumped embeddings to %s/out.emb.bin\n", gOutDir.c_str());
  }
  return 0;
}
#endif
