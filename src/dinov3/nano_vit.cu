// nano_vit.cu — plain-ViT front-end of the nano engine (timm DeiT / ViT-augreg / ... at 224 px).
//
// The same kernels that run the SAM 2.1 hiera-L encoder (nano_sam21.cu), on a model with no
// windows, no pooling and a cls token: a LayerNorm that emits INT8 plus a per-token scale
// (block-Hadamard rotated), INT8 GEMMs with the dequant / bias / GELU fused into the epilogue,
// the fc1 epilogue emitting fc2's INT8 operand with one scale per (row, 128-column group),
// flash attention on raw mma, cuBLASLt for the fp16 GEMMs, everything in one CUDA graph.
//
// Batched: B images x S tokens live in ONE compact [B*S, C] token buffer (S = 197 for a
// /16 model at 224 px with a cls token).  The GEMMs and LayerNorms see M = B*S rows padded up
// to a multiple of 128; the attention kernel treats every image as one "window" of S keys and
// masks the keys past S (the rows it reads belong to the next image, or to 64 slack rows).
//
// build: nvcc -O3 -std=c++17 -gencode arch=compute_86,code=sm_86 nano_vit.cu -lcublasLt -o nano_vit.exe
// run:   nano_vit.exe --model vit/deit_base_patch16_224 --batch 32 --f16acc --int8-fc2 --rot
//        --input-list chunks.txt --out-dir out/      (uint8 NCHW batches -> fp16 logits per file)

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
#include <algorithm>

#define CK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
  printf("CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e_)); exit(1);} }while(0)
#define CL(x) do{ cublasStatus_t s_=(x); if(s_!=CUBLAS_STATUS_SUCCESS){ \
  printf("cublasLt %s:%d status=%d\n",__FILE__,__LINE__,(int)s_); exit(1);} }while(0)

// ---------------------------------------------------------------- model constants
// Head dimension of every plain ViT we run (DeiT-S/B, ViT-B/L, DINOv3): 64, unpadded.
static const int DH  = 64;
static const int DHP = 64;

#include "sam21_kernels.cuh"
#include "sam21_attn3.cuh"    // wide-query fp16-accumulator attention for DH = 64
#include "sam21_gemm_res.cuh"   // sm_120: fp16 GEMM + bias + residual (proj, fp16 fc2)
#include "sam21_int4.cuh"       // --int4 (W4A4 prototype): k_ln_q4, k_quant_w4, k_pack_s4

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }
static inline int roundup(int a, int b) { return cdiv(a, b) * b; }

// ---------------------------------------------------------------- model config
struct Cfg {
  std::string name;
  int C = 768, heads = 12, dh = 64, dhp = 64, depth = 12, S = 197, npatch = 196, patch = 16,
      img = 224, hidden = 3072, ncls = 1000, prefix = 1, rope = 0;
  float eps = 1e-6f, mean[3] = {0.485f, 0.456f, 0.406f}, stdv[3] = {0.229f, 0.224f, 0.225f};
  int Cp() const { return heads * dhp; }
  int Kp() const { return 3 * patch * patch; }
};
static Cfg cfg;

static double jnum(const std::string& js, const char* key, double dflt) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) return dflt;
  p = js.find(':', p); return atof(js.c_str() + p + 1);
}
static void jarr3(const std::string& js, const char* key, float* out) {
  std::string k = std::string("\"") + key + "\"";
  size_t p = js.find(k); if (p == std::string::npos) return;
  p = js.find('[', p) + 1;
  for (int i = 0; i < 3; ++i) { out[i] = (float)atof(js.c_str() + p); p = js.find(',', p) + 1; }
}
static void loadCfg(const std::string& prefix) {
  std::ifstream f(prefix + ".json");
  if (!f) { printf("cannot open %s.json\n", prefix.c_str()); exit(1); }
  std::stringstream ss; ss << f.rdbuf(); std::string js = ss.str();
  size_t p = js.find("\"name\""); p = js.find('"', js.find(':', p)) + 1;
  cfg.name = js.substr(p, js.find('"', p) - p);
  cfg.C = (int)jnum(js, "C", 768); cfg.heads = (int)jnum(js, "heads", 12);
  cfg.dh = (int)jnum(js, "dh", 64); cfg.dhp = (int)jnum(js, "dhp", 64);
  cfg.depth = (int)jnum(js, "depth", 12); cfg.S = (int)jnum(js, "S", 197);
  cfg.npatch = (int)jnum(js, "npatch", 196); cfg.patch = (int)jnum(js, "patch", 16);
  cfg.img = (int)jnum(js, "img", 224); cfg.hidden = (int)jnum(js, "hidden", 4 * cfg.C);
  cfg.ncls = (int)jnum(js, "ncls", 1000); cfg.eps = (float)jnum(js, "eps", 1e-6);
  cfg.prefix = (int)jnum(js, "prefix", 1); cfg.rope = (int)jnum(js, "rope", 0);      // DINOv3: 5 prefix tokens, RoPE
  if (cfg.S != cfg.prefix + cfg.npatch) { printf("config: S %d != prefix %d + npatch %d\n", cfg.S, cfg.prefix, cfg.npatch); exit(1); }
  jarr3(js, "mean", cfg.mean); jarr3(js, "std", cfg.stdv);
  if (cfg.dh != DH || cfg.dhp != DHP) { printf("this build is for head dim %d (model has %d)\n", DH, cfg.dh); exit(1); }
  if (cfg.hidden != 4 * cfg.C) { printf("mlp ratio must be 4 (hidden %d, C %d)\n", cfg.hidden, cfg.C); exit(1); }
  if (cfg.C % 128 != 0 && cfg.C != 192) { printf("unsupported width %d\n", cfg.C); exit(1); }
  printf("model %s: C=%d heads=%d dh=%d depth=%d S=%d (%d prefix%s) patch=%d img=%d ncls=%d eps=%g mean=%.3f/%.3f/%.3f\n",
         cfg.name.c_str(), cfg.C, cfg.heads, cfg.dh, cfg.depth, cfg.S, cfg.prefix, cfg.rope ? ", RoPE" : "", cfg.patch, cfg.img,
         cfg.ncls, cfg.eps, cfg.mean[0], cfg.mean[1], cfg.mean[2]);
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

// ---------------------------------------------------------------- int8 weights (as nano_sam21.cu)
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
__global__ void k_wsum(const int8_t* __restrict__ wt, int K, int N, float* __restrict__ out) {
  const int n = blockIdx.x;
  long long acc = 0;
  for (int k = threadIdx.x; k < K; k += blockDim.x) acc += wt[(size_t)n * K + k];
  __shared__ long long red[32];
  for (int o = 16; o; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = acc;
  __syncthreads();
  if (threadIdx.x == 0) {
    long long t = 0;
    for (int i = 0; i < (blockDim.x + 31) / 32; ++i) t += red[i];
    out[n] = (float)t;
  }
}
__global__ void k_bias_shift(const half* __restrict__ bias, const float* __restrict__ sw,
                             const float* __restrict__ wsum, float shift, int N,
                             half* __restrict__ out) {
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) out[n] = __float2half(__half2float(bias[n]) - shift * sw[n] * wsum[n]);
}

// ---------------------------------------------------------------- ViT-specific kernels
// im2col of the p x p / stride p patch conv straight from the uint8 NCHW batch, normalised
// with the model's mean/std, into the compact token layout: row b*S + 1 + patch; the cls row
// (b*S) and the rows past the batch are zero so the patch GEMM leaves only the pos table there.
__global__ void k_im2col_vit(const uint8_t* __restrict__ img, half* __restrict__ col, int B, int S,
                             int G, int P, int IMG, int Mp, int prefix, float m0, float m1, float m2,
                             float i0, float i1, float i2) {
  const int K = 3 * P * P;
  int t = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (t >= Mp) return;
  int lane = threadIdx.x & 31;
  int b = t / S, j = t - b * S;
  half* dst = col + (size_t)t * K;
  const bool valid = b < B && j >= prefix;
  int q = j - prefix, py = q / G, px = q - py * G;
  for (int k = lane; k < K; k += 32) {
    float v = 0.f;
    if (valid) {
      int c = k / (P * P), r = k - c * P * P, ky = r / P, kx = r - ky * P;
      float mean = c == 0 ? m0 : c == 1 ? m1 : m2, inv = c == 0 ? i0 : c == 1 ? i1 : i2;
      v = (img[((size_t)(b * 3 + c) * IMG + py * P + ky) * IMG + px * P + kx] * (1.f / 255.f) - mean) * inv;
    }
    dst[k] = __float2half(v);
  }
}
// RoPE on Q and K of the patch tokens (prefix tokens untouched), timm's rotate-half convention:
// tab[p] = sin[0..dh) | cos[0..dh); out[d] = x[d] cos[d] - x[d+dh/2] sin[d], out[d+dh/2] = x[d+dh/2] cos[d+dh/2] + x[d] sin[d+dh/2].
// One thread per (token, head, pair); Q was pre-scaled by 1/sqrt(dh) in the weights, which commutes.
__global__ void k_rope(half* __restrict__ qkv, const float* __restrict__ tab, int B, int S, int prefix,
                       int heads, int Cp, int Mp) {
  constexpr int H2 = DH / 2;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int d = idx % H2, rest = idx / H2, h = rest % heads, t = rest / heads;
  if (t >= Mp) return;
  int b = t / S, j = t - b * S;
  if (b >= B || j < prefix) return;
  const float* row = tab + (size_t)(j - prefix) * (2 * DH);
  const float s0 = row[d], s1 = row[d + H2], c0 = row[DH + d], c1 = row[DH + d + H2];
#pragma unroll
  for (int part = 0; part < 2; ++part) {           // Q, then K
    half* x = qkv + (size_t)t * 3 * Cp + part * Cp + h * DHP;
    const float a = __half2float(x[d]), c = __half2float(x[d + H2]);
    x[d] = __float2half(a * c0 - c * s0);
    x[d + H2] = __float2half(c * c1 + a * s1);
  }
}
// the [S, C] per-token table (cls + pos, pos + patch bias) replicated over the batch
__global__ void k_pos_rep(const half* __restrict__ pos, half* __restrict__ out, int B, int S, int C, int Mp) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= (size_t)Mp * (C / 8)) return;
  int t = (int)(i / (C / 8)), c8 = (int)(i % (C / 8));
  float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
  if (t < B * S) v = ((const float4*)(pos + (size_t)(t % S) * C))[c8];
  ((float4*)(out + (size_t)t * C))[c8] = v;
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
static int   gAccMode = 0;        // 0 = fp32 acc, 1 = fp16 acc except fc2, 2 = fp16 acc
static bool  gF16Acc = false;
static int   gAttnAcc = -1;
static bool  gArch80 = false, gArch89 = false;
static bool  gArch120 = false;    // sm_120 (RTX 5090, 170 SMs): per-kernel choices, see the gArch120 sites
static bool  gInt8 = false, gInt8Fc2 = false, gRot = false, gFc2U8 = true, gLtGelu = false;
static int   gFc2Rot = 64;
// --fc2-bfp: power-of-two fc2 group scale carried as a byte exponent (sam21_gemm_s8.cuh)
static bool  gFc2Bfp = false;
static int   gBfpEmax = 0;            // -1 = no guard kernel, 0 = the shape's safe bound
static bool  gBfpCheck = false;
// --fc2-pass: fc1 writes its plain fp16 GELU output and a stand-alone kernel (k_rotquant_fc2)
// rotates and quantises it into fc2's int8 operand, instead of the fused ROT8 / Q8OUT epilogue.
// Same rotation, scales and layout; measures what a separate rotate-and-quantise pass costs.
static bool  gFc2Pass = false;
// --int4 (W4A4 prototype, w4a4_engine): qkv, fc1, fc2 on mma.m16n8k64 s4 with packed int4 operands; implies the
// INT8 path's structure (--int8-fc2 --rot): LN -> rotated per-token int4, fc1 epilogue -> H_64 + per-(token, 128) int4,
// weights per output channel 4-bit (RTN at start-up, or the WBITS=4 GPTQ codes of --gptq).  proj / attention fp16.
static bool  gInt4 = false;
static int8_t* gW4tmp = nullptr;    // --int4: int8 codes [N, K] before packing
// ---- dinov3_engine_20261027 (all off by default; without these flags the code path is w4a4_engine's) ----
// --fc2-static <file> (needs --int4): ONE static symmetric fc2-input scale s per block (NANOF2S1, mode 1).  fc1 epilogue:
//   codes clamp(rint(v * (0.125 / s)), -7, 7) of the H_64-rotated GELU tile (k_gemm_s8 F2S), no row max, no scale store;
//   fc2: the INT4 GEMM with GROUP = 0 + SASCALAR (one int32 accumulator, epilogue acc * (s * sw[col])) on the dynamic
//   fc2's tile rule.  No group-wise activation scale anywhere on this path.
// --fc2-fp16-b0 (needs --int4): block 0's fc1 writes its fp16 GELU output (INT4 GEMM, plain epilogue) and block 0's fc2
//   runs the fp16 path's GEMM (the cuBLASLt plan of the fp16 engine, unrotated fp16 weights) -- the fp16 exemption.
// NANO_DUMP_BLOCK=<b> (env, --dump only): the per-block dumps of --dump refer to block b instead of 0 (file names keep
//   "_b0"); --dump also writes x_in_b0.bin (block input) and, on the --fc2-static path, hid_b0.bin = the fp16 GELU output
//   of the same fc1 tile (one extra plain-epilogue launch, first eager pass only) and f2s_b0.bin = {0.125/s, s}.
static std::string gFc2StaticFile;
static bool  gFc2Fp16B0 = false;
static std::vector<float*> gF2Sinv, gF2Sg;    // per block: device float[1] (0.125 / s), device float[1] (s)
static int   gDumpBlk = 0;
static std::string gGptq, gDump;
static bool  gTune = true, gProf = false;
static half *gTA = nullptr, *gTB = nullptr, *gTD = nullptr;

struct Plan {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t La, Lb, Lc;
  cublasLtMatmulAlgo_t algo;
  int m, n, k;
  bool beta1, f16acc;
};
static std::vector<Plan> gPlans;

static int makePlan(int m, int n, int k, cublasLtEpilogue_t epi, const void* bias, bool beta1,
                    bool deepK = false) {
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
  CL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &gWsBytes, sizeof(gWsBytes)));
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
    for (int c = 0; c < found; ++c) {
      cublasStatus_t st2 = cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc,
                                          gTD, p.Lc, &hr[c].algo, gWs, gWsBytes, 0);
      if (st2 != CUBLAS_STATUS_SUCCESS) continue;
      for (int r = 0; r < 3; ++r)
        cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc,
                       &hr[c].algo, gWs, gWsBytes, 0);
      CK(cudaDeviceSynchronize());
      CK(cudaEventRecord(e0));
      for (int r = 0; r < 12; ++r)
        cublasLtMatmul(LT, p.op, al, gTB, p.La, gTA, p.Lb, be, gTD, p.Lc, gTD, p.Lc,
                       &hr[c].algo, gWs, gWsBytes, 0);
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
static int gB = 32, gMtok = 0, gMp = 0, gMA = 0, gBp = 0;
static half *X, *Xn, *QKV, *HID, *COL, *POS, *CLS, *CLSn, *LOG;
static int8_t *XQ = nullptr, *HQ = nullptr;
static float *XS = nullptr, *HS = nullptr;
static uint8_t *HE = nullptr;   // --fc2-bfp: the group scales as byte exponents
static uint8_t* dIn = nullptr;
static int* mCls = nullptr;
struct Q8W { int8_t* w; float* s; };
static std::vector<Q8W> QWqkv, QWfc1, QWfc2;
static std::vector<half*> Bfc2;
static half* gRotTmp = nullptr;

// every activation buffer carries 64 KB of zeroed slack: some cuBLASLt heuristic kernels over-read
// their last tile (seen as an illegal address on the 3090 with K = 96, tuning on), and the attention
// reads whole key tiles past a window
static half* dAlloc(size_t n) { void* p; CK(cudaMalloc(&p, n * sizeof(half) + 65536)); CK(cudaMemset(p, 0, n * sizeof(half) + 65536)); return (half*)p; }

static void allocAll() {
  const int C = cfg.C, S = cfg.S, Cp = cfg.Cp(), H4 = cfg.hidden;
  // Small batches round to 64 instead of 128: a 261-token DINOv3 image then pads to 320 rather
  // than 384, and every tile used at gMp <= 512 is 64 rows or narrower (1.10x on DINOv3 at batch 1).
  // Batched shapes keep the 128 rounding, so their outputs stay bit-identical.
  gMtok = gB * S; gMp = (gMtok <= 512) ? roundup(gMtok, 64) : roundup(gMtok, 128); gMA = gMp + 64; gBp = roundup(gB, 8);
  X   = dAlloc((size_t)gMA * C);  Xn = dAlloc((size_t)gMA * C);
  QKV = dAlloc((size_t)gMA * 3 * Cp);
  HID = dAlloc((size_t)gMp * H4);
  COL = dAlloc((size_t)gMp * cfg.Kp());
  POS = dAlloc((size_t)gMp * C);
  CLS = dAlloc((size_t)gBp * C); CLSn = dAlloc((size_t)gBp * C);
  LOG = dAlloc((size_t)gBp * roundup(cfg.ncls, 8));
  CK(cudaMalloc(&dIn, (size_t)gB * 3 * cfg.img * cfg.img));
  CK(cudaMemset(dIn, 0, (size_t)gB * 3 * cfg.img * cfg.img));
  gTA = dAlloc((size_t)gMp * std::max(H4, cfg.Kp())); gTB = dAlloc((size_t)std::max(H4, cfg.Kp()) * std::max(H4, 3 * Cp)); gTD = dAlloc((size_t)gMp * std::max(H4, 3 * Cp));
  if (gInt8) {
    CK(cudaMalloc(&XQ, (size_t)gMA * C)); CK(cudaMemset(XQ, 0, (size_t)gMA * C));
    CK(cudaMalloc(&XS, (size_t)gMA * sizeof(float))); CK(cudaMemset(XS, 0, (size_t)gMA * sizeof(float)));
  }
  if (gInt8Fc2) {
    CK(cudaMalloc(&HQ, (size_t)gMp * H4)); CK(cudaMemset(HQ, 0, (size_t)gMp * H4));
    CK(cudaMalloc(&HS, (size_t)gMp * (H4 / 128) * sizeof(float)));
    if (gFc2Bfp) CK(cudaMalloc(&HE, (size_t)gMp * (H4 / 128)));
  }
  std::vector<int> cls(gBp, 0);
  for (int b = 0; b < gB; ++b) cls[b] = b * S;
  CK(cudaMalloc(&mCls, gBp * sizeof(int)));
  CK(cudaMemcpy(mCls, cls.data(), gBp * sizeof(int), cudaMemcpyHostToDevice));
  k_pos_rep<<<cdiv(gMp * (C / 8), 256), 256>>>((const half*)WP("pos"), POS, gB, S, C, gMp);
  CK(cudaDeviceSynchronize());
  printf("batch %d: %d tokens (%d padded), buffers ~%.0f MB\n", gB, gMtok, gMp,
         ((double)gMA * C * 2 * 2 + (double)gMA * 3 * Cp * 2 + (double)gMp * H4 * 2 * 3 + (double)gMp * (cfg.Kp() + C) * 2) / 1e6);
}

// ---------------------------------------------------------------- int8 weight build
static void quantRotated(const half* w, int8_t* wt, float* sw, int K, int N, int kind) {
  if (!gRotTmp) CK(cudaMalloc(&gRotTmp, (size_t)cfg.hidden * std::max(cfg.hidden, 3 * cfg.Cp()) * sizeof(half)));
  const size_t sh = 2ull * K * sizeof(float);
  if (kind == 2) k_rot_w<2><<<N, 256, sh>>>(w, gRotTmp, K, N);            // I (x) H_n along K (the LN rotation)
  else           k_rot_w<1><<<N, 256, sh>>>(w, gRotTmp, K, N, gFc2Rot);   // fc2's column blocks
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
  // --int4: the file's int8 codes are staged in gW4tmp and packed two per byte
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
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB); Bfc2.resize(NB, nullptr);
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
  if (!gGptq.empty()) loadGptq(gGptq);
  printf("int4 weights: %.1f MB packed (qkv + fc1 + fc2 of %d blocks), rotated%s\n", per * NB / 1e6, NB,
         gGptq.empty() ? ", RTN 4-bit" : ", GPTQ codes");
}

static void buildInt8Weights() {
  if (gInt4) { buildInt4Weights(); return; }
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, NB = cfg.depth;
  QWqkv.resize(NB); QWfc1.resize(NB); QWfc2.resize(NB); Bfc2.resize(NB, nullptr);
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
      if (gFc2U8) {                        // fold the u8 shift into a private copy of the bias
        float* ws; CK(cudaMalloc(&ws, C * sizeof(float)));
        CK(cudaMalloc(&Bfc2[i], C * sizeof(half)));
        k_wsum<<<C, 256>>>(QWfc2[i].w, H4, C, ws);
        k_bias_shift<<<cdiv(C, 128), 128>>>((const half*)WP(bk(i, "fc2.b")), QWfc2[i].s, ws, 0.169971f, C, Bfc2[i]);
        CK(cudaDeviceSynchronize()); CK(cudaFree(ws));
      }
    }
  }
  CK(cudaDeviceSynchronize());
  if (!gGptq.empty()) loadGptq(gGptq);
  printf("int8 weights: %.1f MB (%s of %d blocks)%s%s\n", per * NB / 1e6,
         gInt8Fc2 ? "qkv + fc1 + fc2" : "qkv + fc1", NB, gRot ? ", rotated" : "", gGptq.empty() ? "" : ", GPTQ");
}

// ---------------------------------------------------------------- launchers (as nano_sam21.cu)
static void launchLN(const half* x, half* y, const float* g, const float* b, int T, int C, cudaStream_t s) {
  int grid = cdiv(T, 8), thr = 256; size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  switch (C) {
    case 192:  k_ln_t<24><<<grid, thr, sh, s>>>(x, y, g, b, T, nullptr, e); break;
    case 384:  k_ln_t<48><<<grid, thr, sh, s>>>(x, y, g, b, T, nullptr, e); break;
    case 768:  k_ln_t<96><<<grid, thr, sh, s>>>(x, y, g, b, T, nullptr, e); break;
    case 1024: k_ln_t<128><<<grid, thr, sh, s>>>(x, y, g, b, T, nullptr, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}
static void launchLNQ8(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C, cudaStream_t s) {
  dim3 grid(cdiv(T, 8)), thr(256);
  size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  if (gRot) {
    switch (C) {
      case 192:  k_ln_q8<24, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
      case 384:  k_ln_q8<48, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
      case 768:  k_ln_q8<96, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
      case 1024: k_ln_q8<128, 1><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
      default: printf("unexpected LN width %d\n", C); exit(1);
    }
    return;
  }
  switch (C) {
    case 192:  k_ln_q8<24><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
    case 384:  k_ln_q8<48><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
    case 768:  k_ln_q8<96><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
    case 1024: k_ln_q8<128><<<grid, thr, sh, s>>>(x, y, ys, g, b, T, nullptr, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}

template<bool GELU, bool SASCALAR = false, bool RESID = false, int GROUP = 0, bool Q8OUT = false,
         int NGPT = 1, int WM = 2, int WN = 4, bool U8IN = false, bool U8OUT = false, int ROT8 = 0,
         bool DB = false, int MINB = 0, int BM = 128, int BN = 128, int ST = 3, bool BFP = false>
static void runS8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C,
                  int M, int N, int K, cudaStream_t s, const half* resid = nullptr,
                  int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0,
                  uint8_t* ex = nullptr) {
  constexpr int BK = 64;
  if constexpr (!GELU && !SASCALAR && !RESID && GROUP == 0 && !Q8OUT && !U8IN && !U8OUT && ROT8 == 0 &&
                BM == 128 && BN == 128 && ST == 3 && !DB) {
    if (gArch120) {
      // sm_120: the 1024-row stage-3 shapes (N >= 2304) fill the 170 SMs in one wave with a 192-wide tile
      // (160 CTAs instead of 240 = 1.4 waves; build5090/retune/s8sweep_qkv.log: 1.11-1.16x); the rest keep the
      // generic tile
      if (M <= 1024 && N >= 2304) {
        constexpr int BN2 = 192;
        constexpr size_t sh2 = s8_smem<BM, BN2, BK, ST>();
        static bool attr2 = false;
        if (!attr2) {
          CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN2, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB, BFP>,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
          attr2 = true;
        }
        k_gemm_s8<BM, BN2, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB, BFP>
            <<<dim3(cdiv(N, BN2), M / BM), WM * WN * 32, sh2, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride, ex);
        return;
      }
    } else if (gArch89 && K >= 576) {
      constexpr int ST2 = 2;
      constexpr size_t sh2 = s8_smem<BM, BN, BK, ST2>();
      static bool attr2 = false;
      if (!attr2) {
        CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, 2, 2, ST2, false, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, 0, BFP>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh2));
        attr2 = true;
      }
      k_gemm_s8<BM, BN, BK, 2, 2, ST2, false, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, 0, BFP>
          <<<dim3(cdiv(N, BN), M / BM), 128, sh2, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride, ex);
      return;
    }
  }
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB, BFP>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr = true;
  }
  k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SASCALAR, RESID, GROUP, Q8OUT, NGPT, U8IN, U8OUT, ROT8, MINB, BFP>
      <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K, resid, C8, sa2, sa_stride, ex);
}
template<bool U8IN, bool BFP = false>
static void runFc2S8(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C,
                     int M, int N, int K, cudaStream_t s, int sastride, uint8_t* ex = nullptr) {
  if (gArch120) {
    // sm_120 (170 SMs), from build5090/retune/s8sweep_fc2.log: the batch-1 ViT shapes want 32-row tiles (3x),
    // the 1024-row stage-3 shapes 64x128 (1.6-1.8x), the narrow stage-0/1 widths 128x64 with 4x1 warps
    // (1.2-1.4x), everything else 128x128 with 16 warps and a 2-stage ring (1.1x); the Ada 192-wide tile
    // leaves 44 % of the SMs idle here.
    const int ctas = cdiv(M, 128) * cdiv(N, 128);
    if (M <= 512)
      runS8<false, false, true, 128, false, 1, 2, 2, U8IN, false, 0, false, 0, 32, 64, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else if (ctas < 100)
      runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 64, 128, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else if (N < 384 && N % 128 != 0)
      runS8<false, false, true, 128, false, 1, 4, 1, U8IN, false, 0, false, 0, 128, 64, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else
      runS8<false, false, true, 128, false, 1, 4, 4, U8IN, false, 0, false, 0, 128, 128, 2, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
  } else if (gArch89) {
    if (N == 288)
      runS8<false, false, true, 128, false, 1, 4, 2, U8IN, false, 0, false, 0, 128, 64, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else if (N == 1152)
      runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 64, 192, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    // Batch 4-8 of a 197-token ViT (M = 896-2048) leaves 28-78 CTAs on the 128 SMs with the 128x192
    // tile.  Narrower tiles are 1.4-2.3x there and bit-identical (s8sweep_mid_2.log).  Above 2048
    // rows the winner flips with M in a way a rule cannot follow (M = 2176 wants 128x192, M = 3200
    // wants 128x128 w4x4; s8sweep_dino.log), so everything larger keeps the shipped tile and the
    // paper's batch-16 and batch-32 rows do not move.
    else if (M <= 1024)
      runS8<false, false, true, 128, false, 1, 2, 2, U8IN, false, 0, false, 0, 64, 64, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else if (M <= 2048 && N >= 1024)
      runS8<false, false, true, 128, false, 1, 4, 4, U8IN, false, 0, false, 0, 128, 128, 2, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else if (M <= 2048)
      runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 64, 192, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
    else
      runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 128, 192, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
  } else {
    runS8<false, false, true, 128, false, 1, 2, 4, U8IN, false, 0, false, 0, 128, 128, 3, BFP>(A, w, sa, bias, C, M, N, K, s, C, nullptr, nullptr, sastride, ex);
  }
}

// ---------------------------------------------------------------- --int4 launchers (W4A4 prototype)
static void launchLNQ4(const half* x, int8_t* y, float* ys, const float* g, const float* b, int T, int C, cudaStream_t s) {
  dim3 grid(cdiv(T, 8)), thr(256);
  size_t sh = 2 * C * sizeof(float); const float e = cfg.eps;
  uint8_t* yu = (uint8_t*)y;
  switch (C) {
    case 192:  k_ln_q4<24><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, e); break;
    case 384:  k_ln_q4<48><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, e); break;
    case 768:  k_ln_q4<96><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, e); break;
    case 1024: k_ln_q4<128><<<grid, thr, sh, s>>>(x, yu, ys, g, b, T, e); break;
    default: printf("unexpected LN width %d\n", C); exit(1);
  }
}
// k_gemm_s8<..., I4 = true>: K is given in ELEMENTS here and passed to the kernel in bytes (K / 2); BK in bytes.
template<bool GELU, bool RESID, int GROUP, bool Q8OUT, int ROT8, int BM, int BN, int BK, int WM, int WN, int ST,
         bool DB = false, int MINB = 0, bool FF = false, bool SAS = false, bool F2S = false>
static void runS4(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K,
                  cudaStream_t s, const half* resid = nullptr, int8_t* C8 = nullptr, float* sa2 = nullptr, int sa_stride = 0) {
  constexpr size_t sh = s8_smem<BM, BN, BK, ST>();
  static bool attr = false;
  if (!attr) {
    CK(cudaFuncSetAttribute(k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SAS, RESID, GROUP, Q8OUT, 1, false, false, ROT8, MINB, false, false, false, true, FF, F2S>,
                            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
    attr = true;
  }
  k_gemm_s8<BM, BN, BK, WM, WN, ST, DB, GELU, SAS, RESID, GROUP, Q8OUT, 1, false, false, ROT8, MINB, false, false, false, true, FF, F2S>
      <<<dim3(cdiv(N, BN), M / BM), WM * WN * 32, sh, s>>>(A, w.w, sa, w.s, bias, C, M, N, K / 2, resid, C8, sa2, sa_stride);
}
// Tile choices of the --int4 GEMMs, from the micro-benchmark at the ViT-B / ViT-L batch-8 / batch-32 shapes
// (w4a4_engine/logs/micro_3090_run2.log; RESULTS.txt section 4).  BK = 64 bytes (128 int4) everywhere: the
// BK = 128-byte tiles lose 20-40 % (one CTA per SM, and 128x192 / 256x128 spill).  The fast flush (FF) is
// bit-identical but 3-10 % slower than the plain flush on the 3090, so it is not used.
// sm_86 (RTX 3090): qkv 128x128 2x4 warps 3 stages; fc1 128x128 2x2 warps 3 stages double-buffered fragments (within
//   1-3 % of 2x4, better at batch 32); fc2 64x128 2x4 when N < 1024 (ViT-B), 128x192 2x4 at M <= 2048 and N >= 1024
//   (ViT-L batch 8: one wave of 78 CTAs on 82 SMs), 128x128 4x4 2 stages above.
// sm_89 (RTX 4090) default: the INT8 path's own sm_89 tiles (qkv 2x2 warps 2 stages, fc1 2x2 DB, fc2 by M / N);
//   NANO_I4ALT=1 selects the sm_86 rules on sm_89 instead (both are timed end to end on the 4090).
static int gI4Alt = -1;
static bool i4Ada() {
  if (gI4Alt < 0) { const char* e = getenv("NANO_I4ALT"); gI4Alt = e ? atoi(e) : 0; if (gI4Alt) printf("NANO_I4ALT=%d: sm_86 INT4 tiles%c", gI4Alt, 10); }
  return gArch89 && !gI4Alt;
}
static void runQkv4(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* C, int M, int N, int K, cudaStream_t s) {
  if (M <= 512)     runS4<false, false, 0, false, 0, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s);
  else if (i4Ada()) runS4<false, false, 0, false, 0, 128, 128, 64, 2, 2, 2>(A, w, sa, bias, C, M, N, K, s);
  else              runS4<false, false, 0, false, 0, 128, 128, 64, 2, 4, 3>(A, w, sa, bias, C, M, N, K, s);
}
static void runFc14(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K,
                    cudaStream_t s, int8_t* C8, float* sa2) {
  if (M <= 512) runS4<true, false, 0, true, 64, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2);
  else          runS4<true, false, 0, true, 64, 128, 128, 64, 2, 2, 3, true, 1>(A, w, sa, bias, HIDb, M, N, K, s, nullptr, C8, sa2);
}
static void runFc24(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* X_, int M, int N, int K,
                    cudaStream_t s, int sastride) {
  if (M <= 512)
    runS4<false, true, 128, false, 0, 32, 64, 64, 2, 2, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
  else if (i4Ada()) {
    if (M <= 1024)                   runS4<false, true, 128, false, 0, 64, 64, 64, 2, 2, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
    else if (M <= 2048 && N >= 1024) runS4<false, true, 128, false, 0, 128, 128, 64, 4, 4, 2>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
    else if (M <= 2048)              runS4<false, true, 128, false, 0, 64, 192, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
    else                             runS4<false, true, 128, false, 0, 128, 192, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
  } else {
    if (N < 1024)                    runS4<false, true, 128, false, 0, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
    else if (M <= 2048)              runS4<false, true, 128, false, 0, 128, 192, 64, 2, 4, 3>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
    else                             runS4<false, true, 128, false, 0, 128, 128, 64, 4, 4, 2>(A, w, sa, bias, X_, M, N, K, s, X_, nullptr, nullptr, sastride);
  }
}
// ---- dinov3_engine_20261027 launchers ----
// --fc2-static producer: runFc14's tiles with F2S (sa2 = device 0.125 / s, read only; no scale store)
static void runFc14S(const int8_t* A, const Q8W& w, const float* sa, const half* bias, int M, int N, int K,
                     cudaStream_t s, int8_t* C8, float* inv) {
  if (M <= 512) runS4<true, false, 0, true, 64, 64, 128, 64, 2, 4, 3, false, 0, false, false, true>(A, w, sa, bias, nullptr, M, N, K, s, nullptr, C8, inv);
  else          runS4<true, false, 0, true, 64, 128, 128, 64, 2, 2, 3, true, 1, false, false, true>(A, w, sa, bias, nullptr, M, N, K, s, nullptr, C8, inv);
}
// --fc2-fp16-b0 (and the G2 gate dump): runFc14's tiles with the plain fp16 GELU epilogue (no int4 output)
static void runFc14Plain(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K,
                         cudaStream_t s) {
  if (M <= 512) runS4<true, false, 0, false, 0, 64, 128, 64, 2, 4, 3>(A, w, sa, bias, HIDb, M, N, K, s);
  else          runS4<true, false, 0, false, 0, 128, 128, 64, 2, 2, 3, true, 1>(A, w, sa, bias, HIDb, M, N, K, s);
}
// --fc2-static consumer: runFc24's tile rule, GROUP = 0 + SASCALAR (sg = device s): no group flush
static void runFc24S(const int8_t* A, const Q8W& w, const float* sg, const half* bias, half* X_, int M, int N, int K, cudaStream_t s) {
  if (M <= 512)
    runS4<false, true, 0, false, 0, 32, 64, 64, 2, 2, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
  else if (i4Ada()) {
    if (M <= 1024)                   runS4<false, true, 0, false, 0, 64, 64, 64, 2, 2, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
    else if (M <= 2048 && N >= 1024) runS4<false, true, 0, false, 0, 128, 128, 64, 4, 4, 2, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
    else if (M <= 2048)              runS4<false, true, 0, false, 0, 64, 192, 64, 2, 4, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
    else                             runS4<false, true, 0, false, 0, 128, 192, 64, 2, 4, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
  } else {
    if (N < 1024)                    runS4<false, true, 0, false, 0, 64, 128, 64, 2, 4, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
    else if (M <= 2048)              runS4<false, true, 0, false, 0, 128, 192, 64, 2, 4, 3, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
    else                             runS4<false, true, 0, false, 0, 128, 128, 64, 4, 4, 2, false, 0, false, true>(A, w, sg, bias, X_, M, N, K, s, X_);
  }
}
// ---- part 2: the W8A8 (INT8, --rot / --gptq, sm_89) path with a static fc2 scale (--fc2-static without --int4) ----
// producer: the sm_89 rotated INT8 fc1 tile of the dynamic path (128x128 2x2 DB MINB 1; 64x128 2x4 at gMp <= 512) + F2S
static void runFc1W8S(const int8_t* A, const Q8W& w, const float* sa, const half* bias, int M, int N, int K, cudaStream_t s,
                      int8_t* C8, float* inv) {
  if (M <= 512) {
    auto k = k_gemm_s8<64, 128, 64, 2, 4, 3, false, true, false, false, 0, true, 1, false, false, 64, 0,
                       false, false, false, false, false, true>;
    constexpr size_t sh = s8_smem<64, 128, 64, 3>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k<<<dim3(cdiv(N, 128), M / 64), 256, sh, s>>>(A, w.w, sa, w.s, bias, nullptr, M, N, K, nullptr, C8, inv, 0, nullptr, 0);
  } else {
    auto k = k_gemm_s8<128, 128, 64, 2, 2, 3, true, true, false, false, 0, true, 1, false, false, 64, 1,
                       false, false, false, false, false, true>;
    constexpr size_t sh = s8_smem<128, 128, 64, 3>();
    static bool attr = false;
    if (!attr) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh)); attr = true; }
    k<<<dim3(cdiv(N, 128), M / 128), 128, sh, s>>>(A, w.w, sa, w.s, bias, nullptr, M, N, K, nullptr, C8, inv, 0, nullptr, 0);
  }
}
// the same tiles with the plain fp16 GELU epilogue (block-0 fp16 exemption, calibration and gate dumps)
static void runFc1W8Plain(const int8_t* A, const Q8W& w, const float* sa, const half* bias, half* HIDb, int M, int N, int K, cudaStream_t s) {
  if (M <= 512) runS8<true, false, false, 0, false, 1, 2, 4, false, false, 0, false, 0, 64>(A, w, sa, bias, HIDb, M, N, K, s);
  else          runS8<true, false, false, 0, false, 1, 2, 2, false, false, 0, true, 1>(A, w, sa, bias, HIDb, M, N, K, s);
}
// consumer: the dynamic INT8 fc2's sm_89 tile rule (runFc2S8 / small-M 32x64) with GROUP = 0 + SASCALAR (sg = device s)
static void runFc2W8S(const int8_t* A, const Q8W& w, const float* sg, const half* bias, half* C, int M, int N, int K, cudaStream_t s) {
  if (M <= 512)                    runS8<false, true, true, 0, false, 1, 2, 2, false, false, 0, false, 0, 32, 64, 3>(A, w, sg, bias, C, M, N, K, s, C);
  else if (N == 288)               runS8<false, true, true, 0, false, 1, 4, 2, false, false, 0, false, 0, 128, 64, 3>(A, w, sg, bias, C, M, N, K, s, C);
  else if (N == 1152)              runS8<false, true, true, 0, false, 1, 2, 4, false, false, 0, false, 0, 64, 192, 3>(A, w, sg, bias, C, M, N, K, s, C);
  else if (M <= 1024)              runS8<false, true, true, 0, false, 1, 2, 2, false, false, 0, false, 0, 64, 64, 3>(A, w, sg, bias, C, M, N, K, s, C);
  else if (M <= 2048 && N >= 1024) runS8<false, true, true, 0, false, 1, 4, 4, false, false, 0, false, 0, 128, 128, 2>(A, w, sg, bias, C, M, N, K, s, C);
  else if (M <= 2048)              runS8<false, true, true, 0, false, 1, 2, 4, false, false, 0, false, 0, 64, 192, 3>(A, w, sg, bias, C, M, N, K, s, C);
  else                             runS8<false, true, true, 0, false, 1, 2, 4, false, false, 0, false, 0, 128, 192, 3>(A, w, sg, bias, C, M, N, K, s, C);
}
static std::string gF2SCalib;     // NANO_F2S_CALIB=<dir>: dynamic W8A8 path, first eager pass: per block the fp16 GELU output
static bool gFirstPass = true;    //   (valid rows) + the INT8 fc2 weight / column scales, for the static-scale calibration
// NANOF2S1 (the format of fc2static_engine_20261016 / w8a8_static_fc2_20261026), mode 1 only: char[8] magic, int32 nblk,
// int32 mode; per block int32 K (= hidden), int32 G (= K / 128), float s[G] (all equal), float sgemm (= s)
static void buildFc2Static() {
  std::ifstream f(gFc2StaticFile, std::ios::binary); char magic[8]; int n = 0, mode = 0;
  f.read(magic, 8); f.read((char*)&n, 4); f.read((char*)&mode, 4);
  if (!f || memcmp(magic, "NANOF2S1", 8) || n != cfg.depth || mode != 1) { printf("Invalid fc2-static header (mode must be 1 = sym, nblk = depth)\n"); exit(1); }
  gF2Sinv.resize(n); gF2Sg.resize(n);
  double smin = 1e30, smax = 0;
  for (int bi = 0; bi < n; ++bi) {
    int K = 0, G = 0; f.read((char*)&K, 4); f.read((char*)&G, 4);
    if (!f || K != cfg.hidden || G != cdiv(K, 128)) { printf("Invalid fc2-static dimensions (block %d)\n", bi); exit(1); }
    std::vector<float> sv(G); float sg = 0.f;
    f.read((char*)sv.data(), G * 4); f.read((char*)&sg, 4);
    if (!f) { printf("Truncated fc2-static file\n"); exit(1); }
    for (int g = 0; g < G; ++g)
      if (!std::isfinite(sv[g]) || sv[g] <= 0.f || sv[g] != sv[0]) { printf("Invalid fc2-static scale (block %d group %d)\n", bi, g); exit(1); }
    if (sg != sv[0]) { printf("Invalid fc2-static GEMM scale (block %d)\n", bi); exit(1); }
    const float hv[2] = {0.125f / sv[0], sv[0]};
    CK(cudaMalloc(&gF2Sinv[bi], 4)); CK(cudaMalloc(&gF2Sg[bi], 4));
    CK(cudaMemcpy(gF2Sinv[bi], &hv[0], 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(gF2Sg[bi], &hv[1], 4, cudaMemcpyHostToDevice));
    smin = std::min(smin, (double)sv[0]); smax = std::max(smax, (double)sv[0]);
  }
  char extra; if (f.read(&extra, 1)) { printf("Trailing fc2-static bytes\n"); exit(1); }
  printf("fc2-static: static symmetric %s fc2-input scales (%d blocks, s in [%.4g, %.4g]); fc1 epilogue without the row "
         "maximum, fc2 %s GEMM without the group flush%s\n", gInt4 ? "INT4" : "INT8", n, smin, smax, gInt4 ? "INT4" : "INT8", gFc2Fp16B0 ? "; block 0's fc2 in fp16" : "");
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
    if (!attr) {
      CK(cudaFuncSetAttribute(k_gemm_gelu<128, 128, 32, 2, 4, 3, true, true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
      attr = true;
    }
    k_gemm_gelu<128, 128, 32, 2, 4, 3, true, true, false><<<dim3(cdiv(N, 128), M / 128), 256, sh, s>>>(A, Wt, bias, C, M, N, K);
  } else {
    constexpr size_t sh = gemm_gelu_smem<128, 64, 32, 4, true>();
    static bool attr = false;
    if (!attr) {
      CK(cudaFuncSetAttribute(k_gemm_gelu<128, 64, 32, 2, 2, 4, false, true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sh));
      attr = true;
    }
    k_gemm_gelu<128, 64, 32, 2, 2, 4, false, true, true><<<dim3(cdiv(N, 64), M / 128), 128, sh, s>>>(A, Wt, bias, C, M, N, K);
  }
}

// Attention over one image = one window of S keys, B*heads windows.  ws2 = ws2q = TQ = TK = S;
// the kernel masks keys >= TK on its tail tile (the rows past the window belong to the next
// image or the 64 slack rows, all finite).

// Default score accumulation, overridden by `--attn-acc N`.  acc 2 keeps Q.K^T in fp16, which on
// DH = 64 also frees the registers k_attn3 needs (below), so it is two changes at once.
// Swept against acc 1 on the RTX 4090 (sm_89) and the RTX 3090 and A10 (sm_86) for DeiT-S, DeiT-B,
// ViT-B/16 and ViT-L/16, at batch 32 and batch 1, fp16 and INT8
// (bench/paper/remeasure_2026-09/accsweep_{4090,local}.log, 3 alternating repetitions each):
// at batch 32 acc 2 is 1.1-4.0 % faster on the 4090, 0.5-4.5 % on the 3090 and 1.2-2.5 % on the
// A10; at batch 1 it is 0.4-1.8 % faster on the 4090 and 3090 and within +-0.2 % on the A10.
// ImageNet top-1 over 50 000 images moves by at most 0.04 points in any mode.
//
// The DINOv3 family (rope != 0) is EXCLUDED, because its register tokens carry massive activations
// and the attention logits do not fit an fp16 accumulator: the largest |q.k| of block 0 is 17-28 on
// DeiT-S/B and ViT-B/L, 613 on DINOv3-L (which the exporter already divides by 8) but 1.26e5 on
// DINOv3-B, against an fp16 maximum of 65504, and acc 2 there returns NaN for 16 % of the tokens
// (verified 2026-09-21; the accuracy run in remeasure_2026-09/acc_eval.log shows it as CLS cos nan).
// DINOv3-L has headroom and is 1.3-2.8 % faster with `--attn-acc 2`, but the margin depends on an
// exporter-side rescale, so it is not the default either.
// sm_120 is excluded: the 5090 had another tenant during this session, so its default is unchanged
// and its numbers stay bit-identical.  DH != 64 is excluded too (this build is DH = 64 only, but
// the hiera engines share the flag and their 72/96/56 head dimensions were not a win, see
// accsweep_hiera_4090.log), which is what the DH test below says.
static int defAttnAcc() { return (DH == 64 && gArch80 && !gArch120 && cfg.rope == 0) ? 2 : 1; }

static void runAttnVit(const half* qkv, half* xout, cudaStream_t s) {
  const int S = cfg.S, Cp = cfg.Cp(), Co = cfg.C, Bw = gB * cfg.heads;
  int acc = gAttnAcc >= 0 ? gAttnAcc : defAttnAcc();
#define ATTNV(NW, BK, QF, PF) \
  k_attn2<DH, NW, BK, 2, QF, PF><<<dim3(cdiv(S, NW * 16), Bw), NW * 32, attn2_smem<DH, BK, 2>(), s>>>( \
      qkv, nullptr, xout, S, S, cfg.heads, Cp, Co, S, S)
#define ATTNV_ACC(NW, BK) do { if (acc == 2) ATTNV(NW, BK, true, true); else if (acc == 1) ATTNV(NW, BK, false, true); else ATTNV(NW, BK, false, false); } while (0)
  // k_attn3 (sam21_attn3.cuh): 128-query block, fp16 mma accumulators used as the output
  // accumulator, warps whose rows all lie past S skipped.  See runAttn in nano_sam3.cu for
  // why; the same grid test (>= ~2 CTAs per SM) keeps batch 1 on k_attn2, where the wide
  // block is 30 % slower.  DH = 64 only, because that is where it was measured.
#define ATTNV3(QF) \
  k_attn3<DH, 4, 2, 32, 2, -1, false, 4, 0, QF, true><<<dim3(cdiv(S, 128), Bw), 128, attn2_smem<DH, 32, 2>(), s>>>( \
      qkv, nullptr, xout, S, S, cfg.heads, Cp, Co, S, S)
  const bool a3grid = DH == 64 && (long long)Bw * cdiv(S, 128) >= 128;
  if (acc == 2 && a3grid) ATTNV3(true);
  // sm_120 (build5090/retune/attnsweep_64.log): 8-warp CTAs on the 197-token images at batch 32, 2-warp CTAs at batch 1
  else if (gArch120 && gB >= 8 && S <= 224) ATTNV_ACC(8, 32);
  else if (gArch120 && gB == 1) ATTNV_ACC(2, 32);
  else ATTNV_ACC(4, 32);
#undef ATTNV3
#undef ATTNV_ACC
#undef ATTNV
}

// ---------------------------------------------------------------- profiling
enum PTag { P_IM2COL, P_PATCH, P_LN, P_QKV, P_ATTN, P_PROJ, P_FC1, P_FC2Q, P_FC2, P_HEAD, P_NTAG };
static const char* PNAME[P_NTAG] = {"im2col", "patch_gemm", "layernorm", "qkv_gemm", "attention", "proj_gemm", "fc1_gemm", "fc2_quant", "fc2_gemm", "head"};
static std::vector<cudaEvent_t> gEv; static std::vector<int> gTag; static int gEvN = 0;
static void mark(int tag, cudaStream_t s) {
  if (!gProf) return;
  if (gEvN >= (int)gEv.size()) { cudaEvent_t e; CK(cudaEventCreate(&e)); gEv.push_back(e); }
  CK(cudaEventRecord(gEv[gEvN], s));
  if ((int)gTag.size() <= gEvN) gTag.push_back(tag); else gTag[gEvN] = tag;
  ++gEvN;
}
static void profReport() {
  double acc[P_NTAG] = {0}; double tot = 0;
  for (int i = 0; i + 1 < gEvN; ++i) {
    float ms; CK(cudaEventElapsedTime(&ms, gEv[i], gEv[i + 1]));
    if (gTag[i] >= 0 && gTag[i] < P_NTAG) acc[gTag[i]] += ms;
    tot += ms;
  }
  printf("---- per-category GPU time (single un-captured pass, batch %d) ----\n", gB);
  for (int t = 0; t < P_NTAG; ++t)
    if (acc[t] > 0) printf("  %-12s %8.3f ms  %5.1f%%\n", PNAME[t], acc[t], 100 * acc[t] / tot);
  printf("  %-12s %8.3f ms\n", "TOTAL", tot);
}

// ---------------------------------------------------------------- forward
static int gpPatch, gpHead; static std::vector<int> gpQkv, gpProj, gpFc1, gpFc2;
static void buildPlans() {
  const int C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, NB = cfg.depth;
  gpPatch = makePlan(gMp, C, cfg.Kp(), CUBLASLT_EPILOGUE_DEFAULT, nullptr, true);
  gpQkv.resize(NB); gpProj.resize(NB); gpFc1.resize(NB); gpFc2.resize(NB);
  for (int i = 0; i < NB; ++i) {
    gpQkv[i]  = makePlan(gMp, 3 * Cp, C, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "qkv.b")), false);
    gpProj[i] = makePlan(gMp, C, C, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "proj.b")), true);
    gpFc1[i]  = makePlan(gMp, H4, C, gLtGelu ? CUBLASLT_EPILOGUE_GELU_BIAS : CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "fc1.b")), false);
    gpFc2[i]  = makePlan(gMp, C, H4, CUBLASLT_EPILOGUE_BIAS, WP(bk(i, "fc2.b")), true, true);
  }
  gpHead = cfg.ncls ? makePlan(gBp, cfg.ncls, C, CUBLASLT_EPILOGUE_BIAS, WP("head.b"), false) : -1;
}

static void forward(cudaStream_t s) {
  const int TH = 256, C = cfg.C, Cp = cfg.Cp(), H4 = cfg.hidden, S = cfg.S, G = cfg.img / cfg.patch;
  mark(P_IM2COL, s);
  k_im2col_vit<<<cdiv(gMp, 8), 256, 0, s>>>(dIn, COL, gB, S, G, cfg.patch, cfg.img, gMp, cfg.prefix,
      cfg.mean[0], cfg.mean[1], cfg.mean[2], 1.f / cfg.stdv[0], 1.f / cfg.stdv[1], 1.f / cfg.stdv[2]);
  mark(P_PATCH, s);
  runPlan(gpPatch, COL, WP("patch.w"), POS, X, s);
  if (!gDump.empty()) writeBin((gDump + "/x_embed.bin").c_str(), X, (size_t)gMtok * C * 2);
  if (!gDump.empty() && gInt8 && gInt8Fc2) {         // block-0 weights as the GEMMs read them (int4: packed [N, K/2])
    const int dv = gInt4 ? 2 : 1;
    writeBin((gDump + "/w_qkv_b0.bin").c_str(), QWqkv[gDumpBlk].w, (size_t)3 * Cp * C / dv);
    writeBin((gDump + "/s_qkv_b0.bin").c_str(), QWqkv[gDumpBlk].s, (size_t)3 * Cp * 4);
    writeBin((gDump + "/w_fc1_b0.bin").c_str(), QWfc1[gDumpBlk].w, (size_t)H4 * C / dv);
    writeBin((gDump + "/s_fc1_b0.bin").c_str(), QWfc1[gDumpBlk].s, (size_t)H4 * 4);
    writeBin((gDump + "/w_fc2_b0.bin").c_str(), QWfc2[gDumpBlk].w, (size_t)C * H4 / dv);
    writeBin((gDump + "/s_fc2_b0.bin").c_str(), QWfc2[gDumpBlk].s, (size_t)C * 4);
  }

  for (int i = 0; i < cfg.depth; ++i) {
    const float* g1 = (const float*)WP(bk(i, "ln1.w")); const float* b1 = (const float*)WP(bk(i, "ln1.b"));
    const float* g2 = (const float*)WP(bk(i, "ln2.w")); const float* b2 = (const float*)WP(bk(i, "ln2.b"));
    if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/x_in_b0.bin").c_str(), X, (size_t)gMtok * C * 2);
    mark(P_LN, s);
    if (gInt4)     launchLNQ4(X, XQ, XS, g1, b1, gMp, C, s);
    else if (gInt8) launchLNQ8(X, XQ, XS, g1, b1, gMp, C, s);
    else       launchLN(X, Xn, g1, b1, gMp, C, s);
    if (!gDump.empty() && i == gDumpBlk && gInt8) {        // the quantised LN1 output of block 0 and its per-token scales
      writeBin((gDump + "/xq_ln1_b0.bin").c_str(), XQ, (size_t)gMtok * C / (gInt4 ? 2 : 1));
      writeBin((gDump + "/xs_ln1_b0.bin").c_str(), XS, (size_t)gMtok * 4);
    }
    mark(P_QKV, s);
    // small batches (M <= 512 rows) use 64-row tiles: twice the CTAs of the 128-row tile, which
    // otherwise leaves most of the SMs idle at batch 1 (M = 256 -> 2 row tiles)
    const bool smallM = gMp <= 512;
    if (gInt4) runQkv4(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, gMp, 3 * Cp, C, s);
    else if (gInt8 && smallM && 3 * Cp <= 1536)   // narrow qkv at small M: 32x64 triples the CTAs (1.3x, s8sweep_qkv_89_b1.log)
      runS8<false, false, false, 0, false, 1, 2, 2, false, false, 0, false, 0, 32, 64, 3>(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, gMp, 3 * Cp, C, s);
    else if (gInt8 && smallM) runS8<false, false, false, 0, false, 1, 2, 4, false, false, 0, false, 0, 64>(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, gMp, 3 * Cp, C, s);
    else if (gInt8) runS8<false>(XQ, QWqkv[i], XS, (const half*)WP(bk(i, "qkv.b")), QKV, gMp, 3 * Cp, C, s);
    else       runPlan(gpQkv[i], Xn, WP(bk(i, "qkv.w")), nullptr, QKV, s);
    if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/qkv_b0.bin").c_str(), QKV, (size_t)gMtok * 3 * Cp * 2);
    if (cfg.rope) k_rope<<<cdiv(gMp * cfg.heads * (DH / 2), 256), 256, 0, s>>>(QKV, (const float*)WP("rope"), gB, S, cfg.prefix, cfg.heads, Cp, gMp);
    mark(P_ATTN, s);
    runAttnVit(QKV, Xn, s);
    if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/attn_b0.bin").c_str(), Xn, (size_t)gMtok * C * 2);
    mark(P_PROJ, s);
    if (useResF16(gMp)) runResF16(Xn, (const half*)WP(bk(i, "proj.w")), (const half*)WP(bk(i, "proj.b")), X, X, gMp, C, C, s);
    else runPlan(gpProj[i], Xn, WP(bk(i, "proj.w")), X, X, s);          // + residual, in place
    if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/x_mid_b0.bin").c_str(), X, (size_t)gMtok * C * 2);
    mark(P_LN, s);
    if (gInt4)     launchLNQ4(X, XQ, XS, g2, b2, gMp, C, s);
    else if (gInt8) launchLNQ8(X, XQ, XS, g2, b2, gMp, C, s);
    else       launchLN(X, Xn, g2, b2, gMp, C, s);
    if (!gDump.empty() && i == gDumpBlk && gInt8) {
      writeBin((gDump + "/xq_ln2_b0.bin").c_str(), XQ, (size_t)gMtok * C / (gInt4 ? 2 : 1));
      writeBin((gDump + "/xs_ln2_b0.bin").c_str(), XS, (size_t)gMtok * 4);
    }
    mark(P_FC1, s);
    const half* fb1 = (const half*)WP(bk(i, "fc1.b"));
    if (gInt4 && gFc2Fp16B0 && i == 0) {
      runFc14Plain(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
      if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/hid_b0.bin").c_str(), HID, (size_t)gMtok * H4 * 2);
    } else if (gInt4 && !gFc2StaticFile.empty()) {
      runFc14S(XQ, QWfc1[i], XS, fb1, gMp, H4, C, s, HQ, gF2Sinv[i]);
      if (!gDump.empty() && i == gDumpBlk) {      // G2 gate: the fp16 GELU output of the same tile, and {0.125/s, s}
        runFc14Plain(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
        CK(cudaStreamSynchronize(s));
        writeBin((gDump + "/hid_b0.bin").c_str(), HID, (size_t)gMtok * H4 * 2);
        float hv[2]; CK(cudaMemcpy(&hv[0], gF2Sinv[i], 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hv[1], gF2Sg[i], 4, cudaMemcpyDeviceToHost));
        std::ofstream ff(gDump + "/f2s_b0.bin", std::ios::binary); ff.write((const char*)hv, 8);
      }
    } else if (gInt4) {
      runFc14(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, HQ, HS);
    } else if (gInt8 && gInt8Fc2 && gFc2Fp16B0 && i == 0) {          // part 2: W8A8, block 0's fc2 in fp16
      runFc1W8Plain(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
      if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/hid_b0.bin").c_str(), HID, (size_t)gMtok * H4 * 2);
    } else if (gInt8 && gInt8Fc2 && !gFc2StaticFile.empty()) {       // part 2: W8A8 static fc2 producer
      runFc1W8S(XQ, QWfc1[i], XS, fb1, gMp, H4, C, s, HQ, gF2Sinv[i]);
      if (!gDump.empty() && i == gDumpBlk) {
        runFc1W8Plain(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
        CK(cudaStreamSynchronize(s));
        writeBin((gDump + "/hid_b0.bin").c_str(), HID, (size_t)gMtok * H4 * 2);
        float hv[2]; CK(cudaMemcpy(&hv[0], gF2Sinv[i], 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hv[1], gF2Sg[i], 4, cudaMemcpyDeviceToHost));
        std::ofstream ff(gDump + "/f2s_b0.bin", std::ios::binary); ff.write((const char*)hv, 8);
      }
    } else if (gInt8 && gInt8Fc2 && gFc2Pass) {
      // --fc2-pass: the fused kernel's tile with the plain fp16 GELU epilogue, then the pass (gMp % 128 == 0, checked at start-up)
      if (gArch89 && gFc2Rot == 64) runS8<true, false, false, 0, false, 1, 2, 2, false, false, 0, true, 1>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
      else                          runS8<true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
      mark(P_FC2Q, s);
      const dim3 grid(cdiv(H4, 128), gMp / 128);
      if (gFc2Rot == 128) k_rotquant_fc2<128><<<grid, 256, 0, s>>>(HID, HQ, HS, H4, cdiv(H4, 128));
      else                k_rotquant_fc2<64><<<grid, 256, 0, s>>>(HID, HQ, HS, H4, cdiv(H4, 128));
    } else if (gInt8 && gInt8Fc2 && smallM) {  // 64-row tiles for small batches (see qkv)
      if (gRot && gFc2Bfp) runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64, false, 0, 64, 128, 3, true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS, 0, HE);
      else if (gRot && H4 <= 1536)   // narrow fc1 at small M: 32x128 (1.2x, s8sweep_fc1_89_b1.log)
                       runS8<true, false, false, 0, true, 1, 2, 2, false, false, 64, false, 0, 32, 128, 3>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gRot)   runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64, false, 0, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gFc2U8) runS8<true, false, false, 0, true, 1, 2, 4, false, true, 0, false, 0, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else             runS8<true, false, false, 0, true, 1, 2, 4, false, false, 0, false, 0, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
    } else if (gInt8 && gInt8Fc2) {     // fc1 emits int8 + one scale per (row, 128-column group)
      if (gRot && gFc2Rot == 128)
        runS8<true, false, false, 0, true, 1, 2, 4, false, false, 128>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gRot && gFc2Bfp && gArch89)     // --fc2-bfp: power-of-two scale + byte exponent
        runS8<true, false, false, 0, true, 1, 2, 2, false, false, 64, true, 1, 128, 128, 3, true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS, 0, HE);
      else if (gRot && gFc2Bfp)
        runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64, false, 0, 128, 128, 3, true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS, 0, HE);
      else if (gRot && gArch120)   // sm_120 placeholder (generic tile)
        runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gRot && gArch89)
        runS8<true, false, false, 0, true, 1, 2, 2, false, false, 64, true, 1>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gRot)
        runS8<true, false, false, 0, true, 1, 2, 4, false, false, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else if (gFc2U8)
        runS8<true, false, false, 0, true, 1, 2, 4, false, true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
      else
        runS8<true, false, false, 0, true, 1>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s, nullptr, HQ, HS);
    } else if (gInt8 && smallM) {
      runS8<true, false, false, 0, false, 1, 2, 4, false, false, 0, false, 0, 64>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
    } else if (gInt8) {
      runS8<true>(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
    } else if (gArch80 && !gLtGelu) {
      runFc1Gelu(Xn, (const half*)WP(bk(i, "fc1.w")), fb1, HID, gMp, H4, C, gAccMode >= 1, s);
    } else {
      runPlan(gpFc1[i], Xn, WP(bk(i, "fc1.w")), nullptr, HID, s);
      if (!gLtGelu) k_gelu<<<cdiv(gMp * H4 / 8, TH), TH, 0, s>>>(HID, gMp * H4);
    }
    if (gFirstPass && !gF2SCalib.empty()) {           // part 2 calibration dump (dynamic W8A8 path, validated in main)
      runFc1W8Plain(XQ, QWfc1[i], XS, fb1, HID, gMp, H4, C, s);
      CK(cudaStreamSynchronize(s));
      char nm[64];
      snprintf(nm, sizeof nm, "/b%02d_hid.bin", i); writeBin((gF2SCalib + nm).c_str(), HID, (size_t)gMtok * H4 * 2);
      snprintf(nm, sizeof nm, "/b%02d_w.bin", i);   writeBin((gF2SCalib + nm).c_str(), QWfc2[i].w, (size_t)C * H4);
      snprintf(nm, sizeof nm, "/b%02d_sw.bin", i);  writeBin((gF2SCalib + nm).c_str(), QWfc2[i].s, (size_t)C * 4);
    }
    if (!gDump.empty() && i == gDumpBlk && gInt8Fc2) {        // fc2's quantised operand of block 0 and its (row, group) scales
      writeBin((gDump + "/hq_b0.bin").c_str(), HQ, (size_t)gMtok * H4 / (gInt4 ? 2 : 1));
      writeBin((gDump + "/hs_b0.bin").c_str(), HS, (size_t)gMtok * cdiv(H4, 128) * 4);
    }
    mark(P_FC2, s);
    if (gInt4 && gFc2Fp16B0 && i == 0) {      // the fp16 exemption: the fp16 engine's fc2 GEMM (+ bias + residual)
      if (useResF16(gMp)) runResF16(HID, (const half*)WP(bk(i, "fc2.w")), (const half*)WP(bk(i, "fc2.b")), X, X, gMp, C, H4, s);
      else runPlan(gpFc2[i], HID, WP(bk(i, "fc2.w")), X, X, s);
    } else if (gInt4 && !gFc2StaticFile.empty()) {
      runFc24S(HQ, QWfc2[i], gF2Sg[i], (const half*)WP(bk(i, "fc2.b")), X, gMp, C, H4, s);
    } else if (gInt4) {
      runFc24(HQ, QWfc2[i], HS, (const half*)WP(bk(i, "fc2.b")), X, gMp, C, H4, s, cdiv(H4, 128));
    } else if (gInt8 && gInt8Fc2 && gFc2Fp16B0 && i == 0) {          // part 2: the fp16 exemption (fp16 engine's fc2 GEMM)
      if (useResF16(gMp)) runResF16(HID, (const half*)WP(bk(i, "fc2.w")), (const half*)WP(bk(i, "fc2.b")), X, X, gMp, C, H4, s);
      else runPlan(gpFc2[i], HID, WP(bk(i, "fc2.w")), X, X, s);
    } else if (gInt8 && gInt8Fc2 && !gFc2StaticFile.empty()) {       // part 2: W8A8 static fc2 consumer
      runFc2W8S(HQ, QWfc2[i], gF2Sg[i], (const half*)WP(bk(i, "fc2.b")), X, gMp, C, H4, s);
    } else if (gInt8 && gInt8Fc2) {
      const half* fb = gFc2U8 ? Bfc2[i] : (const half*)WP(bk(i, "fc2.b"));
      const int sastride = cdiv(H4, 128);
      if (gFc2Bfp && gBfpEmax >= 0) {   // clamp the per-row exponent spread into int32 headroom
        const int autoe = (int)floor(31.0 - log2((double)sastride * 128.0 * 16129.0));
        k_bfp_guard<<<cdiv(gMp, 256), 256, 0, s>>>(HE, HQ, gMp, sastride, H4, 128,
                                                 gBfpEmax ? gBfpEmax : autoe, gBfpCheck ? 0 : 1);
      }
      if (smallM && gFc2Bfp)
        runS8<false, false, true, 128, false, 1, 2, 4, false, false, 0, false, 0, 64, 128, 3, true>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, X, nullptr, nullptr, sastride, HE);
      else if (smallM) {   // 32x64: at gMp <= 512 the 64x128 tile leaves 12-48 CTAs on 128 SMs (1.6-2.2x, s8sweep_fc2_89_b1.log)
        if (gFc2U8) runS8<false, false, true, 128, false, 1, 2, 2, true, false, 0, false, 0, 32, 64, 3>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, X, nullptr, nullptr, sastride);
        else        runS8<false, false, true, 128, false, 1, 2, 2, false, false, 0, false, 0, 32, 64, 3>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, X, nullptr, nullptr, sastride);
      }
      else if (gFc2Bfp) runFc2S8<false, true>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, sastride, HE);
      else if (gFc2U8) runFc2S8<true>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, sastride);
      else             runFc2S8<false>(HQ, QWfc2[i], HS, fb, X, gMp, C, H4, s, sastride);
    } else if (useResF16(gMp)) {
      runResF16(HID, (const half*)WP(bk(i, "fc2.w")), (const half*)WP(bk(i, "fc2.b")), X, X, gMp, C, H4, s);
    } else {
      runPlan(gpFc2[i], HID, WP(bk(i, "fc2.w")), X, X, s);
    }
    if (!gDump.empty() && i == gDumpBlk) writeBin((gDump + "/x_b0.bin").c_str(), X, (size_t)gMtok * C * 2);
  }
  mark(P_HEAD, s);
  if (cfg.ncls == 0) {                 // feature model (DINOv3): the final LayerNorm over every token is the output
    launchLN(X, Xn, (const float*)WP("norm.w"), (const float*)WP("norm.b"), gMp, C, s);
  } else {
    k_gather<<<cdiv(gBp * (C / 8), TH), TH, 0, s>>>(X, CLS, mCls, gBp, C);
    launchLN(CLS, CLSn, (const float*)WP("norm.w"), (const float*)WP("norm.b"), gBp, C, s);
    if (gpHead >= 0) runPlan(gpHead, CLSn, WP("head.w"), nullptr, LOG, s);
  }
  mark(-1, s);
  if (!gDump.empty()) {
    writeBin((gDump + "/x_final.bin").c_str(), X, (size_t)gMtok * C * 2);
    if (cfg.ncls == 0) writeBin((gDump + "/tokens.bin").c_str(), Xn, (size_t)gMtok * C * 2);
    else {
      writeBin((gDump + "/cls_norm.bin").c_str(), CLSn, (size_t)gB * C * 2);
      if (gpHead >= 0) writeBin((gDump + "/logits.bin").c_str(), LOG, (size_t)gB * cfg.ncls * 2);
    }
  }
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

int main(int argc, char** argv) {
  std::string model = "vit/deit_base_patch16_224", input, inputList, outDir;
  int warmup = 20, iters = 100, useGraph = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) model = argv[++i];
    else if (a == "--batch" && i + 1 < argc) gB = atoi(argv[++i]);
    else if (a == "--input" && i + 1 < argc) input = argv[++i];
    else if (a == "--input-list" && i + 1 < argc) inputList = argv[++i];
    else if (a == "--out-dir" && i + 1 < argc) outDir = argv[++i];
    else if (a == "--dump" && i + 1 < argc) gDump = argv[++i];
    else if (a == "--warmup" && i + 1 < argc) warmup = atoi(argv[++i]);
    else if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
    else if (a == "--no-graph") useGraph = 0;
    else if (a == "--f16acc") gAccMode = 2;
    else if (a == "--f16acc-shallow") gAccMode = 1;
    else if (a == "--attn-acc" && i + 1 < argc) gAttnAcc = atoi(argv[++i]);
    else if (a == "--int8") gInt8 = true;
    else if (a == "--int8-fc2") { gInt8 = true; gInt8Fc2 = true; }
    else if (a == "--int4") { gInt4 = true; gInt8 = true; gInt8Fc2 = true; gRot = true; }
    else if (a == "--fc2-s8") gFc2U8 = false;
    else if (a == "--rot") { gInt8 = true; gRot = true; }
    else if (a == "--fc2-rot" && i + 1 < argc) gFc2Rot = atoi(argv[++i]);
    else if (a == "--fc2-bfp") gFc2Bfp = true;
    else if (a == "--fc2-bfp-raw") { gFc2Bfp = true; gBfpEmax = -1; }
    else if (a == "--fc2-bfp-guard" && i + 1 < argc) { gFc2Bfp = true; gBfpEmax = atoi(argv[++i]); }
    else if (a == "--fc2-bfp-check" && i + 1 < argc) { gFc2Bfp = true; gBfpCheck = true; gBfpEmax = atoi(argv[++i]); }
    else if (a == "--gptq" && i + 1 < argc) { gInt8 = true; gRot = true; gGptq = argv[++i]; }
    else if (a == "--fc2-pass") gFc2Pass = true;
    else if (a == "--lt-gelu") gLtGelu = true;
    else if (a == "--profile") gProf = true;
    else if (a == "--no-tune") gTune = false;
    else if (a == "--fc2-static" && i + 1 < argc) gFc2StaticFile = argv[++i];
    else if (a == "--fc2-fp16-b0") gFc2Fp16B0 = true;
    else { printf("unknown arg %s\n", argv[i]); return 1; }
  }
  if (const char* e = getenv("NANO_F2S_CALIB")) gF2SCalib = e;
  if ((!gFc2StaticFile.empty() || gFc2Fp16B0 || !gF2SCalib.empty()) && !gInt4) {   // part 2: the sm_89 W8A8 path
    cudaDeviceProp p2; CK(cudaGetDeviceProperties(&p2, 0));
    if (!(gInt8Fc2 && gRot && gFc2Rot == 64 && !gFc2Pass && !gFc2Bfp && p2.major == 8 && p2.minor == 9)) {
      printf("--fc2-static / --fc2-fp16-b0 / NANO_F2S_CALIB: --int4, or the sm_89 W8A8 path (--int8-fc2 --rot/--gptq, H_64, no --fc2-pass / --fc2-bfp)\n"); return 1;
    }
    if (!gF2SCalib.empty() && (!gFc2StaticFile.empty() || gFc2Fp16B0)) { printf("NANO_F2S_CALIB is a dump of the DYNAMIC W8A8 path\n"); return 1; }
  }
  if (!gF2SCalib.empty() && gInt4) { printf("NANO_F2S_CALIB is for the W8A8 path\n"); return 1; }
  if (const char* e = getenv("NANO_DUMP_BLOCK")) {
    gDumpBlk = atoi(e);
    if (gDump.empty()) { printf("NANO_DUMP_BLOCK needs --dump\n"); return 1; }
    printf("NANO_DUMP_BLOCK=%d: the --dump files (named _b0) are those of block %d\n", gDumpBlk, gDumpBlk);
  }
  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
  gArch80 = prop.major >= 8;
  gArch89 = prop.major == 8 && prop.minor == 9;
  if (const char* e = getenv("NANO_ARCH89")) { gArch89 = atoi(e) != 0; printf("NANO_ARCH89 override: gArch89=%d%c", (int)gArch89, 10); }
  gArch120 = prop.major == 12;
  if (const char* e = getenv("NANO_ARCH120")) { gArch120 = atoi(e) != 0; printf("NANO_ARCH120 override: gArch120=%d%c", (int)gArch120, 10); }
  if (!gArch80) { printf("nano_vit needs sm_80+\n"); return 1; }
  if (gInt4) {
    if (prop.major != 8) { printf("--int4 needs the sm_80/86/89 INT4 mma (mma.m16n8k64.s4); this is sm_%d%d\n", prop.major, prop.minor); return 1; }
    if (gFc2Pass || gFc2Bfp || gFc2Rot != 64) { printf("--int4 runs the fused H_64 fc1 producer only (no --fc2-pass / --fc2-bfp / --fc2-rot 128)\n"); return 1; }
  }
  if (gRot && gInt8Fc2) {
    gFc2U8 = false;                       // the rotated GELU output has no sign bound
    if (gFc2Rot != 64 && gFc2Rot != 128) { printf("--fc2-rot must be 64 or 128\n"); return 1; }
  }
  if (gFc2Pass && !(gRot && gInt8Fc2 && !gFc2Bfp)) { printf("--fc2-pass needs --int8-fc2 with --rot / --gptq (not --fc2-bfp)\n"); return 1; }
  if (gFc2Pass) printf("fc2-pass: fc1 writes fp16, k_rotquant_fc2<%d> makes the int8 fc2 operand\n", gFc2Rot);
  if (gFc2Bfp) {
    if (!gInt8Fc2) { printf("--fc2-bfp needs --int8-fc2\n"); return 1; }
    if (gRot && gFc2Rot != 64) { printf("--fc2-bfp needs --fc2-rot 64\n"); return 1; }
    gFc2U8 = false;
  }
  printf("GPU: %s sm_%d%d, SMs=%d, accmode=%d gemms=%s%s%s\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount, gAccMode,
         gInt4 ? "int4(qkv+fc1+fc2)" : gInt8Fc2 ? "int8(qkv+fc1+fc2)" : gInt8 ? "int8(qkv+fc1)" : "fp16",
         gRot ? " rot" : "", gGptq.empty() ? "" : " gptq");

  loadCfg(model);
  loadWeights(model);
  allocAll();
  if (gFc2Pass && gMp % 128) { printf("--fc2-pass needs the token count padded to 128 rows (gMp = %d)\n", gMp); return 1; }
  CL(cublasLtCreate(&LT));
  CK(cudaMalloc(&gWs, gWsBytes));
  buildPlans();
  if (gInt8) buildInt8Weights();
  if (gDumpBlk < 0 || gDumpBlk >= cfg.depth) { printf("NANO_DUMP_BLOCK out of range\n"); return 1; }
  if (!gFc2StaticFile.empty()) buildFc2Static();
  else if (gFc2Fp16B0) printf("fc2-fp16-b0: block 0's fc2 in fp16 (the other blocks on the dynamic fc2)\n");

  const size_t inBytes = (size_t)gB * 3 * cfg.img * cfg.img;
  std::vector<char> hin;
  auto loadInput = [&](const std::string& path) {
    size_t n = readFile(path.c_str(), hin);
    if (n > inBytes) n = inBytes;
    CK(cudaMemcpy(dIn, hin.data(), n, cudaMemcpyHostToDevice));
    if (n < inBytes) CK(cudaMemset(dIn + n, 0, inBytes - n));
    return n;
  };
  if (!input.empty()) loadInput(input);

  cudaStream_t s; CK(cudaStreamCreate(&s));
  forward(s); CK(cudaStreamSynchronize(s));       // first pass: func attributes, dumps
  CK(cudaGetLastError());
  gDump.clear();
  gFirstPass = false;

  if (gProf) {
    for (int r = 0; r < 3; ++r) { gEvN = 0; forward(s); CK(cudaStreamSynchronize(s)); }
    profReport();
    gProf = false;
  }

  cudaGraphExec_t exec = nullptr;
  if (useGraph) {
    cudaGraph_t g;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    forward(s);
    CK(cudaStreamEndCapture(s, &g));
    CK(cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0));
    CK(cudaGraphDestroy(g));
  }
  auto once = [&] { if (exec) CK(cudaGraphLaunch(exec, s)); else forward(s); };

  const bool feat = cfg.ncls == 0;
  const size_t logBytes = feat ? (size_t)gB * cfg.S * cfg.C * 2 : (size_t)gB * cfg.ncls * 2;
  const half* outBuf = feat ? Xn : LOG;
  const char* outExt = feat ? ".tokens" : ".logits";
  std::vector<char> hlog(logBytes);
  if (!inputList.empty()) {               // accuracy run: one batch file per line -> logits per file
    if (outDir.empty()) { printf("--input-list needs --out-dir\n"); return 1; }
    std::ifstream lf(inputList); std::string line; int nfile = 0;
    while (std::getline(lf, line)) {
      if (line.empty()) continue;
      loadInput(line);
      once(); CK(cudaStreamSynchronize(s));
      CK(cudaMemcpy(hlog.data(), outBuf, logBytes, cudaMemcpyDeviceToHost));
      std::ofstream of(outDir + "/" + baseName(line) + outExt, std::ios::binary);
      of.write(hlog.data(), logBytes);
      if (++nfile % 100 == 0) { printf("  %d batches\n", nfile); fflush(stdout); }
    }
    printf("wrote %s for %d batches to %s\n", feat ? "tokens" : "logits", nfile, outDir.c_str());
    return 0;
  }

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
  printf("latency_ms/batch mean=%.4f p50=%.4f min=%.4f  per_image=%.4f ms  %.1f img/s  (batch %d, iters %d, graph %d)\n",
         sum / ts.size(), ts[ts.size() / 2], ts[0], ts[ts.size() / 2] / gB, 1000.0 * gB / ts[ts.size() / 2], gB, iters, useGraph);
  if (gFc2Bfp && gBfpEmax >= 0) {
    unsigned long long v = 0; int msp = 0; int sng[64];
    CK(cudaMemcpyFromSymbol(&v, g_bfp_viol, sizeof v));
    CK(cudaMemcpyFromSymbol(&msp, g_bfp_maxspread, sizeof msp));
    CK(cudaMemcpyFromSymbol(sng, g_bfp_spread_ng, sizeof sng));
    printf("bfp: emax=%d fix=%d  max per-row exponent spread=%d  clamped (row,group) pairs "
           "per forward=%.3f\n", gBfpEmax, gBfpCheck ? 0 : 1, msp,
           (double)v / (iters + warmup + 1));
    printf("bfp: spread by group count:");
    for (int i = 0; i < 64; ++i) if (sng[i]) printf("  NG=%d:%d", i, sng[i]);
    unsigned long long hist[64]; CK(cudaMemcpyFromSymbol(hist, g_bfp_hist, sizeof hist));
    unsigned long long tot = 0; for (int i = 0; i < 64; ++i) tot += hist[i];
    printf("%cbfp: rows by spread (of %llu):", 10, tot);
    for (int i = 0; i < 64; ++i) if (hist[i]) printf("  %d:%.4f%%", i, 100.0 * hist[i] / (tot ? tot : 1));
    printf("\n");
  }
  if (!outDir.empty()) {
    CK(cudaMemcpy(hlog.data(), outBuf, logBytes, cudaMemcpyDeviceToHost));
    std::ofstream of(outDir + (feat ? "/tokens.bin" : "/logits.bin"), std::ios::binary); of.write(hlog.data(), logBytes);
    printf("dumped %s to %s\n", feat ? "tokens" : "logits", outDir.c_str());
  }
  return 0;
}
