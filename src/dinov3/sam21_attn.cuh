// sam21_attn.cuh — fused flash attention for the hiera trunk.
//
// The qkv GEMM emits every head padded 72 -> 80 channels, so Q, K and V are read as
// strided wmma fragments straight out of the qkv buffer (ld = 3*Cp): no split kernel, no
// Q/K/V staging traffic, and the [Tq,Tk] scores never reach global memory either.  O stays
// in accumulator fragments and Q in matrix_a fragments for the whole KV loop, so shared
// memory holds little more than the current score tile.
//
// STAGE picks how K/V reach the tensor cores:
//   * STAGE=1 copies each KV tile into shared once per block, which pays off when TK is
//     large enough that the four warps would otherwise thrash L1 (the global-attention
//     blocks, TK=4096);
//   * STAGE=0 loads the fragments directly from global and makes every warp independent,
//     so the kernel needs no __syncthreads at all -- better for the short windows.
//
// The online-softmax correction is lazy: a row only re-bases its running max when it
// grows past m + MAXSLACK, turning the fragment rescale from a per-tile cost into a
// once-or-twice-per-row cost.
//
// Query rows past TQ read a neighbouring window's tokens rather than being masked; the
// results stay finite and are simply not written out.

template<int BQ, int NW, int BK, int STAGE>
__global__ __launch_bounds__(NW * 32) void k_attn(
    const half* __restrict__ qkv,     // [T, 3*Cp] padded qkv, Q third pre-scaled
    const half* __restrict__ Qpool,   // pooled Q for q_pool blocks, else nullptr
    half* __restrict__ Xout,          // [Tq, Co] token-major attention output
    int ws2, int ws2q, int heads, int Cp, int Co, int TQ, int TK) {
  const int LDK = DHP + 8;      // staged K/V row stride, halves
  const int LDS = BK + 4;       // score row stride, floats (multiple of 4 for wmma)
  const int LDP = BK + 8;       // P row stride, halves    (multiple of 8 for wmma)
  const int LDO = 20;           // rescale-scratch row stride, floats
  const int KVS = STAGE ? 2 * BK * LDK : 0;

  extern __shared__ char smem[];
  half*  Ks = (half*)smem;                        // [BK][LDK] (STAGE only)
  half*  Vs = Ks + (STAGE ? BK * LDK : 0);
  float* Ss = (float*)((half*)smem + KVS);        // [NW][16][LDS] scores, then O scratch
  half*  Ps = (half*)(Ss + NW * 16 * LDS);        // [NW][16][LDP]
  float* MLC = (float*)(Ps + NW * 16 * LDP);      // [NW][3][16] max, denom, correction

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, nthr = NW * 32, warp = tid >> 5, lane = tid & 31;
  const int row0 = q0 + warp * 16;                // first query row owned by this warp
  const int ldk = 3 * Cp;
  const half* K = qkv + (size_t)(w * ws2) * ldk + Cp + h * DHP;
  const half* V = K + Cp;

  float* Sw = Ss + warp * 16 * LDS;
  half*  Pw = Ps + warp * 16 * LDP;
  float* Mw = MLC + warp * 48;
  float* Lw = Mw + 16;
  float* Cw = Lw + 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> qf[DHP / 16];
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> oa[DHP / 16];
  {
    const half* Qrow; int ldq;
    if (Qpool) { Qrow = Qpool + ((size_t)b * ws2q + row0) * DHP; ldq = DHP; }
    else       { Qrow = qkv + (size_t)(w * ws2 + row0) * ldk + h * DHP; ldq = ldk; }
#pragma unroll
    for (int kk = 0; kk < DHP / 16; ++kk) {
      wmma::load_matrix_sync(qf[kk], Qrow + kk * 16, ldq);
      wmma::fill_fragment(oa[kk], 0.f);
    }
  }
  if (lane < 16) { Mw[lane] = -1e30f; Lw[lane] = 0.f; }
  if (STAGE) __syncthreads(); else __syncwarp();

  const int LPR = BK / 16;          // lanes cooperating on one score row

  // Staged path: the KV tile for step t+1 is issued into registers right after the barrier
  // that publishes tile t, so its global latency hides behind the QK^T/softmax/PV work
  // instead of stalling every warp at the top of the loop.
  const int NPT = (BK * (DHP / 8) + NW * 32 - 1) / (NW * 32);
  float4 pk[STAGE ? NPT : 1], pv[STAGE ? NPT : 1];
  if (STAGE) {
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
      int i = tid + j * nthr;
      if (i < BK * (DHP / 8)) {
        int r = i / (DHP / 8), c = i % (DHP / 8);
        pk[j] = ((const float4*)(K + (size_t)r * ldk))[c];
        pv[j] = ((const float4*)(V + (size_t)r * ldk))[c];
      }
    }
  }

  for (int k0 = 0; k0 < TK; k0 += BK) {
    if (STAGE) {
#pragma unroll
      for (int j = 0; j < NPT; ++j) {
        int i = tid + j * nthr;
        if (i < BK * (DHP / 8)) {
          int r = i / (DHP / 8), c = i % (DHP / 8);
          ((float4*)(Ks + r * LDK))[c] = pk[j];
          ((float4*)(Vs + r * LDK))[c] = pv[j];
        }
      }
      __syncthreads();
      if (k0 + BK < TK) {
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
          int i = tid + j * nthr;
          if (i < BK * (DHP / 8)) {
            int r = i / (DHP / 8), c = i % (DHP / 8);
            pk[j] = ((const float4*)(K + (size_t)(k0 + BK + r) * ldk))[c];
            pv[j] = ((const float4*)(V + (size_t)(k0 + BK + r) * ldk))[c];
          }
        }
      }
    }

    // ---- S = Q . K^T ----
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[BK / 16];
#pragma unroll
    for (int n = 0; n < BK / 16; ++n) wmma::fill_fragment(acc[n], 0.f);
#pragma unroll
    for (int n = 0; n < BK / 16; ++n) {
      const half* Kt = STAGE ? (Ks + n * 16 * LDK) : (K + (size_t)(k0 + n * 16) * ldk);
      int lk = STAGE ? LDK : ldk;
#pragma unroll
      for (int kk = 0; kk < DHP / 16; ++kk) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> bf;
        wmma::load_matrix_sync(bf, Kt + kk * 16, lk);
        wmma::mma_sync(acc[n], qf[kk], bf, acc[n]);
      }
    }
#pragma unroll
    for (int n = 0; n < BK / 16; ++n)
      wmma::store_matrix_sync(Sw + n * 16, acc[n], LDS, wmma::mem_row_major);
    __syncwarp();

    // ---- running row max / correction; scores are parked in registers ----
    float v[16]; int r = 0, cs = 0;
    float corr = 1.f;
    bool active = (lane / LPR) < 16;
    if (active) {
      r = lane / LPR; cs = (lane % LPR) * 16;
      float mx = -1e30f;
#pragma unroll
      for (int c = 0; c < 16; ++c) { v[c] = Sw[r * LDS + cs + c]; mx = fmaxf(mx, v[c]); }
#pragma unroll
      for (int o = 1; o < LPR; o <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o));
      // Slack in the running max: only re-base when a row grows past m + MAXSLACK.
      // exp(s - m) then stays <= e^8 = 2981, well inside fp16 for P, and O accumulates
      // in fp32 -- free accuracy-wise, and it makes the rescale rare.
      const float MAXSLACK = 8.f;
      float mo = Mw[r];
      float mn = (mx > mo + MAXSLACK || mo < -1e29f) ? mx : mo;
      corr = (mo < -1e29f) ? 0.f : __expf(mo - mn);
      if (cs == 0) { Mw[r] = mn; Cw[r] = corr; }
      float sum = 0.f;
#pragma unroll
      for (int c = 0; c < 16; ++c) { v[c] = __expf(v[c] - mn); sum += v[c]; }
#pragma unroll
      for (int o = 1; o < LPR; o <<= 1) sum += __shfl_xor_sync(0xffffffff, sum, o);
      if (cs == 0) Lw[r] = Lw[r] * corr + sum;
    }
    // Sw is free now (the scores live in v[]); reuse it as the fragment rescale scratch.
    bool need = __any_sync(0xffffffff, active && corr != 1.f && k0 > 0);
    __syncwarp();
    if (need) {
#pragma unroll
      for (int n = 0; n < DHP / 16; ++n) {
        wmma::store_matrix_sync(Sw, oa[n], LDO, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * LDO; i += 32)
          if ((i % LDO) < 16) Sw[i] *= Cw[i / LDO];
        __syncwarp();
        wmma::load_matrix_sync(oa[n], Sw, LDO, wmma::mem_row_major);
        __syncwarp();
      }
    }
    if (active)
#pragma unroll
      for (int c = 0; c < 16; ++c) Pw[r * LDP + cs + c] = __float2half(v[c]);
    __syncwarp();

    // ---- O += P . V ----
#pragma unroll
    for (int kk = 0; kk < BK / 16; ++kk) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> af;
      wmma::load_matrix_sync(af, Pw + kk * 16, LDP);
      const half* Vt = STAGE ? (Vs + kk * 16 * LDK) : (V + (size_t)(k0 + kk * 16) * ldk);
      int lv = STAGE ? LDK : ldk;
#pragma unroll
      for (int n = 0; n < DHP / 16; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> bf;
        wmma::load_matrix_sync(bf, Vt + n * 16, lv);
        wmma::mma_sync(oa[n], af, bf, oa[n]);
      }
    }
    if (STAGE) __syncthreads(); else __syncwarp();
  }

  // ---- normalise and scatter straight into the token-major activation buffer ----
#pragma unroll
  for (int n = 0; n < DHP / 16; ++n) {
    if (n * 16 >= DH) break;
    wmma::store_matrix_sync(Sw, oa[n], LDO, wmma::mem_row_major);
    __syncwarp();
    for (int i = lane; i < 16 * LDO; i += 32) {
      int rr = i / LDO, cc = i % LDO, col = n * 16 + cc;
      if (cc < 16 && col < DH && row0 + rr < TQ)
        Xout[(size_t)(w * ws2q + row0 + rr) * Co + h * DH + col] =
            __float2half(Sw[i] / Lw[rr]);
    }
    __syncwarp();
  }
}

// pooled Q for the three q_pool blocks: 2x2 max over each window, padded head layout
__global__ void k_pool_q(const half* __restrict__ qkv, half* __restrict__ Qb,
                         int nW, int ws, int ws2, int heads, int Cp) {
  int wsq = ws / 2, ws2q = ws2 / 4;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nW * heads * ws2q * (DHP / 2)) return;
  int d2 = idx % (DHP / 2), rest = idx / (DHP / 2);
  int tq = rest % ws2q, b = rest / ws2q;
  int h = b % heads, w = b / heads;
  int oh = tq / wsq, ow = tq - oh * wsq;
  const half2* base = (const half2*)(qkv + (size_t)(w * ws2) * (3 * Cp) + h * DHP);
  int row3 = (3 * Cp) / 2;
  float2 mx = make_float2(-65504.f, -65504.f);
#pragma unroll
  for (int a = 0; a < 2; ++a)
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      float2 p = __half22float2(base[(size_t)((2 * oh + a) * ws + (2 * ow + c)) * row3 + d2]);
      mx.x = fmaxf(mx.x, p.x); mx.y = fmaxf(mx.y, p.y);
    }
  ((half2*)(Qb + (size_t)(b * ws2q + tq) * DHP))[d2] = __floats2half2_rn(mx.x, mx.y);
}
