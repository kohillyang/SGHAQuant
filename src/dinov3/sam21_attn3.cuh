#pragma once
// sam21_attn3.cuh -- experimental variants of k_attn2 / k_attn2m for the d = 64 shapes of the
// SAM 3 image encoder (576-key windows, 5184-key global) and the plain ViTs.  Requires
// sam21_attn2.cuh first (namespace attn2 helpers, attn2_dhp / attn2_lds).
//
// Why another kernel.  ncu on k_attn2<64,4,32,2,false,true> at the 9x16 576x576 shape (3090):
// 56 % tensor-pipe utilisation, 15 warps/SM, "No Eligible" 61 %, and 363 issued instructions
// per warp per key tile of which only 48 are mma + ldmatrix.  The kernel is issue/latency
// bound on the *scalar* half of the flash loop, not on the tensor pipe and not on L2 (raising
// the query block from 64 to 192 cuts L2 traffic from 313 MB to 125 MB per call and makes the
// kernel 34 % SLOWER).  The two levers this header adds:
//
//  FOLD : -1 drops the fp32 output accumulator altogether: the fp16 mma accumulators ARE the
//         output, they are never folded and the rescale on a moving row max is one half2
//         multiply per register instead of two float multiplies plus the half2.  This frees
//         MT*NO*4 registers (64 at DH = 64, MT = 2), which is what makes a 128-query block fit
//         in four CTAs per SM, and removes the ~80 instructions per warp per key tile that the
//         fold costs.  The output is then a sum of TK products rounded to fp16 at every step.
//         Otherwise: how many key tiles of P.V accumulate in the fp16 mma accumulators (oh) before they
//         are folded into the fp32 output accumulator (o).  FOLD = 1 is exactly k_attn2 /
//         k_attn2m; FOLD = 0 folds once after the loop.  One fold costs NO*(4 cvt + 4 add)
//         per row tile = 64 instructions per warp per key tile at DH = 64, plus NO*2 moves to
//         re-zero oh, i.e. about 22 % of everything the kernel issues.  The fp16 slice then
//         holds FOLD key tiles instead of one, so its magnitude grows by at most FOLD and the
//         rounding error of the slice grows with it -- see the accuracy note below.
//         When the running row max moves, o AND oh are rescaled; oh is rescaled with one
//         half2 multiply per register instead of two float multiplies.
//
//  PIPE : compute the scores of key tile t+1 *before* the softmax and the P.V of tile t, so
//         the tensor pipe has NKS*NT independent mma to chew on while the exp2 dependency
//         chain of the softmax runs.  Needs a 3-deep cp.async ring (tile t is still being
//         read by P.V while tile t+1 is read by Q.K^T) and one extra __syncthreads per tile.
//         The key-tile loop is unrolled by two so the two score register sets alternate by
//         name and no register copy happens at the back edge.
//
//  DIAG : timing-only ablations, results are WRONG, never dispatched:
//         1 = no row max / exp2 / row sum (P = the raw scores), 2 = no P.V, 3 = no Q.K^T,
//         4 = cp.async staging + barriers only (no ldmatrix, no mma, no softmax),
//         6 = everything except the cp.async of the tiles after the prologue (no L2 traffic).
//
// Everything else -- the packed-qkv addressing, the cp.async staging, the tail-tile key mask,
// the Q-in-registers layout, the clamp of query rows past TQ and the output scatter -- is
// character for character the contract of k_attn2 / k_attn2m.

// WSKIP : a query block of BQ = NW*16*MT rows that overhangs TQ leaves whole warps with no
//         real row (with MT tiles per warp a warp owns 16*MT consecutive rows, so the overhang
//         is warp granular).  Those warps still stage K/V with cp.async and still hit every
//         __syncthreads, but skip Q.K^T, the softmax, P.V and the epilogue.  576 = 4.5 x 128:
//         without this, a 128-query block computes 640 rows for 576 useful ones.
template<int DH, int NW, int MT, int BK, int STAGES, int FOLD, bool PIPE,
         int MINB = 1, int DIAG = 0, bool QF16 = false, bool WSKIP = false>
__global__ __launch_bounds__(NW * 32, MINB) void k_attn3(
    const half* __restrict__ qkv, const half* __restrict__ Qpool, half* __restrict__ Xout,
    int ws2, int ws2q, int heads, int Cp, int Co, int TQ, int TK) {
#if ATTN2_ENABLED
  using namespace attn2;
  constexpr int BQ = NW * 16 * MT;
  constexpr int DHP = attn2_dhp(DH);
  constexpr int LDS = attn2_lds(DH);
  constexpr int TILE = BK * LDS;
  constexpr int NT = BK / 8, NK = BK / 16, NO = DH / 8;
  constexpr int NKS = DH / 16;
  constexpr bool KTAIL = (DH % 16) == 8;
  constexpr int CHR = DH / 8;
  constexpr float LOG2E = 1.4426950408889634f;
  static_assert(BK % 16 == 0, "BK must be a multiple of 16");
  static_assert(DH % 8 == 0 && DH >= 16, "DH must be a multiple of 8");
  static_assert(LDS % 16 == 8, "staged row stride must be 8 mod 16 for conflict-free ldmatrix");
  static_assert(!PIPE || STAGES >= 3, "PIPE keeps tiles t and t+1 live, so the ring needs 3");
  static_assert(FOLD <= 0 || (FOLD & (FOLD - 1)) == 0, "FOLD must be -1, 0 or a power of two");

  extern __shared__ __align__(16) char smem_raw[];
  half* Ks = (half*)smem_raw;
  half* Vs = Ks + STAGES * TILE;

  const int b = blockIdx.y, q0 = blockIdx.x * BQ;
  const int w = b / heads, h = b - w * heads;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, q4 = lane & 3;
  const int ldk = 3 * Cp;
  const half* K = qkv + (size_t)(w * ws2) * ldk + Cp + h * DHP;
  const int row0 = q0 + warp * 16 * MT;
  const bool wact = !WSKIP || row0 < TQ;      // warp-uniform: does this warp own a real row?

  uint32_t qa[MT][NKS][4], qt[MT][KTAIL ? 2 : 1];
  if (wact) {
    const half* Qrow; int ldq;
    if (Qpool) { Qrow = Qpool + (size_t)b * ws2q * DHP; ldq = DHP; }
    else       { Qrow = qkv + (size_t)(w * ws2) * ldk + h * DHP; ldq = ldk; }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      const half* p0 = Qrow + (size_t)min(row0 + mt * 16 + g, TQ - 1) * ldq + 2 * q4;
      const half* p1 = Qrow + (size_t)min(row0 + mt * 16 + g + 8, TQ - 1) * ldq + 2 * q4;
#pragma unroll
      for (int kk = 0; kk < NKS; ++kk) {
        qa[mt][kk][0] = ld32(p0 + kk * 16);     qa[mt][kk][1] = ld32(p1 + kk * 16);
        qa[mt][kk][2] = ld32(p0 + kk * 16 + 8); qa[mt][kk][3] = ld32(p1 + kk * 16 + 8);
      }
      if (KTAIL) { qt[mt][0] = ld32(p0 + NKS * 16); qt[mt][1] = ld32(p1 + NKS * 16); }
    }
  }

  constexpr int NCH = BK * CHR, NPT = (NCH + NW * 32 - 1) / (NW * 32);
  const half* gsrc[NPT]; uint32_t sdst[NPT];
#pragma unroll
  for (int j = 0; j < NPT; ++j) {
    int i = tid + j * NW * 32; if (i >= NCH) i = NCH - 1;
    int r = i / CHR, c = i - r * CHR;
    gsrc[j] = K + (size_t)r * ldk + c * 8;
    sdst[j] = smem_u32(Ks) + (uint32_t)(r * LDS + c * 8) * 2;
  }
  const uint32_t vsoff = (uint32_t)(STAGES * TILE) * 2;
  auto load_tile = [&](int stage, int k0) {
    const int koff = k0 * ldk;
    const uint32_t soff = (uint32_t)(stage * TILE) * 2;
#pragma unroll
    for (int j = 0; j < NPT; ++j) {
      if (NCH % (NW * 32) == 0 || tid + j * NW * 32 < NCH) {
        cp_async16(sdst[j] + soff, gsrc[j] + koff);
        cp_async16(sdst[j] + soff + vsoff, gsrc[j] + koff + Cp);
      }
    }
  };

  constexpr bool OHONLY = (FOLD < 0);        // the fp16 slices are the output accumulator
  float o[OHONLY ? 1 : MT][OHONLY ? 1 : NO][4];
  uint32_t oh[MT][NO][2];
  float m[MT][2], l[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      if (!OHONLY) { o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f; }
      oh[mt][n][0] = oh[mt][n][1] = 0u;
    }
    m[mt][0] = m[mt][1] = -1e30f; l[mt][0] = l[mt][1] = 0.f;
  }

  const int ntiles = (TK + BK - 1) / BK;
  const int tmask = (TK % BK) ? (ntiles - 1) : -1;
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) { if (s < ntiles) load_tile(s, s * BK); cp_commit(); }
  const int mi = lane >> 3, ri = lane & 7;

  struct Sc { float v[MT][NT][4]; };
  Sc sa, sb;

  // ---- S = Q . K^T for one staged tile, all MT row tiles from one pass over the fragments ----
  auto qk = [&](const half* Kt, Sc& ss) {
    uint32_t sh[QF16 ? MT : 1][QF16 ? NT : 1][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (QF16) { sh[mt][n][0] = sh[mt][n][1] = 0u; }
        else { ss.v[mt][n][0] = ss.v[mt][n][1] = ss.v[mt][n][2] = ss.v[mt][n][3] = 0.f; }
      }
    if (DIAG == 3) { if (QF16) { for (int mt = 0; mt < MT; ++mt) for (int n = 0; n < NT; ++n) ss.v[mt][n][0]=ss.v[mt][n][1]=ss.v[mt][n][2]=ss.v[mt][n][3]=0.f; } return; }
#pragma unroll
    for (int n2 = 0; n2 < NT / 2; ++n2) {
      const half* kb = Kt + (16 * n2 + ri + ((mi & 2) ? 8 : 0)) * LDS + ((mi & 1) ? 8 : 0);
      uint32_t bb[4];
#pragma unroll
      for (int kk = 0; kk < NKS; ++kk) {
        ldsm_x4(bb, smem_u32(kb + kk * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          if (QF16) {
            mma16816_h(sh[mt][2 * n2], qa[mt][kk], bb[0], bb[1]);
            mma16816_h(sh[mt][2 * n2 + 1], qa[mt][kk], bb[2], bb[3]);
          } else {
            mma16816(ss.v[mt][2 * n2], qa[mt][kk], bb[0], bb[1]);
            mma16816(ss.v[mt][2 * n2 + 1], qa[mt][kk], bb[2], bb[3]);
          }
        }
      }
      if (KTAIL) {
        ldsm_x2(bb, smem_u32(Kt + (16 * n2 + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          if (QF16) {
            mma1688_h(sh[mt][2 * n2], qt[mt], bb[0]);
            mma1688_h(sh[mt][2 * n2 + 1], qt[mt], bb[1]);
          } else {
            mma1688(ss.v[mt][2 * n2], qt[mt], bb[0]);
            mma1688(ss.v[mt][2 * n2 + 1], qt[mt], bb[1]);
          }
        }
      }
    }
    if (QF16) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          float2 a = unpack2(sh[mt][n][0]), c = unpack2(sh[mt][n][1]);
          ss.v[mt][n][0] = a.x; ss.v[mt][n][1] = a.y; ss.v[mt][n][2] = c.x; ss.v[mt][n][3] = c.y;
        }
    }
  };

  // ---- fold the fp16 P.V slices into the fp32 accumulator and re-zero them ----
  auto fold = [&]() {
    if (OHONLY) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int n = 0; n < NO; ++n) {
        float2 a = unpack2(oh[mt][n][0]), d = unpack2(oh[mt][n][1]);
        o[mt][n][0] += a.x; o[mt][n][1] += a.y; o[mt][n][2] += d.x; o[mt][n][3] += d.y;
        oh[mt][n][0] = oh[mt][n][1] = 0u;
      }
  };

  // ---- one key tile: mask, softmax, optional pipelined Q.K^T of the next tile, P.V ----
  auto step = [&](int t, Sc& cur, Sc& nxt) {
    const int ct = t % STAGES;
    if (PIPE) {
      __syncthreads();                      // every warp is done with stage (t-1) % STAGES
      if (t + STAGES - 1 < ntiles && DIAG != 6) { load_tile((t + STAGES - 1) % STAGES, (t + STAGES - 1) * BK); }
      cp_commit();
      if (t + 1 < ntiles) { cp_wait<STAGES - 2>(); __syncthreads(); }
    } else {
      cp_wait<STAGES - 2>();
      __syncthreads();
      if (t + STAGES - 1 < ntiles && DIAG != 6) load_tile((t + STAGES - 1) % STAGES, (t + STAGES - 1) * BK);
      cp_commit();
      if (DIAG != 4) qk(Ks + ct * TILE, cur);
    }
    if (DIAG == 4) return;
    if (WSKIP && !wact) return;               // staged the tile, nothing to compute
    // the scores of the next tile are issued first: independent of everything below, so the
    // tensor pipe stays busy across the softmax's exp2 chain
    if (PIPE && t + 1 < ntiles) qk(Ks + ((t + 1) % STAGES) * TILE, nxt);

    if (t == tmask) {
      const int kb0 = t * BK + 2 * q4;
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (kb0 + n * 8     >= TK) { cur.v[mt][n][0] = -1e30f; cur.v[mt][n][2] = -1e30f; }
          if (kb0 + n * 8 + 1 >= TK) { cur.v[mt][n][1] = -1e30f; cur.v[mt][n][3] = -1e30f; }
        }
    }

    uint32_t pa[MT][NK][4];
    float cf[MT][2];
    bool moved = false;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
      if (DIAG == 1) {                      // ablation: no max, no exp2, no row sum
        cf[mt][0] = cf[mt][1] = 1.f;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          pa[mt][n >> 1][(n & 1) * 2]     = pack2(cur.v[mt][n][0] * 1e-3f, cur.v[mt][n][1] * 1e-3f);
          pa[mt][n >> 1][(n & 1) * 2 + 1] = pack2(cur.v[mt][n][2] * 1e-3f, cur.v[mt][n][3] * 1e-3f);
        }
        l[mt][0] = l[mt][1] = 1.f;
        continue;
      }
      float tm0 = -1e30f, tm1 = -1e30f;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        tm0 = fmaxf(tm0, fmaxf(cur.v[mt][n][0], cur.v[mt][n][1]));
        tm1 = fmaxf(tm1, fmaxf(cur.v[mt][n][2], cur.v[mt][n][3]));
      }
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 1));
      tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffff, tm0, 2));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 1));
      tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffff, tm1, 2));
      const float mn0 = fmaxf(m[mt][0], tm0), mn1 = fmaxf(m[mt][1], tm1);
      cf[mt][0] = exp2f((m[mt][0] - mn0) * LOG2E);
      cf[mt][1] = exp2f((m[mt][1] - mn1) * LOG2E);
      m[mt][0] = mn0; m[mt][1] = mn1;
      const float ms0 = mn0 * LOG2E, ms1 = mn1 * LOG2E;
      float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        float p0 = exp2f(fmaf(cur.v[mt][n][0], LOG2E, -ms0)), p1 = exp2f(fmaf(cur.v[mt][n][1], LOG2E, -ms0));
        float p2 = exp2f(fmaf(cur.v[mt][n][2], LOG2E, -ms1)), p3 = exp2f(fmaf(cur.v[mt][n][3], LOG2E, -ms1));
        rs0 += p0 + p1; rs1 += p2 + p3;
        pa[mt][n >> 1][(n & 1) * 2]     = pack2(p0, p1);
        pa[mt][n >> 1][(n & 1) * 2 + 1] = pack2(p2, p3);
      }
      l[mt][0] = l[mt][0] * cf[mt][0] + rs0; l[mt][1] = l[mt][1] * cf[mt][1] + rs1;
      moved |= (cf[mt][0] != 1.f) | (cf[mt][1] != 1.f);
    }
    if (DIAG != 1 && __any_sync(0xffffffff, moved)) {
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const half2 h0 = __floats2half2_rn(cf[mt][0], cf[mt][0]);
        const half2 h1 = __floats2half2_rn(cf[mt][1], cf[mt][1]);
#pragma unroll
        for (int n = 0; n < NO; ++n) {
          if (!OHONLY) {
            o[mt][n][0] *= cf[mt][0]; o[mt][n][1] *= cf[mt][0];
            o[mt][n][2] *= cf[mt][1]; o[mt][n][3] *= cf[mt][1];
          }
          half2 a = *(half2*)&oh[mt][n][0], d = *(half2*)&oh[mt][n][1];
          a = __hmul2(a, h0); d = __hmul2(d, h1);
          oh[mt][n][0] = *(uint32_t*)&a; oh[mt][n][1] = *(uint32_t*)&d;
        }
      }
    }

    if (DIAG != 2) {
      const half* Vt = Vs + ct * TILE;
#pragma unroll
      for (int kk = 0; kk < NK; ++kk) {
        const half* vb = Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + ((mi & 2) ? 8 : 0);
        uint32_t bb[4];
#pragma unroll
        for (int n2 = 0; n2 < NKS; ++n2) {
          ldsm_x4_t(bb, smem_u32(vb + n2 * 16));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) {
            mma16816_h(oh[mt][2 * n2], pa[mt][kk], bb[0], bb[1]);
            mma16816_h(oh[mt][2 * n2 + 1], pa[mt][kk], bb[2], bb[3]);
          }
        }
        if (KTAIL) {
          ldsm_x2_t(bb, smem_u32(Vt + (16 * kk + ri + ((mi & 1) ? 8 : 0)) * LDS + NKS * 16));
#pragma unroll
          for (int mt = 0; mt < MT; ++mt) mma16816_h(oh[mt][NO - 1], pa[mt][kk], bb[0], bb[1]);
        }
      }
    }
    if (FOLD == 1 || (FOLD > 1 && (t & (FOLD - 1)) == FOLD - 1)) fold();
    (void)nxt;
  };

  if (PIPE) {
    cp_wait<STAGES - 2>();
    __syncthreads();
    if (wact) qk(Ks, sa);                        // scores of tile 0 from stage 0
  }
  if (PIPE) {
    for (int t = 0; t < ntiles; t += 2) {        // unrolled by two: sa / sb alternate by name
      step(t, sa, sb);
      if (t + 1 < ntiles) step(t + 1, sb, sa);
    }
  } else {
    for (int t = 0; t < ntiles; ++t) step(t, sa, sb);   // sb is never read, one score set
  }
  if (FOLD != 1) fold();
  if (WSKIP && !wact) return;


#pragma unroll
  for (int mt = 0; mt < MT; ++mt) {
    float l0 = l[mt][0], l1 = l[mt][1];
    l0 += __shfl_xor_sync(0xffffffff, l0, 1); l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1); l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    const float inv0 = 1.f / l0, inv1 = 1.f / l1;
    const int r0 = row0 + mt * 16 + g, r1 = r0 + 8;
    half* out0 = Xout + (size_t)(w * ws2q + r0) * Co + h * DH + 2 * q4;
    half* out1 = Xout + (size_t)(w * ws2q + r1) * Co + h * DH + 2 * q4;
#pragma unroll
    for (int n = 0; n < NO; ++n) {
      float a0, a1, a2, a3;
      if (OHONLY) { float2 a = unpack2(oh[mt][n][0]), d = unpack2(oh[mt][n][1]);
                    a0 = a.x; a1 = a.y; a2 = d.x; a3 = d.y; }
      else        { a0 = o[mt][n][0]; a1 = o[mt][n][1]; a2 = o[mt][n][2]; a3 = o[mt][n][3]; }
      if (r0 < TQ) *(uint32_t*)(out0 + 8 * n) = pack2(a0 * inv0, a1 * inv0);
      if (r1 < TQ) *(uint32_t*)(out1 + 8 * n) = pack2(a2 * inv1, a3 * inv1);
    }
  }
#endif  // ATTN2_ENABLED
}
