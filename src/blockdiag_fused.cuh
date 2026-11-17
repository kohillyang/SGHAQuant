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

#pragma once
#include <mma.h>
using namespace nvcuda;
template<int WARPS=8>
__global__ void k_blockdiag_fused(const half*x,uint8_t*q,float*sc,const float*gamma,const float*beta,
 const half*B,int K,int a,int b,int rows,float clip,int group,const int*map,const int*factorMap=nullptr){
 extern __shared__ __align__(32) unsigned char storage[];
 int LD=K+8;half*raw=(half*)storage;const half*mid=x;
 half*aa=raw+8*LD;half*bb=aa+WARPS*256;float*cc=(float*)(bb+WARPS*256);
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,row0=blockIdx.x*8;
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;
 int tiles;
 if(gamma){for(int r=warp;r<8;r+=WARPS){int row=row0+r,src=row<rows?(map?map[row]:row):-1;float sum=0,ss=0;for(int k=lane;k<K;k+=32){float v=src<0?0:__half2float(mid[(size_t)src*K+k]);sum+=v;ss+=v*v;}for(int o=16;o;o>>=1){sum+=__shfl_xor_sync(0xffffffff,sum,o);ss+=__shfl_xor_sync(0xffffffff,ss,o);}if(lane==0){cc[WARPS*256+r]=sum/K;cc[WARPS*256+8+r]=rsqrtf(fmaxf(ss/K-(sum/K)*(sum/K),0.f)+1e-6f);}}__syncthreads();}

 // Single independent block factor, natural input/output. FP16 LN output, FP32 WMMA accumulation.
 tiles=(b+15)/16;
 for(int task=warp;task<a*tiles;task+=WARPS){int ib=task/tiles,col=(task%tiles)*16;wmma::fill_fragment(acc,0.f);
  for(int k0=0;k0<b;k0+=16){for(int z=lane;z<256;z+=32){int r=z/16,c=z%16;float v=0;int row=row0+r,src=row<rows?(map?map[row]:row):-1;
    if(r<8&&src>=0&&k0+c<b){v=__half2float(mid[(size_t)src*K+ib*b+k0+c]);if(gamma)v=(v-cc[WARPS*256+r])*cc[WARPS*256+8+r]*gamma[ib*b+k0+c]+beta[ib*b+k0+c];}
    aa[warp*256+z]=__float2half_rn(v);bb[warp*256+z]=(k0+r<b&&col+c<b)?B[((factorMap?factorMap[ib]:ib)*b+k0+r)*b+col+c]:__float2half(0);
   }__syncwarp();wmma::load_matrix_sync(fa,aa+warp*256,16);wmma::load_matrix_sync(fb,bb+warp*256,16);wmma::mma_sync(acc,fa,fb,acc);}
  wmma::store_matrix_sync(cc+warp*256,acc,16,wmma::mem_row_major);__syncwarp();for(int z=lane;z<256;z+=32){int r=z/16,c=col+z%16;if(r<8&&c<b)raw[r*LD+ib*b+c]=__float2half_rn(cc[warp*256+z]);}
 }__syncthreads();
 // Each row warp computes EXACT free-axis/group absmax and signed nibble output.
 int gs=group?group:K,ng=(K+gs-1)/gs,kp=(K+31)/32*32;
 for(int r=warp;r<8;r+=WARPS){int row=row0+r;if(row>=rows)continue;
  for(int gi=0;gi<ng;gi++){float mx=0;for(int k=gi*gs+lane;k<min(K,(gi+1)*gs);k+=32)mx=fmaxf(mx,fabsf(__half2float(raw[r*LD+k])));for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));float scale=mx*(clip/7.f)+1e-12f;if(lane==0)sc[(size_t)row*ng+gi]=scale;
   for(int pair=(gi*gs)/2+lane;pair<min(K,(gi+1)*gs)/2;pair+=32){int k=2*pair,lo=max(-7,min(7,__float2int_rn(__half2float(raw[r*LD+k])/scale))),hi=max(-7,min(7,__float2int_rn(__half2float(raw[r*LD+k+1])/scale)));q[(size_t)row*kp/2+pair]=(lo&15)|((hi&15)<<4);}
  }for(int pair=K/2+lane;pair<kp/2;pair+=32)q[(size_t)row*kp/2+pair]=0;
 }
}
