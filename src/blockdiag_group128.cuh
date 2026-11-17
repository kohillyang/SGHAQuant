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

// Gamma-null FC2: one global128-outputgroup perCTA, blockfactorhalo input; nofullKsmem.
// Same16tokenWMMA precision; outputgroupmax after natural128reassembly includingremainder.
#pragma once
#include <mma.h>
using namespace nvcuda;
__global__ void k_blockdiag_group128(const half*x,uint8_t*q,float*sc,const half*F,int K,int a,int b,int rows,float clip,const int*factorMap=nullptr){
 __shared__ __align__(32) half aa[4][256],bb[4][256],out[16][136];
 __shared__ __align__(32) float cc[4][256];
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,row0=blockIdx.y*16,start=blockIdx.x*128,end=min(K,start+128),ng=(K+127)/128;
 for(int z=tid;z<16*136;z+=128)((half*)out)[z]=__float2half(0);__syncthreads();
 int block0=start/b,block1=(end+b-1)/b,nt=(b+15)/16;
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major>A;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major>B;
 wmma::fragment<wmma::accumulator,16,16,16,float>C;
 for(int task=warp;task<(block1-block0)*nt;task+=4){int ib=block0+task/nt,col=task%nt*16;
  if(ib*b+col>=end||ib*b+col+16<=start)continue;wmma::fill_fragment(C,0);
  for(int k0=0;k0<b;k0+=16){for(int z=lane;z<256;z+=32){int r=z/16,c=z%16;aa[warp][z]=(row0+r<rows&&k0+c<b)?x[(size_t)(row0+r)*K+ib*b+k0+c]:__float2half(0);bb[warp][z]=(k0+r<b&&col+c<b)?F[((factorMap?factorMap[ib]:ib)*b+k0+r)*b+col+c]:__float2half(0);}__syncwarp();wmma::load_matrix_sync(A,aa[warp],16);wmma::load_matrix_sync(B,bb[warp],16);wmma::mma_sync(C,A,B,C);}
  wmma::store_matrix_sync(cc[warp],C,16,wmma::mem_row_major);__syncwarp();for(int z=lane;z<256;z+=32){int r=z/16,c=ib*b+col+z%16;if(c>=start&&c<end&&c<ib*b+b)out[r][c-start]=__float2half_rn(cc[warp][z]);}
 }__syncthreads();
 for(int r=warp;r<16;r+=4){if(row0+r>=rows)continue;float mx=0;for(int c=lane;c<end-start;c+=32)mx=fmaxf(mx,fabsf(__half2float(out[r][c])));for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));float s=mx*(clip/7.f)+1e-12f;if(lane==0)sc[(size_t)(row0+r)*ng+blockIdx.x]=s;
  for(int p=lane;p<(end-start)/2;p+=32){int lo=max(-7,min(7,__float2int_rn(__half2float(out[r][2*p])/s))),hi=max(-7,min(7,__float2int_rn(__half2float(out[r][2*p+1])/s)));q[(size_t)(row0+r)*((K+31)/32*16)+start/2+p]=(lo&15)|((hi&15)<<4);}
 }
}
