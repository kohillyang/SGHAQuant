// Hiera-only production-style Monarch fused LN/raw -> A Pi B Pi^-1 -> clip/G128/pack.
// SAME prospective TC precision: FP16 LN/factors/interstage/final, FP32 MMA acc.
// Initial fusion covers K<=1152; largefc2 must use separately counted fallback, never silently omitted.
#pragma once
#include <mma.h>
using namespace nvcuda;
template<int WARPS=8>
__global__ void k_mon_fused(const half*x,uint8_t*q,float*sc,const float*gamma,const float*beta,
 const half*A,const half*B,int K,int a,int b,int rows,float clip,int group,const int*map){
 extern __shared__ __align__(32) unsigned char storage[];
 int LD=K+8;half*raw=(half*)storage;half*mid=raw+16*LD;
 half*aa=mid+16*LD;half*bb=aa+WARPS*256;float*cc=(float*)(bb+WARPS*256);
 int tid=threadIdx.x,warp=tid/32,lane=tid%32,row0=blockIdx.x*16;
 // Row-owned warp LN, shared row storage natural layout.
 for(int r=warp;r<16;r+=WARPS){int row=row0+r,src=row<rows?(map?map[row]:row):-1;float sum=0,ss=0;
  for(int k=lane;k<K;k+=32){float v=src<0?0:__half2float(x[(size_t)src*K+k]);sum+=v;ss+=v*v;}
  for(int o=16;o;o>>=1){sum+=__shfl_xor_sync(0xffffffff,sum,o);ss+=__shfl_xor_sync(0xffffffff,ss,o);}float mean=sum/K,inv=rsqrtf(fmaxf(ss/K-mean*mean,0.f)+1e-6f);
  for(int k=lane;k<K;k+=32){float v=src<0?0:__half2float(x[(size_t)src*K+k]);if(gamma)v=src<0?0:(v-mean)*inv*gamma[k]+beta[k];raw[r*LD+k]=__float2half_rn(v);}
 }__syncthreads();
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> fa;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> fb;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;
 // First factor: each independent A_i shared by all16tokens; coalesced natural access.
 int tiles=(b+15)/16;
 for(int task=warp;task<a*tiles;task+=WARPS){int ib=task/tiles,col=(task%tiles)*16;wmma::fill_fragment(acc,0.f);
  for(int k0=0;k0<b;k0+=16){for(int z=lane;z<256;z+=32){int r=z/16,c=z%16;aa[warp*256+z]=k0+c<b?raw[r*LD+ib*b+k0+c]:__float2half(0);bb[warp*256+z]=(k0+r<b&&col+c<b)?A[(ib*b+k0+r)*b+col+c]:__float2half(0);}__syncwarp();wmma::load_matrix_sync(fa,aa+warp*256,16);wmma::load_matrix_sync(fb,bb+warp*256,16);wmma::mma_sync(acc,fa,fb,acc);}
  wmma::store_matrix_sync(cc+warp*256,acc,16,wmma::mem_row_major);__syncwarp();for(int z=lane;z<256;z+=32){int r=z/16,c=col+z%16;if(c<b)mid[r*LD+ib*b+c]=__float2half_rn(cc[warp*256+z]);}
 }__syncthreads();
 // Second factor: stridepi gather in shared, output invpi natural store. All traffic accounted.
 tiles=(a+15)/16;
 for(int task=warp;task<b*tiles;task+=WARPS){int jb=task/tiles,col=(task%tiles)*16;wmma::fill_fragment(acc,0.f);
  for(int k0=0;k0<a;k0+=16){for(int z=lane;z<256;z+=32){int r=z/16,c=z%16;aa[warp*256+z]=k0+c<a?mid[r*LD+(k0+c)*b+jb]:__float2half(0);bb[warp*256+z]=(k0+r<a&&col+c<a)?B[(jb*a+k0+r)*a+col+c]:__float2half(0);}__syncwarp();wmma::load_matrix_sync(fa,aa+warp*256,16);wmma::load_matrix_sync(fb,bb+warp*256,16);wmma::mma_sync(acc,fa,fb,acc);}
  wmma::store_matrix_sync(cc+warp*256,acc,16,wmma::mem_row_major);__syncwarp();for(int z=lane;z<256;z+=32){int r=z/16,c=col+z%16;if(c<a)raw[r*LD+c*b+jb]=__float2half_rn(cc[warp*256+z]);}
 }__syncthreads();
 // Each row warp computes EXACT free-axis/group absmax and signed nibble output.
 int gs=group?group:K,ng=(K+gs-1)/gs,kp=(K+31)/32*32;
 for(int r=warp;r<16;r+=WARPS){int row=row0+r;if(row>=rows)continue;
  for(int gi=0;gi<ng;gi++){float mx=0;for(int k=gi*gs+lane;k<min(K,(gi+1)*gs);k+=32)mx=fmaxf(mx,fabsf(__half2float(raw[r*LD+k])));for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));float scale=mx*(clip/7.f)+1e-12f;if(lane==0)sc[(size_t)row*ng+gi]=scale;
   for(int pair=(gi*gs)/2+lane;pair<min(K,(gi+1)*gs)/2;pair+=32){int k=2*pair,lo=max(-7,min(7,__float2int_rn(__half2float(raw[r*LD+k])/scale))),hi=max(-7,min(7,__float2int_rn(__half2float(raw[r*LD+k+1])/scale)));q[(size_t)row*kp/2+pair]=(lo&15)|((hi&15)<<4);}
  }for(int pair=K/2+lane;pair<kp/2;pair+=32)q[(size_t)row*kp/2+pair]=0;
 }
}
