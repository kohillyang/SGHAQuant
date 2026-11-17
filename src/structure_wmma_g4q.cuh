// Common prospective precision: FP16 operands/factors/interstage, FP32 WMMA accumulator.
// Stages keep explicit layout/permutation costs. No quantizer range/group changes.
#pragma once
#include <mma.h>
using namespace nvcuda;
// Each block handles 16 tokens and one 16-output tile of one factor block.
// Stage0: Kron computes x[:,j] @ L (j groups), Mon computes x[i,:] @ A_i.
// Stage1: Kron intermediate[i,:] @ R, Mon intermediate[:,j] @ B_j.
// Every output is stored natural[i*b+j]; implicit gather/store accounts pi+invpi.
__global__ void k_structure_wmma(const half*input,half*output,const half*factor,int rows,int a,int b,int type,int stage){
 __shared__ __align__(32) half ast[4][16*16],bst[4][16*16];
 __shared__ __align__(32) float cst[4][16*16];
 int warp=threadIdx.x/32,lane=threadIdx.x%32;int group=blockIdx.z*4+warp,row0=blockIdx.y*16,col0=blockIdx.x*16;half*as=ast[warp];half*bs=bst[warp];float*cs=cst[warp];int nGroups=(type==0?(stage==0?b:a):(stage==0?a:b));if(group>=nGroups)return;
 int dim=(type==0?(stage==0?a:b):(stage==0?b:a));int K=a*b;
 wmma::fragment<wmma::matrix_a,16,16,16,half,wmma::row_major> af;
 wmma::fragment<wmma::matrix_b,16,16,16,half,wmma::row_major> bf;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc;wmma::fill_fragment(acc,0.f);
 for(int k0=0;k0<dim;k0+=16){
  for(int n=lane;n<256;n+=32){int rr=n/16,cc=n%16;int row=row0+rr,k=k0+cc;
   int index;
   if((type==0&&stage==0)||(type==1&&stage==1))index=k*b+group;
   else index=group*b+k;
   as[n]=(row<rows&&k<dim)?input[(size_t)row*K+index]:__float2half(0.f);
   int fk=k0+rr,fc=col0+cc;int off=type==0?fk*dim+fc:group*dim*dim+fk*dim+fc;
   bs[n]=(fk<dim&&fc<dim)?factor[off]:__float2half(0.f);
  }__syncwarp();wmma::load_matrix_sync(af,as,16);wmma::load_matrix_sync(bf,bs,16);wmma::mma_sync(acc,af,bf,acc);__syncwarp();
 }
 wmma::store_matrix_sync(cs,acc,16,wmma::mem_row_major);__syncwarp();
 for(int n=lane;n<256;n+=32){int row=row0+n/16,c=col0+n%16;int index;
  if((type==0&&stage==0)||(type==1&&stage==1))index=c*b+group;
  else index=group*b+c;
  if(row<rows&&c<dim)output[(size_t)row*K+index]=__float2half_rn(cs[n]);
 }
}
__global__ void k_structure_ln_half(const half*x,half*y,const float*g,const float*be,int K,int rows,const int*map){
 __shared__ float red[18];int row=blockIdx.x,tid=threadIdx.x,src=map?map[row]:row;float sum=0,ss=0;
 for(int k=tid;k<K;k+=256){float v=src<0?0.f:__half2float(x[(size_t)src*K+k]);sum+=v;ss+=v*v;}
 for(int o=16;o;o>>=1){sum+=__shfl_down_sync(0xffffffff,sum,o);ss+=__shfl_down_sync(0xffffffff,ss,o);}if((tid&31)==0){red[tid/32]=sum;red[8+tid/32]=ss;}__syncthreads();
 if(tid<32){sum=tid<8?red[tid]:0;ss=tid<8?red[8+tid]:0;for(int o=16;o;o>>=1){sum+=__shfl_down_sync(0xffffffff,sum,o);ss+=__shfl_down_sync(0xffffffff,ss,o);}if(tid==0){red[16]=sum/K;red[17]=rsqrtf(fmaxf(ss/K-(sum/K)*(sum/K),0.f)+TC_LN_EPS);}}__syncthreads();
 for(int k=tid;k<K;k+=256){float v=src<0?0.f:(__half2float(x[(size_t)src*K+k])-red[16])*red[17]*g[k]+be[k];y[(size_t)row*K+k]=__float2half_rn(v);}
}
__global__ void k_structure_pack_half(const half*x,uint8_t*q,float*sc,int K,int rows,float clip,int group){
 __shared__ float scales[36];int row=blockIdx.x,tid=threadIdx.x,gs=group?group:K,ng=(K+gs-1)/gs;
 if(!group){float mx=0;for(int k=tid;k<K;k+=256)mx=fmaxf(mx,fabsf(__half2float(x[(size_t)row*K+k])));for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if((tid&31)==0)scales[tid/32]=mx;__syncthreads();if(tid<32){mx=tid<8?scales[tid]:0;for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if(tid==0){scales[0]=mx*(clip/7.f)+1e-12f;sc[row]=scales[0];}}__syncthreads();}
 else{int warp=tid/32,lane=tid%32;for(int gi=warp;gi<ng;gi+=8){float mx=0;for(int k=gi*128+lane;k<min(K,(gi+1)*128);k+=32)mx=fmaxf(mx,fabsf(__half2float(x[(size_t)row*K+k])));for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_down_sync(0xffffffff,mx,o));if(lane==0){scales[gi]=mx*(clip/7.f)+1e-12f;sc[(size_t)row*ng+gi]=scales[gi];}}__syncthreads();}
 int kp=(K+31)/32*32;for(int p=tid;p<kp/2;p+=256){int k=2*p,lo=0,hi=0;if(k<K)lo=max(-7,min(7,__float2int_rn(__half2float(x[(size_t)row*K+k])/scales[k/gs])));if(k+1<K)hi=max(-7,min(7,__float2int_rn(__half2float(x[(size_t)row*K+k+1])/scales[(k+1)/gs])));q[(size_t)row*kp/2+p]=(lo&15)|((hi&15)<<4);}
}
