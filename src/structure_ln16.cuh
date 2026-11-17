// New matched structures ONLY: no Hadamard, no GPTQ. Both all3 learnedclip/G128.
#pragma once
struct StructureGPU {int K,a,b,type;float clip;float *first,*second;uint2*packed=nullptr;int*factorMap=nullptr;int factorCount=0;};
__global__ void k_structure_producer(const half*x,uint8_t*q,float*scale,const float*gamma,const float*beta,
 const float*first,const float*second,int K,int a,int b,int type,float clip,int group,const int*map,float*transformed){
 extern __shared__ float sm[];float*raw=sm;float*tmp=sm+K;float*red=sm+2*K;int row=blockIdx.x,tid=threadIdx.x,src=map?map[row]:row;
 for(int k=tid;k<K;k+=256)raw[k]=src<0?0:__half2float(x[(size_t)src*K+k]);__syncthreads();
 if(gamma){float sum=0,ss=0;for(int k=tid;k<K;k+=256){sum+=raw[k];ss+=raw[k]*raw[k];}for(int o=16;o;o>>=1){sum+=__shfl_down_sync(0xffffffff,sum,o);ss+=__shfl_down_sync(0xffffffff,ss,o);}if((tid&31)==0){red[tid/32]=sum;red[8+tid/32]=ss;}__syncthreads();if(tid<32){sum=tid<8?red[tid]:0;ss=tid<8?red[8+tid]:0;for(int o=16;o;o>>=1){sum+=__shfl_down_sync(0xffffffff,sum,o);ss+=__shfl_down_sync(0xffffffff,ss,o);}if(tid==0){red[16]=sum/K;red[17]=rsqrtf(fmaxf(ss/K-(sum/K)*(sum/K),0.f)+1e-6f);}}__syncthreads();for(int k=tid;k<K;k+=256)raw[k]=src<0?0:(raw[k]-red[16])*red[17]*gamma[k]+beta[k];__syncthreads();}
 // Kron L^T X R; Mon row-block A -> pi -> row-block B -> invpi.
 // For Mon final natural index (i,j), pi-layout index j*a+i, so inverse permutation is folded into store.
 for(int k=tid;k<K;k+=256){int i=k/b,j=k%b;float v=0;
  if(type==0){for(int t=0;t<a;t++)v=fmaf(first[t*a+i],raw[t*b+j],v);}
  else{for(int t=0;t<b;t++)v=fmaf(raw[i*b+t],first[(i*b+t)*b+j],v);}
  tmp[k]=v;
 }__syncthreads();
 for(int k=tid;k<K;k+=256){int i=k/b,j=k%b;float v=0;
  if(type==0){for(int t=0;t<b;t++)v=fmaf(tmp[i*b+t],second[t*b+j],v);}
  else{for(int t=0;t<a;t++)v=fmaf(tmp[t*b+j],second[(j*a+t)*a+i],v);}
  raw[k]=v;if(transformed)transformed[(size_t)row*K+k]=v;
 }__syncthreads();
 int gs=group?group:K,ng=(K+gs-1)/gs;for(int g=tid;g<ng;g+=256){float mx=0;for(int k=g*gs;k<min(K,(g+1)*gs);k++)mx=fmaxf(mx,fabsf(raw[k]));scale[(size_t)row*ng+g]=mx*(clip/7)+1e-12f;}__syncthreads();
 int kp=(K+31)/32*32;for(int p=tid;p<kp/2;p+=256){int k=2*p,lo=0,hi=0;if(k<K)lo=max(-7,min(7,__float2int_rn(raw[k]/scale[(size_t)row*ng+k/gs])));if(k+1<K)hi=max(-7,min(7,__float2int_rn(raw[k+1]/scale[(size_t)row*ng+(k+1)/gs])));q[(size_t)row*kp/2+p]=(lo&15)|((hi&15)<<4);}
}
// Orientation/quant formula frozen against shared.py. Dedicated tensorcore optimization pending new exports.

static std::string gKronFile;
static std::vector<StructureGPU> gKron1,gKron2,gKron3;
#ifndef TC_LN_EPS
#define TC_LN_EPS 1e-6f
#endif
#include "structure_wmma_g4q.cuh"
#include "mon_fused_pad.cuh"
#include "mon_second_fused_pad.cuh"
static half *tcRaw=nullptr,*tcMid=nullptr,*tcOut=nullptr;
static size_t tcCapacity=0;
#include "blockdiag_fused.cuh"
#include "blockdiag_group128.cuh"
#include "blockdiag_ln16.cuh"
#include "ln_oldstyle.cuh"

static int oldstyleMode(){const char*e=getenv("NANO_LN_OLDSTYLE");return e?atoi(e):0;}
template<int C,int BB,int RC>static void oldstyleRun(const half*x,int8_t*q,float*sc,const float*g,const float*be,int rows,const StructureGPU&m,cudaStream_t stream,const int*map){
 constexpr int BP=(BB+15)/16*16;constexpr size_t sh=16*((C/BB)*BP+8+C+8)*2;auto kern=k_ln_bd_oldstyle<C,BB,RC>;
 static int occ=0,sms=0;static bool init=false;
 if(!init){CK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh));CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ,kern,256,sh));cudaDeviceProp dp;CK(cudaGetDeviceProperties(&dp,0));sms=dp.multiProcessorCount;cudaFuncAttributes at;CK(cudaFuncGetAttributes(&at,kern));printf("OLDSTYLE_RESOURCE C%d b%d RC%d reg%d local%zu static%zu dynamic%zu occupancyCTAs%d SMs%d\n",C,BB,RC,at.numRegs,at.localSizeBytes,at.sharedSizeBytes,sh,occ,sms);init=true;}
 int tiles=(rows+15)/16,waves=1;if(const char*e=getenv("NANO_LN_OLDSTYLE_WAVES"))waves=atoi(e);if(waves<1||waves>4){printf("BAD_WAVES\n");exit(2);}int grid=oldstyleMode()==3?tiles:std::min(tiles,sms*std::max(1,occ)*waves);
 kern<<<grid,256,sh,stream>>>(x,(uint8_t*)q,sc,g,be,m.packed,rows,m.clip,map,m.factorMap);CK(cudaGetLastError());
}
static void oldstyleLaunch(const half*x,int8_t*q,float*sc,const float*g,const float*be,int rows,const StructureGPU&m,cudaStream_t stream,const int*map){
#define OLS(C,B,R) if(oldstyleMode()==2)oldstyleRun<C,B,0>(x,q,sc,g,be,rows,m,stream,map);else oldstyleRun<C,B,R>(x,q,sc,g,be,rows,m,stream,map)
 if(m.K==144&&m.b==12){OLS(144,12,3);}else if(m.K==288&&m.b==18){OLS(288,18,3);}else if(m.K==576&&m.b==24){OLS(576,24,1);}else if(m.K==1152&&m.b==36){OLS(1152,36,0);}else if(m.K==1152&&m.b==32){OLS(1152,32,1);}else{printf("BAD_OLDSTYLE_SHAPE K%d b%d\n",m.K,m.b);exit(2);}
#undef OLS
}
static void launchKron(const half*x,int8_t*q,float*sc,const float*g,const float*be,int rows,const StructureGPU&m,cudaStream_t stream,const int*map=nullptr,float*transformed=nullptr,int group=0){
 if(m.type!=2){printf("blockdiag type2 required%c",10);exit(1);}
 if(!g&&group==128){k_blockdiag_group128<<<dim3((m.K+127)/128,(rows+15)/16),128,0,stream>>>(x,(uint8_t*)q,sc,(const half*)m.first,m.K,m.a,m.b,rows,m.clip,m.factorMap);CK(cudaGetLastError());return;}
 if(g&&oldstyleMode()){oldstyleLaunch(x,q,sc,g,be,rows,m,stream,map);return;}
 if(g&&m.K<=1152){size_t sh16=(size_t)32*(m.K+8)+16512;static bool attr16=false;if(!attr16){CK(cudaFuncSetAttribute(k_blockdiag_ln16<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,53632));attr16=true;}k_blockdiag_ln16<8><<<(rows+15)/16,256,sh16,stream>>>(x,(uint8_t*)q,sc,g,be,(const half*)m.first,m.K,m.a,m.b,rows,m.clip,group,map,m.factorMap);CK(cudaGetLastError());return;}
 size_t sh=(size_t)16*(m.K+8)+16448;
 static bool attr=false;if(!attr){CK(cudaFuncSetAttribute(k_blockdiag_fused<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,90304));attr=true;}
 k_blockdiag_fused<8><<<(rows+7)/8,256,sh,stream>>>(x,(uint8_t*)q,sc,g,be,(const half*)m.first,m.K,m.a,m.b,rows,m.clip,group,map,m.factorMap);CK(cudaGetLastError());
}
#ifndef STRUCTURE_STANDALONE
static void buildKron(){
 std::ifstream f(gKronFile,std::ios::binary);char magic[8];int nb;f.read(magic,8);f.read((char*)&nb,4);bool compact=!memcmp(magic,"NANOSTR2",8);if(!f||(!compact&&memcmp(magic,"NANOSTR1",8))||nb!=cfg.nblk){printf("bad structure header%c",10);exit(1);}
 gKron1.resize(nb);gKron2.resize(nb);gKron3.resize(nb);
 for(int i=0;i<nb;i++)for(int j=0;j<3;j++){StructureGPU m;int h[4],sz[2];f.read((char*)h,16);m.K=h[0];m.a=h[1];m.b=h[2];m.type=h[3];f.read((char*)&m.clip,4);f.read((char*)sz,8);int expected=j==0?cfg.b[i].din:j==1?cfg.b[i].dout:4*cfg.b[i].dout;
 int nf=m.a,mc=0;std::vector<int>mapping;if(compact){f.read((char*)&nf,4);f.read((char*)&mc,4);if(nf!=(m.a+3)/4||mc!=m.a){printf("badcompacteach4shape\n");exit(2);}mapping.resize(mc);f.read((char*)mapping.data(),mc*4);for(int z=0;z<mc;z++)if(mapping[z]!=z/4){printf("badcompacteach4map\n");exit(2);}}m.factorCount=nf;
 if(m.K!=expected||m.a*m.b!=m.K||m.type!=2||!(m.clip>0&&m.clip<=1)||sz[0]!=nf*m.b*m.b||sz[1]!=0){printf("bad matched structure shape%c",10);exit(1);}
 std::vector<float>v(sz[0]),w(sz[1]);f.read((char*)v.data(),v.size()*4);f.read((char*)w.data(),w.size()*4);if(!f){printf("truncated structure file%c",10);exit(1);}std::vector<half>vh(v.size()),wh(w.size());for(size_t z=0;z<v.size();z++)vh[z]=__float2half(v[z]);for(size_t z=0;z<w.size();z++)wh[z]=__float2half(w[z]);CK(cudaMalloc(&m.first,vh.size()*2));if(!wh.empty())CK(cudaMalloc(&m.second,wh.size()*2));CK(cudaMemcpy(m.first,vh.data(),vh.size()*2,cudaMemcpyHostToDevice));if(!wh.empty())CK(cudaMemcpy(m.second,wh.data(),wh.size()*2,cudaMemcpyHostToDevice));if(j<2){auto packed=bd_oldstyle_pack(nf*m.b,m.b,vh);CK(cudaMalloc(&m.packed,packed.size()*sizeof(uint2)));CK(cudaMemcpy(m.packed,packed.data(),packed.size()*sizeof(uint2),cudaMemcpyHostToDevice));printf("BD_PACK_EXACT block%d family%d K%d b%d bytes%zu\n",i,j,m.K,m.b,packed.size()*sizeof(uint2));}if(compact){CK(cudaMalloc(&m.factorMap,mapping.size()*4));CK(cudaMemcpy(m.factorMap,mapping.data(),mapping.size()*4,cudaMemcpyHostToDevice));}(j==0?gKron1:j==1?gKron2:gKron3)[i]=m;}
 for(const auto&bb:cfg.b)tcCapacity=std::max(tcCapacity,(size_t)std::max(bb.Tprb*bb.din,bb.Tqb*4*bb.dout));CK(cudaMalloc(&tcRaw,tcCapacity*2));CK(cudaMalloc(&tcMid,tcCapacity*2));CK(cudaMalloc(&tcOut,tcCapacity*2));
 printf("MATCHED all3structures noHad noGPTQ learnedclip/G128 FP32factors/FMA%c",10);
}
#endif
