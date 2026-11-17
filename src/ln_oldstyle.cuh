#pragma once
// Singleblockdiag mathematics. Organization migrated from k_ln_mon4: packed B fragments,
// perwarp partial cache outsidepersistent tokenloop, direct MMA andregister outputstores.
template<int C,int BB,int RC>
__global__ __launch_bounds__(256) void k_ln_bd_oldstyle(const half* __restrict__ x,uint8_t* __restrict__ q,float* __restrict__ sc,
 const float* __restrict__ gamma,const float* __restrict__ beta,const uint2* __restrict__ packed,int rows,float clip,const int* __restrict__ map,const int* __restrict__ factorMap=nullptr){
 constexpr int BP=(BB+15)/16*16,NK=BP/16,NO=(BB+7)/8,NB=C/BB,NU=NB*NO,UPW=(NU+7)/8,XS=NB*BP+8,YS=C+8;
 constexpr int CACHE=RC<UPW?RC:UPW;
 static_assert(C%BB==0,"fullfactor");
 extern __shared__ __align__(32) unsigned char oldstyle_bd_smem[];
 half*X=(half*)oldstyle_bd_smem;half*Y=X+16*XS;
 int lane=threadIdx.x&31,warp=threadIdx.x>>5,gid=lane>>2,tig=lane&3;
 uint2 cache[CACHE?CACHE:1][NK];
 // Coefficients remain alive across ALL persistent16token iterations, not reloadedper tile.
#pragma unroll
 for(int slot=0;slot<CACHE;slot++){int u=warp+8*slot;if(u<NU){
#pragma unroll
  for(int ks=0;ks<NK;ks++)cache[slot][ks]=__ldg(packed+((size_t)((factorMap?factorMap[u/NO]:u/NO)*NO+u%NO)*NK+ks)*32+lane);
 }}
 for(int tile=blockIdx.x;tile*16<rows;tile+=gridDim.x){
  // Current blockdiag LN arithmetic, NOT the oldMon vector reduction order.
  for(int r=warp;r<16;r+=8){int row=tile*16+r,src=row<rows?(map?map[row]:row):-1;float sum=0,ss=0;
   for(int k=lane;k<C;k+=32){float v=src<0?0:__half2float(x[(size_t)src*C+k]);sum+=v;ss+=v*v;}
   for(int o=16;o;o>>=1){sum+=__shfl_xor_sync(0xffffffff,sum,o);ss+=__shfl_xor_sync(0xffffffff,ss,o);}
   float mean=__fdiv_rn(sum,(float)C);float variance=__fsub_rn(__fdiv_rn(ss,(float)C),__fmul_rn(mean,mean));float inv=rsqrtf(fmaxf(variance,0.f)+1e-6f);
   for(int p=lane;p<NB*BP;p+=32){int ib=p/BP,j=p%BP,k=ib*BB+j;float v=0;
    if(src>=0&&j<BB)v=(__half2float(x[(size_t)src*C+k])-mean)*inv*gamma[k]+beta[k];
    X[r*XS+p]=__float2half_rn(v);
   }
  }
  __syncthreads();
  auto unit=[&](int u,int slot){int ib=u/NO,nt=u%NO;float acc[4]={0.f,0.f,0.f,0.f};
#pragma unroll
   for(int ks=0;ks<NK;ks++){
    uint32_t af[4];int lr=(lane&7)+((lane>>3)&1)*8,lhi=lane>>4;
    gemmk::ldsm_x4(af,gemmk::smem_u32(X+lr*XS+ib*BP+ks*16+lhi*8));
    uint2 bf;
    if constexpr(CACHE>0){if(slot<CACHE)bf=cache[slot][ks];else bf=__ldg(packed+((size_t)((factorMap?factorMap[u/NO]:u/NO)*NO+u%NO)*NK+ks)*32+lane);}
    else bf=__ldg(packed+((size_t)((factorMap?factorMap[u/NO]:u/NO)*NO+u%NO)*NK+ks)*32+lane);
    gemmk::mma16816(acc,af,bf.x,bf.y);
   }
   int j=nt*8+tig*2,idx=ib*BB+j;
   if(j<BB){Y[gid*YS+idx]=__float2half_rn(acc[0]);Y[(gid+8)*YS+idx]=__float2half_rn(acc[2]);}
   if(j+1<BB){Y[gid*YS+idx+1]=__float2half_rn(acc[1]);Y[(gid+8)*YS+idx+1]=__float2half_rn(acc[3]);}
  };
#pragma unroll
  for(int slot=0;slot<CACHE;slot++)if(warp+8*slot<NU)unit(warp+8*slot,slot);
  for(int slot=CACHE,u=warp+8*CACHE;u<NU;slot++,u+=8)unit(u,slot);
  __syncthreads();
  constexpr int KP=(C+31)/32*32;
  for(int r=warp;r<16;r+=8){int row=tile*16+r;if(row>=rows)continue;float mx=0;
   for(int k=lane;k<C;k+=32)mx=fmaxf(mx,fabsf(__half2float(Y[r*YS+k])));
   for(int o=16;o;o>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,o));float s=mx*(clip/7.f)+1e-12f;if(lane==0)sc[row]=s;
   for(int pair=lane;pair<KP/2;pair+=32){int k=pair*2,lo=0,hi=0;if(k<C)lo=max(-7,min(7,__float2int_rn(__half2float(Y[r*YS+k])/s)));if(k+1<C)hi=max(-7,min(7,__float2int_rn(__half2float(Y[r*YS+k+1])/s)));q[(size_t)row*KP/2+pair]=(lo&15)|((hi&15)<<4);}
  }
  __syncthreads(); // no iteration writes X/Y while anotherwarp stillpacks previous tile.
 }
}
// Hostpack the already-roundedFP16factor, preserving rowxT orientation. Bregisteruint2/lane.
static std::vector<uint2> bd_oldstyle_pack(int C,int b,const std::vector<half>&f){
 int a=C/b,nk=(b+15)/16,no=(b+7)/8;std::vector<uint2>p((size_t)a*no*nk*32);
 auto pair=[&](int ib,int k,int col){uint16_t lo=0,hi=0;if(k<b&&col<b)memcpy(&lo,&f[(ib*b+k)*b+col],2);if(k+1<b&&col<b)memcpy(&hi,&f[(ib*b+k+1)*b+col],2);return (uint32_t)lo|((uint32_t)hi<<16);};
 for(int ib=0;ib<a;ib++)for(int nt=0;nt<no;nt++)for(int ks=0;ks<nk;ks++)for(int lane=0;lane<32;lane++){int col=nt*8+(lane>>2),k=ks*16+(lane&3)*2;p[(((size_t)ib*no+nt)*nk+ks)*32+lane]=make_uint2(pair(ib,k,col),pair(ib,k+8,col));}
 // Independently decodeeverypackedword, allvalid/tail entries, exacthalfbitcomparison.
 for(int ib=0;ib<a;ib++)for(int nt=0;nt<no;nt++)for(int ks=0;ks<nk;ks++)for(int lane=0;lane<32;lane++){
  int col=nt*8+(lane>>2),k=ks*16+(lane&3)*2;uint2 v=p[(((size_t)ib*no+nt)*nk+ks)*32+lane];uint32_t words[2]={v.x,v.y};
  for(int w=0;w<2;w++)for(int h=0;h<2;h++){int kk=k+w*8+h;uint16_t expected=0;if(kk<b&&col<b)memcpy(&expected,&f[(ib*b+kk)*b+col],2);if(((words[w]>>(16*h))&65535)!=expected){printf("BDFRAGMENT_PACK_FAIL\n");exit(2);}}
 }
 return p;
}
