#include <cuda_runtime.h>
#include <math.h>
#define THREADS 256
#define TILE 64
__global__ void softmax_attention_kernel(const float* __restrict__ Q,const float* __restrict__ K,const float* __restrict__ V,float* __restrict__ output,int M,int N,int d){
    extern __shared__ float smem[];
    float*q=smem,*acc=q+d,*s=acc+d;
    int row=blockIdx.x,tid=threadIdx.x,lane=tid&31,warp=tid>>5,nw=blockDim.x>>5;
    float scale=rsqrtf((float)d);
    for(int k=tid;k<d;k+=blockDim.x){q[k]=Q[(size_t)row*d+k]*scale;acc[k]=0.0f;}
    __syncthreads();
    float m=-INFINITY,l=0.0f;
    for(int j0=0;j0<N;j0+=TILE){
        int tn=min(TILE,N-j0);
        for(int jj=warp;jj<tn;jj+=nw){
            const float*kr=K+(size_t)(j0+jj)*d;
            float dot=0.0f;
            for(int k=lane;k<d;k+=32)dot+=q[k]*kr[k];
            for(int o=16;o>0;o>>=1)dot+=__shfl_down_sync(0xffffffff,dot,o);
            if(lane==0)s[jj]=dot;
        }
        __syncthreads();
        float tm=-INFINITY;
        for(int jj=0;jj<tn;++jj)tm=fmaxf(tm,s[jj]);
        float mn=fmaxf(m,tm),a=expf(m-mn);
        __syncthreads();
        if(tid<tn)s[tid]=expf(s[tid]-mn);
        __syncthreads();
        float ts=0.0f;
        for(int jj=0;jj<tn;++jj)ts+=s[jj];
        l=l*a+ts;
        for(int k=tid;k<d;k+=blockDim.x){
            float x=acc[k]*a;
            for(int jj=0;jj<tn;++jj)x+=s[jj]*V[(size_t)(j0+jj)*d+k];
            acc[k]=x;
        }
        m=mn;
        __syncthreads();
    }
    float il=1.0f/l;
    for(int k=tid;k<d;k+=blockDim.x)output[(size_t)row*d+k]=acc[k]*il;
}
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int M,int N,int d){
    if(M<=0||N<=0||d<=0)return;
    size_t sh=(2*(size_t)d+TILE)*sizeof(float);
    if(sh>48*1024)cudaFuncSetAttribute(softmax_attention_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)sh);
    softmax_attention_kernel<<<M,THREADS,sh>>>(Q,K,V,output,M,N,d);
    cudaDeviceSynchronize();
}