#include <cuda_runtime.h>
#include <math.h>
__global__ void qkt_kernel(const float*Q,const float*K,float*S,int N,int d_model,int h){
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int head=blockIdx.z;
    if(row>=N||col>=N)return;
    int dk=d_model/h;
    float val=0.0f;
    for(int i=0;i<dk;i++)val+=Q[(size_t)row*d_model+head*dk+i]*K[(size_t)col*d_model+head*dk+i];
    S[(size_t)head*N*N+(size_t)row*N+col]=val/sqrtf((float)dk);
}
__global__ void softmax_kernel(float*S,int N,int h){
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    int head=blockIdx.y;
    if(row>=N)return;
    float*r=S+(size_t)head*N*N+(size_t)row*N;
    float mx=-INFINITY;
    for(int c=0;c<N;c++)mx=fmaxf(mx,r[c]);
    float sum=0.0f;
    for(int c=0;c<N;c++){r[c]=expf(r[c]-mx);sum+=r[c];}
    for(int c=0;c<N;c++)r[c]/=sum;
}
__global__ void sv_kernel(const float*S,const float*V,float*O,int N,int d_model,int h){
    int dk=d_model/h;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int head=blockIdx.z;
    if(row>=N||col>=dk)return;
    const float*s=S+(size_t)head*N*N+(size_t)row*N;
    float val=0.0f;
    for(int i=0;i<N;i++)val+=s[i]*V[(size_t)i*d_model+head*dk+col];
    O[(size_t)row*d_model+head*dk+col]=val;
}
extern "C" void solve(const float* Q,const float* K,const float* V,float* output,int N,int d_model,int h){
    int dk=d_model/h;
    float*S;
    cudaMalloc(&S,(size_t)h*N*N*sizeof(float));
    dim3 b1(16,16),g1((N+15)/16,(N+15)/16,h);
    qkt_kernel<<<g1,b1>>>(Q,K,S,N,d_model,h);
    softmax_kernel<<<dim3((N+255)/256,h),256>>>(S,N,h);
    dim3 b3(16,16),g3((dk+15)/16,(N+15)/16,h);
    sv_kernel<<<g3,b3>>>(S,V,output,N,d_model,h);
    cudaDeviceSynchronize();
    cudaFree(S);
}