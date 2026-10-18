#include <cuda_runtime.h>
#include <math.h>

__global__ void fused_layernorm_linear_gelu_kernel(const float* x,const float* gamma,const float* beta,const float* W,const float* bias,float* y,int M,int K,int N,float eps){
    int tid=threadIdx.x;
    int row=blockIdx.x;
    float temp=0.0f;
    extern __shared__ float s[];

    // Find mean
    for(int i=tid;i<K;i+=blockDim.x){
        temp+=x[row*K+i];
    }
    s[tid]=temp;
    __syncthreads();

    temp=0.0f;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i+=32){
            temp+=s[i];
        }
        float val=temp;
        val+=__shfl_down_sync(0xffffffff,val,16);
        val+=__shfl_down_sync(0xffffffff,val,8);
        val+=__shfl_down_sync(0xffffffff,val,4);
        val+=__shfl_down_sync(0xffffffff,val,2);
        val+=__shfl_down_sync(0xffffffff,val,1);
        if(tid==0){
            s[0]=val/K;
        }
    }
    __syncthreads();

    float mean=s[0];

    // Find variance
    temp=0.0f;
    for(int i=tid;i<K;i+=blockDim.x){
        float diff=x[row*K+i]-mean;
        temp+=diff*diff;
    }
    s[tid]=temp;
    __syncthreads();

    temp=0.0f;
    if(tid<32){
        for(int i=tid;i<blockDim.x;i+=32){
            temp+=s[i];
        }
        float val=temp;
        val+=__shfl_down_sync(0xffffffff,val,16);
        val+=__shfl_down_sync(0xffffffff,val,8);
        val+=__shfl_down_sync(0xffffffff,val,4);
        val+=__shfl_down_sync(0xffffffff,val,2);
        val+=__shfl_down_sync(0xffffffff,val,1);
        if(tid==0){
            s[0]=val/K;
        }
    }
    __syncthreads();

    float variance=s[0];
    float inv_std=rsqrtf(variance+eps);

    // LayerNorm affine + Linear + bias + exact GELU
    for(int j=tid;j<N;j+=blockDim.x){
        float sum=0.0f;
        for(int k=0;k<K;k++){
            float normalized=(x[row*K+k]-mean)*inv_std;
            float ln=normalized*gamma[k]+beta[k];
            sum+=ln*W[k*N+j];
        }
        sum+=bias[j];
        y[row*N+j]=0.5f*sum*(1.0f+erff(sum/sqrtf(2.0f)));
    }
}

extern "C" void solve(const float* x,const float* gamma,const float* beta,const float* W,const float* bias,float* y,int M,int K,int N,float eps){
    int threads=256;
    size_t shared_bytes=threads*sizeof(float);
    fused_layernorm_linear_gelu_kernel<<<M,threads,shared_bytes>>>(x,gamma,beta,W,bias,y,M,K,N,eps);
}
