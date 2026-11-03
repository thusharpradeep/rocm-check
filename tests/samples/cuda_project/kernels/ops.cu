#include <cuda_runtime.h>

__global__ void add(float* a, float* b, float* c) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void launch(float* a, float* b, float* c, int n) {
    float* d_a;
    cudaMalloc(&d_a, n * sizeof(float));
    add<<<1, 256>>>(a, b, c);
    cudaDeviceSynchronize();
}
