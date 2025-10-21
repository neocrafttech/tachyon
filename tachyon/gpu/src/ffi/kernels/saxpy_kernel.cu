
__device__ float helper_mul(float a, float b) {
    printf("Manish here\n");
    return a * b;
}

__global__ void saxpy_kernel(float a, const float *x, float *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = helper_mul(a, x[idx]) + y[idx];
}
