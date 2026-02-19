#include <cuda_runtime.h>
#include <cstdio>

extern "C" cudaError_t launch_saxpy(float a, float *x, float *y, int n);

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err__));                                   \
            return 1;                                                             \
        }                                                                         \
    } while (0)

int main() {
    const int N = 4;

    float hx[N] = {1, 2, 3, 4};
    float hy[N] = {10, 20, 30, 40};

    float *dx, *dy;

    CUDA_CHECK(cudaMalloc(&dx, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy, N * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK(launch_saxpy(2.0f, dx, dy, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i)
        printf("%f\n", hy[i]);

    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));

    return 0;
}
