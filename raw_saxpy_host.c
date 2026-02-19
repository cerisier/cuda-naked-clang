#include <cuda.h>

#include <stdio.h>
#include <stdlib.h>

#define N 4

#define CU_CHECK(call)                                                          \
    do {                                                                        \
        CUresult _err = (call);                                                 \
        if (_err != CUDA_SUCCESS) {                                             \
            const char *name = NULL;                                            \
            const char *msg = NULL;                                             \
            cuGetErrorName(_err, &name);                                        \
            cuGetErrorString(_err, &msg);                                       \
            fprintf(stderr, "CUDA driver error at %s:%d: %s (%s)\\n",         \
                    __FILE__, __LINE__,                                         \
                    name ? name : "<unknown>",                                 \
                    msg ? msg : "<no message>");                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

int main(void) {
    const float a = 2.0f;
    const float h_x[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float h_y[N] = {10.0f, 20.0f, 30.0f, 40.0f};
    float h_out[N] = {0.0f, 0.0f, 0.0f, 0.0f};

    CUdevice device;
    CUcontext ctx;
    CUmodule module;
    CUfunction kernel;
    CUdeviceptr d_x;
    CUdeviceptr d_y;
    CUdeviceptr d_out;

    CU_CHECK(cuInit(0));
    CU_CHECK(cuDeviceGet(&device, 0));
    CU_CHECK(cuCtxCreate(&ctx, 0, device));

    CU_CHECK(cuMemAlloc(&d_x, N * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_y, N * sizeof(float)));
    CU_CHECK(cuMemAlloc(&d_out, N * sizeof(float)));

    CU_CHECK(cuMemcpyHtoD(d_x, h_x, N * sizeof(float)));
    CU_CHECK(cuMemcpyHtoD(d_y, h_y, N * sizeof(float)));

    CU_CHECK(cuModuleLoad(&module, "raw_saxpy.sm_120.cubin"));
    CU_CHECK(cuModuleGetFunction(&kernel, module, "saxpy_raw"));

    int n = N;
    void *args[] = {
        &n,
        (void *)&a,
        &d_x,
        &d_y,
        &d_out,
    };

    const unsigned int block_x = 256;
    const unsigned int grid_x = (N + block_x - 1) / block_x;

    CU_CHECK(cuLaunchKernel(kernel,
                            grid_x, 1, 1,
                            block_x, 1, 1,
                            0,
                            NULL,
                            args,
                            NULL));
    CU_CHECK(cuCtxSynchronize());

    CU_CHECK(cuMemcpyDtoH(h_out, d_out, N * sizeof(float)));

    for (int i = 0; i < N; ++i) {
        printf("%f\n", h_out[i]);
    }

    CU_CHECK(cuMemFree(d_out));
    CU_CHECK(cuMemFree(d_y));
    CU_CHECK(cuMemFree(d_x));
    CU_CHECK(cuModuleUnload(module));
    CU_CHECK(cuCtxDestroy(ctx));

    return 0;
}
