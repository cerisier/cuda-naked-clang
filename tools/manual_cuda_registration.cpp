#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdint>

extern "C" {

struct FatbinWrapper {
    uint32_t magic;
    uint32_t version;
    const void *data;
    void *unused;
};

extern const FatbinWrapper __cuda_fatbin_wrapper;

void **__cudaRegisterFatBinary(void *fatCubin);
void __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void __cudaUnregisterFatBinary(void **fatCubinHandle);
int __cudaRegisterFunction(void **fatCubinHandle,
                           const void *hostFun,
                           char *deviceFun,
                           const char *deviceName,
                           int thread_limit,
                           uint3 *tid,
                           uint3 *bid,
                           dim3 *bDim,
                           dim3 *gDim,
                           int *wSize);

// Host-side function token used by cudart registration and cudaLaunchKernel.
void __cuda_stub_saxpy() {}
}

static void **g_fatbin_handle = nullptr;

static void cuda_module_dtor() {
    if (g_fatbin_handle) {
        __cudaUnregisterFatBinary(g_fatbin_handle);
        g_fatbin_handle = nullptr;
    }
}

__attribute__((constructor)) static void cuda_module_ctor() {
    g_fatbin_handle = __cudaRegisterFatBinary(
        const_cast<void *>(static_cast<const void *>(&__cuda_fatbin_wrapper)));

    // Device entry name from PTX/cubin (`.entry _Z5saxpyfPfS_`).
    __cudaRegisterFunction(g_fatbin_handle,
                           reinterpret_cast<const void *>(&__cuda_stub_saxpy),
                           const_cast<char *>("_Z5saxpyfPfS_"),
                           "_Z5saxpyfPfS_",
                           -1,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr,
                           nullptr);

    __cudaRegisterFatBinaryEnd(g_fatbin_handle);
    atexit(cuda_module_dtor);
}

extern "C" cudaError_t launch_saxpy(float a, float *x, float *y, int n) {
    dim3 grid(1, 1, 1);
    dim3 block(static_cast<unsigned>(n), 1, 1);
    void *args[] = {&a, &x, &y};
    return cudaLaunchKernel(reinterpret_cast<const void *>(&__cuda_stub_saxpy),
                            grid,
                            block,
                            args,
                            0,
                            nullptr);
}
