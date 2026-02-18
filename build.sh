#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME=/usr/local/cuda-12.9
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_LIB=${CUDA_HOME}/lib64
COMPAT_INCLUDE=compat/cuda

CLANG=./llvm/bin/clang++
LLC=./llvm/bin/llc
LLVM_LINK=./llvm/bin/llvm-link
LLVM_NM=./llvm/bin/llvm-nm
PTXAS=${CUDA_HOME}/bin/ptxas
FATBINARY=${CUDA_HOME}/bin/fatbinary
LIBDEVICE=${CUDA_HOME}/nvvm/libdevice/libdevice.10.bc

SRC=kernel.cu
SMS=(sm_120 sm_90 sm_80)

build_device() {
    local SM=$1

    echo "== Device IR generation (${SM})"
    $CLANG \
        -std=c++20 \
        --cuda-device-only \
        --cuda-gpu-arch=${SM} \
        -nocudalib \
        -nocudainc \
        -include __clang_cuda_runtime_wrapper.h \
        -Xclang -internal-isystem -Xclang ${CUDA_INCLUDE} \
        -emit-llvm -c \
        ${SRC} \
        -o kernel.${SM}.bc

    echo "== Validate unresolved libdevice symbol before link (${SM})"
    if ! $LLVM_NM kernel.${SM}.bc | rg -q " U __nv_erfcinvf$"; then
        echo "Expected unresolved __nv_erfcinvf in kernel.${SM}.bc"
        exit 1
    fi

    echo "== Link libdevice (${SM})"
    $LLVM_LINK \
        --only-needed \
        kernel.${SM}.bc \
        ${LIBDEVICE} \
        -o kernel.${SM}.linked.bc

    echo "== Validate linked libdevice symbol (${SM})"
    if ! $LLVM_NM kernel.${SM}.linked.bc | rg -q " T __nv_erfcinvf$"; then
        echo "Expected defined __nv_erfcinvf in kernel.${SM}.linked.bc"
        exit 1
    fi

    echo "== Lower to PTX (${SM})"
    $LLC \
        -march=nvptx64 \
        -mcpu=${SM} \
        kernel.${SM}.linked.bc \
        -o kernel.${SM}.ptx

    echo "== Assemble to cubin (${SM})"
    $PTXAS \
        -arch=${SM} \
        kernel.${SM}.ptx \
        -o kernel.${SM}.cubin
}

# Build all configured architectures
for SM in "${SMS[@]}"; do
    build_device "${SM}"
done

echo "== Create fatbin"
FATBIN_ARGS=(--64 --create=kernel.fatbin)
for SM in "${SMS[@]}"; do
    SM_NUM=${SM#sm_}
    FATBIN_ARGS+=(--image3=kind=elf,sm=${SM_NUM},file=kernel.${SM}.cubin)
done
$FATBINARY "${FATBIN_ARGS[@]}"

echo "== Compile host side only"
$CLANG \
    -std=c++20 \
    --cuda-host-only \
    -I${CUDA_INCLUDE} \
    -Xclang -fcuda-include-gpubinary \
    -Xclang kernel.fatbin \
    -c ${SRC} \
    -o kernel.host.o

echo "== Link final executable"
$CLANG \
    kernel.host.o \
    -L${CUDA_LIB} \
    -lcudart \
    -ldl -lrt -pthread \
    -o app

echo "Build complete: ./app"
