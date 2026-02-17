#!/usr/bin/env bash
set -euo pipefail
set -x

CUDA_HOME=/usr/local/cuda-12.9
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_LIB=${CUDA_HOME}/lib64
COMPAT_INCLUDE=compat/cuda

CLANG=./llvm/bin/clang++
LLC=./llvm/bin/llc
LLVM_LINK=./llvm/bin/llvm-link
PTXAS=${CUDA_HOME}/bin/ptxas
FATBINARY=${CUDA_HOME}/bin/fatbinary

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
        -I${CUDA_INCLUDE} \
        -emit-llvm -S \
        ${SRC} \
        -o kernel.${SM}.ll

    echo "== Lower to PTX (${SM})"
    $LLC \
        -march=nvptx64 \
        -mcpu=${SM} \
        kernel.${SM}.ll \
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
