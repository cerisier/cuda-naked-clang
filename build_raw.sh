#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME=/usr/local/cuda-12.9
CUDA_INCLUDE=${CUDA_HOME}/include
CUDA_LIB=${CUDA_HOME}/lib64

CLANG=./llvm/bin/clang
PTXAS=${CUDA_HOME}/bin/ptxas

DEVICE_SRC=raw_saxpy_device.c
HOST_SRC=raw_saxpy_host.c
SM=sm_120

echo "== Device compile to PTX (${SM})"
$CLANG \
    -v \
    --target=nvptx64-nvidia-cuda \
    -O3 \
    -S \
    ${DEVICE_SRC} \
    -o raw_saxpy.ptx

echo "== Assemble PTX to cubin (${SM})"
$PTXAS \
    -arch=${SM} \
    raw_saxpy.ptx \
    -o raw_saxpy.${SM}.cubin

echo "== Build host executable (driver API only, no cudart)"
$CLANG \
    -O2 \
    ${HOST_SRC} \
    -I${CUDA_INCLUDE} \
    -L${CUDA_LIB} \
    -Wl,-rpath,${CUDA_LIB} \
    -lcuda \
    -o app_raw

echo "Build complete: ./app_raw"
