# naked CUDA split-compilation demo (Clang + LLVM + NVIDIA tools)

This repository showcases a minimal end-to-end CUDA build pipeline where device and host compilation are handled separately.

## What it demonstrates

- Device-only compilation from `kernel.cu` to LLVM bitcode with `clang++`
- Explicit linking of `libdevice`
- Lowering linked bitcode to PTX with `llc`, then assembling to cubin with `ptxas`
- Building multi-architecture images (`sm_80`, `sm_90`, `sm_120`) and packaging them into `kernel.fatbin`
- Host-only compilation and embedding of the generated fatbin, then final link against `cudart`

This is to illustrate a fully explicit CUDA compilation graph without `--cuda-path`.

## Requirements

- LLVM 21 (installed in ./llvm)
- Nvidia 50xx series

## Quick start

```bash
./build.sh
./app
```
