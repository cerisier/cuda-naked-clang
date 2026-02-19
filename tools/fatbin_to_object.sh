#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <input.fatbin> <output.o> [clang++]" >&2
  exit 1
fi

IN_FATBIN=$1
OUT_OBJ=$2
CC=${3:-./llvm/bin/clang++}

if [[ ! -f "${IN_FATBIN}" ]]; then
  echo "Input fatbin does not exist: ${IN_FATBIN}" >&2
  exit 1
fi

TMP_ASM=$(mktemp /tmp/fatbinwrap.XXXXXX.s)
trap 'rm -f "${TMP_ASM}"' EXIT

# CUDA fatbin wrapper ABI used by host registration:
# struct { uint32_t magic; uint32_t version; void *data; void *unused; }
# magic: 0x466243b1, version: 1
cat > "${TMP_ASM}" <<EOF
.section .nv_fatbin,"a",@progbits
.p2align 3
.globl __cuda_fatbin
__cuda_fatbin:
  .incbin "${IN_FATBIN}"
.size __cuda_fatbin, .-__cuda_fatbin

.section .nvFatBinSegment,"aw",@progbits
.p2align 3
.globl __cuda_fatbin_wrapper
__cuda_fatbin_wrapper:
  .long 0x466243b1
  .long 0x1
  .quad __cuda_fatbin
  .quad 0
.size __cuda_fatbin_wrapper, .-__cuda_fatbin_wrapper
EOF

"${CC}" -c -x assembler "${TMP_ASM}" -o "${OUT_OBJ}"

echo "Wrote ${OUT_OBJ} from ${IN_FATBIN}"
