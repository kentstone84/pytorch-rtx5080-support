#!/bin/bash
set -e

echo "[+] Patching PyTorch for Blackwell (sm_120)..."

# Apply patch
patch -p1 < patch_blackwell.diff

echo "[✓] Patch applied successfully. You can now build with TORCH_CUDA_ARCH_LIST=\"Blackwell\""
