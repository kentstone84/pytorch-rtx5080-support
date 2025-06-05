#!/bin/bash

set -e

echo "[+] Patching PyTorch for Blackwell (sm_120)..."

PATCH_FILE="$(dirname "$0")/patch_blackwell.diff"

if git apply --check "$PATCH_FILE" >/dev/null 2>&1; then
  patch -p1 < "$PATCH_FILE"
  echo "[+] Patch applied successfully."
else
  echo "[!] Patch appears to be already applied or cannot be applied cleanly."
  exit 1
fi

patch -p1 < "$PATCH_FILE"

echo "[+] Patch applied successfully."
