#!/bin/bash

set -e

echo "[+] Patching PyTorch for Blackwell (sm_120)..."

PATCH_FILE="$(dirname "$0")/patch_blackwell.diff"

if [ ! -f "$PATCH_FILE" ]; then
    echo "[!] Patch file not found: $PATCH_FILE"
    exit 1
fi

patch -p1 < "$PATCH_FILE"

echo "[+] Patch applied successfully."
