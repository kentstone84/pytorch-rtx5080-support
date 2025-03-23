# Blackwell (sm_120) Patch for PyTorch

This patch adds support for NVIDIA's next-gen Blackwell GPUs (`sm_120`) to PyTorch's build system.

## What It Does

✅ Adds `Blackwell` as a CUDA architecture alias  
✅ Enables `sm_120` compilation via `TORCH_CUDA_ARCH_LIST`  
✅ Useful for forward compatibility with RTX 5090 / B100

## Usage

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
/path/to/torch-blackwell-patch/patch_blackwell.sh
