# 🧠 Blackwell (sm_120) Patch for PyTorch

**🚀 Unlock full RTX 5080 performance in PyTorch!**  
PyTorch does not yet support `sm_120` (Blackwell) natively — so I built custom CUDA 12.8 drivers and patched the PyTorch build system.

This repo includes the patch, a script, and build instructions.

---

## ✅ What It Does

- ✅ Adds `"Blackwell"` as a CUDA architecture alias
- ✅ Enables `sm_120` compilation via `TORCH_CUDA_ARCH_LIST`
- ✅ Future-ready for RTX 5090, B100, GB200 series
- ✅ Compatible with CUDA 12.8, PyTorch 2.5.0+

---

## 🛠 Usage

### Step 1 – Clone PyTorch and This Patch Repo
git clone --recursive https://github.com/pytorch/pytorch
git clone https://github.com/kentstone84/pytorch-rtx5080-support.git

### Step 2 – Apply Patch
cd pytorch
../pytorch-rtx5080-support/patch_blackwell.sh

### Step 3 – Build PyTorch
export TORCH_CUDA_ARCH_LIST="Blackwell"
python setup.py install


✅ Test It Worked
  python
    import torch
    print(torch.cuda.get_device_properties(0))
    # Should show major=12, minor=0 → sm_120
