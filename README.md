Troubleshooting & How to Get sm_120 Support in PyTorch
Problem
You’re using a modern NVIDIA GPU with compute capability sm_120 (e.g., RTX 5080, 5060 Ti, 5090), but PyTorch throws errors saying your device isn’t supported or doesn’t recognize sm_120.

This happens because:

Official PyTorch pre-built binaries may not yet fully support sm_120.

Nightly builds claim to support it but sometimes fail due to packaging or dependency issues.

CUDA toolkit compatibility and PyTorch build flags may not include the latest architecture by default.

How to Verify Your Current Setup
Run this simple snippet to check your PyTorch version, CUDA version, and supported architectures:

python
Copy
Edit
import torch

print("PyTorch version:", torch.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA arch list:", torch.cuda.get_arch_list())
print("CUDA device:", torch.cuda.get_device_name(0))
print("Device capability:", torch.cuda.get_device_capability(0))
print("Test tensor on CUDA:", torch.randn(1).cuda())
If 'sm_120' is not in the arch list or your device capability isn’t (12, 0), your build likely does not support your GPU properly.

Solution: Build PyTorch from Source for sm_120 Support
Clone the PyTorch repo

bash
Copy
Edit
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
Checkout the latest stable or nightly branch

bash
Copy
Edit
git checkout tags/v2.8.0  # or master for nightly
git submodule sync
git submodule update --init --recursive
Set environment variables to enable sm_120

This is the crucial step. Add the sm_120 architecture to CUDA flags so it compiles kernels for your GPU:

On Linux/macOS:

bash
Copy
Edit
export TORCH_CUDA_ARCH_LIST="5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;10.0;12.0"
On Windows (PowerShell):

powershell
Copy
Edit
setx TORCH_CUDA_ARCH_LIST "5.0;6.0;6.1;7.0;7.5;8.0;8.6;9.0;10.0;12.0"
Install dependencies

Make sure you have CUDA 12.8 installed and your compiler is compatible.

Build PyTorch

bash
Copy
Edit
python setup.py clean
python setup.py install
This can take 30–60 minutes depending on your machine.

Additional Tips
Always use a clean environment (conda or virtualenv) to avoid package conflicts.

Make sure your NVIDIA drivers are up to date and compatible with CUDA 12.8.

Check dependencies of other PyTorch-related packages (torchaudio, torchvision) to avoid version mismatches.

If you’re using tools like torch-directml, beware of compatibility conflicts as they often require specific PyTorch versions.

For any errors during build, carefully check the logs; common issues are missing dependencies or incompatible compiler versions.

Why bother?
Pre-built PyTorch binaries are lagging behind the bleeding edge GPU releases and CUDA support. Building from source is the only guaranteed way to fully enable new architectures like sm_120 today without waiting months for official support.



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
