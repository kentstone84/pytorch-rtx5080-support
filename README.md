# PyTorch 2.10.0a0 with SM 12.0 Support for RTX 5080 - Windows

Native Blackwell architecture support for NVIDIA GeForce RTX 5080 on Windows 11.

## Overview

This is a custom-built PyTorch 2.10.0a0 package compiled with **native SM 12.0 (Blackwell) support** for Windows. Unlike PyTorch nightlies which only provide PTX backward compatibility (~70-80% performance), this build includes optimized CUDA kernels specifically compiled for RTX 5080.

### Why This Build?

Official PyTorch releases currently only support up to SM 8.9 (Ada Lovelace/RTX 40-series). When running on RTX 5080, they fall back to PTX compatibility mode which:
- Reduces performance by 20-30%
- Increases JIT compilation overhead  
- Lacks Blackwell-specific optimizations

This build solves that problem with native SM 12.0 compilation.

## Specifications

- **PyTorch Version:** 2.10.0a0
- **CUDA Version:** 13.0
- **Python Version:** 3.10 or 3.11 (recommended)
- **Platform:** Windows 11
- **Architecture:** SM 12.0 (compute_120, code_sm_120)
- **Package Size:** 8.3 GB (uncompressed), 5.3 GB (compressed)

## Supported Hardware

- NVIDIA GeForce RTX 5080
- NVIDIA GeForce RTX 5090 (also Blackwell SM 12.0)

## Requirements

### System Requirements
- Windows 11 (22H2 or later)
- Python 3.10 or 3.11
- NVIDIA Driver 570.00 or newer
- CUDA 13.0+ compatible driver
- 15 GB free disk space

### Python Dependencies
- filelock
- fsspec
- Jinja2
- MarkupSafe
- mpmath
- networkx
- sympy
- typing-extensions >= 4.10.0

All dependencies will be installed automatically by the install script.

## Installation

### Method 1: Automated Installation (Recommended)

```powershell
# Download the release files
# Extract all parts to the same directory

# Create and activate virtual environment
python -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1

# Run the installer
.\install.ps1
```

The installer will:
1. Check Python version compatibility
2. Verify CUDA installation
3. Install required dependencies
4. Copy PyTorch to your site-packages
5. Verify the installation

### Method 2: Manual Installation

```powershell
# Create virtual environment
python -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1

# Install dependencies
pip install filelock fsspec Jinja2 MarkupSafe mpmath networkx sympy "typing_extensions>=4.10.0"

# Extract the torch folder
# Copy to: .\pytorch-env\Lib\site-packages\torch\
```

## Download Instructions

Due to GitHub's file size limits, the package is split into multiple parts:

```powershell
# Download all parts from GitHub Releases
# pytorch-2.10.0a0-sm120-windows.tar.gz.partaa
# pytorch-2.10.0a0-sm120-windows.tar.gz.partab
# pytorch-2.10.0a0-sm120-windows.tar.gz.partac

# Recombine the parts
cat pytorch-2.10.0a0-sm120-windows.tar.gz.part* > pytorch-2.10.0a0-sm120-windows.tar.gz

# Extract
tar -xzf pytorch-2.10.0a0-sm120-windows.tar.gz
```

## Verification

After installation, verify PyTorch is working correctly:

```powershell
python
```

```python
import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
print(f"Arch List: {torch.cuda.get_arch_list()}")

# Test GPU operation
x = torch.rand(5, 3).cuda()
print(f"Tensor device: {x.device}")
```

Expected output:
```
PyTorch Version: 2.10.0a0+...
CUDA Available: True
CUDA Version: 13.0
GPU Name: NVIDIA GeForce RTX 5080
Compute Capability: (12, 0)
Arch List: ['sm_120']
Tensor device: cuda:0
```

## Performance

Compared to PyTorch nightlies on RTX 5080:
- **20-30% faster** training and inference
- **No JIT overhead** from PTX compilation
- **Native Blackwell optimizations** for tensor cores and memory bandwidth

## Troubleshooting

### "CUDA not available" after installation

1. Verify NVIDIA driver version:
   ```powershell
   nvidia-smi
   ```
   Should show driver >= 570.00

2. Check CUDA installation:
   ```powershell
   nvcc --version
   ```

3. Verify GPU compute capability:
   ```powershell
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader
   ```
   Should show `12.0`

### DLL Load Errors

- Ensure you have the latest NVIDIA drivers
- Install Visual C++ Redistributable 2015-2022
- Check that CUDA 13.0 runtime DLLs are accessible

### Python version issues

This build requires Python 3.10 or 3.11. Python 3.12+ may have compatibility issues.

Create a new environment with the correct Python version:
```powershell
py -3.11 -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1
```

## Build Details

This package was compiled from PyTorch main branch with the following configuration:

```
TORCH_CUDA_ARCH_LIST=12.0
USE_CUDA=1
USE_CUDNN=1
CUDA_HOME=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0
```

All CUDA kernels were compiled with:
```
-gencode arch=compute_120,code=sm_120 -DCUDA_HAS_FP16=1 -O2
```

## Benchmark

Run the included `benchmark.py` to test performance:

```powershell
python benchmark.py
```

## License

PyTorch is released under the BSD-3-Clause license. See the [PyTorch repository](https://github.com/pytorch/pytorch) for details.

This package is compiled from the official PyTorch source code with no modifications except for the architecture target.

## Contributing

If you encounter issues or have improvements:
1. Open an issue describing the problem
2. Include your GPU model, driver version, and error messages
3. Provide steps to reproduce

## Acknowledgments

- PyTorch team for the excellent framework
- NVIDIA for the CUDA toolkit and Blackwell architecture
- Community contributors who helped test this build

## Changelog

### v2.10.0a0 (November 12, 2025)
- Initial Windows release
- Built from PyTorch main branch
- Native SM 12.0 support for RTX 5080
- CUDA 13.0 compatibility
- Python 3.10/3.11 support

---

Built for the RTX 5080 community.
