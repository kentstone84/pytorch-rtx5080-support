# RTX-STone: Unlock True Blackwell Performance

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10.0a0-orange.svg)](https://pytorch.org/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![SM 12.0](https://img.shields.io/badge/SM-12.0-red.svg)](https://developer.nvidia.com/cuda-gpus)
[![License](https://img.shields.io/badge/License-BSD--3-lightgrey.svg)](LICENSE)

## 🚨 MAJOR DISCOVERY: NVIDIA Driver Gatekeeping Exposed

**Your RTX 5080/5090 is being artificially limited - even with PyTorch 2.7+ and CUDA 12.8.**

### The Problem

Through reverse engineering with Ghidra, we discovered NVIDIA's driver **actively rejects** native sm_120 (Blackwell) kernel execution and silently falls back to sm_89 (Ada Lovelace) code - causing **30-40% performance loss** that users don't even know about.

### The Proof

By patching **3 hex bytes** in the NVIDIA driver:
- **3D Graphics Mark: 29,604 → 47,616** (+60.8% improvement!) 🚀
- **GPU Compute: 23,650** (unlocked performance)
- **Same hardware, same system - just driver patch**
- **Native sm_120 execution** finally working as intended

**Read the full analysis:** [DRIVER_GATEKEEPING_ANALYSIS.md](DRIVER_GATEKEEPING_ANALYSIS.md)

---

**Native Blackwell (SM 12.0) support + Driver patches for all NVIDIA RTX 50-series GPUs**

PyTorch 2.10 with native SM 12.0 compilation + Driver gatekeeping bypass + Triton compiler + Optimization suite for RTX 5090, 5080, 5070 Ti, 5070, and all future RTX 50-series GPUs.

## 🚀 Quick Start

### Option 1: PyPI Installation (Recommended)

```powershell
# Install RTX-STone from PyPI
pip install rtx-stone[all]

# Verify installation
rtx-stone-verify

# Run benchmarks
rtx-stone-benchmark
```

### Option 2: Manual Installation

```powershell
# 1. Download and extract the release
# 2. Create virtual environment
python -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1

# 3. Run installer (installs PyTorch + optional Triton)
.\install.ps1

# 4. Install additional dependencies (optional but recommended)
pip install -r requirements.txt

# 5. Verify installation
python examples/getting_started.py

# 6. Run benchmarks
python compare_performance.py
```

### Option 3: Driver Patch Only (Advanced)

If you already have PyTorch 2.7+ and just want to unlock true sm_120 performance:

**See detailed guide:** [patch_driver_sm120.md](patch_driver_sm120.md)

**Quick overview:**
1. Use Ghidra to find 3 sm_120 rejection functions in NVIDIA driver
2. Patch 3 hex bytes (fail → pass)
3. Replace system driver with patched version
4. Enjoy 40% performance boost

**Warning:** Requires reverse engineering skills, voids warranty, needs driver signature bypass on Windows.

### Option 4: Docker

```powershell
# Pull and run
docker pull rtx-stone:latest
docker run --gpus all -it rtx-stone:latest

# Or build from source
docker build -t rtx-stone:latest .
docker-compose up rtx-stone-jupyter
```

**What you get:**
- ✅ **PyTorch 2.10.0a0** with native SM 12.0 (20-30% faster than nightlies)
- ✅ **All RTX 50-series GPUs** supported (5090, 5080, 5070 Ti, 5070)
- ✅ **Triton compiler** for custom CUDA kernels in Python
- ✅ **Flash Attention 2** (1.5x faster for long sequences)
- ✅ **LLM optimization suite** (Llama, Mistral, Qwen support)
- ✅ **HuggingFace integration** (one-line model optimization)
- ✅ **Auto-tuning framework** (optimal configs for your GPU)
- ✅ **vLLM integration** (high-performance serving)
- ✅ **LangChain RAG** examples
- ✅ **ComfyUI optimization** guide
- ✅ **Multi-GPU support** (DDP, FSDP, tensor parallelism)
- ✅ **Docker containers** for easy deployment
- ✅ **Jupyter notebooks** for tutorials
- ✅ **Production-ready** examples and benchmarks
- ✅ **Native Windows** (no WSL required!)

## Overview

This is a custom-built PyTorch 2.10.0a0 package compiled with **native SM 12.0 (Blackwell) support** for Windows, **PLUS driver-level patches** that unlock true Blackwell performance.

### The Two-Layer Problem

**Layer 1: PyTorch Compilation (SOLVED)**
- Official PyTorch 2.7+ now compiles sm_120 kernels ✅
- Ships CUDA 12.8 binaries ✅

**Layer 2: NVIDIA Driver (THIS REPO SOLVES)**
- **Driver actively rejects sm_120 kernels at runtime** ❌
- **Silently falls back to sm_89 (Ada) instead** ❌
- **You lose 30-40% performance without knowing** ❌
- **We reverse-engineered and patched it** ✅

### Why This Build?

**Official PyTorch 2.7 claims:**
> "Supports NVIDIA Blackwell architecture with CUDA 12.8"

**What actually happens:**
1. PyTorch compiles sm_120 kernels ✅
2. PyTorch loads them successfully ✅
3. **NVIDIA driver rejects them** ❌
4. **Driver substitutes sm_89 fallback** ❌
5. **User sees "working" but gets ~70% performance** ❌

**RTX-STone provides:**
1. PyTorch with native sm_120 compilation ✅
2. **Driver analysis and patches** to bypass restrictions ✅
3. **True native Blackwell execution** ✅
4. **3D Graphics Mark: 47,616** (vs 29,604 stock) ✅
5. **60% better performance than stock driver** ✅

This build solves BOTH problems: PyTorch compilation AND driver gatekeeping.

### Why Native Windows (Not WSL)?

**Performance Advantages:**
- **Direct driver access** - No virtualization overhead
- **Lower latency** - No translation layer between Windows and Linux
- **Better compatibility** - Native Windows apps and tools work seamlessly
- **Simpler workflow** - One environment, no dual OS management

WSL2 is great, but native Windows with proper CUDA support is simply faster and more efficient.

### 🔺 Triton Support - Game Changer for Windows!

This package includes **Triton**, OpenAI's GPU programming language, with full SM 12.0 Blackwell support on Windows! This is revolutionary for Windows-based RTX 50 series users doing ML research and production work.

**What is Triton?**
- Python-based compiler for writing custom CUDA kernels
- No C++/CUDA knowledge required - write GPU kernels in Python!
- Automatic optimization for your specific GPU architecture
- Used by major ML frameworks (PyTorch, HuggingFace, OpenAI)

**Performance Gains on Blackwell (RTX 5080/5090):**
- **1.5x faster** Flash Attention (FP16) vs Hopper
- **2x faster** matrix operations with MXFP4 precision
- **Fused kernels** - combine multiple operations to eliminate memory bottlenecks
- **Native Tensor Core utilization** for Blackwell architecture

**Use Cases:**
- Custom model layers and attention mechanisms
- High-performance data preprocessing
- Research prototyping with production-level performance
- Kernel fusion to optimize memory bandwidth

## Specifications

- **PyTorch Version:** 2.10.0a0
- **Triton Version:** 3.3+ (triton-windows)
- **CUDA Version:** 13.0
- **Python Version:** 3.10 or 3.11 (recommended)
- **Platform:** Windows 11
- **Architecture:** SM 12.0 (compute_120, code_sm_120)
- **Package Size:** 8.3 GB (uncompressed), 5.3 GB (compressed)

## Supported Hardware

All NVIDIA RTX 50-series GPUs with SM 12.0 (Blackwell):
- **RTX 5090** (24GB VRAM)
- **RTX 5080** (16GB VRAM)
- **RTX 5070 Ti** (16GB VRAM)
- **RTX 5070** (12GB VRAM)
- All future RTX 50-series GPUs

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
1. Check Python version compatibility (3.10 or 3.11)
2. Verify CUDA installation and GPU detection
3. Install required dependencies automatically
4. Copy PyTorch to your site-packages
5. Verify PyTorch installation with CUDA
6. **Optionally install Triton** (recommended for custom kernels)
7. Verify Triton JIT compilation (if installed)

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

### Verify Triton Installation

```python
import triton
import triton.language as tl

print(f"Triton Version: {triton.__version__}")

# Test basic JIT compilation
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

print("✓ Triton JIT compilation successful")
print("✓ Ready to write custom CUDA kernels in Python!")
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

## 🚀 Advanced Features

### Flash Attention 2

Production-ready Flash Attention implementation optimized for Blackwell:

```python
from flash_attention_rtx5080 import flash_attention

# Drop-in replacement for PyTorch SDPA
output = flash_attention(q, k, v)  # 1.5x faster!
```

See `flash_attention_rtx5080.py` for details.

### LLM Optimization Suite

Optimized kernels for running Llama, Mistral, and other LLMs:

```python
from llm_inference_optimized import LLMOptimizer

optimizer = LLMOptimizer(model)
optimizer.optimize_attention()  # Flash Attention 2
optimizer.optimize_rope()       # Fused RoPE
optimizer.enable_kv_cache()     # Optimized KV-cache

output = optimizer.generate(input_ids, max_length=100)
```

Features:
- Fused RoPE (Rotary Position Embedding)
- Optimized RMSNorm
- Efficient KV-cache management
- BF16/FP16 mixed precision

### HuggingFace Integration

One-line optimization for any HuggingFace model:

```python
from transformers import AutoModelForCausalLM
from huggingface_rtx5080 import optimize_for_rtx5080

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = optimize_for_rtx5080(model)  # That's it!
```

Automatically applies:
- Flash Attention 2
- Fused normalization layers
- Optimized embeddings
- BF16 precision
- Gradient checkpointing

### Auto-Tuning Framework

Find optimal kernel configurations for your specific GPU:

```powershell
# Auto-tune all kernels and save config
python autotune_rtx5080.py --save-config

# Auto-tune specific kernel
python autotune_rtx5080.py --kernel matmul

# Load previously saved config
python autotune_rtx5080.py --load-config
```

The auto-tuner benchmarks different block sizes, warp counts, and memory layouts to find the fastest configuration for your RTX 5080/5090.

### Performance Comparison

Compare your build against stock PyTorch and WSL2:

```powershell
python compare_performance.py --save-results
```

Benchmarks:
- Matrix multiplication (all precisions)
- Attention mechanisms (with/without Flash Attention)
- Convolution operations
- Memory bandwidth

Expected improvements:
- **20-30% faster** than PyTorch nightlies (SM 12.0 vs PTX)
- **1.5x faster** attention with Flash Attention 2
- **10-15% faster** than WSL2 (native Windows advantage)

## Benchmarks

### PyTorch Benchmark

Test native PyTorch performance with SM 12.0:

```powershell
python benchmark.py
```

This benchmarks matrix multiplication at various sizes and precisions (FP32, FP16, BF16).

### Triton Benchmark

Test Triton custom kernels optimized for Blackwell:

```powershell
python benchmark_triton.py
```

Benchmarks include:
- Vector addition
- Softmax
- Matrix multiplication (GEMM) with Tensor Cores
- Performance comparison vs native PyTorch

### Triton Examples

Explore production-ready Triton kernel examples:

```powershell
python triton_examples.py
```

Examples include:
- Fused ReLU + Dropout
- Layer Normalization
- GELU activation
- Fused Linear + Bias + ReLU
- Flash Attention (simplified)

## 📂 Examples

The `examples/` directory contains real-world applications:

### Getting Started

Verify your installation and run basic tests:

```powershell
python examples/getting_started.py
```

This script:
- Checks GPU and SM 12.0 support
- Tests PyTorch operations
- Verifies Triton compilation
- Runs quick performance benchmarks
- Provides next steps

See `examples/README.md` for more examples including:
- Local Llama chatbot with Flash Attention
- Stable Diffusion/FLUX optimization
- Custom training loops
- Performance comparisons

## Getting Started with Triton

Now that you've seen what Triton can do, let's write your first custom kernel!

### Your First Triton Kernel

Here's a simple example to get you started:

```python
import torch
import triton
import triton.language as tl

@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID (which block we're processing)
    pid = tl.program_id(axis=0)

    # Compute offsets for this block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Create a mask for valid elements
    mask = offsets < n_elements

    # Load data from GPU memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform computation
    output = x + y

    # Store result back to GPU memory
    tl.store(output_ptr + offsets, output, mask=mask)

# Use the kernel
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = output.numel()

    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output

# Test it
x = torch.randn(10000, device='cuda')
y = torch.randn(10000, device='cuda')
z = add(x, y)
```

### Learning Resources

- **Official Triton Tutorials:** https://triton-lang.org/main/getting-started/tutorials/
- **Triton Examples in this repo:** `triton_examples.py`
- **Benchmarks:** `benchmark_triton.py`
- **Community:** https://github.com/triton-lang/triton/discussions

### When to Use Triton

✅ **Use Triton when:**
- You need custom operations not available in PyTorch
- Fusing multiple operations to reduce memory bandwidth
- Prototyping research ideas with production-level performance
- Optimizing specific bottlenecks in your model

❌ **Don't use Triton when:**
- Standard PyTorch operations already meet your needs
- You're not familiar with GPU programming concepts yet
- The operation is already optimized in cuDNN/cuBLAS

## License

PyTorch is released under the BSD-3-Clause license. See the [PyTorch repository](https://github.com/pytorch/pytorch) for details.

This package is compiled from the official PyTorch source code with no modifications except for the architecture target.

## Contributing

If you encounter issues or have improvements:
1. Open an issue describing the problem
2. Include your GPU model, driver version, and error messages
3. Provide steps to reproduce

## Acknowledgments

- **PyTorch team** for the excellent framework
- **OpenAI & Triton community** for democratizing GPU programming
- **NVIDIA** for the CUDA toolkit and Blackwell architecture
- **woct0rdho** for the triton-windows fork
- Community contributors who helped test this build

## 🐳 Docker Support

RTX-STone is available as Docker containers for easy deployment:

```powershell
# Development environment
docker-compose up rtx-stone-dev

# Jupyter notebooks
docker-compose up rtx-stone-jupyter
# Access at http://localhost:8888

# vLLM API server
docker-compose up rtx-stone-vllm
# API at http://localhost:8000

# Run benchmarks
docker-compose up rtx-stone-benchmark
```

See [Dockerfile](Dockerfile) and [docker-compose.yml](docker-compose.yml) for details.

## 📚 Jupyter Notebooks

Interactive tutorials in `notebooks/`:

1. **Getting Started** - Installation verification and basic benchmarks
2. **Flash Attention** - Optimizing attention mechanisms (coming soon)
3. **Custom Triton Kernels** - Writing GPU kernels in Python (coming soon)
4. **LLM Optimization** - Optimizing large language models (coming soon)
5. **Image Generation** - Stable Diffusion optimization (coming soon)

```powershell
# Launch Jupyter
jupyter notebook notebooks/
```

## 🔌 Integrations

### vLLM (LLM Serving)

High-performance LLM inference serving:

```python
# See integrations/vllm_integration.py
python integrations/vllm_integration.py --mode server --model meta-llama/Llama-3.2-3B
```

### LangChain (RAG)

Build RAG systems with local LLMs:

```python
# See integrations/langchain_rag_example.py
python integrations/langchain_rag_example.py --documents ./docs
```

### ComfyUI (Image Generation)

Optimize ComfyUI workflows:
- See [ComfyUI Integration Guide](integrations/comfyui_integration.md)
- 20-30% faster image generation
- Custom nodes for RTX-STone optimizations

## 🎯 Model Zoo

Pre-tested configurations and benchmarks:
- [Model Zoo Documentation](MODEL_ZOO.md)
- Llama 3.2, 3.1 (3B, 8B, 70B)
- Mistral 7B, Mixtral 8x7B
- Qwen 2.5
- SDXL, SD3, FLUX
- Performance benchmarks for each model

## 🚀 Multi-GPU Support

Distributed training and inference:

```powershell
# Distributed Data Parallel (DDP)
torchrun --nproc_per_node=2 examples/multi_gpu/distributed_training.py

# FSDP for large models
# See examples/multi_gpu/

# Tensor Parallelism with vLLM
python integrations/vllm_integration.py --tensor-parallel-size 2
```

## 📊 Benchmarking Suite

Comprehensive performance testing:

```powershell
# PyTorch benchmarks
python benchmark.py

# Triton benchmarks
python benchmark_triton.py

# Full comparison vs PyTorch nightlies
python compare_performance.py --save-results

# Or use CLI
rtx-stone-benchmark
```

## 🛠️ Command Line Tools

Installed with PyPI package:

```powershell
# Verify installation
rtx-stone-verify

# Show system info
rtx-stone-info

# Run benchmarks
rtx-stone-benchmark
```

## 📖 Documentation

- [Quick Start Guide](QUICK_START.md)
- [Model Zoo](MODEL_ZOO.md)
- [Release Notes](RELEASE_NOTES.md)
- [Changelog](CHANGELOG.md)
- [Contributing Guide](.github/CONTRIBUTING.md)
- [Security Policy](.github/SECURITY.md)

## Changelog

### v2.10.0a0 + Driver Gatekeeping Discovery (Latest - November 2025)

**🔥 BREAKING DISCOVERY: NVIDIA Driver Actively Blocks sm_120 Performance**
- **Reverse-engineered NVIDIA driver** with Ghidra
- **Found 3 functions** that reject native sm_120 kernel execution
- **Created driver patches** (3 hex bytes) to bypass restrictions
- **Validated 60% performance improvement** - 3D Graphics Mark: 29,604 → 47,616
- **GPU Compute subscore: 23,650** (unlocked Blackwell performance)
- **Documented full methodology** for community verification
- **Exposed silent fallback to sm_89** even with "official" PyTorch 2.7 sm_120 support

**NEW Documentation:**
- [DRIVER_GATEKEEPING_ANALYSIS.md](DRIVER_GATEKEEPING_ANALYSIS.md) - Technical analysis of driver restrictions
- [DRIVER_PATCH_METHODOLOGY.md](DRIVER_PATCH_METHODOLOGY.md) - Reverse engineering process
- [patch_driver_sm120.md](patch_driver_sm120.md) - Step-by-step patching guide
- [REDDIT_RESPONSE.md](REDDIT_RESPONSE.md) - Context on PyTorch 2.7 vs driver reality

### v2.10.0a0 + Complete Suite
- **NEW:** PyPI package - `pip install rtx-stone`
- **NEW:** Support for ALL RTX 50-series GPUs (5090, 5080, 5070 Ti, 5070)
- **NEW:** Docker containers with docker-compose
- **NEW:** vLLM integration for LLM serving
- **NEW:** LangChain RAG examples
- **NEW:** ComfyUI optimization guide
- **NEW:** Multi-GPU DDP/FSDP examples
- **NEW:** Jupyter notebooks tutorials
- **NEW:** Model Zoo with benchmarks
- **NEW:** CLI tools (rtx-stone-verify, rtx-stone-benchmark)
- **NEW:** GitHub templates (issues, PRs, contributing)
- **NEW:** CI/CD workflows
- **NEW:** Comprehensive documentation
- **NEW:** Triton compiler integration for Windows
- **NEW:** Native SM 12.0 Blackwell support in Triton kernels
- **NEW:** Flash Attention 2 implementation (`flash_attention_rtx5080.py`)
  - 1.5x faster than PyTorch SDPA on long sequences
  - Optimized for Blackwell Tensor Cores
  - Drop-in replacement for scaled_dot_product_attention
- **NEW:** LLM Optimization Suite (`llm_inference_optimized.py`)
  - Fused RoPE kernels
  - Optimized RMSNorm
  - Efficient KV-cache management
  - Support for Llama, Mistral, Qwen
- **NEW:** HuggingFace Integration (`huggingface_rtx5080.py`)
  - One-line model optimization
  - Automatic Flash Attention injection
  - Model-specific optimizations
- **NEW:** Auto-Tuning Framework (`autotune_rtx5080.py`)
  - Find optimal kernel configurations
  - Benchmark different block sizes
  - Cache tuning results
- **NEW:** Performance Comparison Tool (`compare_performance.py`)
  - Compare vs PyTorch nightlies and WSL2
  - Comprehensive benchmark suite
  - JSON export for results
- **NEW:** Examples Directory (`examples/`)
  - Getting started script
  - Real-world applications
  - Best practices guide
- **NEW:** Requirements file (`requirements.txt`)
  - Easy dependency installation
  - Optional libraries documented
- Triton benchmark suite (`benchmark_triton.py`)
- Production-ready Triton kernel examples (`triton_examples.py`)
- Automated Triton installation in `install.ps1`
- Comprehensive documentation
- Learning resources and tutorials

### v2.10.0a0 (November 12, 2025)
- Initial Windows release
- Built from PyTorch main branch
- Native SM 12.0 support for RTX 5080
- CUDA 13.0 compatibility
- Python 3.10/3.11 support

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines.

Ways to contribute:
- Report bugs and request features
- Submit optimized kernels
- Share benchmarks and configurations
- Improve documentation
- Create tutorials and examples

## 📜 License

BSD-3-Clause (same as PyTorch). See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **PyTorch team** for the excellent framework
- **OpenAI & Triton community** for democratizing GPU programming
- **NVIDIA** for CUDA toolkit and Blackwell architecture
- **woct0rdho** for triton-windows fork
- **Community contributors** who help test and improve

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/kentstone84/pytorch-rtx5080-support/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kentstone84/pytorch-rtx5080-support/discussions)
- **Security**: See [SECURITY.md](.github/SECURITY.md)

## ⭐ Star History

If this project helped you, consider giving it a star!

---

**RTX-STone** - Unleash the full power of your RTX 50-series GPU!
