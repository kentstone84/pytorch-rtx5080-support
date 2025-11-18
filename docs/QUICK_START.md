# Quick Start Guide - PyTorch RTX 5080/5090

Get up and running in **5 minutes**! ⚡

---

## ⚡ TL;DR - Fastest Path

```powershell
# 1. Extract release files
# 2. Create environment
python -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1

# 3. Install
.\install.ps1

# 4. Test
python examples/getting_started.py
```

**Done!** You now have PyTorch with native SM 12.0 + Triton! 🎉

---

## 📦 Step-by-Step Installation

### Prerequisites

✅ Windows 11 (22H2+)
✅ RTX 5080 or RTX 5090 GPU
✅ NVIDIA Driver 570.00+
✅ Python 3.10 or 3.11
✅ 15 GB free space

### Installation Steps

**1. Download Release**
- Go to [Releases](https://github.com/kentstone84/pytorch-rtx5080-support/releases)
- Download the latest release
- Extract to a folder (e.g., `C:\ML\pytorch-rtx5080`)

**2. Create Virtual Environment**
```powershell
cd C:\ML\pytorch-rtx5080
python -m venv pytorch-env
.\pytorch-env\Scripts\Activate.ps1
```

**3. Run Installer**
```powershell
.\install.ps1
```

The installer will:
- ✅ Check Python version
- ✅ Verify GPU and drivers
- ✅ Install PyTorch with SM 12.0
- ✅ Optionally install Triton
- ✅ Verify installation

**4. Install Optional Dependencies** (Recommended)
```powershell
pip install -r requirements.txt
```

This adds:
- HuggingFace transformers
- Jupyter notebooks
- Performance monitoring
- Quantization support

**5. Verify Everything Works**
```powershell
python examples/getting_started.py
```

---

## 🚀 Your First 10 Minutes

### Test Basic Operations

```python
import torch

# Check GPU
print(torch.cuda.get_device_name(0))
# Output: NVIDIA GeForce RTX 5080

# Check SM 12.0
print(torch.cuda.get_arch_list())
# Output: ['sm_120']

# Run on GPU
x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = x @ y
print(f"✓ GPU computation works! Device: {z.device}")
```

### Test Triton

```python
import triton
print(f"Triton version: {triton.__version__}")
# Output: Triton version: 3.5.x
```

### Test Flash Attention

```python
from flash_attention_rtx5080 import flash_attention
import torch

q = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 512, 64, device='cuda', dtype=torch.float16)

output = flash_attention(q, k, v)
print(f"✓ Flash Attention works! Shape: {output.shape}")
```

---

## 🎯 Common Tasks

### Run Benchmarks

```powershell
# PyTorch benchmark
python benchmark.py

# Triton benchmark
python benchmark_triton.py

# Full comparison
python compare_performance.py
```

### Optimize a HuggingFace Model

```python
from transformers import AutoModelForCausalLM
from huggingface_rtx5080 import optimize_for_rtx5080

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    torch_dtype=torch.float16,
    device_map="cuda"
)

# One-line optimization!
model = optimize_for_rtx5080(model)

# Now 20-30% faster! 🚀
```

### Auto-Tune Kernels

```powershell
# Find optimal configs for your GPU
python autotune_rtx5080.py --save-config

# Results cached at: ~/.pytorch_rtx5080/autotune_cache.json
```

---

## 📚 Next Steps

### Learn Triton
1. Read: `triton_examples.py` - 5 production kernels
2. Experiment: Modify examples
3. Build: Your custom kernel

### Optimize LLMs
1. Check: `llm_inference_optimized.py`
2. Try: Llama 3 with optimizations
3. Deploy: Your chatbot

### Explore Flash Attention
1. Study: `flash_attention_rtx5080.py`
2. Benchmark: Your model
3. Compare: vs standard attention

---

## 🐛 Troubleshooting

### CUDA Not Available

```powershell
# Check driver
nvidia-smi

# Should show driver >= 570.00
```

**Fix:** Update NVIDIA drivers

### Triton Import Error

```powershell
# Install Triton
pip install "triton-windows<3.6"
```

### Out of Memory

```python
# Reduce batch size
batch_size = 16  # Try 8 or 4

# Or enable gradient checkpointing
model.gradient_checkpointing_enable()
```

### Slow Performance

```python
# Check if using SM 12.0
import torch
print(torch.cuda.get_arch_list())

# Should contain 'sm_120'
# If not, reinstall with SM 12.0 build
```

---

## 💡 Pro Tips

### Maximum Performance

```python
# Enable TF32 for Tensor Cores
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

# Use BF16 for best Blackwell performance
model = model.to(torch.bfloat16)
```

### Memory Optimization

```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache between runs
torch.cuda.empty_cache()

# Monitor memory
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
```

### Debugging

```python
# Enable detailed CUDA errors
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Check for compilation issues
torch._C._cuda_init()
```

---

## 🎓 Learning Resources

### Official Docs
- [PyTorch Docs](https://pytorch.org/docs)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

### This Repository
- `README.md` - Complete documentation
- `RELEASE_NOTES.md` - What's new
- `examples/` - Real-world examples
- `triton_examples.py` - Kernel examples

### Community
- [GitHub Issues](https://github.com/kentstone84/pytorch-rtx5080-support/issues)
- [GitHub Discussions](https://github.com/kentstone84/pytorch-rtx5080-support/discussions)

---

## ✅ Checklist for Success

Before deploying to production:

- [ ] Ran `examples/getting_started.py` successfully
- [ ] Verified SM 12.0 with `torch.cuda.get_arch_list()`
- [ ] Benchmarked performance with `compare_performance.py`
- [ ] Tested Flash Attention with your model
- [ ] Auto-tuned kernels with `autotune_rtx5080.py`
- [ ] Read troubleshooting section
- [ ] Have fun building amazing ML applications! 🚀

---

## 🎉 You're Ready!

You now have the most powerful Windows ML platform for RTX 5080/5090!

**What will you build?** Share your projects and help grow the community!

---

**Questions?** Open an issue or discussion on GitHub!
**Found a bug?** Please report it so we can fix it!
**Built something cool?** Show it off in Discussions!

Happy coding! 🔥
