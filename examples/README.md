# Examples - RTX 5080/5090 Optimized Applications

Real-world examples demonstrating the power of native SM 12.0 + Triton on Windows.

## 📁 Directory Structure

```
examples/
├── getting_started.py       # Quick start guide
├── llama_chatbot.py         # Local Llama chatbot with Flash Attention
├── image_generation.py      # Stable Diffusion/FLUX optimization
├── benchmark_comparison.py  # Compare against stock PyTorch
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Getting Started

Run the getting started script to verify your installation:

```powershell
python examples/getting_started.py
```

This will:
- Check your GPU and SM 12.0 support
- Run basic PyTorch operations
- Test Triton compilation
- Benchmark performance

### 2. LLM Chatbot

Run a local Llama chatbot optimized for RTX 5080:

```powershell
python examples/llama_chatbot.py --model meta-llama/Llama-3.2-1B
```

Features:
- Flash Attention 2 (1.5x faster)
- Optimized KV-cache
- BF16 precision
- Interactive chat interface

### 3. Image Generation

Generate images with optimized Stable Diffusion:

```powershell
python examples/image_generation.py --prompt "A futuristic city at sunset"
```

Features:
- Fused attention kernels
- Optimized UNet forward pass
- Mixed precision support

## 📊 Performance Comparison

Compare your RTX 5080 build against stock PyTorch:

```powershell
python examples/benchmark_comparison.py
```

Expected results:
- **20-30% faster** matrix operations (SM 12.0 vs PTX)
- **1.5x faster** attention (Flash Attention 2)
- **10-15% faster** vs WSL2 (native Windows)

## 💡 Tips

1. **Always use BF16 or FP16** for best performance on Blackwell
2. **Enable Flash Attention** for any sequence length > 512
3. **Use auto-tuning** for custom kernels: `python autotune_rtx5080.py`
4. **Monitor GPU usage** with `nvidia-smi -l 1`

## 🔗 Additional Resources

- [Main README](../README.md)
- [Triton Examples](../triton_examples.py)
- [Flash Attention Implementation](../flash_attention_rtx5080.py)
- [HuggingFace Integration](../huggingface_rtx5080.py)

## 🐛 Troubleshooting

**Problem:** Flash Attention not working
**Solution:** Ensure Triton is installed: `pip install "triton-windows<3.6"`

**Problem:** Out of memory
**Solution:** Try smaller batch sizes or enable gradient checkpointing

**Problem:** Slow performance
**Solution:** Run `python compare_performance.py` to verify SM 12.0 is being used

## 📝 Contributing

Have a cool example? Submit a PR!

Requirements:
- Clear documentation
- Works on RTX 5080/5090
- Demonstrates performance improvement
- Includes error handling
