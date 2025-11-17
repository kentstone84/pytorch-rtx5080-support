# RTX-STone Jupyter Notebooks

Interactive tutorials and examples for RTX-STone.

## Notebooks

### 1. Getting Started
**File:** `01_Getting_Started.ipynb`

Learn the basics:
- Verify installation
- Check GPU capabilities
- Run basic operations
- Benchmark performance
- Compare precisions (FP32, FP16, BF16)

### 2. Flash Attention 2
**File:** `02_Flash_Attention.ipynb`

Optimize attention mechanisms:
- What is Flash Attention?
- 1.5x speedup on long sequences
- Drop-in replacement for PyTorch SDPA
- Benchmark vs standard attention
- Use in transformers

### 3. Custom Triton Kernels
**File:** `03_Custom_Triton_Kernels.ipynb`

Write GPU kernels in Python:
- Introduction to Triton
- Vector addition kernel
- Fused operations (ReLU + Dropout)
- LayerNorm implementation
- Performance tuning

### 4. LLM Optimization
**File:** `04_LLM_Optimization.ipynb`

Optimize large language models:
- Flash Attention for transformers
- Fused RoPE and RMSNorm
- KV-cache optimization
- Mixed precision training
- Llama, Mistral, Qwen support

### 5. Image Generation
**File:** `05_Image_Generation.ipynb`

Optimize diffusion models:
- Stable Diffusion XL optimization
- FLUX model optimization
- Attention optimization
- VAE acceleration
- Memory optimization

## Running the Notebooks

### Prerequisites

```powershell
# Install RTX-STone
pip install rtx-stone

# Install Jupyter and dependencies
pip install jupyter matplotlib
```

### Launch Jupyter

```powershell
# Navigate to notebooks directory
cd notebooks

# Launch Jupyter
jupyter notebook
```

Your browser will open with the notebook interface.

## Tips

1. **GPU Memory**: Close other applications using the GPU
2. **Kernel Crashes**: Restart kernel if GPU runs out of memory
3. **Performance**: Results vary by specific GPU model and cooling
4. **Save Results**: Export benchmark results for comparison

## Requirements

- Windows 11
- Python 3.10 or 3.11
- RTX 50-series GPU (5090, 5080, 5070 Ti, 5070)
- NVIDIA Driver 570.00+
- 8GB+ GPU memory recommended

## Troubleshooting

### Out of Memory Errors

```python
# Clear GPU memory
import torch
torch.cuda.empty_cache()

# Reduce batch size or model size
```

### Kernel Crashes

- Restart Jupyter kernel
- Check GPU temperature
- Update NVIDIA drivers

### Slow Performance

- Ensure GPU is not throttling (check temperature)
- Close background applications
- Check if running on correct GPU: `torch.cuda.current_device()`

## Contributing

Have ideas for new notebooks? Open an issue or submit a PR!

- Example use cases
- Advanced optimization techniques
- Integration examples
- Performance comparisons

## Resources

- [RTX-STone Documentation](https://github.com/kentstone84/pytorch-rtx5080-support)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Triton Documentation](https://triton-lang.org/)

---

Happy Learning!
