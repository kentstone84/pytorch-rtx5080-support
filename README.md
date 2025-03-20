# 🚀 PyTorch Support for RTX 5080 (sm_120) – CUDA 12.8  

### ✅ **What is this?**
PyTorch **does not support the NVIDIA RTX 5080 (sm_120) GPU natively**, which means AI developers **cannot take full advantage of its power**. This project **fixes that** by building **custom CUDA 12.8 PyTorch binaries** and adding **RTX 5080 support** for AI workloads.

---

## **⚡ Features**
- **Full PyTorch CUDA Support for RTX 5080**
- **Optimized drivers for CUDA 12.8**
- **Prebuilt wheels & source build guide**
- **Matrix multiplication & AI model benchmarks**
- **Docker container setup for easy builds**
- **Works with: TensorRT, FlashAttention, LLaMA, Stable Diffusion, etc.**

---

## **📥 Installation**
### **🔹 Option 1: Use Prebuilt PyTorch Binaries**
1. Download the **prebuilt PyTorch CUDA 12.8 wheels** from the [Releases](https://github.com/YOUR_USERNAME/RTX5080-PyTorch/releases) page.
2. Install it:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
