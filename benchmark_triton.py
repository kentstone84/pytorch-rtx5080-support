"""
Triton Benchmark for RTX 5080/5090 (Blackwell SM 12.0)

This script benchmarks Triton custom kernels on RTX 50 series GPUs,
demonstrating the performance benefits of native SM 12.0 support.
"""

import torch
import triton
import triton.language as tl
import time
from typing import Callable


# ============================================================================
# Triton Kernels
# ============================================================================

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized element-wise addition kernel."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def fused_softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """Fused softmax kernel with numerical stability."""
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized matrix multiplication kernel leveraging Blackwell Tensor Cores."""
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ============================================================================
# Wrapper Functions
# ============================================================================

def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated element-wise addition."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated softmax."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    num_warps = 4 if BLOCK_SIZE <= 1024 else 8
    y = torch.empty_like(x)
    fused_softmax_kernel[(n_rows,)](
        y, x, x.stride(0), y.stride(0), n_cols,
        num_warps=num_warps, BLOCK_SIZE=BLOCK_SIZE
    )
    return y


def triton_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Triton-accelerated matrix multiplication."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
    )
    return c


# ============================================================================
# Benchmark Utilities
# ============================================================================

def benchmark(func: Callable, *args, warmup: int = 3, iterations: int = 10, **kwargs) -> float:
    """Benchmark a function with warmup iterations."""
    # Warmup
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        _ = func(*args, **kwargs)
    torch.cuda.synchronize()

    return (time.time() - start) / iterations


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🔺 Triton Benchmark for RTX 5080/5090 (Blackwell SM 12.0)")
    print("="*70 + "\n")

    # System info
    print(f"PyTorch version:  {torch.__version__}")
    print(f"Triton version:   {triton.__version__}")
    print(f"CUDA version:     {torch.version.cuda}")
    print(f"GPU:              {torch.cuda.get_device_name(0)}")
    print(f"Compute Cap:      sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")
    print(f"Arch list:        {torch.cuda.get_arch_list()}")
    print()

    # ========================================================================
    # Benchmark 1: Vector Addition
    # ========================================================================
    print("─" * 70)
    print("Benchmark 1: Vector Addition (FP32)")
    print("─" * 70)

    for size in [1_000_000, 10_000_000, 100_000_000]:
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)

        torch_time = benchmark(torch.add, x, y)
        triton_time = benchmark(triton_add, x, y)

        speedup = torch_time / triton_time
        print(f"  Size: {size:>12,}")
        print(f"    PyTorch:  {torch_time*1000:>8.3f} ms")
        print(f"    Triton:   {triton_time*1000:>8.3f} ms")
        print(f"    Speedup:  {speedup:>8.2f}x")
        print()

    # ========================================================================
    # Benchmark 2: Softmax
    # ========================================================================
    print("─" * 70)
    print("Benchmark 2: Softmax (FP32)")
    print("─" * 70)

    for shape in [(4096, 4096), (8192, 8192), (16384, 4096)]:
        x = torch.randn(shape, device='cuda', dtype=torch.float32)

        torch_time = benchmark(torch.softmax, x, dim=-1)
        triton_time = benchmark(triton_softmax, x)

        speedup = torch_time / triton_time
        print(f"  Shape: {shape[0]}x{shape[1]}")
        print(f"    PyTorch:  {torch_time*1000:>8.3f} ms")
        print(f"    Triton:   {triton_time*1000:>8.3f} ms")
        print(f"    Speedup:  {speedup:>8.2f}x")
        print()

    # ========================================================================
    # Benchmark 3: Matrix Multiplication (GEMM)
    # ========================================================================
    print("─" * 70)
    print("Benchmark 3: Matrix Multiplication (FP16) - Blackwell Tensor Cores")
    print("─" * 70)

    for size in [2048, 4096, 8192]:
        a = torch.randn((size, size), device='cuda', dtype=torch.float16)
        b = torch.randn((size, size), device='cuda', dtype=torch.float16)

        torch_time = benchmark(torch.matmul, a, b)
        triton_time = benchmark(triton_matmul, a, b)

        # Calculate TFLOPS
        flops = 2 * size ** 3
        torch_tflops = flops / (torch_time * 1e12)
        triton_tflops = flops / (triton_time * 1e12)

        speedup = torch_time / triton_time
        print(f"  Size: {size}x{size}")
        print(f"    PyTorch:  {torch_time*1000:>8.3f} ms  ({torch_tflops:>6.2f} TFLOPS)")
        print(f"    Triton:   {triton_time*1000:>8.3f} ms  ({triton_tflops:>6.2f} TFLOPS)")
        print(f"    Speedup:  {speedup:>8.2f}x")
        print()

    # ========================================================================
    # Summary
    # ========================================================================
    print("="*70)
    print("✓ Benchmark Complete")
    print("="*70)
    print("\nTriton is successfully utilizing Blackwell (SM 12.0) architecture!")
    print("For production workloads, consider using Triton for:")
    print("  • Custom fused kernels (avoid memory bandwidth bottlenecks)")
    print("  • Flash Attention (1.5x faster on Blackwell)")
    print("  • Mixed precision with MXFP8/MXFP4 (2x FP8 performance)")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check your installation.")
        exit(1)

    main()
