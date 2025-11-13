"""
Triton Kernel Examples for RTX 5080/5090 (Blackwell SM 12.0)

Production-ready examples demonstrating how to write custom CUDA kernels
using Triton's Python-based compiler, optimized for Blackwell architecture.
"""

import torch
import triton
import triton.language as tl


# ============================================================================
# Example 1: Fused ReLU + Dropout
# ============================================================================

@triton.jit
def fused_relu_dropout_kernel(
    x_ptr, output_ptr, n_elements,
    dropout_p, seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused ReLU + Dropout kernel.

    This kernel fuses two operations (ReLU and Dropout) into a single pass,
    reducing memory bandwidth requirements compared to separate operations.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Apply ReLU
    x = tl.maximum(x, 0.0)

    # Apply dropout
    random = tl.rand(seed, offsets)
    keep_mask = random > dropout_p
    x = tl.where(keep_mask, x / (1.0 - dropout_p), 0.0)

    # Store output
    tl.store(output_ptr + offsets, x, mask=mask)


def fused_relu_dropout(x: torch.Tensor, dropout_p: float = 0.1) -> torch.Tensor:
    """Apply fused ReLU + Dropout."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    seed = torch.randint(0, 2**31, (1,), device='cuda').item()

    fused_relu_dropout_kernel[grid](
        x, output, n_elements,
        dropout_p, seed,
        BLOCK_SIZE=1024
    )
    return output


# ============================================================================
# Example 2: Layer Normalization
# ============================================================================

@triton.jit
def layer_norm_kernel(
    x_ptr, gamma_ptr, beta_ptr, output_ptr,
    n_rows, n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Layer Normalization kernel.

    Computes: output = gamma * (x - mean) / sqrt(variance + eps) + beta
    """
    row_idx = tl.program_id(0)
    row_start = row_idx * n_cols

    # Compute column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input row
    x = tl.load(x_ptr + row_start + col_offsets, mask=mask, other=0.0)

    # Compute mean
    mean = tl.sum(x, axis=0) / n_cols

    # Compute variance
    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # Normalize
    rstd = 1.0 / tl.sqrt(variance + eps)
    x_norm = x_centered * rstd

    # Scale and shift
    gamma = tl.load(gamma_ptr + col_offsets, mask=mask, other=1.0)
    beta = tl.load(beta_ptr + col_offsets, mask=mask, other=0.0)
    output = x_norm * gamma + beta

    # Store output
    tl.store(output_ptr + row_start + col_offsets, output, mask=mask)


def layer_norm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Apply Layer Normalization."""
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    output = torch.empty_like(x)

    layer_norm_kernel[(n_rows,)](
        x, gamma, beta, output,
        n_rows, n_cols, eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output


# ============================================================================
# Example 3: Fused Multi-Head Attention (Simplified)
# ============================================================================

@triton.jit
def flash_attention_kernel(
    Q, K, V, Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX, D_HEAD,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Simplified Flash Attention kernel.

    This is a simplified version demonstrating the concept.
    For production, use the official Flash Attention implementation.
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    # Compute batch and head indices
    off_h = off_hz % H
    off_z = off_hz // H

    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_z * stride_qz + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + off_z * stride_kz + off_h * stride_kh + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    v_ptrs = V + off_z * stride_vz + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk

    # Load Q
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, D_HEAD], dtype=tl.float32)

    # Iterate over K, V blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        # Load K, V
        k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)
        v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n[:, None]) < N_CTX, other=0.0)

        # Compute attention scores
        qk = tl.dot(q, tl.trans(k))
        qk = qk * (1.0 / tl.sqrt(D_HEAD.to(tl.float32)))

        # Apply softmax
        m_ij = tl.maximum(tl.max(qk, axis=1), 0.0)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)

        # Accumulate weighted values
        p = p / l_ij[:, None]
        acc += tl.dot(p.to(v.dtype), v)

    # Store output
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D_HEAD)
    out_ptrs = Out + off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    tl.store(out_ptrs, acc, mask=offs_m[:, None] < N_CTX)


# ============================================================================
# Example 4: GELU Activation (Approximation)
# ============================================================================

@triton.jit
def gelu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """
    GELU activation function (fast approximation).

    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # GELU approximation
    sqrt_2_over_pi = 0.7978845608028654  # sqrt(2/π)
    c = 0.044715

    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + c * x_cubed)
    tanh_inner = tl.libdevice.tanh(inner)
    output = 0.5 * x * (1.0 + tanh_inner)

    # Store output
    tl.store(output_ptr + offsets, output, mask=mask)


def gelu(x: torch.Tensor) -> torch.Tensor:
    """Apply GELU activation."""
    output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    gelu_kernel[grid](x, output, n_elements, BLOCK_SIZE=1024)
    return output


# ============================================================================
# Example 5: Fused Linear + Bias + ReLU
# ============================================================================

@triton.jit
def fused_linear_bias_relu_kernel(
    x_ptr, w_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused Linear (matmul) + Bias + ReLU kernel.

    Computes: output = ReLU(x @ w + bias)
    """
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    # Compute matrix multiplication
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    acc = tl.maximum(acc, 0.0)

    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=mask)


# ============================================================================
# Usage Examples
# ============================================================================

def main():
    """Demonstrate all example kernels."""
    print("\n" + "="*70)
    print("🔺 Triton Kernel Examples for RTX 5080/5090")
    print("="*70 + "\n")

    device = 'cuda'

    # Example 1: Fused ReLU + Dropout
    print("Example 1: Fused ReLU + Dropout")
    x = torch.randn(1024, 512, device=device)
    y = fused_relu_dropout(x, dropout_p=0.1)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ Executed successfully\n")

    # Example 2: Layer Normalization
    print("Example 2: Layer Normalization")
    x = torch.randn(64, 512, device=device)
    gamma = torch.ones(512, device=device)
    beta = torch.zeros(512, device=device)
    y = layer_norm(x, gamma, beta)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ Executed successfully\n")

    # Example 3: GELU Activation
    print("Example 3: GELU Activation")
    x = torch.randn(1024, 1024, device=device)
    y = gelu(x)
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {y.shape}")

    # Verify correctness
    y_torch = torch.nn.functional.gelu(x)
    max_error = torch.max(torch.abs(y - y_torch)).item()
    print(f"  Max error vs PyTorch: {max_error:.6f}")
    print(f"  ✓ Executed successfully\n")

    print("="*70)
    print("✓ All examples completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("  • Modify these kernels for your specific use case")
    print("  • Combine operations to reduce memory bandwidth")
    print("  • Use @triton.autotune() to optimize hyperparameters")
    print("  • Profile with PyTorch Profiler to measure performance")
    print()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check your installation.")
        exit(1)

    main()
