"""
Auto-Tuning Framework for RTX 5080/5090 (Blackwell SM 12.0)

Automatically finds optimal kernel configurations for your specific GPU:
- Block sizes
- Warps per SM
- Shared memory usage
- Register allocation

Usage:
    python autotune_rtx5080.py --kernel matmul --save-config
"""

import torch
import triton
import triton.language as tl
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import itertools


class KernelAutoTuner:
    """
    Automatically tune Triton kernels for RTX 5080/5090.

    This class benchmarks different kernel configurations and
    finds the optimal settings for your specific GPU.
    """

    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        self.config_cache_path = Path.home() / '.pytorch_rtx5080' / 'autotune_cache.json'

        # Get GPU properties
        props = torch.cuda.get_device_properties(device)
        self.gpu_name = props.name
        self.sm_count = props.multi_processor_count
        self.max_shared_mem = props.shared_memory_per_multiprocessor

        print(f"\n🔧 Auto-Tuner for {self.gpu_name}")
        print(f"   SM Count: {self.sm_count}")
        print(f"   Shared Memory per SM: {self.max_shared_mem / 1024:.0f} KB\n")

    def tune_matmul(self, sizes=[2048, 4096, 8192]):
        """
        Auto-tune matrix multiplication kernel.

        Tests different block sizes and finds the best configuration.
        """
        print("="*70)
        print("Auto-Tuning Matrix Multiplication")
        print("="*70 + "\n")

        # Configuration space to search
        block_sizes = [16, 32, 64, 128, 256]
        configs = list(itertools.product(block_sizes, block_sizes, block_sizes))

        best_configs = {}

        for size in sizes:
            print(f"Matrix Size: {size}x{size}")
            best_time = float('inf')
            best_config = None

            # Create test matrices
            a = torch.randn(size, size, device=self.device, dtype=torch.float16)
            b = torch.randn(size, size, device=self.device, dtype=torch.float16)

            # Test each configuration
            for block_m, block_n, block_k in configs:
                # Skip invalid configs
                if block_m > size or block_n > size or block_k > size:
                    continue

                try:
                    # Benchmark this config
                    time_ms = self._benchmark_matmul_config(
                        a, b, block_m, block_n, block_k
                    )

                    if time_ms < best_time:
                        best_time = time_ms
                        best_config = (block_m, block_n, block_k)

                except Exception as e:
                    # Skip configs that fail
                    continue

            if best_config:
                block_m, block_n, block_k = best_config
                flops = 2 * size ** 3
                tflops = flops / (best_time * 1e-3) / 1e12

                print(f"  Best Config: BLOCK_M={block_m}, BLOCK_N={block_n}, BLOCK_K={block_k}")
                print(f"  Performance: {best_time:.2f} ms ({tflops:.2f} TFLOPS)")
                print()

                best_configs[size] = {
                    'block_m': block_m,
                    'block_n': block_n,
                    'block_k': block_k,
                    'time_ms': best_time,
                    'tflops': tflops,
                }

        self.results['matmul'] = best_configs
        return best_configs

    def _benchmark_matmul_config(self, a, b, block_m, block_n, block_k, iterations=5):
        """Benchmark a specific matmul configuration."""
        from triton_examples import matmul_kernel

        m, k = a.shape
        k, n = b.shape
        c = torch.empty((m, n), device=a.device, dtype=a.dtype)

        grid = lambda META: (
            triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(n, META['BLOCK_SIZE_N']),
        )

        # Warmup
        matmul_kernel[grid](
            a, b, c, m, n, k,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
        )

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            matmul_kernel[grid](
                a, b, c, m, n, k,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                BLOCK_SIZE_M=block_m,
                BLOCK_SIZE_N=block_n,
                BLOCK_SIZE_K=block_k,
            )

        torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations

        return elapsed * 1000  # Return in milliseconds

    def tune_softmax(self, sizes=[1024, 2048, 4096, 8192]):
        """Auto-tune softmax kernel."""
        print("="*70)
        print("Auto-Tuning Softmax")
        print("="*70 + "\n")

        block_sizes = [32, 64, 128, 256, 512, 1024, 2048]
        best_configs = {}

        for size in sizes:
            print(f"Softmax Size: {size}")
            best_time = float('inf')
            best_block = None

            x = torch.randn(128, size, device=self.device, dtype=torch.float32)

            for block_size in block_sizes:
                if block_size < size:
                    continue

                try:
                    time_ms = self._benchmark_softmax_config(x, block_size)

                    if time_ms < best_time:
                        best_time = time_ms
                        best_block = block_size

                except Exception:
                    continue

            if best_block:
                print(f"  Best Config: BLOCK_SIZE={best_block}")
                print(f"  Performance: {best_time:.2f} ms")
                print()

                best_configs[size] = {
                    'block_size': best_block,
                    'time_ms': best_time,
                }

        self.results['softmax'] = best_configs
        return best_configs

    def _benchmark_softmax_config(self, x, block_size, iterations=10):
        """Benchmark a specific softmax configuration."""
        from triton_examples import fused_softmax_kernel

        n_rows, n_cols = x.shape
        y = torch.empty_like(x)

        num_warps = 4 if block_size <= 1024 else 8

        # Warmup
        fused_softmax_kernel[(n_rows,)](
            y, x, x.stride(0), y.stride(0), n_cols,
            num_warps=num_warps,
            BLOCK_SIZE=block_size,
        )

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(iterations):
            fused_softmax_kernel[(n_rows,)](
                y, x, x.stride(0), y.stride(0), n_cols,
                num_warps=num_warps,
                BLOCK_SIZE=block_size,
            )

        torch.cuda.synchronize()
        elapsed = (time.time() - start) / iterations

        return elapsed * 1000

    def save_config(self, filepath=None):
        """Save tuned configurations to disk."""
        if filepath is None:
            filepath = self.config_cache_path

        # Create directory if needed
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Add GPU info
        config = {
            'gpu_name': self.gpu_name,
            'sm_count': self.sm_count,
            'results': self.results,
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"\n✓ Saved configuration to: {filepath}\n")

    def load_config(self, filepath=None):
        """Load previously tuned configurations."""
        if filepath is None:
            filepath = self.config_cache_path

        if not filepath.exists():
            print(f"⚠️  No cached config found at {filepath}")
            return None

        with open(filepath, 'r') as f:
            config = json.load(f)

        # Verify GPU matches
        if config['gpu_name'] != self.gpu_name:
            print(f"⚠️  Config is for {config['gpu_name']}, you have {self.gpu_name}")
            print("   Re-tuning recommended")

        self.results = config.get('results', {})
        print(f"✓ Loaded configuration from: {filepath}\n")

        return self.results

    def run_full_autotune(self):
        """Run comprehensive auto-tuning for all kernels."""
        print("\n" + "="*70)
        print("Running Full Auto-Tune for RTX 5080/5090")
        print("="*70 + "\n")
        print("This may take 5-10 minutes...\n")

        # Tune all kernels
        self.tune_matmul()
        self.tune_softmax()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print summary of tuning results."""
        print("\n" + "="*70)
        print("Auto-Tune Summary")
        print("="*70 + "\n")

        for kernel, configs in self.results.items():
            print(f"{kernel.upper()}:")
            for size, config in configs.items():
                print(f"  Size {size}: {config}")
            print()


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Auto-tune Triton kernels for RTX 5080/5090"
    )
    parser.add_argument(
        '--kernel',
        choices=['matmul', 'softmax', 'all'],
        default='all',
        help="Which kernel to tune"
    )
    parser.add_argument(
        '--save-config',
        action='store_true',
        help="Save tuning results to disk"
    )
    parser.add_argument(
        '--load-config',
        action='store_true',
        help="Load previously saved config"
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return

    tuner = KernelAutoTuner()

    # Load existing config if requested
    if args.load_config:
        tuner.load_config()
        tuner.print_summary()
        return

    # Run tuning
    if args.kernel == 'matmul':
        tuner.tune_matmul()
    elif args.kernel == 'softmax':
        tuner.tune_softmax()
    else:
        tuner.run_full_autotune()

    # Save if requested
    if args.save_config:
        tuner.save_config()

    tuner.print_summary()


if __name__ == "__main__":
    main()
