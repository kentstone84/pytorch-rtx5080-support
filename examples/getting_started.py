"""
Getting Started with PyTorch + Triton on RTX 5080/5090

This script verifies your installation and runs basic tests to ensure
everything is working correctly.

Usage:
    python examples/getting_started.py
"""

import torch
import sys
import platform


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title)
    print("="*70 + "\n")


def check_system_info():
    """Check and display system information."""
    print_section("System Information")

    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python Version: {platform.python_version()}")
    print(f"PyTorch Version: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    else:
        print("❌ CUDA not available!")
        return False

    return True


def check_gpu_info():
    """Check and display GPU information."""
    print_section("GPU Information")

    if not torch.cuda.is_available():
        print("❌ No CUDA GPU detected")
        return False

    props = torch.cuda.get_device_properties(0)

    print(f"GPU Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"SM Count: {props.multi_processor_count}")
    print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"CUDA Arch List: {torch.cuda.get_arch_list()}")

    # Check for SM 12.0
    if 'sm_120' in torch.cuda.get_arch_list():
        print("\n✓ Native SM 12.0 (Blackwell) support ENABLED")
        print("  You're getting maximum performance!")
    else:
        print("\n⚠️  Native SM 12.0 support NOT detected")
        print("  You may be using PTX compatibility mode (20-30% slower)")
        print("  Ensure you installed the SM 12.0 build")

    return True


def test_basic_operations():
    """Test basic PyTorch operations."""
    print_section("Testing Basic Operations")

    try:
        # Test tensor creation
        x = torch.randn(1000, 1000, device='cuda')
        print("✓ Tensor creation on GPU")

        # Test matrix multiplication
        y = torch.randn(1000, 1000, device='cuda')
        z = torch.matmul(x, y)
        print("✓ Matrix multiplication")

        # Test mixed precision
        with torch.cuda.amp.autocast():
            z_fp16 = torch.matmul(x.half(), y.half())
        print("✓ Mixed precision (FP16)")

        # Test BF16 (Blackwell optimized)
        z_bf16 = torch.matmul(x.bfloat16(), y.bfloat16())
        print("✓ BFloat16 operations")

        print("\n✓ All basic operations passed!")
        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_triton():
    """Test Triton compilation."""
    print_section("Testing Triton")

    try:
        import triton
        import triton.language as tl

        print(f"Triton Version: {triton.__version__}")

        # Test simple kernel compilation
        @triton.jit
        def add_kernel(x_ptr, y_ptr, output_ptr, n, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            output = x + y
            tl.store(output_ptr + offsets, output, mask=mask)

        # Test kernel launch
        size = 1024
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        output = torch.empty_like(x)

        grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
        add_kernel[grid](x, y, output, size, BLOCK_SIZE=256)

        # Verify correctness
        expected = x + y
        if torch.allclose(output, expected, rtol=1e-5):
            print("✓ Triton kernel compilation successful")
            print("✓ Triton kernel execution correct")
            print("\n✓ Triton is working correctly!")
            return True
        else:
            print("❌ Triton kernel output incorrect")
            return False

    except ImportError:
        print("⚠️  Triton not installed")
        print("   Install with: pip install 'triton-windows<3.6'")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def run_quick_benchmark():
    """Run a quick performance benchmark."""
    print_section("Quick Performance Benchmark")

    import time

    sizes = [2048, 4096, 8192]
    print("Matrix Multiplication (FP16):\n")

    for size in sizes:
        a = torch.randn(size, size, device='cuda', dtype=torch.float16)
        b = torch.randn(size, size, device='cuda', dtype=torch.float16)

        # Warmup
        _ = torch.matmul(a, b)

        # Benchmark
        torch.cuda.synchronize()
        start = time.time()

        for _ in range(10):
            c = torch.matmul(a, b)

        torch.cuda.synchronize()
        elapsed = (time.time() - start) / 10

        # Calculate TFLOPS
        flops = 2 * size ** 3
        tflops = flops / elapsed / 1e12

        print(f"  {size}x{size}: {elapsed*1000:.2f} ms ({tflops:.2f} TFLOPS)")

    print("\nExpected on RTX 5080:")
    print("  ~120 TFLOPS for FP16/BF16")
    print("  ~60 TFLOPS for FP32")


def print_next_steps():
    """Print recommendations for next steps."""
    print_section("Next Steps")

    print("✓ Your installation is working!")
    print("\nRecommended next steps:\n")

    print("1. Run comprehensive benchmarks:")
    print("   python compare_performance.py\n")

    print("2. Try Flash Attention 2:")
    print("   python flash_attention_rtx5080.py\n")

    print("3. Optimize a HuggingFace model:")
    print("   python huggingface_rtx5080.py\n")

    print("4. Auto-tune kernels for your GPU:")
    print("   python autotune_rtx5080.py --save-config\n")

    print("5. Explore Triton examples:")
    print("   python triton_examples.py\n")

    print("For documentation, see README.md")


def main():
    """Run all checks and tests."""
    print("\n" + "🔥"*35)
    print("  Getting Started with PyTorch RTX 5080/5090")
    print("🔥"*35)

    all_passed = True

    # Run checks
    if not check_system_info():
        sys.exit(1)

    if not check_gpu_info():
        sys.exit(1)

    if not test_basic_operations():
        all_passed = False

    if not test_triton():
        all_passed = False

    if all_passed:
        run_quick_benchmark()
        print_next_steps()
        print("\n✓ All checks passed!")
    else:
        print("\n⚠️  Some checks failed. Please review the errors above.")

    print()


if __name__ == "__main__":
    main()
