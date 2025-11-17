"""
Diagnostic and verification tools for RTX-STone
"""

import sys
import subprocess
from typing import Dict, Optional, Tuple


def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor in (10, 11):
        return True, f"Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"Python {version.major}.{version.minor}.{version.micro} (requires 3.10 or 3.11)"


def check_platform() -> Tuple[bool, str]:
    """Check if running on Windows."""
    if sys.platform == "win32":
        try:
            import platform
            return True, f"Windows {platform.release()} {platform.version()}"
        except:
            return True, "Windows (version unknown)"
    else:
        return False, f"{sys.platform} (requires Windows)"


def check_pytorch() -> Tuple[bool, str, Dict]:
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()

        info = {
            "version": version,
            "cuda_available": cuda_available,
        }

        if cuda_available:
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["device_count"] = torch.cuda.device_count()

        return True, version, info
    except ImportError:
        return False, "Not installed", {}


def check_gpu() -> Tuple[bool, str, Dict]:
    """Check GPU information."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No CUDA GPU detected", {}

        gpu_name = torch.cuda.get_device_name(0)
        compute_cap = torch.cuda.get_device_capability(0)
        props = torch.cuda.get_device_properties(0)

        info = {
            "name": gpu_name,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}",
            "total_memory_gb": props.total_memory / (1024**3),
            "sm_version": f"SM {compute_cap[0]}.{compute_cap[1]}",
            "arch_list": torch.cuda.get_arch_list(),
        }

        # Check if SM 12.0 (RTX 50-series)
        is_rtx_50 = compute_cap[0] == 12 and compute_cap[1] == 0
        status = "RTX 50-series (Blackwell)" if is_rtx_50 else f"SM {compute_cap[0]}.{compute_cap[1]}"

        return is_rtx_50, status, info
    except Exception as e:
        return False, f"Error: {e}", {}


def check_nvidia_driver() -> Tuple[bool, str]:
    """Check NVIDIA driver version."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            driver_version = result.stdout.strip()
            # Check if driver is recent enough (570+)
            try:
                major_version = int(driver_version.split(".")[0])
                is_compatible = major_version >= 570
                return is_compatible, driver_version
            except:
                return True, driver_version
        else:
            return False, "nvidia-smi failed"
    except FileNotFoundError:
        return False, "nvidia-smi not found"
    except Exception as e:
        return False, f"Error: {e}"


def check_triton() -> Tuple[bool, str]:
    """Check Triton installation."""
    try:
        import triton
        version = triton.__version__
        return True, version
    except ImportError:
        return False, "Not installed (optional)"


def check_gpu_support() -> bool:
    """Quick check if current GPU is supported (RTX 50-series)."""
    is_supported, _, _ = check_gpu()
    return is_supported


def show_info():
    """Display system and GPU information."""
    print("=" * 70)
    print("RTX-STone System Information")
    print("=" * 70)

    # Python version
    py_ok, py_ver = check_python_version()
    status = "✓" if py_ok else "✗"
    print(f"\n{status} Python: {py_ver}")

    # Platform
    plat_ok, plat_info = check_platform()
    status = "✓" if plat_ok else "✗"
    print(f"{status} Platform: {plat_info}")

    # PyTorch
    pt_ok, pt_ver, pt_info = check_pytorch()
    status = "✓" if pt_ok else "✗"
    print(f"{status} PyTorch: {pt_ver}")
    if pt_ok and pt_info.get("cuda_available"):
        print(f"  - CUDA: {pt_info.get('cuda_version', 'unknown')}")
        print(f"  - cuDNN: {pt_info.get('cudnn_version', 'unknown')}")
        print(f"  - Devices: {pt_info.get('device_count', 0)}")

    # NVIDIA Driver
    drv_ok, drv_ver = check_nvidia_driver()
    status = "✓" if drv_ok else "✗"
    print(f"{status} NVIDIA Driver: {drv_ver}")

    # GPU
    gpu_ok, gpu_status, gpu_info = check_gpu()
    status = "✓" if gpu_ok else "⚠" if gpu_info else "✗"
    print(f"\n{status} GPU: {gpu_status}")
    if gpu_info:
        print(f"  - Name: {gpu_info.get('name', 'unknown')}")
        print(f"  - Compute Capability: {gpu_info.get('compute_capability', 'unknown')}")
        print(f"  - Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")
        print(f"  - Architecture: {gpu_info.get('sm_version', 'unknown')}")
        if gpu_info.get('arch_list'):
            print(f"  - Compiled Architectures: {', '.join(gpu_info['arch_list'])}")

    # Triton
    triton_ok, triton_ver = check_triton()
    status = "✓" if triton_ok else "○"
    print(f"\n{status} Triton: {triton_ver}")

    # Overall status
    print("\n" + "=" * 70)
    if gpu_ok:
        print("✓ RTX-STone is fully supported on this system!")
        print("  Your RTX 50-series GPU will run at optimal performance.")
    elif gpu_info:
        print("⚠ RTX-STone is installed but GPU is not RTX 50-series")
        print(f"  Detected: {gpu_info.get('name', 'unknown')}")
        print("  Performance optimizations are designed for RTX 50-series (SM 12.0)")
    else:
        print("✗ No compatible GPU detected")
        print("  RTX-STone requires an NVIDIA RTX 50-series GPU")

    print("=" * 70)


def verify_installation():
    """Verify RTX-STone installation and run basic tests."""
    print("\n" + "=" * 70)
    print("RTX-STone Installation Verification")
    print("=" * 70)

    all_ok = True

    # Check Python
    py_ok, py_ver = check_python_version()
    print(f"\n{'✓' if py_ok else '✗'} Python version: {py_ver}")
    all_ok = all_ok and py_ok

    # Check platform
    plat_ok, plat_info = check_platform()
    print(f"{'✓' if plat_ok else '✗'} Platform: {plat_info}")
    all_ok = all_ok and plat_ok

    # Check PyTorch
    pt_ok, pt_ver, pt_info = check_pytorch()
    print(f"{'✓' if pt_ok else '✗'} PyTorch: {pt_ver}")
    all_ok = all_ok and pt_ok

    # Check CUDA
    if pt_ok:
        cuda_ok = pt_info.get("cuda_available", False)
        print(f"{'✓' if cuda_ok else '✗'} CUDA available: {cuda_ok}")
        all_ok = all_ok and cuda_ok

    # Check driver
    drv_ok, drv_ver = check_nvidia_driver()
    print(f"{'✓' if drv_ok else '✗'} NVIDIA Driver: {drv_ver}")
    all_ok = all_ok and drv_ok

    # Check GPU
    gpu_ok, gpu_status, gpu_info = check_gpu()
    print(f"{'✓' if gpu_ok else '⚠' if gpu_info else '✗'} GPU: {gpu_status}")
    if gpu_info:
        print(f"    Name: {gpu_info.get('name', 'unknown')}")
        print(f"    Compute: {gpu_info.get('compute_capability', 'unknown')}")
        print(f"    Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")

    # Run basic GPU test
    if pt_ok and pt_info.get("cuda_available"):
        print("\nRunning basic GPU test...")
        try:
            import torch
            x = torch.randn(1000, 1000, device="cuda")
            y = torch.randn(1000, 1000, device="cuda")
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            print("✓ GPU computation test: PASSED")
        except Exception as e:
            print(f"✗ GPU computation test: FAILED ({e})")
            all_ok = False

    # Check Triton
    triton_ok, triton_ver = check_triton()
    print(f"\n{'✓' if triton_ok else '○'} Triton: {triton_ver}")

    # Final status
    print("\n" + "=" * 70)
    if all_ok and gpu_ok:
        print("✓ Installation verified! RTX-STone is ready to use.")
        print("\nNext steps:")
        print("  - Run benchmarks: rtx-stone-benchmark")
        print("  - Try examples: python examples/getting_started.py")
        print("  - Read docs: https://github.com/kentstone84/pytorch-rtx5080-support")
    elif all_ok and gpu_info:
        print("⚠ Installation verified, but GPU is not RTX 50-series")
        print(f"  Detected: {gpu_info.get('name', 'unknown')}")
        print("  RTX-STone optimizations are designed for RTX 50-series GPUs")
    else:
        print("✗ Installation verification failed")
        print("  Please check the errors above and ensure:")
        print("  - Python 3.10 or 3.11 is installed")
        print("  - NVIDIA Driver 570+ is installed")
        print("  - RTX 50-series GPU is present")

    print("=" * 70 + "\n")

    return all_ok and gpu_ok


def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("\n" + "=" * 70)
    print("RTX-STone Benchmarking Suite")
    print("=" * 70)

    # Check if GPU is available
    gpu_ok, gpu_status, gpu_info = check_gpu()
    if not gpu_ok:
        print(f"\n✗ Cannot run benchmarks: {gpu_status}")
        return

    print(f"\nGPU: {gpu_info.get('name', 'unknown')}")
    print(f"Compute Capability: {gpu_info.get('compute_capability', 'unknown')}")
    print(f"Memory: {gpu_info.get('total_memory_gb', 0):.1f} GB")

    print("\nRunning benchmarks...")
    print("This may take a few minutes...\n")

    # Import and run benchmarks
    try:
        import sys
        from pathlib import Path
        import subprocess

        # Run benchmark scripts
        scripts = [
            ("PyTorch Benchmark", "benchmark.py"),
            ("Triton Benchmark", "benchmark_triton.py"),
            ("Performance Comparison", "compare_performance.py"),
        ]

        for name, script in scripts:
            print(f"Running {name}...")
            try:
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
                if result.returncode == 0:
                    print(result.stdout)
                else:
                    print(f"  Warning: {script} exited with code {result.returncode}")
            except FileNotFoundError:
                print(f"  Skipping (script not found): {script}")
            except subprocess.TimeoutExpired:
                print(f"  Timeout: {script} took too long")
            except Exception as e:
                print(f"  Error running {script}: {e}")

    except Exception as e:
        print(f"Error running benchmarks: {e}")

    print("\n" + "=" * 70)
    print("Benchmarking complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RTX-STone diagnostic tools")
    parser.add_argument(
        "command",
        nargs="?",
        default="verify",
        choices=["verify", "info", "benchmark"],
        help="Command to run (default: verify)",
    )

    args = parser.parse_args()

    if args.command == "verify":
        verify_installation()
    elif args.command == "info":
        show_info()
    elif args.command == "benchmark":
        run_benchmarks()
