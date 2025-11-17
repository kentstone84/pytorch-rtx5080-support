#!/usr/bin/env python
"""
RTX-STone: PyTorch with native SM 12.0 support for RTX 50-series GPUs
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install

# Ensure we're on Windows
if sys.platform != "win32":
    print("ERROR: RTX-STone is currently only supported on Windows.")
    print("For Linux support, please use WSL2 or wait for native Linux builds.")
    sys.exit(1)

# Ensure Python version is 3.10 or 3.11
if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    print("ERROR: RTX-STone requires Python 3.10 or 3.11")
    print(f"Current Python version: {sys.version}")
    sys.exit(1)


class PostInstallCommand(install):
    """Post-installation customization"""

    def run(self):
        install.run(self)
        print("\n" + "=" * 70)
        print("RTX-STone Installation")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Run verification: python -m rtx_stone.diagnostic")
        print("2. Check GPU support: rtx-stone-info")
        print("3. Run benchmarks: rtx-stone-benchmark")
        print("4. Read documentation: https://github.com/kentstone84/pytorch-rtx5080-support")
        print("\nSupported GPUs:")
        print("  - RTX 5090 (24GB)")
        print("  - RTX 5080 (16GB)")
        print("  - RTX 5070 Ti (16GB)")
        print("  - RTX 5070 (12GB)")
        print("  - All future RTX 50-series GPUs with SM 12.0")
        print("\nFor issues or questions:")
        print("  https://github.com/kentstone84/pytorch-rtx5080-support/issues")
        print("=" * 70 + "\n")


# Read the long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read version from pyproject.toml to keep in sync
# In practice, version should be defined in one place
VERSION = "2.10.0a0"

setup(
    cmdclass={
        "install": PostInstallCommand,
    },
    long_description=long_description,
    long_description_content_type="text/markdown",
)
