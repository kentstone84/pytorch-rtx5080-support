# Contributing to PyTorch RTX 5080 Support

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Performance Testing](#performance-testing)

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow:

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on constructive feedback
- Accept gracefully when others disagree
- Show empathy towards other community members

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
1. Check the [existing issues](https://github.com/kentstone84/pytorch-rtx5080-support/issues)
2. Gather system information (GPU model, driver version, CUDA version, etc.)
3. Create a minimal reproducible example

Use the bug report template when filing an issue.

### Suggesting Features

We welcome feature suggestions! Please:
1. Check if the feature has already been requested
2. Clearly describe the use case and benefits
3. Provide examples of how it would work
4. Consider if you can implement it yourself

### Contributing Code

We welcome contributions in these areas:

**High Priority:**
- New Triton kernels for common operations
- Performance optimizations for existing kernels
- Integration examples (vLLM, LangChain, ComfyUI, etc.)
- Documentation improvements
- Jupyter notebook tutorials

**Good First Issues:**
- Look for issues labeled `good-first-issue`
- Documentation typos and clarifications
- Adding tests for existing functionality
- Example scripts and tutorials

## Development Setup

### Prerequisites

- Windows 11
- RTX 5080 or RTX 5090
- Python 3.10 or 3.11
- NVIDIA Driver 570.00+
- Git

### Setting Up Your Development Environment

```powershell
# Clone the repository
git clone https://github.com/kentstone84/pytorch-rtx5080-support.git
cd pytorch-rtx5080-support

# Create virtual environment
python -m venv dev-env
.\dev-env\Scripts\Activate.ps1

# Install PyTorch RTX 5080
.\install.ps1

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 jupyter

# Install pre-commit hooks (if using)
pip install pre-commit
pre-commit install
```

### Running Tests

```powershell
# Run basic verification
python examples/getting_started.py

# Run benchmarks
python compare_performance.py

# Run Triton tests
python benchmark_triton.py

# Run pytest (when available)
pytest tests/
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://pep8.org/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use type hints where appropriate
- Add docstrings for all public functions and classes

### Code Formatting

We use `black` for code formatting:

```powershell
# Format your code
black *.py examples/*.py

# Check formatting
black --check *.py
```

### Linting

Use `flake8` for linting:

```powershell
flake8 *.py examples/*.py
```

### Triton Kernel Guidelines

When writing Triton kernels:

1. **Performance First**: Benchmark against PyTorch equivalent
2. **Documentation**: Explain the algorithm and optimizations
3. **Configurability**: Use `tl.constexpr` for block sizes
4. **Testing**: Include correctness tests against PyTorch
5. **Benchmarking**: Provide performance comparisons

Example structure:

```python
@triton.jit
def my_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Brief description of what the kernel does.

    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        n_elements: Number of elements to process
        BLOCK_SIZE: Block size for processing (must be power of 2)
    """
    # Implementation
    pass


def my_operation(x: torch.Tensor) -> torch.Tensor:
    """
    High-level function that launches the kernel.

    Args:
        x: Input tensor

    Returns:
        Output tensor

    Performance:
        - Speedup over PyTorch: 1.5x on RTX 5080
        - Memory usage: Same as PyTorch
    """
    # Launch kernel
    pass


# Benchmark
def benchmark_my_operation():
    """Benchmark against PyTorch baseline."""
    pass
```

### Documentation

- Use NumPy-style docstrings
- Include usage examples in docstrings
- Document performance characteristics
- Add inline comments for complex logic

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `perf`: Performance improvement
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**

```
feat(triton): add fused LayerNorm kernel

Implement fused LayerNorm with configurable epsilon.
Achieves 1.3x speedup over PyTorch on RTX 5080.

Closes #123
```

```
fix(flash-attention): correct mask handling for causal attention

Fixed issue where causal mask was not properly applied in
flash_attention_rtx5080.py, causing incorrect outputs.

Fixes #456
```

### Commit Best Practices

- Keep commits atomic (one logical change per commit)
- Write clear, descriptive commit messages
- Reference related issues in commit messages
- Ensure each commit builds and passes tests

## Pull Request Process

### Before Submitting

1. **Test Your Changes**
   - Run all existing tests
   - Add new tests for new functionality
   - Run benchmarks if performance-related

2. **Update Documentation**
   - Update README if needed
   - Add docstrings to new functions
   - Update relevant examples

3. **Code Quality**
   - Format with `black`
   - Lint with `flake8`
   - Fix any warnings

### Submitting a PR

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Commit your changes (`git commit -m 'feat: add amazing feature'`)
5. Push to your fork (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### PR Review Process

1. **Automated Checks**: CI/CD will run tests
2. **Code Review**: Maintainers will review your code
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, maintainers will merge

### PR Best Practices

- Keep PRs focused (one feature/fix per PR)
- Provide clear description of changes
- Include benchmark results for performance changes
- Respond to review feedback promptly
- Keep your PR up to date with main branch

## Performance Testing

For performance-related contributions, please include:

### Benchmark Script

```python
import torch
import time

def benchmark_operation(iterations=100):
    # Your benchmark code
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        # Operation to benchmark
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    return {
        'mean': sum(times) / len(times),
        'min': min(times),
        'max': max(times),
    }
```

### Performance Report Template

```
## Performance Results

**Test Configuration:**
- GPU: RTX 5080 16GB
- Driver: 570.00
- CUDA: 13.0
- Batch Size: 16
- Sequence Length: 2048

**Results:**
- Baseline (PyTorch): 150ms
- This PR: 100ms
- Speedup: 1.5x
- Memory Usage: Same as baseline

**Methodology:**
- 100 iterations, warm-up: 10 iterations
- CUDA events for timing
- Results averaged over all iterations
```

## Getting Help

- **Questions**: Open a [Discussion](https://github.com/kentstone84/pytorch-rtx5080-support/discussions)
- **Bugs**: File an [Issue](https://github.com/kentstone84/pytorch-rtx5080-support/issues)
- **Real-time**: Check if there's a Discord/Slack (TBD)

## Recognition

Contributors will be recognized in:
- CHANGELOG.md for their contributions
- README.md acknowledgments section
- Release notes when applicable

## License

By contributing, you agree that your contributions will be licensed under the same license as the project (BSD-3-Clause, following PyTorch).

---

Thank you for contributing to PyTorch RTX 5080 Support! Your efforts help make this the best PyTorch distribution for RTX 50-series GPUs.
