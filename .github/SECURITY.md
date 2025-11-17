# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 2.10.0a0+ (latest) | :white_check_mark: |
| < 2.10.0a0 | :x: |

## Reporting a Vulnerability

The PyTorch RTX 5080 project takes security seriously. We appreciate your efforts to responsibly disclose your findings.

### Where to Report

**Please DO NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via:

1. **Email**: Send an email to the repository maintainer at the email listed in the GitHub profile
2. **Subject Line**: Use `[SECURITY] Brief description of the issue`

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, code injection, privilege escalation)
- Full paths of source file(s) related to the issue
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

### What to Expect

- **Initial Response**: Within 48 hours, you'll receive an acknowledgment of your report
- **Investigation**: We'll investigate and validate the issue
- **Updates**: We'll keep you informed of our progress
- **Resolution**: Once resolved, we'll notify you and publicly disclose (with credit)

### Timeline

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Development**: Varies based on complexity
- **Public Disclosure**: After fix is released (coordinated with you)

## Security Best Practices

### For Users

When using PyTorch RTX 5080:

1. **Download from Official Sources**
   - Only download releases from the official GitHub repository
   - Verify file checksums (when provided)
   - Use official installation methods

2. **Keep Software Updated**
   - Use the latest version of the package
   - Update NVIDIA drivers regularly
   - Keep Python and dependencies up to date

3. **Secure Your Environment**
   - Use virtual environments
   - Don't run untrusted code
   - Be cautious with user-supplied input to models

4. **CUDA/Triton Security**
   - Be aware that CUDA kernels run with GPU privileges
   - Review custom Triton kernels before execution
   - Understand that malicious kernels could potentially:
     - Crash the GPU
     - Leak GPU memory
     - Cause system instability

### For Contributors

When contributing code:

1. **Input Validation**
   - Validate all user inputs
   - Check tensor dimensions and types
   - Handle edge cases safely

2. **Memory Safety**
   - Use proper bounds checking in Triton kernels
   - Avoid buffer overflows
   - Clean up resources properly

3. **Triton Kernel Safety**
   - Use masks for boundary conditions
   - Validate pointer offsets
   - Test with various input sizes

4. **Dependencies**
   - Use pinned versions in requirements.txt
   - Audit dependency security regularly
   - Avoid dependencies with known vulnerabilities

## Known Security Considerations

### CUDA/GPU Access

This package requires GPU access with the following implications:

- **Driver-level access**: Requires NVIDIA drivers with system privileges
- **GPU memory**: Kernels can access all GPU memory
- **System stability**: Malicious or buggy kernels can crash the GPU/system
- **Shared resources**: GPU is shared across processes

### Triton Kernels

Custom Triton kernels:

- Execute directly on the GPU
- Have access to allocated GPU memory
- Could potentially crash the GPU if malformed
- Should be reviewed before execution in production

**Recommendation**: Only run Triton kernels from trusted sources.

### Model Weights and Checkpoints

When loading model weights:

- PyTorch's `torch.load()` uses pickle, which can execute arbitrary code
- Only load weights from trusted sources
- Consider using `weights_only=True` parameter when available
- Use `safetensors` format when possible (more secure than pickle)

### Network Access

Some features may require network access:

- Downloading models from HuggingFace Hub
- Auto-update checks (if implemented)
- Performance telemetry (opt-in only)

**Privacy Note**: This package does not collect or transmit user data by default.

## Vulnerability Disclosure Policy

### Public Disclosure

- Security issues will be disclosed publicly after a fix is released
- Credit will be given to the reporter (unless they request anonymity)
- Details will be published in:
  - GitHub Security Advisories
  - CHANGELOG.md
  - Release notes

### Coordinated Disclosure

We follow a coordinated disclosure policy:

1. Security issue is reported privately
2. We investigate and develop a fix
3. We coordinate with the reporter on disclosure timeline
4. Fix is released
5. Security advisory is published

## Security Updates

Security updates will be:

- Released as soon as possible after validation
- Clearly marked in release notes
- Announced via GitHub Security Advisories
- Backported to supported versions if necessary

## Scope

### In Scope

- Security vulnerabilities in this codebase
- Dependency vulnerabilities affecting this project
- Configuration issues leading to security risks
- Issues with installation scripts

### Out of Scope

- Vulnerabilities in PyTorch core (report to PyTorch project)
- Vulnerabilities in Triton (report to Triton project)
- NVIDIA driver vulnerabilities (report to NVIDIA)
- Operating system vulnerabilities
- Issues requiring physical access to the machine
- Social engineering attacks

## Contact

For security-related questions or concerns:

- Check this SECURITY.md file
- Review open Security Advisories
- Contact maintainers via email (for sensitive issues)
- Use GitHub Discussions (for general security questions)

---

**Thank you for helping keep PyTorch RTX 5080 Support secure!**
