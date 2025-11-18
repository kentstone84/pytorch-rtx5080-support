---
name: Driver Patch Verification
about: Report your driver patch findings and performance results
title: '[DRIVER-PATCH] Driver XXX.XX on [Platform] - [GPU Model]'
labels: driver-patch, verification
assignees: ''
---

## Driver Information

**Driver Version:** (e.g., 570.00, 571.23)
**Platform:** (Linux / Windows)
**OS Version:** (e.g., Ubuntu 24.04, Windows 11 23H2)

## GPU Information

**GPU Model:** (e.g., RTX 5080, RTX 5090)
**VRAM:** (e.g., 16GB, 24GB)

## Patch Details

### Did you successfully find the 3 sm_120 rejection functions?

- [ ] Yes
- [ ] No (if no, describe what you found)

### Hex Patch Addresses

**Function 1:**
- Address: `0x[ADDRESS]`
- Original byte: `0x[VALUE]`
- Patched byte: `0x[VALUE]`

**Function 2:**
- Address: `0x[ADDRESS]`
- Original byte: `0x[VALUE]`
- Patched byte: `0x[VALUE]`

**Function 3:**
- Address: `0x[ADDRESS]`
- Original byte: `0x[VALUE]`
- Patched byte: `0x[VALUE]`

## Performance Results

### Before Patch (Stock Driver)

**PerformanceTest Score:** (if applicable)

**PyTorch Benchmark:**
```
# Paste output from: python compare_performance.py --baseline
```

**Observed Issues:**
- [ ] "No kernel image available" errors
- [ ] Lower than expected TFLOPS
- [ ] Other (describe):

### After Patch

**PerformanceTest Score:** (if applicable)

**PyTorch Benchmark:**
```
# Paste output from: python compare_performance.py --patched
```

**Performance Improvement:** (e.g., +40%, +35%)

**Any Issues:**
- [ ] No issues, working perfectly
- [ ] System instability
- [ ] Driver crashes
- [ ] Other (describe):

## Verification

### Can others reproduce your findings?

- [ ] Yes, I've documented exact steps
- [ ] Partially, some details unclear
- [ ] Need help documenting

### PyTorch Info

```python
# Paste output of:
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Capability: {torch.cuda.get_device_capability(0)}")
print(f"Arch List: {torch.cuda.get_arch_list()}")
```

## Screenshots (Optional)

If you have PerformanceTest results, Ghidra screenshots, or benchmark graphs, attach them here.

## Additional Notes

Any other observations, tips, or insights from your patching process:

---

**By submitting this issue, I confirm:**
- [ ] I performed this patch for personal research/educational purposes
- [ ] I understand this may void my GPU warranty
- [ ] I'm sharing findings to help the community
- [ ] All information provided is accurate to the best of my knowledge
