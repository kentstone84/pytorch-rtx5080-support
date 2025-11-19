# Screenshot Upload Guide

## Required Screenshots Per Driver Version

For each driver version you've patched, provide these screenshots:

### 1. Before Patch - PerformanceTest Results
**Filename:** `before_patch_XXXXX.png` (where XXXXX is the actual score)
- PerformanceTest 11.1 results window
- **Must show:**
  - GPU model (RTX 5080)
  - Driver version
  - 3D Graphics Mark score (~29,604 for 570.00)
  - Timestamp/date if visible

### 2. After Patch - PerformanceTest Results
**Filename:** `after_patch_XXXXX.png` (where XXXXX is the actual score)
- PerformanceTest 11.1 results window
- **Must show:**
  - Same GPU model (RTX 5080)
  - Same driver version (but patched)
  - 3D Graphics Mark score (~47,616 for 570.00)
  - Timestamp/date if visible

### 3. Ghidra - Scalar Search (0x2c)
**Filename:** `ghidra_scalar_search_0x2c.png`
- Ghidra search results window
- **Must show:**
  - Search query: "0x2c" or "44" (decimal)
  - Number of results found (2,616 instances)
  - libcuda.so.XXX.XX file being analyzed

### 4. Ghidra - String Search (sm_120)
**Filename:** `ghidra_string_search_sm120.png`
- Ghidra string search results
- **Must show:**
  - Search query: "sm_120"
  - All references found (10 instances)
  - Addresses where strings appear

### 5. Ghidra - Function Cross-References
**Filename:** `ghidra_function_005e0020_xrefs.png`
- FUN_005e0020 with XREFs visible
- **Must show:**
  - Function address: 005e0020
  - 7 cross-references to sm_120 strings
  - Function decompilation if helpful

### 6. System Information
**Filename:** `system_info.png`
- nvidia-smi output OR system info dialog
- **Must show:**
  - GPU: RTX 5080
  - Driver version
  - CUDA version
  - Linux distribution/kernel version

## Optional But Helpful

### 7. Hex Editor - Before/After Comparison
**Filename:** `hex_comparison_patch_N.png` (N = patch number 1-3)
- Hex editor showing the specific byte change
- **Must show:**
  - File offset
  - Original byte value
  - New byte value
  - Surrounding context

### 8. Terminal Output - Patch Verification
**Filename:** `patch_verification.png`
- Running the automated patcher
- Shows successful backup and patch application

## Directory Structure

```
docs/evidence/screenshots/
├── driver_570.00/
│   ├── before_patch_29604.png
│   ├── after_patch_47616.png
│   ├── ghidra_scalar_search_0x2c.png
│   ├── ghidra_string_search_sm120.png
│   ├── ghidra_function_005e0020_xrefs.png
│   ├── system_info.png
│   └── [optional hex comparison screenshots]
├── driver_571.xx/
│   └── [same structure]
├── driver_572.xx/
│   └── [same structure]
└── driver_573.xx/
    └── [same structure]
```

## Upload Commands

```bash
# From your Pictures or Screenshots folder:

# Driver 570.00 evidence
cp ~/Pictures/performancetest_before.png \
   docs/evidence/screenshots/driver_570.00/before_patch_29604.png

cp ~/Pictures/performancetest_after.png \
   docs/evidence/screenshots/driver_570.00/after_patch_47616.png

cp ~/Pictures/ghidra_*.png \
   docs/evidence/screenshots/driver_570.00/

cp ~/Pictures/nvidia_smi.png \
   docs/evidence/screenshots/driver_570.00/system_info.png
```

## Verification Checklist

Before committing screenshots, verify:
- [ ] All filenames follow the naming convention
- [ ] Screenshots are clear and readable (no blur)
- [ ] All required information is visible
- [ ] Timestamps are consistent within each driver version
- [ ] GPU model and driver versions match across screenshots
- [ ] EXIF data preserved (don't strip metadata)

## Privacy Notes

**Safe to share:**
- GPU model, driver version, benchmark scores
- Ghidra analysis screenshots
- System info (GPU/driver/CUDA version)

**Consider redacting:**
- Personal username (if visible in paths)
- IP addresses
- Serial numbers
- Full system hostname

---

*Once uploaded, commit with: `git add docs/evidence/screenshots/ && git commit -m "Add screenshot evidence for driver X.XX"`*
