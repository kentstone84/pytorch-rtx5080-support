# Community Verification Checklist

## For Community Members Testing This Repository

This repository makes significant claims about NVIDIA driver gatekeeping. Here's how to independently verify or refute these claims.

## What You Can Verify Right Now

### ✅ Level 1: Ghidra Analysis (No patching required)

**Requirements:**
- NVIDIA driver 570.00+ with RTX 50-series support
- Ghidra (free, open-source)
- Basic reverse engineering knowledge

**Steps:**

1. **Locate the driver library:**
   ```bash
   # Linux
   find /usr -name "libcuda.so.570.*"

   # Should find: /usr/lib/x86_64-linux-gnu/libcuda.so.570.XX
   ```

2. **Open in Ghidra:**
   - Import libcuda.so.570.XX
   - Let Ghidra auto-analyze
   - Wait for analysis to complete (~10-15 minutes)

3. **Verify scalar 0x2c (rejection code):**
   - Search → For Scalars
   - Enter: `2c` (hex) or `44` (decimal)
   - **Expected: ~2,616 instances found**
   - Does this match? ⬜ YES ⬜ NO

4. **Verify "sm_120" string references:**
   - Search → For Strings
   - Enter: `sm_120`
   - **Expected: 10 references found**
   - Does this match? ⬜ YES ⬜ NO
   - List addresses found:
     ```
     ⬜ 00fbb9ae
     ⬜ 0108b027
     ⬜ 01090fe0
     ⬜ [record others]
     ```

5. **Verify FUN_005e0020 (sm_120 handler):**
   - Navigate to address: `005e0020`
   - Check cross-references (XREFs) to sm_120 strings
   - **Expected: 7 XREFs to "sm_120"**
   - Count found: _____ XREFs
   - Does this match? ⬜ YES ⬜ NO

6. **Verify FUN_00f682d0 (primary orchestrator):**
   - Navigate to address: `00f682d0`
   - Examine decompiled code
   - Look for calls to subfunctions
   - **Expected: Calls multiple validation functions**
   - Can you identify the validation flow? ⬜ YES ⬜ NO

**Conclusion from Level 1:**
- ⬜ Confirmed: Ghidra findings match documentation
- ⬜ Partial: Some findings match, others don't
- ⬜ Refuted: Findings significantly differ
- ⬜ Unable to verify: [explain why]

---

### ✅ Level 2: Patch Methodology Validation (Requires hex editing skills)

**Requirements:**
- All Level 1 requirements
- Hex editor (xxd, hexedit, or similar)
- Understanding of x86-64 assembly
- Backup of original driver

**Steps:**

1. **Create backup:**
   ```bash
   sudo cp /usr/lib/x86_64-linux-gnu/libcuda.so.570.XX \
           /usr/lib/x86_64-linux-gnu/libcuda.so.570.XX.backup
   ```

2. **Document original findings:**
   - In Ghidra, find the 3 patch locations
   - Record exact file offsets
   - Record original byte values
   - Document expected byte changes

3. **Apply patches:**
   - Use hex editor to modify the 3 bytes
   - Verify changes with: `cmp original.so patched.so`

4. **Safety check:**
   - Only patch if you understand what you're changing
   - Have a recovery plan (boot to previous kernel if needed)
   - Can restore original driver quickly

**Conclusion from Level 2:**
- ⬜ Successfully identified patch locations
- ⬜ Patches apply without errors
- ⬜ Unable to find exact locations: [explain]
- ⬜ Assembly doesn't match description: [explain]

---

### ✅ Level 3: Performance Verification (Full reproduction)

**Requirements:**
- All Level 2 requirements
- RTX 5080 or RTX 5090 GPU
- PerformanceTest 11.1 (or similar benchmark)
- PyTorch with CUDA 12.8+ support

**Steps:**

1. **Baseline measurement (before patch):**
   ```bash
   # Run PerformanceTest 11.1
   # Record: 3D Graphics Mark score
   # Record: GPU Compute score
   # Record: Overall G3D Mark
   ```
   - 3D Graphics Mark: ______
   - GPU Compute: ______
   - G3D Mark: ______

2. **Apply patches:**
   - Use findings from Level 2
   - Or use automated patcher (if patch_config.json has real addresses)

3. **Reboot and verify driver loaded:**
   ```bash
   nvidia-smi
   # Verify driver version matches
   ```

4. **Post-patch measurement:**
   ```bash
   # Run PerformanceTest 11.1 again
   # Record same metrics
   ```
   - 3D Graphics Mark: ______
   - GPU Compute: ______
   - G3D Mark: ______

5. **Calculate improvement:**
   - 3D Graphics: ____% improvement
   - **Expected: ~60% improvement (29,604 → 47,616)**
   - Does your result match? ⬜ YES ⬜ NO

6. **Verify sm_120 kernel execution:**
   ```python
   import torch
   print(torch.cuda.get_device_capability())
   # Expected: (12, 0) for sm_120
   # Before patch might show: (8, 9) for sm_89 fallback
   ```
   - Capability detected: ______
   - Using sm_120 kernels? ⬜ YES ⬜ NO

**Conclusion from Level 3:**
- ⬜ Fully reproduced: ~60% performance gain confirmed
- ⬜ Partial reproduction: ____% gain (less than expected)
- ⬜ No improvement: Patches didn't change performance
- ⬜ Regression: Performance got worse
- ⬜ Unable to test: [explain why]

---

## Evidence Evaluation

### Author's Evidence (When Available)

Once the author uploads evidence to `docs/evidence/`, verify:

**Screenshots:**
- ⬜ Before/after benchmark scores are clear and readable
- ⬜ GPU model and driver version visible
- ⬜ Scores match claimed values (29,604 → 47,616)
- ⬜ Ghidra screenshots match documented findings
- ⬜ System info confirms RTX 5080 hardware
- ⬜ EXIF data shows consistent timestamps

**ChatGPT History:**
- ⬜ Shows genuine discovery process (not scripted)
- ⬜ Timestamps show investigation took time
- ⬜ Contains dead ends and corrections (authentic)
- ⬜ Technical details emerge progressively
- ⬜ Timeline is consistent and logical

**Reproducibility:**
- ⬜ Author claims 4 driver versions patched successfully
- ⬜ Evidence exists for multiple driver versions
- ⬜ Consistency across different patches

---

## Red Flags to Watch For

### Evidence Against Claims:

- ❌ Ghidra findings don't match (different number of sm_120 references)
- ❌ Functions at documented addresses don't exist
- ❌ Assembly code doesn't match described validation logic
- ❌ Performance gains are inconsistent or irreproducible
- ❌ Screenshots show different GPU model or driver
- ❌ ChatGPT history looks fabricated or too perfect
- ❌ Community members can't reproduce Level 1 findings

### Evidence Supporting Claims:

- ✅ Multiple people independently verify Ghidra findings
- ✅ Performance improvements are reproducible
- ✅ PyTorch actually uses sm_120 kernels after patch
- ✅ Evidence timeline is consistent and authentic
- ✅ Technical details hold up under scrutiny
- ✅ Author responsive to questions and provides data

---

## Reporting Your Findings

**If you verify the claims:**
```markdown
## Verification Report

**Level reached:** [1/2/3]
**Date:** YYYY-MM-DD
**Tester:** [your GitHub username]

**Results:**
- Ghidra analysis: ✅ Matches documentation
- Patch locations: ✅ Found all 3 addresses
- Performance: ✅ ~60% improvement confirmed (before: XXXXX, after: XXXXX)

**Hardware:**
- GPU: RTX 5080 / RTX 5090
- Driver: 570.XX
- OS: [your Linux distro]

**Conclusion:** Claims verified. Successfully reproduced results.
```

**If you refute the claims:**
```markdown
## Verification Report

**Level reached:** [1/2/3]
**Date:** YYYY-MM-DD
**Tester:** [your GitHub username]

**Results:**
- Ghidra analysis: ❌ Found only X instances of 0x2c (expected 2,616)
- String search: ❌ Found Y references to sm_120 (expected 10)
- Performance: ❌ No improvement (before: XXXXX, after: XXXXX)

**Hardware:**
- GPU: RTX 5080 / RTX 5090
- Driver: 570.XX
- OS: [your Linux distro]

**Conclusion:** Unable to reproduce. [Explain discrepancies]
```

**If you partially verify:**
```markdown
## Verification Report

**Level reached:** [1/2/3]
**Date:** YYYY-MM-DD
**Tester:** [your GitHub username]

**Results:**
- Ghidra analysis: ✅ Matches (but with minor differences)
- Patch locations: ⚠️ Found 2 of 3 addresses
- Performance: ⚠️ ~20% improvement (less than claimed 60%)

**Hardware:**
- GPU: RTX 5080 / RTX 5090
- Driver: 571.XX (different version than documented)
- OS: [your Linux distro]

**Conclusion:** Partial verification. [Explain what worked and what didn't]
```

---

## For the Author

This checklist helps you understand what the community will test. Make sure:

- ✅ Your documentation is detailed enough for independent verification
- ✅ You provide exact addresses and byte values (when ready)
- ✅ You upload screenshots and chat history to `docs/evidence/`
- ✅ You respond to community questions and verification attempts
- ✅ You update documentation if community finds errors

## Questions or Issues?

- Open a GitHub issue for specific technical questions
- Include your verification level (1/2/3) and results
- Provide screenshots or logs if results differ
- Be respectful - we're all trying to find the truth

---

*Science works through independent verification. Thank you for taking the time to verify these claims rigorously.*
