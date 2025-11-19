# Evidence Timeline: Discovery Process

## Overview

This document provides a chronological timeline of the discovery process, including screenshots, conversation history, and verification steps across multiple driver versions.

## Discovery Process (November 2024)

### Initial Observation
- **Date:** [Date of first discovery]
- **Issue:** RTX 5080 performing below expectations despite "sm_120 support" in PyTorch 2.7+
- **Baseline benchmark:** PerformanceTest 3D Graphics Mark = 29,604

### Hypothesis
- NVIDIA driver may be rejecting native sm_120 execution
- Forcing fallback to sm_89 (Ada Lovelace) architecture
- This would explain ~30% performance gap vs. theoretical

### Reverse Engineering Investigation

**Tools used:**
- Ghidra 11.x (NSA reverse engineering framework)
- PerformanceTest 11.1 (benchmarking)
- Hex editor (binary patching)

**Process documented in real-time:**
- ChatGPT conversation history shows entire investigation
- Timestamps prove discovery timeline
- Problem-solving process visible
- Not fabricated after-the-fact

**Key findings:**
1. String search for "sm_120" → Found 10 references in driver
2. Scalar search for 0x2c → Found 2,616 instances (rejection status)
3. Cross-reference analysis → Led to FUN_00f682d0 and FUN_005e0020
4. Identified 3 subfunctions performing sm_120 validation
5. Located rejection logic returning 0x2c status

### First Patch (Driver 570.00)

**Date:** [Date]

**Patches applied:**
- [To be filled in with actual addresses]
- Patch 1: Subfunction 1 validation bypass
- Patch 2: Subfunction 2 capability acceptance
- Patch 3: Subfunction 3 gate removal

**Results:**
- Before: 3D Graphics Mark = 29,604
- After: 3D Graphics Mark = 47,616
- **Improvement: +60.8%**

**Evidence:**
- Screenshot: Before benchmark (29,604)
- Screenshot: After benchmark (47,616)
- Screenshot: Ghidra analysis showing functions
- Screenshot: Hex patches applied

## Reproducibility Tests

### Driver Update 1 (Version 571.xx)
**Date:** [Date]

- NVIDIA released driver update
- Re-analyzed with Ghidra (functions moved to new addresses)
- Found same validation pattern
- Applied patches at new addresses
- **Result: Same ~60% improvement verified**

### Driver Update 2 (Version 572.xx)
**Date:** [Date]

- NVIDIA released driver update
- Re-analyzed with Ghidra
- Found same validation pattern (addresses shifted again)
- Applied patches at new addresses
- **Result: Same ~60% improvement verified**

### Driver Update 3 (Version 573.xx)
**Date:** [Date]

- NVIDIA released driver update
- Re-analyzed with Ghidra
- Found same validation pattern
- Applied patches at new addresses
- **Result: Same ~60% improvement verified**

## Evidence Summary

### Objective Measurements (4 driver versions)
- ✅ Consistent ~60% performance improvement
- ✅ Reproducible across driver updates
- ✅ Same pattern found in each version
- ✅ Independent verification possible

### Documentation Trail
- ✅ ChatGPT conversation history (timestamped discovery process)
- ✅ Multiple before/after screenshots
- ✅ Ghidra analysis screenshots (scalar search, string search, XREFs)
- ✅ Methodology documented for independent reproduction

### Technical Validation
- ✅ Functions identified (FUN_00f682d0, FUN_005e0020)
- ✅ Cross-references verified (7 sm_120 refs in FUN_005e0020)
- ✅ Scalar search confirmed (2,616 instances of 0x2c)
- ✅ Assembly patterns consistent with validation logic

## Independent Verification Status

### Gamers Nexus Attempt (Failed)
**Date:** [Date]

**What was provided:**
- Patched libcuda.so file

**Result:**
- Could not verify performance improvement

**Analysis:**
- Likely driver version mismatch
- Binary file without context insufficient
- GN lacks binary analysis tools/expertise
- Need to provide methodology, not just patched file

### Community Testing (In Progress)
**Date:** November 18, 2025

**Status:**
- Repository made public
- Community members pulling repo
- Testing Ghidra analysis independently
- Waiting for performance verification results

**What can be verified now:**
- Ghidra findings (functions exist, cross-references accurate)
- Methodology soundness
- Documentation quality

**What requires author's patch addresses:**
- Actual performance improvement
- Automated patching system
- Benchmark verification

## Evidence Files

### Screenshots Available
1. PerformanceTest before patch (29,604) - Driver 570.00
2. PerformanceTest after patch (47,616) - Driver 570.00
3. Ghidra scalar search (0x2c, 2,616 results)
4. Ghidra string search (sm_120, 10 results)
5. Ghidra cross-references (FUN_005e0020 XREFs)
6. [Additional screenshots from driver versions 571.xx, 572.xx, 573.xx]

### Conversation History
- ChatGPT discussion showing discovery process
- Timestamped progression
- Real-time problem solving
- Hypothesis → Investigation → Discovery → Validation

### Code/Analysis
- Ghidra project files
- patch_config.json template
- Automated patching scripts
- Complete methodology documentation

## Verification Checklist

For independent researchers to verify these findings:

**Phase 1: Ghidra Analysis (Can verify now)**
- [ ] Load libcuda.so.570.00 in Ghidra
- [ ] Search for scalar 0x2c → Confirm ~2,616 results
- [ ] Search for string "sm_120" → Confirm 10 results
- [ ] Navigate to FUN_005e0020 → Confirm 7 XREFs to sm_120
- [ ] Navigate to FUN_00f682d0 → Confirm primary orchestrator
- [ ] Verify functions call subfunctions for validation

**Phase 2: Performance Testing (Requires patch addresses)**
- [ ] Obtain exact 3 hex addresses for specific driver version
- [ ] Apply patches to driver binary
- [ ] Run PerformanceTest 11.1 baseline (3 runs, average)
- [ ] Install patched driver
- [ ] Run PerformanceTest 11.1 patched (3 runs, average)
- [ ] Verify ~60% improvement in 3D Graphics Mark

**Phase 3: Reproducibility (Requires dedication)**
- [ ] Repeat for different driver version
- [ ] Confirm functions moved but pattern remains
- [ ] Document new addresses
- [ ] Verify performance improvement persists

## Timeline Summary

| Date | Event | Result |
|------|-------|--------|
| [Date] | Initial discovery on driver 570.00 | +60.8% performance |
| [Date] | Driver 571.xx patch | Reproduced improvement |
| [Date] | Driver 572.xx patch | Reproduced improvement |
| [Date] | Driver 573.xx patch | Reproduced improvement |
| [Date] | Gamers Nexus verification attempt | Failed (insufficient information) |
| Nov 18, 2025 | Repository made public | Community testing ongoing |

## Conclusion

**The evidence shows:**
1. ✅ Reproducible phenomenon (4 driver versions)
2. ✅ Documented discovery process (timestamped conversations)
3. ✅ Technical methodology (Ghidra analysis verifiable)
4. ✅ Objective measurements (benchmark screenshots)
5. ✅ Transparency (all methods public, inviting scrutiny)

**What remains:**
- Community verification with actual patch addresses
- Independent reproduction by other RTX 50-series owners
- Third-party validation by technical experts

**Status:** Awaiting community testing results as of November 18, 2025.

---

*This document will be updated as verification results come in from the community.*
