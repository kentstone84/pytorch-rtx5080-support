# Evidence Directory

## What's Here

This directory contains objective evidence supporting the driver gatekeeping claims:

### Screenshots
- Before/after PerformanceTest results
- Ghidra analysis screenshots
- Multiple driver version results
- System information

### Chat History
- ChatGPT conversation logs showing discovery process
- Timestamped investigation timeline
- Real-time problem solving
- Proves findings weren't fabricated after-the-fact

## Why This Matters

### Screenshots Prove:
- ✅ Objective performance measurements
- ✅ Consistent results across tests
- ✅ Technical analysis (Ghidra)
- ✅ System configuration

### Chat History Proves:
- ✅ Discovery process was genuine
- ✅ Timeline of investigation
- ✅ Hypothesis → Testing → Discovery → Validation
- ✅ Not reverse-engineered conclusion (working backwards from fake claim)

## Organization

```
evidence/
├── screenshots/
│   ├── driver_570.00/
│   │   ├── before_patch_29604.png
│   │   ├── after_patch_47616.png
│   │   ├── ghidra_scalar_search_0x2c.png
│   │   ├── ghidra_string_search_sm120.png
│   │   └── system_info.png
│   ├── driver_571.xx/
│   │   └── [similar structure]
│   ├── driver_572.xx/
│   │   └── [similar structure]
│   └── driver_573.xx/
│       └── [similar structure]
└── chat_history/
    ├── discovery_conversation.md (or .txt)
    └── timestamps.txt
```

## How to Verify Evidence

### For Screenshots:
1. Check EXIF data for timestamps (if present)
2. Verify PerformanceTest version shown (11.1)
3. Confirm GPU model (RTX 5080)
4. Check score consistency across multiple tests
5. Verify Ghidra screenshots match documented findings

### For Chat History:
1. Verify timestamps show progression over time
2. Check for consistent narrative (hypothesis → discovery)
3. Look for genuine problem-solving (not scripted)
4. Confirm technical details match final documentation

## Adding Your Evidence

**To add your screenshots here:**

```bash
# Copy screenshots to appropriate driver version folder
cp ~/Pictures/performancetest_before.png docs/evidence/screenshots/driver_570.00/before_patch_29604.png
cp ~/Pictures/performancetest_after.png docs/evidence/screenshots/driver_570.00/after_patch_47616.png

# Copy Ghidra screenshots
cp ~/Pictures/ghidra_*.png docs/evidence/screenshots/driver_570.00/

# Export ChatGPT conversation
# (Copy from ChatGPT UI → Save as text/markdown)
```

## For Community Reviewers

**When reviewing this evidence, ask:**

1. **Consistency:** Do the screenshots show the same GPU/system?
2. **Reproducibility:** Are results consistent across driver versions?
3. **Timeline:** Does the chat history show logical progression?
4. **Technical depth:** Do Ghidra screenshots match documented analysis?
5. **Authenticity:** Any signs of manipulation or fabrication?

## Evidence Status

**Current status:** Waiting for author to upload screenshots and chat history

**What's verifiable now:**
- Ghidra methodology (follow the guide)
- Technical approach (documented in detail)
- Function locations (can be found independently)

**What requires evidence files:**
- Actual benchmark scores
- Visual proof of improvements
- Discovery timeline validation

---

*This directory will be populated with evidence files to support all claims made in the repository.*
