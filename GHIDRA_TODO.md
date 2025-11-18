# Ghidra Analysis TODO - What You Need to Find

## Overview

You need to find **3 hex addresses and byte values** to fill into `patch_config.json`. This document tells you exactly what to look for.

## What You Already Know

From your previous analysis:
- ✅ **FUN_00f682d0** is the primary orchestrator (address: 00f682d0)
- ✅ **FUN_005e0020** handles sm_120 strings (7 references found)
- ✅ **3 subfunctions** called by FUN_00f682d0 need to be patched
- ✅ Patches change rejection (0x2c) to acceptance

## Your Mission

Find the **exact 3 addresses and bytes** you changed to get the 60% performance boost.

---

## Step-by-Step: What to Do in Ghidra

### Step 1: Open Your Ghidra Project

```bash
# Launch Ghidra
./ghidraRun

# Open your project: blackwell_gpu_unlock
# Load file: libcuda.so.1.1
```

### Step 2: Navigate to FUN_00f682d0

1. Press **G** (Go To)
2. Enter address: `00f682d0`
3. Press Enter

You should see the primary orchestrator function.

### Step 3: Find the 3 Subfunctions

**Look for CALL instructions within FUN_00f682d0:**

```assembly
; Example pattern:
00f682d0:  PUSH RBP
00f682d1:  MOV RBP, RSP
...
00f682xx:  CALL subfunction_1    ; ← Find these 3 CALL instructions
...
00f682yy:  CALL subfunction_2    ; ← Find these 3 CALL instructions
...
00f682zz:  CALL subfunction_3    ; ← Find these 3 CALL instructions
```

**Which subfunctions to patch?**
- Look for CALLs to functions that check architecture
- Look for functions called multiple times
- Look for functions that interact with 0x2c status byte

**Hint:** Click on each CALL → Follow it → Check if it references sm_120 or 0x2c

### Step 4: For Each Subfunction - Find What You Changed

Once you identify the 3 subfunctions, navigate to each one and find:

#### Patch 1: Conditional Jump Bypass

**What to look for:**
```assembly
; Example pattern in subfunction:
CMP     EAX, 0x120         ; Compare with sm_120 value
JE      reject_label       ; Jump if equal (REJECT sm_120)
                           ; ← THIS IS WHAT YOU LIKELY CHANGED
```

**What you need:**
- **Offset:** Address of the JE instruction (e.g., `0x005e1234`)
- **Original byte:** The JE opcode (likely `0x74` or `0x84`)
- **New byte:** What you changed it to (likely `0xEB` for JMP, or `0x90` for NOP)

#### Patch 2: Return Value Change

**What to look for:**
```assembly
; Example pattern:
reject_label:
    MOV     EAX, 0x2c       ; Return rejection status
                            ; ← THIS IS WHAT YOU LIKELY CHANGED
    RET
```

**What you need:**
- **Offset:** Address of the MOV instruction or the 0x2c byte
- **Original byte:** `0x2c` (rejection status)
- **New byte:** `0x00` (success status)

#### Patch 3: Validation Gate Disable

**What to look for:**
```assembly
; Example pattern:
TEST    EAX, EAX            ; Test return value
JNZ     fail_path           ; Jump if not zero (fail)
                            ; ← THIS IS WHAT YOU LIKELY CHANGED
```

**What you need:**
- **Offset:** Address of the conditional jump or test
- **Original byte:** The conditional jump opcode
- **New byte:** NOP (0x90) or unconditional jump

### Step 5: Record Your Findings

For each of the 3 patches, write down:

```
Patch 1:
- Address: 0x________
- Original byte: 0x__
- New byte: 0x__
- Description: (what this instruction does)

Patch 2:
- Address: 0x________
- Original byte: 0x__
- New byte: 0x__
- Description: (what this instruction does)

Patch 3:
- Address: 0x________
- Original byte: 0x__
- New byte: 0x__
- Description: (what this instruction does)
```

---

## Quick Reference: Common Opcodes

### Conditional Jumps (Common Targets for Patch)

| Opcode | Instruction | Meaning |
|--------|-------------|---------|
| `0x74` | JE / JZ | Jump if equal / Jump if zero |
| `0x75` | JNE / JNZ | Jump if not equal / Jump if not zero |
| `0x84` | JE (long) | Jump if equal (32-bit offset) |
| `0x85` | JNE (long) | Jump if not equal (32-bit offset) |

### What You Likely Changed Them To

| New Opcode | Instruction | Effect |
|------------|-------------|--------|
| `0xEB` | JMP (short) | Unconditional jump (always bypass) |
| `0x90` | NOP | No operation (skip the check entirely) |
| `0xE9` | JMP (long) | Unconditional jump (32-bit offset) |

### Return Values

| Byte | Meaning |
|------|---------|
| `0x2c` | Rejection status (fail) |
| `0x00` | Success status (accept) |

---

## How to Find Addresses in Ghidra

### Method 1: Click on Instruction

1. Click on the instruction you want
2. Look at the **Address column** (left side)
3. That's your offset (e.g., `005e1234`)

### Method 2: View Hex Bytes

1. Right-click in Listing window
2. Select "Bytes" → "Show Bytes"
3. You'll see the hex bytes next to each instruction
4. The byte you need is usually the first byte of the instruction

### Example

```
Address   Bytes           Assembly
005e1234  74 0A           JE 005e1240
                          ^^
                          This is original byte (0x74)
```

If you changed this to:
```
Address   Bytes           Assembly
005e1234  EB 0A           JMP 005e1240
                          ^^
                          New byte (0xEB)
```

Then your patch is:
- Offset: `0x005e1234`
- Original: `0x74`
- New: `0xEB`

---

## Filling in patch_config.json

Once you have all 3 patches documented, edit `patch_config.json`:

```json
{
  "driver_version": "570.00",
  "platform": "linux",

  "patches": [
    {
      "name": "Subfunction 1 - Conditional jump bypass",
      "offset": "0x005e1234",        ← YOUR ADDRESS HERE
      "original_byte": "0x74",       ← YOUR ORIGINAL BYTE
      "new_byte": "0xEB",            ← YOUR NEW BYTE
      "_todo": "DONE"
    },
    {
      "name": "Subfunction 2 - Return value change",
      "offset": "0x005e5678",        ← YOUR ADDRESS HERE
      "original_byte": "0x2c",       ← YOUR ORIGINAL BYTE
      "new_byte": "0x00",            ← YOUR NEW BYTE
      "_todo": "DONE"
    },
    {
      "name": "Subfunction 3 - Validation gate disable",
      "offset": "0x00f68abc",        ← YOUR ADDRESS HERE
      "original_byte": "0x75",       ← YOUR ORIGINAL BYTE
      "new_byte": "0x90",            ← YOUR NEW BYTE
      "_todo": "DONE"
    }
  ]
}
```

---

## Verification Checklist

Before running the patching script, verify:

- [ ] All 3 offsets start with `0x` and are valid hex addresses
- [ ] All original_byte and new_byte values are 2-digit hex (e.g., `0x74`, not `0x7`)
- [ ] Offsets look reasonable (between 0x00100000 and 0x02000000 for this driver)
- [ ] You remember making these exact changes when you got the 60% boost
- [ ] You have backup of original driver

---

## Timeline

**When you have time:**
1. Open Ghidra project (you already have this)
2. Navigate to FUN_00f682d0
3. Find the 3 subfunctions you patched
4. Document the exact addresses and bytes
5. Update patch_config.json
6. Run the automated patching script

**Estimated time:** 30-60 minutes (since you already know what to look for)

---

## What Happens After You Fill This In

Once `patch_config.json` has real values:

1. Anyone with RTX 5080 can use the automated script
2. They just run: `sudo python3 patch_rtx50_driver.py`
3. Script reads config, patches driver, creates backup
4. Reboot → 60% performance unlock
5. **No Ghidra knowledge required for other users**

**You're doing the hard reverse engineering work once, so everyone else can benefit.**

---

## Questions While Doing This?

If you get stuck:
- Reference your original Ghidra analysis screenshots
- Look at the 0x2c scalar search results (2,616 instances)
- Check sm_120 string XREFs (7 references in FUN_005e0020)
- Remember: You already did this successfully - just need to document what you changed

**Good luck! Take your time - accurate addresses are critical.**
