#!/usr/bin/env python3
"""
RTX 50-Series Driver Patch Tool - Reads from patch_config.json
See GHIDRA_TODO.md for what information you need to provide.
"""
import sys, os, shutil, json
from pathlib import Path
from datetime import datetime

CONFIG = json.load(open(Path(__file__).parent / "patch_config.json"))
DRIVER_VERSION = CONFIG["driver_version"]
PLATFORM = CONFIG["platform"]
DRIVER_PATHS = {
    "linux": f"/usr/lib/x86_64-linux-gnu/libcuda.so.{DRIVER_VERSION}",
    "windows": "C:\\Windows\\System32\\nvcuda.dll"
}
PATCHES = [(int(p["offset"],16), int(p["original_byte"],16), int(p["new_byte"],16), p["name"]) for p in CONFIG["patches"]]
