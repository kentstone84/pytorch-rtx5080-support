# 🧠 Blackwell (sm_120) Patch for PyTorch

**🚀 Unlock full RTX 5080 performance in PyTorch!**  
PyTorch does not yet support `sm_120` (Blackwell) natively — so I built custom CUDA 12.8 drivers and patched the PyTorch build system.

This repo includes the patch, a script, and build instructions.

---

## ✅ What It Does

- ✅ Adds `"Blackwell"` as a CUDA architecture alias
- ✅ Enables `sm_120` compilation via `TORCH_CUDA_ARCH_LIST`
- ✅ Future-ready for RTX 5090, B100, GB200 series
- ✅ Compatible with CUDA 12.8, PyTorch 2.5.0+

---

## 🛠 Usage

### Step 1 – Clone PyTorch and This Patch Repo
git clone --recursive https://github.com/pytorch/pytorch
git clone https://github.com/kentstone84/pytorch-rtx5080-support.git

### Step 2 – Apply Patch
cd pytorch
../pytorch-rtx5080-support/patch_blackwell.sh

### Step 3 – Build PyTorch
export TORCH_CUDA_ARCH_LIST="Blackwell"
python setup.py install


✅ Test It Worked
  python
    import torch
    print(torch.cuda.get_device_properties(0))
    # Should show major=12, minor=0 → sm_120


How to Run the Project on Windows Using Command Prompt (CMD)
Step 1: Open Command Prompt
Press Win + R, type cmd, then press Enter.

Step 2: Navigate to Your Project Folder
Use the cd command to change directories.

Example if your project folder is on the Desktop:

bash
Copy
Edit
cd Desktop\tic_tac_toe_modern
If your folder path contains spaces, use quotes:

bash
Copy
Edit
cd "C:\Users\YourUserName\Desktop\tic_tac_toe_modern"
Step 3: Verify You’re in the Right Folder
Type:

bash
Copy
Edit
dir
You should see your project files listed.

Step 4: Run Your Script or Command
If you have a batch file (.bat) or command script, just type its name, for example:

Copy
Edit
path_blackwell.bat
For shell scripts (.sh), it’s best to use Git Bash or Windows Subsystem for Linux (WSL) to run:

bash
Copy
Edit
./path_blackwell.sh
Pro Tips:
You can open the terminal directly in the folder by Shift + Right Click inside the folder and selecting:

Open PowerShell window here or

Open Command Window here

Double-clicking shell scripts on Windows may not work without the right shell environment installed.
