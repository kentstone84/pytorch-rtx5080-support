# PyTorch RTX 5080 Installer for Windows 11
# Run this inside your activated virtual environment

Write-Host "`n🔥 PyTorch RTX 5080 Installer" -ForegroundColor Cyan
Write-Host "================================`n" -ForegroundColor Cyan

# ---- Check if running in virtual environment --------------------------------
$inVenv = $env:VIRTUAL_ENV -or (Test-Path ".\venv") -or (Test-Path ".\.venv")
if (-not $inVenv) {
    Write-Warning "⚠️  Not in a virtual environment. Highly recommended!"
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y') { exit 0 }
}

# ---- Check Python version ----------------------------------------------------
$pyVersion = (& python --version) 2>$null
Write-Host "Detected $pyVersion"

python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) and sys.version_info < (3,12) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Python 3.10 or 3.11 required. Current: $pyVersion"
    Write-Host "Note: Python 3.12+ may have compatibility issues" -ForegroundColor Yellow
    exit 1
}

# ---- Check for CUDA installation --------------------------------------------
Write-Host "`nChecking for CUDA installation..." -ForegroundColor Yellow
$cudaPath = $env:CUDA_PATH
if (-not $cudaPath -or -not (Test-Path $cudaPath)) {
    Write-Warning "⚠️  CUDA_PATH not found or invalid"
    Write-Host "Please install CUDA Toolkit 12.4, 12.8, or 13.0 from:" -ForegroundColor Yellow
    Write-Host "https://developer.nvidia.com/cuda-downloads`n" -ForegroundColor Cyan
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne 'y') { exit 0 }
} else {
    Write-Host "✓ Found CUDA at: $cudaPath" -ForegroundColor Green
}

# ---- Check for NVIDIA GPU ---------------------------------------------------
Write-Host "`nChecking for NVIDIA GPU..." -ForegroundColor Yellow
$gpu = (nvidia-smi --query-gpu=name --format=csv,noheader 2>$null)
if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Detected: $gpu" -ForegroundColor Green
} else {
    Write-Warning "⚠️  Could not detect NVIDIA GPU or drivers not installed"
}

# ---- Install required dependencies ------------------------------------------
Write-Host "`nInstalling dependencies..." -ForegroundColor Yellow
pip install --quiet filelock fsspec Jinja2 MarkupSafe mpmath networkx sympy "typing_extensions>=4.10.0"
if ($LASTEXITCODE -ne 0) {
    Write-Error "❌ Failed to install dependencies."
    exit 1
}
Write-Host "✓ Dependencies installed" -ForegroundColor Green

# ---- Locate site-packages and copy torch ------------------------------------
$sitePackages = python -c "import site; print(site.getsitepackages()[0])"
if (-not (Test-Path $sitePackages)) {
    Write-Error "Could not locate site-packages directory."
    exit 1
}

Write-Host "`nInstalling to: $sitePackages" -ForegroundColor Yellow

# Use PSScriptRoot to get the directory containing this script
$torchSource = Join-Path $PSScriptRoot "torch"
Write-Host "Looking for torch at: $torchSource" -ForegroundColor Yellow

if (-not (Test-Path $torchSource)) {
    Write-Error "❌ torch folder not found at: $torchSource"
    exit 1
}

# Calculate size
$torchSize = (Get-ChildItem $torchSource -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "Package size: $([math]::Round($torchSize, 2)) GB" -ForegroundColor Cyan

Write-Host "`nCopying torch folder (this may take 1-2 minutes)..." -ForegroundColor Yellow
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    # Remove old torch if it exists
    $torchDest = Join-Path $sitePackages "torch"
    if (Test-Path $torchDest) {
        Write-Host "Removing existing torch installation..." -ForegroundColor Yellow
        Remove-Item $torchDest -Recurse -Force -ErrorAction Stop
    }
    
    Copy-Item $torchSource -Destination $sitePackages -Recurse -Force -ErrorAction Stop
    $stopwatch.Stop()
    Write-Host "✓ Copy complete ($($stopwatch.Elapsed.TotalSeconds.ToString('0.0'))s)" -ForegroundColor Green
}
catch {
    Write-Error "❌ Failed to copy torch: $_"
    exit 1
}

# ---- Verify installation ----------------------------------------------------
Write-Host "`nVerifying installation..." -ForegroundColor Yellow
python -c @"
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'GPU:             {torch.cuda.get_device_name(0)}')
    print(f'Compute Cap:     sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}')
    print(f'Arch list:       {torch.cuda.get_arch_list()}')
else:
    print('⚠️  CUDA not available - check drivers and CUDA installation')
"@

if ($LASTEXITCODE -ne 0) {
    Write-Error "`n❌ Installation verification failed"
    exit 1
}

Write-Host "`n✅ Installation complete!" -ForegroundColor Green
Write-Host "`nQuick test:" -ForegroundColor Cyan
Write-Host "  python -c `"import torch; print(torch.cuda.is_available())`"" -ForegroundColor Gray