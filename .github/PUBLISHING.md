# RTX-STone PyPI Publishing Guide

This document explains how to publish RTX-STone to PyPI using GitHub Actions and Trusted Publishers.

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org
2. **TestPyPI Account**: Create account at https://test.pypi.org (for testing)
3. **GitHub Repository**: Admin access to kentstone84/pytorch-rtx5080-support

## Setup PyPI Trusted Publisher

### Step 1: Configure Trusted Publisher on PyPI

1. Go to https://pypi.org/manage/account/publishing/
2. Scroll to "Add a new pending publisher"
3. Fill in the form:

```
PyPI Project Name: rtx-stone
Owner: kentstone84
Repository name: pytorch-rtx5080-support
Workflow name: publish.yml
Environment name: pypi
```

4. Click "Add"

### Step 2: Configure Trusted Publisher on TestPyPI (Optional but Recommended)

1. Go to https://test.pypi.org/manage/account/publishing/
2. Add the same configuration with environment name: `testpypi`

### Step 3: Create GitHub Environments

1. Go to repository Settings → Environments
2. Create environment `pypi`:
   - Add protection rules (require reviewers if desired)
   - No secrets needed (uses OIDC)

3. Create environment `testpypi` (optional):
   - For testing before production

## Publishing Workflow

### Test Publish (TestPyPI)

```bash
# Trigger manually from GitHub Actions
1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Select environment: testpypi
4. Click "Run workflow"

# Verify on TestPyPI
pip install --index-url https://test.pypi.org/simple/ rtx-stone
```

### Production Publish (PyPI)

**Option 1: Automatic on Release (Recommended)**

```bash
# Create a GitHub release
1. Go to Releases → Draft a new release
2. Create a new tag: v2.10.0a0
3. Release title: RTX-STone v2.10.0a0
4. Add release notes
5. Click "Publish release"

# Workflow triggers automatically
# Package is published to PyPI
```

**Option 2: Manual Trigger**

```bash
# Trigger manually from GitHub Actions
1. Go to Actions → Publish to PyPI
2. Click "Run workflow"
3. Select environment: pypi
4. Click "Run workflow"
```

## Versioning

Update version in `pyproject.toml`:

```toml
[project]
name = "rtx-stone"
version = "2.10.0a0"  # Update this
```

Version scheme:
- `2.10.0a0` - Alpha release (current)
- `2.10.0b1` - Beta release
- `2.10.0rc1` - Release candidate
- `2.10.0` - Stable release

## Package Structure

The package includes:

```
rtx-stone/
├── rtx_stone/              # Main package
│   ├── __init__.py
│   └── diagnostic.py
├── examples/               # Example scripts
├── notebooks/              # Jupyter tutorials
├── integrations/           # Integration examples
├── pyproject.toml         # Package metadata
├── setup.py               # Build script
└── MANIFEST.in            # File inclusion
```

## What Gets Published

Files included in distribution (defined in MANIFEST.in):
- ✅ README.md, LICENSE, CHANGELOG.md
- ✅ rtx_stone/ Python package
- ✅ examples/ directory
- ✅ notebooks/ directory
- ✅ integrations/ directory
- ✅ Documentation files
- ❌ .git/, __pycache__, build/

## Installation After Publishing

Users can install with:

```bash
# Basic installation
pip install rtx-stone

# With optional dependencies
pip install rtx-stone[triton]
pip install rtx-stone[huggingface]
pip install rtx-stone[all]
```

## Troubleshooting

### "Project name not found" error

**Solution**: First publish must be done AFTER configuring Trusted Publisher on PyPI.

### "Environment not found" error

**Solution**: Create the `pypi` environment in GitHub repository settings.

### "Permission denied" error

**Solution**: Ensure workflow has `id-token: write` permission.

### Build fails

**Solution**:
1. Test locally: `python -m build`
2. Check for syntax errors
3. Verify all files in MANIFEST.in exist

### Package too large

**Solution**: PyPI has a 100MB limit per file. For large packages:
1. Exclude large files in MANIFEST.in
2. Host binaries separately (GitHub Releases)
3. Document manual installation for large files

## Important Notes

### PyTorch Binary

⚠️ **Critical**: The actual PyTorch binary (8.3GB) is too large for PyPI.

**Solution**: The PyPI package is a "meta-package" that:
1. Provides the optimization modules (flash_attention_rtx5080.py, etc.)
2. Provides utilities and integrations
3. Requires users to install PyTorch separately

Update `pyproject.toml` if needed:

```toml
dependencies = [
    # Don't include torch - users install separately
    "filelock",
    "fsspec",
    # ... other dependencies
]
```

Document installation:

```bash
# 1. Install PyTorch from release
# Download from: https://github.com/kentstone84/pytorch-rtx5080-support/releases

# 2. Install RTX-STone utilities
pip install rtx-stone[all]
```

### Alternative: Separate Packages

Consider creating two packages:
1. `rtx-stone` - Utilities and optimizations (this repo)
2. `rtx-stone-pytorch` - PyTorch binary (separate, manual download)

## Security

- ✅ Uses OpenID Connect (no API tokens needed)
- ✅ Workflow only runs on release or manual trigger
- ✅ Requires environment protection
- ✅ Signs packages with Sigstore
- ✅ Uploads signatures to GitHub Releases

## Maintenance

### Update Package

1. Make changes to code
2. Update version in `pyproject.toml`
3. Update CHANGELOG.md
4. Create GitHub release
5. Workflow publishes automatically

### Yank a Release

```bash
# If you need to remove a broken release
pip install twine
twine upload --skip-existing  # Manual upload
# Or use PyPI web interface to yank version
```

## Resources

- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [GitHub Actions OpenID Connect](https://docs.github.com/en/actions/deployment/security-hardening-your-deployments/about-security-hardening-with-openid-connect)
- [Python Packaging Guide](https://packaging.python.org/)

---

**Ready to publish?** Follow the steps above to get RTX-STone on PyPI!
