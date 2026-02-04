# Sirchmunk MCP - Build and Package Guide

This guide explains how to build and package the Sirchmunk MCP server as an independent module.

## Directory Structure

```
src/sirchmunk_mcp/
├── pyproject.toml       # Project configuration and metadata
├── setup.py             # Backwards compatibility setup script
├── MANIFEST.in          # Package distribution manifest
├── LICENSE              # Apache 2.0 license
├── README_MCP.md        # Main documentation
├── INSTALL_MCP.md       # Installation guide
├── MCP_QUICKSTART.md    # Quick start guide
├── BUILD.md             # This file
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface
├── config.py            # Configuration management
├── server.py            # MCP server implementation
├── service.py           # Sirchmunk service wrapper
├── tools.py             # MCP tools definitions
├── config/              # Configuration templates
│   ├── env.example
│   └── mcp_config.json
└── tests/               # Test suite
    ├── __init__.py
    └── test_mcp_server.py
```

## Prerequisites

- Python 3.9 or higher
- pip 21.0 or higher (for PEP 517 support)
- build tools: `pip install build twine`

## Building from Source

### Step 1: Navigate to Module Directory

```bash
cd src/sirchmunk_mcp
```

### Step 2: Install Build Dependencies

```bash
pip install build twine
```

### Step 3: Build the Package

```bash
# Build both wheel and source distribution
python -m build

# This creates:
# - dist/sirchmunk_mcp-0.1.0-py3-none-any.whl (wheel)
# - dist/sirchmunk_mcp-0.1.0.tar.gz (source)
```

### Step 4: Verify Build

```bash
# List built files
ls -lh dist/

# Check package contents
tar -tzf dist/sirchmunk_mcp-0.1.0.tar.gz | head -20
```

## Local Installation

### Install from Built Wheel

```bash
# Install the wheel file
pip install dist/sirchmunk_mcp-0.1.0-py3-none-any.whl

# Or with full dependencies
pip install "dist/sirchmunk_mcp-0.1.0-py3-none-any.whl[full]"
```

### Install in Development Mode

```bash
# From the module directory
pip install -e .

# Or with dev dependencies
pip install -e ".[dev]"
```

## Testing the Build

### Run Unit Tests

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/
```

### Test CLI Installation

```bash
# Verify command is available
which sirchmunk-mcp

# Test version
sirchmunk-mcp version

# Test configuration
sirchmunk-mcp config
```

### Test Import

```python
# Test Python import
python -c "from sirchmunk_mcp import Config, create_server; print('OK')"
```

## Publishing to PyPI

### Prerequisites

- PyPI account and API token
- Configured `~/.pypirc` or environment variable `TWINE_TOKEN`

### Step 1: Build Package

```bash
python -m build
```

### Step 2: Check Package

```bash
# Validate package
twine check dist/*
```

### Step 3: Upload to Test PyPI (Optional)

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ sirchmunk-mcp
```

### Step 4: Upload to PyPI

```bash
# Upload to production PyPI
twine upload dist/*
```

## Version Management

Version is defined in `pyproject.toml`:

```toml
[project]
name = "sirchmunk-mcp"
version = "0.1.0"
```

To release a new version:

1. Update version in `pyproject.toml`
2. Update version in `src/sirchmunk_mcp/__init__.py`
3. Update `MCP_SERVER_VERSION` in `config.py`
4. Rebuild package
5. Publish to PyPI

## Dependency Management

### Core Dependencies

Defined in `pyproject.toml` under `dependencies`:
- `mcp>=0.9.0` - MCP SDK
- `pydantic>=2.0.0` - Data validation
- `openai>=1.0.0` - LLM client
- `loguru>=0.7.0` - Logging
- `duckdb>=0.10.0` - Database

### Optional Dependencies

- `[full]` - Embedding support (sentence-transformers, modelscope, torch)
- `[dev]` - Development tools (pytest, black, ruff, mypy)

### Installing with Different Dependencies

```bash
# Minimal installation
pip install sirchmunk-mcp

# With embedding support
pip install sirchmunk-mcp[full]

# Development installation
pip install sirchmunk-mcp[dev]

# All dependencies
pip install sirchmunk-mcp[full,dev]
```

## Build Configuration

### Hatchling Backend

Using `hatchling` as build backend (modern, PEP 517 compliant):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### Package Discovery

Packages are auto-discovered from the current directory:

```toml
[tool.hatch.build.targets.wheel]
packages = ["."]
```

### Included Files

Controlled by `MANIFEST.in`:
- Documentation (*.md)
- Configuration templates
- Tests
- Python source files

## Troubleshooting

### Build Fails

**Problem**: `ModuleNotFoundError` during build

**Solution**:
```bash
# Ensure build dependencies are installed
pip install --upgrade build hatchling

# Clean previous builds
rm -rf dist/ build/ *.egg-info
```

### Import Errors After Installation

**Problem**: Cannot import `sirchmunk_mcp`

**Solution**:
```bash
# Check installation
pip show sirchmunk-mcp

# Verify package contents
python -c "import sirchmunk_mcp; print(sirchmunk_mcp.__file__)"
```

### Missing Dependencies

**Problem**: Runtime errors due to missing dependencies

**Solution**:
```bash
# Reinstall with dependencies
pip install --force-reinstall sirchmunk-mcp[full]
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Build and Publish

on:
  release:
    types: [published]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install build twine
    
    - name: Build package
      run: |
        cd src/sirchmunk_mcp
        python -m build
    
    - name: Publish to PyPI
      env:
        TWINE_USERNAME: __token__
        TWINE_PASSWORD: ${{ secrets.PYPI_TOKEN }}
      run: |
        cd src/sirchmunk_mcp
        twine upload dist/*
```

## Best Practices

1. **Version Pinning**: Use `~=` or `>=` for dependencies to allow updates
2. **Testing**: Always test package before publishing
3. **Documentation**: Keep README and docs up to date
4. **Changelog**: Maintain a CHANGELOG.md for version history
5. **Semantic Versioning**: Follow semver (MAJOR.MINOR.PATCH)

## Additional Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [Hatchling Documentation](https://hatch.pypa.io/)
- [MCP Specification](https://modelcontextprotocol.io/)
- [PyPI Publishing Guide](https://packaging.python.org/tutorials/packaging-projects/)

---

For installation instructions, see [INSTALL_MCP.md](INSTALL_MCP.md).

For usage instructions, see [README_MCP.md](README_MCP.md).
