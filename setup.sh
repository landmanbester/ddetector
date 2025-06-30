#!/bin/bash

# DDetector Project Setup Script
# Run this script in your ddetector directory

echo "Setting up DDetector project structure..."

# Create directory structure
mkdir -p src/ddetector
mkdir -p tests/data
mkdir -p docs/examples
mkdir -p examples/config_examples
mkdir -p .github/workflows

# Create __init__.py files
touch src/ddetector/__init__.py
touch tests/__init__.py

# Create main package files
cat > src/ddetector/__init__.py << 'EOF'
"""
DDetector: Automated Direction-Dependent Calibration Region Detection

A tool for detecting regions requiring direction-dependent calibration
in radio interferometric residual images.
"""

__version__ = "0.1.0"
__author__ = "Landman Bester"
__email__ = "lbester@sarao.ac.za"

from .core import DetectionParameters, RegionInfo
from .detection import ResidualImageAnalyzer
from .io_utils import write_ds9_region_file, write_casa_region_file
from .visualization import DetectionVisualizer, AdvancedAnalyzer

__all__ = [
    "DetectionParameters",
    "RegionInfo",
    "ResidualImageAnalyzer",
    "write_ds9_region_file",
    "write_casa_region_file",
    "DetectionVisualizer",
    "AdvancedAnalyzer"
]
EOF

# Create pyproject.toml
cat > pyproject.toml << 'EOF'
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ddetector"
version = "0.1.0"
description = "Automated detection of direction-dependent calibration regions in radio interferometric residual images"
authors = [
    {name = "Landman Bester", email = "lbester@sarao.ac.za"}
]
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.8"
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Astronomy",
]
keywords = ["radio astronomy", "interferometry", "calibration", "machine learning"]

dependencies = [
    "numpy>=1.20.0",
    "jax>=0.4.0",
    "jaxlib>=0.4.0",
    "scipy>=1.7.0",
    "scikit-learn>=1.0.0",
    "scikit-image>=0.18.0",
    "astropy>=5.0.0",
    "matplotlib>=3.5.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=6.0.0",
    "pytest-cov>=3.0.0",
    "pytest-xdist>=2.0.0",
    "black>=22.0.0",
    "isort>=5.0.0",
    "flake8>=4.0.0",
    "mypy>=0.950",
    "pre-commit>=2.0.0",
]
docs = [
    "sphinx>=4.0.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinxcontrib-napoleon>=0.7",
]
ray = [
    "ray[default]>=2.0.0",
]
dask = [
    "dask[complete]>=2022.0.0",
]

[project.urls]
Homepage = "https://github.com/landmanbester/ddetector"
Documentation = "https://ddetector.readthedocs.io"
Repository = "https://github.com/landmanbester/ddetector"
Issues = "https://github.com/landmanbester/ddetector/issues"

[project.scripts]
ddetect = "ddetector.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
ddetector = ["data/*.fits", "configs/*.json"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--cov=ddetector",
    "--cov-report=html",
    "--cov-report=term-missing",
    "--strict-markers",
    "-v"
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
    "gpu: marks tests requiring GPU",
]

[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?


[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
known_first_party = ["ddetector"]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
strict_equality = true

[[tool.mypy.overrides]]
module = [
    "scipy.*",
    "sklearn.*",
    "skimage.*",
    "ray.*",
    "dask.*",
]
ignore_missing_imports = true
EOF

# Create GitHub Actions workflow
cat > .github/workflows/ci.yml << 'EOF'
name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11']
        exclude:
          - os: macos-latest
            python-version: '3.8'

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip packages
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/pyproject.toml') }}
        restore-keys: |
          ${{ runner.os }}-pip-

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"

    - name: Lint with flake8
      run: |
        flake8 src/ddetector --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ddetector --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

    - name: Format check with black
      run: |
        black --check src/ddetector tests

    - name: Import sort check
      run: |
        isort --check-only src/ddetector tests

    - name: Type check with mypy
      run: |
        mypy src/ddetector

    - name: Test with pytest
      run: |
        pytest tests/ --cov=ddetector --cov-report=xml

    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
EOF

# Create .gitignore
cat > .gitignore << 'EOF'
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
pip-wheel-metadata/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Project specific
*.fits
*.reg
*.crtf
*.json
!examples/config_examples/*.json
!src/ddetector/configs/*.json
ddetector_analysis/
comprehensive_analysis/
*.png
*.html
EOF

# Create LICENSE
cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2025 Landman Bester

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF

# Create example config files
cat > examples/config_examples/lband_config.json << 'EOF'
{
  "negative_threshold_sigma": 3.0,
  "spike_threshold_percentile": 95.0,
  "ripple_frequency_range": [0.1, 1.5],
  "min_region_size_pixels": 50,
  "clustering_eps": 10.0,
  "clustering_min_samples": 5,
  "morphology_disk_size": 3,
  "edge_buffer_pixels": 20,
  "description": "Standard configuration for L-band (1-2 GHz) observations"
}
EOF

cat > examples/config_examples/conservative_config.json << 'EOF'
{
  "negative_threshold_sigma": 4.0,
  "spike_threshold_percentile": 98.0,
  "ripple_frequency_range": [0.2, 1.8],
  "min_region_size_pixels": 80,
  "clustering_eps": 20.0,
  "clustering_min_samples": 10,
  "morphology_disk_size": 4,
  "edge_buffer_pixels": 30,
  "description": "Conservative settings to minimize false positives"
}
EOF

echo "✅ DDetector project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Copy the main code files into src/ddetector/"
echo "2. Copy the test files into tests/"
echo "3. Update README.md with your information"
echo "4. Initialize git and push to GitHub"
echo ""
echo "To initialize git and connect to GitHub:"
echo "git add ."
echo "git commit -m 'Initial commit: DDetector tool for direction-dependent calibration'"
echo "git remote add origin https://github.com/landmanbester/ddetector.git"
echo "git push -u origin main"
