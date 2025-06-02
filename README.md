# AuroraGCAFS

AuroraGCAFS is a project that adapts Microsoft's Aurora AI code for use with NOAA's UFS aerosol or GEFS-aerosol output instead of CAMS data.

## Overview

This repository contains code to process, analyze, and visualize NOAA's UFS/GEFS-aerosol output using techniques inspired by Microsoft's Aurora AI system. The goal is to provide enhanced aerosol forecasting and analysis capabilities.

## Directory Structure

- `sorc/` - Source code including the main Python package
- `scripts/` - Operational scripts for running the system
  - Training scripts for model development
  - Rocoto workflow for URSA RDHPCS automation
  - SLURM job submission scripts
  - Evaluation and metrics tracking scripts
- `parm/` - Parameter and configuration files
- `util/` - Utility programs and helper scripts
- `docs/` - Documentation
- `lib/` - Libraries and dependencies
- `modulefiles/` - Environment module files
- `versions/` - Version information

## Installation

### Using pip (recommended)

You can install the package directly from the repository:

```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

### Using conda

Alternatively, you can create a conda environment:

```bash
# Create conda environment
conda env create -f environment.yml

# Activate the environment
conda activate auroragcafs
```

## Usage

### Local Development

For local development and testing:

```bash
# Train the model with specific date ranges
python scripts/train_model.py --start-date 20250101 --end-date 20250131

# Evaluate model performance
python scripts/evaluate_model.py --model-path /path/to/model.pt --data-start 20250201 --data-end 20250228
```

### URSA HPC Deployment

For running on URSA RDHPCS:

1. Set up the Rocoto workflow:

```bash
cd scripts
./rocoto_controller.sh setup 20250101 20251231 30
```

2. Start the workflow:

```bash
./rocoto_controller.sh start
```

3. Check workflow status:

```bash
./rocoto_controller.sh status
```

See `scripts/ROCOTO_WORKFLOW_GUIDE.md` and `scripts/URSA_TRAINING_GUIDE.md` for detailed instructions.

## Contributing

(Instructions for contributing to the project)

## License

(License information)
