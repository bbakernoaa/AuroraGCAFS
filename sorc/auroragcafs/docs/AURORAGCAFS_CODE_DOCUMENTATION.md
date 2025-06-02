
# AuroraGCAFS Code Documentation

## Overview
AuroraGCAFS is a modular, extensible framework for global coupled data assimilation and forecasting, designed for NOAA's RDHPCS systems. It supports distributed training, continual learning, and evaluation workflows, optimized for H100 GPUs on URSA.

## Directory Structure
- `sorc/auroragcafs/auroragcafs/` — Main Python package
  - `config/` — Configuration management
  - `data/` — Data loading and preprocessing
  - `model/` — Model architectures and training logic
  - `utils/` — Utility functions
- `scripts/` — Workflow scripts and SLURM job scripts
  - `train_model.py` — Main training entry point
  - `evaluate_model.py` — Model evaluation
  - `update_metrics.py` — Metrics aggregation
  - `ursa_train_job.sh`, `ursa_continuous_learning.sh` — SLURM job scripts
  - `workflow_auroragcafs.xml` — Rocoto workflow definition
- `parm/config.yaml` — Main configuration file for model and data settings
- `docs/` — Documentation (this file, config module docs, etc.)

## Key Components
### Training
- Run via `train_model.py` (called by SLURM scripts)
- Uses configuration from `parm/config.yaml`
- Supports both initial and continual learning

### Evaluation
- Run via `evaluate_model.py`
- Loads trained model checkpoints and computes evaluation metrics

### Metrics Update
- Run via `update_metrics.py`
- Aggregates evaluation results and produces summary reports

### Workflow Automation
- `workflow_auroragcafs.xml` defines the full Rocoto workflow
- `rocoto_controller.sh` manages workflow setup and execution
- `test_rocoto_workflow.sh` validates workflow configuration

## Configuration
- All major settings (data paths, model hyperparameters, resource allocation) are in `parm/config.yaml`
- Modify this file to change model, data, or resource settings

## Extending AuroraGCAFS
- Add new models in `sorc/auroragcafs/auroragcafs/model/`
- Add new data sources in `sorc/auroragcafs/auroragcafs/data/`
- Update configuration in `parm/config.yaml`
- Update workflow or SLURM scripts as needed for new tasks

## Best Practices
- Use the provided SLURM scripts for all training and evaluation on RDHPCS
- Validate workflow changes with `test_rocoto_workflow.sh` before deployment
- Keep documentation up to date in `docs/`

## Further Reading
- [URSA_TRAINING_GUIDE.md](../../../../scripts/URSA_TRAINING_GUIDE.md)
- [H100_MIGRATION_SUMMARY.md](../../../../H100_MIGRATION_SUMMARY.md)
- [URSA_H100_QUICK_REFERENCE.md](../../../../URSA_H100_QUICK_REFERENCE.md)
