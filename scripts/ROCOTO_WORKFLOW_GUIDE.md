# AuroraGCAFS Rocoto Workflow Guide

## Overview

This document describes the Rocoto workflow management system implemented for the AuroraGCAFS project. The workflow automates the process of model training, continuous learning, model evaluation, and metrics tracking on URSA RDHPCS.

## Features

- **Initial Training**: Trains the initial model using data from a specific time period
- **Continual Learning**: Updates the model incrementally using Elastic Weight Consolidation (EWC)
- **Model Evaluation**: Evaluates model performance on test data and generates metrics
- **Metrics Tracking**: Aggregates metrics over time to track model improvement

## Prerequisites

1. Access to URSA RDHPCS
2. Rocoto workflow management system (typically available on HPC systems)
3. AuroraGCAFS codebase properly installed

## Directory Structure

The workflow uses the following directory structure:

```
$SCRATCH/auroragcafs/
├── logs/              # Log files from all jobs
├── models/            # Trained model checkpoints
├── training/          # Training data and intermediate results
│   └── YYYYMMDD/      # Results by date
├── evaluation/        # Model evaluation results
│   └── YYYYMMDD/      # Results by date
└── metrics/           # Aggregated metrics and visualizations
    └── YYYYMMDD/      # Results by date
```

## Workflow Configuration

The workflow is defined in `workflow_auroragcafs.xml`. Key configuration elements:

- **Account**: Your project code for resource allocation
- **Time Periods**: Start date, end date, and interval for training cycles
- **Resources**: CPU, memory, and GPU allocations for each task
- **Dependencies**: Task execution order and dependencies

## Tasks

### Initial Training

Trains a new model from scratch using data from a specified time period.

- **Script**: `train_model.py`
- **Resources**: 1 node, 16 CPUs, 1 GPU, 128GB memory
- **Outputs**: Model checkpoint, training logs

### Continual Learning

Updates an existing model with new data while preserving knowledge of previous data.

- **Script**: `train_model.py` (with EWC parameters)
- **Dependencies**: Initial training or previous continual learning task
- **Resources**: 1 node, 16 CPUs, 1 GPU, 128GB memory
- **Outputs**: Updated model checkpoint, training logs

### Model Evaluation

Evaluates model performance on test data and generates metrics and visualizations.

- **Script**: `evaluate_model.py`
- **Dependencies**: Initial training or continual learning task
- **Resources**: 1 node, 8 CPUs, 1 GPU, 128GB memory
- **Outputs**: Metrics JSON, visualization images

### Metrics Update

Aggregates metrics from all evaluation runs and tracks model improvement over time.

- **Script**: `update_metrics.py`
- **Dependencies**: At least one model evaluation task
- **Resources**: 1 node, 4 CPUs, 32GB memory
- **Outputs**: Aggregated metrics JSON, trend visualization images

## Using the Rocoto Controller

The `rocoto_controller.sh` script provides a simple interface for managing the workflow:

```bash
# Setup the workflow (first time or after changes)
./rocoto_controller.sh setup YYYYMMDD YYYYMMDD [interval_days]

# Start or resume the workflow
./rocoto_controller.sh start

# Check workflow status
./rocoto_controller.sh status
./rocoto_controller.sh status detailed  # For more detailed status

# Rerun a specific task that failed
./rocoto_controller.sh rewind task_name cycle_time

# Stop the workflow
./rocoto_controller.sh stop

# Clean up workflow database and (optionally) logs
./rocoto_controller.sh clean
./rocoto_controller.sh clean all  # Clean logs too
```

## Automation with Cron

For continuous execution, add a cron job to run the workflow periodically:

```bash
# Add to crontab to run every 10 minutes
*/10 * * * * cd /path/to/AuroraGCAFS/scripts && ./rocoto_controller.sh start > $SCRATCH/auroragcafs/logs/rocoto_cron.log 2>&1
```

## Monitoring and Debugging

- Check task status: `./rocoto_controller.sh status`
- View detailed task information: `./rocoto_controller.sh status detailed`
- View log files in `$SCRATCH/auroragcafs/logs/`
- View SLURM job details: `scontrol show job [jobid]`

## Best Practices

1. **Initial Setup**: Always run the setup command before starting the workflow
2. **Regular Monitoring**: Check status regularly to ensure tasks are completing
3. **Resource Allocation**: Adjust memory and walltime in XML file if needed
4. **Fault Tolerance**: Use the rewind command to restart failed tasks
5. **Clean Up**: Periodically clean old log files to save disk space

## Common Issues and Solutions

- **Missing Model Dependencies**: Check that model files exist at expected locations
- **Resource Limitations**: If jobs fail due to memory/time limits, adjust in the XML file
- **Workflow Database Corruption**: Use `clean` command and set up again
- **Module Conflicts**: Ensure environment modules are loaded in correct order

## For More Information

- Rocoto Documentation: https://github.com/NOAA-EMC/Rocoto/wiki
- URSA RDHPCS User Guide: [URSA documentation link]
- AuroraGCAFS Documentation: See `docs/` directory
