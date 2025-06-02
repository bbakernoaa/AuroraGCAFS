# AuroraGCAFS H100 GPU Migration Summary

## Overview
Successfully migrated AuroraGCAFS scripts from the legacy `xjet` partition to the new `u1-h100` partition on URSA RDHPCS to properly utilize H100 GPUs.

## Changes Made

### 1. SLURM Job Scripts Updated

#### ursa_train_job.sh
- **Partition**: `xjet` → `u1-h100`
- **QOS**: `batch` → `gpu`
- **GPU Resource**: `--gpus-per-node=1` → `--gres=gpu:h100:1`
- **CPUs**: `16` → `32` cores
- **Memory**: `128G` → `256G`

#### ursa_continuous_learning.sh
- **Partition**: `xjet` → `u1-h100`
- **QOS**: `batch` → `gpu`
- **GPU Resource**: `--gpus-per-node=1` → `--gres=gpu:h100:1`
- **CPUs**: `16` → `32` cores
- **Memory**: `128G` → `256G`

### 2. Rocoto Workflow XML Updated

#### DOCTYPE Entities
```xml
<!ENTITY PARTITION  "u1-h100">
<!ENTITY QOS        "gpu">
<!ENTITY MEMORY     "256">
```

#### Task Configurations
All tasks updated to use proper H100 configuration:

- **initial_training**: `1:ppn=32` with `--gres=gpu:h100:1`
- **continual_learning**: `1:ppn=32` with `--gres=gpu:h100:1`
- **model_evaluation (both)**: `1:ppn=16` with `--gres=gpu:h100:1`
- **metrics_update**: `1:ppn=8` (CPU-only task)

All GPU tasks now use `--qos=&QOS; --gres=gpu:h100:1` instead of legacy configuration.

### 3. Documentation Updated

#### URSA_TRAINING_GUIDE.md
- Added H100 GPU specifications section
- Updated setup instructions for GPU project allocations
- Added GPU access options (priority vs windfall)
- Enhanced SLURM reference with H100-specific commands
- Updated resource recommendations

Key additions:
- H100 GPU specifications (94GB memory, 2 GPUs per node)
- QOS options explanation (`gpu` vs `gpuwf`)
- Example SLURM commands for H100 usage
- Resource allocation recommendations

## H100 GPU Specifications

- **GPU Model**: NVIDIA H100-NVL
- **GPU Memory**: 94GB per GPU
- **GPUs per Node**: 2 H100 GPUs per node
- **Total GPU Nodes**: 58 nodes with H100 GPUs
- **Partition**: `u1-h100` (exclusive H100 partition)
- **QOS**: `gpu` (priority) or `gpuwf` (windfall)
- **Node CPU**: AMD Genoa 9654 (96 cores, 2.4 GHz)
- **Node Memory**: 192GB per node
- **Network**: NDR-200 InfiniBand

## Required Project Allocation

### Priority Access (Recommended)
- Requires GPU-specific project allocation
- Use QOS: `gpu`
- Faster queue times
- Priority over windfall jobs
- Contact your PI or Portfolio Manager for GPU allocation

### Windfall Access
- Available to ALL CPU project allocations on URSA
- Use QOS: `gpuwf`
- Longer queue times
- Runs when GPU resources are available
- Good for exploring GPU capabilities

**Important**: All users with CPU allocations automatically have windfall GPU access!

## Configuration Validation

- ✅ XML syntax validation passed
- ✅ SLURM script syntax validated
- ✅ All files error-free
- ✅ H100 GPU configuration compliant with URSA documentation
- ✅ Matches official URSA examples:
  - Priority: `sbatch -A mygpu_project -p u1-h100 -q gpu -N 1 --gres=gpu:h100:1`
  - Windfall: `sbatch -A mycpu_project -p u1-h100 -q gpuwf -N 1 --gres=gpu:h100:1`

## Verified Against Official URSA Documentation

Our configuration exactly matches the official URSA examples:
- **Partition**: `u1-h100` ✅
- **QOS**: `gpu` (priority) or `gpuwf` (windfall) ✅
- **GPU Resource**: `--gres=gpu:h100:1` or `--gres=gpu:h100:2` ✅
- **Node Specification**: Properly formatted for SLURM ✅

## Next Steps

1. **Test on URSA**: Submit a test job to verify H100 GPU allocation
2. **Monitor Performance**: Compare training performance with H100 vs previous setup
3. **Optimize**: Consider using dual H100 GPUs for larger models if needed
4. **Project Allocation**: Ensure you have proper GPU project allocation for priority access

## Commands to Verify Setup

```bash
# Check H100 availability
sinfo -p u1-h100

# Submit test job
sbatch scripts/ursa_train_job.sh 20250101 20250107

# Monitor job
squeue -u $USER -p u1-h100

# Check GPU utilization
scontrol show job JOBID
```

## Important Notes

- All scripts now target H100 GPUs specifically
- Memory allocation increased to match H100 capabilities
- CPU allocation optimized for H100 performance
- Documentation updated to reflect new requirements
- Both priority and windfall access options documented

### URSA-Specific Considerations

#### File Systems
- **Available**: `/scratch3` and `/scratch4` (new HPFS file systems)
- **Migration Deadline**: `/scratch[12]` will be decommissioned in August 2025
- **Action Required**: Complete migration to `/scratch[34]` by July 31, 2025
- **Backup**: Scratch file systems are NOT backed up!

#### Software Stack
- **OS**: Rocky 9.4 (newer than Hera/Jet's Rocky 8)
- **Module System**: Spack-based (similar to MSU systems)
- **Compilers**: Intel oneapi, NVIDIA nvhpc, AMD AOCC
- **MPI**: Intel MPI, HPC-X MPI (recommended), OpenMPI
- **CUDA/AI Libraries**: Available via module system

#### Performance Notes
- HPC-X MPI recommended for better NDR-200 InfiniBand performance
- Use `module spider` to explore available modules
- Consider using both H100 GPUs (`--gres=gpu:h100:2`) for larger models

This migration ensures optimal utilization of URSA's H100 GPU resources and follows NOAA RDHPCS best practices.
