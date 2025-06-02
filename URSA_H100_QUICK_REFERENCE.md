# URSA H100 Quick Reference Card

## Essential H100 Configuration

### SLURM Job Submission
```bash
# Priority Access (GPU allocation required)
sbatch -A your-gpu-project -p u1-h100 -q gpu -N 1 --gres=gpu:h100:1 script.sh

# Windfall Access (any CPU allocation)
sbatch -A your-cpu-project -p u1-h100 -q gpuwf -N 1 --gres=gpu:h100:1 script.sh
```

### Key Parameters
- **Partition**: `u1-h100` (required)
- **QOS**: `gpu` (priority) or `gpuwf` (windfall)
- **GPU**: `--gres=gpu:h100:1` (single) or `--gres=gpu:h100:2` (dual)
- **CPUs**: `--cpus-per-task=32` (recommended for single GPU)
- **Memory**: `--mem=256G` (recommended for single GPU)

### H100 Specifications
- **GPU Model**: NVIDIA H100-NVL
- **GPU Memory**: 94GB per GPU
- **GPUs per Node**: 2 H100 GPUs
- **Total GPU Nodes**: 58
- **Node CPU**: AMD Genoa 9654 (96 cores @ 2.4GHz)
- **Node Memory**: 192GB
- **Network**: NDR-200 InfiniBand

### File Systems
- **Current**: `/scratch3`, `/scratch4`
- **Deprecated**: `/scratch1`, `/scratch2` (decommissioned Aug 2025)
- **Migration Deadline**: July 31, 2025

### AuroraGCAFS Scripts
- `scripts/ursa_train_job.sh` - Single training job
- `scripts/ursa_continuous_learning.sh` - Continuous learning
- `scripts/ursa_job_manager.sh` - Job management utility
- `scripts/workflow_auroragcafs.xml` - Rocoto workflow

### Quick Commands
```bash
# Check H100 availability
sinfo -p u1-h100

# View your jobs
squeue -u $USER -p u1-h100

# Check job details
scontrol show job JOBID

# Load modules
module spider cuda
module load conda cuda cudnn
```

### Software Stack
- **OS**: Rocky 9.4
- **Modules**: Spack-based (use `module spider`)
- **Compilers**: Intel oneapi, NVIDIA nvhpc, AMD AOCC
- **MPI**: HPC-X (recommended), Intel MPI, OpenMPI

### Contact
- GPU allocation requests: Contact your PI or Portfolio Manager
- Technical issues: [RDHPCS Help](https://docs.rdhpcs.noaa.gov/help/index.html#getting-help)
