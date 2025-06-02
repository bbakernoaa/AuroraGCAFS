# URSA RDHPCS Training Guide for AuroraGCAFS

This document provides instructions for running the AuroraGCAFS training on NOAA's URSA RDHPCS system using H100 GPUs.

## Important GPU Requirements

**URSA H100 GPU Configuration:**
- Partition: `u1-h100` (required for H100 access)
- QOS: `gpu` (for priority access) or `gpuwf` (for windfall access)
- GPU Resource: `--gres=gpu:h100:1` or `--gres=gpu:h100:2`
- Memory: Each H100 has 94GB of GPU memory
- CPU allocation: 32+ cores recommended for optimal performance

## Setup Instructions

1. **Connect to URSA**:
   ```bash
   ssh username@ursa-login.rdhpcs.noaa.gov
   ```

2. **Clone the repository**:
   ```bash
   cd $HOME
   git clone https://github.com/your-username/AuroraGCAFS.git
   cd AuroraGCAFS
   ```

3. **Create conda environment**:
   ```bash
   module purge
   module use /apps/modules
   module load conda
   conda env create -f environment.yml
   conda activate auroragcafs
   ```

4. **Configure your project account**:
   Edit the scripts to use your project account:
   ```bash
   # Open scripts/ursa_train_job.sh and scripts/ursa_continuous_learning.sh
   # Replace 'your-project-code' with your actual GPU project allocation
   # Replace 'your.email@noaa.gov' with your email address
   # Note: You need a GPU-specific project allocation for priority access
   ```

5. **Important File System Migration**:
   ```bash
   # URSA uses new file systems /scratch3 and /scratch4
   # Old /scratch[12] will be decommissioned in August 2025
   # Ensure your data is on /scratch3 or /scratch4
   echo $SCRATCH  # Should point to /scratch3 or /scratch4
   ```

## Running Jobs

### Using the Job Manager Script

We provide a utility script to manage your URSA jobs:

1. **Training Job**:
   ```bash
   scripts/ursa_job_manager.sh train 20250101 20250131
   ```
   To use a pre-trained model for fine-tuning:
   ```bash
   scripts/ursa_job_manager.sh train 20250101 20250131 /path/to/model.pt
   ```

2. **Continuous Learning**:
   ```bash
   scripts/ursa_job_manager.sh continuous 20250101 20251231 30
   ```
   This will train incrementally from Jan 1, 2025 to Dec 31, 2025 in 30-day intervals.

3. **Check Job Status**:
   ```bash
   scripts/ursa_job_manager.sh status
   ```

4. **Cancel a Job**:
   ```bash
   scripts/ursa_job_manager.sh cancel JOB_ID
   ```

### Submitting Jobs Directly

You can also submit jobs directly using sbatch:

1. **Training Job**:
   ```bash
   sbatch scripts/ursa_train_job.sh 20250101 20250131
   ```

2. **Continuous Learning**:
   ```bash
   sbatch scripts/ursa_continuous_learning.sh 20250101 20251231 30 10 64
   ```
   Where the parameters are:
   - Start date (YYYYMMDD)
   - End date (YYYYMMDD)
   - Interval days (default: 30)
   - Epochs per period (default: 10)
   - Batch size (default: 64)

## GPU Access Options

### Priority Access (Recommended)
If you have a GPU-specific project allocation:
- Use QOS: `gpu`
- Your jobs will have priority over windfall jobs
- Faster queue times

### Windfall Access
If you only have a CPU project allocation:
- **Good News**: ALL CPU allocations automatically have windfall GPU access!
- Change QOS to `gpuwf` in the scripts
- Jobs run when GPU resources are available
- Longer queue times but fully functional
- Great for exploring H100 capabilities

To use windfall access, modify the `--qos=gpu` line in the scripts to `--qos=gpuwf`.

**Example windfall submission**:
```bash
sbatch -A your-cpu-project -p u1-h100 -q gpuwf -N 1 --gres=gpu:h100:1 your_script.sh
```

## Output Files

Training outputs are stored in `$SCRATCH/auroragcafs/`:

- **Individual training runs**: `$SCRATCH/auroragcafs/training/[START_DATE]_[END_DATE]_[TIMESTAMP]/`
- **Continuous learning runs**: `$SCRATCH/auroragcafs/continuous_training_[TIMESTAMP]/`
- **Latest model link**: `$SCRATCH/auroragcafs/latest_model.pt`
- **Final model from continuous learning**: `$SCRATCH/auroragcafs/final_model_[TIMESTAMP].pt`

## URSA H100 SLURM Reference

### Key H100 GPU Specifications
- **GPU Model**: NVIDIA H100-NVL
- **GPU Memory**: 94GB per GPU
- **GPUs per Node**: 2 H100 GPUs
- **Partition**: `u1-h100` (exclusive for H100 access)
- **QOS Options**: `gpu` (priority) or `gpuwf` (windfall)

### Example SLURM Commands
```bash
# Submit a job with 1 H100 GPU
sbatch -A your-gpu-project -p u1-h100 -q gpu -N 1 --gres=gpu:h100:1 your_script.sh

# Submit a job with 2 H100 GPUs (for larger models)
sbatch -A your-gpu-project -p u1-h100 -q gpu -N 1 --gres=gpu:h100:2 your_script.sh

# Check GPU resource availability
sinfo -p u1-h100

# View GPU usage
squeue -p u1-h100
```

### Resource Recommendations for AuroraGCAFS
- **Single GPU**: `--gres=gpu:h100:1 --cpus-per-task=32 --mem=256G`
- **Dual GPU**: `--gres=gpu:h100:2 --cpus-per-task=64 --mem=512G`

For more information on using SLURM on URSA, see:
- [URSA User Guide](https://docs.rdhpcs.noaa.gov/systems/ursa_user_guide.html)
- [URSA GPU Documentation](https://docs.rdhpcs.noaa.gov/systems/ursa_user_guide.html#using-gpu-resources-on-ursa)
- [URSA SLURM Documentation](https://docs.rdhpcs.noaa.gov/slurm/index.html)

## Job Monitoring

To monitor job progress:
```bash
# View job details
scontrol show job JOB_ID

# View job efficiency
seff JOB_ID

# View job logs in real time
tail -f logs/train_JOB_ID.log
```

## Transferring Models

To copy trained models to other systems:
```bash
# From your local system
scp username@ursa-dtn.rdhpcs.noaa.gov:$SCRATCH/auroragcafs/final_model_*.pt /local/path/

# Or using Globus (recommended for large files)
# See URSA documentation for Globus instructions
```
