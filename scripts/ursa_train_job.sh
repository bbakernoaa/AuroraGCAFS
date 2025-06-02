#!/bin/bash
###############################################################################
# URSA Training Job script for AuroraGCAFS
#
# This script submits a training job to the URSA RDHPCS SLURM scheduler.
# It handles both training from scratch and fine-tuning from a pre-trained model.
#
# Usage:
#   sbatch ursa_train_job.sh <start_date> <end_date> [model_path]
#
# Arguments:
#   start_date : Training data start date (YYYYMMDD format)
#   end_date   : Training data end date (YYYYMMDD format)
#   model_path : Optional path to pre-trained model for fine-tuning
#
# Outputs:
#   Model checkpoints and logs are saved to $SCRATCH/auroragcafs/training/
#
# Author: NOAA Team
# Date: May 2025
###############################################################################

#SBATCH --job-name=aurora_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=96
#SBATCH --mem=192G
#SBATCH --partition=u1-h100
#SBATCH --account=your-project-code
#SBATCH --qos=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@noaa.gov

# Enable job checkpointing for long-running jobs
#SBATCH --checkpoint=01:00:00
#SBATCH --checkpoint-dir=$SCRATCH/checkpoints

# Set URSA environment variables
export SLURM_EXPORT_ENV=ALL
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# Load required modules for URSA
module purge
module use /apps/modules
module load conda
module load cuda
module load cudnn

# Activate conda environment (adjust path if needed)
source /usr/local/other/miniconda3/etc/profile.d/conda.sh
conda activate auroragcafs

# Set environment variables
export PYTHONPATH=${PWD}:$PYTHONPATH

# Create directory for outputs
DATE_NOW=$(date +"%Y%m%d_%H%M%S")
START_DATE=${1:-"20250101"}
END_DATE=${2:-"20250131"}
MODEL_PATH=${3:-""}  # Optional path to pre-trained model
OUTPUT_DIR="${SCRATCH}/auroragcafs/training/${START_DATE}_${END_DATE}_${DATE_NOW}"
mkdir -p ${OUTPUT_DIR}/logs
mkdir -p ${OUTPUT_DIR}/checkpoints

# Log information about job
echo "Starting job at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM JOB ID: ${SLURM_JOB_ID}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "Training period: ${START_DATE} to ${END_DATE}"
echo "Outputs will be saved to: ${OUTPUT_DIR}"

# Run training with srun to get proper GPU allocation
srun python scripts/train_model.py \
    --start-date ${START_DATE} \
    --end-date ${END_DATE} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size 64 \
    --epochs 30 \
    --learning-rate 0.001 \
    --ewc-lambda 100.0 \
    ${MODEL_PATH:+--model-path ${MODEL_PATH}}

# Create a symlink to latest model
if [ -f "${OUTPUT_DIR}/checkpoints/model_${START_DATE}_${END_DATE}_best.pt" ]; then
    ln -sf ${OUTPUT_DIR}/checkpoints/model_${START_DATE}_${END_DATE}_best.pt ${SCRATCH}/auroragcafs/latest_model.pt
    echo "Latest model linked at: ${SCRATCH}/auroragcafs/latest_model.pt"
fi

echo "Job completed at: $(date)"
