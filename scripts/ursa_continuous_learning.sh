#!/bin/bash
###############################################################################
# URSA Continuous Learning Job script for AuroraGCAFS
#
# This script implements a continuous learning pipeline that incrementally
# trains the AuroraGCAFS model on data from multiple time periods while
# preventing catastrophic forgetting using Elastic Weight Consolidation (EWC).
#
# Usage:
#   sbatch ursa_continuous_learning.sh <start_date> <end_date> [interval_days] [epochs_per_period] [batch_size]
#
# Arguments:
#   start_date       : Overall training start date (YYYYMMDD format)
#   end_date         : Overall training end date (YYYYMMDD format)
#   interval_days    : Days per training period (default: 30)
#   epochs_per_period: Training epochs per period (default: 10)
#   batch_size       : Training batch size (default: 64)
#
# Outputs:
#   Each period's model is saved to $SCRATCH/auroragcafs/continuous_training_<timestamp>/
#   The final model is copied to $SCRATCH/auroragcafs/final_model_<timestamp>.pt
#
# Author: NOAA/NESDIS Team
# Date: May 2025
###############################################################################

#SBATCH --job-name=aurora_continuous
#SBATCH --output=logs/continuous_%j.log
#SBATCH --error=logs/continuous_%j.err
#SBATCH --time=72:00:00
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

# Enable job checkpointing
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

# Activate conda environment
source /usr/local/other/miniconda3/etc/profile.d/conda.sh
conda activate auroragcafs

# Set environment variables
export PYTHONPATH=${PWD}:$PYTHONPATH

# Create base directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BASE_DIR="${SCRATCH}/auroragcafs/continuous_training_${TIMESTAMP}"
mkdir -p ${BASE_DIR}/logs
mkdir -p ${BASE_DIR}/checkpoints

# Log information about job
echo "Starting continuous learning job at: $(date)"
echo "Running on node: $(hostname)"
echo "SLURM JOB ID: ${SLURM_JOB_ID}"
echo "Base directory: ${BASE_DIR}"

# Arguments for the continuous training
INITIAL_DATE=${1:-"20250101"}
FINAL_DATE=${2:-"20251231"}
INTERVAL_DAYS=${3:-30}
EPOCHS_PER_PERIOD=${4:-10}
BATCH_SIZE=${5:-64}

# Function to increment date by N days (works on URSA)
increment_date() {
    date -d "$1 + $2 days" +%Y%m%d
}

# Start with initial training
current_date=$INITIAL_DATE
end_date=$(increment_date $current_date $((INTERVAL_DAYS - 1)))
current_model=""

echo "Starting continuous learning from $INITIAL_DATE to $FINAL_DATE"
echo "Training in intervals of $INTERVAL_DAYS days, $EPOCHS_PER_PERIOD epochs per period"

while [[ $current_date < $FINAL_DATE ]]; do
    echo "===== Training period: $current_date to $end_date ====="

    # Set output directory for this period
    period_dir="$BASE_DIR/${current_date}_${end_date}"
    mkdir -p $period_dir/logs
    mkdir -p $period_dir/checkpoints

    echo "$(date): Starting training for period $current_date to $end_date"
    echo "Outputs will be saved to: $period_dir"

    # Execute training for this period
    if [[ -z $current_model ]]; then
        # Initial training without previous model
        srun python scripts/train_model.py \
            --start-date $current_date \
            --end-date $end_date \
            --output-dir $period_dir \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS_PER_PERIOD \
            --learning-rate 0.001 \
            --ewc-lambda 0.0
    else
        # Continual learning with previous model
        srun python scripts/train_model.py \
            --start-date $current_date \
            --end-date $end_date \
            --model-path $current_model \
            --output-dir $period_dir \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS_PER_PERIOD \
            --learning-rate 0.0005 \
            --ewc-lambda 100.0
    fi

    # Check if the training was successful
    if [ $? -ne 0 ]; then
        echo "ERROR: Training failed for period $current_date to $end_date"
        echo "Check logs for details"
        exit 1
    fi

    # Update current model to best checkpoint from this period
    current_model="$period_dir/checkpoints/model_${current_date}_${end_date}_best.pt"

    # Create symbolic link to latest model
    if [ -f "$current_model" ]; then
        ln -sf $current_model "$BASE_DIR/latest_model.pt"
        echo "Updated latest model link to: $current_model"
    else
        echo "WARNING: Model checkpoint not found at $current_model"
    fi

    # Increment dates
    current_date=$(increment_date $end_date 1)
    end_date=$(increment_date $current_date $((INTERVAL_DAYS - 1)))

    # Ensure we don't go past the final date
    if [[ $end_date > $FINAL_DATE ]]; then
        end_date=$FINAL_DATE
    fi

    # Quick status report
    echo "Completed period: $current_date to $end_date"
    echo "Next period: $current_date to $end_date"
    echo "Current model: $current_model"
    echo ""
done

echo "Continuous learning completed at: $(date)"
echo "Final model: $current_model"
echo "All outputs saved to: $BASE_DIR"

# Copy the final model to a well-known location
cp $current_model "${SCRATCH}/auroragcafs/final_model_${TIMESTAMP}.pt"
echo "Final model copied to: ${SCRATCH}/auroragcafs/final_model_${TIMESTAMP}.pt"
