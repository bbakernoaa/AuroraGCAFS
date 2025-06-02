#!/bin/bash
#SBATCH --job-name=aurora_incremental
#SBATCH --output=logs/incremental_%j.log
#SBATCH --error=logs/incremental_%j.err
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=xjet
#SBATCH --account=your-project-code
#SBATCH --qos=batch

# Load required modules
module purge
module use /apps/modules
module load conda

# Activate conda environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate auroragcafs

# Set environment variables
export PYTHONPATH=/path/to/AuroraGCAFS:$PYTHONPATH

# Arguments
INITIAL_DATE=${1:-"20250101"}
FINAL_DATE=${2:-"20251231"}
INTERVAL_DAYS=${3:-30}
OUTPUT_BASE_DIR="outputs/incremental_training"

# Create base output directory
mkdir -p $OUTPUT_BASE_DIR/logs
mkdir -p $OUTPUT_BASE_DIR/checkpoints

# Function to increment date by N days
increment_date() {
    date -d "$1 + $2 days" +%Y%m%d
}

# Start with initial training
current_date=$INITIAL_DATE
end_date=$(increment_date $current_date $((INTERVAL_DAYS - 1)))
current_model=""

echo "Starting incremental training from $INITIAL_DATE to $FINAL_DATE"

while [[ $current_date < $FINAL_DATE ]]; do
    echo "Training period: $current_date to $end_date"

    # Set output directory for this period
    period_dir="$OUTPUT_BASE_DIR/${current_date}_${end_date}"
    mkdir -p $period_dir/logs
    mkdir -p $period_dir/checkpoints

    # Submit training job and wait for completion
    if [[ -z $current_model ]]; then
        # Initial training without previous model
        srun python scripts/train_model.py \
            --start-date $current_date \
            --end-date $end_date \
            --output-dir $period_dir \
            --batch-size 32 \
            --epochs 10 \
            --learning-rate 0.001 \
            --ewc-lambda 100.0
    else
        # Continual learning with previous model
        srun python scripts/train_model.py \
            --start-date $current_date \
            --end-date $end_date \
            --model-path $current_model \
            --output-dir $period_dir \
            --batch-size 32 \
            --epochs 10 \
            --learning-rate 0.001 \
            --ewc-lambda 100.0
    fi

    # Update current model to best checkpoint from this period
    current_model="$period_dir/checkpoints/model_${current_date}_${end_date}_best.pt"

    # Create symbolic link to latest model
    ln -sf $current_model "$OUTPUT_BASE_DIR/latest_model.pt"

    # Increment dates
    current_date=$(increment_date $end_date 1)
    end_date=$(increment_date $current_date $((INTERVAL_DAYS - 1)))

    # Ensure we don't go past the final date
    if [[ $end_date > $FINAL_DATE ]]; then
        end_date=$FINAL_DATE
    fi
done

echo "Incremental training completed. Final model: $current_model"
