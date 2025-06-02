#!/bin/bash
#SBATCH --job-name=aurora_train
#SBATCH --output=logs/train_%j.log
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
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
START_DATE=${1:-"20250101"}
END_DATE=${2:-"20250131"}
MODEL_PATH=${3:-""}  # Optional path to pre-trained model
OUTPUT_DIR="outputs/training/${START_DATE}_${END_DATE}"

# Create output directory
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/checkpoints

# Run training script
srun python scripts/train_model.py \
    --start-date $START_DATE \
    --end-date $END_DATE \
    --output-dir $OUTPUT_DIR \
    --batch-size 32 \
    --epochs 10 \
    --learning-rate 0.001 \
    --ewc-lambda 100.0 \
    ${MODEL_PATH:+--model-path $MODEL_PATH}
