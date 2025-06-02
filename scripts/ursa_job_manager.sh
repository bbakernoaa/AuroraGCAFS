#!/bin/bash
###############################################################################
# URSA Job Management Utility for AuroraGCAFS
#
# This utility script provides a simple interface for managing URSA SLURM jobs
# for the AuroraGCAFS project, including training, continuous learning, and
# status monitoring.
#
# Usage:
#   ursa_job_manager.sh [command] [options]
#
# Commands:
#   train       - Submit a training job
#   continuous  - Submit a continuous learning job
#   status      - Show status of running jobs
#   cancel      - Cancel a job
#   help        - Show help message
#
# Examples:
#   ursa_job_manager.sh train 20250101 20250131
#   ursa_job_manager.sh continuous 20250101 20251231 30
#   ursa_job_manager.sh status
#   ursa_job_manager.sh cancel 123456
#
# Author: NOAA/NESDIS Team
# Date: May 2025
###############################################################################

# Function to display help information
show_help() {
    echo "URSA Job Management Utility for AuroraGCAFS"
    echo ""
    echo "Usage:"
    echo "  $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  train       Submit a training job"
    echo "  continuous  Submit a continuous learning job"
    echo "  status      Show status of running jobs"
    echo "  cancel      Cancel a job"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 train 20250101 20250131"
    echo "  $0 continuous 20250101 20251231 30"
    echo "  $0 status"
    echo "  $0 cancel 123456"
}

# Function to submit a training job
submit_training() {
    if [ $# -lt 2 ]; then
        echo "Error: Missing required parameters"
        echo "Usage: $0 train START_DATE END_DATE [MODEL_PATH]"
        exit 1
    fi

    START_DATE=$1
    END_DATE=$2
    MODEL_PATH=$3

    echo "Submitting training job for period: $START_DATE to $END_DATE"
    if [ ! -z "$MODEL_PATH" ]; then
        echo "Using pre-trained model: $MODEL_PATH"
    fi

    JOBID=$(sbatch scripts/ursa_train_job.sh $START_DATE $END_DATE $MODEL_PATH | awk '{print $NF}')
    echo "Job submitted with ID: $JOBID"
}

# Function to submit a continuous learning job
submit_continuous() {
    if [ $# -lt 2 ]; then
        echo "Error: Missing required parameters"
        echo "Usage: $0 continuous START_DATE END_DATE [INTERVAL_DAYS] [EPOCHS_PER_PERIOD] [BATCH_SIZE]"
        exit 1
    fi

    START_DATE=$1
    END_DATE=$2
    INTERVAL_DAYS=${3:-30}
    EPOCHS_PER_PERIOD=${4:-10}
    BATCH_SIZE=${5:-64}

    echo "Submitting continuous learning job from $START_DATE to $END_DATE"
    echo "Training in intervals of $INTERVAL_DAYS days with $EPOCHS_PER_PERIOD epochs per period"

    JOBID=$(sbatch scripts/ursa_continuous_learning.sh $START_DATE $END_DATE $INTERVAL_DAYS $EPOCHS_PER_PERIOD $BATCH_SIZE | awk '{print $NF}')
    echo "Job submitted with ID: $JOBID"
}

# Function to show job status
show_status() {
    echo "=== Your running jobs ==="
    squeue -u $USER
}

# Function to cancel a job
cancel_job() {
    if [ $# -lt 1 ]; then
        echo "Error: Missing job ID"
        echo "Usage: $0 cancel JOB_ID"
        exit 1
    fi

    JOBID=$1
    echo "Cancelling job $JOBID"
    scancel $JOBID
}

# Main execution
if [ $# -lt 1 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    train)
        submit_training "$@"
        ;;
    continuous)
        submit_continuous "$@"
        ;;
    status)
        show_status
        ;;
    cancel)
        cancel_job "$@"
        ;;
    help)
        show_help
        ;;
    *)
        echo "Unknown command: $COMMAND"
        show_help
        exit 1
        ;;
esac
