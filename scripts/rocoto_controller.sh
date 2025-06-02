#!/bin/bash
###############################################################################
# Rocoto Workflow Controller for AuroraGCAFS
#
# This script provides a simple interface for managing AuroraGCAFS training
# workflows using Rocoto on URSA RDHPCS.
#
# Usage:
#   ./rocoto_controller.sh [command] [options]
#
# Commands:
#   setup       - Set up directories and prepare workflow
#   start       - Start or resume the workflow
#   status      - Show workflow status
#   stop        - Stop the workflow
#   clean       - Clean workflow logs and database
#   help        - Show this help message
#
# Examples:
#   ./rocoto_controller.sh setup 20250101 20251231
#   ./rocoto_controller.sh start
#   ./rocoto_controller.sh status
#   ./rocoto_controller.sh stop
#
# Author: NOAA/NESDIS Team
# Date: May 28, 2025
###############################################################################

# Default values
WORKFLOW_XML="workflow_auroragcafs.xml"
DATABASE="workflow_auroragcafs.db"
SCRATCH_BASE="${SCRATCH:-/scratch/$USER}"
WORKFLOW_DIR="${PWD}"
START_DATE="20250101"
END_DATE="20251231"
INTERVAL_DAYS=30

# Function to display help
show_help() {
    echo "Rocoto Workflow Controller for AuroraGCAFS"
    echo ""
    echo "Usage:"
    echo "  $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  setup [start_date] [end_date] [interval_days] - Set up directories and prepare workflow"
    echo "  start                                         - Start or resume the workflow"
    echo "  status                                        - Show workflow status"
    echo "  stop                                          - Stop the workflow"
    echo "  rewind [task] [cycle]                         - Rewind a task to run again"
    echo "  clean                                         - Clean workflow logs and database"
    echo "  help                                          - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup 20250101 20251231 30"
    echo "  $0 start"
    echo "  $0 status"
    echo "  $0 rewind continual_learning 202503010000"
    echo "  $0 stop"
}

# Function to set up workflow directories and files
setup_workflow() {
    # Parse arguments
    if [ $# -ge 1 ]; then
        START_DATE=$1
    fi
    if [ $# -ge 2 ]; then
        END_DATE=$2
    fi
    if [ $# -ge 3 ]; then
        INTERVAL_DAYS=$3
    fi

    echo "Setting up AuroraGCAFS workflow with:"
    echo "  Start Date: $START_DATE"
    echo "  End Date: $END_DATE"
    echo "  Interval: $INTERVAL_DAYS days"

    # Create required directories
    RUNDIR="${SCRATCH_BASE}/auroragcafs"
    mkdir -p ${RUNDIR}/{logs,models,training,evaluation,metrics}
    echo "Created directories at ${RUNDIR}"

    # Copy workflow file to a working copy
    WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_active.xml"
    cp ${WORKFLOW_DIR}/${WORKFLOW_XML} ${WORKING_XML}

    # Update XML with user-specific settings
    sed -i "s/@USER@/$USER/g" ${WORKING_XML}

    # Update cycle definitions based on start/end dates and interval
    START_CYCLE="${START_DATE}0000"
    END_CYCLE="${END_DATE}0000"
    TRAINING_INTERVAL_HOURS=$((INTERVAL_DAYS * 24))
    EVAL_INTERVAL_HOURS=$((INTERVAL_DAYS * 24))
    METRICS_INTERVAL_HOURS=$((INTERVAL_DAYS * 48)) # Less frequent metrics updates

    # Update cycledefs in the XML - handle multiple cycle definitions
    sed -i "s/<cycledef group=\"training\">.*<\/cycledef>/<cycledef group=\"training\">$START_CYCLE $END_CYCLE 00:00:$TRAINING_INTERVAL_HOURS:00:00<\/cycledef>/g" ${WORKING_XML}
    sed -i "s/<cycledef group=\"evaluation\">.*<\/cycledef>/<cycledef group=\"evaluation\">$START_CYCLE $END_CYCLE 00:00:$EVAL_INTERVAL_HOURS:00:00<\/cycledef>/g" ${WORKING_XML}
    sed -i "s/<cycledef group=\"metrics\">.*<\/cycledef>/<cycledef group=\"metrics\">$START_CYCLE $END_CYCLE 00:01:$METRICS_INTERVAL_HOURS:00:00<\/cycledef>/g" ${WORKING_XML}

    echo "Workflow configuration updated. Ready to start with:"
    echo "  rocoto_controller.sh start"
}

# Function to start or resume the workflow
start_workflow() {
    echo "Starting/resuming AuroraGCAFS workflow..."

    # Check if working XML exists
    WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_active.xml"
    if [ ! -f ${WORKING_XML} ]; then
        echo "ERROR: Workflow configuration not found at ${WORKING_XML}"
        echo "Please run 'rocoto_controller.sh setup' first"
        return 1
    fi

    # Start the workflow
    rocotorun -w ${WORKING_XML} -d ${WORKFLOW_DIR}/${DATABASE} -v 10

    # Set up a cron job to automatically run the workflow
    (crontab -l 2>/dev/null; echo "*/10 * * * * cd ${WORKFLOW_DIR} && rocotorun -w ${WORKING_XML} -d ${WORKFLOW_DIR}/${DATABASE} >> ${SCRATCH_BASE}/auroragcafs/logs/rocoto_cron.log 2>&1") | crontab -

    echo "Workflow started and cron job set to run every 10 minutes"
    echo "Check status with: rocoto_controller.sh status"
}

# Function to show workflow status
show_status() {
    WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_active.xml"
    if [ ! -f ${WORKING_XML} ]; then
        echo "ERROR: Workflow configuration not found at ${WORKING_XML}"
        return 1
    fi

    echo "AuroraGCAFS Workflow Status:"
    rocotostat -w ${WORKING_XML} -d ${WORKFLOW_DIR}/${DATABASE} -v 10
}

# Function to stop the workflow
stop_workflow() {
    echo "Stopping AuroraGCAFS workflow..."

    # Remove cron job
    crontab -l | grep -v "rocotorun -w ${WORKFLOW_DIR}" | crontab -

    # Check if working XML exists
    WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_active.xml"
    if [ ! -f ${WORKING_XML} ]; then
        echo "ERROR: Workflow configuration not found at ${WORKING_XML}"
        return 1
    fi

    # Cancel all pending jobs
    running_jobs=$(rocotostat -w ${WORKING_XML} -d ${WORKFLOW_DIR}/${DATABASE} | grep "RUNNING\|QUEUED" | awk '{print $2}' | cut -d '.' -f1)

    if [ ! -z "$running_jobs" ]; then
        echo "Cancelling running jobs: $running_jobs"
        for job in $running_jobs; do
            scancel $job
        done
    fi

    echo "Workflow stopped"
}

# Function to rewind a task to run again
rewind_task() {
    if [ $# -lt 2 ]; then
        echo "ERROR: Missing task name or cycle"
        echo "Usage: $0 rewind <task_name> <cycle>"
        echo "Example: $0 rewind continual_learning 202503010000"
        return 1
    fi

    TASK=$1
    CYCLE=$2

    WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_active.xml"
    if [ ! -f ${WORKING_XML} ]; then
        echo "ERROR: Workflow configuration not found at ${WORKING_XML}"
        return 1
    fi

    echo "Rewinding task '$TASK' for cycle '$CYCLE'..."
    rocotorewind -w ${WORKING_XML} -d ${WORKFLOW_DIR}/${DATABASE} -c $CYCLE -t $TASK -v 10

    echo "Task rewound. Run 'rocoto_controller.sh start' to resume workflow."
}

# Function to clean workflow artifacts
clean_workflow() {
    echo "Cleaning AuroraGCAFS workflow..."

    # Remove cron job
    crontab -l | grep -v "rocotorun -w ${WORKFLOW_DIR}" | crontab -

    # Remove database and working XML
    rm -f ${WORKFLOW_DIR}/${DATABASE}
    rm -f ${WORKFLOW_DIR}/workflow_auroragcafs_active.xml

    echo "Workflow cleaned"
}

# Main execution
if [ $# -lt 1 ]; then
    show_help
    exit 0
fi

COMMAND=$1
shift

case $COMMAND in
    setup)
        setup_workflow "$@"
        ;;
    start)
        start_workflow
        ;;
    status)
        show_status
        ;;
    stop)
        stop_workflow
        ;;
    rewind)
        rewind_task "$@"
        ;;
    clean)
        clean_workflow
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
