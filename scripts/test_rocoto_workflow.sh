#!/bin/bash
# filepath: /Users/l22-n04127-res/Documents/GitHub/AuroraGCAFS/scripts/test_rocoto_workflow.sh
###############################################################################
# Test script for AuroraGCAFS Rocoto Workflow
#
# This script validates the Rocoto XML workflow configuration and dependencies
# without actually submitting jobs.
#
# Usage:
#   ./test_rocoto_workflow.sh
#
# Author: NOAA/NESDIS Team
# Date: May 28, 2025
###############################################################################

# Set default paths
WORKFLOW_XML="workflow_auroragcafs.xml"
WORKFLOW_DIR="${PWD}"
WORKING_XML="${WORKFLOW_DIR}/workflow_auroragcafs_test.xml"
DATABASE="${WORKFLOW_DIR}/workflow_test.db"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== AuroraGCAFS Rocoto Workflow Test ===${NC}"

# Check if XML file exists
if [ ! -f "${WORKFLOW_DIR}/${WORKFLOW_XML}" ]; then
    echo -e "${RED}ERROR: Workflow XML not found at ${WORKFLOW_DIR}/${WORKFLOW_XML}${NC}"
    exit 1
fi

echo "Creating test configuration..."

# Create a test copy of the workflow XML
cp "${WORKFLOW_DIR}/${WORKFLOW_XML}" "${WORKING_XML}"

# Update user in XML
sed -i "s/@USER@/${USER}/g" "${WORKING_XML}"

# Create test cycle definitions (a single short cycle)
TEST_START="202501010000"
TEST_END="202501020000"
sed -i "s/<cycledef group=\"training\">.*<\/cycledef>/<cycledef group=\"training\">$TEST_START $TEST_END 00:00:24:00:00<\/cycledef>/g" "${WORKING_XML}"
sed -i "s/<cycledef group=\"evaluation\">.*<\/cycledef>/<cycledef group=\"evaluation\">$TEST_START $TEST_END 00:00:24:00:00<\/cycledef>/g" "${WORKING_XML}"
sed -i "s/<cycledef group=\"metrics\">.*<\/cycledef>/<cycledef group=\"metrics\">$TEST_START $TEST_END 00:00:24:00:00<\/cycledef>/g" "${WORKING_XML}"

# Add test mode to the XML to prevent actual job submission
sed -i "s/<workflow realtime=\"false\" scheduler=\"slurm\" cyclethrottle=\"1\">/<workflow realtime=\"false\" scheduler=\"slurm\" cyclethrottle=\"1\" testonly=\"true\">/g" "${WORKING_XML}"

echo "Validating XML syntax..."

# Check XML syntax
xmllint --noout "${WORKING_XML}" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: XML validation failed${NC}"
    echo "Running xmllint to show errors:"
    xmllint "${WORKING_XML}"
    exit 1
else
    echo -e "${GREEN}XML syntax is valid${NC}"
fi

echo "Testing workflow configuration..."

# Clean up any existing test database
rm -f "${DATABASE}"

# Run rocotorun in test mode
rocotorun -w "${WORKING_XML}" -d "${DATABASE}" -v 10 -t

if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Rocoto workflow validation failed${NC}"
    exit 1
else
    echo -e "${GREEN}Workflow configuration is valid${NC}"
fi

echo "Testing task dependencies..."

# Check task dependencies
rocotostat -w "${WORKING_XML}" -d "${DATABASE}" -v 10

echo -e "\nChecking for potential issues:"

# Check for common issues
echo "1. Checking paths and entities..."
grep -n "&RUNDIR;" "${WORKING_XML}" | grep -v "cyclestr"
if [ $? -eq 0 ]; then
    echo -e "${YELLOW}WARNING: Found &RUNDIR; references without cyclestr wrapper${NC}"
fi

echo "2. Checking job resources..."
grep -n "nodes=" "${WORKING_XML}"
echo "   (Verify that node specifications are correct for your HPC system)"

echo "3. Checking dependencies..."
grep -n "<dependency>" "${WORKING_XML}" -A 5

echo -e "\n${GREEN}Test completed. Review any warnings above.${NC}"
echo "To deploy the workflow, run: ./rocoto_controller.sh setup <start_date> <end_date> <interval>"

# Clean up
echo "Cleaning up test files..."
rm -f "${WORKING_XML}" "${DATABASE}"
