#!/bin/bash
# Convenient script: Extract numbered items from knowledge base

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root directory
cd "$PROJECT_ROOT"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <knowledge_base_name>"
    echo "Example: $0 hkuds-lab"
    exit 1
fi

KB_NAME="$1"

# Run extraction script
python src/knowledge/extract_numbered_items.py \
    --kb "$KB_NAME" \
    --base-dir ./data/knowledge_bases \
    "${@:2}"  # Pass other arguments
