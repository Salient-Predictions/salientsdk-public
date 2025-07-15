#!/bin/bash
set -e

# Script to test GitHub Actions workflows locally using act
#
# USAGE:
#   First time setup:
#     chmod +x scripts/test-workflows.sh
#
#   Run from project root:
#     ./scripts/test-workflows.sh                    # Test default release.yaml workflow
#     ./scripts/test-workflows.sh --workflow all     # Test all workflows
#     ./scripts/test-workflows.sh --workflow my.yaml # Test specific workflow
#     ./scripts/test-workflows.sh --help             # Show help
#
# NOTE: Always runs in dry-run mode for safety - no real deployments will occur

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_ROOT/tests/.env.test"

echo "📂 Script directory: $SCRIPT_DIR"
echo "📂 Project root: $PROJECT_ROOT"

# Find act binary - check local bin first, then system PATH
ACT_CMD=""
if [ -x "$PROJECT_ROOT/bin/act" ]; then
    ACT_CMD="$PROJECT_ROOT/bin/act"
    echo "✅ Using local act: $ACT_CMD"
elif command -v act &> /dev/null; then
    ACT_CMD="act"
    echo "✅ Using system act"
else
    echo "🔧 act not found, installing..."
    if command -v curl &> /dev/null; then
        curl -s https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash
        
        # Check if it was installed locally or system-wide
        if [ -x "$PROJECT_ROOT/bin/act" ]; then
            ACT_CMD="$PROJECT_ROOT/bin/act"
            echo "✅ act installed locally: $ACT_CMD"
        elif command -v act &> /dev/null; then
            ACT_CMD="act"
            echo "✅ act installed system-wide"
        else
            echo "❌ act installation failed"
            exit 1
        fi
    else
        echo "❌ curl not found. Please install act manually:"
        echo "   Visit: https://github.com/nektos/act#installation"
        exit 1
    fi
fi

# Create test environment file if it doesn't exist
if [ ! -f "$ENV_FILE" ]; then
    echo "Creating test environment file: $ENV_FILE"
    mkdir -p "$(dirname "$ENV_FILE")"
    cat > "$ENV_FILE" << EOF
PYPI_TOKEN=fake-token
ACTIONS_APP_ID=12345
ACTIONS_APP_PRIVATE_KEY=fake-key
GITHUB_TOKEN=fake-github-token
EOF
fi

# Default workflow
WORKFLOW="release.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workflow)
            WORKFLOW="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--workflow <workflow.yaml|all>]"
            echo "  --workflow FILE    Specify workflow file (default: release.yaml)"
            echo "  --workflow all     Test all workflows in .github/workflows/"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

cd "$PROJECT_ROOT"

# Function to test a single workflow
test_workflow() {
    local workflow_file="$1"
    echo "🧪 Testing workflow: $workflow_file"
    echo "📁 Environment file: $ENV_FILE"
    echo

    # Try different event types to find one that works
    local events=("push" "pull_request" "workflow_dispatch")
    local success=false
    
    for event in "${events[@]}"; do
        if $ACT_CMD $event \
            -W ".github/workflows/$workflow_file" \
            --env-file "$ENV_FILE" \
            -P ubuntu-latest=catthehacker/ubuntu:act-latest \
            --dryrun > /dev/null 2>&1; then
            
            echo "Using event: $event"
            echo
            
            # Run it with full output
            $ACT_CMD $event \
                -W ".github/workflows/$workflow_file" \
                --env-file "$ENV_FILE" \
                -P ubuntu-latest=catthehacker/ubuntu:act-latest \
                --dryrun
            
            success=true
            break
        fi
    done
    
    if [ "$success" = false ]; then
        echo "  ⚠️  Could not find a working event type for $workflow_file"
        return 1
    fi

    echo
    echo "✅ Workflow $workflow_file completed successfully!"
    echo
}

# Handle "all" workflows or single workflow
if [ "$WORKFLOW" = "all" ]; then
    echo "🔍 Discovering all workflows in .github/workflows/"
    
    # Find all .yaml and .yml files
    WORKFLOW_FILES=$(find .github/workflows/ -name "*.yaml" -o -name "*.yml" | sort)
    
    if [ -z "$WORKFLOW_FILES" ]; then
        echo "❌ No workflow files found in .github/workflows/"
        exit 1
    fi
    
    echo "Found workflows:"
    echo "$WORKFLOW_FILES" | sed 's/^/  - /'
    echo
    
    # Test each workflow
    for workflow_path in $WORKFLOW_FILES; do
        workflow_name=$(basename "$workflow_path")
        test_workflow "$workflow_name"
    done
    
    echo "🎉 All workflows tested successfully!"
else
    # Test single workflow
    test_workflow "$WORKFLOW"
fi 