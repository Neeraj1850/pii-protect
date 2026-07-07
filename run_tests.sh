#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Setup colors for console output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================================================${NC}"
echo -e "${BLUE}                   pii-protect Test Runner Script                     ${NC}"
echo -e "${BLUE}======================================================================${NC}"

# Check python and pip installation
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed.${NC}"
    exit 1
fi

# Determine python command
PYTHON="python3"

# Verify pip
if ! $PYTHON -m pip --version &> /dev/null; then
    echo -e "${YELLOW}Warning: pip is not available via python module. Trying global pip...${NC}"
    if ! command -v pip &> /dev/null; then
        echo -e "${RED}Error: pip is not installed.${NC}"
        exit 1
    fi
fi

# Install dependencies if requested
if [[ "$1" == "--install" || "$1" == "-i" ]]; then
    echo -e "${YELLOW}Installing/Updating project dependencies...${NC}"
    $PYTHON -m pip install -e ".[dev]"
    # Download Spacy model if spacy extra is installed
    if $PYTHON -c "import spacy" &> /dev/null; then
        echo -e "${YELLOW}Downloading spaCy model en_core_web_sm...${NC}"
        $PYTHON -m spacy download en_core_web_sm || echo -e "${YELLOW}Warning: Failed to download spaCy model. Some NER tests may be skipped.${NC}"
    fi
fi

# Test path selection. Defaults to the whole suite; a *-only flag narrows
# TARGET_PATHS to one directory. --llm-live opts IN to the live Groq API
# tests (real network calls, needs GROQ_API_KEY) since they're excluded by
# default to keep a plain ./run_tests.sh run offline and quota-free.
TARGET_PATHS="tests"
INCLUDE_LLM_LIVE=false
RUN_BENCHMARK=false
GEN_HTML=true
GEN_COV=true

# Parse remaining arguments
shift $((OPTIND - 1))
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --unit-only) TARGET_PATHS="tests/unit" ;;
        --integration-only) TARGET_PATHS="tests/integration" ;;
        --detection-only) TARGET_PATHS="tests/detection" ;;
        --edge-only) TARGET_PATHS="tests/edge_cases" ;;
        --dataset-only) TARGET_PATHS="tests/dataset" ;;
        --llm-only) TARGET_PATHS="tests/llm" ;;
        --llm-live-only) TARGET_PATHS="tests/llm/live"; INCLUDE_LLM_LIVE=true ;;
        --postgres-only) TARGET_PATHS="tests/unit/storage/test_postgres.py tests/integration/test_postgres_engine.py" ;;
        --redis-only) TARGET_PATHS="tests/unit/storage/test_redis.py tests/integration/test_redis_engine.py" ;;
        --llm-live) INCLUDE_LLM_LIVE=true ;;
        --benchmark) RUN_BENCHMARK=true; TARGET_PATHS="tests/benchmarks" ;;
        --no-html) GEN_HTML=false ;;
        --no-cov) GEN_COV=false ;;
        *) echo -e "${YELLOW}Unknown option: $1. Running all tests.${NC}" ;;
    esac
    shift
done

# Build pytest flags
PYTEST_FLAGS=""
if [ "$GEN_HTML" = true ]; then
    PYTEST_FLAGS="$PYTEST_FLAGS --html=reports/report.html --self-contained-html"
fi

if [ "$GEN_COV" = true ]; then
    PYTEST_FLAGS="$PYTEST_FLAGS --cov=pii_protect --cov-report=html:reports/coverage --cov-report=term"
fi

# Create reports directory
mkdir -p reports

# Execute pytest
echo -e "${GREEN}Running selected test suites: ${TARGET_PATHS}${NC}"
pytest_cmd="pytest $PYTEST_FLAGS $TARGET_PATHS"

# If benchmarking, run benchmark only; otherwise exclude benchmark markers,
# and exclude live Groq tests unless explicitly opted into.
if [ "$RUN_BENCHMARK" = true ]; then
    echo -e "${YELLOW}Running performance benchmarks...${NC}"
    pytest_cmd="$pytest_cmd --benchmark-only"
elif [ "$INCLUDE_LLM_LIVE" = true ]; then
    pytest_cmd="$pytest_cmd -m 'not benchmark'"
else
    pytest_cmd="$pytest_cmd -m 'not benchmark and not llm_integration'"
fi

# Run the command
echo -e "${BLUE}Executing: $pytest_cmd${NC}"
eval $pytest_cmd

echo -e "${GREEN}======================================================================${NC}"
echo -e "${GREEN}Tests Completed Successfully!${NC}"
if [ "$GEN_HTML" = true ]; then
    echo -e "HTML Report generated: ${YELLOW}reports/report.html${NC}"
fi
if [ "$GEN_COV" = true ]; then
    echo -e "Coverage Report generated: ${YELLOW}reports/coverage/index.html${NC}"
fi
echo -e "${GREEN}======================================================================${NC}"
