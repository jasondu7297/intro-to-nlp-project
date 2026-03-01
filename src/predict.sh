#!/usr/bin/env bash
set -e
set -v

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHONPATH="$PROJECT_ROOT/src" python3 -m main test --work_dir "$PROJECT_ROOT/work" --test_data "$1" --test_output "$2"
