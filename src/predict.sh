#!/usr/bin/env bash
set -e
set -v

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ ! -f "$PROJECT_ROOT/work/n_gram_lm.checkpoint" ]; then
  echo "Missing work/n_gram_lm.checkpoint. Run training first: PYTHONPATH=src python3 -m main train --work_dir work" >&2
  exit 1
fi

PYTHONPATH="$PROJECT_ROOT/src" python3 -m main test --work_dir "$PROJECT_ROOT/work" --test_data "$1" --test_output "$2"
