#!/usr/bin/env bash
set -e
set -v

if [ ! -f work/n_gram_lm.checkpoint ]; then
  echo "Missing work/n_gram_lm.checkpoint. Run training first: python3 src/n_gram_lm.py train --work_dir work" >&2
  exit 1
fi

python3 src/n_gram_lm.py test --work_dir work --test_data "$1" --test_output "$2"
