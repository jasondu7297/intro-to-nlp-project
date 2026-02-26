#!/usr/bin/env bash
set -x
set -e

rm -rf submit submit.zip
mkdir -p submit

# submit team.txt
printf "Victor Zhong,vzhng\nStudent Name,WaterlooID\nStudent Name,WaterlooID" > submit/team.txt

# train model
PYTHONPATH=src python3 -m main train --work_dir work

# make predictions on open-dev data and submit it in pred.txt
PYTHONPATH=src python3 -m main test --work_dir work --test_data data/raw/open-dev/input.txt --test_output submit/pred.txt

# submit docker file
cp Dockerfile submit/Dockerfile

# submit source code
cp -r src submit/src

# submit checkpoints
cp -r work submit/work

# make zip file
zip -r submit.zip submit
