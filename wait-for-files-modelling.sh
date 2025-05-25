#!/bin/sh
# filepath: wait-for-files-modelling.sh

set -e

for f in X_train.csv X_test.csv y_train.csv y_test.csv; do
  while [ ! -f data/processed/$f ]; do
    echo "Waiting for data/processed/$f..."
    sleep 2
  done
done

exec python src/pipelines/run_modelling.py