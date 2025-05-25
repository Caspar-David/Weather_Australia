#!/bin/sh
# filepath: wait-for-files-api.sh

set -e
while [ ! -f data/processed/xgboost_model.pkl ]; do
  echo "Waiting for data/processed/xgboost_model.pkl..."
  sleep 2
done
exec uvicorn src.api.basicAPI:app --host 0.0.0.0 --port 8000