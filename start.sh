#!/bin/bash
set -e
echo "Guardian AI v8.0 Starting..."
mkdir -p /app/models /app/data /app/logs
MODEL_PATH="${MODEL_PATH:-/app/models/guardian_model.pkl}"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Training ML model (~2 min)..."
    cd /app && python3 train.py --dataset synthetic --samples 30000
    echo "Model ready!"
else
    echo "Model found!"
fi
cd /app
exec uvicorn api:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1 --log-level info
