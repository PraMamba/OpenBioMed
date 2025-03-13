#!/bin/bash

echo "server start"

# Activate the Conda environment
source /root/miniconda3/bin/activate OpenBioMed

python -m uvicorn open_biomed.scripts.run_server:app \
    --host 0.0.0.0 \
    --port 8082 \
    --log-level info > ./tmp/server.log 2>&1 &

python -m uvicorn open_biomed.scripts.run_server_workflow:app \
    --host 0.0.0.0 \
    --port 8083 \
    --log-level info > ./tmp/workflow.log 2>&1 &

tail -f /dev/null