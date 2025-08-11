#!/bin/bash

echo "📊 Monitoring merge progress..."
echo "================================"

while true; do
    # Get current time
    TIME=$(date +"%H:%M:%S")
    
    # Check file size
    if [ -f "data/large_dataset_with_embeddings_merged.parquet" ]; then
        SIZE=$(ls -lh data/large_dataset_with_embeddings_merged.parquet | awk '{print $5}')
    else
        SIZE="Not created"
    fi
    
    # Check memory usage of python process
    PID=$(pgrep -f "merge_chunks_memory_efficient.py")
    if [ ! -z "$PID" ]; then
        # Get memory in GB
        MEM_KB=$(ps -o rss= -p $PID 2>/dev/null)
        if [ ! -z "$MEM_KB" ]; then
            MEM_GB=$(echo "scale=2; $MEM_KB / 1048576" | bc)
            STATUS="Running"
        else
            MEM_GB="N/A"
            STATUS="Finished/Crashed"
        fi
    else
        MEM_GB="N/A"
        STATUS="Not running"
    fi
    
    # Print status
    echo "[$TIME] Status: $STATUS | File: $SIZE | Memory: ${MEM_GB} GB"
    
    # Exit if process is done
    if [ "$STATUS" != "Running" ]; then
        echo "Process completed or crashed!"
        break
    fi
    
    # Wait 30 seconds
    sleep 30
done