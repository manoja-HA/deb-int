#!/bin/bash
set -e

echo "=== Starting Personalized Shopping API ==="

# Check if ChromaDB collection exists
if [ ! -d "/app/data/embeddings/chroma" ]; then
    echo "ChromaDB collection not found. Building..."
    python /app/scripts/build_vector_index.py

    if [ $? -eq 0 ]; then
        echo "ChromaDB collection built successfully ✓"
    else
        echo "WARNING: Failed to build ChromaDB collection. API may not work correctly."
    fi
else
    echo "ChromaDB collection found ✓"
fi

# Start the application
echo "Starting FastAPI application..."
exec "$@"
