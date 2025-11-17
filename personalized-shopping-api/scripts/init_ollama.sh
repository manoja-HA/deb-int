#!/bin/bash
set -e

echo "=== Initializing Ollama Models ==="

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
    echo "Waiting for Ollama..."
done

echo "Ollama is ready!"

# Pull required models
echo "Pulling llama3.2:3b..."
ollama pull llama3.2:3b

echo "Pulling llama3.1:8b..."
ollama pull llama3.1:8b

echo "=== Ollama initialization complete ==="
