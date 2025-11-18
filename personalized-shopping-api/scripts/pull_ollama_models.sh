#!/bin/bash
# Pull required Ollama models for CPU-only operation
# Run this script after starting docker-compose

set -e

echo "🚀 Pulling Ollama models for CPU-only operation..."
echo ""

# Configuration
OLLAMA_CONTAINER="shopping-ollama"
MODELS=("llama3.2:3b" "llama3.1:8b")

# Check if Ollama container is running
if ! docker ps | grep -q $OLLAMA_CONTAINER; then
    echo "❌ Error: Ollama container '$OLLAMA_CONTAINER' is not running"
    echo "   Please run: docker-compose up -d"
    exit 1
fi

echo "✓ Ollama container is running"
echo ""

# Pull each model
for model in "${MODELS[@]}"; do
    echo "📥 Pulling $model..."
    docker exec $OLLAMA_CONTAINER ollama pull $model

    if [ $? -eq 0 ]; then
        echo "✓ Successfully pulled $model"
    else
        echo "❌ Failed to pull $model"
        exit 1
    fi
    echo ""
done

# List all available models
echo "📋 Available models:"
docker exec $OLLAMA_CONTAINER ollama list

echo ""
echo "✅ All models pulled successfully!"
echo ""
echo "📊 Disk usage:"
docker exec $OLLAMA_CONTAINER du -sh /root/.ollama/models

echo ""
echo "Next steps:"
echo "  1. Restart the API to load models: docker-compose restart api"
echo "  2. Test the API: curl http://localhost:8002/health"
echo "  3. Monitor logs: docker-compose logs -f api"
