#!/bin/bash
set -e

echo "=== Personalized Shopping API Setup ==="

# Check if running in Docker
if [ -f /.dockerenv ]; then
    echo "Running in Docker container"
    IN_DOCKER=true
else
    echo "Running on host system"
    IN_DOCKER=false
fi

# Create directories
echo "Creating necessary directories..."
mkdir -p data/raw data/embeddings logs

# Check if CSV data files exist
if [ ! -f "data/raw/customer_purchase_data.csv" ]; then
    echo "ERROR: customer_purchase_data.csv not found!"
    echo "Please add your data files to data/raw/"
    exit 1
fi

if [ ! -f "data/raw/customer_reviews_data.csv" ]; then
    echo "ERROR: customer_reviews_data.csv not found!"
    echo "Please add your data files to data/raw/"
    exit 1
fi

echo "Data files found ✓"

# Build vector index if it doesn't exist
if [ ! -f "data/embeddings/customer_index.faiss" ]; then
    echo "Building vector index..."
    python scripts/build_vector_index.py
    
    if [ $? -eq 0 ]; then
        echo "Vector index built successfully ✓"
    else
        echo "ERROR: Failed to build vector index"
        exit 1
    fi
else
    echo "Vector index already exists ✓"
fi

# Check Ollama connection (only if not in Docker)
if [ "$IN_DOCKER" = false ]; then
    echo "Checking Ollama connection..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama is running ✓"
        
        # Check if models are available
        if ollama list | grep -q "llama3.2:3b"; then
            echo "llama3.2:3b model found ✓"
        else
            echo "Pulling llama3.2:3b..."
            ollama pull llama3.2:3b
        fi
        
        if ollama list | grep -q "llama3.1:8b"; then
            echo "llama3.1:8b model found ✓"
        else
            echo "Pulling llama3.1:8b..."
            ollama pull llama3.1:8b
        fi
    else
        echo "WARNING: Ollama is not running at http://localhost:11434"
        echo "Please start Ollama with: ollama serve"
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the API:"
echo "  Development: uvicorn app.main:app --reload"
echo "  Production:  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4"
echo "  Docker:      docker-compose up -d"
echo ""
echo "API Documentation:"
echo "  Swagger UI: http://localhost:8000/api/v1/docs"
echo "  ReDoc:      http://localhost:8000/api/v1/redoc"
echo ""
