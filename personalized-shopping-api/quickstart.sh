#!/bin/bash
set -e

echo "=== Personalized Shopping API - Quick Start ==="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

echo "Docker is running ✓"
echo ""

# Check if Ollama is running on host
echo "Checking if Ollama is running on host..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama detected on host ✓"
    echo ""
    echo "Pulling required models..."
    
    if ollama list | grep -q "llama3.2:3b"; then
        echo "  llama3.2:3b ✓"
    else
        echo "  Pulling llama3.2:3b..."
        ollama pull llama3.2:3b
    fi
    
    if ollama list | grep -q "llama3.1:8b"; then
        echo "  llama3.1:8b ✓"
    else
        echo "  Pulling llama3.1:8b..."
        ollama pull llama3.1:8b
    fi
else
    echo "WARNING: Ollama not detected on host"
    echo "The Docker container will use its own Ollama instance"
    echo "This may take longer as models need to be downloaded inside the container"
fi

echo ""
echo "Starting services with Docker Compose..."
echo ""

# Start services
docker-compose up -d

echo ""
echo "=== Services Starting ==="
echo ""
echo "Waiting for services to be healthy..."

# Wait for API to be ready
echo -n "Waiting for API"
for i in {1..60}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo " ✓"
        break
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "=== Services Ready! ==="
echo ""
echo "API:          http://localhost:8000"
echo "Docs:         http://localhost:8000/api/v1/docs"
echo "Prometheus:   http://localhost:9090"
echo "Grafana:      http://localhost:3000 (admin/admin)"
echo ""
echo "View logs:"
echo "  docker-compose logs -f api"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
echo "Test the API:"
echo "  curl -X POST http://localhost:8000/api/v1/recommendations/personalized \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"query\": \"What else would Kenneth Martinez like?\", \"customer_name\": \"Kenneth Martinez\"}'"
echo ""
