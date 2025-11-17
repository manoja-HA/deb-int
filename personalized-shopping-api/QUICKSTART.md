# Quick Start Guide

Get the Personalized Shopping API running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Ollama running on your host (recommended) or will use Docker Ollama
- 8GB+ RAM available
- 10GB+ disk space

## Option 1: Quick Start (Recommended)

Run the quick start script:

```bash
./quickstart.sh
```

This will:
1. Check Docker is running
2. Pull required Ollama models (if Ollama is on host)
3. Start all services with docker-compose
4. Wait for services to be ready
5. Display API URLs

## Option 2: Manual Setup

### Step 1: Ensure Ollama is Running

If you have Ollama on your host:

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull required models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

### Step 2: Start Services

```bash
docker-compose up -d
```

### Step 3: Wait for Services

```bash
# Watch logs
docker-compose logs -f api

# Wait for "Application startup complete"
```

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations for Kenneth Martinez
curl -X POST http://localhost:8000/api/v1/recommendations/personalized \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What else would Kenneth Martinez like based on his purchase history?",
    "customer_name": "Kenneth Martinez",
    "top_n": 5,
    "include_reasoning": true
  }'
```

## Accessing Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | Main API endpoint |
| Swagger Docs | http://localhost:8000/api/v1/docs | Interactive API documentation |
| ReDoc | http://localhost:8000/api/v1/redoc | Alternative API documentation |
| Prometheus | http://localhost:9090 | Metrics and monitoring |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Redis | localhost:6379 | Cache (internal) |
| Ollama | http://localhost:11434 | LLM service |

## Example Usage

### 1. Get Recommendations

```bash
curl -X POST http://localhost:8000/api/v1/recommendations/personalized \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What would Kenneth Martinez like?",
    "customer_name": "Kenneth Martinez"
  }'
```

### 2. Get Customer Profile

```bash
curl http://localhost:8000/api/v1/customers/887/profile
```

### 3. Find Similar Customers

```bash
curl 'http://localhost:8000/api/v1/customers/887/similar?top_k=10'
```

### 4. Get Product Reviews

```bash
curl http://localhost:8000/api/v1/products/291/reviews
```

## Multi-Agent Workflow

The API uses 5 specialized agents:

1. **Customer Profiling Agent**: Analyzes customer purchase history
2. **Similar Customer Discovery Agent**: Finds customers with similar behavior (vector similarity)
3. **Review-Based Filtering Agent**: Filters products by sentiment analysis
4. **Cross-Category Recommendation Agent**: Scores products using collaborative filtering
5. **Response Generation Agent**: Creates natural language explanations

## Stopping Services

```bash
docker-compose down
```

To also remove volumes:

```bash
docker-compose down -v
```

## Troubleshooting

### Vector index not found

The index is built automatically on first startup. If it fails:

```bash
# Enter the container
docker exec -it shopping-api bash

# Build index manually
python scripts/build_vector_index.py
```

### Ollama connection errors

If using host Ollama, ensure it's accessible from Docker:

```bash
# On Linux, Ollama listens on localhost by default
# Update docker-compose.yml:
# OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Out of memory

Reduce the number of workers or model sizes:

```bash
# In docker-compose.yml, reduce:
# WORKERS: 1
# Or use smaller models:
# RECOMMENDATION_MODEL=llama3.2:1b
```

## Development Mode

To run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector index
python scripts/build_vector_index.py

# Start API
uvicorn app.main:app --reload

# Access at http://localhost:8000
```

## Next Steps

- Explore API docs: http://localhost:8000/api/v1/docs
- View metrics: http://localhost:9090
- Add your own customer data in `data/raw/`
- Customize agent parameters in `.env`

## Support

- Check logs: `docker-compose logs -f`
- View health: http://localhost:8000/health
- Check metrics: http://localhost:8000/metrics
