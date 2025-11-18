# Quick Start Guide: Ollama with CPU-Only Models

## Prerequisites
- 16GB+ RAM
- 4+ CPU cores
- Docker & Docker Compose installed

## 1. Pull Required Models

```bash
# Pull the 3B model (smaller, faster)
docker exec shopping-ollama ollama pull llama3.2:3b

# Pull the 8B model (larger, more capable)
docker exec shopping-ollama ollama pull llama3.1:8b
```

**Note**: First-time pull may take 10-30 minutes depending on internet speed.

## 2. Verify Models are Available

```bash
# List all pulled models
docker exec shopping-ollama ollama list

# Expected output:
# NAME              ID              SIZE      MODIFIED
# llama3.2:3b       a80c4f17acd5    2.0 GB    2 minutes ago
# llama3.1:8b       91ab477bec9d    4.7 GB    5 minutes ago
```

## 3. Start the Stack

```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps

# Expected: All services should show "Up" and "healthy"
```

## 4. Test Ollama is Working

```bash
# Test 3B model (should respond in 20-30 seconds)
curl http://localhost:11435/api/generate -d '{
  "model": "llama3.2:3b",
  "prompt": "Hello!",
  "stream": false
}'

# Test 8B model (should respond in 60-90 seconds)
curl http://localhost:11435/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello!",
  "stream": false
}'
```

## 5. Test API Endpoint

```bash
# Health check
curl http://localhost:8002/health

# Test recommendation endpoint (will be slow first time)
curl -X POST http://localhost:8002/api/v1/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "Kenneth Martinez",
    "query": "Recommend some products",
    "top_n": 3
  }'
```

**Note**: First request will take 90-150 seconds (cold start). Subsequent requests: 20-90 seconds.

## 6. Monitor Performance

```bash
# Watch Ollama container stats
docker stats shopping-ollama

# Watch Ollama logs
docker logs -f shopping-ollama

# Check which models are currently loaded in memory
curl http://localhost:11435/api/ps
```

## Common Issues & Solutions

### Issue: "Model not found"
**Solution**: Pull the model first:
```bash
docker exec shopping-ollama ollama pull llama3.2:3b
```

### Issue: Request times out after 120 seconds
**Solution**: Check CPU usage - if maxed out, reduce MAX_TOKENS or use only 3B models

### Issue: Out of memory
**Solution**:
```bash
# Check memory usage
docker stats

# Restart Ollama to clear memory
docker-compose restart ollama
```

### Issue: Models keep reloading (slow)
**Solution**: Verify OLLAMA_KEEP_ALIVE is set in docker-compose.yml (should be 10m)

## Performance Expectations

### Cold Start (First Request)
- Model loading: 30-60s
- Inference: 60-90s
- **Total: 90-150s**

### Warm Requests (Model in Memory)
- llama3.2:3b: 20-45s
- llama3.1:8b: 60-90s

### With Cache Hit (Redis)
- Response: < 1s

## Resource Usage (Normal Operation)

| Component | CPU | Memory | Notes |
|-----------|-----|--------|-------|
| Ollama | 60-80% | 8-12GB | 2 models loaded |
| API | 5-10% | 1-2GB | |
| Redis | < 5% | 500MB | |
| PostgreSQL | < 5% | 500MB | |
| LangFuse | < 5% | 500MB | |

## Next Steps

1. Enable Redis caching:
   ```bash
   # In .env
   ENABLE_CACHE=true
   ```

2. Monitor LangFuse for LLM observability:
   - Visit: http://localhost:3002
   - Login with default credentials

3. Review optimization guide:
   - See: [OLLAMA_CPU_OPTIMIZATION.md](OLLAMA_CPU_OPTIMIZATION.md)

## Useful Commands

```bash
# Restart just Ollama
docker-compose restart ollama

# View all models
docker exec shopping-ollama ollama list

# Remove unused models
docker exec shopping-ollama ollama rm model-name

# Check Ollama version
docker exec shopping-ollama ollama --version

# Force pull latest model version
docker exec shopping-ollama ollama pull llama3.2:3b --force
```

## Production Checklist

- [ ] Models pre-pulled (llama3.2:3b, llama3.1:8b)
- [ ] First test request successful (90-150s)
- [ ] Subsequent requests successful (20-90s)
- [ ] Memory usage stable (< 12GB)
- [ ] OLLAMA_KEEP_ALIVE working (models stay loaded)
- [ ] Redis caching enabled
- [ ] Monitoring configured (LangFuse)
- [ ] Resource limits set in docker-compose.yml
- [ ] Timeout set to 120s in config

---

For detailed optimization information, see: [OLLAMA_CPU_OPTIMIZATION.md](OLLAMA_CPU_OPTIMIZATION.md)
