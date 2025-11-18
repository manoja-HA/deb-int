# Quick Start Guide - Shopping Assistant with LangFuse

## Prerequisites

- Docker and Docker Compose installed
- Ollama running locally on port 11434
- At least 8GB RAM available for containers

## 🚀 Start Everything

```bash
# Start all services
docker-compose up -d

# Check services are running
docker-compose ps
```

You should see:
- ✅ `langfuse-db` (healthy) - PostgreSQL database
- ✅ `langfuse` - LangFuse observability platform
- ✅ `shopping-api` (healthy) - Your FastAPI application
- ✅ `shopping-prometheus` - Metrics collection
- ✅ `shopping-grafana` - Metrics visualization

## 📊 Access the Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Shopping API** | http://localhost:8002 | Main API |
| **API Docs** | http://localhost:8002/api/v1/docs | Swagger UI |
| **LangFuse UI** | http://localhost:3002 | LLM Observability |
| **Grafana** | http://localhost:3001 | Metrics Dashboard |
| **Prometheus** | http://localhost:9091 | Metrics Storage |

## 🔧 Set Up LangFuse

### 1. Open LangFuse UI
Visit: http://localhost:3002

### 2. Create Account
- Click **"Sign Up"**
- Email: `admin@localhost`
- Password: Choose a password
- Click **"Create Account"**

### 3. Create Project
- Click **"New Project"** or **"+"**
- Name: `shopping-assistant`
- Click **"Create"**

### 4. Get API Keys
- Go to **Project Settings** (gear icon)
- Navigate to **"API Keys"** tab
- Click **"Create new API keys"**
- **Copy both keys** (you can't view the secret key again!)

### 5. Configure Application

Add the keys to your `.env` file:

```bash
# Update these lines in .env
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 6. Restart API

```bash
docker-compose restart api
```

## 🧪 Test the Setup

### Test Informational Query

```bash
curl -X POST "http://localhost:8002/api/v1/recommendations/personalized" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Kenneth Martinez",
    "query": "what is the total purchase of Kenneth Martinez?",
    "include_reasoning": true,
    "top_n": 5
  }'
```

### Test Recommendation Query

```bash
curl -X POST "http://localhost:8002/api/v1/recommendations/personalized" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Kenneth Martinez",
    "query": "recommend some products for me",
    "include_reasoning": true,
    "top_n": 5
  }'
```

### View Traces

1. Go to http://localhost:3002
2. Navigate to your project
3. Click **"Traces"** in the sidebar
4. You'll see your API requests with full details!

## 📈 What Gets Traced

Every request traces:
- **Intent Classification** - Query analysis and routing
- **Customer Profiling** - Profile retrieval and analysis
- **Similar Customer Discovery** - Vector similarity search
- **Sentiment Filtering** - Review analysis and filtering
- **Recommendation Generation** - Scoring and ranking
- **Response Generation** - LLM-based explanations

## 🛑 Stop Everything

```bash
docker-compose down
```

To also remove data:
```bash
docker-compose down -v
```

## 📚 More Information

- [Full LangFuse Setup Guide](docs/LANGFUSE_SETUP.md)
- [LangFuse Integration Docs](docs/LANGFUSE_INTEGRATION.md)
- [API Documentation](http://localhost:8002/api/v1/docs)

## 🐛 Troubleshooting

### Services won't start
```bash
# Check logs
docker-compose logs langfuse
docker-compose logs api

# Restart services
docker-compose restart
```

### API can't connect to Ollama
Make sure Ollama is running on your host:
```bash
ollama list
curl http://localhost:11434/api/tags
```

### No traces in LangFuse
1. Check API keys are correct in `.env`
2. Verify `LANGFUSE_ENABLED=true`
3. Restart API: `docker-compose restart api`
4. Check logs: `docker logs shopping-api | grep -i langfuse`

## 💡 Tips

- Use `docker-compose logs -f api` to watch API logs in real-time
- LangFuse data persists in Docker volumes
- First request may be slow while models load in Ollama
- Check http://localhost:3001 for Grafana dashboards

---

**Ready to start?** Run `docker-compose up -d` and follow the steps above!
