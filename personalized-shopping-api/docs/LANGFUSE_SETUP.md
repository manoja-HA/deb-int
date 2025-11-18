# LangFuse Local Setup Guide

This guide will help you set up LangFuse for LLM observability in your local development environment.

## Overview

LangFuse is now included in the Docker Compose setup and will automatically start with all its dependencies:
- **PostgreSQL** (port 5433) - Database for LangFuse
- **LangFuse UI** (port 3002) - Web interface for trace visualization
- **Shopping API** (port 8002) - Your FastAPI application with tracing enabled

## Quick Start

### 1. Start All Services

```bash
cd personalized-shopping-api
docker-compose up -d
```

This will start:
- PostgreSQL database for LangFuse
- LangFuse application
- Your FastAPI application (after LangFuse is healthy)
- Prometheus & Grafana (optional monitoring)

### 2. Run Setup Script (Optional)

For guided setup with instructions:

```bash
./scripts/setup-langfuse.sh
```

### 3. Access LangFuse UI

Open your browser and navigate to: **http://localhost:3002**

### 4. Create Your Account

On first access:
1. Click **"Sign Up"**
2. Enter credentials:
   - Email: `admin@localhost` (or any email)
   - Password: Choose a secure password
3. Click **"Create Account"**

### 5. Create a Project

After logging in:
1. Click **"New Project"** or the **"+"** button
2. Enter project details:
   - Name: `shopping-assistant`
   - Description: `Personalized Shopping Assistant API`
3. Click **"Create"**

### 6. Get API Keys

1. Navigate to **Project Settings** (gear icon)
2. Go to **"API Keys"** tab
3. Click **"Create new API keys"**
4. **Important**: Copy both keys immediately:
   - **Public Key**: `pk-lf-...`
   - **Secret Key**: `sk-lf-...`

### 7. Configure Your Application

Update your `.env` file with the API keys:

```bash
# LangFuse (LLM Observability)
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxxxxxxxxxxxxxxxxxxxxx
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_RELEASE=v1.0.0
LANGFUSE_ENVIRONMENT=development
LANGFUSE_DEBUG=false
```

### 8. Restart the API

```bash
docker-compose restart api
```

Wait for the API to be healthy:
```bash
docker logs -f shopping-api
```

Look for: `"Application startup completed successfully"`

## Testing the Integration

### 1. Send a Test Request

**Informational Query:**
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

**Recommendation Query:**
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

### 2. View Traces in LangFuse

1. Go to **http://localhost:3002**
2. Navigate to your project
3. Click **"Traces"** in the sidebar
4. You should see your API requests with full trace details

## What Gets Traced

The application automatically traces:

### 🎯 Intent Classification
- Query analysis
- Intent detection (informational vs recommendation)
- Confidence scores
- Classification reasoning

### 👤 Customer Profiling
- Profile retrieval
- Purchase history analysis
- Preference extraction

### 🔍 Similar Customer Discovery
- Vector similarity search
- Top similar customers
- Similarity scores

### 💭 Sentiment Filtering
- Review sentiment analysis
- Product filtering by sentiment threshold
- Filtered product counts

### 🎁 Recommendation Generation
- Collaborative filtering scores
- Category affinity calculations
- Final ranking and selection

### 📝 Response Generation
- LLM-based reasoning generation
- Natural language explanations

### 📊 Metrics & Scores
- Processing times
- Confidence scores
- Quality metrics
- Error tracking

## Understanding the Dashboard

### Traces View
- See all API requests chronologically
- Filter by session ID, user, or tags
- View latency and token usage

### Sessions
- Group related requests by customer
- Track customer journey
- Analyze patterns

### LLM Calls
- Monitor all LLM invocations
- Track prompt templates
- Analyze token usage and costs

### Scores
- View quality metrics
- Track recommendation accuracy
- Monitor user satisfaction

## Troubleshooting

### Services Not Starting

Check service health:
```bash
docker-compose ps
```

View logs:
```bash
docker-compose logs langfuse
docker-compose logs langfuse-db
docker-compose logs api
```

### API Keys Not Working

1. Verify keys in `.env` file
2. Check keys in LangFuse UI (Project Settings > API Keys)
3. Ensure `LANGFUSE_ENABLED=true`
4. Restart API: `docker-compose restart api`

### No Traces Appearing

1. Check API logs for LangFuse errors:
   ```bash
   docker logs shopping-api | grep -i langfuse
   ```

2. Verify network connectivity:
   ```bash
   docker exec shopping-api wget -O- http://langfuse:3000/api/public/health
   ```

3. Check LangFuse logs:
   ```bash
   docker logs langfuse
   ```

### Database Connection Issues

Reset the database:
```bash
docker-compose down langfuse-db langfuse
docker volume rm personalized-shopping-api_langfuse-db-data
docker-compose up -d langfuse-db langfuse
```

Then recreate your account and API keys.

## Advanced Configuration

### Environment Variables

Full configuration options in `.env`:

```bash
# Enable/disable tracing
LANGFUSE_ENABLED=true

# API keys (from LangFuse UI)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Host (internal Docker network)
LANGFUSE_HOST=http://langfuse:3000

# Metadata
LANGFUSE_RELEASE=v1.0.0
LANGFUSE_ENVIRONMENT=development

# Debug mode (verbose logging)
LANGFUSE_DEBUG=false
```

### Custom Tags and Metadata

Traces are automatically tagged with:
- `intent_classification` - Intent detection traces
- `profiling` - Customer profiling traces
- `similarity` - Similar customer discovery
- `filtering` - Sentiment filtering
- `recommendation` - Recommendation generation
- `reasoning` - Response generation

### Session IDs

Each API request gets a unique session ID: `req-xxxxxxxx`

This helps track related operations across multiple traces.

## Production Considerations

### Security

For production:
1. Use strong passwords
2. Store API keys in secrets manager
3. Use HTTPS for LangFuse UI
4. Restrict database access
5. Enable authentication middleware

### Performance

- LangFuse adds minimal latency (~10-50ms per trace)
- Traces are sent asynchronously
- Failed traces don't break your API
- Consider sampling in high-traffic scenarios

### Data Retention

Configure in LangFuse UI:
- Project Settings > Data Retention
- Set retention period (default: 90 days)
- Auto-delete old traces

## Resources

- **LangFuse Documentation**: https://langfuse.com/docs
- **LangFuse GitHub**: https://github.com/langfuse/langfuse
- **Python SDK Docs**: https://langfuse.com/docs/sdk/python

## Support

If you encounter issues:
1. Check this guide's troubleshooting section
2. Review LangFuse logs: `docker logs langfuse`
3. Check API logs: `docker logs shopping-api`
4. Consult LangFuse documentation

---

**Last Updated**: 2025-01-17
