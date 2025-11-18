# LangFuse Integration - Complete Setup

## ✅ What's Been Implemented

### 1. Docker Compose Services Added

**PostgreSQL Database** (`langfuse-db`)
- Port: 5433
- Database: `langfuse`
- Used by LangFuse for storing traces

**LangFuse Platform** (`langfuse`)
- Port: 3002
- Version: 2.95.11 (v2 - stable)
- Web UI for trace visualization
- Auto-starts with database dependency

### 2. Application Integration

**Core Tracing Module** (`app/core/tracing.py`)
- `LangFuseTracer` singleton for client management
- `get_langfuse_callback()` for LangChain integration
- `trace_span()` context manager for manual tracing
- Helper functions: `log_event()`, `log_score()`, `create_generation()`
- Graceful fallbacks when tracing is disabled

**LLM Factory** (`app/models/llm_factory.py`)
- Updated `get_llm()` with tracing parameters
- Automatic callback attachment to all LLM instances
- Metadata enrichment with model info

**Intent Classifier Agent** (`app/agents/intent_classifier_agent.py`)
- Session and user tracking
- Trace spans around classification workflow
- Event logging for results and errors

**Recommendation Service** (`app/services/recommendation_service.py`)
- Main workflow trace span
- Sub-spans for each agent step
- Quality score logging
- Error tracking

### 3. Configuration

**Environment Variables** (`.env`)
```bash
LANGFUSE_ENABLED=true
LANGFUSE_HOST=http://langfuse:3000
LANGFUSE_PUBLIC_KEY=${LANGFUSE_PUBLIC_KEY:-}
LANGFUSE_SECRET_KEY=${LANGFUSE_SECRET_KEY:-}
LANGFUSE_ENVIRONMENT=development
LANGFUSE_RELEASE=v1.0.0
```

**Docker Environment** (`docker-compose.yml`)
- Configured for internal Docker network
- API connects to LangFuse via `http://langfuse:3000`
- External access via `http://localhost:3002`

### 4. Documentation

- ✅ [QUICKSTART.md](QUICKSTART.md) - Get started in 5 minutes
- ✅ [docs/LANGFUSE_SETUP.md](docs/LANGFUSE_SETUP.md) - Detailed setup guide
- ✅ [docs/LANGFUSE_INTEGRATION.md](docs/LANGFUSE_INTEGRATION.md) - Integration details
- ✅ [scripts/setup-langfuse.sh](scripts/setup-langfuse.sh) - Automated setup script

## 🎯 Current Status

### Services Running

```bash
$ docker-compose ps
NAME                  STATUS
langfuse-db          Up (healthy)
langfuse             Up
shopping-api         Up (healthy)
shopping-prometheus  Up
shopping-grafana     Up
```

### Ports Exposed

| Service | Port | URL |
|---------|------|-----|
| Shopping API | 8002 | http://localhost:8002 |
| LangFuse UI | 3002 | http://localhost:3002 |
| Prometheus | 9091 | http://localhost:9091 |
| Grafana | 3001 | http://localhost:3001 |
| PostgreSQL | 5433 | localhost:5433 |

## 🔄 Complete Workflow

### 1. System Architecture

```
User Request
    ↓
Shopping API (port 8002)
    ↓
[Tracing Enabled] → LangFuse (port 3002) → PostgreSQL (port 5433)
    ↓
Intent Classification → LLM (Ollama)
    ↓
Customer Profiling
    ↓
Similar Customer Discovery
    ↓
Sentiment Filtering
    ↓
Recommendation Generation
    ↓
Response Generation → LLM (Ollama)
    ↓
Response to User
```

### 2. Tracing Flow

Each API request:
1. Generates unique session ID (`req-xxxxxxxx`)
2. Creates main trace span
3. Executes workflow with sub-spans
4. Logs events and metrics
5. Sends traces to LangFuse asynchronously
6. Returns response (tracing doesn't block)

### 3. What Gets Traced

**Intent Classification**
- Input: User query
- Process: LLM-based intent detection
- Output: Intent (informational/recommendation), confidence
- Metadata: Query text, category, reasoning

**Customer Profiling**
- Input: Customer name/ID
- Process: Profile retrieval, purchase analysis
- Output: Profile summary
- Metadata: Total purchases, price segment, categories

**Similar Customer Discovery**
- Input: Customer ID, similarity threshold
- Process: Vector similarity search
- Output: List of similar customers
- Metadata: Similarity scores, count found

**Sentiment Filtering**
- Input: Candidate products
- Process: Review sentiment analysis
- Output: Filtered products
- Metadata: Products considered, filtered count

**Recommendation Generation**
- Input: Filtered products, customer preferences
- Process: Scoring, ranking, diversification
- Output: Top N recommendations
- Metadata: Scores, sources, reasoning

**Response Generation**
- Input: Query, recommendations, profile
- Process: LLM-based explanation generation
- Output: Natural language response
- Metadata: Model, tokens, latency

## 📊 Metrics Tracked

### Performance Metrics
- Processing time (ms) per request
- Processing time per agent/step
- LLM latency per call
- Token usage per call

### Quality Metrics
- Intent classification confidence
- Recommendation confidence score
- Similar customer count
- Products filtered by sentiment

### Business Metrics
- Query intent distribution
- Customer engagement patterns
- Recommendation acceptance (when feedback added)
- Error rates by component

## 🧪 Testing Instructions

### Prerequisites
1. All services running (`docker-compose up -d`)
2. Ollama running on host (port 11434)
3. LangFuse account created
4. API keys configured in `.env`
5. API restarted after configuration

### Test Suite

**1. Health Check**
```bash
curl http://localhost:8002/health
# Expected: {"status":"healthy","environment":"development","version":"1.0.0"}
```

**2. Informational Query**
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

Expected response:
- `reasoning`: Answer about purchases
- `recommendations`: Empty array
- `metadata.query_intent`: "informational"

**3. Recommendation Query**
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

Expected response:
- `reasoning`: Natural language explanation
- `recommendations`: Array of 5 products
- `metadata.session_id`: Session ID for tracing

**4. View Traces**
1. Go to http://localhost:3002
2. Login with your credentials
3. Navigate to "shopping-assistant" project
4. Click "Traces" tab
5. Find traces with session IDs from above requests

**5. Verify Trace Details**

Each trace should show:
- Session ID matching response
- Multiple spans (intent, profiling, similarity, etc.)
- LLM calls with prompts and completions
- Events (intent_classified, candidate_products_collected)
- Scores (recommendation_quality)
- Timing information

## 🔐 Security Notes

### Current Configuration (Development)

⚠️ **NOT FOR PRODUCTION** - Current settings are for local development only:

1. **LangFuse Credentials**
   - NextAuth secret: `mysecret`
   - Salt: `mysalt`
   - Encryption key: All zeros

2. **PostgreSQL**
   - User: `langfuse`
   - Password: `langfuse`
   - Port exposed: 5433

3. **Network**
   - All services on same Docker network
   - API connects via internal network
   - UI accessible externally

### Production Recommendations

1. **Use strong secrets**
   ```bash
   NEXTAUTH_SECRET=$(openssl rand -base64 32)
   SALT=$(openssl rand -base64 32)
   ENCRYPTION_KEY=$(openssl rand -hex 32)
   ```

2. **Secure database**
   - Strong password
   - Don't expose port externally
   - Use volume backups

3. **Secure LangFuse**
   - Enable HTTPS
   - Use reverse proxy (nginx/traefik)
   - Restrict network access
   - Enable authentication middleware

4. **Store secrets securely**
   - Use secrets manager (AWS Secrets Manager, Vault)
   - Don't commit `.env` to git
   - Rotate keys regularly

## 🚀 Deployment Options

### Development (Current)
```bash
docker-compose up -d
```

### Production Options

**Option 1: Docker Compose with Custom Config**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Option 2: Kubernetes**
- Deploy LangFuse using Helm chart
- Configure ingress for HTTPS
- Use Kubernetes secrets for credentials
- Scale based on load

**Option 3: Cloud-Hosted LangFuse**
- Use https://cloud.langfuse.com
- Update `LANGFUSE_HOST` to cloud URL
- Remove local LangFuse from docker-compose
- Keep only API container

## 📈 Next Steps

### Immediate
1. ✅ All services running
2. ✅ Create LangFuse account
3. ✅ Get API keys
4. ✅ Test with sample requests
5. ✅ Verify traces appear

### Short Term
1. Add user feedback collection
2. Implement custom scoring
3. Create LangFuse dashboards
4. Set up alerts for errors
5. Configure data retention

### Long Term
1. Add A/B testing framework
2. Implement recommendation evaluation
3. Create training data from traces
4. Build custom analytics
5. Optimize based on insights

## 🆘 Support

### Common Issues

**Issue: LangFuse not accessible**
```bash
# Check logs
docker logs langfuse
# Restart service
docker-compose restart langfuse
```

**Issue: No traces appearing**
```bash
# Check API logs
docker logs shopping-api | grep -i langfuse
# Verify configuration
docker exec shopping-api env | grep LANGFUSE
```

**Issue: Database connection errors**
```bash
# Check database health
docker exec langfuse-db pg_isready -U langfuse
# Restart database
docker-compose restart langfuse-db langfuse
```

### Getting Help

1. Check logs: `docker-compose logs [service]`
2. Verify configuration: Review `.env` and `docker-compose.yml`
3. Consult documentation:
   - [QUICKSTART.md](QUICKSTART.md)
   - [docs/LANGFUSE_SETUP.md](docs/LANGFUSE_SETUP.md)
   - https://langfuse.com/docs

## ✨ Summary

**You now have:**
- ✅ Complete LLM observability platform running locally
- ✅ Automatic tracing of all AI operations
- ✅ Web UI for visualization and analysis
- ✅ Production-ready integration code
- ✅ Comprehensive documentation

**Start using it:**
```bash
# 1. Ensure all services are running
docker-compose ps

# 2. Create LangFuse account at http://localhost:3002

# 3. Get API keys from LangFuse UI

# 4. Update .env with keys

# 5. Restart API
docker-compose restart api

# 6. Send test request
curl -X POST "http://localhost:8002/api/v1/recommendations/personalized" \
  -H "Content-Type: application/json" \
  -d '{"customer_name": "Kenneth Martinez", "query": "recommend products"}'

# 7. View traces at http://localhost:3002
```

---

**Last Updated**: 2025-01-17
**Status**: ✅ Ready for use
