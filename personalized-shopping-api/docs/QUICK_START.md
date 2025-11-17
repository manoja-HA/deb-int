# Quick Start Guide - Personalized Shopping API

## 🚀 Running the Application

### Start All Services
```bash
cd personalized-shopping-api
docker-compose up -d
```

### Check Status
```bash
# API Health
curl http://localhost:8002/api/v1/health

# Container Logs
docker logs shopping-api --tail 50

# ChromaDB Status
docker exec shopping-api python -c "
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path='/app/data/embeddings/chroma', settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(name='customers')
print(f'Customers indexed: {collection.count()}')
"
```

### Stop Services
```bash
docker-compose down
```

---

## 📡 API Endpoints

### 1. Get Personalized Recommendations (Main Use Case)

**Endpoint**: `POST /api/v1/recommendations/personalized`

**Example: Kenneth Martinez Query**
```bash
curl -X POST http://localhost:8002/api/v1/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What else would Kenneth Martinez like based on his purchase history?",
    "customer_name": "Kenneth Martinez",
    "top_n": 5,
    "include_reasoning": true
  }' | jq '.'
```

**Response Structure**:
```json
{
  "query": "What else would Kenneth Martinez like...",
  "customer_profile": {
    "customer_id": "887",
    "customer_name": "Kenneth Martinez",
    "total_purchases": 1,
    "avg_purchase_price": 689.99,
    "favorite_categories": ["Electronics"],
    "price_segment": "premium"
  },
  "recommendations": [
    {
      "product_id": "207",
      "product_name": "Television",
      "product_category": "Electronics",
      "avg_price": 991.08,
      "recommendation_score": 0.864,
      "reason": "Matches your preference for Electronics...",
      "similar_customer_count": 1,
      "avg_sentiment": 0.82
    }
  ],
  "reasoning": "Based on Kenneth Martinez's purchase history...",
  "similar_customers_analyzed": 20,
  "processing_time_ms": 2480.92,
  "agent_execution_order": [
    "customer_profiling",
    "similar_customer_discovery",
    "review_filtering",
    "recommendation",
    "response_generation"
  ]
}
```

### 2. Find Similar Customers

**Endpoint**: `GET /api/v1/customers/{customer_id}/similar`

**Example**:
```bash
# Find 10 customers similar to Kenneth Martinez
curl http://localhost:8002/api/v1/customers/887/similar?top_k=10 | jq '.'
```

**Response**:
```json
[
  {
    "customer_id": "560",
    "customer_name": "Justin Arnold",
    "similarity_score": 0.8886,
    "common_categories": ["Electronics"],
    "purchase_overlap_count": 1
  }
]
```

### 3. Get Customer Profile

**Endpoint**: `GET /api/v1/customers/{customer_id}`

**Example**:
```bash
curl http://localhost:8002/api/v1/customers/887 | jq '.'
```

### 4. Search Customers by Name

**Endpoint**: `GET /api/v1/customers/search`

**Example**:
```bash
curl "http://localhost:8002/api/v1/customers/search?name=Kenneth" | jq '.'
```

### 5. Get Product Details

**Endpoint**: `GET /api/v1/products/{product_id}`

**Example**:
```bash
curl http://localhost:8002/api/v1/products/240 | jq '.'
```

### 6. List All Products

**Endpoint**: `GET /api/v1/products`

**Example**:
```bash
# All products
curl http://localhost:8002/api/v1/products | jq '.'

# Filter by category
curl "http://localhost:8002/api/v1/products?category=Electronics" | jq '.'
```

---

## 🧪 Example Queries

### 1. Premium Electronics Buyer (Kenneth Martinez)
```bash
curl -X POST http://localhost:8002/api/v1/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Kenneth Martinez",
    "top_n": 5
  }' | jq '.recommendations[] | {product_name, avg_price, reason}'
```

**Expected**: Electronics recommendations (Television, Tablet) in premium price range

### 2. Budget Buyer
```bash
# Find a budget customer first
curl "http://localhost:8002/api/v1/customers" | jq '.[] | select(.avg_purchase_price < 200) | .customer_name' | head -1

# Get recommendations
curl -X POST http://localhost:8002/api/v1/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "John Smith",
    "top_n": 5
  }' | jq '.recommendations[] | {product_name, avg_price}'
```

**Expected**: Budget-friendly products (< $200)

### 3. High-Frequency Buyer
```bash
# Find high-frequency customer
curl "http://localhost:8002/api/v1/customers" | jq '.[] | select(.total_purchases > 5) | .customer_name' | head -1

# Get recommendations
curl -X POST http://localhost:8002/api/v1/recommendations/personalized \
  -H "Content-Type: application/json" \
  -d '{
    "customer_name": "Sarah Johnson",
    "top_n": 5
  }' | jq '.'
```

**Expected**: Diverse recommendations based on frequent purchase patterns

---

## 🔍 Understanding the 4-Agent Workflow

### Agent 1: Customer Profiling
**What it does**: Analyzes Kenneth's purchase history
**Output**:
- Buyer type: "quantity buyer" (5 units)
- Price segment: "premium" ($689.99)
- Category preference: "Electronics"

### Agent 2: Similar Customer Discovery
**What it does**: Finds customers with similar behavioral patterns
**Method**: Vector similarity using behavioral embeddings
**Output**: 20 similar customers (87-89% match)

### Agent 3: Review-Based Filtering
**What it does**: Filters products by review sentiment
**Threshold**: 60% positive reviews minimum
**Output**: 26 products with good reviews from 35 candidates

### Agent 4: Cross-Category Recommendation
**What it does**: Ranks and selects final recommendations
**Scoring**:
- Collaborative filtering (60%)
- Category affinity (40%)
- Sentiment boost (20%)
**Output**: Top 5 products with reasoning

---

## 📊 Monitoring & Debugging

### Check ChromaDB Collection
```bash
docker exec shopping-api python -c "
import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path='/app/data/embeddings/chroma',
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_collection(name='customers')

print(f'Total customers: {collection.count()}')
print(f'Embedding dimension: 768 (BGE-base-en-v1.5)')

# Sample a customer
result = collection.get(ids=['887'], include=['documents', 'metadatas'])
print(f'\nSample (Kenneth Martinez):')
print(f'Behavior: {result[\"documents\"][0]}')
print(f'Metadata: {result[\"metadatas\"][0]}')
"
```

### Rebuild ChromaDB Index
```bash
# If you update the dataset, rebuild the index
docker exec shopping-api python /app/scripts/build_vector_index.py
docker restart shopping-api
```

### View Logs
```bash
# Real-time logs
docker logs -f shopping-api

# Last 100 lines
docker logs shopping-api --tail 100

# Filter for errors
docker logs shopping-api 2>&1 | grep ERROR

# Filter for recommendations
docker logs shopping-api 2>&1 | grep "Processing recommendation"
```

### Prometheus Metrics
```bash
# Access Prometheus UI
open http://localhost:9091

# Query API metrics
curl http://localhost:8002/metrics
```

### Grafana Dashboards
```bash
# Access Grafana
open http://localhost:3001

# Login: admin / admin
```

---

## 🛠️ Troubleshooting

### Problem: API returns 500 error
**Solution**:
```bash
# Check logs
docker logs shopping-api --tail 50

# Verify ChromaDB is loaded
docker exec shopping-api python -c "
import chromadb
client = chromadb.PersistentClient(path='/app/data/embeddings/chroma')
collection = client.get_collection(name='customers')
print(f'Loaded: {collection.count()} customers')
"

# Restart if needed
docker restart shopping-api
```

### Problem: No similar customers found
**Solution**:
```bash
# Rebuild ChromaDB index
docker exec shopping-api python /app/scripts/build_vector_index.py

# Verify customer exists
curl http://localhost:8002/api/v1/customers/887
```

### Problem: Ollama connection refused
**Solution**:
```bash
# Check if Ollama is running on host
curl http://localhost:11434/api/tags

# If not, start Ollama
ollama serve &

# Pull required model
ollama pull llama3.1:8b
```

### Problem: Container keeps restarting
**Solution**:
```bash
# Check startup logs
docker logs shopping-api

# Common issues:
# 1. ChromaDB build failed → Check data/raw/*.csv files exist
# 2. Permission denied → chmod -R 777 logs data
# 3. Port conflict → Change port in docker-compose.yml
```

---

## 📝 Configuration

### Environment Variables (docker-compose.yml)
```yaml
environment:
  - ENVIRONMENT=production
  - DEBUG=false
  - OLLAMA_BASE_URL=http://host.docker.internal:11434
  - PURCHASE_DATA_PATH=/app/data/raw/customer_purchase_data.csv
  - REVIEW_DATA_PATH=/app/data/raw/customer_reviews_data.csv
  - REDIS_URL=redis://host.docker.internal:6379
  - LOG_LEVEL=INFO
  - LOG_FORMAT=json
```

### Recommendation Parameters (app/core/config.py)
```python
SIMILARITY_THRESHOLD = 0.75          # Min similarity for similar customers
SENTIMENT_THRESHOLD = 0.6            # Min review sentiment (60%)
MIN_REVIEWS_FOR_INCLUSION = 1       # Min reviews required
COLLABORATIVE_WEIGHT = 0.6           # Collaborative filtering weight
CATEGORY_AFFINITY_WEIGHT = 0.4       # Category matching weight
```

---

## 🎯 Success Criteria

✅ **API is working** if:
- Health endpoint returns `{"status": "healthy"}`
- ChromaDB collection has 609 customers
- Kenneth Martinez query returns 4-5 recommendations
- Similar customers have 87-89% similarity
- Recommendations include Electronics products
- Processing time < 5 seconds

✅ **4-Agent workflow is working** if:
- Agent 1 profiles customer correctly (premium, Electronics, quantity buyer)
- Agent 2 finds 20+ similar customers (>75% similarity)
- Agent 3 filters by sentiment (>60% positive)
- Agent 4 returns ranked recommendations with reasoning

---

## 📚 Documentation

- [STATUS.md](STATUS.md) - Implementation status and architecture
- [TEST_RESULTS.md](TEST_RESULTS.md) - Detailed test results for Kenneth Martinez
- [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) - Original 7-agent roadmap
- [PRODUCT_STRATEGY.md](PRODUCT_STRATEGY.md) - Product strategy and analysis
- [README.md](README.md) - Project overview

---

**API Base URL**: http://localhost:8002
**Swagger Docs**: http://localhost:8002/docs
**ReDoc**: http://localhost:8002/redoc
**Metrics**: http://localhost:8002/metrics
**Prometheus**: http://localhost:9091
**Grafana**: http://localhost:3001
