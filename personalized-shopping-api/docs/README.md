# Personalized Shopping Assistant API

A production-ready FastAPI service that provides personalized product recommendations using a **5-agent multi-agent architecture** with vector similarity search, collaborative filtering, sentiment analysis, and LLM-powered reasoning.

## 🎯 Overview

This API implements a sophisticated recommendation system that analyzes customer purchase history, finds similar customers using vector embeddings, filters products by review sentiment, scores recommendations using collaborative filtering, and generates natural language explanations.

### Key Features

- **5-Agent Multi-Agent Workflow**: Sequential agent execution for comprehensive recommendations
- **Vector Similarity Search**: FAISS-based customer behavior matching  
- **Collaborative Filtering**: Smart product scoring based on similar customers
- **Sentiment Analysis**: Review-based product filtering
- **LLM-Powered Reasoning**: Natural language explanations using Ollama
- **Production-Ready**: Docker Compose deployment with monitoring
- **Layered Architecture**: Clean separation (API → Services → Repositories → Infrastructure)
- **Dependency Injection**: Fully testable with FastAPI's `Depends()`

## 🏗️ Architecture

### Multi-Agent Workflow

```
Query: "What else would Kenneth Martinez like?"
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 1: Customer Profiling                                │
│ - Analyze purchase history                                 │
│ - Calculate metrics (avg price, frequency)                 │
│ - Determine price segment & favorite categories            │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 2: Similar Customer Discovery                        │
│ - Generate behavioral embedding (BGE-base-en-v1.5)         │
│ - FAISS vector similarity search                           │
│ - Find top-K similar customers (threshold > 0.75)          │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 3: Review-Based Filtering                            │
│ - Get products purchased by similar customers              │
│ - Analyze review sentiment (LLM or rule-based)             │
│ - Filter by sentiment threshold (> 0.6)                    │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 4: Cross-Category Recommendation                     │
│ - Score = 0.6×collaborative + 0.4×category + 0.2×sentiment │
│ - Apply diversity constraints (max 2 per category)         │
│ - Rank and select top-N products                           │
└─────────────────────────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 5: Response Generation                               │
│ - Generate natural language reasoning (Llama 3.1 8B)       │
│ - Explain why products match customer preferences          │
└─────────────────────────────────────────────────────────────┘
  ↓
Result: Top-5 recommendations with explanations
```

### System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     API Layer                            │
│  FastAPI Endpoints (Pydantic validation)                 │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│                   Service Layer                          │
│  - RecommendationService (5-agent orchestration)         │
│  - CustomerService (profiling, similarity)               │
│  - ProductService (reviews, sentiment)                   │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│                  Repository Layer                        │
│  - CustomerRepository (purchase data)                    │
│  - ProductRepository (product data)                      │
│  - ReviewRepository (review data)                        │
│  - VectorRepository (FAISS index)                        │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│               Infrastructure Layer                       │
│  - LLM Factory (Ollama Llama 3.1/3.2)                   │
│  - Embedding Model (BGE-base-en-v1.5)                   │
│  - Sentiment Analyzer (LLM/rule-based)                  │
└──────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 8GB+ RAM
- 10GB+ disk space
- Ollama (optional, Docker will use its own)

### One-Command Start

```bash
cd personalized-shopping-api
./quickstart.sh
```

This will:
1. ✅ Check Docker is running
2. ✅ Pull Ollama models (llama3.2:3b, llama3.1:8b)
3. ✅ Start all services (API, Ollama, Redis, Prometheus, Grafana)
4. ✅ Build vector index automatically
5. ✅ Wait for services to be healthy
6. ✅ Display access URLs

### Manual Start

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

## 📡 API Endpoints

### Recommendations

**Get Personalized Recommendations**
```bash
POST /api/v1/recommendations/personalized
Content-Type: application/json

{
  "query": "What else would Kenneth Martinez like based on his purchase history?",
  "customer_name": "Kenneth Martinez",
  "top_n": 5,
  "include_reasoning": true
}
```

**Response:**
```json
{
  "query": "What else would Kenneth Martinez like?",
  "customer_profile": {
    "customer_id": "887",
    "customer_name": "Kenneth Martinez",
    "total_purchases": 5,
    "avg_purchase_price": 639.99,
    "favorite_categories": ["Electronics"],
    "price_segment": "premium"
  },
  "recommendations": [
    {
      "product_id": "291",
      "product_name": "Laptop",
      "product_category": "Electronics",
      "avg_price": 520.30,
      "recommendation_score": 1.18,
      "reason": "Highly popular with 5 similar premium electronics buyers",
      "similar_customer_count": 5,
      "avg_sentiment": 0.9,
      "source": "collaborative"
    }
  ],
  "reasoning": "Based on Kenneth's purchase history of premium electronics...",
  "confidence_score": 0.85,
  "processing_time_ms": 847,
  "similar_customers_analyzed": 3,
  "products_considered": 8,
  "agent_execution_order": [
    "customer_profiling",
    "similar_customer_discovery",
    "review_filtering",
    "recommendation",
    "response_generation"
  ]
}
```

### Customers

**Get Customer Profile**
```bash
GET /api/v1/customers/{customer_id}/profile
```

**Find Similar Customers**
```bash
GET /api/v1/customers/{customer_id}/similar?top_k=10
```

### Products

**Get Product Reviews with Sentiment**
```bash
GET /api/v1/products/{product_id}/reviews
```

## 🔧 Configuration

Edit `.env` to customize:

```bash
# LLM Models
PROFILING_MODEL=llama3.2:3b      # Fast profiling
SENTIMENT_MODEL=llama3.1:8b      # Accurate sentiment
RECOMMENDATION_MODEL=llama3.1:8b # Smart recommendations
RESPONSE_MODEL=llama3.1:8b       # Quality responses

# Agent Parameters
SENTIMENT_THRESHOLD=0.6          # Min sentiment score
COLLABORATIVE_WEIGHT=0.6         # Collaborative filtering weight
CATEGORY_AFFINITY_WEIGHT=0.4     # Category match weight

# Vector Search
SIMILARITY_THRESHOLD=0.75        # Min similarity score
SIMILARITY_TOP_K=20              # Max similar customers

# Embeddings
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
EMBEDDING_DIMENSION=768
```

## 📊 Services

| Service | URL | Credentials |
|---------|-----|-------------|
| **API** | http://localhost:8000 | - |
| **Swagger UI** | http://localhost:8000/api/v1/docs | - |
| **ReDoc** | http://localhost:8000/api/v1/redoc | - |
| **Prometheus** | http://localhost:9090 | - |
| **Grafana** | http://localhost:3000 | admin / admin |
| **Redis** | localhost:6379 | - |
| **Ollama** | http://localhost:11434 | - |

## 🧪 Testing

### Quick Test

```bash
./test_api.sh
```

### Manual Tests

**1. Health Check**
```bash
curl http://localhost:8000/health
```

**2. Kenneth Martinez Use Case**
```bash
curl -X POST http://localhost:8000/api/v1/recommendations/personalized \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What else would Kenneth Martinez like?",
    "customer_name": "Kenneth Martinez"
  }' | jq
```

**3. Get Profile**
```bash
curl http://localhost:8000/api/v1/customers/887/profile | jq
```

**4. Similar Customers**
```bash
curl 'http://localhost:8000/api/v1/customers/887/similar?top_k=5' | jq
```

## 💾 Data

### Sample Data Included

**Customers**: 18 unique customers  
**Transactions**: 60 purchases  
**Reviews**: 30 product reviews  

**Kenneth Martinez (ID: 887)** - Premium Electronics Buyer:
- Smartphone ($899.99)
- Laptop ($1,299.99)
- Wireless Headphones ($249.99)
- 4K Monitor ($599.99)
- Mechanical Keyboard ($149.99)

### Adding Your Own Data

Replace CSV files in `data/raw/`:

**customer_purchase_data.csv**
```csv
transaction_id,customer_id,customer_name,product_id,product_name,product_category,price,purchase_date
1,887,Kenneth Martinez,101,Smartphone,Electronics,899.99,2024-01-15
```

**customer_reviews_data.csv**
```csv
review_id,product_id,product_name,customer_id,customer_name,rating,review_text,review_date
1,101,Smartphone,887,Kenneth Martinez,5,Excellent phone!,2024-01-20
```

Then rebuild the vector index:
```bash
docker exec -it shopping-api python scripts/build_vector_index.py
```

## 🛠️ Development

### Local Development (Without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector index
python scripts/build_vector_index.py

# Start Ollama (in another terminal)
ollama serve

# Pull models
ollama pull llama3.2:3b
ollama pull llama3.1:8b

# Start API
uvicorn app.main:app --reload

# Access at http://localhost:8000
```

### Project Structure

```
personalized-shopping-api/
├── app/
│   ├── api/v1/endpoints/          # API endpoints
│   ├── services/                  # Business logic
│   │   ├── recommendation_service.py  # 5-agent orchestration
│   │   ├── customer_service.py        # Customer profiling
│   │   └── product_service.py         # Product reviews
│   ├── repositories/              # Data access
│   ├── models/                    # LLM, embeddings
│   ├── domain/schemas/            # Pydantic models
│   └── core/                      # Config, logging
├── data/
│   ├── raw/                       # CSV data files
│   └── embeddings/                # FAISS index
├── scripts/
│   ├── build_vector_index.py     # Index builder
│   └── setup.sh                   # Setup automation
├── docker-compose.yml             # Service orchestration
├── Dockerfile                     # API container
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔍 Technical Details

### Recommendation Scoring

```python
final_score = (
    0.6 × collaborative_score +    # How many similar customers bought it
    0.4 × category_affinity +      # Matches favorite categories
    0.2 × avg_sentiment            # Product review sentiment
)
```

### Vector Embeddings

Customer behavior is encoded as:
```python
behavior_text = f"Frequency: {frequency} buyer, Price: {price_segment}, Categories: {categories}, Avg: ${avg_price}"
embedding = BGE_model.encode(behavior_text)  # 768-dim vector
```

### Similarity Search

```python
# FAISS IVF index
index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)
index.nprobe = 10

# Search
distances, indices = index.search(query_embedding, k=20)
similarity_scores = np.exp(-distances)  # Convert to similarity
```

## 📈 Monitoring

**Metrics** (Prometheus): http://localhost:9090
- Request latency
- Agent execution times
- Error rates
- Cache hit rates

**Dashboards** (Grafana): http://localhost:3000
- Login: admin / admin
- Pre-configured Prometheus datasource

**Logs**
```bash
# View all logs
docker-compose logs -f

# API logs only
docker-compose logs -f api

# Follow logs
docker-compose logs --tail=100 -f api
```

## 🐛 Troubleshooting

### Vector Index Not Found

```bash
docker exec -it shopping-api python scripts/build_vector_index.py
```

### Ollama Connection Error

Update `docker-compose.yml`:
```yaml
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### Out of Memory

Reduce workers or use smaller models:
```yaml
# docker-compose.yml
environment:
  WORKERS: 1
  RECOMMENDATION_MODEL: llama3.2:1b
```

### Rebuild Everything

```bash
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

## 📚 Documentation

- **API Docs**: http://localhost:8000/api/v1/docs
- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Implementation**: [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **LangChain** - LLM framework
- **Ollama** - Local LLM inference
- **FAISS** - Vector similarity search
- **FastAPI** - Modern web framework
- **Sentence Transformers** - Embedding models

---

**Built with ❤️ using Claude Code**

For support, open an issue or check the documentation.
