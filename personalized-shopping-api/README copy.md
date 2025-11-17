# Personalized Shopping Assistant with Behavioral Memory

A production-ready multi-agent recommendation system that uses customer purchase history, collaborative filtering, and sentiment analysis to generate personalized product recommendations.

## 🎯 Features

- **Multi-Agent Architecture**: 5 specialized agents working together
- **Vector Similarity Search**: FAISS-based customer behavior matching
- **Sentiment Analysis**: LLM-powered review filtering
- **Collaborative Filtering**: Recommendations based on similar customers
- **Natural Language Responses**: Conversational AI-powered responses
- **Production-Ready**: Comprehensive error handling, logging, and metrics
- **Modular Design**: Easy to extend and customize

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query Input                          │
└────────────────────┬────────────────────────────────────────┘
                     │
          ┌──────────▼──────────┐
          │  Agent 1: Customer  │
          │     Profiling       │
          │  (Llama 3.2 3B)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Agent 2: Similar   │
          │ Customer Discovery  │
          │  (BGE Embeddings)  │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Agent 3: Review    │
          │    Filtering        │
          │  (Llama 3.1 8B)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Agent 4: Cross-    │
          │   Category Rec      │
          │  (Algorithm)        │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │  Agent 5: Response  │
          │    Generation       │
          │  (Llama 3.1 8B)    │
          └──────────┬──────────┘
                     │
          ┌──────────▼──────────┐
          │   Final Response    │
          └─────────────────────┘
```

## 📋 Prerequisites

- Python 3.10+
- Ollama running locally with:
  - `llama3.2:3b`
  - `llama3.1:8b`
- Customer purchase data (CSV)
- Customer review data (CSV)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and navigate to project
cd personalized-shopping-assistant

# Run setup script
chmod +x scripts/setup_environment.sh
./scripts/setup_environment.sh

# Activate virtual environment
source venv/bin/activate
```

### 2. Configuration

Edit `.env` file with your settings:

```env
# Ollama endpoint
OLLAMA_BASE_URL=http://localhost:11434

# Data paths
PURCHASE_DATA_PATH=./data/raw/customer_purchase_data.csv
REVIEW_DATA_PATH=./data/raw/customer_reviews_data.csv

# Model settings
PROFILING_MODEL=llama3.2:3b
SENTIMENT_MODEL=llama3.1:8b
RECOMMENDATION_MODEL=llama3.1:8b
RESPONSE_MODEL=llama3.1:8b

# Embedding model
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

### 3. Prepare Data

Place your CSV files in `data/raw/`:

**customer_purchase_data.csv**:
```csv
TransactionID,CustomerID,CustomerName,ProductID,ProductName,ProductCategory,PurchaseQuantity,PurchasePrice,PurchaseDate,Country
1,887,Kenneth Martinez,101,Laptop,Electronics,1,699,2024-01-15,USA
```

**customer_reviews_data.csv**:
```csv
ReviewID,ProductID,ReviewText,ReviewDate
1,101,Great laptop! Very fast and reliable,2024-01-16
```

### 4. Build Vector Index

```bash
# Generate embeddings for all customers
python scripts/generate_embeddings.py

# Build FAISS vector index
python scripts/build_vector_index.py
```

### 5. Run the Assistant

```bash
# Single query mode
python main.py "What else would Kenneth Martinez like based on his purchase history?" --customer-name "Kenneth Martinez"

# Interactive mode
python main.py

# With detailed metrics
python main.py "Your query" --customer-name "John Doe" --metrics
```

## 📊 Example Output

```
🔍 Processing Query: "What else would Kenneth Martinez like based on his purchase history?"

[Agent 1: Customer Profiling]
✓ Customer found: Kenneth Martinez (ID: 887)
✓ Profile: 5 purchases, $689 avg price, Premium segment
✓ Favorite categories: Electronics (100%)

[Agent 2: Similar Customer Discovery]
✓ Found 20 similar customers (similarity > 0.75)
✓ Top match: CustomerID 560 (similarity: 0.89)

[Agent 3: Review-Based Filtering]
✓ Filtered: 8 products with sentiment > 0.6
✓ Removed: 4 products with poor reviews

[Agent 4: Cross-Category Recommendation]
✓ Generated 5 recommendations

[Agent 5: Response Generation]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Based on Kenneth's purchase history and customers with similar preferences,
I recommend:

1. **Laptop** ($520) - Highly rated by similar premium electronics buyers
   └─ Reason: 8 similar customers purchased, avg sentiment 0.85

2. **Smartwatch** ($489) - Popular complementary product
   └─ Reason: 6 similar customers, fits Kenneth's premium segment

3. **Camera** ($79) - Strong reviews from electronics enthusiasts
   └─ Reason: 5 similar customers, 0.82 sentiment score

These recommendations are based on purchase patterns from 20 customers
similar to Kenneth, all with positive product reviews.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Metadata:
   - Processing time: 847ms
   - Confidence: 0.78
   - Agents executed: customer_profiling → similar_customers → review_filtering → recommendation → response_generation
   - Similar customers: 20
   - Products evaluated: 12
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/unit/test_agents/test_customer_profiling.py -v
```

## 📁 Project Structure

```
personalized-shopping-assistant/
├── src/
│   ├── agents/              # 5 agent implementations
│   ├── data/                # Data loading and processing
│   ├── graph/               # LangGraph workflow
│   ├── models/              # LLM and embedding models
│   ├── utils/               # Utilities (logging, metrics)
│   ├── vector_store/        # FAISS vector store
│   ├── memory/              # Conversation storage
│   ├── config.py            # Configuration
│   └── state.py             # State schema
├── scripts/
│   ├── setup_environment.sh
│   ├── generate_embeddings.py
│   └── build_vector_index.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── main.py                  # CLI entry point
└── requirements.txt
```

## ⚙️ Configuration Options

### Agent Settings

```env
# Similarity threshold (0.0-1.0)
SIMILARITY_THRESHOLD=0.75

# Number of similar customers to analyze
SIMILARITY_TOP_K=20

# Sentiment threshold for filtering (0.0-1.0)
SENTIMENT_THRESHOLD=0.6

# Final recommendations count
RECOMMENDATION_TOP_N=5
```

### Performance Tuning

```env
# Enable caching
CACHE_ENABLED=true
CACHE_TTL_SECONDS=3600

# FAISS index type (flat, ivf, hnsw)
FAISS_INDEX_TYPE=ivf

# Vector database type (faiss, chroma)
VECTOR_DB_TYPE=faiss
```

## 🔧 Advanced Usage

### Custom Agent Development

```python
from src.state import ShoppingAssistantState
from src.utils.metrics import track_agent_performance

@track_agent_performance("custom_agent")
def custom_agent(state: ShoppingAssistantState) -> dict:
    # Your logic here
    return {
        "agent_execution_order": ["custom_agent"],
        # ... other state updates
    }
```

### Integrating with Existing Workflow

```python
from src.graph.workflow import create_workflow

# Get workflow
workflow = create_workflow()

# Add custom node
workflow.add_node("custom_agent", custom_agent)

# Update edges
workflow.add_edge("recommendation", "custom_agent")
workflow.add_edge("custom_agent", "response_generation")
```

## 📈 Performance Benchmarks

| Metric | Target | Typical |
|--------|--------|---------|
| End-to-end latency (p95) | <1000ms | 850ms |
| Customer profiling | <100ms | 45ms |
| Vector search | <200ms | 120ms |
| Sentiment analysis | <500ms | 380ms |

## 🐛 Troubleshooting

### "Vector store not initialized"

```bash
# Solution: Build the vector index
python scripts/build_vector_index.py
```

### "Customer not found"

- Check customer name spelling (case-insensitive)
- Verify customer exists in purchase data
- Try using customer ID instead

### Ollama connection errors

```bash
# Check Ollama is running
ollama list

# Pull required models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

## 📝 Data Format Requirements

### Purchase Data CSV

Required columns:
- `TransactionID`, `CustomerID`, `CustomerName`
- `ProductID`, `ProductName`, `ProductCategory`
- `PurchaseQuantity`, `PurchasePrice`, `PurchaseDate`
- `Country`

### Review Data CSV

Required columns:
- `ReviewID`, `ProductID`
- `ReviewText`, `ReviewDate`

## 🤝 Contributing

1. Follow existing code structure
2. Add tests for new features
3. Update documentation
4. Use type hints and docstrings

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- LangGraph for workflow orchestration
- Sentence Transformers for embeddings
- FAISS for vector similarity search
- Ollama for local LLM inference

## 📞 Support

For issues or questions:
- Check troubleshooting section
- Review example notebooks in `notebooks/`
- Open an issue on GitHub

---

**Built with ❤️ for personalized e-commerce recommendations**
