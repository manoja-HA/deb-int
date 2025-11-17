# Project Index

Quick navigation guide for the Personalized Shopping Assistant codebase.

## 📚 Documentation (Start Here!)

| File | Purpose | Read Time |
|------|---------|-----------|
| [README.md](README.md) | Complete setup and usage guide | 10 min |
| [QUICKSTART.md](QUICKSTART.md) | Get running in 5 minutes | 5 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical deep dive | 20 min |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Completion status and overview | 5 min |

## 🎯 Getting Started Path

1. **First Time Setup**
   ```
   QUICKSTART.md → ./scripts/setup_environment.sh → Build index → Run!
   ```

2. **Understanding the System**
   ```
   README.md → ARCHITECTURE.md → src/graph/workflow.py
   ```

3. **Customizing**
   ```
   .env → src/config.py → src/agents/ → src/graph/routers.py
   ```

## 📂 Directory Structure

```
personalized-shopping-assistant/
│
├── 📄 Core Entry Points
│   ├── main.py                      # CLI entry point
│   ├── .env.example                 # Configuration template
│   └── requirements.txt             # Dependencies
│
├── 📁 src/ - Source Code
│   │
│   ├── 🤖 agents/                   # 5 Agent Implementations
│   │   ├── customer_profiling.py   # Agent 1: Profile extraction
│   │   ├── similar_customers.py    # Agent 2: Vector similarity
│   │   ├── review_filtering.py     # Agent 3: Sentiment filtering
│   │   ├── recommendation.py       # Agent 4: Scoring & ranking
│   │   └── response_generation.py  # Agent 5: LLM response
│   │
│   ├── 🌊 graph/                    # LangGraph Workflow
│   │   ├── workflow.py             # Workflow definition
│   │   └── routers.py              # Conditional routing
│   │
│   ├── 💾 data/                     # Data Processing
│   │   ├── loaders.py              # CSV loading
│   │   ├── processors.py           # Data transformations
│   │   └── embeddings_generator.py # Embedding creation
│   │
│   ├── 🧠 models/                   # Model Management
│   │   ├── llm_factory.py          # LLM initialization
│   │   ├── embedding_model.py      # BGE wrapper
│   │   └── sentiment_analyzer.py   # Sentiment analysis
│   │
│   ├── 🗄️ vector_store/             # Vector Database
│   │   ├── customer_embeddings.py  # FAISS index
│   │   └── product_embeddings.py   # Product vectors
│   │
│   ├── 🛠️ utils/                     # Utilities
│   │   ├── logging.py              # Structured logging
│   │   ├── metrics.py              # Performance tracking
│   │   └── validators.py           # Input validation
│   │
│   ├── 💬 memory/                   # Conversation Storage
│   │   └── conversation_store.py   # Session persistence
│   │
│   ├── state.py                    # State schema (TypedDict)
│   └── config.py                   # Configuration (Pydantic)
│
├── 🔧 scripts/ - Automation
│   ├── setup_environment.sh        # Initial setup
│   ├── generate_embeddings.py     # Pre-compute embeddings
│   ├── build_vector_index.py      # Build FAISS index
│   └── run_evaluation.py          # System evaluation
│
├── 🧪 tests/ - Test Suite
│   ├── conftest.py                # Pytest fixtures
│   ├── unit/                      # Unit tests
│   │   ├── test_agents/
│   │   └── test_data.py
│   └── integration/               # Integration tests
│       └── test_workflow.py
│
└── 📦 data/ - Data Storage
    ├── raw/                       # CSV input files
    ├── embeddings/                # Cached embeddings
    └── processed/                 # Intermediate data
```

## 🔍 Key Files by Use Case

### "I want to understand the system"

1. [README.md](README.md) - Overview and setup
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [src/graph/workflow.py](src/graph/workflow.py) - Agent orchestration
4. [src/state.py](src/state.py) - Data flow

### "I want to customize agents"

1. [src/agents/](src/agents/) - All agent implementations
2. [src/graph/routers.py](src/graph/routers.py) - Routing logic
3. [src/config.py](src/config.py) - Configuration options

### "I want to modify data processing"

1. [src/data/loaders.py](src/data/loaders.py) - CSV loading
2. [src/data/processors.py](src/data/processors.py) - Transformations
3. [src/data/embeddings_generator.py](src/data/embeddings_generator.py) - Embeddings

### "I want to change models"

1. [src/models/llm_factory.py](src/models/llm_factory.py) - LLM setup
2. [src/models/embedding_model.py](src/models/embedding_model.py) - Embeddings
3. [.env](.env.example) - Model configuration

### "I want to add new features"

1. Create new agent in [src/agents/](src/agents/)
2. Update [src/graph/workflow.py](src/graph/workflow.py)
3. Add routing in [src/graph/routers.py](src/graph/routers.py)
4. Update [src/state.py](src/state.py) if needed

### "I want to run and test"

1. [scripts/setup_environment.sh](scripts/setup_environment.sh) - Setup
2. [scripts/build_vector_index.py](scripts/build_vector_index.py) - Index
3. [main.py](main.py) - Run queries
4. [tests/](tests/) - Test suite

## 📖 Code Reading Order

### For Understanding the Flow

```
1. src/state.py              # Understand state structure
2. src/graph/workflow.py     # See agent orchestration
3. src/agents/customer_profiling.py  # Follow a simple agent
4. src/graph/routers.py      # Understand routing
5. src/agents/similar_customers.py   # See vector search
6. src/vector_store/customer_embeddings.py  # FAISS implementation
```

### For Implementation Details

```
1. src/config.py             # Configuration system
2. src/data/loaders.py       # Data loading
3. src/models/llm_factory.py # Model management
4. src/utils/metrics.py      # Observability
5. src/agents/               # All agent logic
```

## 🔧 Configuration Files

| File | Purpose |
|------|---------|
| `.env.example` | Configuration template |
| `src/config.py` | Configuration validation |
| `requirements.txt` | Python dependencies |
| `pyproject.toml` | Project metadata |

## 🧪 Testing Files

| File | Tests |
|------|-------|
| `tests/conftest.py` | Test fixtures |
| `tests/unit/test_agents/` | Agent unit tests |
| `tests/unit/test_data.py` | Data processing tests |
| `tests/integration/test_workflow.py` | E2E workflow tests |

## 📊 Stats

- **Total Lines of Code:** ~3,341
- **Python Files:** 30
- **Test Files:** 6
- **Documentation Files:** 4
- **Scripts:** 4

## 🚀 Common Workflows

### Setup and Run

```bash
# 1. Setup
./scripts/setup_environment.sh
source venv/bin/activate

# 2. Configure
cp .env.example .env
# Edit .env with your settings

# 3. Add data
# Copy CSVs to data/raw/

# 4. Build index
python scripts/build_vector_index.py

# 5. Run
python main.py "Your query" --customer-name "Name"
```

### Development

```bash
# Run tests
pytest

# Run specific test
pytest tests/unit/test_agents/test_customer_profiling.py -v

# Check coverage
pytest --cov=src --cov-report=html

# Run evaluation
python scripts/run_evaluation.py
```

### Debugging

```bash
# Enable debug logging
python main.py "Query" --customer-name "Name" --log-level DEBUG

# View logs
tail -f logs/shopping_assistant_*.log

# Check metrics
python main.py "Query" --customer-name "Name" --metrics
```

## 🎯 Quick Reference

### Agent Files
- Agent 1: `src/agents/customer_profiling.py`
- Agent 2: `src/agents/similar_customers.py`
- Agent 3: `src/agents/review_filtering.py`
- Agent 4: `src/agents/recommendation.py`
- Agent 5: `src/agents/response_generation.py`

### Configuration
- Main config: `src/config.py`
- Environment: `.env`
- Models: `src/models/llm_factory.py`

### Data
- Loaders: `src/data/loaders.py`
- Processing: `src/data/processors.py`
- Embeddings: `src/data/embeddings_generator.py`

### Workflow
- Main workflow: `src/graph/workflow.py`
- Routing: `src/graph/routers.py`
- State: `src/state.py`

## 💡 Tips

1. **Start with QUICKSTART.md** for fastest setup
2. **Read ARCHITECTURE.md** for deep understanding
3. **Check tests/** for usage examples
4. **Use --log-level DEBUG** for troubleshooting
5. **Modify .env** for easy configuration changes

## 🔗 External Resources

- LangGraph Docs: https://langchain-ai.github.io/langgraph/
- FAISS Wiki: https://github.com/facebookresearch/faiss/wiki
- BGE Embeddings: https://huggingface.co/BAAI/bge-base-en-v1.5
- Ollama: https://ollama.com/

---

**Need help?** Check the troubleshooting section in [README.md](README.md)
