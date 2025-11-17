# System Architecture

Complete technical architecture documentation for the Personalized Shopping Assistant.

## Table of Contents

1. [System Overview](#system-overview)
2. [Multi-Agent Architecture](#multi-agent-architecture)
3. [Data Flow](#data-flow)
4. [State Management](#state-management)
5. [Vector Store Design](#vector-store-design)
6. [Error Handling](#error-handling)
7. [Performance Optimization](#performance-optimization)
8. [Observability](#observability)

## System Overview

The system implements a **collaborative filtering recommendation engine** using a multi-agent architecture orchestrated by LangGraph.

### Key Components

```
┌─────────────────────────────────────────────────────────┐
│                    Application Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  CLI (main)  │  │   API (opt)  │  │  Tests       │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Orchestration Layer                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │          LangGraph Workflow Engine               │   │
│  │  (State management, routing, checkpointing)      │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                      Agent Layer                         │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │Agent │  │Agent │  │Agent │  │Agent │  │Agent │     │
│  │  1   │→ │  2   │→ │  3   │→ │  4   │→ │  5   │     │
│  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘     │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                     Service Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │   LLM    │  │Embedding │  │ Vector   │             │
│  │ Factory  │  │  Model   │  │  Store   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                      Data Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │Purchase  │  │ Reviews  │  │Embeddings│             │
│  │   CSV    │  │   CSV    │  │  Cache   │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

## Multi-Agent Architecture

### Agent 1: Customer Profiling Agent

**Responsibility:** Extract customer behavior profile from purchase history

**Input:**
- Customer name or ID
- Purchase history lookback period

**Processing:**
1. Lookup customer ID (if name provided)
2. Load purchase transactions
3. Calculate aggregate metrics:
   - Total purchases, total spent, average price
   - Favorite categories (top 3 by frequency)
   - Purchase frequency classification
   - Price segment classification
4. Build CustomerProfile object

**Output:**
- CustomerProfile with confidence score
- Recent purchases (last 5)
- Behavioral segments

**Model:** None (algorithmic processing with pandas)

**Performance Target:** <100ms

### Agent 2: Similar Customer Discovery Agent

**Responsibility:** Find customers with similar purchase behaviors

**Input:**
- Target customer profile
- Similarity threshold
- Top-K parameter

**Processing:**
1. Generate embedding from customer behavior:
   ```
   Text = "Frequency: {freq} buyer, Price: {segment},
           Categories: {cats}, Avg: ${avg}"
   ```
2. Encode using BGE model (768-dim vector)
3. Search FAISS index for similar vectors
4. Convert L2 distance to similarity score: `exp(-distance)`
5. Filter by threshold (default: 0.75)
6. Load purchase histories for similar customers

**Output:**
- List of SimilarCustomer objects
- Similarity scores
- Common categories
- Their purchase histories

**Model:** `BAAI/bge-base-en-v1.5` (sentence-transformers)

**Performance Target:** <200ms

### Agent 3: Review-Based Filtering Agent

**Responsibility:** Filter products to high-quality items only

**Input:**
- Candidate products from similar customers
- Sentiment threshold

**Processing:**
1. Extract unique product IDs
2. Load reviews for each product
3. Analyze sentiment using LLM or rules:
   - **LLM Method:** Llama 3.1 8B scores 0.0-1.0
   - **Rule-based:** Keyword matching for speed
4. Calculate average sentiment per product
5. Filter products below threshold (default: 0.6 = 4 stars)
6. Handle products with no reviews (neutral score)

**Output:**
- Filtered product list
- Sentiment scores and confidence
- Review counts
- Products filtered out count

**Model:** `llama3.1:8b` (sentiment analysis)

**Performance Target:** <500ms

### Agent 4: Cross-Category Recommendation Agent

**Responsibility:** Generate final recommendations with scoring

**Input:**
- High-quality filtered products
- Customer profile
- Similar customers data

**Processing:**
1. **Collaborative Filtering Score:**
   ```
   collab_score = (# similar customers who bought) / max_count
   ```

2. **Category Affinity Score:**
   ```
   category_score = calculate_affinity(customer_favorites, product_category)
   ```

3. **Final Score:**
   ```
   final_score = (0.6 × collab_score) +
                 (0.4 × category_score) +
                 (0.2 × sentiment_score)
   ```

4. **Diversity Filter:** Max 2 products per category

5. **Explanation Generation:** Determine reason for each recommendation

**Output:**
- Top-N ProductRecommendation objects
- Scores and reasoning
- Similar customer counts
- Source attribution

**Model:** None (algorithmic)

**Performance Target:** <100ms

### Agent 5: Response Generation Agent

**Responsibility:** Format as natural language response

**Input:**
- Final recommendations
- Customer context
- Original query

**Processing:**
1. Build context summary from profile
2. Format recommendations as structured data
3. Create LLM prompt with:
   - Customer context
   - Recommendations with details
   - Reasoning for each
4. Generate conversational response
5. Fallback to template if LLM fails

**Output:**
- Natural language response
- Formatted recommendations
- Conversational tone

**Model:** `llama3.1:8b` (response generation)

**Performance Target:** <500ms

## Data Flow

### Query Processing Flow

```
User Query
    │
    ├─ Extract: customer_name/customer_id
    ├─ Validate: input format
    └─ Initialize: ShoppingAssistantState
    │
    ▼
Agent 1: Customer Profiling
    │
    ├─ Load: purchase_data.csv → List[Purchase]
    ├─ Calculate: metrics (pandas aggregations)
    └─ Output: CustomerProfile
    │
    ▼
Agent 2: Similar Customer Discovery
    │
    ├─ Generate: customer embedding (BGE)
    ├─ Search: FAISS index → List[(customer_id, score)]
    ├─ Load: their purchase histories
    └─ Output: List[SimilarCustomer]
    │
    ▼
Agent 3: Review-Based Filtering
    │
    ├─ Aggregate: products from similar customers
    ├─ Load: review_data.csv → List[Review]
    ├─ Analyze: sentiment (LLM or rules)
    └─ Output: filtered_products (sentiment > threshold)
    │
    ▼
Agent 4: Cross-Category Recommendation
    │
    ├─ Score: collaborative filtering
    ├─ Score: category affinity
    ├─ Combine: weighted scores
    ├─ Apply: diversity constraints
    └─ Output: List[ProductRecommendation]
    │
    ▼
Agent 5: Response Generation
    │
    ├─ Format: context + recommendations
    ├─ Generate: LLM response (conversational)
    └─ Output: final_response (string)
    │
    ▼
Final Response to User
```

## State Management

### State Schema Design

The `ShoppingAssistantState` uses **TypedDict** for type safety and **Annotated reducers** for list accumulation:

```python
class ShoppingAssistantState(TypedDict):
    # Input
    query: str
    customer_name: Optional[str]
    customer_id: Optional[str]

    # Agent outputs (accumulated with operator.add)
    similar_customers: Annotated[List[SimilarCustomer], operator.add]
    candidate_products: Annotated[List[Dict], operator.add]

    # Metadata
    processing_time_ms: float
    confidence_score: float

    # Error handling
    errors: Annotated[List[str], operator.add]
    warnings: Annotated[List[str], operator.add]
```

### State Reducer Pattern

Lists use `operator.add` for automatic accumulation:

```python
# Agent returns
return {
    "similar_customers": [customer1, customer2],  # Appends to existing
    "errors": ["Some error"]  # Appends to error list
}
```

### Checkpointing

LangGraph uses **MemorySaver** for workflow checkpointing:

```python
memory = MemorySaver()
workflow.compile(checkpointer=memory)
```

This enables:
- Resume on failure
- Partial replay
- State inspection

## Vector Store Design

### FAISS Index Architecture

```
CustomerEmbeddingStore
    │
    ├─ Index Type: IVF (Inverted File)
    │  ├─ Clusters (nlist): 100
    │  ├─ Search probes (nprobe): 10
    │  └─ Metric: L2 distance
    │
    ├─ Embeddings: float32[768]
    │
    ├─ Metadata Store:
    │  ├─ customer_ids: List[str]
    │  └─ customer_metadata: Dict[str, Dict]
    │
    └─ Persistence:
       ├─ Index: .faiss file
       └─ Metadata: .metadata.pkl
```

### Index Building Process

```python
1. Load all customers from purchase data
2. For each customer:
   a. Calculate profile metrics
   b. Generate behavior text
   c. Encode with BGE model → embedding
   d. Cache embedding to disk
3. Create FAISS index (IVF type)
4. Train index with all embeddings
5. Add all embeddings in batch
6. Save index + metadata to disk
```

### Search Process

```python
1. Generate query embedding for target customer
2. Set nprobe parameter (clusters to search)
3. FAISS search → (distances, indices)
4. Convert L2 distance to similarity: exp(-distance)
5. Filter by threshold (default: 0.75)
6. Return (customer_id, similarity_score) tuples
```

## Error Handling

### Multi-Level Error Strategy

1. **Agent-Level Errors:**
   ```python
   try:
       result = process_agent()
   except Exception as e:
       return {
           "errors": [f"Agent error: {str(e)}"],
           "fallback_used": True
       }
   ```

2. **Workflow-Level Routing:**
   ```python
   def route_after_profiling(state):
       if state.get("customer_profile"):
           return "similar_customers"
       else:
           return "error"  # Route to error handler
   ```

3. **Graceful Degradation:**
   - No similar customers → Popular items fallback
   - No reviews → Neutral sentiment assumption
   - LLM failure → Template-based response

4. **Retry Logic:**
   ```python
   retry_count = state.get("retry_count", 0)
   if retry_count < max_retries:
       return retry_agent()
   else:
       return fallback_response()
   ```

## Performance Optimization

### 1. Caching Strategy

**Embedding Cache:**
```python
# Cache customer embeddings to disk
cache_path = f"{embeddings_dir}/customer_{id}.pkl"
if cache_exists:
    return load_cached_embedding()
```

**LLM Response Cache (Semantic):**
```python
# Cache similar queries (>0.95 similarity)
if semantic_similarity(query, cached_query) > 0.95:
    return cached_response
```

### 2. Model Cascading

Use smaller models for simpler tasks:
- **Profiling:** Llama 3.2 3B (fast pattern extraction)
- **Sentiment:** Llama 3.1 8B (balanced accuracy/speed)
- **Response:** Llama 3.1 8B (quality responses)

### 3. Batch Processing

```python
# Batch embeddings generation
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True
)
```

### 4. Index Optimization

**FAISS IVF Index:**
- Train with representative sample
- Use GPU version for large datasets: `faiss-gpu`
- Adjust nprobe for speed/accuracy tradeoff

## Observability

### Metrics Tracking

```python
@track_agent_performance("agent_name")
def agent_function(state):
    # Automatically tracks:
    # - Execution latency
    # - Success/failure counts
    # - Error rates
    pass
```

### Logged Metrics

1. **Performance:**
   - `{agent}_latency_ms` (per agent)
   - `end_to_end_latency_ms`
   - `processing_time_ms`

2. **Quality:**
   - `confidence_score` (overall)
   - `similarity_threshold`
   - `products_filtered_out`

3. **Business:**
   - `similar_customer_count`
   - `recommendation_count`
   - `final_recommendations`

### Structured Logging

```python
logger.info(
    f"Agent completed: {agent_name}",
    extra={
        "agent": agent_name,
        "latency_ms": elapsed,
        "success": True,
        "customer_id": customer_id
    }
)
```

### Monitoring Dashboard (Future)

- Prometheus metrics export
- Grafana dashboards
- LangSmith tracing integration

## Design Decisions

### Why LangGraph?

✅ **Stateful workflows** with checkpointing
✅ **Conditional routing** based on agent outputs
✅ **Built-in error handling** and retry logic
✅ **Production-ready** observability

### Why FAISS?

✅ **Fast similarity search** (millions of vectors)
✅ **Memory efficient** with IVF indexing
✅ **No external dependencies** (unlike Pinecone)
✅ **GPU support** for scaling

### Why BGE Embeddings?

✅ **State-of-the-art** quality for retrieval
✅ **Open-source** and self-hosted
✅ **768 dimensions** (good balance)
✅ **Optimized** for semantic similarity

### Why Multi-Agent?

✅ **Separation of concerns** (easier to debug)
✅ **Modular** (swap agents independently)
✅ **Testable** (unit test each agent)
✅ **Observable** (track each step)

## Scaling Considerations

### Horizontal Scaling

- **Stateless agents** → Parallelize across workers
- **FAISS sharding** → Distribute index across nodes
- **LLM serving** → Multiple Ollama instances

### Vertical Scaling

- **GPU acceleration** → FAISS-GPU, vLLM for LLMs
- **Batch processing** → Process multiple queries together
- **Model quantization** → GGUF quantized models

### Cost Optimization

- **Smaller models** for simple tasks (3B vs 8B)
- **Semantic caching** (avoid redundant LLM calls)
- **Pre-computation** (embeddings, index building)
- **Lazy loading** (load models only when needed)

---

**Architecture Version:** 1.0
**Last Updated:** 2025-01-16
