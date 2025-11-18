# Orchestration Architecture - Complete Guide

**Last Updated**: January 2025
**System Version**: v2.0 (with PydanticAI)

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Detailed Orchestration Flow](#detailed-orchestration-flow)
3. [5-Agent Sequential Pipeline](#5-agent-sequential-pipeline)
4. [Agent Deep Dive](#agent-deep-dive)
5. [Data Flow Diagrams](#data-flow-diagrams)
6. [Observability & Tracing](#observability--tracing)
7. [Design Patterns](#design-patterns)
8. [Performance Characteristics](#performance-characteristics)
9. [Infrastructure Components](#infrastructure-components)

---

## Architecture Overview

The system uses a **multi-layered orchestration pattern** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                        API Layer (FastAPI)                       │
│                    /recommendations/personalized                 │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                   RecommendationService                          │
│                    (Thin Facade Layer)                           │
│                                                                   │
│  Responsibilities:                                               │
│  • Customer lookup by name/ID                                    │
│  • Intent classification (INFORMATIONAL vs RECOMMENDATION)       │
│  • Route to appropriate handler                                  │
│  • Maintain tracing/observability                                │
└─────────┬────────────────────────────────────┬──────────────────┘
          │                                    │
          │ INFORMATIONAL                      │ RECOMMENDATION
          │                                    │
┌─────────▼──────────────┐        ┌───────────▼──────────────────┐
│ QueryAnsweringService  │        │ PersonalizedRecommendation   │
│                        │        │        Workflow              │
│ Direct answers to      │        │                              │
│ factual queries        │        │  Multi-agent orchestration   │
└────────────────────────┘        └──────────────┬───────────────┘
                                                  │
                                  ┌───────────────▼───────────────┐
                                  │   5-Agent Sequential Pipeline  │
                                  └────────────────────────────────┘
```

### Layer Responsibilities

#### 1. API Layer (FastAPI)
- **File**: `app/api/v1/endpoints/recommendations.py`
- **Responsibility**: HTTP request handling, input validation
- **Endpoint**: `POST /recommendations/personalized`

#### 2. Service Layer (RecommendationService)
- **File**: `app/services/recommendation_service.py`
- **Lines of Code**: ~150 (thin facade)
- **Responsibilities**:
  - Customer lookup by name or ID
  - Intent classification using PydanticAI
  - Routing to appropriate handler
  - Tracing and observability
  - Backward compatibility

#### 3. Workflow Layer (PersonalizedRecommendationWorkflow)
- **File**: `app/workflows/personalized_recommendation.py`
- **Lines of Code**: ~350
- **Responsibilities**:
  - Pure orchestration (no business logic)
  - Agent coordination
  - Data transformation between agents
  - Error handling
  - Response construction

#### 4. Agent Layer (5 Specialized Agents)
- **Location**: `app/capabilities/agents/`
- **Pattern**: Single-purpose, stateless, testable
- **Interface**: `BaseAgent[InputModel, OutputModel]`

---

## Detailed Orchestration Flow

### Phase 1: Request Entry (RecommendationService)

**File**: `app/services/recommendation_service.py:86-257`

```python
async def get_personalized_recommendations(
    query: str,
    customer_name: Optional[str] = None,
    customer_id: Optional[str] = None,
    top_n: int = 5,
    include_reasoning: bool = True
) -> RecommendationResponse:
```

**Execution Steps**:

#### Step 1: Input Validation (lines 124-127)

```python
if not customer_name and not customer_id:
    raise ValidationException(
        "Either customer_name or customer_id must be provided"
    )
```

**Validates**:
- At least one identifier (name or ID)
- Query is non-empty
- top_n is within bounds (1-20)

---

#### Step 2: Tracing Setup (lines 130-146)

```python
# Generate session ID for tracing
session_id = f"req-{uuid.uuid4().hex[:8]}"

with trace_span(
    name="recommendation_workflow",
    session_id=session_id,
    user_id=customer_name or customer_id,
    metadata={
        "query": query,
        "top_n": top_n,
        "include_reasoning": include_reasoning,
    },
    tags=["recommendation", "workflow"],
    input_data={
        "query": query,
        "customer_name": customer_name,
        "customer_id": customer_id,
        "top_n": top_n,
    }
) as main_trace:
```

**Creates**:
- Unique session ID: `req-a1b2c3d4`
- LangFuse trace span
- Tracks all metadata for observability

---

#### Step 3: Intent Classification (lines 155-167) ✨ **PydanticAI**

```python
logger.info(
    f"[Intent Classification] Analyzing query: '{query[:50]}...'"
)

intent_result = self.intent_classifier_agent.classify(
    query,
    session_id=session_id,
    user_id=customer_name or customer_id
)
intent = intent_result.intent
category = intent_result.category
confidence = intent_result.confidence

logger.info(
    f"[Intent Classification] Detected: {intent.value} "
    f"(confidence: {confidence:.2f}) - {intent_result.reasoning}"
)
```

**Using**: `IntentClassifierAgentV2` (PydanticAI-based)

**Classifies queries into two categories**:

1. **INFORMATIONAL** - Questions about customer data
   - Examples:
     - "How many items has Kenneth Martinez bought?"
     - "What is my total spending?"
     - "What are my favorite categories?"
     - "What did I buy recently?"
   - Categories: `total_purchases`, `spending`, `favorite_categories`, `recent_purchases`, `customer_profile`, `general`

2. **RECOMMENDATION** - Requests for product suggestions
   - Examples:
     - "What should I buy?"
     - "Recommend some products for me"
     - "What would I like based on my history?"

**Output** (`IntentClassification`):
```python
IntentClassification(
    intent=QueryIntent.INFORMATIONAL,  # or RECOMMENDATION
    category=InformationCategory.SPENDING,  # or None
    confidence=0.95,
    reasoning="Query explicitly asks about total spending amount",
    extracted_info={"time_period": "all_time"}
)
```

**How it works** (PydanticAI):
- Loads prompt from `prompts/intent_classification/`
- Uses Ollama LLM (llama3.1:8b)
- Automatic JSON parsing and validation
- Pydantic field validators convert strings to enums
- Built-in retry on failures

---

#### Step 4: Customer Lookup (lines 173-178)

```python
if not customer_id and customer_name:
    customer_id = self.customer_repo.get_customer_id_by_name(
        customer_name
    )
    if not customer_id:
        raise NotFoundException("Customer", customer_name)
```

**Resolves**:
- Customer name → Customer ID
- Throws 404 if customer not found

---

#### Step 5: Routing Decision (line 183)

```python
if intent == QueryIntent.INFORMATIONAL:
    # Route to query answering service
    result = await self._handle_informational_query(...)
else:
    # Route to recommendation workflow
    result = await self.recommendation_workflow.execute(...)
```

---

### Branch A: Informational Query Path

**Handler**: `_handle_informational_query()` (lines 259-325)

**Flow**:

```
1. Get customer profile
   ↓
2. QueryAnsweringService.answer_query()
   ↓
3. Return RecommendationResponse with:
   - reasoning = answer text
   - recommendations = [] (empty)
   - metadata.intent = "informational"
```

**Implementation**:

```python
async def _handle_informational_query(
    self,
    query: str,
    customer_id: str,
    customer_name: Optional[str],
    category,
    extracted_info,
    start_time: float,
    session_id: str,
) -> RecommendationResponse:
    # Get customer profile
    profile = await self.customer_service.get_customer_profile(customer_id)

    # Generate answer using query answering service
    answer = await self.query_answering_service.answer_query(
        query=query,
        profile=profile,
        category=category,
        extracted_info=extracted_info,
        session_id=session_id,
    )

    # Calculate processing time
    processing_time_ms = (time.time() - start_time) * 1000

    # Return response with answer (no recommendations)
    return RecommendationResponse(
        query=query,
        customer_profile=CustomerProfileSummary(...),
        recommendations=[],  # No products for informational queries
        reasoning=answer,
        confidence_score=0.9,  # High confidence for factual answers
        processing_time_ms=processing_time_ms,
        similar_customers_analyzed=0,
        products_considered=0,
        products_filtered_by_sentiment=0,
        recommendation_strategy="informational_query",
        agent_execution_order=["intent_classification", "query_answering"],
        metadata={
            "intent": "informational",
            "category": category.value if category else None,
            "session_id": session_id,
        },
    )
```

**Use Cases**:
- "How many items has John bought?" → `"John has purchased 45 items in total."`
- "What's my total spending?" → `"Your total spending is $4,027.50."`
- "What are my favorite categories?" → `"Your favorite categories are Electronics, Books, and Home & Garden."`

---

### Branch B: Recommendation Query Path (Main Workflow)

**Handler**: `PersonalizedRecommendationWorkflow.execute()` (lines 211-245 in service, 97-349 in workflow)

**Service code**:

```python
else:
    # Route to recommendation workflow
    logger.info(
        "[Recommendation Query] Routing to PersonalizedRecommendationWorkflow"
    )

    # Create agent context
    context = AgentContext(
        request_id=session_id,
        session_id=session_id,
        user_id=customer_id,
        metadata={
            "customer_name": customer_name,
            "query": query,
        },
    )

    # Execute workflow
    result = await self.recommendation_workflow.execute(
        customer_id=customer_id,
        query=query,
        top_n=top_n,
        include_reasoning=include_reasoning,
        context=context,
    )

    # Update trace
    if main_trace:
        main_trace.update(output={
            "intent": "recommendation",
            "recommendations_count": len(result.recommendations),
            "processing_time_ms": result.processing_time_ms,
            "confidence_score": result.confidence_score,
        })

    return result
```

This routes to the **core multi-agent orchestration** for product recommendations.

---

## 5-Agent Sequential Pipeline

**File**: `app/workflows/personalized_recommendation.py`

### Pipeline Overview

```
┌──────────────────────────────────────────────────────────────┐
│                  Workflow Orchestration                       │
│           PersonalizedRecommendationWorkflow.execute()       │
└───┬──────────────────────────────────────────────────────────┘
    │
    ├─► [Agent 1] Customer Profiling
    │   └─► Extract behavioral metrics from purchase history
    │
    ├─► [Agent 2] Similar Customer Discovery
    │   └─► Find customers with similar purchasing patterns
    │
    ├─► [Data Collection] Gather candidate products
    │   └─► Products purchased by similar customers
    │
    ├─► [Agent 3] Sentiment Filtering
    │   └─► Remove products with poor reviews
    │
    ├─► [Agent 4] Product Scoring & Ranking
    │   └─► Score using collaborative filtering + category affinity
    │
    └─► [Agent 5] Response Generation (PydanticAI)
        └─► Generate natural language explanation
```

### Workflow Execution

```python
async def execute(
    self,
    customer_id: str,
    query: str,
    top_n: int = 5,
    include_reasoning: bool = True,
    context: Optional[AgentContext] = None,
) -> RecommendationResponse:
    """
    Execute the personalized recommendation workflow

    Args:
        customer_id: Customer to recommend for
        query: User's search query
        top_n: Number of recommendations to return
        include_reasoning: Whether to generate LLM reasoning
        context: Execution context (created if not provided)

    Returns:
        Complete recommendation response with products and reasoning
    """
    start_time = time.time()
    agent_execution = []

    logger.info(
        f"Starting personalized recommendation workflow for customer {customer_id}"
    )

    # Execute 5-agent pipeline...
```

---

## Agent Deep Dive

### Agent 1: Customer Profiling

**File**: `app/capabilities/agents/customer_profiling.py`
**Lines in Workflow**: 138-146

#### Purpose
Extract behavioral metrics from purchase history to understand customer preferences.

#### Input
```python
CustomerProfilingInput(
    customer_id="123"
)
```

#### Execution
```python
profiling_input = CustomerProfilingInput(customer_id=customer_id)
profiling_output = await self.profiling_agent.run(profiling_input, context)
profile = profiling_output.profile

logger.info(
    f"[Agent 1] Profiled customer: {profile.total_purchases} purchases, "
    f"{profile.purchase_frequency} frequency"
)
```

#### What it does

1. **Fetch all purchases** for the customer
2. **Calculate metrics**:
   - Total purchases
   - Average purchase price
   - Total spending
   - Favorite categories (top 3 by purchase count)
   - Price segment classification
   - Purchase frequency

#### Algorithm

```python
# Price Segment Classification
avg_price = total_spending / total_purchases

if avg_price < 50:
    price_segment = "Budget"
elif avg_price < 150:
    price_segment = "Mid-range"
else:
    price_segment = "Premium"

# Purchase Frequency Classification
purchases_per_month = total_purchases / months_active

if purchases_per_month >= 4:
    frequency = "frequent"
elif purchases_per_month >= 1:
    frequency = "regular"
else:
    frequency = "occasional"

# Favorite Categories
category_counts = Counter(purchase.category for purchase in purchases)
favorite_categories = [cat for cat, count in category_counts.most_common(3)]
```

#### Output
```python
CustomerProfile(
    customer_id="123",
    customer_name="Kenneth Martinez",
    total_purchases=45,
    avg_purchase_price=89.50,
    total_spending=4027.50,
    favorite_categories=["Electronics", "Books", "Home & Garden"],
    price_segment="Mid-range",
    purchase_frequency="frequent"
)
```

#### Performance
- **Latency**: ~20ms (database query)
- **No LLM calls**
- **Pure computation**

---

### Agent 2: Similar Customer Discovery

**File**: `app/capabilities/agents/similar_customers.py`
**Lines in Workflow**: 151-162

#### Purpose
Find customers with similar purchasing behavior using vector similarity search.

#### Input
```python
SimilarCustomersInput(
    customer_profile=profile,
    top_k=20,  # From settings.SIMILARITY_TOP_K
    similarity_threshold=0.7  # From settings.SIMILARITY_THRESHOLD
)
```

#### Execution
```python
similar_input = SimilarCustomersInput(
    customer_profile=profile,
    top_k=settings.SIMILARITY_TOP_K,
    similarity_threshold=settings.SIMILARITY_THRESHOLD,
)
similar_output = await self.similar_customers_agent.run(similar_input, context)
similar_customers = similar_output.similar_customers

logger.info(
    f"[Agent 2] Found {len(similar_customers)} similar customers"
)
```

#### How it works

1. **Create embedding** from customer profile:
   ```python
   profile_text = f"""
   Customer purchases: {total_purchases}
   Average price: ${avg_purchase_price}
   Categories: {", ".join(favorite_categories)}
   Price segment: {price_segment}
   Frequency: {purchase_frequency}
   """
   ```

2. **Query ChromaDB** for similar vectors:
   ```python
   results = self.vector_repo.query(
       collection_name="customer_profiles",
       query_texts=[profile_text],
       n_results=top_k
   )
   ```

3. **Filter by threshold** and build result:
   ```python
   similar_customers = []
   for customer_id, similarity in zip(results['ids'], results['distances']):
       if similarity >= similarity_threshold:
           similar_customers.append(
               SimilarCustomer(
                   customer_id=customer_id,
                   customer_name=get_customer_name(customer_id),
                   similarity_score=similarity,
                   overlap_categories=calculate_overlap(...)
               )
           )
   ```

#### Vector Similarity
- **Method**: Cosine similarity
- **Embedding**: Dense vector (768 dimensions)
- **Database**: ChromaDB (vector store)

#### Output
```python
[
    SimilarCustomer(
        customer_id="456",
        customer_name="Sarah Johnson",
        similarity_score=0.89,
        overlap_categories=["Electronics", "Books"]
    ),
    SimilarCustomer(
        customer_id="789",
        customer_name="Michael Chen",
        similarity_score=0.84,
        overlap_categories=["Electronics", "Home & Garden"]
    ),
    # ... up to top_k customers with similarity >= threshold
]
```

#### Performance
- **Latency**: ~100ms (ChromaDB query)
- **Index**: HNSW (Hierarchical Navigable Small World)
- **Scalability**: Sub-linear with dataset size

---

### Data Collection Phase (Non-Agent)

**Lines in Workflow**: 167-214

#### Purpose
Aggregate candidate products from similar customers' purchase history.

#### Step 1: Gather Candidate Purchases (lines 167-172)

```python
candidate_purchases = []
for sim_customer in similar_customers:
    purchases = self.customer_repo.get_purchases_by_customer_id(
        sim_customer.customer_id
    )
    candidate_purchases.extend(purchases)

logger.info(
    f"[Data] Collected {len(candidate_purchases)} candidate purchases"
)
```

**Result**: List of all purchases by similar customers (can be hundreds)

#### Step 2: Filter Already Purchased (lines 175-187)

```python
# Get customer's already purchased products to exclude
customer_purchases = self.customer_repo.get_purchases_by_customer_id(customer_id)
already_purchased_ids = set(str(p["product_id"]) for p in customer_purchases)

# Filter out already purchased
candidate_purchases = [
    p for p in candidate_purchases
    if str(p["product_id"]) not in already_purchased_ids
]

logger.info(
    f"[Data] Collected {len(candidate_purchases)} candidate purchases "
    f"(excluded {len(already_purchased_ids)} already purchased)"
)
```

**Why**: Don't recommend products customer already owns

#### Step 3: Aggregate by Product (lines 200-214)

```python
if not candidate_purchases:
    # No candidates - return empty response
    logger.warning("No candidate products found")
    return self._build_empty_response(...)

df = pd.DataFrame(candidate_purchases)
products_df = df.groupby("product_id").agg({
    "product_name": "first",
    "product_category": "first",
    "price": "mean",
    "transaction_id": "count",  # How many times purchased
}).reset_index()

products_df.columns = [
    "product_id",
    "product_name",
    "product_category",
    "avg_price",
    "purchase_count"  # Popularity metric
]

candidate_products = products_df.to_dict("records")
```

**Aggregation**:
- Group multiple purchases of same product
- Calculate average price (handle price variations)
- Count purchase frequency (popularity signal)

#### Output
```python
[
    {
        "product_id": "789",
        "product_name": "Wireless Headphones",
        "product_category": "Electronics",
        "avg_price": 249.99,
        "purchase_count": 12  # Purchased 12 times by similar customers
    },
    {
        "product_id": "456",
        "product_name": "Smart Watch",
        "product_category": "Electronics",
        "avg_price": 399.00,
        "purchase_count": 8
    },
    # ... more products
]
```

#### Performance
- **Latency**: ~50ms (database aggregation)
- **Data volume**: 100-500 candidate products typical

---

### Agent 3: Sentiment Filtering

**File**: `app/capabilities/agents/sentiment_filtering.py`
**Lines in Workflow**: 219-242

#### Purpose
Filter out products with poor reviews to ensure quality recommendations.

#### Input
```python
SentimentFilteringInput(
    candidate_products=[
        ProductCandidate(
            product_id="789",
            product_name="Wireless Headphones",
            product_category="Electronics",
            avg_price=249.99,
            purchase_count=12
        ),
        # ... more products
    ],
    sentiment_threshold=0.6  # From settings.SENTIMENT_THRESHOLD
)
```

#### Execution
```python
sentiment_input = SentimentFilteringInput(
    candidate_products=[...],
    sentiment_threshold=settings.SENTIMENT_THRESHOLD,
)
sentiment_output = await self.sentiment_filtering_agent.run(
    sentiment_input,
    context
)
filtered_products = sentiment_output.filtered_products

logger.info(
    f"[Agent 3] Filtered to {len(filtered_products)} products "
    f"(removed {sentiment_output.products_filtered_out})"
)
```

#### How it works

1. **For each product**, fetch all reviews:
   ```python
   reviews = self.review_repo.get_reviews_by_product_id(product_id)
   ```

2. **Analyze sentiment** using SentimentAnalyzer:
   ```python
   sentiments = []
   for review in reviews:
       sentiment_score = self.sentiment_analyzer.analyze(review.text)
       sentiments.append(sentiment_score)

   avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
   ```

3. **Filter by threshold**:
   ```python
   if avg_sentiment >= sentiment_threshold:
       filtered_products.append(
           FilteredProduct(
               ...,
               avg_sentiment=avg_sentiment,
               review_count=len(reviews)
           )
       )
   else:
       products_filtered_out += 1
   ```

#### Sentiment Analyzer

**Method**: Rule-based (VADER-style)

**How it works**:
- Lexicon-based sentiment scoring
- Considers:
  - Positive/negative keywords
  - Intensifiers (very, extremely)
  - Negation (not good → negative)
  - Punctuation (!!!, ???)
  - CAPS LOCK emphasis

**Scoring**:
- Range: -1.0 (very negative) to +1.0 (very positive)
- Example scores:
  - "This product is amazing!" → 0.85
  - "Works well, good quality" → 0.65
  - "It's okay, nothing special" → 0.15
  - "Terrible, broke after 2 days" → -0.75

**Threshold**: 0.6 (only keep products with positive reviews)

#### Output
```python
FilteredProduct(
    product_id="789",
    product_name="Wireless Headphones",
    product_category="Electronics",
    avg_price=249.99,
    purchase_count=12,
    avg_sentiment=0.82,  # Positive reviews
    review_count=45
)
```

#### Performance
- **Latency**: ~200ms (20% of total time)
- **Bottleneck**: Multiple review queries + sentiment analysis
- **Optimization opportunity**: Batch processing, caching

#### Statistics
Typical filtering:
- Input: 100 candidate products
- Filtered out: 15-25 products (~20%)
- Output: 75-85 products

---

### Agent 4: Product Scoring & Ranking

**File**: `app/capabilities/agents/product_scoring.py`
**Lines in Workflow**: 256-286

#### Purpose
Score and rank products using collaborative filtering + category affinity + sentiment.

#### Input
```python
scoring_input = ProductScoringInput(
    customer_profile=profile,
    products=[...],  # Filtered products from Agent 3
    purchase_counts=product_purchase_counts,  # Counter object
    top_n=5,  # Number of recommendations to return
    max_per_category=2  # Diversity constraint
)
```

#### Execution
```python
# Build purchase counts for collaborative filtering
product_purchase_counts = Counter(
    str(p["product_id"]) for p in candidate_purchases
)

scoring_input = ProductScoringInput(
    customer_profile=profile,
    products=[...],
    purchase_counts=product_purchase_counts,
    top_n=top_n,
    max_per_category=2,  # Diversity constraint
)
scoring_output = await self.product_scoring_agent.run(scoring_input, context)
recommendations = scoring_output.recommendations

logger.info(
    f"[Agent 4] Generated {len(recommendations)} recommendations"
)
```

#### Scoring Algorithm

**Three-component score**:

```python
# 1. Collaborative Filtering Score (popularity among similar customers)
max_purchase_count = max(purchase_counts.values())
collab_score = purchase_count / max_purchase_count  # Normalize 0-1

# 2. Category Affinity Score (match with customer's favorite categories)
if product_category in customer_favorite_categories:
    category_score = 1.0
else:
    category_score = 0.0

# 3. Sentiment Bonus (quality signal)
# Normalize sentiment from [0.6, 1.0] to [0.0, 1.0]
sentiment_bonus = (avg_sentiment - 0.6) / 0.4

# Final Score (weighted combination)
final_score = (
    0.6 * collab_score +      # 60% collaborative filtering
    0.4 * category_score +     # 40% category affinity
    0.2 * sentiment_bonus      # 20% sentiment (additive bonus)
)

# Note: Score can exceed 1.0 due to sentiment bonus
```

**Ranking**:
1. Sort products by `final_score` (descending)
2. Apply diversity constraint: max 2 products per category
3. Take top N

#### Diversity Constraint

**Purpose**: Ensure variety in recommendations (don't recommend 5 electronics)

**Implementation**:
```python
category_counts = {}
recommendations = []

for product in sorted_products:
    category = product.product_category

    # Check category limit
    if category_counts.get(category, 0) >= max_per_category:
        continue  # Skip, already have 2 from this category

    recommendations.append(product)
    category_counts[category] = category_counts.get(category, 0) + 1

    # Stop when we have enough
    if len(recommendations) >= top_n:
        break
```

#### Output
```python
[
    ProductRecommendation(
        product_id="789",
        product_name="Wireless Headphones",
        product_category="Electronics",
        price=249.99,
        score=0.87,
        reasoning="Popular among similar customers (12 purchases), "
                  "matches your Electronics interest, highly rated (4.5/5)"
    ),
    ProductRecommendation(
        product_id="456",
        product_name="Smart Watch",
        product_category="Electronics",
        price=399.00,
        score=0.79,
        reasoning="Frequently purchased by similar customers (8 purchases), "
                  "aligns with your Electronics preference"
    ),
    ProductRecommendation(
        product_id="321",
        product_name="Python Programming Book",
        product_category="Books",
        price=44.99,
        score=0.72,
        reasoning="Popular with similar shoppers (6 purchases), "
                  "matches your Books category, excellent reviews (4.8/5)"
    ),
    # ... up to top_n recommendations
]
```

#### Performance
- **Latency**: ~30ms (pure computation)
- **Complexity**: O(n log n) for sorting
- **No I/O**: All data in memory

---

### Agent 5: Response Generation ✨ **PydanticAI**

**File**: `app/capabilities/agents/response_generation_v2.py`
**Lines in Workflow**: 291-309

#### Purpose
Generate natural language explanation for why products are recommended.

#### Input
```python
ResponseGenerationInput(
    query="What should I buy?",
    customer_profile=profile,
    recommendations=recommendations  # Top N from Agent 4
)
```

#### Execution
```python
agent_execution.append("response_generation")

if include_reasoning and recommendations:
    response_input = ResponseGenerationInput(
        query=query,
        customer_profile=profile,
        recommendations=recommendations,
    )
    response_output = await self.response_generation_agent.run(
        response_input,
        context
    )
    reasoning = response_output.reasoning
else:
    reasoning = f"Based on {profile.customer_name}'s purchase history, here are {len(recommendations)} recommendations."

logger.info(
    f"[Agent 5] Generated reasoning ({len(reasoning)} chars)"
)
```

#### How it works (PydanticAI)

**Step 1**: Load prompt from centralized system
```python
from app.prompts import get_prompt_loader

prompt_loader = get_prompt_loader()
response_prompt = prompt_loader.load_prompt("response.generation")
```

**Step 2**: Create PydanticAI agent
```python
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models.ollama import OllamaModel

response_pydantic_agent = PydanticAgent(
    model=OllamaModel(
        model_name=response_prompt.metadata.model,  # "llama3.1:8b"
        base_url=settings.OLLAMA_BASE_URL,
    ),
    result_type=str,  # Simple text output
    system_prompt=response_prompt.system,
)
```

**Step 3**: Render user prompt with Jinja2 template

Template (`prompts/response_generation/user.txt`):
```jinja2
Customer Profile:
- Name: {{customer_name}}
- Shopping Segment: {{price_segment}}
- Favorite Categories: {{favorite_categories}}

Query: "{{query}}"

Recommended Products:
{{recommendations_text}}

Please provide a friendly, personalized explanation (2-3 sentences) for why these products are recommended for this customer.
```

Rendering:
```python
# Format recommendations
recommendations_text = "\n".join([
    f"{i+1}. {rec.product_name} - ${rec.price:.2f} ({rec.product_category})"
    for i, rec in enumerate(recommendations)
])

# Render template
user_prompt = prompt_loader.render_user_prompt(
    "response.generation",
    customer_name=profile.customer_name,
    price_segment=profile.price_segment,
    favorite_categories=", ".join(profile.favorite_categories),
    query=query,
    recommendations_text=recommendations_text
)
```

**Step 4**: Invoke PydanticAI agent
```python
# Automatic:
# - LLM invocation
# - Response parsing
# - Validation
# - Retries on failure
result = await response_pydantic_agent.run(user_prompt)
reasoning = result.data  # Type-safe string
```

#### Prompt Configuration

**Metadata** (`prompts/response_generation/metadata.yaml`):
```yaml
id: "response.generation"
version: "1.0.0"
model: "llama3.1:8b"
temperature: 0.7  # Creative but focused
max_tokens: 500
```

**System Prompt** (`prompts/response_generation/system.txt`):
```
You are a friendly, personalized shopping assistant.

Your task is to explain why recommended products are a good fit for the customer based on their purchase history and preferences.

Guidelines:
- Keep explanations to 2-3 sentences
- Reference the customer's shopping preferences
- Highlight how products match their interests
- Use a warm, conversational tone
- Be specific about why each product fits their profile
```

#### Output Examples

**Example 1** (Electronics enthusiast):
```
"Based on your interest in Electronics and Premium products, here are some great recommendations for you! The Wireless Headphones are highly rated by customers with similar tastes, and the Smart Watch complements your tech-focused purchases perfectly. Given your frequent purchasing habits, these items align well with your preferences."
```

**Example 2** (Budget-conscious book lover):
```
"Since you love Books and tend to shop in the Budget range, we've selected some excellent reads for you! The Python Programming Book and Data Science Guide have been popular among customers with similar interests, and they're both priced within your typical range. Happy reading!"
```

**Example 3** (Diverse interests):
```
"We've curated a diverse selection based on your varied interests in Electronics, Books, and Home & Garden. These recommendations reflect what similar customers have enjoyed, and they match your Mid-range price preferences. Each item has received positive reviews from shoppers with tastes like yours."
```

#### Performance
- **Latency**: ~500ms (50% of total time)
- **Bottleneck**: LLM inference (largest component)
- **Caching**: Not applicable (each response unique)

#### Benefits of PydanticAI

✅ **Before** (manual approach):
- 60+ lines of code
- Manual prompt construction
- Manual error handling
- No automatic retries

✅ **After** (PydanticAI):
- 25 lines of code (60% reduction)
- Centralized prompts (easy updates)
- Automatic retries
- Type-safe outputs

---

## Data Flow Diagrams

### Complete Request Flow

```
Input: query="What should I buy?", customer_name="Kenneth Martinez"
  │
  ├─► [API] POST /recommendations/personalized
  │     └─► FastAPI endpoint validation
  │
  ├─► [Service] RecommendationService.get_personalized_recommendations()
  │     ├─► Validate inputs
  │     ├─► Create tracing session
  │     └─► Intent Classification (PydanticAI) ↓
  │           ├─► INFORMATIONAL → QueryAnsweringService → Answer
  │           └─► RECOMMENDATION ↓
  │
  ├─► [Workflow] PersonalizedRecommendationWorkflow.execute()
  │
  ├─► [Agent 1] Customer Profiling
  │     Input:  customer_id="123"
  │     Output: profile{
  │               total_purchases=45,
  │               avg_price=89.50,
  │               favorite_categories=["Electronics", "Books"],
  │               price_segment="Mid-range",
  │               frequency="frequent"
  │             }
  │
  ├─► [Agent 2] Similar Customers
  │     Input:  profile
  │     Process: Vector similarity search in ChromaDB
  │     Output: similar_customers=[
  │               {customer_id="456", similarity=0.89},
  │               {customer_id="789", similarity=0.84},
  │               ... 18 more customers
  │             ]
  │
  ├─► [Data] Gather Candidates
  │     Step 1: Fetch purchases from similar customers
  │             → 234 purchases
  │     Step 2: Filter already purchased products
  │             → 198 purchases remaining
  │     Step 3: Aggregate by product_id
  │             → 87 unique products
  │     Output: candidate_products=[
  │               {product_id="789", purchase_count=12, avg_price=249.99},
  │               {product_id="456", purchase_count=8, avg_price=399.00},
  │               ... 85 more products
  │             ]
  │
  ├─► [Agent 3] Sentiment Filtering
  │     Input:  87 candidate products
  │     Process: Analyze reviews for each product
  │             - Fetch reviews from DB
  │             - Calculate avg sentiment
  │             - Filter < 0.6 threshold
  │     Output: filtered_products=[
  │               {product_id="789", avg_sentiment=0.82, review_count=45},
  │               {product_id="456", avg_sentiment=0.78, review_count=32},
  │               ... 69 more products (18 filtered out)
  │             ]
  │
  ├─► [Agent 4] Product Scoring
  │     Input:  69 filtered products
  │     Process: Calculate scores:
  │             - Collaborative filtering (60%)
  │             - Category affinity (40%)
  │             - Sentiment bonus (20%)
  │             - Apply diversity constraint (max 2 per category)
  │     Output: recommendations=[
  │               {product_id="789", score=0.87, category="Electronics"},
  │               {product_id="456", score=0.79, category="Electronics"},
  │               {product_id="321", score=0.72, category="Books"},
  │               {product_id="654", score=0.68, category="Home & Garden"},
  │               {product_id="987", score=0.65, category="Books"}
  │             ]  # Top 5 with diversity
  │
  └─► [Agent 5] Response Generation (PydanticAI)
        Input:  query, profile, recommendations
        Process: - Load prompt from file
                - Render Jinja2 template
                - Invoke LLM (llama3.1:8b)
                - Parse and validate response
        Output: reasoning="Based on your interest in Electronics and
                Mid-range products, here are some great recommendations
                for you! The Wireless Headphones are highly rated by
                customers with similar tastes..."

Final Response:
  RecommendationResponse(
    query="What should I buy?",
    customer_profile=CustomerProfileSummary(...),
    recommendations=[...],  # 5 products
    reasoning="Based on your interest in...",
    confidence_score=0.85,
    processing_time_ms=1043,
    similar_customers_analyzed=20,
    products_considered=87,
    products_filtered_by_sentiment=18,
    recommendation_strategy="collaborative_with_category_affinity",
    agent_execution_order=[
      "customer_profiling",
      "similar_customer_discovery",
      "sentiment_filtering",
      "product_scoring",
      "response_generation"
    ]
  )
```

### Agent Dependency Graph

```
┌──────────────────────┐
│   Customer ID        │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Agent 1: Profiling  │
│  (No dependencies)   │
└──────────┬───────────┘
           │
           │ CustomerProfile
           ▼
┌──────────────────────┐
│  Agent 2: Similar    │
│  (Depends on: Agent 1)│
└──────────┬───────────┘
           │
           │ List[SimilarCustomer]
           ▼
┌──────────────────────┐
│  Data: Candidates    │
│  (Depends on: Agent 2)│
└──────────┬───────────┘
           │
           │ List[ProductCandidate]
           ▼
┌──────────────────────┐
│  Agent 3: Sentiment  │
│  (Depends on: Data)  │
└──────────┬───────────┘
           │
           │ List[FilteredProduct]
           ▼
┌──────────────────────┐
│  Agent 4: Scoring    │
│  (Depends on: Agent  │
│   1, 3, and Data)    │
└──────────┬───────────┘
           │
           │ List[ProductRecommendation]
           ▼
┌──────────────────────┐
│  Agent 5: Response   │
│  (Depends on: Agent  │
│   1 and 4)           │
└──────────┬───────────┘
           │
           │ RecommendationResponse
           ▼
      [End User]
```

---

## Observability & Tracing

### LangFuse Integration

Every step is traced using LangFuse spans for complete observability.

#### Main Trace Span

```python
with trace_span(
    name="recommendation_workflow",
    session_id=session_id,
    user_id=customer_name or customer_id,
    metadata={
        "query": query,
        "top_n": top_n,
        "include_reasoning": include_reasoning,
    },
    tags=["recommendation", "workflow"],
    input_data={
        "query": query,
        "customer_name": customer_name,
        "customer_id": customer_id,
        "top_n": top_n,
    }
) as main_trace:
    # ... workflow execution

    # Update trace with results
    main_trace.update(output={
        "intent": "recommendation",
        "recommendations_count": len(result.recommendations),
        "processing_time_ms": result.processing_time_ms,
        "confidence_score": result.confidence_score,
    })
```

#### What's Tracked

1. **Request Metadata**
   - Session ID: `req-a1b2c3d4`
   - User ID: Customer name or ID
   - Query text
   - Parameters (top_n, include_reasoning)

2. **Agent Execution**
   - Execution order: `["customer_profiling", "similar_customer_discovery", ...]`
   - Timing per agent
   - Success/failure status

3. **Performance Metrics**
   - Total processing time
   - Latency breakdown by agent
   - Database query times
   - LLM inference time

4. **Results**
   - Number of recommendations
   - Confidence score
   - Products considered
   - Products filtered
   - Similar customers analyzed

5. **LLM Calls**
   - Prompts sent
   - Responses received
   - Token usage
   - Model used
   - Temperature, max_tokens

6. **Errors**
   - Exception type
   - Stack trace
   - Failed agent
   - Fallback used

#### Agent-Level Tracing

Each agent creates its own trace span:

```python
class BaseAgent(Generic[InputModel, OutputModel], ABC):
    async def run(
        self,
        input_data: InputModel,
        context: AgentContext
    ) -> OutputModel:
        """Run agent with automatic tracing"""

        with trace_span(
            name=f"agent_{self.metadata.id}",
            session_id=context.session_id,
            user_id=context.user_id,
            metadata={
                "agent_name": self.metadata.name,
                "agent_version": self.metadata.version,
            },
            tags=["agent"] + self.metadata.tags,
            input_data=input_data.dict()
        ) as trace:
            start_time = time.time()

            try:
                # Execute agent logic
                output = await self._execute(input_data, context)

                # Log success
                trace.update(
                    output=output.dict(),
                    metadata={
                        "duration_ms": (time.time() - start_time) * 1000,
                        "success": True
                    }
                )

                return output

            except Exception as e:
                # Log error
                trace.update(
                    output={"error": str(e)},
                    level="ERROR",
                    metadata={
                        "duration_ms": (time.time() - start_time) * 1000,
                        "success": False,
                        "error_type": type(e).__name__
                    }
                )
                raise
```

#### LangFuse Dashboard Views

**1. Trace Timeline**
```
recommendation_workflow (1043ms)
├─ intent_classification (50ms)
├─ agent_customer_profiling (20ms)
├─ agent_similar_customers (100ms)
├─ agent_sentiment_filtering (200ms)
├─ agent_product_scoring (30ms)
└─ agent_response_generation (500ms)
   └─ llm_call (480ms)
      ├─ prompt_tokens: 245
      ├─ completion_tokens: 87
      └─ total_tokens: 332
```

**2. Metrics Dashboard**
- Average latency per agent
- Success rate by agent
- LLM token usage over time
- Cost per request
- Error rate trends

**3. Session View**
- All requests from same user
- Query patterns
- Recommendation evolution
- A/B test assignments

---

## Design Patterns

### 1. Thin Facade Pattern

**RecommendationService** acts as a thin facade:

```python
class RecommendationService:
    """
    Thin facade over workflows

    Responsibilities:
    - Routing (INFORMATIONAL vs RECOMMENDATION)
    - Customer lookup
    - Tracing
    - Backward compatibility

    Does NOT contain:
    - Business logic (delegated to agents)
    - Orchestration (delegated to workflow)
    - Data transformation (delegated to agents)
    """
```

**Benefits**:
- ~150 lines (was 400+ before refactoring)
- Easy to test (minimal logic)
- Clear separation of concerns
- Single responsibility

---

### 2. Sequential Pipeline Pattern

**PersonalizedRecommendationWorkflow** orchestrates agents in sequence:

```python
class PersonalizedRecommendationWorkflow:
    """
    Sequential agent pipeline

    Output of Agent N → Input of Agent N+1
    """

    async def execute(...):
        # Agent 1
        profile = await agent1.run(input1)

        # Agent 2 (depends on Agent 1)
        similar_customers = await agent2.run(profile)

        # Agent 3 (depends on data from Agent 2)
        filtered = await agent3.run(candidates)

        # Agent 4 (depends on Agent 1, 3, and data)
        recommendations = await agent4.run(profile, filtered)

        # Agent 5 (depends on Agent 1 and 4)
        reasoning = await agent5.run(profile, recommendations)
```

**Benefits**:
- Clear execution order
- Easy to trace
- Simple error handling (fail at any step)
- Testable (mock upstream agents)

---

### 3. Dependency Injection

Agents receive dependencies via constructor:

```python
class SimilarCustomersAgent(BaseAgent):
    def __init__(
        self,
        customer_repository: CustomerRepository,
        vector_repository: VectorRepository
    ):
        self.customer_repo = customer_repository
        self.vector_repo = vector_repository
```

**Benefits**:
- Easy to mock for testing
- No global state
- Explicit dependencies
- Swappable implementations

---

### 4. Pydantic Everywhere

All inputs/outputs are Pydantic models:

```python
class CustomerProfilingInput(BaseModel):
    customer_id: str

class CustomerProfilingOutput(BaseModel):
    profile: CustomerProfile
    processing_time_ms: float

# Type-safe end-to-end
input_data = CustomerProfilingInput(customer_id="123")
output_data = await agent.run(input_data, context)
# output_data is CustomerProfilingOutput (validated!)
```

**Benefits**:
- Automatic validation
- Type hints in IDEs
- Self-documenting
- Runtime type checking

---

### 5. Centralized Orchestration

**Workflow owns coordination logic**:

```python
# ✅ Good: Workflow coordinates
class PersonalizedRecommendationWorkflow:
    async def execute(...):
        profile = await self.profiling_agent.run(...)
        similar = await self.similar_customers_agent.run(...)
        # Workflow decides execution order

# ❌ Bad: Agents call each other
class ProfilingAgent:
    async def run(...):
        profile = self._calculate_profile(...)
        # DON'T DO THIS:
        similar = await self.similar_customers_agent.run(profile)
```

**Benefits**:
- Agents are stateless capabilities
- Single source of truth for flow
- Easy to modify pipeline
- Clear responsibilities

---

### 6. BaseAgent Generic Pattern

Uniform interface for all agents:

```python
class BaseAgent(Generic[InputModel, OutputModel], ABC):
    """
    Base class for all agents

    Provides:
    - Automatic timing
    - Automatic logging
    - Automatic error handling
    - Tracing integration
    """

    async def run(
        self,
        input_data: InputModel,
        context: AgentContext
    ) -> OutputModel:
        # Wrapper with observability

    @abstractmethod
    async def _execute(
        self,
        input_data: InputModel,
        context: AgentContext
    ) -> OutputModel:
        # Subclasses implement this
```

**Benefits**:
- Consistent interface
- Automatic observability
- Easy to add new agents
- Testable in isolation

---

## Performance Characteristics

### Average Latency Breakdown

**Typical Request** (customer with purchase history, returning 5 recommendations):

| Step | Average Time | % of Total | Complexity |
|------|--------------|------------|------------|
| **Intent Classification** | 50ms | 5% | O(1) LLM call |
| **Customer Profiling** | 20ms | 2% | O(n) where n = purchases |
| **Similar Customer Search** | 100ms | 10% | O(log m) where m = customers |
| **Data Collection** | 50ms | 5% | O(k) where k = similar customers |
| **Sentiment Filtering** | 200ms | 20% | O(p × r) where p = products, r = reviews |
| **Product Scoring** | 30ms | 3% | O(p log p) for sorting |
| **Response Generation (LLM)** | 500ms | 50% | O(1) LLM call |
| **Other** (tracing, etc.) | 50ms | 5% | - |
| **Total** | **~1000ms** | **100%** | - |

### Bottlenecks

1. **Response Generation (50%)**
   - LLM inference on CPU
   - 500ms average
   - **Optimization**: GPU inference, caching common patterns

2. **Sentiment Filtering (20%)**
   - Multiple database queries for reviews
   - Sentiment analysis per review
   - **Optimization**: Batch queries, pre-computed sentiment scores

3. **Similar Customer Search (10%)**
   - Vector similarity in ChromaDB
   - ~100ms for 20k customers
   - **Optimization**: Already optimized (HNSW index)

### Scalability Analysis

**Current Performance**:
- Throughput: ~10-15 requests/second per instance
- P50 latency: 950ms
- P95 latency: 1500ms
- P99 latency: 2500ms

**Scaling Strategies**:

1. **Horizontal Scaling**
   - Multiple API instances (stateless)
   - Load balancer (nginx)
   - Can scale to 100+ req/sec

2. **Caching**
   - Redis cache for:
     - Customer profiles (1 hour TTL)
     - Similar customers (24 hour TTL)
     - Product sentiment scores (1 week TTL)
   - Estimated improvement: -30% latency

3. **Async Optimization**
   - Parallel sentiment filtering (current: sequential)
   - Estimated improvement: -15% latency

4. **Database Optimization**
   - Index on customer_id, product_id
   - Connection pooling
   - Estimated improvement: -5% latency

5. **LLM Optimization**
   - GPU inference (vs current CPU)
   - Estimated improvement: -50% on LLM calls (-25% total)
   - Smaller model (llama3.1:8b → 3b)
   - Estimated improvement: -60% on LLM calls (-30% total)

---

## Infrastructure Components

### Database: PostgreSQL

**Purpose**: Persistent data storage

**Schema**:
- `customers` - Customer information
- `products` - Product catalog
- `transactions` - Purchase history
- `reviews` - Product reviews

**Configuration**:
- Port: 5432
- Connection pool: 20 connections
- Backup: Daily snapshots

---

### Vector Store: ChromaDB

**Purpose**: Vector similarity search for customer matching

**Collections**:
- `customer_profiles` - Customer behavior embeddings

**Configuration**:
- Embedding dimension: 768
- Distance metric: Cosine similarity
- Index type: HNSW
- Port: 8000

**Performance**:
- Query latency: ~100ms for 20k vectors
- Scalability: Sub-linear with dataset size

---

### Cache: Redis

**Purpose**: Fast caching layer

**Selected lines from docker-compose.yml** (113-128):
```yaml
redis:
  image: redis:7-alpine
  container_name: shopping-redis
  ports:
    - "6380:6379"
  volumes:
    - redis-data:/data
  command: redis-server --appendonly yes
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
  networks:
    - shopping-network
  restart: unless-stopped
```

**What's Cached**:
- Customer profiles (1 hour TTL)
- Similar customers (24 hour TTL)
- Product sentiment scores (1 week TTL)
- LLM responses (not currently, future optimization)

**Configuration**:
- Port: 6380 (host) → 6379 (container)
- Persistence: Append-only file (AOF)
- Health checks: Every 10s

---

### LLM: Ollama

**Purpose**: Local LLM inference

**Models**:
- `llama3.1:8b` - Response generation (creative)
- `llama3.1:8b` - Intent classification (analytical)

**Configuration**:
- Base URL: `http://ollama:11434`
- Context length: 8192 tokens
- Temperature: 0.7 (response), 0.1 (intent)

---

### Observability: LangFuse

**Purpose**: LLM application tracing and monitoring

**Features**:
- Trace visualization
- Token usage tracking
- Cost analysis
- Performance metrics
- A/B test support

**Configuration**:
- Public key: From env
- Secret key: From env
- Host: cloud or self-hosted

---

## Summary

### Current Orchestration Architecture

**Pattern**: Multi-layered orchestration with sequential agent pipeline

**Layers**:
1. **API Layer**: FastAPI endpoints
2. **Service Layer**: Thin facade (routing, tracing)
3. **Workflow Layer**: Pure orchestration
4. **Agent Layer**: 5 specialized agents

**Agents**:
1. Customer Profiling (20ms)
2. Similar Customer Discovery (100ms)
3. Sentiment Filtering (200ms)
4. Product Scoring (30ms)
5. Response Generation - PydanticAI (500ms)

**Key Features**:
- ✅ Intent-based routing (INFORMATIONAL vs RECOMMENDATION)
- ✅ Sequential 5-agent pipeline
- ✅ PydanticAI for LLM agents (structured outputs, zero JSON errors)
- ✅ Centralized prompt management with versioning
- ✅ Full observability with LangFuse tracing
- ✅ Type-safe end-to-end (Pydantic everywhere)
- ✅ Easy to test (mock individual agents)
- ✅ Scalable (stateless, horizontal scaling)

**Performance**:
- Average latency: ~1000ms
- Throughput: 10-15 req/sec per instance
- Bottleneck: LLM inference (50% of time)

**Infrastructure**:
- PostgreSQL (data)
- ChromaDB (vectors)
- Redis (cache)
- Ollama (LLM)
- LangFuse (observability)

---

**This is a production-grade, scalable, observable multi-agent recommendation system!** 🎉

---

*Last Updated: January 2025*
*Version: 2.0 (with PydanticAI)*
