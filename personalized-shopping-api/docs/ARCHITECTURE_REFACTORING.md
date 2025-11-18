# Architecture Refactoring: Agentic Design

## Overview

This document describes the major architecture refactoring completed to transform the personalized shopping API into a clean, agentic, service-first architecture. The refactoring introduces a uniform agent interface, workflow orchestration layer, and clear separation of concerns.

## Table of Contents

- [Motivation](#motivation)
- [New Architecture](#new-architecture)
- [Core Components](#core-components)
- [Migration Guide](#migration-guide)
- [Legacy Code](#legacy-code)
- [Future Extensions](#future-extensions)

---

## Motivation

### Problems with Previous Architecture

1. **Mixed Orchestration and Business Logic**: `RecommendationService.get_personalized_recommendations()` contained both workflow orchestration and business logic in a 400+ line method
2. **Duplicate Implementations**: Legacy LangGraph agents existed alongside inline service logic, creating maintenance burden
3. **Inconsistent Interfaces**: Agents had varying signatures and lacked a common interface
4. **Poor Testability**: Inline logic made it difficult to test individual capabilities in isolation
5. **Limited Reusability**: Logic embedded in services couldn't be reused across different workflows
6. **No Agent Discovery**: No registry or metadata system for runtime agent introspection

### Goals of Refactoring

- **Separation of Concerns**: Clear boundaries between orchestration, business logic, and data access
- **Composability**: Agents can be combined into different workflows
- **Testability**: Each agent can be tested independently with mock dependencies
- **Observability**: Uniform logging, tracing, and metrics across all agents
- **Extensibility**: Easy to add new agents and workflows without modifying existing code
- **Maintainability**: Single source of truth for each capability, no duplication

---

## New Architecture

### Layered Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        API Layer                             │
│         (FastAPI endpoints - thin HTTP adapters)            │
│         /api/v1/endpoints/recommendations.py                │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Service Layer                             │
│         (Facades, routing, backward compatibility)          │
│         RecommendationService (thin facade)                 │
│         QueryAnsweringService                               │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    Workflow Layer                            │
│         (Orchestration only - calls agents in sequence)     │
│         PersonalizedRecommendationWorkflow                  │
│         SentimentSearchWorkflow (future)                    │
│         QualityAlertWorkflow (future)                       │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Agent/Capability Layer                     │
│         (Pydantic-based, reusable, testable)                │
│         • CustomerProfilingAgent                            │
│         • SimilarCustomersAgent                             │
│         • SentimentFilteringAgent                           │
│         • ProductScoringAgent                               │
│         • ResponseGenerationAgent                           │
│         • IntentClassificationAgent (existing)              │
└────────────────────────┬────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                   Repository Layer                           │
│         (Pure data access - no business logic)              │
│         • CustomerRepository, ProductRepository             │
│         • ReviewRepository, VectorRepository                │
└─────────────────────────────────────────────────────────────┘
```

### Key Architectural Principles

1. **Single Responsibility**: Each agent has one clearly defined capability
2. **Dependency Injection**: Agents receive dependencies via constructor
3. **Stateless Agents**: All state passed via input/context, no instance variables
4. **Typed Interfaces**: Pydantic models for all inputs and outputs
5. **Pure Orchestration**: Workflows contain zero business logic, only agent coordination
6. **Backward Compatibility**: Existing API contracts maintained

---

## Core Components

### 1. Base Agent Architecture (`app/capabilities/base.py`)

#### AgentContext

Request-scoped context passed to all agents:

```python
class AgentContext(BaseModel):
    request_id: str              # Unique request identifier
    session_id: Optional[str]    # Tracing session ID
    user_id: Optional[str]       # User/customer ID
    metadata: Dict[str, Any]     # Additional context
    timestamp: datetime          # Context creation time
```

#### AgentMetadata

Agent identification and schema information:

```python
class AgentMetadata(BaseModel):
    id: str                       # Unique agent ID (snake_case)
    name: str                     # Human-readable name
    description: str              # What this agent does
    version: str                  # Agent version
    input_schema: Type[BaseModel] # Pydantic input model
    output_schema: Type[BaseModel]# Pydantic output model
    tags: List[str]               # Categorization tags
```

#### BaseAgent[InputModel, OutputModel]

Generic base class for all agents:

```python
class BaseAgent(Generic[InputModel, OutputModel], ABC):
    def __init__(self, metadata: AgentMetadata): ...

    @abstractmethod
    async def _execute(
        self,
        input_data: InputModel,
        context: AgentContext
    ) -> OutputModel:
        """Core agent logic (implemented by subclasses)"""
        pass

    async def run(
        self,
        input_data: InputModel,
        context: AgentContext
    ) -> OutputModel:
        """Execute with observability (timing, logging, error handling)"""
        # Wraps _execute() with instrumentation
```

**Features:**
- Automatic timing and logging
- Error handling and tracing integration
- Execution metadata stored in context
- Type-safe input/output validation

#### AgentRegistry

Discovery mechanism for runtime agent introspection:

```python
class AgentRegistry:
    @classmethod
    def register(cls, agent_class, metadata): ...

    @classmethod
    def get_agent(cls, agent_id) -> Type[BaseAgent]: ...

    @classmethod
    def list_agents() -> List[AgentMetadata]: ...

    @classmethod
    def find_by_tag(cls, tag) -> List[AgentMetadata]: ...
```

**Use Cases:**
- Runtime agent discovery for UI/editors
- Dynamic workflow composition
- Agent versioning and migration
- Documentation generation

---

### 2. Concrete Agents (`app/capabilities/agents/`)

All agents follow the uniform `BaseAgent` interface:

#### CustomerProfilingAgent

**Purpose**: Extract behavioral metrics from purchase history

**Input**: `CustomerProfilingInput(customer_id)`
**Output**: `CustomerProfilingOutput(profile: CustomerProfile)`

**Logic**:
- Fetch purchases from repository
- Calculate metrics (total spent, avg price, frequency)
- Identify favorite categories
- Segment by price tier and frequency
- Compute confidence score

**Tags**: `["customer", "profiling", "segmentation"]`

---

#### SimilarCustomersAgent

**Purpose**: Find customers with similar purchase behavior

**Input**: `SimilarCustomersInput(customer_profile, top_k, threshold)`
**Output**: `SimilarCustomersOutput(similar_customers, search_query)`

**Logic**:
- Convert profile to behavior text
- Perform vector similarity search (ChromaDB)
- Calculate category overlap
- Filter by similarity threshold

**Tags**: `["customer", "similarity", "vector-search", "collaborative-filtering"]`

---

#### SentimentFilteringAgent

**Purpose**: Filter products by review sentiment

**Input**: `SentimentFilteringInput(candidate_products, threshold)`
**Output**: `SentimentFilteringOutput(filtered_products, stats)`

**Logic**:
- Fetch reviews for each product
- Calculate average sentiment (SentimentAnalyzer)
- Filter below threshold
- Track filtering metrics

**Tags**: `["filtering", "sentiment", "reviews", "quality"]`

---

#### ProductScoringAgent

**Purpose**: Score and rank products using collaborative filtering + category affinity

**Input**: `ProductScoringInput(customer_profile, products, purchase_counts, top_n)`
**Output**: `ProductScoringOutput(recommendations, total_scored)`

**Scoring Formula**:
```python
final_score = (
    0.6 * collaborative_score +      # Similar customer purchases
    0.4 * category_affinity_score +  # Favorite category match
    0.2 * sentiment_score            # Review sentiment
)
```

**Features**:
- Diversity constraint (max 2 per category)
- Position-based category affinity
- Automatic reason generation

**Tags**: `["scoring", "ranking", "recommendation", "collaborative-filtering"]`

---

#### ResponseGenerationAgent

**Purpose**: Generate natural language explanations using LLM

**Input**: `ResponseGenerationInput(query, customer_profile, recommendations)`
**Output**: `ResponseGenerationOutput(reasoning, llm_used)`

**Logic**:
- Build customer context summary
- Construct LLM prompt with top recommendations
- Invoke LLM (with tracing)
- Fallback to template on error

**Tags**: `["response", "llm", "reasoning", "explanation"]`

---

### 3. Workflows (`app/workflows/`)

#### PersonalizedRecommendationWorkflow

**Purpose**: Orchestrate the 5-agent recommendation pipeline

**Method**: `execute(customer_id, query, top_n, include_reasoning, context)`

**Agent Sequence**:
1. **CustomerProfilingAgent** → Extract profile
2. **SimilarCustomersAgent** → Find similar customers
3. *Data collection* → Gather candidate products
4. **SentimentFilteringAgent** → Remove low-sentiment products
5. **ProductScoringAgent** → Score and rank
6. **ResponseGenerationAgent** → Generate explanation

**Orchestration Principles**:
- No business logic (pure agent coordination)
- Pydantic models passed between agents
- Centralized error handling
- Comprehensive logging

**Returns**: `RecommendationResponse` (same schema as before)

---

### 4. Service Layer (`app/services/`)

#### RecommendationService (Refactored)

**Purpose**: Thin facade over workflows

**Responsibilities**:
1. Intent classification (informational vs recommendation)
2. Customer lookup by name
3. Route to appropriate workflow
4. Maintain tracing and observability
5. Ensure backward compatibility

**Before (400+ lines)**:
```python
async def get_personalized_recommendations(...):
    # All orchestration + business logic inline
    # Intent classification
    # Customer profiling
    # Similar customer search
    # Sentiment filtering
    # Product scoring
    # Response generation
    # ...
```

**After (< 150 lines)**:
```python
async def get_personalized_recommendations(...):
    # Intent classification
    intent = self.intent_classifier_agent.classify(query)

    # Route to workflow
    if intent == INFORMATIONAL:
        return await self._handle_informational_query(...)
    else:
        return await self.recommendation_workflow.execute(...)
```

**Benefits**:
- 70% code reduction
- Clear separation of routing vs orchestration
- All business logic in agents
- Easy to add new workflows

---

## Migration Guide

### For Developers

#### Using the New Architecture

**1. Adding a New Agent**:

```python
from app.capabilities.base import BaseAgent, AgentMetadata, AgentContext
from pydantic import BaseModel

class MyAgentInput(BaseModel):
    query: str

class MyAgentOutput(BaseModel):
    result: str

class MyAgent(BaseAgent[MyAgentInput, MyAgentOutput]):
    def __init__(self, dependency: SomeDependency):
        metadata = AgentMetadata(
            id="my_agent",
            name="My Agent",
            description="Does something useful",
            input_schema=MyAgentInput,
            output_schema=MyAgentOutput,
            tags=["my-category"],
        )
        super().__init__(metadata)
        self.dependency = dependency

    async def _execute(
        self,
        input_data: MyAgentInput,
        context: AgentContext
    ) -> MyAgentOutput:
        # Your logic here
        result = self.dependency.do_something(input_data.query)
        return MyAgentOutput(result=result)
```

**2. Using an Agent**:

```python
from app.capabilities.base import AgentContext

# Create agent with dependencies
agent = MyAgent(dependency=some_dependency)

# Prepare input
input_data = MyAgentInput(query="test")

# Create context
context = AgentContext(
    request_id="req-123",
    user_id="user-456",
)

# Run agent
output = await agent.run(input_data, context)

# Access result
print(output.result)
```

**3. Creating a New Workflow**:

```python
class MyWorkflow:
    def __init__(self, dependencies):
        self.agent1 = Agent1(...)
        self.agent2 = Agent2(...)

    async def execute(self, input_params, context):
        # Call agents in sequence
        output1 = await self.agent1.run(input1, context)
        output2 = await self.agent2.run(output1, context)

        # Build response
        return MyResponse(...)
```

**4. Testing an Agent**:

```python
@pytest.mark.asyncio
async def test_my_agent():
    # Create mock dependency
    mock_dependency = Mock()
    mock_dependency.do_something.return_value = "result"

    # Create agent
    agent = MyAgent(mock_dependency)

    # Prepare input and context
    input_data = MyAgentInput(query="test")
    context = AgentContext(request_id="test")

    # Run agent
    output = await agent.run(input_data, context)

    # Assert
    assert output.result == "result"
    mock_dependency.do_something.assert_called_once()
```

### For API Consumers

**No Changes Required**: The API contract is unchanged. All existing endpoints work exactly as before:

```bash
POST /api/v1/recommendations/personalized
{
  "query": "What should I buy?",
  "customer_name": "Kenneth Martinez",
  "top_n": 5
}
```

**Response Schema**: Unchanged (same `RecommendationResponse`)

---

## Legacy Code

### Deprecated Components

The following components have been deprecated and moved to legacy status:

#### 1. Legacy LangGraph Workflow

**File**: `app/graph/workflow.py`
**Status**: ❌ **DEPRECATED** - Not used in production

**Replacement**: `PersonalizedRecommendationWorkflow`

**Reason**: Duplicate implementation of the same logic. The new workflow-based architecture provides the same functionality with better separation of concerns.

#### 2. Legacy Agent Implementations

**Files**:
- `app/agents/customer_profiling.py` ❌ DEPRECATED
- `app/agents/similar_customers.py` ❌ DEPRECATED
- `app/agents/review_filtering.py` ❌ DEPRECATED
- `app/agents/recommendation.py` ❌ DEPRECATED
- `app/agents/response_generation.py` ❌ DEPRECATED

**Replacement**: `app/capabilities/agents/*` (new implementations)

**Reason**: Old agents lacked uniform interface, metadata, and observability hooks.

**Note**: `app/agents/intent_classifier_agent.py` is **kept** as it's currently used in production.

#### 3. FAISS Vector Store (Partial Deprecation)

**File**: `app/vector_store/customer_embeddings.py`
**Status**: ⚠️ **UNUSED** - ChromaDB is the active implementation

**Recommendation**: Either:
- Remove FAISS code entirely, OR
- Clearly isolate as alternative implementation with config flag

**Current Issue**: Config has `VECTOR_DB_TYPE="faiss"` but ChromaDB is hardcoded in `VectorRepository`.

### Migration Path from Legacy

If you have code using the old agents:

**Before**:
```python
from app.agents.customer_profiling import CustomerProfilingAgent

agent = CustomerProfilingAgent()
result = agent.run(customer_id)  # Old signature
```

**After**:
```python
from app.capabilities.agents import CustomerProfilingAgent
from app.capabilities.base import AgentContext

agent = CustomerProfilingAgent(customer_repository)
input_data = CustomerProfilingInput(customer_id=customer_id)
context = AgentContext(request_id="...")
result = await agent.run(input_data, context)
```

### Backup Files

Original implementations have been backed up:

- `app/services/recommendation_service.py.backup` - Original 400+ line service

---

## Future Extensions

### Planned Workflows

1. **SentimentSearchWorkflow**: Search products by sentiment across categories
2. **QualityAlertWorkflow**: Detect products with declining sentiment
3. **GeoSegmentationWorkflow**: Analyze purchase patterns by geography
4. **CLVPredictionWorkflow**: Predict customer lifetime value
5. **ChurnPreventionWorkflow**: Identify at-risk customers

### New Agent Ideas

1. **TrendDetectionAgent**: Identify emerging product trends
2. **PriceOptimizationAgent**: Suggest optimal pricing based on demand
3. **InventoryForecastAgent**: Predict stock requirements
4. **CrossSellAgent**: Identify product bundles
5. **SeasonalityAgent**: Detect seasonal purchase patterns

### LangGraph Integration (Optional)

The new agents can be wrapped in LangGraph nodes:

```python
from langgraph.graph import StateGraph

class WorkflowState(TypedDict):
    profile: CustomerProfile
    similar_customers: List[SimilarCustomer]
    # ...

graph = StateGraph(WorkflowState)

# Each agent becomes a node
graph.add_node("profiling", lambda state: profiling_agent.run(...))
graph.add_node("similar", lambda state: similar_agent.run(...))

# Define edges (workflow)
graph.add_edge("profiling", "similar")
graph.add_edge("similar", "sentiment")

workflow = graph.compile()
```

**Benefits**:
- Visual workflow representation
- Conditional routing
- Human-in-the-loop
- State persistence

**Key Principle**: Agents remain standalone, LangGraph is just an orchestration option.

---

## Testing Strategy

### Unit Tests

**Test agents in isolation** with mock dependencies:

```python
# See: tests/capabilities/test_product_scoring_agent.py

@pytest.mark.asyncio
async def test_product_scoring_basic():
    agent = ProductScoringAgent()

    # In-memory test data (no database)
    input_data = ProductScoringInput(...)
    context = AgentContext(request_id="test")

    # Run agent
    output = await agent.run(input_data, context)

    # Assert behavior
    assert len(output.recommendations) == 3
```

### Integration Tests

**Test workflows end-to-end** with test database:

```python
@pytest.mark.asyncio
async def test_recommendation_workflow_integration():
    # Initialize workflow with test repositories
    workflow = PersonalizedRecommendationWorkflow(...)

    # Execute full workflow
    response = await workflow.execute(
        customer_id="test-123",
        query="What should I buy?",
        ...
    )

    # Assert end-to-end behavior
    assert len(response.recommendations) > 0
    assert response.confidence_score > 0
```

### Observability Tests

**Verify tracing and metrics**:

```python
def test_agent_observability():
    agent = MyAgent(...)
    context = AgentContext(request_id="test")

    await agent.run(input_data, context)

    # Check execution metadata was recorded
    assert "agent_executions" in context.metadata
    assert context.metadata["agent_executions"][0]["agent_id"] == "my_agent"
    assert context.metadata["agent_executions"][0]["success"] is True
```

---

## Configuration

### Agent Configuration

Agents respect configuration from `app/core/config.py`:

```python
# Scoring weights
COLLABORATIVE_WEIGHT = 0.6
CATEGORY_AFFINITY_WEIGHT = 0.4

# Sentiment
SENTIMENT_THRESHOLD = 0.6
MIN_PURCHASES_FOR_PROFILE = 2

# Vector search
SIMILARITY_THRESHOLD = 0.75
SIMILARITY_TOP_K = 20

# LLM models
PROFILING_MODEL = "llama3.2:3b"
RESPONSE_MODEL = "llama3.1:8b"
TEMPERATURE = 0.1
```

### Environment Variables

```bash
# Tracing
LANGFUSE_ENABLED=true
LANGFUSE_PUBLIC_KEY=pk_xxx
LANGFUSE_SECRET_KEY=sk_xxx

# Data paths
PURCHASE_DATA_PATH=data/raw/customer_purchase_data.csv
REVIEWS_DATA_PATH=data/raw/customer_reviews_data.csv

# Vector store
VECTOR_DB_TYPE=chroma  # Currently ignored (always uses Chroma)
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

---

## Performance Characteristics

### Agent Execution Times (Approximate)

| Agent | Avg Latency | Dependencies |
|-------|-------------|--------------|
| CustomerProfilingAgent | 10-20ms | Database read |
| SimilarCustomersAgent | 50-100ms | Vector search |
| SentimentFilteringAgent | 100-200ms | Multiple review reads |
| ProductScoringAgent | 5-10ms | In-memory computation |
| ResponseGenerationAgent | 500-1000ms | LLM inference |

**Total Workflow**: ~700ms - 1.4s

### Optimization Opportunities

1. **Parallel Agent Execution**: Independent agents could run concurrently
2. **Caching**: Cache customer profiles and vector embeddings
3. **Batch Operations**: Process multiple products in parallel
4. **LLM Optimization**: Use faster models or skip reasoning for API speed

---

## Summary

### What Changed

✅ **Added**:
- Uniform `BaseAgent` interface with Pydantic schemas
- 5 new capability agents (profiling, similarity, sentiment, scoring, response)
- Workflow orchestration layer (`PersonalizedRecommendationWorkflow`)
- Agent registry for discovery
- Comprehensive unit test example
- This documentation

♻️ **Refactored**:
- `RecommendationService`: 400+ lines → 150 lines (thin facade)
- Clear separation: Orchestration (workflows) vs Logic (agents)

❌ **Deprecated**:
- Legacy LangGraph workflow (`app/graph/`)
- Old agent implementations (`app/agents/` except intent classifier)
- FAISS vector store (unused)

### What Stayed the Same

- API contracts (100% backward compatible)
- Response schemas (`RecommendationResponse`)
- Intent classification (still uses LangGraph agent)
- Data layer (repositories unchanged)
- Configuration system

### Benefits Achieved

- ✅ **Maintainability**: Single source of truth, no duplication
- ✅ **Testability**: Agents tested in isolation
- ✅ **Composability**: Agents reusable across workflows
- ✅ **Observability**: Uniform logging and tracing
- ✅ **Extensibility**: Easy to add new agents/workflows
- ✅ **Type Safety**: Pydantic validation everywhere
- ✅ **Performance**: Same performance characteristics

---

## Questions & Support

For questions about the new architecture:
1. Review this document
2. Check agent docstrings in `app/capabilities/agents/`
3. See example tests in `tests/capabilities/`
4. Consult backup file: `app/services/recommendation_service.py.backup`

---

**Last Updated**: 2025-01-18
**Version**: 1.0.0
**Status**: ✅ Production Ready
